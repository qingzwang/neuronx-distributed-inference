"""End-to-end VL benchmark for Qwen3.5-2B on Neuron.

For each image configuration:
  * vision encoder latency (standalone forward)
  * TTFT -- time from generate() start to the first output token
    (vision encode + mRoPE + CTE prefill)
  * TKG latency per token and decode throughput
  * total task latency (vision + CTE + TKG decode)
  * token counts: input_ids, merged_vision_tokens, output tokens
  * HBM peak, via neuron-monitor
  * sample output text

The contrib PR's NKI DeltaNet kernel is numerically unstable on prefill
sequences longer than ~30 tokens, so the full decode path only works for
a very small image grid (--grid-size 4 -> 4 merged tokens). Larger grids
are still useful for measuring the Neuron vision encoder alone; for
those we report vision-only metrics and flag E2E as "decoder NaN".

Usage:
    python3 examples/benchmark_vl.py \\
        --image /home/ubuntu/qwen-omini-on-trn/0.png \\
        --configs 4 8 64 \\
        --iters 10 --warmup 3 \\
        --out-json /tmp/qwen35_vl_bench.json
"""

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import torch
from PIL import Image

_CONTRIB_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _CONTRIB_ROOT not in sys.path:
    sys.path.insert(0, _CONTRIB_ROOT)

from src.modeling_qwen35 import NeuronQwen35ForCausalLM, Qwen35InferenceConfig
from src.modeling_qwen35_vl import (
    NeuronQwen35VLForCausalLM,
    Qwen35VLInferenceConfig,
)
from neuronx_distributed_inference.models.config import (
    NeuronConfig,
    OnDeviceSamplingConfig,
)


# ---------------------------------------------------------------------------
# HBM sampling via `neuron-monitor`
# ---------------------------------------------------------------------------

_MONITOR_CONFIG = {
    "period": "0.1s",
    "neuron_runtimes": [{
        "tag_filter": ".*",
        "metrics": [{"type": "memory_used"}],
    }],
}


class HBMSampler:
    """Spawn neuron-monitor, parse json-lines, track peak HBM for our pid."""

    def __init__(self):
        self._proc = None
        self._thr = None
        self.pid = os.getpid()
        self._stop_flag = False
        self.samples = []  # (ts, device_total_bytes, per_core_dict)
        self._cfg_path = None

    def _write_cfg(self):
        import tempfile
        fd, path = tempfile.mkstemp(prefix="neuronmon_", suffix=".json")
        with os.fdopen(fd, "w") as f:
            json.dump(_MONITOR_CONFIG, f)
        self._cfg_path = path

    def start(self):
        self._write_cfg()
        self.samples = []
        self._stop_flag = False
        self._proc = subprocess.Popen(
            ["neuron-monitor", "-c", self._cfg_path],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, bufsize=1,
        )
        self._thr = threading.Thread(target=self._loop, daemon=True)
        self._thr.start()

    def _loop(self):
        for line in self._proc.stdout:
            if self._stop_flag:
                break
            try:
                rec = json.loads(line)
            except Exception:
                continue
            for rt in rec.get("neuron_runtime_data", []) or []:
                if rt.get("pid") != self.pid:
                    continue
                mu = rt.get("report", {}).get("memory_used", {})
                dev_total = (mu.get("neuron_runtime_used_bytes", {})
                               .get("neuron_device", 0))
                cores = (mu.get("neuron_runtime_used_bytes", {})
                             .get("usage_breakdown", {})
                             .get("neuroncore_memory_usage", {}))
                per_core = {}
                for cid, cu in cores.items():
                    per_core[int(cid)] = sum(
                        cu.get(k, 0) or 0
                        for k in ("constants", "model_code",
                                  "model_shared_scratchpad",
                                  "runtime_memory", "tensors")
                    )
                self.samples.append((time.time(), dev_total, per_core))

    def stop(self):
        self._stop_flag = True
        if self._proc is not None:
            try:
                self._proc.terminate()
                self._proc.wait(timeout=2)
            except Exception:
                self._proc.kill()
        if self._thr is not None:
            self._thr.join(timeout=2)
        if self._cfg_path and os.path.exists(self._cfg_path):
            os.unlink(self._cfg_path)

    def peak_total_bytes(self):
        return max((s[1] for s in self.samples), default=0)

    def peak_per_core_bytes(self):
        peaks = {}
        for _, _, per_core in self.samples:
            for cid, v in per_core.items():
                peaks[cid] = max(peaks.get(cid, 0), v)
        return peaks


# ---------------------------------------------------------------------------
# Model setup
# ---------------------------------------------------------------------------


def build_models(args):
    with open(os.path.join(args.model_path, "config.json")) as f:
        full_cfg = json.load(f)
    text_cfg = dict(full_cfg["text_config"])
    text_cfg["pad_token_id"] = text_cfg.get("eos_token_id", 248044)
    text_cfg.setdefault("tie_word_embeddings", full_cfg.get("tie_word_embeddings", True))
    if "rope_parameters" in text_cfg:
        text_cfg["rope_theta"] = text_cfg["rope_parameters"].get("rope_theta", 10000000)

    # Match bucketing config to the compiled model so the runtime picks the
    # smallest bucket that fits the prompt instead of always using max seq_len.
    buckets = args.buckets or [args.seq_len]
    nc = NeuronConfig(
        tp_degree=args.tp_degree, batch_size=1, ctx_batch_size=1, tkg_batch_size=1,
        seq_len=max(buckets), torch_dtype=torch.bfloat16,
        on_device_sampling_config=OnDeviceSamplingConfig(top_k=1),
        enable_bucketing=len(buckets) > 1,
        buckets=buckets,
        context_encoding_buckets=buckets,
        flash_decoding_enabled=False, logical_nc_config=2,
        save_sharded_checkpoint=True,
    )
    ic = Qwen35InferenceConfig(neuron_config=nc, **text_cfg)

    vl_cfg = Qwen35VLInferenceConfig(
        text_config=ic,
        vision_config=full_cfg["vision_config"],
        image_token_id=full_cfg.get("image_token_id", 248056),
        video_token_id=full_cfg.get("video_token_id", 248057),
        vision_start_token_id=full_cfg.get("vision_start_token_id", 248053),
        vision_end_token_id=full_cfg.get("vision_end_token_id", 248054),
        spatial_merge_size=full_cfg["vision_config"].get("spatial_merge_size", 2),
    )

    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(args.model_path)

    vl_model = NeuronQwen35VLForCausalLM(
        model_path=args.model_path,
        text_config=ic,
        vision_config=vl_cfg,
        processor=processor,
    )
    vl_model.text_model.load(args.compiled_path)
    vl_model.vision_model_wrapper.load_vision_weights_from_hf(args.model_path)
    vl_model.vision_model_wrapper.load_compiled(args.vision_compiled_path)

    return vl_model, processor, full_cfg


# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------


def _prepare_inputs(processor, full_cfg, image_path, grid_size, prompt):
    """Build model inputs for a given patch-grid size.

    grid_size must be an even number; after spatial-merge by 2 the
    number of merged vision tokens is (grid_size / 2) ** 2.
    """
    patch = full_cfg["vision_config"]["patch_size"]
    merge = full_cfg["vision_config"]["spatial_merge_size"]
    # Force the HF processor to emit exactly grid_size x grid_size patches.
    pixel_budget = (grid_size * patch) ** 2
    processor.image_processor.min_pixels = pixel_budget
    processor.image_processor.max_pixels = pixel_budget

    img = Image.open(image_path).convert("RGB").resize(
        (grid_size * patch, grid_size * patch), Image.BILINEAR
    )

    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": img},
            {"type": "text", "text": prompt},
        ],
    }]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )
    grid_thw = inputs["image_grid_thw"]
    merged = int((grid_thw[0, 1] // merge) * (grid_thw[0, 2] // merge))
    return dict(
        input_ids=inputs["input_ids"],
        attention_mask=inputs.get("attention_mask",
                                  torch.ones_like(inputs["input_ids"])),
        pixel_values=inputs["pixel_values"],
        image_grid_thw=grid_thw,
        grid_size=grid_size,
        patches=int(grid_thw[0, 1] * grid_thw[0, 2]),
        merged_tokens=merged,
        input_ids_len=int(inputs["input_ids"].shape[1]),
    )


def bench_config(vl_model, processor, full_cfg, args, grid_size, sampler):
    print(f"\n=== grid_size={grid_size} ===")
    inp = _prepare_inputs(
        processor, full_cfg, args.image, grid_size, args.prompt
    )
    print(f"  patches={inp['patches']}  merged_tokens={inp['merged_tokens']}  "
          f"input_ids={inp['input_ids_len']}")

    result = {
        "grid_size": grid_size,
        "patches": inp["patches"],
        "merged_tokens": inp["merged_tokens"],
        "input_ids_len": inp["input_ids_len"],
        "prompt": args.prompt,
    }

    # ---- Vision encoder only ----------------------------------------------
    print(f"  vision-only warmup x{args.warmup}...")
    for _ in range(args.warmup):
        _ = vl_model.vision_model_wrapper(
            inp["pixel_values"], inp["image_grid_thw"]
        )

    # Run vision long enough for the HBM sampler (100 ms period) to catch
    # multiple samples when a single forward is sub-millisecond.
    min_runtime_s = 2.0
    sampler.start()
    vision_ms = []
    t_loop_start = time.perf_counter()
    i = 0
    while True:
        t0 = time.perf_counter()
        _ = vl_model.vision_model_wrapper(
            inp["pixel_values"], inp["image_grid_thw"]
        )
        vision_ms.append((time.perf_counter() - t0) * 1000.0)
        i += 1
        # stop when we've done the requested iters AND covered min_runtime_s
        if i >= args.iters and (time.perf_counter() - t_loop_start) >= min_runtime_s:
            break
    sampler.stop()
    result["vision_latency_ms"] = {
        "mean": sum(vision_ms) / len(vision_ms),
        "min":  min(vision_ms),
        "max":  max(vision_ms),
    }
    result["vision_hbm_peak_bytes_total"] = sampler.peak_total_bytes()
    result["vision_hbm_peak_bytes_per_core"] = sampler.peak_per_core_bytes()
    print(f"  vision ms: mean={result['vision_latency_ms']['mean']:.1f} "
          f"min={result['vision_latency_ms']['min']:.1f} "
          f"max={result['vision_latency_ms']['max']:.1f}")

    # ---- Full task (vision + CTE + TKG) -----------------------------------
    max_supported = max(args.buckets) if args.buckets else args.seq_len
    if inp["input_ids_len"] > max_supported:
        print(f"  !! input_ids {inp['input_ids_len']} > max compiled bucket "
              f"{max_supported}; skipping E2E (vision-only result kept)")
        result["e2e_status"] = "seq_len_exceeded"
        result["sample_reply"] = ""
        return result

    print(f"  e2e warmup x{max(1, args.warmup // 2)}...")
    ok_e2e = True
    try:
        for _ in range(max(1, args.warmup // 2)):
            out_ids, _ = vl_model.generate(
                input_ids=inp["input_ids"], attention_mask=inp["attention_mask"],
                pixel_values=inp["pixel_values"], image_grid_thw=inp["image_grid_thw"],
                max_new_tokens=args.max_new_tokens, temperature=0.0,
                return_timings=True,
            )
            new = out_ids[0, inp["input_ids_len"]:]
            if new.numel() < 2:
                ok_e2e = False
    except Exception as e:
        print(f"  !! E2E warmup raised {type(e).__name__}: {e}")
        result["e2e_status"] = f"error: {type(e).__name__}"
        result["sample_reply"] = ""
        return result
    if not ok_e2e:
        print(f"  !! decoder produced <= 1 new token -- likely NKI DeltaNet NaN. "
              f"Running timed iterations anyway to report latency.")
        result["e2e_status"] = "decoder_nan"
        # Fall through to the timed loop; timings are still meaningful --
        # they show how long the compiled CTE graph takes even when logits
        # are NaN.

    sampler.start()
    timings_list = []
    sample_reply = None
    for i in range(args.iters):
        out_ids, timings = vl_model.generate(
            input_ids=inp["input_ids"], attention_mask=inp["attention_mask"],
            pixel_values=inp["pixel_values"], image_grid_thw=inp["image_grid_thw"],
            max_new_tokens=args.max_new_tokens, temperature=0.0,
            return_timings=True,
        )
        timings_list.append(timings)
        if i == 0:
            new = out_ids[0, inp["input_ids_len"]:]
            sample_reply = processor.tokenizer.decode(
                new, skip_special_tokens=True
            )
    sampler.stop()

    def agg(key):
        vals = [t[key] for t in timings_list if key in t]
        return {
            "mean": sum(vals) / len(vals),
            "min":  min(vals),
            "max":  max(vals),
        } if vals else None

    new_tokens = [t["new_tokens"] for t in timings_list]
    tkg_tokens = [t["tkg_tokens"] for t in timings_list]
    tkg_ms     = [t["tkg_ms"] for t in timings_list]
    decode_tps = [
        (tt / (tm / 1000.0)) if tm > 0 else 0.0
        for tt, tm in zip(tkg_tokens, tkg_ms)
    ]

    if result.get("e2e_status") != "decoder_nan":
        result["e2e_status"] = "ok"
    result["e2e"] = {
        "vision_encode_ms": agg("vision_encode_ms"),
        "cte_ms":           agg("cte_ms"),
        "ttft_ms":          agg("ttft_ms"),
        "tkg_ms":           agg("tkg_ms"),
        "total_ms":         agg("total_ms"),
        "new_tokens_mean":  sum(new_tokens) / len(new_tokens),
        "tkg_tokens_mean":  sum(tkg_tokens) / len(tkg_tokens),
        "decode_tok_per_s_mean": sum(decode_tps) / len(decode_tps),
        "decode_tok_per_s_min":  min(decode_tps) if decode_tps else 0,
        "decode_tok_per_s_max":  max(decode_tps) if decode_tps else 0,
    }
    result["e2e_hbm_peak_bytes_total"] = sampler.peak_total_bytes()
    result["e2e_hbm_peak_bytes_per_core"] = sampler.peak_per_core_bytes()
    result["sample_reply"] = sample_reply

    e = result["e2e"]
    print(f"  E2E over {args.iters} runs:")
    print(f"    vision_encode: mean {e['vision_encode_ms']['mean']:.1f} ms")
    print(f"    CTE prefill : mean {e['cte_ms']['mean']:.1f} ms")
    print(f"    TTFT        : mean {e['ttft_ms']['mean']:.1f} ms")
    print(f"    TKG total   : mean {e['tkg_ms']['mean']:.1f} ms "
          f"({e['tkg_tokens_mean']:.1f} tok -> {e['decode_tok_per_s_mean']:.1f} tok/s)")
    print(f"    Total task  : mean {e['total_ms']['mean']:.1f} ms "
          f"({e['new_tokens_mean']:.1f} new tokens)")
    print(f"    sample reply: {sample_reply!r}"[:200])
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default="/home/ubuntu/models/Qwen3.5-2B")
    ap.add_argument("--compiled-path", default="/home/ubuntu/traced_model/Qwen3.5-2B")
    ap.add_argument("--vision-compiled-path",
                    default="/home/ubuntu/traced_model/Qwen3.5-2B/vision")
    ap.add_argument("--image", default="/home/ubuntu/qwen-omini-on-trn/0.png")
    ap.add_argument("--configs", type=int, nargs="+", default=[4, 64],
                    help="List of patch-grid sizes to benchmark "
                         "(4 -> 64x64 image, 64 -> 1024x1024 image)")
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--prompt", default="What is in this image?")
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--buckets", type=int, nargs="+", default=None,
                    help="CTE buckets for the compiled model. If given, enables "
                         "multi-bucket so short prompts pick a shorter graph.")
    ap.add_argument("--tp-degree", type=int, default=4)
    ap.add_argument("--max-new-tokens", type=int, default=40)
    ap.add_argument("--out-json", default="/tmp/qwen35_vl_bench.json")
    args = ap.parse_args()

    print("[setup] loading models...")
    vl_model, processor, full_cfg = build_models(args)

    sampler = HBMSampler()
    sampler.start()
    time.sleep(1.0)
    sampler.stop()
    baseline_total = sampler.peak_total_bytes()
    baseline_per_core = sampler.peak_per_core_bytes()
    print(f"[baseline] HBM after load: {baseline_total/1e9:.2f} GB")

    all_results = []
    for grid in args.configs:
        try:
            r = bench_config(vl_model, processor, full_cfg, args, grid, sampler)
            all_results.append(r)
        except Exception as e:
            import traceback
            traceback.print_exc()
            all_results.append({"grid_size": grid, "error": str(e)})

    summary = {
        "config": {
            "tp_degree": args.tp_degree,
            "seq_len": args.seq_len,
            "iters": args.iters,
            "warmup": args.warmup,
            "prompt": args.prompt,
            "max_new_tokens": args.max_new_tokens,
            "compiled_path": args.compiled_path,
            "vision_compiled_path": args.vision_compiled_path,
        },
        "baseline_hbm_bytes_total": baseline_total,
        "baseline_hbm_bytes_per_core": baseline_per_core,
        "results": all_results,
    }
    with open(args.out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[done] wrote {args.out_json}")


if __name__ == "__main__":
    main()
