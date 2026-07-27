#!/usr/bin/env python
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compare the compiled Neuron model's logits against a CPU run of the same graph.

Everything up to this point only proved the graph *compiles*. This is the
first check that it computes the right thing: same weights, same input, CPU
vs Neuron, compare logits.

The CPU side deliberately reuses `compile_neuron`'s own factory patches
(real-valued RoPE, static-shape MoE) so a mismatch points at the Neuron
lowering rather than at the CPU/GPU reference gap. `--vs-hf-reference` adds
a third run using HF's *unpatched* model.py to separately confirm the
patches themselves are faithful.

The CPU comparison model is built at world_size=1, so it holds full-size
tensors and needs no sharding; the Neuron artifact is the TP=N build. Any
gap between them is TP-reduction + bf16 + Neuron lowering.

Run:
    source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
    cd contrib/models/DeepSeek-V4-Flash
    DSV4_MODEL_PATH=/mnt/data/models/DeepSeek-V4-Flash python \
        test/test_neuron_vs_cpu.py --artifact /tmp/dsv4_w_tp8_L1 \
        --tp 8 --n-layers 1 --seq-len 16
"""

import argparse
import json
import os
import sys
import time

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.abspath(os.path.join(_HERE, "..", "src"))
sys.path.insert(0, _SRC)

import compile_neuron as C  # noqa: E402
import paths  # noqa: E402
import shard_loader as SL  # noqa: E402


def build_cpu_model(n_layers, seq_len, batch, patched=True):
    """HF Transformer at world_size=1 with the real weights loaded.

    patched=True applies compile_neuron's XLA patches (what Neuron actually
    runs); patched=False keeps HF's original complex RoPE + dispatch MoE.
    """
    import torch.distributed as dist
    dist.is_initialized = lambda: False

    C._wire_shims()
    paths.add_hf_inference_to_syspath()
    import model as hf
    if patched:
        C.apply_xla_patches(hf)
    torch.set_default_dtype(torch.bfloat16)

    cfg = json.load(open(paths.config_json()))
    cfg.update(dtype="bf16", expert_dtype=None, n_mtp_layers=0,
               max_batch_size=batch, max_seq_len=seq_len)
    cfg["n_layers"] = n_layers
    cfg["compress_ratios"] = cfg["compress_ratios"][:n_layers]
    model = hf.Transformer(hf.ModelArgs(**cfg)).eval()
    SL.load_rank_weights(model, paths.model_path(), rank=0, world_size=1)
    return model, hf


def run_cpu(model, hf, ids):
    raw = getattr(model.forward, "__wrapped__", None)
    with torch.no_grad():
        if raw is not None:
            return raw(model, ids, 0)
        return model(ids, 0)


def report(label, ref, got, tol_rel):
    ref_f, got_f = ref.float().flatten(), got.float().flatten()
    diff = (ref_f - got_f).abs()
    scale = ref_f.abs().mean().clamp_min(1e-6)
    rel = (diff.mean() / scale).item()
    max_abs = diff.max().item()

    k = min(5, ref_f.numel())
    ref_top = ref_f.topk(k).indices.tolist()
    got_top = got_f.topk(k).indices.tolist()
    top1_match = ref_top[0] == got_top[0]
    overlap = len(set(ref_top) & set(got_top))

    cos = torch.nn.functional.cosine_similarity(
        ref_f.unsqueeze(0), got_f.unsqueeze(0)
    ).item()

    print(f"\n--- {label} ---")
    print(f"  ref |mean|      : {ref_f.abs().mean().item():.4f}")
    print(f"  max abs diff    : {max_abs:.4f}")
    print(f"  mean rel diff   : {rel:.5f}   (tol {tol_rel})")
    print(f"  cosine sim      : {cos:.6f}")
    print(f"  top-1 token     : ref={ref_top[0]}  got={got_top[0]}  "
          f"{'MATCH' if top1_match else 'MISMATCH'}")
    print(f"  top-5 overlap   : {overlap}/5")
    print(f"  ref top5        : {ref_top}")
    print(f"  got top5        : {got_top}")

    ok = top1_match and rel < tol_rel
    print(f"  => {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", required=True, help="dir with tp_*.pt")
    ap.add_argument("--tp", type=int, default=8)
    ap.add_argument("--n-layers", type=int, default=1)
    ap.add_argument("--seq-len", type=int, default=16)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--tol-rel", type=float, default=0.02,
                    help="Allowed mean relative logit diff (bf16 + TP reduction)")
    ap.add_argument("--vs-hf-reference", action="store_true",
                    help="Also run HF's unpatched model.py to validate the "
                         "RoPE / MoE rewrites independently of Neuron.")
    ap.add_argument("--prompt-ids", default=None,
                    help="Comma-separated token ids; default is a fixed pseudo-"
                         "random sequence (tokenizer-independent).")
    args = ap.parse_args()

    if args.prompt_ids:
        ids_list = [int(x) for x in args.prompt_ids.split(",")]
    else:
        g = torch.Generator().manual_seed(0)
        ids_list = torch.randint(0, 129280, (args.seq_len,), generator=g).tolist()
    ids_list = (ids_list + [0] * args.seq_len)[: args.seq_len]
    ids = torch.tensor([ids_list] * args.batch, dtype=torch.long)
    print(f"[input] batch={args.batch} seq_len={args.seq_len} ids[:8]={ids_list[:8]}")

    results = []

    print(f"\n[cpu] building n_layers={args.n_layers} (patched graph, world_size=1)")
    t0 = time.perf_counter()
    cpu_model, hf = build_cpu_model(args.n_layers, args.seq_len, args.batch)
    print(f"[cpu] built + loaded in {time.perf_counter()-t0:.1f}s")
    t0 = time.perf_counter()
    cpu_logits = run_cpu(cpu_model, hf, ids)
    print(f"[cpu] forward in {time.perf_counter()-t0:.1f}s  "
          f"shape={list(cpu_logits.shape)}")

    if args.vs_hf_reference:
        print("\n[hf] building unpatched HF reference for patch validation")
        for mod in ("model",):
            sys.modules.pop(mod, None)
        hf_model, hf_raw = build_cpu_model(
            args.n_layers, args.seq_len, args.batch, patched=False)
        hf_logits = run_cpu(hf_model, hf_raw, ids)
        results.append(report(
            "HF unpatched (complex RoPE + dispatch MoE) vs patched CPU graph",
            hf_logits, cpu_logits, args.tol_rel))
        del hf_model

    del cpu_model

    print(f"\n[neuron] loading {args.artifact}")
    from neuronx_distributed.trace import parallel_model_load
    t0 = time.perf_counter()
    neuron_model = parallel_model_load(args.artifact)
    print(f"[neuron] loaded in {time.perf_counter()-t0:.1f}s")

    start_pos = torch.zeros((), dtype=torch.int32)
    t0 = time.perf_counter()
    neuron_logits = neuron_model(ids, start_pos)
    print(f"[neuron] forward in {time.perf_counter()-t0:.1f}s  "
          f"shape={list(neuron_logits.shape)}")

    nan = torch.isnan(neuron_logits).any().item()
    inf = torch.isinf(neuron_logits).any().item()
    print(f"[neuron] NaN={nan} Inf={inf}")
    if nan or inf:
        print("  => FAIL: non-finite logits")
        results.append(False)

    results.append(report("Neuron TP vs CPU (same weights)",
                          cpu_logits, neuron_logits, args.tol_rel))

    print("\n" + "=" * 62)
    ok = all(results)
    print("ALL CHECKS PASSED" if ok else "FAILURES PRESENT")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
