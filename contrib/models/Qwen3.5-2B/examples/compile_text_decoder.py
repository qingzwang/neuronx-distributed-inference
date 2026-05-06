"""Compile the Qwen3.5-2B text decoder with multiple CTE buckets.

DeltaNet CTE is O(seq_len) even for short prompts because the NKI
recurrent kernel walks every position in the compiled bucket. Multi-
bucket compilation lets short prompts pick a short compiled graph,
cutting TTFT proportionally:

  bucket 128  -> CTE ~305 ms   (grid <= 8,  ~128 image px)
  bucket 512  -> CTE ~690 ms   (grid <= 22, ~352 image px)
  bucket 1024 -> CTE ~1367 ms  (grid <= 30, ~480 image px; also covers 62 via padding)
  bucket 2048 -> CTE ~2735 ms  (grid <= 44, ~704 image px; grid 62 reuses 1024)

NOTE: `USE_NKI=1` is required for the recurrent kernel. The default
fused Neumann kernel is faster on short prompts but NaNs on long
prefill; the recurrent kernel is numerically stable up through
grid=62 (992x992 image, 981-token prompt).

Usage:
    USE_NKI=1 python3 examples/compile_text_decoder.py \\
        --out /home/ubuntu/traced_model/Qwen3.5-2B-s2048-multibucket \\
        --buckets 128 512 1024 2048
"""

import argparse
import json
import os
import sys
import time

import torch

_CONTRIB_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _CONTRIB_ROOT not in sys.path:
    sys.path.insert(0, _CONTRIB_ROOT)

from neuronx_distributed_inference.models.config import (
    NeuronConfig,
    OnDeviceSamplingConfig,
)
from src.modeling_qwen35 import (
    NeuronQwen35ForCausalLM,
    Qwen35InferenceConfig,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default="/home/ubuntu/models/Qwen3.5-2B")
    ap.add_argument("--out",
                    default="/home/ubuntu/traced_model/Qwen3.5-2B-s2048-multibucket")
    ap.add_argument("--buckets", type=int, nargs="+",
                    default=[128, 512, 1024, 2048])
    ap.add_argument("--tp-degree", type=int, default=4)
    args = ap.parse_args()

    with open(os.path.join(args.model_path, "config.json")) as f:
        full = json.load(f)
    tc = dict(full["text_config"])
    tc["pad_token_id"] = tc.get("eos_token_id", 248044)
    tc.setdefault("tie_word_embeddings", full.get("tie_word_embeddings", True))
    if "rope_parameters" in tc:
        tc["rope_theta"] = tc["rope_parameters"].get("rope_theta", 10000000)

    nc = NeuronConfig(
        tp_degree=args.tp_degree,
        batch_size=1, ctx_batch_size=1, tkg_batch_size=1,
        seq_len=max(args.buckets),
        torch_dtype=torch.bfloat16,
        on_device_sampling_config=OnDeviceSamplingConfig(top_k=1),
        enable_bucketing=len(args.buckets) > 1,
        buckets=args.buckets,
        context_encoding_buckets=args.buckets,
        flash_decoding_enabled=False,
        logical_nc_config=2,
        save_sharded_checkpoint=True,
    )
    ic = Qwen35InferenceConfig(neuron_config=nc, **tc)

    os.makedirs(args.out, exist_ok=True)
    print(f"Compiling {len(args.buckets)} CTE buckets {args.buckets} -> {args.out}")
    if os.environ.get("USE_NKI") != "1":
        print("  WARNING: USE_NKI=1 is not set; falling back to default NKI fused "
              "kernel which NaNs on prefill > ~30 tokens. Re-run with "
              "USE_NKI=1 in front of the python command.")
    t0 = time.perf_counter()
    m = NeuronQwen35ForCausalLM(args.model_path, ic)
    m.compile(args.out)
    print(f"done in {time.perf_counter()-t0:.1f}s")


if __name__ == "__main__":
    main()
