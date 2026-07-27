#!/usr/bin/env python
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Dequantize DeepSeek-V4-Flash weights: FP4 experts + FP8 attention/o_lora → BF16.

The checkpoint from `deepseek-ai/DeepSeek-V4-Flash` uses a mixed-precision
layout that Neuron cannot consume directly:

  * MoE routed expert weights (`ffn.experts.N.{w1,w2,w3}.weight`):
      stored as int8 with each byte packing 2 FP4 (E2M1) values.
      Per-block scale in `.scale`  fp8_e8m0fnu with block size 32 on the
      packed inner dim (i.e. 64 on the unpacked dim).
      Shape convention: weight [out, in_packed] where in_unpacked = 2 * in_packed.

  * Attention / o_lora projections (`attn.{wkv,wo_a,...}.weight`):
      stored as fp8_e4m3fn with per-block scale fp8_e8m0fnu, block_size 128×128.

  * Everything else (norms, embed, head, hc_*, attn_sink, tid2eid, ape, ...)
      stored in its native dtype (bf16 / fp32 / int64).

This module:
  1. Walks all 46 safetensors shards.
  2. For each tensor that has a paired `.scale` sibling, dequantizes to bf16.
  3. Passes through non-quantized tensors unchanged.
  4. Optionally emits merged bf16 safetensors (default: dry-run + summary).

Usage:
    python src/dequant_checkpoint.py --ckpt /mnt/nvme/models/DeepSeek-V4-Flash \\
        --out-dir /mnt/nvme/models/DeepSeek-V4-Flash-BF16     # writes ~570 GB
    python src/dequant_checkpoint.py --ckpt <path> --dry-run   # just validate
"""

import argparse
import os
import sys
import time
from glob import glob
from pathlib import Path

import torch
from safetensors import safe_open

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths  # noqa: E402


# FP4 (E2M1) decode table from HF's convert.py. High-bit-1 indicates sign, low 3 bits value.
FP4_TABLE = torch.tensor(
    [
        0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
        0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
    ],
    dtype=torch.float32,
)


def _decode_e8m0_scale(scale_e8m0: torch.Tensor) -> torch.Tensor:
    """fp8_e8m0fnu → fp32.

    e8m0 encodes a power of 2 in the range [2^-127, 2^127] using an 8-bit
    biased exponent (bias=127). Bit pattern 0xFF encodes NaN.
    """
    if scale_e8m0.dtype != torch.float8_e8m0fnu:
        # Some scales come as fp32 already; nothing to do
        return scale_e8m0.float()
    # torch.float8_e8m0fnu -> fp32 is a native cast
    return scale_e8m0.to(torch.float32)


def dequant_fp4_expert(weight_int8: torch.Tensor, scale_e8m0: torch.Tensor) -> torch.Tensor:
    """FP4 (E2M1) packed × per-block e8m0 scale → BF16 dense weight.

    Args:
        weight_int8: shape [out, in_packed] where each byte holds 2 FP4 vals.
        scale_e8m0:  shape [out, in_unpacked / fp4_block_size], fp8_e8m0fnu.
                     fp4_block_size = 32 on the unpacked dim.

    Returns:
        bf16 tensor of shape [out, in_unpacked = 2 * in_packed].
    """
    assert weight_int8.dtype == torch.int8, weight_int8.dtype
    assert weight_int8.ndim == 2
    out_dim, in_packed = weight_int8.shape
    in_unpacked = 2 * in_packed
    fp4_block_size = 32  # scales are per 32 unpacked FP4 values along the inner dim
    assert in_unpacked % fp4_block_size == 0
    assert scale_e8m0.shape == (out_dim, in_unpacked // fp4_block_size), (
        f"scale shape {scale_e8m0.shape} vs expected "
        f"{(out_dim, in_unpacked // fp4_block_size)}"
    )

    # Unpack two FP4 nibbles per byte -> FP4 values in fp32
    x = weight_int8.view(torch.uint8)
    low = x & 0x0F
    high = (x >> 4) & 0x0F
    # stack low+high along the last dim then flatten -> [out, in_unpacked]
    x = torch.stack(
        [FP4_TABLE[low.long()], FP4_TABLE[high.long()]], dim=-1
    ).flatten(1)  # [out, in_unpacked], fp32

    # Broadcast scale to per-element multiplier.
    # scale[o, b] applies to x[o, b*32 : (b+1)*32].
    scale = _decode_e8m0_scale(scale_e8m0)  # [out, in_unpacked // 32], fp32
    scale_expanded = scale.repeat_interleave(fp4_block_size, dim=1)  # [out, in_unpacked]

    dequant = x * scale_expanded
    return dequant.to(torch.bfloat16)


def dequant_fp8_block(weight_fp8: torch.Tensor, scale_e8m0: torch.Tensor) -> torch.Tensor:
    """FP8 (E4M3) × per-block e8m0 scale → BF16.

    Block size is 128×128 (both dims). scale shape must be
    (out // 128, in // 128).
    """
    assert weight_fp8.dtype == torch.float8_e4m3fn, weight_fp8.dtype
    assert weight_fp8.ndim == 2
    out_dim, in_dim = weight_fp8.shape
    fp8_block = 128
    assert out_dim % fp8_block == 0 and in_dim % fp8_block == 0
    exp_scale_shape = (out_dim // fp8_block, in_dim // fp8_block)
    assert scale_e8m0.shape == exp_scale_shape, (
        f"scale shape {scale_e8m0.shape} vs expected {exp_scale_shape}"
    )

    w = weight_fp8.to(torch.float32)
    s = _decode_e8m0_scale(scale_e8m0)  # [out // 128, in // 128]
    # Broadcast: (out // 128, in // 128) -> (out // 128, 1, in // 128, 1) -> (out, in)
    s_full = s[:, None, :, None].expand(-1, fp8_block, -1, fp8_block).reshape(out_dim, in_dim)
    dequant = w * s_full
    return dequant.to(torch.bfloat16)


def dequant_wo_a(weight_fp8: torch.Tensor, scale_e8m0: torch.Tensor) -> torch.Tensor:
    """`wo_a.weight` uses a different scale layout: 64x32 for a 8192x4096 weight.

    The block layout for wo_a is (128, 128) along the *output* dim only:
    scale shape (out_dim // 128, in_dim // 128) but with out_dim=8192, in_dim=4096
    that gives (64, 32) — matches. Reuse the standard fp8 block dequant.
    """
    return dequant_fp8_block(weight_fp8, scale_e8m0)


# --- Key classification: which tensors need dequantization ---

def _is_expert_weight(key: str) -> bool:
    """`layers.N.ffn.experts.M.w1.weight` — routed expert FP4 weight."""
    return (
        "experts" in key
        and "shared_experts" not in key
        and key.endswith(".weight")
        and not key.endswith(".bias")
    )


def _is_fp8_attn_weight(key: str) -> bool:
    """Attention/o-lora FP8-quantized linear weights."""
    return (
        key.endswith(".weight")
        and (
            ".attn.wkv" in key or
            ".attn.wq_b" in key or
            ".attn.wq_a" in key or
            ".attn.wo_a" in key or
            ".attn.wo_b" in key or
            (".attn.indexer.wq_b" in key) or
            ("shared_experts" in key and ("w1" in key or "w2" in key or "w3" in key))
        )
    )


# --- Main walk ---

def walk_shards(ckpt_dir: str):
    """Yields (shard_path, key, tensor) tuples across all shards in order."""
    shards = sorted(glob(os.path.join(ckpt_dir, "model-*.safetensors")))
    for p in shards:
        with safe_open(p, framework="pt", device="cpu") as f:
            keys = list(f.keys())
            yield p, f, keys


def dequant_shard(path: str, dry_run: bool):
    """Dequantize all tensors in one shard. Returns dict[name] -> bf16 tensor."""
    out = {}
    with safe_open(path, framework="pt", device="cpu") as f:
        all_keys = list(f.keys())
        # Build scale lookup by weight-name -> scale-name
        scale_keys = {k for k in all_keys if k.endswith(".scale")}
        weight_keys = [k for k in all_keys if not k.endswith(".scale")]

        stats = {"fp4_experts": 0, "fp8_attn": 0, "passthrough": 0, "orphan_scales": 0}
        for k in weight_keys:
            scale_name = k.replace(".weight", ".scale")

            if _is_expert_weight(k) and scale_name in scale_keys:
                w = f.get_tensor(k)
                s = f.get_tensor(scale_name)
                deq = dequant_fp4_expert(w, s)
                if not dry_run:
                    out[k] = deq
                stats["fp4_experts"] += 1
                # verify
                if torch.isnan(deq).any() or torch.isinf(deq).any():
                    raise RuntimeError(f"{k}: NaN/Inf in dequant output")
            elif _is_fp8_attn_weight(k) and scale_name in scale_keys:
                w = f.get_tensor(k)
                s = f.get_tensor(scale_name)
                deq = dequant_fp8_block(w, s)
                if not dry_run:
                    out[k] = deq
                stats["fp8_attn"] += 1
                if torch.isnan(deq).any() or torch.isinf(deq).any():
                    raise RuntimeError(f"{k}: NaN/Inf in dequant output")
            else:
                # Not quantized (bf16 / fp32 / int64), passthrough
                t = f.get_tensor(k)
                if not dry_run:
                    out[k] = t
                stats["passthrough"] += 1

        # Check for orphan scales (scale without matching weight)
        for sk in scale_keys:
            wk = sk.replace(".scale", ".weight")
            if wk not in weight_keys:
                stats["orphan_scales"] += 1

    return out, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=paths.model_path(),
                    help="HF checkpoint dir with 46 safetensors shards")
    ap.add_argument("--out-dir", default=None,
                    help="Write dequantized bf16 safetensors here. Default: dry-run.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Validate only, do not write output.")
    ap.add_argument("--limit-shards", type=int, default=0,
                    help="Only process this many shards (0 = all).")
    args = ap.parse_args()

    dry = args.dry_run or args.out_dir is None
    if not dry:
        os.makedirs(args.out_dir, exist_ok=True)

    shards = sorted(glob(os.path.join(args.ckpt, "model-*.safetensors")))
    if args.limit_shards > 0:
        shards = shards[: args.limit_shards]
    print(f"[dequant] {len(shards)} shard(s), dry_run={dry}")

    total = {"fp4_experts": 0, "fp8_attn": 0, "passthrough": 0, "orphan_scales": 0}
    from safetensors.torch import save_file
    t0 = time.perf_counter()
    for i, shard in enumerate(shards):
        t_shard = time.perf_counter()
        out, stats = dequant_shard(shard, dry_run=dry)
        for k, v in stats.items():
            total[k] += v
        dt = time.perf_counter() - t_shard
        print(f"  [{i+1}/{len(shards)}] {Path(shard).name} : "
              f"fp4={stats['fp4_experts']:>4}  fp8={stats['fp8_attn']:>3}  "
              f"pass={stats['passthrough']:>4}  ({dt:.1f}s)")
        if not dry:
            out_path = os.path.join(args.out_dir, Path(shard).name)
            save_file(out, out_path)

    print(f"\n[done] {time.perf_counter()-t0:.1f}s")
    print(f"  fp4 experts     : {total['fp4_experts']}")
    print(f"  fp8 attn/dense  : {total['fp8_attn']}")
    print(f"  passthrough     : {total['passthrough']}")
    print(f"  orphan scales   : {total['orphan_scales']}  (should be 0)")


if __name__ == "__main__":
    main()
