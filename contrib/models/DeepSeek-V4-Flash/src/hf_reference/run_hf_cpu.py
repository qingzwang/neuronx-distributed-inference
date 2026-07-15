#!/usr/bin/env python
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Run DeepSeek-V4-Flash's HuggingFace reference model on CPU as an accuracy oracle.

The reference `inference/model.py` imports `kernel.py`, which uses TileLang/
Triton and only runs on NVIDIA GPUs. This driver:

  1. Injects `kernel_cpu` in place of `kernel` in sys.modules before
     `model` is imported, so the reference `sparse_attn`/`hc_split_sinkhorn`
     calls resolve to our PyTorch CPU implementations.
  2. Builds `ModelArgs` with `dtype="bf16"` and `expert_dtype=None`,
     so every `Linear`/`ColumnParallelLinear` allocates a BF16 weight
     matching what `dequant_checkpoint.dequant_shard` produces.
  3. Iterates the 46 FP4/FP8 safetensors shards, dequants each on the fly,
     and copies the tensors into the pre-allocated model parameters. No
     ~570 GB intermediate checkpoint is written to disk.
  4. Runs a single 1-token forward pass on a tiny prompt to verify the
     graph is wired correctly and the logits look reasonable.

Usage:
    source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
    cd contrib/models/DeepSeek-V4-Flash
    python src/hf_reference/run_hf_cpu.py --limit-shards 5   # smoke on 5 shards
    python src/hf_reference/run_hf_cpu.py                    # full model (SLOW)

Warning: even with careful memory planning, a single forward pass through
284 B params of BF16 weights on CPU is DRAM-bandwidth bound and will take
a long time — expect O(minutes) per token.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

# Ensure our package is importable as `hf_reference.kernel_cpu` (i.e. put
# `src/` on sys.path). Also add `src/` for `from src.dequant_checkpoint import ...`
# by putting the contrib root on sys.path too.
_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.abspath(os.path.join(_HERE, ".."))            # .../src
_CONTRIB_MODEL_DIR = os.path.abspath(os.path.join(_HERE, "..", "..")) # .../DeepSeek-V4-Flash
for p in (_SRC_DIR, _CONTRIB_MODEL_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

# Path to HF's cloned inference/ dir (comes with the checkpoint download)
_HF_INFERENCE_DIR = "/mnt/nvme/models/DeepSeek-V4-Flash/inference"


def _wire_cpu_kernel():
    """Make `import kernel` in HF's model.py resolve to our CPU shim, and
    provide a pure-PyTorch stand-in for `fast_hadamard_transform` (Tri Dao's
    package; the pip install requires CUDA)."""
    from hf_reference import kernel_cpu
    sys.modules["kernel"] = kernel_cpu

    # Fake fast_hadamard_transform via a natural-order Hadamard matrix.
    # rotate_activation() calls hadamard_transform(x, scale=1/sqrt(d)); for the
    # dims used in DeepSeek-V4 (rope_head_dim=64, index_head_dim=128), a plain
    # torch matmul against a precomputed H_d is fine on CPU.
    import types
    import torch as _torch

    def _hadamard_matrix(n: int) -> "_torch.Tensor":
        assert n > 0 and (n & (n - 1)) == 0, f"n must be power-of-2, got {n}"
        H = _torch.tensor([[1.0]], dtype=_torch.float32)
        while H.size(0) < n:
            H = _torch.cat([_torch.cat([H, H], dim=1),
                            _torch.cat([H, -H], dim=1)], dim=0)
        return H

    _H_CACHE: dict = {}

    def hadamard_transform(x: "_torch.Tensor", scale: float = 1.0) -> "_torch.Tensor":
        d = x.size(-1)
        if d not in _H_CACHE:
            _H_CACHE[d] = _hadamard_matrix(d)
        H = _H_CACHE[d].to(x.dtype).to(x.device)
        return _torch.matmul(x, H) * scale

    mod = types.ModuleType("fast_hadamard_transform")
    mod.hadamard_transform = hadamard_transform
    # importlib.util.find_spec (used by transformers>=5.0's package probe)
    # requires __spec__ to be set on faked modules.
    import importlib.machinery
    mod.__spec__ = importlib.machinery.ModuleSpec(
        "fast_hadamard_transform", loader=None,
    )
    sys.modules["fast_hadamard_transform"] = mod


def _import_hf_model():
    """Import HF's `model.py` after wiring the CPU kernel shim."""
    _wire_cpu_kernel()
    if _HF_INFERENCE_DIR not in sys.path:
        sys.path.insert(0, _HF_INFERENCE_DIR)
    import model as hf_model  # noqa: F401 — imported for side effects (dtype globals)
    return hf_model


def build_args(config_path: str, max_seq_len: int, hf_mod):
    """Read inference_config.json and override to BF16 fast-path.

    Overrides:
      dtype        "fp8"  -> "bf16"     (all Linear() use BF16 weights)
      expert_dtype "fp4"  -> None       (routed experts also BF16)
      n_mtp_layers        -> 0          (skip speculative-decoding head)
      max_batch_size      -> 1          (single-request oracle)
      max_seq_len         -> as passed  (shrink from 4096 to save memory)
    """
    with open(config_path) as f:
        cfg = json.load(f)
    cfg.update(
        dtype="bf16",
        expert_dtype=None,
        n_mtp_layers=0,
        max_batch_size=1,
        max_seq_len=max_seq_len,
    )
    return hf_mod.ModelArgs(**cfg)


def _fixup_key(name: str) -> str:
    """Map safetensors key -> model state_dict key.

    The checkpoint uses `layers.N.attn.wkv.weight`; the model registers this
    as `layers.N.attn.wkv.weight` too (Linear.weight). No renaming needed for
    weight tensors, but `.scale` companions must be dropped in load_state_dict
    (we take care of that during the walk).
    """
    return name


def load_weights_streaming(model, ckpt_dir: str, limit_shards: int = 0):
    """Dequant + copy tensors into pre-allocated model parameters, one shard
    at a time so we never hold >1 shard of dequantized weights in RAM."""
    # We reuse dequant_shard, which returns dict[str, bf16 tensor] for weights
    # and passthrough for everything else. Then we copy_ into model params.
    from dequant_checkpoint import dequant_shard  # type: ignore  # via _SRC_DIR on sys.path

    from glob import glob
    shards = sorted(glob(os.path.join(ckpt_dir, "model-*.safetensors")))
    if limit_shards > 0:
        shards = shards[:limit_shards]
    print(f"[load] {len(shards)} shard(s)")

    state = dict(model.named_parameters())
    state.update(dict(model.named_buffers()))
    n_copied = 0
    n_missing = 0
    for i, shard in enumerate(shards):
        t0 = time.perf_counter()
        deq, _stats = dequant_shard(shard, dry_run=False)
        for key, tensor in deq.items():
            # Drop `.scale` — those are not model parameters when we run BF16
            if key.endswith(".scale"):
                continue
            target_key = _fixup_key(key)
            if target_key in state:
                param = state[target_key]
                # Match dtype: model params are bf16, some passthrough tensors
                # may come as fp32/int64 — copy_ handles dtype conversion.
                if param.shape != tensor.shape:
                    raise RuntimeError(
                        f"shape mismatch for {target_key}: "
                        f"model={list(param.shape)} vs ckpt={list(tensor.shape)}"
                    )
                with torch.no_grad():
                    param.copy_(tensor.to(param.dtype))
                n_copied += 1
            else:
                n_missing += 1
        dt = time.perf_counter() - t0
        print(f"  [{i+1}/{len(shards)}] {Path(shard).name}  "
              f"copied={n_copied} missing={n_missing}  ({dt:.1f}s)")
    print(f"[load] done: copied={n_copied}, missing={n_missing}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", default="/mnt/nvme/models/DeepSeek-V4-Flash")
    ap.add_argument("--config",
                    default="/mnt/nvme/models/DeepSeek-V4-Flash/inference/config.json")
    ap.add_argument("--max-seq-len", type=int, default=64,
                    help="Shrink from 4096 to reduce KV cache RAM in this smoke")
    ap.add_argument("--limit-shards", type=int, default=0,
                    help="Load only this many shards (0 = all). Only useful for "
                         "verifying the plumbing since a partial model will not "
                         "produce sensible logits.")
    ap.add_argument("--forward-smoke", action="store_true",
                    help="After load, do a 1-token forward and print top-K logits.")
    ap.add_argument("--prompt", default="The capital of France is")
    args = ap.parse_args()

    torch.set_default_dtype(torch.bfloat16)
    torch.manual_seed(0)

    hf = _import_hf_model()
    m_args = build_args(args.config, args.max_seq_len, hf)
    print(f"[cfg] dtype={m_args.dtype} expert_dtype={m_args.expert_dtype} "
          f"n_layers={m_args.n_layers} n_mtp={m_args.n_mtp_layers} "
          f"max_seq_len={m_args.max_seq_len}")

    t0 = time.perf_counter()
    model = hf.Transformer(m_args)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[build] Transformer on CPU: {n_params/1e9:.2f} B params  "
          f"({time.perf_counter()-t0:.1f}s)")

    load_weights_streaming(model, args.ckpt_dir, args.limit_shards)

    if args.forward_smoke:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.ckpt_dir)
        ids = tok(args.prompt, return_tensors="pt").input_ids
        print(f"[fwd] prompt='{args.prompt}'  ids.shape={list(ids.shape)}")
        t0 = time.perf_counter()
        with torch.inference_mode():
            logits = model(ids, 0)
        print(f"[fwd] logits.shape={list(logits.shape)}  "
              f"({time.perf_counter()-t0:.1f}s)")
        # Print top-5 next tokens
        topk = logits[0].topk(5)
        for score, idx in zip(topk.values.tolist(), topk.indices.tolist()):
            tok_str = tok.decode([idx])
            print(f"    {idx:>7}  {score:7.2f}  {tok_str!r}")


if __name__ == "__main__":
    main()
