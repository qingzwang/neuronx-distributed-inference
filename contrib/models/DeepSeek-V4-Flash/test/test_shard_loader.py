#!/usr/bin/env python
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify shard_loader puts the *right* bytes in the *right* rank.

A weight loader that silently does nothing still lets the compile pass, so
"Compiler status PASS" proves nothing about accuracy. These checks assert on
tensor values, not on shapes or exit codes.

  1. sharded_tensors_differ_across_ranks
       wq_b / wo_a / attn_sink / embed must hold *different* values on rank 0
       vs rank r. If NxD-style silent no-op sharding were happening, every
       rank would hold the same (full or init) tensor.
  2. slices_concatenate_to_the_full_tensor
       Concatenating the per-rank slices along the shard dim must reproduce
       the checkpoint tensor bit-for-bit.
  3. replicated_tensors_match_everywhere
       Norms / hc_* / gate must be identical across ranks and equal to the
       checkpoint.
  4. no_parameter_left_at_init
       Every parameter that exists in the checkpoint must have moved away
       from the sentinel value we pre-fill, catching keys the loader skips.
  5. expert_ownership_follows_rank
       Rank r must hold routed experts [r*32, (r+1)*32) for TP=8 and their
       weights must match the FP4-dequantized checkpoint values.

Run:
    source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
    cd contrib/models/DeepSeek-V4-Flash
    DSV4_MODEL_PATH=/mnt/data/models/DeepSeek-V4-Flash \
        python test/test_shard_loader.py --tp 8
"""

import argparse
import json
import os
import sys

import torch
from safetensors import safe_open

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.abspath(os.path.join(_HERE, "..", "src"))
sys.path.insert(0, _SRC)

import compile_neuron as C  # noqa: E402
import dequant_checkpoint as dq  # noqa: E402
import paths  # noqa: E402
import shard_loader as SL  # noqa: E402

SENTINEL = -12345.0


def build_rank_model(hf, rank, world_size, n_layers, seq_len=16):
    """Construct HF's Transformer with rank-local shapes, on real memory."""
    import torch.distributed as dist

    dist.is_initialized = lambda: True
    dist.get_world_size = lambda *a, **k: world_size
    dist.get_rank = lambda *a, **k: rank

    cfg = json.load(open(paths.config_json()))
    cfg.update(
        dtype="bf16", expert_dtype=None, n_mtp_layers=0,
        max_batch_size=1, max_seq_len=seq_len,
    )
    cfg["n_layers"] = n_layers
    cfg["compress_ratios"] = cfg["compress_ratios"][:n_layers]
    model = hf.Transformer(hf.ModelArgs(**cfg)).eval()
    # Pre-fill so "still at init" is detectable.
    with torch.no_grad():
        for p in model.parameters():
            if p.is_floating_point():
                p.fill_(SENTINEL)
    return model


def ckpt_tensor(name):
    """Dequantized full checkpoint tensor for `name`."""
    root = paths.model_path()
    wm = SL._load_weight_map(root)
    with safe_open(os.path.join(root, wm[name]), framework="pt", device="cpu") as h:
        return SL._dequant_one(h, name, set(h.keys()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tp", type=int, default=8)
    ap.add_argument("--n-layers", type=int, default=1)
    ap.add_argument("--ranks", default=None,
                    help="Comma-separated ranks to materialize (default 0 and tp-1)")
    args = ap.parse_args()
    ws = args.tp
    ranks = ([int(r) for r in args.ranks.split(",")] if args.ranks
             else [0, ws - 1])

    C._wire_shims()
    paths.add_hf_inference_to_syspath()
    import model as hf
    C._patch_hf_model_for_xla(hf)
    C._patch_moe_forward_for_xla(hf)
    torch.set_default_dtype(torch.bfloat16)

    loaded = {}
    for r in ranks:
        m = build_rank_model(hf, r, ws, args.n_layers)
        SL.load_rank_weights(m, paths.model_path(), rank=r, world_size=ws)
        sd = dict(m.named_parameters())
        sd.update(dict(m.named_buffers()))
        loaded[r] = {k: v.detach().clone() for k, v in sd.items()}
        del m

    failures = []

    def check(cond, label, detail=""):
        status = "PASS" if cond else "FAIL"
        print(f"  [{status}] {label}" + (f"  {detail}" if detail else ""))
        if not cond:
            failures.append(label)

    r0, rN = ranks[0], ranks[-1]

    print(f"\n=== 1. sharded tensors differ across rank {r0} vs {rN} ===")
    for name in ("embed.weight", "layers.0.attn.wq_b.weight",
                 "layers.0.attn.wo_a.weight", "layers.0.attn.wo_b.weight",
                 "layers.0.attn.attn_sink"):
        a, b = loaded[r0][name].float(), loaded[rN][name].float()
        differs = not torch.equal(a, b)
        check(differs, f"{name} differs across ranks",
              f"shape={tuple(a.shape)} maxdiff={(a - b).abs().max():.4f}")

    print("\n=== 2. per-rank slices concatenate to the checkpoint tensor ===")
    if len(ranks) == ws:
        for name, dim in (("layers.0.attn.wq_b.weight", 0),
                          ("layers.0.attn.wo_b.weight", 1),
                          ("layers.0.attn.attn_sink", 0),
                          ("embed.weight", 0)):
            full = ckpt_tensor(name)
            rebuilt = torch.cat([loaded[r][name] for r in range(ws)], dim=dim)
            same = torch.equal(rebuilt.to(full.dtype), full)
            check(same, f"{name} reassembles from {ws} slices",
                  f"dim={dim} shape={tuple(full.shape)}")
    else:
        # Only some ranks materialized: verify each slice against the checkpoint.
        for name, dim in (("layers.0.attn.wq_b.weight", 0),
                          ("layers.0.attn.wo_b.weight", 1),
                          ("layers.0.attn.attn_sink", 0)):
            full = ckpt_tensor(name)
            n = full.size(dim) // ws
            for r in ranks:
                expect = full.narrow(dim, r * n, n)
                got = loaded[r][name].to(full.dtype)
                check(torch.equal(got, expect),
                      f"{name} rank{r} equals checkpoint slice",
                      f"dim={dim} slice={r*n}:{(r+1)*n}")

    print("\n=== 3. replicated tensors identical across ranks & match ckpt ===")
    for name in ("layers.0.attn_norm.weight", "layers.0.hc_attn_fn",
                 "layers.0.ffn.gate.weight", "norm.weight",
                 "layers.0.attn.wkv.weight"):
        a, b = loaded[r0][name], loaded[rN][name]
        check(torch.equal(a, b), f"{name} replicated identically")
        full = ckpt_tensor(name)
        check(torch.equal(a.to(full.dtype), full),
              f"{name} matches checkpoint", f"shape={tuple(full.shape)}")

    print("\n=== 4. no parameter left at the init sentinel ===")
    wm = SL._load_weight_map(paths.model_path())
    for r in ranks:
        stuck = [
            k for k, v in loaded[r].items()
            if k in wm and v.is_floating_point()
            and torch.all(v.float() == SENTINEL)
        ]
        check(not stuck, f"rank{r}: every checkpoint-backed param was written",
              f"stuck={stuck[:5]}" if stuck else "")

    print("\n=== 5. routed-expert ownership follows rank ===")
    n_local = 256 // ws
    for r in ranks:
        have = sorted(
            int(k.split(".")[4]) for k in loaded[r]
            if k.startswith("layers.0.ffn.experts.") and k.endswith("w1.weight")
        )
        expect = list(range(r * n_local, (r + 1) * n_local))
        check(have == expect, f"rank{r} owns experts {expect[0]}..{expect[-1]}",
              f"got {len(have)} experts")
        eid = expect[0]
        name = f"layers.0.ffn.experts.{eid}.w1.weight"
        full = ckpt_tensor(name)
        check(torch.equal(loaded[r][name].to(full.dtype), full),
              f"rank{r} expert {eid} w1 matches FP4-dequantized checkpoint",
              f"shape={tuple(full.shape)}")

    print("\n" + "=" * 62)
    if failures:
        print(f"FAILED {len(failures)} check(s):")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
