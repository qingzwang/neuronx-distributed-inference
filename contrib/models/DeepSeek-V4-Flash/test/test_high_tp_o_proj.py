#!/usr/bin/env python
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify the tp > o_groups O-projection split reproduces the tp=1 result.

HF caps TP at o_groups (8): `n_local_groups = o_groups // world_size` hits 0 at
tp >= 16, the wo_a view gets a zero-size dim, and the O-projection silently
contributes nothing. But tp=8 leaves 70.9 GB of weights per core against a
24 GB budget, so production needs tp >= 32.

compile_neuron._patch_attention_o_proj_for_high_tp moves the split to the
contraction dim past o_groups, relying on wo_b's existing all_reduce to sum the
partials. This runs one Attention layer at tp=1 and at tp=16/32/64, summing the
per-rank partials by hand (standing in for the all_reduce), and checks the
outputs agree.

Run:
    source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
    cd contrib/models/DeepSeek-V4-Flash
    DSV4_MODEL_PATH=/mnt/data/models/DeepSeek-V4-Flash \
        python test/test_high_tp_o_proj.py
"""

import argparse
import json
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.abspath(os.path.join(_HERE, "..", "src"))
sys.path.insert(0, _SRC)

import compile_neuron as C  # noqa: E402
import paths  # noqa: E402
import shard_loader as SL  # noqa: E402

failures = []


def check(cond, label, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {label}" + (f"  {detail}" if detail else ""))
    if not cond:
        failures.append(label)


def build(hf, world_size, rank, seq_len, n_layers=1):
    import torch.distributed as dist
    dist.is_initialized = lambda: True
    dist.get_world_size = lambda *a, **k: world_size
    dist.get_rank = lambda *a, **k: rank
    # wo_b all-reduces; we sum the per-rank partials ourselves instead.
    dist.all_reduce = lambda t, *a, **k: t

    cfg = json.load(open(paths.config_json()))
    cfg.update(dtype="bf16", expert_dtype=None, n_mtp_layers=0,
               max_batch_size=1, max_seq_len=seq_len)
    cfg["n_layers"] = n_layers
    cfg["compress_ratios"] = cfg["compress_ratios"][:n_layers]
    m = hf.Transformer(hf.ModelArgs(**cfg)).eval()
    SL.load_rank_weights(m, paths.model_path(), rank=rank,
                         world_size=world_size, verbose=False,
                         o_groups=cfg["o_groups"])
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq-len", type=int, default=16)
    ap.add_argument("--tps", default="16,32,64")
    args = ap.parse_args()

    C._wire_shims()
    paths.add_hf_inference_to_syspath()
    import model as hf
    C.apply_xla_patches(hf)
    torch.set_default_dtype(torch.bfloat16)

    cfg = json.load(open(paths.config_json()))
    o_groups, n_heads = cfg["o_groups"], cfg["n_heads"]
    S, D = args.seq_len, cfg["dim"]

    g = torch.Generator().manual_seed(0)
    x = torch.randn(1, S, D, generator=g, dtype=torch.bfloat16)

    print(f"\n=== reference: tp=1 (unsharded, o_groups={o_groups}) ===")
    m1 = build(hf, 1, 0, S)
    attn1 = m1.layers[0].attn
    check(attn1.n_local_groups == o_groups, "tp=1 keeps all groups",
          f"n_local_groups={attn1.n_local_groups}")
    with torch.no_grad():
        ref = attn1.forward(x, 0).float()
    print(f"  ref |mean|={ref.abs().mean():.5f} shape={list(ref.shape)}")
    del m1

    for ws in [int(t) for t in args.tps.split(",")]:
        print(f"\n=== tp={ws} (> o_groups: contraction-dim split) ===")
        acc = torch.zeros_like(ref)
        shapes = set()
        for r in range(ws):
            m = build(hf, ws, r, S)
            a = m.layers[0].attn
            if r == 0:
                check(a.n_local_groups == 0,
                      "n_local_groups is 0 (HF's path would be a no-op)")
                check(a.wo_a.weight.shape[0] == cfg["o_lora_rank"],
                      "wo_a re-allocated to [o_lora_rank, slice]",
                      f"shape={tuple(a.wo_a.weight.shape)}")
                check(a.n_local_heads == n_heads // ws,
                      f"n_local_heads={a.n_local_heads}")
            shapes.add(tuple(a.wo_a.weight.shape))
            with torch.no_grad():
                acc += a.forward(x, 0).float()
            del m

        check(len(shapes) == 1, "every rank has the same wo_a shape",
              f"{shapes}")
        diff = (acc - ref).abs()
        rel = (diff.max() / ref.abs().max()).item()
        cos = torch.nn.functional.cosine_similarity(
            acc.flatten().unsqueeze(0), ref.flatten().unsqueeze(0)).item()
        print(f"  summed |mean|={acc.abs().mean():.5f}  max abs diff={diff.max():.5f}"
              f"  rel={rel:.2e}  cos={cos:.6f}")
        # bf16 activations through a 4096-wide contraction: ~1e-2 relative is
        # the dtype floor, not a sharding error.
        check(rel < 0.05, f"tp={ws} matches tp=1 within bf16 tolerance",
              f"rel={rel:.2e}")
        check(cos > 0.999, f"tp={ws} cosine similarity", f"cos={cos:.6f}")

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
