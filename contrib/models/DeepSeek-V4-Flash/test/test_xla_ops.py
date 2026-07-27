#!/usr/bin/env python
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify xla_ops.topk_indices is a drop-in for torch.topk, then that it
compiles on trn2 (torch.topk does not: HLO `sort` is unsupported).

The CPU equivalence checks run without a device; pass --device to also compile
and execute on Neuron.

Run:
    source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
    cd contrib/models/DeepSeek-V4-Flash
    python test/test_xla_ops.py --device
"""

import argparse
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "src")))

import xla_ops  # noqa: E402

failures = []


def check(cond, label, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {label}" + (f"  {detail}" if detail else ""))
    if not cond:
        failures.append(label)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", action="store_true",
                    help="Also compile/run on Neuron (slow).")
    args = ap.parse_args()

    g = torch.Generator().manual_seed(0)

    print("\n=== 1. topk_indices matches torch.topk on random scores ===")
    for shape, k in (((16, 256), 6), ((1, 256), 6), ((64, 256), 6),
                     ((16, 512), 1), ((8, 32), 32), ((4, 4096), 8)):
        s = torch.randn(*shape, generator=g)
        want = s.topk(k, dim=-1)
        got_v, got_i = xla_ops.topk(s, k)
        check(torch.equal(got_i, want.indices),
              f"indices match shape={shape} k={k}")
        check(torch.equal(got_v, want.values),
              f"values match shape={shape} k={k}")

    print("\n=== 2. ties: same values selected, indices deterministic ===")
    # torch.topk's tie-break is unspecified (an all-zero row can come back as
    # [10, 11, 12, 9]), so only the *values* are comparable. Ours always takes
    # the lowest index, which is what makes the traced graph reproducible.
    s = torch.zeros(2, 16)
    check(torch.equal(xla_ops.topk(s, 4)[0], s.topk(4, dim=-1).values),
          "all-equal row: values match")
    check(torch.equal(xla_ops.topk_indices(s, 4),
                      torch.arange(4).expand(2, 4)),
          "all-equal row: lowest indices chosen")

    s = torch.zeros(1, 8)
    s[0, 5] = s[0, 2] = 1.0
    got = xla_ops.topk_indices(s, 3)
    check(torch.equal(xla_ops.topk(s, 3)[0], s.topk(3, dim=-1).values),
          "duplicate maxima: values match")
    check(sorted(got[0, :2].tolist()) == [2, 5],
          "duplicate maxima: both peaks selected", f"got {got[0].tolist()}")

    print("\n=== 3. 3-D input (Indexer's [b, s, t] index_score) ===")
    s = torch.randn(1, 16, 128, generator=g)
    check(torch.equal(xla_ops.topk_indices(s, 6), s.topk(6, dim=-1).indices),
          "3-D last-dim selection")

    print("\n=== 4. -inf masked positions rank last, as in the Indexer ===")
    s = torch.randn(4, 32, generator=g)
    s[:, 16:] = float("-inf")
    got = xla_ops.topk_indices(s, 16)
    check(bool((got < 16).all()), "no masked position selected",
          f"max idx={int(got.max())}")

    print("\n=== 5. topk_indices_unordered covers the full axis when k>=n ===")
    s = torch.randn(1, 16, 128, generator=g)
    idx = xla_ops.topk_indices_unordered(s, 512)   # index_topk > n
    check(tuple(idx.shape) == (1, 16, 128), "shape is full axis",
          f"got {tuple(idx.shape)}")
    check(torch.equal(idx[0, 0], torch.arange(128)), "identity ordering")
    try:
        xla_ops.topk_indices_unordered(s, 64)
        check(False, "large-k raises NotImplementedError")
    except NotImplementedError:
        check(True, "large-k raises NotImplementedError")

    print("\n=== 6. k > n is rejected ===")
    try:
        xla_ops.topk_indices(torch.zeros(2, 4), 5)
        check(False, "k>n raises ValueError")
    except ValueError:
        check(True, "k>n raises ValueError")

    if args.device:
        print("\n=== 7. compiles and runs on trn2 (torch.topk does not) ===")
        import torch_neuronx

        scores = torch.randn(16, 256, generator=g)
        ref = xla_ops.topk_indices(scores, 6)

        class Sel(torch.nn.Module):
            def forward(self, s):
                return xla_ops.topk_indices(s, 6)

        try:
            tr = torch_neuronx.trace(
                Sel(), (scores,),
                compiler_args=["--model-type=transformer", "-O1",
                               "--auto-cast=none"])
            out = tr(scores)
            check(torch.equal(out.int(), ref.int()),
                  "device result equals CPU result")
        except Exception as e:
            check(False, "compiles + runs on device", f"{type(e).__name__}: {e}")

        # Confirm the thing we are replacing really is rejected, so this test
        # documents *why* xla_ops exists rather than asserting it blindly.
        class TorchTopk(torch.nn.Module):
            def forward(self, s):
                return s.topk(6, dim=-1)[1]

        try:
            torch_neuronx.trace(
                TorchTopk(), (scores,),
                compiler_args=["--model-type=transformer", "-O1",
                               "--auto-cast=none"])
            print("  [NOTE] torch.topk now compiles; xla_ops may be unnecessary")
        except Exception:
            check(True, "torch.topk still rejected by neuronx-cc (baseline)")

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
