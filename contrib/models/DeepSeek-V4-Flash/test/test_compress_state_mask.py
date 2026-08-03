#!/usr/bin/env python
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Gate for step 2 of NXDI_PORT_DESIGN.md: the -inf masking rewrite.

Two claims, and the second is the one that matters:

  1. Masking in the graph reproduces HF's -inf-initialised ring exactly.
  2. It does so **regardless of what the buffer was initialised to**. That is the
     point: NxD's StateInitializer hardcodes torch.zeros with no hook to change
     the fill, so any port that depends on the -inf init is depending on something
     the framework will silently overwrite.

Also measures what getting it wrong costs, so the number in the design doc is
grounded rather than asserted: an unmasked zero-initialised ring is compared
against the correct result and the error reported.

CPU only, seconds to run.

    python test/test_compress_state_mask.py
"""

import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(os.path.dirname(_HERE), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import compress_state as cs  # noqa: E402

failures = []


def check(cond, label, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {label}" + (f"  {detail}" if detail else ""))
    if not cond:
        failures.append(label)
    return cond


def hf_reference(kv, score_written, ring_len, lo, hi, ratio, overlap):
    """What HF computes: ring initialised to -inf, real slots overwritten.

    Builds the ring the way the reference does — full(-inf) then assign — and
    compresses with a plain softmax, no mask.
    """
    B, D = kv.shape[0], kv.shape[2]
    score = torch.full((B, ring_len, D), float("-inf"))
    if not overlap:
        if lo:
            score[:, :lo] = score_written[:, :lo]
    else:
        if lo:
            score[:, :lo] = score_written[:, :lo]
        if hi:
            score[:, ratio:ratio + hi] = score_written[:, ratio:ratio + hi]
    weights = score.softmax(dim=1)
    return (kv * weights).sum(dim=1, keepdim=True)


def masked_version(kv, score_written, ring_len, lo, hi, ratio, overlap, fill):
    """The rewrite: ring initialised to `fill`, mask applied before softmax."""
    B, D = kv.shape[0], kv.shape[2]
    score = torch.full((B, ring_len, D), fill)
    if not overlap:
        if lo:
            score[:, :lo] = score_written[:, :lo]
    else:
        if lo:
            score[:, :lo] = score_written[:, :lo]
        if hi:
            score[:, ratio:ratio + hi] = score_written[:, ratio:ratio + hi]
    mask = cs.ring_slot_mask(ring_len, lo, hi, ratio, overlap, kv.device)
    return cs.compress_with_mask(kv, score, mask)


def main():
    torch.manual_seed(0)
    B, D, ratio = 1, 16, 4

    print("=== 1. masked rewrite == HF's -inf ring, across fill states ===")
    cases = []
    for overlap in (False, True):
        ring_len = (2 if overlap else 1) * ratio
        for lo in range(0, ratio + 1):
            for hi in (range(0, ratio + 1) if overlap else (0,)):
                if lo == 0 and hi == 0:
                    continue  # an entirely unwritten ring is all -inf -> NaN
                cases.append((overlap, ring_len, lo, hi))

    worst = 0.0
    worst_case = None
    for overlap, ring_len, lo, hi in cases:
        kv = torch.randn(B, ring_len, D)
        sw = torch.randn(B, ring_len, D)
        ref = hf_reference(kv, sw, ring_len, lo, hi, ratio, overlap)
        for fill in (0.0, float("-inf"), 1.0, -7.5):
            got = masked_version(kv, sw, ring_len, lo, hi, ratio, overlap, fill)
            d = float((got - ref).abs().max())
            if d > worst:
                worst, worst_case = d, (overlap, lo, hi, fill)
    check(worst < 1e-5,
          f"all {len(cases)} ring states x 4 init values match HF",
          f"worst max|d|={worst:.2e} at overlap/lo/hi/fill={worst_case}")

    print("\n=== 2. invariance to the buffer's init value ===")
    # The specific thing StateInitializer breaks: zeros vs -inf must not matter.
    kv = torch.randn(B, 2 * ratio, D)
    sw = torch.randn(B, 2 * ratio, D)
    a = masked_version(kv, sw, 2 * ratio, ratio, 2, ratio, True, 0.0)
    b = masked_version(kv, sw, 2 * ratio, ratio, 2, ratio, True, float("-inf"))
    check(float((a - b).abs().max()) < 1e-6,
          "zeros-init and -inf-init give identical results once masked",
          f"max|d|={float((a - b).abs().max()):.2e}")

    print("\n=== 3. cost of getting it wrong ===")
    # No mask, zero-initialised ring: what a naive ModelBuilder port computes.
    # The error grows as the ring empties, so it is worst at the START of every
    # sequence -- exactly where a short prompt or a fresh decode sits. Averaged
    # over 200 draws per fill level.
    print("    written/ring   mean relative error")
    worst_rel, finite_everywhere = 0.0, True
    for lo, hi in ((ratio, 2), (ratio, 1), (2, 1), (1, 1), (1, 0)):
        rels = []
        for _ in range(200):
            k2 = torch.randn(B, 2 * ratio, D)
            s2 = torch.randn(B, 2 * ratio, D)
            correct = hf_reference(k2, s2, 2 * ratio, lo, hi, ratio, True)
            z = torch.zeros(B, 2 * ratio, D)
            if lo:
                z[:, :lo] = s2[:, :lo]
            if hi:
                z[:, ratio:ratio + hi] = s2[:, ratio:ratio + hi]
            naive = (k2 * z.softmax(dim=1)).sum(dim=1, keepdim=True)
            finite_everywhere &= bool(torch.isfinite(naive).all())
            rels.append(float((correct - naive).abs().mean()
                              / correct.abs().mean().clamp_min(1e-9)))
        rel = sum(rels) / len(rels)
        worst_rel = max(worst_rel, rel)
        print(f"      {lo + hi}/{2 * ratio}          {rel:.1%}")
    check(finite_everywhere,
          "the wrong version is finite throughout (which is why it is dangerous)")
    check(worst_rel > 0.25,
          "unmasked zero-init is materially wrong, not a rounding difference",
          f"up to {worst_rel:.0%} relative error, worst on an empty ring")

    print("\n=== 4. decode: n_written as a traced tensor ===")
    # Decode advances the ring one slot per step, so lo/hi arrive as 0-d tensors.
    for pos in (0, 1, 3, 4, 7, 130):
        lo_t = torch.tensor(min(ratio, max(0, pos // ratio * ratio and ratio)))
        hi_t = torch.tensor((pos % ratio) + 1)
        m = cs.ring_slot_mask(2 * ratio, lo_t, hi_t, ratio, True,
                              torch.device("cpu"))
        ok = tuple(m.shape) == (1, 2 * ratio, 1)
        if not ok:
            check(False, f"tensor n_written at pos={pos}", f"shape {tuple(m.shape)}")
            break
    else:
        check(True, "tensor-valued n_written produces the right mask shape",
              "pos 0,1,3,4,7,130")

    print("\n=== 5. prefill slot accounting matches HF's branch ===")
    for seqlen in (1, 3, 4, 5, 8, 127, 128, 130):
        for overlap in (False, True):
            lo, hi = cs.n_written_prefill(seqlen, ratio, overlap)
            rem = seqlen % ratio
            cut = seqlen - rem
            exp_lo = (ratio if (overlap and cut >= ratio) else
                      (0 if overlap else rem))
            exp_hi = rem if overlap else 0
            if (lo, hi) != (exp_lo, exp_hi):
                check(False, f"n_written_prefill(seqlen={seqlen}, "
                             f"overlap={overlap})", f"{(lo, hi)} != "
                                                    f"{(exp_lo, exp_hi)}")
                break
        else:
            continue
        break
    else:
        check(True, "n_written_prefill agrees with HF's cutoff/remainder logic",
              "seqlen 1,3,4,5,8,127,128,130 x overlap on/off")

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
