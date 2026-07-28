#!/usr/bin/env python
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Probe the XLA primitives a decode graph needs, before building one.

Decode differs from prefill in exactly one way that matters: `start_pos` must
be a *runtime value* rather than a compile-time constant, or we would need one
NEFF per position. Everything the reference model does with `start_pos` as a
Python int has to become a tensor op:

    freqs_cis[p : p+1]              -> index_select(freqs_cis, 0, p)
    kv_cache[:, p % win] = kv       -> index_copy_ / scatter with tensor index
    arange(0, (p+1) // ratio)       -> fixed length + -1 masking
    if (p+1) % ratio == 0: write    -> masked select on the write value

Each of those is a separate bet on trn2 support, and the last port taught us
that guessing is expensive: `torch.topk` lowers to HLO `sort` which trn2
rejects outright, and `as_strided` has no XLA implementation at all. So probe
the primitives on tiny shapes first — a few KB, no model weights — and only
then write the real thing.

Run:
    source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
    cd contrib/models/DeepSeek-V4-Flash
    python test/spike/test_decode_primitives.py
"""

import os
import sys
import traceback

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.abspath(os.path.join(_HERE, "..", "..", "src"))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

# Keep the probe tiny: these run on one core, no TP, no weights.
WIN = 8
RATIO = 4
DIM = 4
MAX_COMP = 4


# --- The candidate implementations, as plain torch on whatever device ---

def freqs_at(freqs, pos):
    """freqs_cis[p:p+1] with p a tensor. index_select keeps the axis."""
    return torch.index_select(freqs, 0, pos.reshape(1))


def window_idxs(pos, win):
    """Closed form replacing HF's three-way branch on start_pos.

    Slot j of a ring buffer of size `win` holds absolute position
    p - ((p - j) mod win), which is a real position iff it is >= 0, i.e. iff
    j <= p. sparse_attn softmaxes over the gathered slots, so the *order* of
    the index list is irrelevant — only the set and the -1 mask matter. That
    is what lets one expression cover both of HF's decode branches:
      * p >= win-1: every slot valid (HF emits a rotation of arange(win),
        a permutation of what we emit)
      * 0 < p < win-1: slots 0..p valid, rest -1 (identical to HF)
    """
    j = torch.arange(win, device=pos.device)
    return torch.where(j <= pos, j, torch.full_like(j, -1))


def compress_idxs(pos, ratio, max_comp, offset):
    """HF: arange(0, (p+1)//ratio) + offset — a *variable-length* list.

    A graph cannot have a data-dependent output shape, so emit the full
    max_comp and mark the not-yet-written tail with -1.
    """
    n_valid = (pos + 1) // ratio
    j = torch.arange(max_comp, device=pos.device)
    return torch.where(j < n_valid, j + offset, torch.full_like(j, -1))


def ring_write(cache, pos, value, win):
    """kv_cache[:, p % win] = value, with p a tensor.

    index_copy insists on a Long index on XLA ("Copy index is expected to be
    of scalar type Long, but it is Int"), while start_pos arrives as int32
    because that is what torch.jit.trace accepts cleanly as a scalar input.
    So cast at the boundary.
    """
    slot = (pos % win).reshape(1).long()
    return cache.index_copy(1, slot, value)


def masked_ring_write(cache, pos, value, ratio, max_comp):
    """The Compressor's conditional write.

    HF returns early when (p+1) % ratio != 0, so the cache write happens on
    only one step in `ratio`. An early return changes the graph, so instead
    always write, but write back the old value when the condition is false.
    """
    should = ((pos + 1) % ratio) == 0
    slot = (pos // ratio).clamp(max=max_comp - 1).reshape(1).long()
    old = torch.index_select(cache, 1, slot)
    merged = torch.where(should, value, old)
    return cache.index_copy(1, slot, merged)


CASES = {
    "index_select (freqs_cis[p:p+1])": lambda dev: (
        freqs_at(torch.arange(16, dtype=torch.float32, device=dev).reshape(8, 2),
                 torch.tensor(5, dtype=torch.int32, device=dev))
    ),
    "window_idxs closed form": lambda dev: (
        window_idxs(torch.tensor(3, dtype=torch.int32, device=dev), WIN)
    ),
    "compress_idxs fixed-shape": lambda dev: (
        compress_idxs(torch.tensor(7, dtype=torch.int32, device=dev),
                      RATIO, MAX_COMP, WIN)
    ),
    "index_copy ring write": lambda dev: (
        ring_write(torch.zeros(1, WIN, DIM, device=dev),
                   torch.tensor(3, dtype=torch.int32, device=dev),
                   torch.ones(1, 1, DIM, device=dev), WIN)
    ),
    "masked conditional write": lambda dev: (
        masked_ring_write(torch.zeros(1, MAX_COMP, DIM, device=dev),
                          torch.tensor(3, dtype=torch.int32, device=dev),
                          torch.ones(1, 1, DIM, device=dev),
                          RATIO, MAX_COMP)
    ),
    # sparse_attn gathers kv[b, idx] with idx from the above; the gather itself
    # already works in prefill, but with a *traced* pos the indices are no
    # longer compile-time constants. Check the gather still lowers.
    "gather with runtime indices": lambda dev: (
        torch.zeros(1, WIN, DIM, device=dev)[
            torch.zeros(1, 1, WIN, dtype=torch.long, device=dev),
            window_idxs(torch.tensor(3, dtype=torch.int32, device=dev),
                        WIN).clamp_min(0).reshape(1, 1, WIN),
        ]
    ),
}


def main():
    print("=" * 70)
    print("CPU reference vs XLA, decode primitives")
    print("=" * 70)

    # CPU first: establishes the expected values, and catches a bug in the
    # candidate expression itself before XLA muddies the picture.
    cpu = {}
    for name, fn in CASES.items():
        try:
            cpu[name] = fn(torch.device("cpu"))
            print(f"[cpu ] {name}: ok  {tuple(cpu[name].shape)}")
        except Exception as e:
            cpu[name] = None
            print(f"[cpu ] {name}: FAILED {type(e).__name__}: {e}")

    # Semantic check of the closed forms against HF's own branches, on CPU.
    print("\n" + "-" * 70)
    print("closed form vs HF reference semantics")
    print("-" * 70)
    ok = True
    for p in range(1, 20):
        pt = torch.tensor(p, dtype=torch.int32)
        got = set(window_idxs(pt, WIN).tolist()) - {-1}
        # HF's own two decode branches, verbatim
        if p >= WIN - 1:
            pm = p % WIN
            want = set(torch.cat([torch.arange(pm + 1, WIN),
                                  torch.arange(0, pm + 1)]).tolist())
        else:
            want = set(range(p + 1))
        if got != want:
            print(f"  p={p}: MISMATCH got={sorted(got)} want={sorted(want)}")
            ok = False
    print(f"  window_idxs matches HF for p=1..19: {ok}")

    ok2 = True
    for p in range(1, 20):
        pt = torch.tensor(p, dtype=torch.int32)
        got = [v for v in compress_idxs(pt, RATIO, MAX_COMP, WIN).tolist()
               if v != -1]
        want = (torch.arange(0, (p + 1) // RATIO) + WIN).tolist()
        # ours is capped at max_comp; only compare where HF fits
        if len(want) <= MAX_COMP and got != want:
            print(f"  p={p}: MISMATCH got={got} want={want}")
            ok2 = False
    print(f"  compress_idxs matches HF for p=1..19: {ok2}")

    # Now XLA.
    print("\n" + "-" * 70)
    print("XLA lowering (this is the part that can reject an op)")
    print("-" * 70)
    import torch_xla.core.xla_model as xm
    dev = xm.xla_device()

    results = {}
    for name, fn in CASES.items():
        try:
            out = fn(dev)
            xm.mark_step()
            out_cpu = out.cpu()
            if cpu[name] is not None and out_cpu.shape == cpu[name].shape:
                same = torch.equal(out_cpu, cpu[name].to(out_cpu.dtype))
            else:
                same = None
            results[name] = ("ok", same)
            print(f"[xla ] {name}: ok  matches_cpu={same}")
        except Exception as e:
            results[name] = ("fail", str(e))
            print(f"[xla ] {name}: FAILED {type(e).__name__}: {e}")
            traceback.print_exc(limit=3)

    print("\n" + "=" * 70)
    failed = [n for n, (s, _) in results.items() if s == "fail"]
    mismatch = [n for n, (s, m) in results.items() if s == "ok" and m is False]
    if failed:
        print(f"BLOCKED: {len(failed)} primitive(s) do not lower on XLA:")
        for n in failed:
            print(f"  - {n}")
    if mismatch:
        print(f"WRONG: {len(mismatch)} primitive(s) lower but disagree with CPU:")
        for n in mismatch:
            print(f"  - {n}")
    if not failed and not mismatch and ok and ok2:
        print("ALL CLEAR: decode can be built from these primitives.")
    print("=" * 70)
    return 1 if (failed or mismatch or not ok or not ok2) else 0


if __name__ == "__main__":
    sys.exit(main())
