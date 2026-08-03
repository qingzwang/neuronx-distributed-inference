# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compression-ring state that does not depend on its buffer's initial value.

The problem
-----------
HF initialises the compressor's score ring to -inf:

    register_buffer("score_state",
                    torch.full((B, coff*ratio, coff*head_dim), float("-inf")))

That value is load-bearing. `score_state` is consumed by `softmax(dim=1)` over the
ring, so a slot that has not been written yet must contribute *zero weight*. A
zero-initialised buffer instead contributes `exp(0)` — a uniform vote for a slot
holding no data. The result stays finite and looks plausible; it is just wrong by
a factor of ~2 in magnitude (measured: compressed-output absmean 0.62 correct vs
0.31 with zero init).

Nothing in the reference is wrong. What breaks is running this model through NxD's
`ModelBuilder`, where device state is built by `StateInitializer`, which hardcodes

    torch.zeros(shape, dtype=..., device=f"privateuseone:{rank}")

(neuronx_distributed/trace/nxd_model/base_nxd_model.py:31). There is no hook to
change the fill. So a port that relies on the buffer's init value is relying on
something that layer of the stack will silently overwrite.

The fix
-------
Stop depending on the init value. Track how many ring slots hold real data and
apply the mask inside the graph:

    masked = where(slot_index < n_written, score, -inf)
    weights = masked.softmax(dim=1)

Now a zero-filled buffer and an -inf-filled buffer give identical results, because
the unwritten tail is masked before the softmax either way. `n_written` is derived
from the position, which is already a graph input, so this adds one comparison per
compression step and no new state.

This is deliberately the more robust of the two options in NXDI_PORT_DESIGN.md:
overriding the buffer's construction would work today but leaves correctness
depending on no other layer re-initialising it. Masking in the graph cannot be
undone from outside.

`ring_slot_mask` is written to serve both paths:
  * prefill fills slots 0..n_written-1 in one shot, n_written a Python int
  * decode advances one slot at a time, n_written a traced tensor
"""

import torch


def n_written_prefill(seqlen, ratio, overlap):
    """How many ring slots hold real data after a prefill of `seqlen` tokens.

    Mirrors HF's prefill branch. With `overlap` the ring is
    [overlap window | current window], each `ratio` long, and prefill seeds the
    first half with the last full block before the cutoff (when there is one) and
    the second half with the ragged tail:

        if overlap and cutoff >= ratio:  kv_state[:, :ratio]        <- full block
        if remainder > 0:                kv_state[:, off:off+rem]   <- tail

    So the written slots are not a simple prefix when overlap is on: the first
    half is written iff `cutoff >= ratio`, and the second half holds `remainder`
    entries. Returns (lo_written, hi_written) counts for the two halves, which
    `ring_slot_mask` turns into a mask. Without overlap there is one half.
    """
    remainder = seqlen % ratio
    cutoff = seqlen - remainder
    if not overlap:
        return (remainder, 0)
    return (ratio if cutoff >= ratio else 0, remainder)


def ring_slot_mask(ring_len, lo_written, hi_written, ratio, overlap, device,
                   dtype=torch.float32):
    """Additive mask over the ring: 0 where a slot is real, -inf where it is not.

    Shaped `(1, ring_len, 1)` so it broadcasts over batch and feature dims onto
    `score_state`, and added rather than multiplied so it composes with whatever
    the score already is.

    `lo_written` / `hi_written` may be Python ints (prefill, known at trace time)
    or 0-d tensors (decode, position-dependent). Both lower identically here
    because the comparison is against an arange, not a slice.
    """
    slots = torch.arange(ring_len, device=device)
    if not overlap:
        real = slots < _as_tensor(lo_written, slots)
    else:
        # [0, ratio) is the overlap half, [ratio, 2*ratio) the current half.
        in_lo = slots < _as_tensor(lo_written, slots)
        in_hi = (slots >= ratio) & (
            slots - ratio < _as_tensor(hi_written, slots))
        real = in_lo | in_hi
    neg_inf = torch.finfo(dtype).min
    # Arithmetic select rather than torch.where: where() does not lower inside
    # ModelBuilder's generate_hlo (verified — all operands correct, still raises
    # "size of tensor a (N) must match tensor b (0)"), while mul/add does.
    keep = real.to(dtype)
    return ((1.0 - keep) * neg_inf).reshape(1, ring_len, 1)


def _as_tensor(v, like):
    """Python int or 0-d tensor -> something comparable against `like`."""
    if isinstance(v, torch.Tensor):
        return v.to(like.device)
    return torch.tensor(v, device=like.device)


def compress_with_mask(kv_state, score_state, mask, dim=1):
    """The compression HF writes as `(kv * score.softmax(dim)).sum(dim)`.

    With `mask` added before the softmax, so unwritten ring slots contribute
    exactly zero regardless of what the buffer was initialised to.
    """
    weights = (score_state + mask.to(score_state.dtype)).softmax(dim=dim)
    return (kv_state * weights).sum(dim=dim, keepdim=True)
