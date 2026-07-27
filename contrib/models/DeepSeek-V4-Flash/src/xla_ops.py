# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""XLA/Neuron-safe replacements for ops neuronx-cc rejects on trn2.

`torch.topk` lowers to HLO `sort`, which the compiler refuses:

    [ERROR] [NCC_EVRF029] Operation sort is not supported on trn2. Use
    supported equivalent operation like TopK or replace it with an alternate
    implementation via Neuron Kernel Interface (NKI).
    %sort.16 = (f32[16,256], s32[16,256]) sort(%p0, %iota.3), dimensions={1}

DeepSeek-V4-Flash hits this in two places:

  * `Gate.forward` -> `scores.topk(6, dim=-1)` for every layer at or past
    `n_hash_layers` (i.e. layer 3 onwards; layers 0-2 use hash routing).
  * `Indexer.forward` -> `index_score.topk(min(index_topk, end_pos // ratio))`
    for layers whose compress_ratio == 4 (layer 2 onwards).

`argmax` lowers to a plain reduce and is supported, so a k-step
argmax-and-mask loop gives bit-identical results to `torch.topk` for small k
at the cost of k reduction passes.
"""

from typing import Tuple

import torch


def topk_indices(scores: torch.Tensor, k: int) -> torch.Tensor:
    """Indices of the k largest elements along the last dim, descending.

    Equivalent to `scores.topk(k, dim=-1)[1]` whenever the k-th and (k+1)-th
    values differ, which is the case for real gate scores. On exact ties the
    two disagree, but only because `torch.topk`'s tie-break is unspecified
    (an all-equal row can come back as [10, 11, 12, 9]); this returns the
    lowest index, which is deterministic. Either choice picks equal values, so
    the selected *scores* — all the gate consumes — are identical.

    Masking is by position rather than by value, so duplicated maxima stay
    individually selectable. Costs k argmax reductions, so keep k small: this
    is meant for the MoE gate (k=6), not for large-k selection.
    """
    n = scores.size(-1)
    if k > n:
        raise ValueError(f"k={k} exceeds last-dim size {n}")
    if k == n:
        # Selecting everything still has to come back in descending order,
        # so there is no shortcut here; fall through to the loop.
        pass

    neg_inf = torch.finfo(scores.dtype).min
    positions = torch.arange(n, device=scores.device).view(
        *([1] * (scores.dim() - 1)), n
    )

    remaining = scores
    picked = []
    for _ in range(k):
        best = remaining.argmax(dim=-1, keepdim=True)
        picked.append(best)
        # Knock out the position just taken (not the value: duplicates must
        # stay selectable, exactly as torch.topk would).
        remaining = torch.where(
            positions == best, torch.full_like(remaining, neg_inf), remaining,
        )
    return torch.cat(picked, dim=-1)


def topk(scores: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """(values, indices) equivalent of `scores.topk(k, dim=-1)`."""
    idx = topk_indices(scores, k)
    return scores.gather(-1, idx), idx


def topk_indices_unordered(scores: torch.Tensor, k: int) -> torch.Tensor:
    """Indices of the k largest elements, order unspecified.

    For the Indexer's sparse-attention selection the returned indices are only
    used as gather positions into KV plus an additive mask, so their order does
    not affect the result. That admits a cheap exact path when k covers the
    whole candidate axis, which is the common case at short sequence lengths:
    with compress_ratio 4 and index_topk 512, `k == n` for every seq_len up to
    2048.

    Beyond that this falls back to the k-step loop, which is impractical for
    k=512. A real large-k selection needs an NKI kernel; we raise rather than
    silently emit a graph that takes minutes to compile.
    """
    n = scores.size(-1)
    if k >= n:
        # Broadcast-add a zero tensor rather than .expand(): expand lowers to
        # as_strided, which torch-xla has no implementation for.
        positions = torch.arange(n, device=scores.device).view(
            *([1] * (scores.dim() - 1)), n
        )
        zeros = torch.zeros(
            *scores.shape[:-1], 1, dtype=positions.dtype, device=scores.device,
        )
        return positions + zeros
    if k > 32:
        raise NotImplementedError(
            f"topk_indices_unordered with k={k} of n={n} needs {k} argmax "
            "passes, which is impractical. This happens once the compressed "
            "KV axis grows past index_topk (seq_len > index_topk * "
            "compress_ratio, i.e. > 2048 with the shipped config). Provide an "
            "NKI top-k kernel for this range."
        )
    return topk_indices(scores, k)
