"""
Pure-PyTorch (CPU) replacement for `inference/kernel.py` from
deepseek-ai/DeepSeek-V4-Flash.

The reference kernel.py uses Triton + TileLang (`import tilelang`), which
requires a CUDA/NVIDIA GPU. This module reimplements the same API in
plain PyTorch so we can run the reference model on CPU as an accuracy
oracle for the Neuron port.

Exported names match kernel.py:

  * sparse_attn(q, kv, attn_sink, topk_idxs, softmax_scale)
      Sparse multi-head attention over gathered top-k KV positions with
      learnable attn_sink. Non-fused, plain torch.gather + softmax.

  * hc_split_sinkhorn(mixes, hc_scale, hc_base, hc_mult, sinkhorn_iters, eps)
      Compute (pre, post, comb) for Hyper-Connections. `comb` is a doubly-
      stochastic matrix built via row/col normalization iterations.

  * act_quant(x, block_size, scale_fmt, scale_dtype, inplace=False)
      Block-wise FP8 quantization. With inplace=True we make it a no-op
      on the BF16 fast-path (the reference uses it for QAT simulation).

  * fp4_act_quant(x, block_size, inplace=False)
      Same as act_quant but for FP4; also a no-op on BF16 fast-path.

  * fp8_gemm(a, a_s, b, b_s, scale_dtype=torch.float32)
      Never called when all weights are BF16 (linear() dispatch chooses
      F.linear branch). Raises to catch accidental usage.

  * fp4_gemm(a, a_s, b, b_s, scale_dtype=torch.float32)
      Same as fp8_gemm — raises.
"""

from typing import Optional, Tuple

import torch


# ---- Constants (match kernel.py) ------------------------------------------

FP8 = "float8_e4m3"
FP4 = "float4_e2m1fn"
FE8M0 = "float8_e8m0fnu"
BF16 = "bfloat16"
FP32 = "float32"
INT32 = "int32"


# ---- FP8 / FP4 GEMM: not needed on BF16 fast-path -------------------------

def fp8_gemm(a, a_s, b, b_s, scale_dtype=torch.float32) -> torch.Tensor:
    """The reference dispatches to this only when the weight is FP8. On our
    BF16 path linear() picks the F.linear branch instead, so this must never
    be reached. Raise loudly if it is."""
    raise RuntimeError(
        "kernel_cpu.fp8_gemm called — the FP8 GEMM path was reached. "
        "This means a weight is still FP8-quantized; dequant_checkpoint.py "
        "must run before feeding weights to the model."
    )


def fp4_gemm(a, a_s, b, b_s, scale_dtype=torch.float32) -> torch.Tensor:
    """Same as fp8_gemm — must never be reached on BF16 fast-path."""
    raise RuntimeError(
        "kernel_cpu.fp4_gemm called — the FP4 GEMM path was reached. "
        "This means an expert weight is still FP4-quantized; "
        "dequant_checkpoint.py must run before feeding weights to the model."
    )


# ---- Activation quantization: no-op on BF16 fast-path ---------------------

def act_quant(
    x: torch.Tensor,
    block_size: int = 128,
    scale_fmt: Optional[str] = None,
    scale_dtype: torch.dtype = torch.float32,
    inplace: bool = False,
):
    """The reference calls this in Attention with `inplace=True` to simulate
    the QAT-training-time FP8 quantization on kv activations. On BF16 we skip
    the quant round-trip — it is a numerical no-op modulo the FP8 rounding
    noise, which is exactly what the model was trained to tolerate.
    """
    if inplace:
        return x  # simulate QAT round-trip as identity
    # Non-inplace path returns (y_fp8, scale). Not called in our flow.
    raise RuntimeError(
        "kernel_cpu.act_quant(inplace=False) unexpected in BF16 flow"
    )


def fp4_act_quant(
    x: torch.Tensor, block_size: int = 32, inplace: bool = False,
):
    """Same story as act_quant — inplace variant is a no-op on BF16."""
    if inplace:
        return x
    raise RuntimeError(
        "kernel_cpu.fp4_act_quant(inplace=False) unexpected in BF16 flow"
    )


# ---- Sparse attention -----------------------------------------------------

def sparse_attn(
    q: torch.Tensor,
    kv: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_idxs: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    """Non-fused reference for the sparse attention kernel.

    Args:
        q: (b, m, h, d) — queries. `m` is the number of query positions in
            this pass (prompt length during prefill, 1 during decode).
        kv: (b, n, d) — cached KV values. `n` is the size of the KV cache
            (window + compressed segments). The reference implementation
            keys and values share the same tensor.
        attn_sink: (h,) — learnable per-head sink logit added to the softmax
            denominator (i.e. weighted by exp(sink) but not by kv values).
        topk_idxs: (b, m, topk) — int32 index into `kv` per query. `-1`
            marks a masked/pad slot.
        softmax_scale: 1/sqrt(d) typically.

    Returns:
        (b, m, h, d) — attention output.
    """
    b, m, h, d = q.shape
    _, n, _ = kv.shape
    _, _, topk = topk_idxs.shape
    device = q.device
    q_dtype = q.dtype

    # Build a mask for the -1 (pad) slots
    idx_i32 = topk_idxs
    pad_mask = idx_i32 < 0                       # (b, m, topk)
    gather_idx = idx_i32.clamp_min(0).long()     # (b, m, topk); use 0 for pad

    # Gather KV: kv[b, gather_idx[b, m, t], :] -> (b, m, topk, d)
    b_range = torch.arange(b, device=device).view(b, 1, 1).expand(b, m, topk)
    kv_gathered = kv[b_range, gather_idx]         # (b, m, topk, d)

    # Compute Q @ K^T over the d dimension, per head.
    # q: (b, m, h, d) ; kv_gathered: (b, m, topk, d)
    # scores: (b, m, h, topk)
    scores = torch.einsum("bmhd,bmtd->bmht", q.float(), kv_gathered.float())
    scores = scores * softmax_scale

    # Mask out pad slots
    if pad_mask.any():
        scores = scores.masked_fill(pad_mask.unsqueeze(2), float("-inf"))

    # Softmax with an attn_sink slot: numerically stable with logsumexp.
    # Effectively softmax over [scores || sink] but we don't materialize the
    # concat — the sink only contributes to the denominator, its "value" is 0.
    max_scores = scores.amax(dim=-1, keepdim=True)  # (b, m, h, 1)
    # sink term: exp(sink - max) has shape (b, m, h, 1) via broadcasting
    sink_bias = attn_sink.float().view(1, 1, h, 1)
    max_with_sink = torch.maximum(max_scores, sink_bias)
    scores = scores - max_with_sink
    sink_term = (sink_bias - max_with_sink).exp()   # (b, m, h, 1)
    weights = scores.exp()                          # (b, m, h, topk)
    if pad_mask.any():
        weights = weights.masked_fill(pad_mask.unsqueeze(2), 0.0)
    denom = weights.sum(dim=-1, keepdim=True) + sink_term
    weights = weights / denom

    # Weighted sum of KV values (same tensor as keys in this model).
    out = torch.einsum("bmht,bmtd->bmhd", weights, kv_gathered.float())
    return out.to(q_dtype)


# ---- Hyper-Connections Sinkhorn ------------------------------------------

def hc_split_sinkhorn(
    mixes: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    hc_mult: int = 4,
    sinkhorn_iters: int = 20,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split `mixes` into (pre, post, comb) with `comb` a doubly-stochastic mix.

    mixes: (b, s, (2 + hc_mult) * hc_mult) FP32
    hc_scale: (3,) FP32
    hc_base: ((2 + hc_mult) * hc_mult,) FP32

    Returns pre (b, s, hc_mult), post (b, s, hc_mult), comb (b, s, hc_mult, hc_mult).
    Layout matches kernel.py:
      mixes[..., :hc]                     -> pre  (sigmoid + eps)
      mixes[..., hc : 2*hc]               -> post (2 * sigmoid)
      mixes[..., 2*hc : (2 + hc) * hc]    -> comb (softmax-normalized rows,
                                                   then Sinkhorn 20 iters)
    """
    hc = hc_mult
    # Fold batch/seq into a leading N dim
    orig_shape = mixes.shape[:-1]
    m = mixes.reshape(-1, mixes.size(-1)).float()  # (N, mix_hc)
    N = m.size(0)

    # pre
    pre = torch.sigmoid(m[:, :hc] * hc_scale[0] + hc_base[:hc]) + eps
    # post
    post = 2.0 * torch.sigmoid(m[:, hc:2 * hc] * hc_scale[1] + hc_base[hc:2 * hc])
    # comb: (N, hc, hc)
    comb_flat = m[:, 2 * hc:] * hc_scale[2] + hc_base[2 * hc:]
    comb = comb_flat.reshape(N, hc, hc)

    # Row softmax + eps
    comb = torch.softmax(comb, dim=-1) + eps
    # Column normalization
    comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    # sinkhorn_iters - 1 more (row -> col) sweeps
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)

    return (
        pre.reshape(*orig_shape, hc),
        post.reshape(*orig_shape, hc),
        comb.reshape(*orig_shape, hc, hc),
    )
