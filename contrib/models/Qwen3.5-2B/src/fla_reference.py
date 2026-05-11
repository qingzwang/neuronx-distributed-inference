# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Self-contained PyTorch reference implementations of the
flash-linear-attention (FLA) gated-delta-rule kernels used by
Qwen3.5-2B's DeltaNet layers.

The official `fla` package wraps Triton kernels that target CUDA. On
Trainium we cannot run those, but the math is small enough that a fp32
PyTorch reference is useful for:

  * accuracy checks of the NKI kernels (fp32 reference vs. NKI bf16/fp32
    outputs, both on the same inputs)
  * a CPU fallback so unit tests can run without a Neuron device
  * a known-good baseline to compare new NKI kernels against during
    development

The two functions exposed here mirror `fla.ops.gated_delta_rule`:

  * `chunk_gated_delta_rule(q, k, v, g, beta, ...)` -- chunked Neumann
    forward used for context encoding (prefill)
  * `recurrent_gated_delta_rule(q, k, v, g, beta, ...)` -- per-token
    sequential recurrence used for token generation

Both accept and return tensors in (B, H, S, D) layout, matching FLA. Both
operate fully in fp32 internally. Both assume queries/keys are NOT
pre-l2-normed (they l2-norm internally), to keep the call signature
identical to FLA's reference.
"""

from typing import Optional, Tuple

import torch
import torch.nn.functional as F


def _l2norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return F.normalize(x, p=2, dim=-1, eps=eps)


@torch.no_grad()
def recurrent_gated_delta_rule(
    q: torch.Tensor,        # (B, H, S, K)
    k: torch.Tensor,        # (B, H, S, K)
    v: torch.Tensor,        # (B, H, S, V)
    g: torch.Tensor,        # (B, H, S)   -- log-decay (g <= 0)
    beta: torch.Tensor,     # (B, H, S)   -- write gate in [0, 1]
    initial_state: Optional[torch.Tensor] = None,  # (B, H, K, V)
    output_final_state: bool = True,
    use_qk_l2norm_in_kernel: bool = True,
    scale: Optional[float] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Per-token recurrent gated delta rule (fp32 reference).

    Implements the recurrence used by Qwen3.5 DeltaNet:

        S_t = exp(g_t) * S_{t-1}
        delta_t = (v_t - S_t^T @ k_t) * beta_t
        S_t' = S_t + outer(k_t, delta_t)
        o_t  = S_t'^T @ q_t

    Args mirror `fla.ops.gated_delta_rule.fused_recurrent_gated_delta_rule`.

    Returns (output, final_state) with shapes (B, H, S, V) and (B, H, K, V).
    """
    B, H, S, K = q.shape
    V = v.shape[-1]

    if use_qk_l2norm_in_kernel:
        q = _l2norm(q.float())
        k = _l2norm(k.float())
    else:
        q = q.float()
        k = k.float()
    v = v.float()
    g = g.float()
    beta = beta.float()

    if scale is None:
        scale = 1.0 / (K ** 0.5)
    q = q * scale

    if initial_state is None:
        state = q.new_zeros(B, H, K, V)
    else:
        state = initial_state.float().clone()

    output = q.new_zeros(B, H, S, V)
    for t in range(S):
        q_t = q[:, :, t]                                   # (B,H,K)
        k_t = k[:, :, t]                                   # (B,H,K)
        v_t = v[:, :, t]                                   # (B,H,V)
        g_t = g[:, :, t].exp().view(B, H, 1, 1)            # (B,H,1,1)
        beta_t = beta[:, :, t].view(B, H, 1)               # (B,H,1)

        state = state * g_t                                # decay
        kv_mem = (state * k_t.unsqueeze(-1)).sum(dim=-2)   # (B,H,V)
        delta = (v_t - kv_mem) * beta_t                    # (B,H,V)
        state = state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        output[:, :, t] = (state * q_t.unsqueeze(-1)).sum(dim=-2)

    if output_final_state:
        return output, state
    return output, None


@torch.no_grad()
def chunk_gated_delta_rule(
    q: torch.Tensor,        # (B, H, S, K)
    k: torch.Tensor,        # (B, H, S, K)
    v: torch.Tensor,        # (B, H, S, V)
    g: torch.Tensor,        # (B, H, S)
    beta: torch.Tensor,     # (B, H, S)
    chunk_size: int = 64,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = True,
    use_qk_l2norm_in_kernel: bool = True,
    scale: Optional[float] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Chunked gated delta rule (fp32 reference, no Neumann approximation).

    Mirrors the math in `fla.ops.gated_delta_rule.chunk_gated_delta_rule`
    and HuggingFace's `torch_chunk_gated_delta_rule`. Each chunk is
    processed with an exact recurrence (the Neumann series is not used --
    we resolve the per-chunk cumulative-decay matrix exactly via a small
    sequential pass).
    """
    B, H, S, K = q.shape
    V = v.shape[-1]

    if use_qk_l2norm_in_kernel:
        q = _l2norm(q.float())
        k = _l2norm(k.float())
    else:
        q = q.float()
        k = k.float()
    v = v.float()
    g = g.float()
    beta = beta.float()

    if scale is None:
        scale = 1.0 / (K ** 0.5)
    q = q * scale

    pad = (chunk_size - S % chunk_size) % chunk_size
    if pad:
        q = F.pad(q, (0, 0, 0, pad))
        k = F.pad(k, (0, 0, 0, pad))
        v = F.pad(v, (0, 0, 0, pad))
        g = F.pad(g, (0, pad))
        beta = F.pad(beta, (0, pad))
    Stot = S + pad
    nC = Stot // chunk_size

    v_beta = v * beta.unsqueeze(-1)
    k_beta = k * beta.unsqueeze(-1)

    # Reshape to (B, H, nC, chunk_size, ...)
    qC = q.reshape(B, H, nC, chunk_size, K)
    kC = k.reshape(B, H, nC, chunk_size, K)
    vC = v.reshape(B, H, nC, chunk_size, V)
    kbC = k_beta.reshape(B, H, nC, chunk_size, K)
    vbC = v_beta.reshape(B, H, nC, chunk_size, V)
    gC = g.reshape(B, H, nC, chunk_size)

    # Cumulative-decay within chunk. gc[i] - gc[j] gives log(prod g[j..i]).
    gc = gC.cumsum(dim=-1)
    decay = (gc.unsqueeze(-1) - gc.unsqueeze(-2)).clamp(max=0.0).exp()  # (B,H,nC,c,c)
    lower = torch.tril(torch.ones(chunk_size, chunk_size, dtype=q.dtype, device=q.device), diagonal=-1)
    lower_diag = torch.tril(torch.ones(chunk_size, chunk_size, dtype=q.dtype, device=q.device), diagonal=0)

    # Solve for the per-chunk correction matrix N such that
    #   value_corr  = N @ v_beta
    #   k_cumdecay = N @ (k_beta * exp(gc))
    # (this matches the Neumann-resolved form in nki_deltanet_fused.py).
    QK = kbC @ kC.transpose(-1, -2)            # (B,H,nC,c,c)
    A = -(QK * decay) * lower                  # strictly lower-triangular
    # Resolve N = (I - A)^-1 @ I exactly. A is strictly lower triangular so
    # (I - A)^-1 has a finite Neumann sum we can compute by sequential
    # forward substitution in chunk_size steps (no nilpotency-exponent loop).
    eye = torch.eye(chunk_size, dtype=q.dtype, device=q.device)
    N = eye[None, None, None, :, :].expand(B, H, nC, chunk_size, chunk_size).contiguous()
    # forward-substitute: row i picks up A[i, :i] @ N[:i, :]
    for i in range(1, chunk_size):
        # N[i, :] = e_i + A[i, :i] @ N[:i, :]
        N[..., i, :] = N[..., i, :] + (A[..., i, :i].unsqueeze(-1) * N[..., :i, :]).sum(-2)

    value_corr = N @ vbC
    k_cumdecay = N @ (kbC * gc.unsqueeze(-1).exp())

    if initial_state is None:
        state = q.new_zeros(B, H, K, V)
    else:
        state = initial_state.float().clone()

    upper_strict = torch.triu(torch.ones(chunk_size, chunk_size, dtype=q.dtype, device=q.device), diagonal=1).bool()
    out = q.new_zeros(B, H, Stot, V)
    for ci in range(nC):
        q_i = qC[:, :, ci]
        k_i = kC[:, :, ci]
        gc_i = gc[:, :, ci]                              # (B,H,c)
        attn_i = (q_i @ k_i.transpose(-1, -2)) * decay[:, :, ci]
        attn_i = attn_i.masked_fill(upper_strict, 0.0)

        v_prime = k_cumdecay[:, :, ci] @ state           # (B,H,c,V)
        v_new = value_corr[:, :, ci] - v_prime

        attn_inter = (q_i * gc_i.unsqueeze(-1).exp()) @ state   # (B,H,c,V)
        out_chunk = attn_inter + attn_i @ v_new
        out[:, :, ci * chunk_size:(ci + 1) * chunk_size] = out_chunk

        gl = gc_i[..., -1:]                              # (B,H,1)
        gl_minus_gc = (gl - gc_i).clamp(max=0.0).exp()   # (B,H,c)
        k_weighted = k_i * gl_minus_gc.unsqueeze(-1)     # (B,H,c,K)
        # state: (B,H,K,V); gl.exp() broadcasts as scalar over K,V
        state = state * gl.exp().unsqueeze(-1) + k_weighted.transpose(-1, -2) @ v_new

    out = out[:, :, :S]
    if output_final_state:
        return out, state
    return out, None


# Convenience: tiny test that fp32 chunk == fp32 recurrent.
def _self_test():
    torch.manual_seed(0)
    B, H, S, K, V = 1, 2, 256, 16, 16
    q = torch.randn(B, H, S, K)
    k = torch.randn(B, H, S, K)
    v = torch.randn(B, H, S, V)
    g = -F.softplus(torch.randn(B, H, S))      # negative log-decay
    beta = torch.sigmoid(torch.randn(B, H, S))
    o_rec, s_rec = recurrent_gated_delta_rule(q, k, v, g, beta)
    o_ch, s_ch = chunk_gated_delta_rule(q, k, v, g, beta, chunk_size=64)
    err_o = (o_rec - o_ch).abs().max().item()
    err_s = (s_rec - s_ch).abs().max().item()
    print(f"recurrent vs chunked  max|err_o|={err_o:.4e}  max|err_s|={err_s:.4e}")
    assert err_o < 1e-4 and err_s < 1e-4, (err_o, err_s)


if __name__ == "__main__":
    _self_test()
