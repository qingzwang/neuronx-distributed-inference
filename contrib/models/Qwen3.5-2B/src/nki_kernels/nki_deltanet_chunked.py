# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""NKI per-chunk DeltaNet kernel for CTE (context encoding / prefill).

Single-chunk kernel: processes one chunk (128 tokens) with Neumann-series
power-doubling for intra-chunk correction. The caller loops over chunks
in PyTorch, passing state between calls.

This is the per-chunk analog of `_deltanet_fused_kernel` (see
`nki_deltanet_fused.py`). The fused kernel keeps state in SBUF across
chunks but is broken under multi-bucket compilation in the current SDK
(produces empty output / EOS for any compiled seq_len in {384, 512, 1024}).
This per-chunk kernel sidesteps that bug by reading/writing state to HBM
between chunks — slightly more DMA traffic but the chunk math itself is
identical and the compiler handles the simple case correctly.

Math: same log-sum-exp-stable formulation as the fused kernel. The
earlier version of this kernel used `exp(gc[i]) * exp(-gc[j])` directly
(two scalar broadcasts) which overflows in fp32 when |gc| is large
(typical for prompts >~30 tokens where vision-token decay accumulates).
The fix is to compute `exp(min(gc[i] - gc[j], 0))` instead — same math
on the strictly-lower-triangular valid side, but every intermediate is
bounded in [0, 1].

Each kernel call:
  - Takes one chunk of data: q, k, v, beta_broadcast, gc_per_token, gl_scalar
  - Takes recurrent state_in (128, 128)
  - Returns chunk output (128, 128) and state_out (128, 128)

Inputs are all full 128x128 tiles. The caller ensures:
  - q is l2-normed and scaled by 1/sqrt(K)
  - k is l2-normed
  - beta_broadcast[i, :] = sigmoid(b_t[i]) (broadcast across feature dim)
  - gc_per_token: (128, 1) column with per-token gc[t] (cumsum within chunk)
  - gl_scalar: (1, 1) scalar with gc[chunk_size - 1]

NKI v3 (SDK 2.29, NKI 0.3.0). Uses nki.* namespace.
"""

import nki
import nki.isa as nisa
import nki.language as nl

P_MAX = 128
_BROADCAST_MASK = [0] * 32


@nki.jit
def deltanet_chunk_step(
    query,           # (128, 128) float32 -- one chunk, l2-normed+scaled
    key,             # (128, 128) float32 -- one chunk, l2-normed
    value,           # (128, 128) float32 -- one chunk
    beta_broadcast,  # (128, 128) float32 -- write gate broadcast to feature dim
    g_cumsum,        # (128, 128) float32 -- per-token gc (cumsum), broadcast to feature dim
    g_last,          # (128, 128) float32 -- gl scalar, broadcast everywhere
    state_in,        # (128, 128) float32 -- recurrent state from previous chunk
    lower_mask,      # (128, 128) float32 -- strict lower triangular
    identity,        # (128, 128) float32 -- identity matrix
    lower_mask_diag, # (128, 128) float32 -- lower tri with diagonal
):
    """Process one chunk of DeltaNet, returning (output, new_state).

    All exponents are clamped to (-inf, 0] before exp(), so every
    intermediate is in [0, 1] regardless of |gc|. This is the fix that
    made the fused kernel stable for vision-language prompts; this
    kernel inherits the same fix.
    """
    C, dim = query.shape  # 128, 128

    # Output tensors in HBM
    output = nl.ndarray((P_MAX, dim), dtype=query.dtype, buffer=nl.shared_hbm)
    state_out = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.shared_hbm)

    # Load all inputs into SBUF
    q_c = nl.ndarray((P_MAX, dim), dtype=query.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=q_c, src=query)

    k_c = nl.ndarray((P_MAX, dim), dtype=key.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=k_c, src=key)

    v_c = nl.ndarray((P_MAX, dim), dtype=value.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=v_c, src=value)

    beta_c = nl.ndarray((P_MAX, dim), dtype=beta_broadcast.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=beta_c, src=beta_broadcast)

    gc_c = nl.ndarray((P_MAX, dim), dtype=g_cumsum.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=gc_c, src=g_cumsum)

    gl_c = nl.ndarray((P_MAX, dim), dtype=g_last.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=gl_c, src=g_last)

    state = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=state, src=state_in)

    # Load masks once
    eye = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=eye, src=identity)

    Lmask = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=Lmask, src=lower_mask)

    Lmask_d = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=Lmask_d, src=lower_mask_diag)

    # ============================================================
    # Build log-sum-exp-style decay matrix.
    #
    #   decay_mat[i, j] = exp(min(gc[i] - gc[j], 0))
    #
    # gc_c is (P_MAX, dim) but only column 0 is meaningful (input was
    # (128, 1) broadcast). To get gc[i] - gc[j] we need:
    #   gc_col_bcast[i, j] = gc[i]                    (constant in j)
    #   gc_row_bcast[i, j] = gc[j]                    (constant in i)
    # gc_col_bcast comes from gc_c column 0 broadcast to all columns.
    # gc_row_bcast comes from transposing gc_c (so row 0 is gc^T) and
    # then broadcasting partition 0 to all partitions via stream shuffle.
    # ============================================================
    # Take column 0 of gc_c — this is gc as (P_MAX, 1)
    gc_col = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=gc_col, src=gc_c[0:P_MAX, 0:1])

    # gc_col_bcast[i, j] = gc[i] (constant in j)
    ones_PxP = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=ones_PxP, value=1.0)
    gc_col_bcast = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(
        dst=gc_col_bcast,
        data=ones_PxP,
        op0=nl.multiply,
        operand0=gc_col,
        engine=nisa.vector_engine,
    )

    # Build gc_row (1, P_MAX): transpose gc_col into row form
    gc_padded = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=gc_padded, value=0.0)
    nisa.tensor_copy(dst=gc_padded[0:P_MAX, 0:1], src=gc_col)

    gc_tp_psum = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_transpose(dst=gc_tp_psum, data=gc_padded)

    gc_row = nl.ndarray((1, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=gc_row[0:1, 0:P_MAX], src=gc_tp_psum[0:1, 0:P_MAX])

    # Broadcast gc_row to (P_MAX, P_MAX): every partition row gets gc^T.
    gc_row_bcast = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
    for i_shuf in nl.static_range(P_MAX // 32):
        nisa.nc_stream_shuffle(
            src=gc_row[0:1, 0:P_MAX],
            dst=gc_row_bcast[i_shuf * 32 : i_shuf * 32 + 32, 0:P_MAX],
            shuffle_mask=_BROADCAST_MASK,
        )

    # diff[i, j] = gc[i] - gc[j]
    diff = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(
        dst=diff, data1=gc_col_bcast, data2=gc_row_bcast, op=nl.subtract
    )

    # Clamp to <=0 so exp() stays in [0, 1] regardless of |gc|.
    diff_safe = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(
        dst=diff_safe,
        data=diff,
        op0=nl.minimum,
        operand0=0.0,
        engine=nisa.vector_engine,
    )

    decay_mat = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(
        dst=decay_mat, op=nl.exp, data=diff_safe, bias=None, scale=1.0
    )

    # exp(gc) as (P_MAX, 1) for q*exp(gc) and (k_beta * exp(gc)) terms.
    # gc is non-positive in normal use, but clamp defensively.
    gc_nonpos = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(
        dst=gc_nonpos,
        data=gc_col,
        op0=nl.minimum,
        operand0=0.0,
        engine=nisa.vector_engine,
    )
    exp_gc_p = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(
        dst=exp_gc_p[0:P_MAX, 0:1],
        op=nl.exp,
        data=gc_nonpos[0:P_MAX, 0:1],
        bias=None,
        scale=1.0,
    )

    # exp(gl) as scalar (1,1) — gl is broadcast in gl_c so any element works.
    gl_11 = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=gl_11, src=gl_c[0:1, 0:1])
    gl_nonpos = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(
        dst=gl_nonpos,
        data=gl_11,
        op0=nl.minimum,
        operand0=0.0,
        engine=nisa.vector_engine,
    )
    exp_gl_11 = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(
        dst=exp_gl_11, op=nl.exp, data=gl_nonpos, bias=None, scale=1.0
    )

    # Broadcast exp(gl) to (P_MAX, 1) for state-decay tensor_scalar.
    exp_gl_p = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
    for i_shuf in nl.static_range(P_MAX // 32):
        nisa.nc_stream_shuffle(
            src=exp_gl_11[0:1, 0:1],
            dst=exp_gl_p[i_shuf * 32 : i_shuf * 32 + 32, 0:1],
            shuffle_mask=_BROADCAST_MASK,
        )

    # ============================================================
    # k_beta = K * beta, v_beta = V * beta
    # beta_c is the (P, dim) broadcast input — multiply elementwise.
    # ============================================================
    k_beta = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=k_beta, data1=k_c, data2=beta_c, op=nl.multiply)

    v_beta = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=v_beta, data1=v_c, data2=beta_c, op=nl.multiply)

    # ============================================================
    # Phase 1: Build A matrix.
    # QK = k_beta @ k^T, contracting features
    # ============================================================
    kb_T_psum = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_transpose(dst=kb_T_psum, data=k_beta)
    kb_T = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=kb_T, src=kb_T_psum)

    k_T_psum = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_transpose(dst=k_T_psum, data=k_c)
    k_T = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=k_T, src=k_T_psum)

    QK_psum = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=QK_psum, stationary=kb_T, moving=k_T)
    QK = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=QK, src=QK_psum)

    # QK_decay[i, j] = QK[i, j] * exp(gc[i] - gc[j])
    QK_decay = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(
        dst=QK_decay, data1=QK, data2=decay_mat, op=nl.multiply
    )

    # A = -QK_decay * lower_mask (strict lower tri)
    neg_QK_decay = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(
        dst=neg_QK_decay,
        data=QK_decay,
        op0=nl.multiply,
        operand0=-1.0,
        engine=nisa.vector_engine,
    )
    A_mat = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=A_mat, data1=neg_QK_decay, data2=Lmask, op=nl.multiply)

    # ============================================================
    # Neumann power-doubling: N = (I+A)(I+A^2)...(I+A^{64}) (6 rounds).
    # A is strictly lower triangular so A^128 = 0 and the truncated
    # series equals (I-A)^{-1} exactly within the chunk.
    # ============================================================
    P_acc = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=P_acc, data1=eye, data2=A_mat, op=nl.add)

    A_pow = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=A_pow, src=A_mat)

    for _round in nl.sequential_range(6):
        Ap_T_psum = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_transpose(dst=Ap_T_psum, data=A_pow)
        Ap_T = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=Ap_T, src=Ap_T_psum)

        Ap_sq_psum = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=Ap_sq_psum, stationary=Ap_T, moving=A_pow)
        nisa.tensor_copy(dst=A_pow, src=Ap_sq_psum)

        IpA = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=IpA, data1=eye, data2=A_pow, op=nl.add)

        IpA_T_psum = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_transpose(dst=IpA_T_psum, data=IpA)
        IpA_T = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=IpA_T, src=IpA_T_psum)

        Pacc_psum = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=Pacc_psum, stationary=IpA_T, moving=P_acc)
        nisa.tensor_copy(dst=P_acc, src=Pacc_psum)

    # ============================================================
    # Apply N: value_corr = N @ v_beta, k_cumdecay = N @ (k_beta * exp(gc))
    # ============================================================
    N_T_psum = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_transpose(dst=N_T_psum, data=P_acc)
    N_T = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=N_T, src=N_T_psum)

    vc_psum = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=vc_psum, stationary=N_T, moving=v_beta)
    value_corr = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=value_corr, src=vc_psum)

    # k_beta * exp(gc) — row-scaled by the (P_MAX, 1) exp_gc_p
    kb_exp_gc = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(
        dst=kb_exp_gc,
        data=k_beta,
        op0=nl.multiply,
        operand0=exp_gc_p,
        engine=nisa.vector_engine,
    )

    kcd_psum = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=kcd_psum, stationary=N_T, moving=kb_exp_gc)
    k_cumdecay = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=k_cumdecay, src=kcd_psum)

    # ============================================================
    # Phase 2: Inter-chunk state propagation.
    # attn_intra = (q @ k^T) * decay_mat * lower_mask_diag
    # ============================================================
    q_T_psum = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_transpose(dst=q_T_psum, data=q_c)
    q_T = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=q_T, src=q_T_psum)

    qk_psum = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=qk_psum, stationary=q_T, moving=k_T)
    qk_raw = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=qk_raw, src=qk_psum)

    qk_decay = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(
        dst=qk_decay, data1=qk_raw, data2=decay_mat, op=nl.multiply
    )

    attn_intra = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(
        dst=attn_intra, data1=qk_decay, data2=Lmask_d, op=nl.multiply
    )

    # ============================================================
    # v_prime = k_cumdecay @ state
    # ============================================================
    kcd_T_psum = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_transpose(dst=kcd_T_psum, data=k_cumdecay)
    kcd_T = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=kcd_T, src=kcd_T_psum)

    vp_psum = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=vp_psum, stationary=kcd_T, moving=state)
    v_prime = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=v_prime, src=vp_psum)

    v_new = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=v_new, data1=value_corr, data2=v_prime, op=nl.subtract)

    # ============================================================
    # attn_inter = (q * exp(gc)) @ state
    # ============================================================
    q_exp = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(
        dst=q_exp,
        data=q_c,
        op0=nl.multiply,
        operand0=exp_gc_p,
        engine=nisa.vector_engine,
    )

    qe_T_psum = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_transpose(dst=qe_T_psum, data=q_exp)
    qe_T = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=qe_T, src=qe_T_psum)

    ai_psum = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=ai_psum, stationary=qe_T, moving=state)
    attn_inter = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=attn_inter, src=ai_psum)

    # ============================================================
    # attn_intra @ v_new
    # ============================================================
    ai_T_psum = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_transpose(dst=ai_T_psum, data=attn_intra)
    ai_T = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=ai_T, src=ai_T_psum)

    intra_psum = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=intra_psum, stationary=ai_T, moving=v_new)
    intra_out = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=intra_out, src=intra_psum)

    # chunk_output = attn_inter + intra_out
    chunk_out = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(
        dst=chunk_out, data1=attn_inter, data2=intra_out, op=nl.add
    )

    nisa.dma_copy(dst=output, src=chunk_out)

    # ============================================================
    # State update — log-sum-exp form, never overflows.
    #
    # Math:
    #   new_state = exp(gl) * state + sum_t k[t] * exp(gl - gc[t]) * v_new[t]^T
    #             = exp(gl) * state + (k * exp(gl - gc))^T @ v_new
    # ============================================================
    # gl_minus_gc[t] = gl - gc[t], (P_MAX, 1)
    gl_p = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
    for i_shuf in nl.static_range(P_MAX // 32):
        nisa.nc_stream_shuffle(
            src=gl_11[0:1, 0:1],
            dst=gl_p[i_shuf * 32 : i_shuf * 32 + 32, 0:1],
            shuffle_mask=_BROADCAST_MASK,
        )

    gl_minus_gc_p = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(
        dst=gl_minus_gc_p, data1=gl_p, data2=gc_col, op=nl.subtract
    )
    gl_minus_gc_safe = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(
        dst=gl_minus_gc_safe,
        data=gl_minus_gc_p,
        op0=nl.minimum,
        operand0=0.0,
        engine=nisa.vector_engine,
    )
    exp_gl_minus_gc_p = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(
        dst=exp_gl_minus_gc_p[0:P_MAX, 0:1],
        op=nl.exp,
        data=gl_minus_gc_safe[0:P_MAX, 0:1],
        bias=None,
        scale=1.0,
    )

    # k_weighted[t, :] = k[t, :] * exp(gl - gc[t])
    k_weighted = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(
        dst=k_weighted,
        data=k_c,
        op0=nl.multiply,
        operand0=exp_gl_minus_gc_p,
        engine=nisa.vector_engine,
    )

    # kv_outer = k_weighted^T @ v_new
    kv_psum = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=kv_psum, stationary=k_weighted, moving=v_new)
    kv_outer = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=kv_outer, src=kv_psum)

    # state_decayed = exp(gl) * state
    state_decayed = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(
        dst=state_decayed,
        data=state,
        op0=nl.multiply,
        operand0=exp_gl_p,
        engine=nisa.vector_engine,
    )

    # state_new = state_decayed + kv_outer
    state_new = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(
        dst=state_new, data1=state_decayed, data2=kv_outer, op=nl.add
    )

    nisa.dma_copy(dst=state_out, src=state_new)

    return output, state_out
