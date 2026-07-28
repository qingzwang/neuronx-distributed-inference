#!/usr/bin/env python
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Compile the DeepSeek-V4-Flash HF reference model to Neuron with TP sharding.

Strategy (approach 2 from the design doc):
  * Use HF's `inference/model.py` verbatim, with kernel_cpu.py monkey-patched
    in place of the Triton `kernel.py` (validated to XLA-compile in the spike).
  * Wrap the whole Transformer in `parallel_model_trace(spmd_mode=True,
    tp_degree=N)`. NxD's ParallelEmbedding / ColumnParallelLinear /
    RowParallelLinear that HF's model uses will shard correctly across N cores
    (`world_size = dist.get_world_size()` inside HF picks up the TP group).
  * checkpoint_loader_callable returns the dequantized full-tensor dict;
    NxD does rank-aware slicing during load.

This mirrors what we did for Qwen3.5-2B vision-encoder TP: the shard-aware
loading is entirely in NxD's `_load_weights` path.

Usage:
    source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
    cd contrib/models/DeepSeek-V4-Flash
    # Tiny smoke: 2 layers, TP=2, seq_len=16
    python src/compile_neuron.py --tp 2 --n-layers 2 --seq-len 16 \
        --out-dir /tmp/dsv4_smoke
    # Full model: 43 layers, TP=32, seq_len=64
    python src/compile_neuron.py --tp 32 --seq-len 64 \
        --out-dir /tmp/dsv4_tp32
"""

import argparse
import json
import os
import sys
import time
from types import SimpleNamespace

import torch


_HERE = os.path.dirname(os.path.abspath(__file__))
_CONTRIB_DIR = os.path.abspath(os.path.join(_HERE, ".."))

for p in (_HERE, _CONTRIB_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

import paths  # noqa: E402  — needs _HERE on sys.path first

_HF_INFERENCE_DIR = paths.hf_inference_dir()
if _HF_INFERENCE_DIR not in sys.path:
    sys.path.insert(0, _HF_INFERENCE_DIR)


# --- Module-level state for the picklable factory + loader ---
# parallel_model_trace re-imports this module in each spawned rank, so any
# state we need at factory time has to come from env vars or globals set at
# module-import time.
_MODEL_PATH = paths.model_path()
_CONFIG_JSON = paths.config_json()
_N_LAYERS = int(os.environ.get("DSV4_N_LAYERS", "43"))
_SEQ_LEN = int(os.environ.get("DSV4_SEQ_LEN", "64"))
_MAX_BATCH_SIZE = int(os.environ.get("DSV4_MAX_BATCH_SIZE", "1"))


def _wire_shims():
    """Inject kernel_cpu + fake fast_hadamard_transform before HF's `model` imports."""
    import importlib.machinery
    import types
    from hf_reference import kernel_cpu
    sys.modules["kernel"] = kernel_cpu

    # Fake fast_hadamard_transform (mirror of run_hf_cpu.py's shim)
    def _hadamard_matrix(n: int) -> "torch.Tensor":
        assert n > 0 and (n & (n - 1)) == 0
        H = torch.tensor([[1.0]], dtype=torch.float32)
        while H.size(0) < n:
            H = torch.cat([torch.cat([H, H], dim=1),
                           torch.cat([H, -H], dim=1)], dim=0)
        return H

    _H_CACHE: dict = {}

    def hadamard_transform(x: "torch.Tensor", scale: float = 1.0) -> "torch.Tensor":
        d = x.size(-1)
        if d not in _H_CACHE:
            _H_CACHE[d] = _hadamard_matrix(d)
        H = _H_CACHE[d].to(x.dtype).to(x.device)
        return torch.matmul(x, H) * scale

    mod = types.ModuleType("fast_hadamard_transform")
    mod.hadamard_transform = hadamard_transform
    mod.__spec__ = importlib.machinery.ModuleSpec(
        "fast_hadamard_transform", loader=None,
    )
    sys.modules["fast_hadamard_transform"] = mod


def _patch_parallel_embedding_for_xla(hf_mod):
    """Replace boolean-mask assignment in ParallelEmbedding with torch.where.

    HF's version is:

        mask = (x < start) | (x >= end)
        x = x - start
        x[mask] = 0                 # index_put_ with a bool mask
        y = F.embedding(x, weight)
        y[mask] = 0                 # ditto
        dist.all_reduce(y)

    `tensor[bool_mask] = 0` lowers on XLA to a `nonzero()`-driven scatter, and
    `nonzero` has a *data-dependent* output size. Under torch.jit.trace that
    size is frozen to whatever the example input produced. parallel_model_trace
    traces with all-zero input_ids, so rank 0 sees mask.sum() == 0 while every
    other rank sees mask.sum() == seq_len. At runtime with real token ids the
    true counts differ from the recorded ones and the indirect DMA walks off the
    end of the index buffer:

        status=1006 Execution Out-Of-Bounds Memory Access
        scatter/gather (indirect memory copy via vector DGE) out-of-bound
        access ... engine=GPSIMD

    torch.where is shape-static and index-free, so the same math compiles to a
    fixed-size select with no indirect addressing. Semantics are identical:
    out-of-range ids read row 0 and then get zeroed before the all-reduce.
    """
    import torch as _torch
    import torch.nn.functional as _F

    def parallel_embedding_forward(self, x: _torch.Tensor) -> _torch.Tensor:
        if hf_mod.world_size > 1:
            in_range = (x >= self.vocab_start_idx) & (x < self.vocab_end_idx)
            # Clamp instead of masked-assign: any id outside our shard reads
            # row 0, whose contribution is discarded right after.
            local = _torch.where(
                in_range, x - self.vocab_start_idx, _torch.zeros_like(x),
            )
            y = _F.embedding(local, self.weight)
            y = _torch.where(in_range.unsqueeze(-1), y, _torch.zeros_like(y))
            import torch.distributed as _dist
            _dist.all_reduce(y)
            return y
        return _F.embedding(x, self.weight)

    hf_mod.ParallelEmbedding.forward = parallel_embedding_forward


def _patch_moe_forward_for_xla(hf_mod):
    """Replace HF's dispatch-based MoE.forward with a static-shape one.

    The HF reference does:

        counts = torch.bincount(indices.flatten(), minlength=N).tolist()
        for i in range(start, end):
            if counts[i] == 0: continue
            idx, top = torch.where(indices == i)
            y[idx] += expert_i(x[idx], weights[idx, top, None])

    This is GPU-only:
      * `bincount(...).tolist()` forces a device->host sync
      * data-dependent `if counts[i] == 0: continue`
      * `torch.where`-driven scatter with variable-length gathers

    For XLA/Neuron all shapes must be static. Rewrite in the "run every
    expert on every token, mask with routing weights" style — mathematically
    identical to the dispatch version, but much more work than the sparse
    reference. Fine for correctness validation at small n_layers; at full
    43 layers × 256 experts this will be prohibitively slow (~256/6 = 42x
    the FLOPs a proper dispatched impl would do). For that we would need
    to plug in NxDI's expert_mlps_v2, but this static rewrite is enough
    to validate the *rest* of the graph compiles.
    """
    import torch as _torch
    import torch.nn.functional as _F

    def moe_forward_xla(self, x: _torch.Tensor, input_ids: _torch.Tensor):
        shape = x.size()
        x = x.view(-1, self.dim)                      # (T, D)
        weights, indices = self.gate(x, input_ids.flatten())
        # weights, indices: (T, top_k)
        T = x.size(0)
        k = weights.size(-1)

        # Build a one-hot-ish routing mask over all local experts.
        # weight_per_expert[t, e] = sum over k of weights[t, k] * (indices[t, k] == e)
        # Vectorize via scatter_add.
        wpe = _torch.zeros(
            T, self.n_local_experts, dtype=weights.dtype, device=weights.device,
        )
        # Only consider indices that fall inside our local expert range.
        local_indices = indices - self.experts_start_idx
        in_range = (local_indices >= 0) & (local_indices < self.n_local_experts)
        safe_local = local_indices.clamp(min=0, max=self.n_local_experts - 1)
        # weights_masked[t, k] = weights[t, k] if in_range else 0.0
        weights_masked = weights * in_range.to(weights.dtype)
        # scatter-add into wpe[t, safe_local[t, k]]
        wpe.scatter_add_(1, safe_local, weights_masked)

        # Run every local expert on every token (static shapes) and combine
        # by wpe (which is zero for tokens the expert wasn't routed to).
        y = _torch.zeros_like(x, dtype=_torch.float32)
        for i in range(self.experts_start_idx, self.experts_end_idx):
            expert = self.experts[i]
            # w_e is per-token gating for this expert
            w_e = wpe[:, i - self.experts_start_idx].unsqueeze(-1)  # (T, 1)
            # expert receives (x, weight) and returns a scaled contribution
            y = y + expert(x, w_e).float()

        # world_size / dist live at the model.py module scope; look them up
        # at call time so this patched forward sees updates that happen after
        # Transformer.__init__ mutates them.
        if hf_mod.world_size > 1:
            import torch.distributed as _dist
            _dist.all_reduce(y)
        y = y + self.shared_experts(x).float()
        return y.type_as(x).view(shape)

    hf_mod.MoE.forward = moe_forward_xla


def _patch_hf_model_for_xla(hf_mod):
    """Replace complex-tensor RoPE with a real-valued equivalent.

    HF's `apply_rotary_emb` uses torch.view_as_complex + complex mul, which
    XLA/Neuron does not support. Rewrite in terms of (cos, sin) real pairs
    while preserving the "in-place-on-a-slice" contract used by call sites:
      apply_rotary_emb(q[..., -rd:], freqs_cis)   # mutates q in place

    We also rewrite `precompute_freqs_cis` to return a stacked (cos, sin) pair
    instead of a complex tensor. The stacked form has shape [S, rd] where each
    pair of adjacent columns is (cos_i, sin_i) — matches how apply_rotary_emb
    unflattens to pairs.
    """
    import math as _math

    def precompute_freqs_cis(
        dim, seqlen, original_seq_len, base, factor, beta_fast, beta_slow,
    ):
        # Same freq schedule as HF's version.
        def find_correction_dim(num_rotations, dim, base, max_seq_len):
            return (
                dim * _math.log(max_seq_len / (num_rotations * 2 * _math.pi))
                / (2 * _math.log(base))
            )

        def find_correction_range(low_rot, high_rot, dim, base, max_seq_len):
            low = _math.floor(find_correction_dim(low_rot, dim, base, max_seq_len))
            high = _math.ceil(find_correction_dim(high_rot, dim, base, max_seq_len))
            return max(low, 0), min(high, dim - 1)

        def linear_ramp_factor(mn, mx, d):
            if mn == mx:
                mx += 0.001
            lf = (torch.arange(d, dtype=torch.float32) - mn) / (mx - mn)
            return torch.clamp(lf, 0, 1)

        freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        if original_seq_len > 0:
            low, high = find_correction_range(
                beta_fast, beta_slow, dim, base, original_seq_len,
            )
            smooth = 1 - linear_ramp_factor(low, high, dim // 2)
            freqs = freqs / factor * (1 - smooth) + freqs * smooth

        t = torch.arange(seqlen, dtype=torch.float32)
        freqs = torch.outer(t, freqs)  # [S, dim/2]
        # Stack (cos, sin) along a new last dim then flatten -> [S, dim]
        # so freqs_cis[:, 2i]   = cos(freqs[:, i])
        #    freqs_cis[:, 2i+1] = sin(freqs[:, i])
        cs = torch.stack([torch.cos(freqs), torch.sin(freqs)], dim=-1).flatten(-2)
        return cs  # float32, [S, dim]

    def apply_rotary_emb(x, freqs_cis, inverse: bool = False):
        """Same semantics as HF's version but real-valued and functional.

        The call sites do `apply_rotary_emb(q[..., -rd:], freqs_cis)` and
        discard the return value, relying on in-place mutation of `q`. We
        preserve that by writing back into `x` via `x.copy_(...)` at the end
        — XLA lowers copy_ into a functional update.
        """
        # freqs_cis: [S, rd] with (cos, sin, cos, sin, ...) layout.
        # Reshape to [1, S, rd/2, 2] or [1, S, 1, rd/2, 2] to broadcast over heads.
        rd = x.size(-1)
        cs = freqs_cis.view(1, freqs_cis.size(0), rd // 2, 2)
        if x.ndim == 4:  # (B, S, H, rd) — attention Q/K case
            cs = cs.unsqueeze(2)  # (1, S, 1, rd/2, 2)
        cos = cs[..., 0]
        sin = cs[..., 1]
        if inverse:
            sin = -sin

        # Pair-wise rotate: (a, b) -> (a*cos - b*sin, a*sin + b*cos)
        y = x.float()
        # Unflatten last dim into pairs
        y_pairs = y.unflatten(-1, (rd // 2, 2))  # (..., rd/2, 2)
        a = y_pairs[..., 0]
        b = y_pairs[..., 1]
        rot_a = a * cos - b * sin
        rot_b = a * sin + b * cos
        rot = torch.stack([rot_a, rot_b], dim=-1).flatten(-2)
        rot = rot.to(x.dtype)
        x.copy_(rot)
        return x

    hf_mod.precompute_freqs_cis = precompute_freqs_cis
    hf_mod.apply_rotary_emb = apply_rotary_emb


def _patch_attention_o_proj_for_high_tp(hf_mod):
    """Make the grouped O-projection work when tp > o_groups.

    HF's Attention does

        self.n_local_groups = self.n_groups // world_size      # o_groups = 8
        wo_a = self.wo_a.weight.view(self.n_local_groups, self.o_lora_rank, -1)
        o = torch.einsum("bsgd,grd->bsgr", o, wo_a)

    so at tp >= 16 `n_local_groups` is 0, the view has a zero-size dim, and the
    einsum silently contributes nothing. That caps TP at 8 — but 8 is not
    enough to hold the model: 283.8 B params is 568 GB in bf16, i.e. 70.9 GB
    per core at tp=8 against a 24 GB per-core budget (bytes_limit reported by
    torch_xla on this trn2 with logical-neuroncore-config 2). Production needs
    tp >= 32.

    Past o_groups the natural split changes axis. Each rank already owns
    n_heads // tp heads, i.e. a *slice of the input dim* of one group's wo_a
    rather than whole groups. So for tp > o_groups rank r handles

        group g = r // (tp // o_groups)
        slice j = r %  (tp // o_groups)   of that group's contraction dim

    which is an ordinary row-parallel (input-sharded) matmul. The partial
    products are summed by the all_reduce already inside wo_b, which is a
    RowParallelLinear — so no extra collective is needed and the result is
    exact up to fp accumulation order (verified: max rel error ~5e-7 vs the
    unsharded reference at tp = 8, 16, 32, 64).

    tp <= o_groups keeps HF's original grouped-einsum path unchanged.
    """
    import torch as _torch

    def attention_forward(self, x: _torch.Tensor, start_pos: int):
        bsz, seqlen, _ = x.size()
        freqs_cis = self.freqs_cis[start_pos:start_pos + seqlen]
        win = self.window_size
        ratio = self.compress_ratio
        rd = self.rope_head_dim
        if self.compress_ratio and self.compressor.kv_cache is None:
            self.compressor.kv_cache = self.kv_cache[:, win:]
            self.compressor.freqs_cis = self.freqs_cis
            if self.indexer is not None:
                self.indexer.freqs_cis = self.freqs_cis

        qr = q = self.q_norm(self.wq_a(x))
        q = self.wq_b(q).unflatten(-1, (self.n_local_heads, self.head_dim))
        q = q * _torch.rsqrt(q.square().mean(-1, keepdim=True) + self.eps)
        hf_mod.apply_rotary_emb(q[..., -rd:], freqs_cis)

        kv = self.wkv(x)
        kv = self.kv_norm(kv)
        hf_mod.apply_rotary_emb(kv[..., -rd:], freqs_cis)
        hf_mod.act_quant(kv[..., :-rd], 64, hf_mod.scale_fmt,
                         hf_mod.scale_dtype, True)
        topk_idxs = hf_mod.get_window_topk_idxs(win, bsz, seqlen, start_pos)
        if self.compress_ratio:
            offset = kv.size(1) if start_pos == 0 else win
            if self.indexer is not None:
                compress_topk_idxs = self.indexer(x, qr, start_pos, offset)
            else:
                compress_topk_idxs = hf_mod.get_compress_topk_idxs(
                    ratio, bsz, seqlen, start_pos, offset)
            topk_idxs = _torch.cat([topk_idxs, compress_topk_idxs], dim=-1)
        topk_idxs = topk_idxs.int()

        if start_pos == 0:
            if seqlen <= win:
                self.kv_cache[:bsz, :seqlen] = kv
            else:
                cutoff = seqlen % win
                (self.kv_cache[:bsz, cutoff:win],
                 self.kv_cache[:bsz, :cutoff]) = kv[:, -win:].split(
                     [win - cutoff, cutoff], dim=1)
            if self.compress_ratio:
                kv_compress = self.compressor(x, start_pos)
                if kv_compress is not None:
                    kv = _torch.cat([kv, kv_compress], dim=1)
            o = hf_mod.sparse_attn(q, kv, self.attn_sink, topk_idxs,
                                   self.softmax_scale)
        else:
            self.kv_cache[:bsz, start_pos % win] = kv.squeeze(1)
            if self.compress_ratio:
                self.compressor(x, start_pos)
            o = hf_mod.sparse_attn(q, self.kv_cache[:bsz], self.attn_sink,
                                   topk_idxs, self.softmax_scale)
        hf_mod.apply_rotary_emb(o[..., -rd:], freqs_cis, True)

        if self.n_local_groups >= 1:
            # HF's path: this rank owns whole groups.
            o = o.view(bsz, seqlen, self.n_local_groups, -1)
            wo_a = self.wo_a.weight.view(self.n_local_groups, self.o_lora_rank, -1)
            o = _torch.einsum("bsgd,grd->bsgr", o, wo_a)
            o = o.flatten(2)
        else:
            # tp > o_groups: this rank owns a contraction-dim slice of one
            # group. wo_a.weight is already [o_lora_rank, slice_width] from
            # shard_loader, so a plain matmul is the whole operation; wo_b's
            # all_reduce sums this rank's partial with its group-mates'.
            o = _torch.nn.functional.linear(
                o.reshape(bsz, seqlen, -1), self.wo_a.weight,
            )
        return self.wo_b(o)

    original_init = hf_mod.Attention.__init__

    def attention_init(self, layer_id, args):
        original_init(self, layer_id, args)
        ws = hf_mod.world_size
        if ws > self.n_groups:
            if ws % self.n_groups:
                raise ValueError(
                    f"tp={ws} must be a multiple of o_groups={self.n_groups} "
                    f"to split each group's contraction dim evenly"
                )
            # wo_a was built as ColumnParallelLinear(n_heads*head_dim//n_groups,
            # n_groups*o_lora_rank) -> [n_groups*o_lora_rank // ws, group_in].
            # Re-allocate as one group's o_lora_rank rows by this rank's slice
            # of that group's contraction dim.
            ranks_per_group = ws // self.n_groups
            group_in = self.n_heads * self.head_dim // self.n_groups
            slice_width = group_in // ranks_per_group
            self.o_ranks_per_group = ranks_per_group
            self.o_group_id = hf_mod.rank // ranks_per_group
            self.o_slice_id = hf_mod.rank % ranks_per_group
            self.wo_a.weight = torch.nn.Parameter(
                torch.empty(self.o_lora_rank, slice_width,
                            dtype=self.wo_a.weight.dtype),
                requires_grad=False,
            )
            # wo_b was RowParallelLinear(n_groups * o_lora_rank, dim), i.e.
            # [dim, n_groups * o_lora_rank // ws] — 512 cols at tp=16. But the
            # partial this rank produces spans its group's *entire* o_lora_rank
            # block, so wo_b needs all o_lora_rank columns of that block,
            # replicated across the group's ranks. Splitting wo_b's columns
            # instead would drop cross terms (verified: 66% relative error).
            # The duplication is 4096 x 1024 x 2 B = 8 MB per layer.
            self.wo_b.weight = torch.nn.Parameter(
                torch.empty(self.dim, self.o_lora_rank,
                            dtype=self.wo_b.weight.dtype),
                requires_grad=False,
            )

    hf_mod.Attention.__init__ = attention_init
    hf_mod.Attention.forward = attention_forward


def _patch_index_helpers_for_xla(hf_mod):
    """Make get_window/compress_topk_idxs build their index tensors on-device.

    Both helpers construct pure position constants with bare `torch.arange`.
    HF gets away with it because its __main__ sets a global default device of
    cuda; under XLA tracing the default device is CPU, so the results are CPU
    tensors. Layers 0-1 tolerate that (compress_ratio == 0, so the window
    indices go straight into sparse_attn, which accepts a host index), but
    layer 2 onwards does

        topk_idxs = torch.cat([topk_idxs, compress_topk_idxs], dim=-1)

    where compress_topk_idxs comes from the Indexer and *is* on device:

        RuntimeError: Expected all tensors in the given list to be XLA
        tensors. Element at index 0 is not an XLA tensor.

    The device is recorded from input_ids at the top of Transformer.forward
    because these are free functions with no module to read it from. Also drops
    HF's lru_cache: the cache key does not include the device, so a cached CPU
    result would be handed back on a later on-device call.
    """
    import torch as _torch
    import torch.nn.functional as _F

    state = {"device": None}

    def _dev():
        return state["device"]

    def _rows(vec, n_rows):
        """Materialize `vec` as n_rows identical rows without a strided view.

        Neither `.repeat(n, 1)` nor `.unsqueeze(0).expand(n, -1)` survives XLA
        tracing here:
            RuntimeError: aten::as_strided ... has no implementation for the
            backend "xla:0". View operators don't support since the tensor's
            storage cannot be shared across devices.
        Adding a zero column vector broadcasts to the same result as a real op.
        """
        zeros = _torch.zeros(n_rows, 1, dtype=vec.dtype, device=vec.device)
        return vec.unsqueeze(0) + zeros

    def _batch(matrix, bsz):
        """Same trick for the leading batch dim."""
        zeros = _torch.zeros(bsz, *([1] * matrix.dim()),
                             dtype=matrix.dtype, device=matrix.device)
        return matrix.unsqueeze(0) + zeros

    def get_window_topk_idxs(window_size, bsz, seqlen, start_pos):
        d = _dev()
        if start_pos >= window_size - 1:
            start_pos %= window_size
            matrix = _torch.cat([
                _torch.arange(start_pos + 1, window_size, device=d),
                _torch.arange(0, start_pos + 1, device=d),
            ], dim=0)
        elif start_pos > 0:
            matrix = _F.pad(_torch.arange(start_pos + 1, device=d),
                            (0, window_size - start_pos - 1), value=-1)
        else:
            base = _torch.arange(seqlen, device=d).unsqueeze(1)
            matrix = ((base - window_size + 1).clamp(0)
                      + _torch.arange(min(seqlen, window_size), device=d))
            matrix = _torch.where(matrix > base, -1, matrix)
        return _batch(matrix, bsz)

    def get_compress_topk_idxs(ratio, bsz, seqlen, start_pos, offset):
        d = _dev()
        if start_pos > 0:
            matrix = _torch.arange(0, (start_pos + 1) // ratio, device=d) + offset
        else:
            matrix = _rows(_torch.arange(seqlen // ratio, device=d), seqlen)
            mask = (matrix
                    >= _torch.arange(1, seqlen + 1, device=d).unsqueeze(1) // ratio)
            matrix = _torch.where(mask, -1, matrix + offset)
        return _batch(matrix, bsz)

    original_forward = getattr(hf_mod.Transformer.forward, "__wrapped__",
                               hf_mod.Transformer.forward)

    def transformer_forward(self, input_ids, start_pos: int = 0):
        state["device"] = input_ids.device
        return original_forward(self, input_ids, start_pos)

    hf_mod.get_window_topk_idxs = get_window_topk_idxs
    hf_mod.get_compress_topk_idxs = get_compress_topk_idxs
    hf_mod.Transformer.forward = transformer_forward
    # _TraceWrapper looks for __wrapped__ to bypass @torch.inference_mode();
    # our replacement is already unwrapped, so point it at itself.
    transformer_forward.__wrapped__ = transformer_forward
    return state


def _patch_topk_for_xla(hf_mod):
    """Route Gate / Indexer top-k selection through xla_ops (no HLO `sort`).

    neuronx-cc rejects `sort` on trn2, and torch.topk lowers to it:
        [NCC_EVRF029] Operation sort is not supported on trn2
    Layers 0-2 use hash routing and layer 0/1 have no compressor, so a 1-3
    layer build never hits this; layer 2 (Indexer) and layer 3 (score-based
    gate) do. See xla_ops.py.
    """
    import torch as _torch
    import torch.nn.functional as _F

    import xla_ops

    def gate_forward_xla(self, x, input_ids=None):
        scores = hf_mod.linear(x.float(), self.weight.float())
        if self.score_func == "softmax":
            scores = scores.softmax(dim=-1)
        elif self.score_func == "sigmoid":
            scores = scores.sigmoid()
        else:
            scores = _F.softplus(scores).sqrt()
        original_scores = scores
        if self.bias is not None:
            scores = scores + self.bias
        if self.hash:
            indices = self.tid2eid[input_ids]
        else:
            indices = xla_ops.topk_indices(scores, self.topk)
        weights = original_scores.gather(1, indices.long())
        if self.score_func != "softmax":
            weights = weights / weights.sum(dim=-1, keepdim=True)
        weights = weights * self.route_scale
        return weights, indices

    def indexer_forward_xla(self, x, qr, start_pos: int, offset: int):
        bsz, seqlen, _ = x.size()
        freqs_cis = self.freqs_cis[start_pos:start_pos + seqlen]
        ratio = self.compress_ratio
        rd = self.rope_head_dim
        end_pos = start_pos + seqlen
        if self.compressor.kv_cache is None:
            self.compressor.kv_cache = self.kv_cache
            self.compressor.freqs_cis = self.freqs_cis
        q = self.wq_b(qr)
        q = q.unflatten(-1, (self.n_local_heads, self.head_dim))
        hf_mod.apply_rotary_emb(q[..., -rd:], freqs_cis)
        q = hf_mod.rotate_activation(q)
        hf_mod.fp4_act_quant(q, hf_mod.fp4_block_size, True)
        self.compressor(x, start_pos)
        weights = self.weights_proj(x) * (
            self.softmax_scale * self.n_heads ** -0.5
        )
        index_score = _torch.einsum(
            "bshd,btd->bsht", q, self.kv_cache[:bsz, :end_pos // ratio],
        )
        index_score = (index_score.relu() * weights.unsqueeze(-1)).sum(dim=2)
        if hf_mod.world_size > 1:
            import torch.distributed as _dist
            _dist.all_reduce(index_score)
        # HF builds these arange masks with the global default device set to
        # cuda; under XLA tracing the default device is CPU, so they must be
        # pinned to the activation's device explicitly or the trace aborts with
        # "Expected XLA tensor. Got: torch.LongTensor".
        dev = index_score.device
        causal_limit = (
            _torch.arange(1, seqlen + 1, device=dev).unsqueeze(1) // ratio
        )
        if start_pos == 0:
            # Broadcast-add rather than repeat/expand: both lower to
            # as_strided, which torch-xla does not implement.
            cols = _torch.arange(seqlen // ratio, device=dev).unsqueeze(0)
            mask = (cols + _torch.zeros(seqlen, 1, dtype=cols.dtype, device=dev)
                    >= causal_limit)
            index_score = index_score + _torch.where(
                mask, float("-inf"), 0.0,
            ).to(index_score.dtype)
        k = min(self.index_topk, end_pos // ratio)
        topk_idxs = xla_ops.topk_indices_unordered(index_score, k)
        if start_pos == 0:
            topk_idxs = _torch.where(
                topk_idxs >= causal_limit, -1, topk_idxs + offset,
            )
        else:
            topk_idxs = topk_idxs + offset
        return topk_idxs

    hf_mod.Gate.forward = gate_forward_xla
    hf_mod.Indexer.forward = indexer_forward_xla


def apply_xla_patches(hf_mod):
    """Apply every XLA-safety patch HF's model.py needs, in one call.

    Call this instead of the individual _patch_* functions so a newly added
    patch can't be silently missed by one of the test harnesses.
    """
    _patch_hf_model_for_xla(hf_mod)
    _patch_parallel_embedding_for_xla(hf_mod)
    _patch_moe_forward_for_xla(hf_mod)
    _patch_topk_for_xla(hf_mod)
    _patch_index_helpers_for_xla(hf_mod)
    _patch_attention_o_proj_for_high_tp(hf_mod)


def _build_model_args(hf_mod):
    """Read inference config, override to BF16 fast-path + small-model overrides."""
    with open(_CONFIG_JSON) as f:
        cfg = json.load(f)
    cfg.update(
        dtype="bf16",
        expert_dtype=None,
        n_mtp_layers=0,
        max_batch_size=_MAX_BATCH_SIZE,
        max_seq_len=_SEQ_LEN,
    )
    if _N_LAYERS < cfg["n_layers"]:
        cfg["n_layers"] = _N_LAYERS
        # compress_ratios must also match n_layers (drop trailing entries)
        if "compress_ratios" in cfg and len(cfg["compress_ratios"]) > _N_LAYERS:
            cfg["compress_ratios"] = cfg["compress_ratios"][:_N_LAYERS]

    # A layer with compress_ratio r has seq_len // r compressed KV entries. At
    # seq_len < r that is zero, and every downstream index tensor is
    # zero-width, which XLA cannot even construct:
    #   aten::as_strided ... has no implementation for the backend "xla:0"
    # Reject it here with an actionable message instead.
    ratios = [r for r in cfg.get("compress_ratios", []) if r]
    if ratios:
        need = max(ratios)
        if _SEQ_LEN < need:
            offenders = sorted({r for r in ratios if r > _SEQ_LEN})
            raise ValueError(
                f"seq_len={_SEQ_LEN} is too short for compress_ratios "
                f"{offenders} present in the first {cfg['n_layers']} layers: "
                f"seq_len // compress_ratio == 0 produces zero-width index "
                f"tensors, which XLA cannot build. Use --seq-len >= {need}, or "
                f"--n-layers <= "
                f"{cfg['compress_ratios'].index(max(offenders))} to stay below "
                f"the first such layer."
            )
    return hf_mod.ModelArgs(**cfg)


class _TraceWrapper(torch.nn.Module):
    """Wrap the HF Transformer for XLA tracing.

    Two adjustments vs the raw HF Transformer.forward:
      1. Bypass the @torch.inference_mode() decorator, which conflicts with
         torch-xla's tensor version counter tracking:
           RuntimeError: Cannot set version_counter for inference tensor
      2. Accept `start_pos` as a scalar int tensor rather than a Python int,
         because torch.jit.trace / torch_neuronx.trace only accepts
         Tensor / List[Tensor] / Dict[..., Tensor] / Tuple[Tensor, ...].
    """

    def __init__(self, inner: torch.nn.Module):
        super().__init__()
        self.inner = inner

    def forward(self, input_ids: torch.Tensor, start_pos: torch.Tensor):
        # start_pos is a 0-d int tensor; the HF code uses it as a Python int
        # to slice freqs_cis and index kv_cache. Convert via .item() — this
        # bakes the value into the trace, which is fine for the CTE (prefill)
        # graph where start_pos = 0 is a compile-time constant.
        with torch.no_grad():
            pos = int(start_pos.item()) if start_pos.dim() == 0 else int(start_pos)
            raw = getattr(self.inner.forward, "__wrapped__", None)
            if raw is not None:
                return raw(self.inner, input_ids, pos)
            return self.inner.forward(input_ids, pos)


def _picklable_factory():
    """Runs in each spawned trace subprocess. Must return (model, aliases).

    When DSV4_LOAD_WEIGHTS=1 this also loads *this rank's* slice of the real
    checkpoint before returning. That has to happen here, inside the spawned
    rank, rather than via NxD's `checkpoint_loader_callable`: NxD's sharding
    only recognizes its own parallel-layer classes, and HF's model.py defines
    same-named classes of its own, so NxD would shard nothing. See
    shard_loader.py for the full explanation.
    """
    for p in (_HERE, _CONTRIB_DIR, _HF_INFERENCE_DIR):
        if p not in sys.path:
            sys.path.insert(0, p)
    _wire_shims()
    import model as hf
    apply_xla_patches(hf)
    torch.set_default_dtype(torch.bfloat16)
    m_args = _build_model_args(hf)
    m = hf.Transformer(m_args).eval()

    if os.environ.get("DSV4_LOAD_WEIGHTS") == "1":
        import shard_loader
        # hf.world_size / hf.rank were set from the process group inside
        # Transformer.__init__, so the model already has rank-local shapes.
        shard_loader.load_rank_weights(
            m, _MODEL_PATH, rank=hf.rank, world_size=hf.world_size,
        )
    return _TraceWrapper(m), {}


def _example_inputs():
    """input_ids: (B, S), start_pos: 0-d int tensor.

    HF's Transformer.forward takes `start_pos: int`; _TraceWrapper accepts
    a scalar tensor and passes .item() through. torch.jit.trace refuses
    Python-int inputs in the example-inputs tuple, so we wrap in a tensor.
    """
    ids = torch.zeros(_MAX_BATCH_SIZE, _SEQ_LEN, dtype=torch.long)
    start_pos = torch.zeros((), dtype=torch.int32)
    return (ids, start_pos)


# Host RAM per concurrently-compiling rank, GB. Measured: 12 layers at tp=32
# peaked at 1465 GB used with 148 GB of that in resident rank weights, so
# 1317/32 ~= 41. Each rank's neuronx-cc fans out to ~3 OS processes.
_COMPILE_GB_PER_RANK = 41.0

# Total parameter count of the DeepSeek-V4-Flash checkpoint, all 43 layers.
_TOTAL_PARAMS = 283.8e9


def _host_ram_gb():
    """(total, available) host RAM in GB, read from /proc/meminfo."""
    vals = {}
    with open("/proc/meminfo") as f:
        for line in f:
            k, _, rest = line.partition(":")
            vals[k] = int(rest.split()[0]) / (1024 ** 2)  # kB -> GB
    return vals["MemTotal"], vals["MemAvailable"]


def _swap_gb():
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("SwapTotal:"):
                return int(line.split()[1]) / (1024 ** 2)
    return 0.0


def preflight_host_ram(cfg, n_layers, tp, max_parallel_compilations, force=False):
    """Refuse to start a trace whose host-RAM peak exceeds what the box has.

    `parallel_model_trace` spawns all `tp` ranks at once and holds every rank
    model resident in host RAM *while* neuronx-cc runs. Two costs add up:

      weights   n_layers/total_layers * total_params * 2 bytes  (bf16, summed
                over all ranks == the whole model, since the ranks partition it)
      compiler  _COMPILE_GB_PER_RANK per concurrently-compiling rank. Note that
                is per *rank*, not per process: max_parallel_compilations gates
                ranks, and each rank's neuronx-cc fans out to ~3 processes
                (tp=32 uncapped showed 97 of them).

    The compiler constant is calibrated against a measured run rather than
    guessed: 12 layers at tp=32 peaked at 1465 GB used with 148 GB of weights,
    leaving ~1317 GB across 32 concurrent ranks.

    This is not a theoretical guard. A 43-layer tp=32 run needs ~529 GB of
    weights, and 529 + 32*41 = 1841 GB exceeded the 1999 GB box. With swap=0
    the kernel livelocked rather than OOM-killing anything: it became
    unreachable over SSH and had to be power-cycled, losing 45 minutes of
    weight loading. The 12-layer run fit at 1465 GB and looked like it had
    headroom, which is exactly why this check has to be arithmetic rather than
    "the last one fit".
    """
    total_layers = int(cfg.get("n_layers", n_layers)) or n_layers
    weights_gb = (n_layers / total_layers) * _TOTAL_PARAMS * 2 / (1024 ** 3)
    compilers = min(max_parallel_compilations or tp, tp)
    compiler_gb = _COMPILE_GB_PER_RANK * compilers
    need = weights_gb + compiler_gb

    total, avail = _host_ram_gb()
    swap = _swap_gb()
    print(f"[preflight] host RAM: total={total:.0f} GB available={avail:.0f} GB "
          f"swap={swap:.0f} GB")
    print(f"[preflight] estimated peak: weights={weights_gb:.0f} GB "
          f"+ {compilers} concurrent ranks x {_COMPILE_GB_PER_RANK:.0f} GB "
          f"= {compiler_gb:.0f} GB  =>  {need:.0f} GB")

    if need > avail * 0.9:
        msg = (
            f"estimated peak host RAM {need:.0f} GB exceeds 90% of available "
            f"{avail:.0f} GB. With swap={swap:.0f} GB the kernel cannot reclaim, "
            f"so overcommitting hangs the machine instead of failing the run.\n"
            f"Options:\n"
            f"  --max-parallel-compilations N   (N x {_COMPILE_GB_PER_RANK:.0f} GB; "
            f"N <= {max(0, int((avail * 0.9 - weights_gb) // _COMPILE_GB_PER_RANK))} "
            f"fits alongside the weights)\n"
            f"  --n-layers <fewer>              (weights scale linearly)\n"
            f"  --force                         (proceed anyway; may hang the host)"
        )
        if not force:
            raise SystemExit(f"[preflight] REFUSING TO START: {msg}")
        print(f"[preflight] WARNING (--force): {msg}")
    return need


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tp", type=int, default=2)
    ap.add_argument("--n-layers", type=int, default=2)
    ap.add_argument("--seq-len", type=int, default=16)
    ap.add_argument("--max-batch-size", type=int, default=1)
    ap.add_argument("--out-dir", default="/tmp/dsv4_smoke")
    ap.add_argument("--model-path", default=paths.model_path())
    ap.add_argument("--config", default=paths.config_json())
    ap.add_argument("--load-weights", action="store_true",
                    help="Load each rank's slice of the real checkpoint before "
                         "tracing. Without this the trace runs on a random-init "
                         "model, which validates the graph but not accuracy. "
                         "NOTE: SPMD mode (NxD's checkpoint_loader_callable) "
                         "cannot be used for this model — NxD only shards its "
                         "own parallel-layer classes and HF's model.py defines "
                         "its own same-named ones, so nothing gets sharded. "
                         "See shard_loader.py.")
    ap.add_argument("--opt-level", default="1", choices=["1", "2", "3"],
                    help="neuronx-cc -O level.")
    ap.add_argument("--no-mixed-precision-accumulation", action="store_true",
                    help="Drop --enable-mixed-precision-accumulation. That flag "
                         "changes how reductions are lowered; useful to toggle "
                         "when chasing runtime faults.")
    ap.add_argument("--extra-compiler-args", default="",
                    help="Space-separated extra neuronx-cc flags.")
    ap.add_argument("--max-parallel-compilations", type=int, default=None,
                    help="Cap concurrent neuronx-cc processes. Default (None) "
                         "lets NxD run one per rank, which at tp=32 measured "
                         "~1.3 TB of host RAM on top of the resident rank "
                         "models. Trades wall-clock for RAM.")
    ap.add_argument("--force", action="store_true",
                    help="Skip the host-RAM preflight refusal. Overcommitting "
                         "on a swapless host hangs the machine rather than "
                         "failing the run.")
    ap.add_argument("--compiler-workdir", default=None,
                    help="Scratch dir for neuronx-cc. Defaults under /tmp, which "
                         "is too small at production depth: the workdir holds one "
                         "HLO + NEFF per rank, so 43 layers at tp=32 needs "
                         "hundreds of GB. Point this at a big filesystem. Do NOT "
                         "set TMPDIR to move it — torch's shm_manager needs "
                         "TMPDIR to already exist and fails the run if it does "
                         "not.")
    args = ap.parse_args()

    # Publish factory args via env (spawned subprocesses re-import this module).
    os.environ["DSV4_MODEL_PATH"] = args.model_path
    os.environ["DSV4_CONFIG"] = args.config
    os.environ["DSV4_N_LAYERS"] = str(args.n_layers)
    os.environ["DSV4_SEQ_LEN"] = str(args.seq_len)
    os.environ["DSV4_MAX_BATCH_SIZE"] = str(args.max_batch_size)
    os.environ["DSV4_LOAD_WEIGHTS"] = "1" if args.load_weights else "0"

    global _MODEL_PATH, _CONFIG_JSON, _N_LAYERS, _SEQ_LEN, _MAX_BATCH_SIZE
    _MODEL_PATH = args.model_path
    _CONFIG_JSON = args.config
    _N_LAYERS = args.n_layers
    _SEQ_LEN = args.seq_len
    _MAX_BATCH_SIZE = args.max_batch_size

    from neuronx_distributed.trace import parallel_model_save, parallel_model_trace

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"[compile] tp={args.tp}  n_layers={args.n_layers}  seq_len={args.seq_len}  "
          f"weights={'real' if args.load_weights else 'random-init'}")

    compiler_args = [
        "--model-type=transformer",
        "--auto-cast=none",
        f"-O{args.opt_level}",
    ]
    if not args.no_mixed_precision_accumulation:
        compiler_args.append("--enable-mixed-precision-accumulation")
    compiler_args += args.extra_compiler_args.split()
    print(f"[compile] compiler_args={compiler_args}")

    with open(args.config) as f:
        preflight_host_ram(json.load(f), args.n_layers, args.tp,
                           args.max_parallel_compilations, force=args.force)

    workdir = args.compiler_workdir or (
        f"/tmp/dsv4_ws_tp{args.tp}_L{args.n_layers}_S{args.seq_len}"
    )
    os.makedirs(workdir, exist_ok=True)
    print(f"[compile] compiler_workdir={workdir}")

    t0 = time.perf_counter()
    kwargs = dict(
        tp_degree=args.tp,
        compiler_workdir=workdir,
        compiler_args=compiler_args,
    )
    if args.max_parallel_compilations is not None:
        kwargs["max_parallel_compilations"] = args.max_parallel_compilations
    parallel_model = parallel_model_trace(
        _picklable_factory,
        _example_inputs(),
        **kwargs,
    )
    dt = time.perf_counter() - t0
    print(f"[compile] done in {dt:.1f}s")

    parallel_model_save(parallel_model, args.out_dir)
    print(f"[compile] saved -> {args.out_dir}")


if __name__ == "__main__":
    main()
