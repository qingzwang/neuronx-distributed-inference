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
    _patch_hf_model_for_xla(hf)
    _patch_moe_forward_for_xla(hf)
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

    t0 = time.perf_counter()
    kwargs = dict(
        tp_degree=args.tp,
        compiler_workdir=f"/tmp/dsv4_ws_tp{args.tp}_L{args.n_layers}_S{args.seq_len}",
        compiler_args=[
            "--model-type=transformer",
            "--auto-cast=none",
            "-O1",
            "--enable-mixed-precision-accumulation",
        ],
    )
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
