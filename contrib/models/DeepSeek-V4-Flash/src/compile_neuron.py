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
_HF_INFERENCE_DIR = "/mnt/nvme/models/DeepSeek-V4-Flash/inference"

for p in (_HERE, _CONTRIB_DIR, _HF_INFERENCE_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)


# --- Module-level state for the picklable factory + loader ---
# parallel_model_trace re-imports this module in each spawned rank, so any
# state we need at factory time has to come from env vars or globals set at
# module-import time.
_MODEL_PATH = os.environ.get("DSV4_MODEL_PATH", "/mnt/nvme/models/DeepSeek-V4-Flash")
_CONFIG_JSON = os.environ.get(
    "DSV4_CONFIG",
    "/mnt/nvme/models/DeepSeek-V4-Flash/inference/config.json",
)
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
    """Wrap the HF Transformer to bypass its @torch.inference_mode() decorator.

    parallel_model_trace uses torch-xla, which needs to bump tensor version
    counters during tracing. Tensors created inside @inference_mode() are
    frozen and cannot record versions, causing:
      RuntimeError: Cannot set version_counter for inference tensor.
    We call the underlying forward directly with plain no_grad instead.
    """

    def __init__(self, inner: torch.nn.Module):
        super().__init__()
        self.inner = inner

    def forward(self, input_ids: torch.Tensor, start_pos: int = 0):
        with torch.no_grad():
            # Peel the @torch.inference_mode() decorator. HF's Transformer.forward
            # is a plain function attribute; getattr(func, "__wrapped__", func)
            # gives the un-decorated function.
            raw = getattr(self.inner.forward, "__wrapped__", None)
            if raw is not None:
                return raw(self.inner, input_ids, start_pos)
            return self.inner.forward(input_ids, start_pos)


def _picklable_factory():
    """Runs in each spawned trace subprocess. Must return (model, aliases)."""
    for p in (_HERE, _CONTRIB_DIR, _HF_INFERENCE_DIR):
        if p not in sys.path:
            sys.path.insert(0, p)
    _wire_shims()
    import model as hf
    _patch_hf_model_for_xla(hf)
    torch.set_default_dtype(torch.bfloat16)
    m_args = _build_model_args(hf)
    m = hf.Transformer(m_args).eval()
    return _TraceWrapper(m), {}


def _picklable_checkpoint_loader():
    """Return the full dequantized BF16 state_dict for NxD to shard rank-aware."""
    for p in (_HERE, _CONTRIB_DIR, _HF_INFERENCE_DIR):
        if p not in sys.path:
            sys.path.insert(0, p)
    from dequant_checkpoint import dequant_shard
    from glob import glob
    state = {}
    for shard in sorted(glob(os.path.join(_MODEL_PATH, "model-*.safetensors"))):
        deq, _ = dequant_shard(shard, dry_run=False)
        # Drop `.scale` and drop MTP-layer tensors we don't need
        for k, v in deq.items():
            if k.endswith(".scale"):
                continue
            if k.startswith("mtp."):
                continue
            # Prune tensors for layers beyond _N_LAYERS
            if k.startswith("layers."):
                lid = int(k.split(".")[1])
                if lid >= _N_LAYERS:
                    continue
            state[k] = v
    print(f"[loader] {len(state)} tensors kept for n_layers={_N_LAYERS}", flush=True)
    return state


def _example_inputs():
    """input_ids: (B, S), start_pos: int → HF Transformer.forward signature."""
    ids = torch.zeros(_MAX_BATCH_SIZE, _SEQ_LEN, dtype=torch.long)
    return (ids, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tp", type=int, default=2)
    ap.add_argument("--n-layers", type=int, default=2)
    ap.add_argument("--seq-len", type=int, default=16)
    ap.add_argument("--max-batch-size", type=int, default=1)
    ap.add_argument("--out-dir", default="/tmp/dsv4_smoke")
    ap.add_argument("--model-path", default="/mnt/nvme/models/DeepSeek-V4-Flash")
    ap.add_argument("--config",
                    default="/mnt/nvme/models/DeepSeek-V4-Flash/inference/config.json")
    ap.add_argument("--spmd", action="store_true",
                    help="Use SPMD mode: compile one rank then generate the rest "
                         "via checkpoint_loader_callable. Much faster + lower "
                         "memory for large models; but the loader dict must be "
                         "picklable / serializable across process boundaries.")
    args = ap.parse_args()

    # Publish factory args via env (spawned subprocesses re-import this module).
    os.environ["DSV4_MODEL_PATH"] = args.model_path
    os.environ["DSV4_CONFIG"] = args.config
    os.environ["DSV4_N_LAYERS"] = str(args.n_layers)
    os.environ["DSV4_SEQ_LEN"] = str(args.seq_len)
    os.environ["DSV4_MAX_BATCH_SIZE"] = str(args.max_batch_size)

    global _MODEL_PATH, _CONFIG_JSON, _N_LAYERS, _SEQ_LEN, _MAX_BATCH_SIZE
    _MODEL_PATH = args.model_path
    _CONFIG_JSON = args.config
    _N_LAYERS = args.n_layers
    _SEQ_LEN = args.seq_len
    _MAX_BATCH_SIZE = args.max_batch_size

    from neuronx_distributed.trace import parallel_model_save, parallel_model_trace

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"[compile] tp={args.tp}  n_layers={args.n_layers}  seq_len={args.seq_len}")

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
    if args.spmd:
        kwargs.update(
            spmd_mode=True,
            inline_weights_to_neff=False,
            checkpoint_loader_callable=_picklable_checkpoint_loader,
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
