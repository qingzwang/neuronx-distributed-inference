# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Rank-aware weight loading for the HF DeepSeek-V4-Flash reference at TP=N.

Why this exists
---------------
The obvious approach — hand NxD's `parallel_model_trace(spmd_mode=True,
checkpoint_loader_callable=...)` the full checkpoint and let it shard — does
not work for this model, and fails *silently*.

NxD shards a checkpoint in `trace.shard_children()`, which early-returns
unless a module is an `isinstance` of NxD's own
`__SUPPORTED_SHARDED_MODULES` (NxD's `ColumnParallelLinear`,
`RowParallelLinear`, `ParallelEmbedding`, ...). HF's `inference/model.py`
declares its *own* classes with those same names:

    class ColumnParallelLinear(Linear):   # model.py — NOT neuronx_distributed's
        def __init__(self, in_features, out_features, ...):
            self.part_out_features = out_features // world_size

They share nothing but the name, so `shard_children` matches none of them
and shards nothing. Every rank then gets handed full-size tensors for
rank-sized parameters, and the load either shape-mismatches or (worse)
quietly leaves weights at their init values.

So we do the sharding ourselves, mirroring HF's convention exactly. HF's
`Transformer.__init__` already allocates *rank-local* shapes (it reads
`world_size`/`rank` from the initialized process group), so all we have to
do is copy the matching slice of each full checkpoint tensor.

HF's sharding convention (read off model.py)
--------------------------------------------
    ParallelEmbedding      weight [vocab // ws, dim]           -> shard dim 0
    ParallelHead           weight [vocab // ws, dim]           -> shard dim 0
    ColumnParallelLinear   weight [out // ws, in]              -> shard dim 0
    RowParallelLinear      weight [out, in // ws]              -> shard dim 1
    Linear                 replicated
    Attention.attn_sink    [n_heads // ws]                     -> shard dim 0
    MoE routed experts     experts[i] is None unless
                           rank*n_local <= i < (rank+1)*n_local -> whole tensor
    everything else (norms, hc_*, gate, shared_experts, ape)   replicated

Rather than trust that table blindly, `_shard_spec()` derives the shard dim
from the *actual* shapes (model param vs checkpoint tensor) and only uses
the module class to cross-check. A tensor whose local and full shapes match
is replicated; exactly one dim may differ, and it must differ by a factor of
`world_size`. Anything else raises instead of silently loading garbage.
"""

import json
import os
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import torch
from safetensors import safe_open

import dequant_checkpoint as dq


def _load_weight_map(ckpt_dir: str) -> Dict[str, str]:
    """key -> shard filename, from the checkpoint's index json."""
    idx_path = os.path.join(ckpt_dir, "model.safetensors.index.json")
    with open(idx_path) as f:
        return json.load(f)["weight_map"]


def _shard_spec(
    name: str,
    local_shape: torch.Size,
    full_shape: Tuple[int, ...],
    world_size: int,
) -> Optional[int]:
    """Return the dim to shard `name` along, or None if it is replicated.

    Derived from shapes, not from a hardcoded name table, so it stays correct
    if HF changes a layer's parallelism. Raises on anything ambiguous.
    """
    local = tuple(local_shape)
    full = tuple(full_shape)

    if local == full:
        return None  # replicated

    if len(local) != len(full):
        raise RuntimeError(
            f"{name}: rank-local rank {len(local)} != checkpoint rank "
            f"{len(full)} (local={local}, full={full})"
        )

    differing = [i for i, (a, b) in enumerate(zip(local, full)) if a != b]
    if len(differing) != 1:
        raise RuntimeError(
            f"{name}: expected exactly one sharded dim, found {differing} "
            f"(local={local}, full={full})"
        )
    d = differing[0]
    if full[d] != local[d] * world_size:
        raise RuntimeError(
            f"{name}: dim {d} is {full[d]} in the checkpoint but "
            f"{local[d]} locally, which is not a 1/{world_size} slice"
        )
    return d


def _slice_for_rank(
    tensor: torch.Tensor, dim: Optional[int], rank: int, world_size: int
) -> torch.Tensor:
    if dim is None:
        return tensor
    n = tensor.size(dim) // world_size
    return tensor.narrow(dim, rank * n, n)


def plan_rank_load(
    model: torch.nn.Module,
    weight_map: Dict[str, str],
    rank: int,
    world_size: int,
) -> Tuple[Dict[str, List[str]], List[str]]:
    """Group the parameters this rank needs by the shard file holding them.

    Returns (shard_file -> [param names], missing names). `missing` is
    expected to be non-empty and benign: non-persistent buffers such as
    `freqs_cis` / `kv_cache` / `kv_state` are computed at init, not stored in
    the checkpoint.
    """
    wanted = dict(model.named_parameters())
    wanted.update(dict(model.named_buffers()))

    by_file: Dict[str, List[str]] = defaultdict(list)
    missing: List[str] = []
    for name in wanted:
        shard = weight_map.get(name)
        if shard is None:
            missing.append(name)
        else:
            by_file[shard].append(name)
    return by_file, missing


def _dequant_one(handle, key: str, available: set) -> torch.Tensor:
    """Fetch `key` from an open safetensors handle, dequantizing if needed.

    Mirrors dequant_checkpoint.dequant_shard's dispatch, but for a single key
    so we never pay to dequantize tensors this rank does not want. That
    matters: a full-shard dequant is ~17 s, and at small `n_layers` we need
    only a handful of keys per shard.
    """
    scale_key = key.replace(".weight", ".scale")
    has_scale = scale_key in available

    if dq._is_expert_weight(key) and has_scale:
        return dq.dequant_fp4_expert(handle.get_tensor(key), handle.get_tensor(scale_key))
    if dq._is_fp8_attn_weight(key) and has_scale:
        return dq.dequant_fp8_block(handle.get_tensor(key), handle.get_tensor(scale_key))

    t = handle.get_tensor(key)
    if t.dtype in (torch.float8_e4m3fn, torch.int8) and has_scale:
        raise RuntimeError(
            f"{key}: dtype {t.dtype} has a .scale sibling but matched no "
            f"dequant rule — it would be loaded as raw quantized bits."
        )
    return t


def load_rank_weights(
    model: torch.nn.Module,
    ckpt_dir: str,
    rank: int,
    world_size: int,
    verbose: bool = True,
) -> Dict[str, int]:
    """Copy this rank's slice of every checkpoint tensor into `model`.

    `model` must already be built with rank-local shapes (i.e. HF's
    Transformer constructed under an initialized process group of size
    `world_size`).
    """
    weight_map = _load_weight_map(ckpt_dir)
    targets = dict(model.named_parameters())
    targets.update(dict(model.named_buffers()))
    by_file, missing = plan_rank_load(model, weight_map, rank, world_size)

    stats = {"copied": 0, "replicated": 0, "sharded": 0, "skipped_absent": len(missing)}

    for shard_file in sorted(by_file):
        path = os.path.join(ckpt_dir, shard_file)
        with safe_open(path, framework="pt", device="cpu") as handle:
            available = set(handle.keys())
            for name in sorted(by_file[shard_file]):
                if name not in available:
                    # Indexed to this shard but absent — treat as a hard error
                    # rather than leaving the parameter at its init value.
                    raise RuntimeError(f"{name}: indexed to {shard_file} but not present")
                param = targets[name]
                full = _dequant_one(handle, name, available)
                dim = _shard_spec(name, param.shape, tuple(full.shape), world_size)
                piece = _slice_for_rank(full, dim, rank, world_size)
                if tuple(piece.shape) != tuple(param.shape):
                    raise RuntimeError(
                        f"{name}: sliced to {tuple(piece.shape)} but the model "
                        f"wants {tuple(param.shape)}"
                    )
                with torch.no_grad():
                    param.copy_(piece.to(param.dtype))
                stats["copied"] += 1
                stats["sharded" if dim is not None else "replicated"] += 1
                del full, piece

    if verbose:
        print(
            f"[shard_loader rank={rank}/{world_size}] copied={stats['copied']} "
            f"(sharded={stats['sharded']} replicated={stats['replicated']}) "
            f"absent_from_ckpt={stats['skipped_absent']}",
            flush=True,
        )
    return stats
