# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""One all-reduce that works under both tracing paths.

HF's model.py calls `torch.distributed.all_reduce(x)` with no group argument,
which resolves the *default* process group. That works under
`parallel_model_trace`, which really does call `init_process_group` in each
spawned rank.

It does not work under `ModelBuilder`. ModelBuilder traces rank 0 only, inside
`mock_distributed(world_size=N)`, which replaces `init_process_group`,
`get_world_size`, `get_rank` and `new_group` with stubs — but leaves
`all_reduce` as the real function. So the call goes through to real c10d, asks
for the default group, and finds none:

    ValueError: Default process group has not been initialized,
                please make sure to call init_process_group.

NxD's own parallel layers never hit this because they pass an explicit group from
`parallel_state.get_tensor_model_parallel_group()`, which `mock_distributed`
*does* provide (as a MagicMock carrying the replica mesh — enough for the XLA
lowering to emit the right collective).

So: resolve the TP group when one exists, and fall back to the default group when
it does not. `parallel_model_trace` initialises the TP group too, so this is the
correct call on both paths rather than a ModelBuilder special case.
"""

import torch


def _dist():
    """`torch.distributed` resolved NOW, not at import time.

    This indirection is the whole point. `mock_distributed` works by *replacing
    the `torch.distributed` attribute on the torch module* for the duration of
    tracing. A module-level `import torch.distributed as dist` captures the real
    one before that swap and keeps using it, so `get_world_size()` returns 1 even
    at tp=32 — and HF, which reads it in `Transformer.__init__`, then builds
    unsharded modules whose shapes disagree with the sharded state around them.

    That mismatch is what produced "size of tensor a (128) must match tensor b
    (0)": with world_size stuck at 1, `n_local_heads` and the o_proj split were
    computed for a single rank, and a downstream index tensor came out empty.

    Resolving per call sees whatever `torch.distributed` is bound to at that
    moment, mocked or not.
    """
    return torch.distributed


def tp_group():
    """The tensor-parallel process group, or None if NxD's state is not up.

    Returning None rather than raising: some callers (single-rank CPU tests) run
    with no parallel state at all and should fall through to the default group.
    """
    try:
        from neuronx_distributed.parallel_layers import parallel_state
    except ImportError:
        return None
    try:
        if not parallel_state.model_parallel_is_initialized():
            return None
        return parallel_state.get_tensor_model_parallel_group()
    except (AssertionError, RuntimeError):
        return None


def all_reduce(tensor):
    """In-place all-reduce over the TP group, usable under mock_distributed.

    Mirrors `dist.all_reduce(tensor)`'s contract (sum, in place, returns None)
    so it is a drop-in for HF's call sites.
    """
    group = tp_group()
    if group is None:
        _dist().all_reduce(tensor)
    else:
        _dist().all_reduce(tensor, group=group)


def all_gather(tensor_list, tensor):
    """`dist.all_gather` over the TP group."""
    group = tp_group()
    if group is None:
        _dist().all_gather(tensor_list, tensor)
    else:
        _dist().all_gather(tensor_list, tensor, group=group)


class _GroupAwareDist:
    """A stand-in for `torch.distributed` that pins collectives to the TP group.

    HF's model.py calls `dist.all_reduce(y)` / `dist.all_gather(...)` in five
    places of its own (ParallelEmbedding, RowParallelLinear, Indexer, MoE,
    ParallelHead). Patching each one means re-implementing five HF methods just
    to change their group argument, and any HF update adds more. Substituting the
    module its `dist` name resolves to fixes all of them at once, including the
    ones this port does not otherwise touch.

    Everything except the group-taking collectives falls through to the real
    torch.distributed.
    """

    all_reduce = staticmethod(all_reduce)
    all_gather = staticmethod(all_gather)

    def __getattr__(self, name):
        # Resolve against the *current* torch.distributed so get_world_size(),
        # get_rank() and is_initialized() see mock_distributed's stubs while
        # tracing. Binding these at import time is what left world_size at 1.
        return getattr(_dist(), name)


def patch_hf_dist(hf_mod):
    """Point HF's module-level `dist` at the group-aware shim.

    Needed under ModelBuilder, which traces rank 0 inside `mock_distributed`.
    That context stubs `init_process_group`/`get_world_size`/`new_group` but
    leaves `all_reduce` real, so a group-less call reaches c10d and raises
    "Default process group has not been initialized". NxD's own layers avoid this
    by always passing an explicit group.

    Harmless on the `parallel_model_trace` path, which does initialise a real
    default group: the TP group it also creates is the correct one to reduce over
    either way.
    """
    hf_mod.dist = _GroupAwareDist()
