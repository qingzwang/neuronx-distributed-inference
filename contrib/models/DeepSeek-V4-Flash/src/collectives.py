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


def patch_parallel_head_all_gather(hf_mod):
    """Replace ParallelHead's in-place `all_gather` with a functional one.

    This is the bug that made the joint graph return all-zero logits while
    correctly writing all 17 KV state buffers. HF's head does:

        all_logits = [torch.empty_like(logits) for _ in range(world_size)]
        dist.all_gather(all_logits, logits)
        logits = torch.cat(all_logits, dim=-1)

    `dist.all_gather` writes into `tensor_list` **in place**. XLA tracing records
    dataflow, not side effects on pre-existing tensors, so the trace keeps the
    `empty_like` placeholders and never connects the gathered values. Reading the
    compiled HLO shows exactly that: root tuple element 0 is
    `add(concatenate(32 x broadcast(constant)), ...)` — all 32 rank slices are
    broadcast constants, so the logits output was a constant the whole time. The
    state outputs were fine because they flow through functional `index_copy`.

    `all_gather_into_tensor` is the functional form: one output tensor, returned
    rather than mutated, so the trace captures it. The output layout is the
    concatenation along dim 0 of each rank's contribution, so for a
    `[batch, vocab_shard]` input the result is `[world_size * batch, vocab_shard]`
    and has to be reshaped to `[batch, world_size * vocab_shard]` to match what
    `cat(..., dim=-1)` produced.

    Worth stating why this did not show up on the `parallel_model_trace` path: that
    path traces each rank in its own process with a real process group, where the
    in-place write happens for real during tracing and the recorded graph picks up
    the resulting values.
    """
    import torch

    def parallel_head_forward(self, x, hc_fn, hc_scale, hc_base, norm):
        x = self.hc_head(x, hc_fn, hc_scale, hc_base)
        logits = self.get_logits(norm(x))
        ws = hf_mod.world_size
        if ws > 1:
            # xm.all_gather RETURNS the gathered tensor rather than filling one
            # that was allocated beforehand. That distinction is the whole fix:
            # both `dist.all_gather(list, t)` and
            # `dist.all_gather_into_tensor(out, t)` write into memory the caller
            # already owns, and XLA tracing records dataflow rather than writes to
            # pre-existing buffers, so the trace keeps the placeholder and the
            # gathered values never reach the output. Confirmed by reading the HLO
            # both times: with the list form root element 0 was
            # concatenate(32 x broadcast(constant)); with the into_tensor form it
            # became a real transpose/reshape chain that still bottomed out in a
            # broadcast constant.
            #
            # xm.all_gather over dim=-1 reproduces HF's `cat(all_logits, dim=-1)`
            # directly, so no reshape gymnastics are needed either.
            import torch_xla.core.xla_model as xm
            # `groups` must be the TP replica mesh. Without it xm.all_gather
            # degenerates to a local copy: the output came back with exactly
            # 4040 = 129280/32 non-zero entries, i.e. only this rank's own vocab
            # shard, the rest untouched.
            #
            # Under mock_distributed the TP group is a MagicMock carrying `_mesh`,
            # which is where NxD's own layers read their replica groups from.
            groups = None
            g = tp_group()
            mesh = getattr(g, "_mesh", None)
            if mesh is not None:
                groups = [list(m) for m in mesh]
            logits = xm.all_gather(logits, dim=-1, groups=groups)
        return logits

    hf_mod.ParallelHead.forward = parallel_head_forward
