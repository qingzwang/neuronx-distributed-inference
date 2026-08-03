# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reset the decode graph's device-resident KV cache between independent prompts.

STATUS: DOES NOT WORK. Kept because the diagnosis is the useful part.
------------------------------------------------------------------------
Three host-side write paths were tried against a loaded 43-layer artifact, and
all three leave what the graph reads unchanged:

    param.fill_(v)                  returns fine, graph unaffected
    param.data.copy_(host_tensor)   returns fine, graph unaffected
    states._parameters[k] = Parameter(host.to(dev))   readback OK, graph unaffected

The last one is the informative failure. Replacing the parameter *does* change
what `named_parameters()` reports — `_verify_reset` reads the new value back and
passes — and the model still generates the pre-reset continuation, byte for byte.
So `NeuronModule.forward`'s `inputs.extend(self.states)` is not the path the
aliased state actually travels on a *loaded* artifact: `forward_v2` binds its
device buffers once and the NEFF updates them in place thereafter. The Python
parameters are the seed for that binding, not a live view of it.

ROOT CAUSE (found later): the port is on the wrong tracing API. NxD *does*
support host-side state access — `NxDModel.read_from_neuron_buffer` and
`.write_to_neuron_buffer` (neuronx_distributed/trace/nxd_model/nxd_model.py:355
and :382) — but those live on `NxDModel`, which is what `ModelBuilder` returns.
`parallel_model_trace` returns `TensorParallelNeuronModel`, which has no such
method:

    [m for m in dir(TensorParallelNeuronModel) if 'buffer' in m]
    -> ['get_buffer', 'named_buffers', 'register_buffer']   # plain nn.Module
    [m for m in dir(NxDModel) if 'buffer' in m]
    -> [..., 'read_from_neuron_buffer', 'write_to_neuron_buffer']

So this is not a missing capability in the runtime, it is a capability the port
cannot reach from the API it was built on. Fixing it properly means moving
`compile_neuron.py` to `ModelBuilder`, which also unblocks joint prefill+decode
for the same reason — see JOINT_INFERENCE.md.

Until then: reloading the artifact (~440 s + warmup) is the only way to get a
clean cache, which is far above the 94 ms/token the graph is worth, so
multi-prompt runs carry the contamination measured below.

`test_cache_reset.py` measures the contamination this leaves: running prompt B
after prompt A changes B's tokens, max|dlogit| = 2.36. Any multi-prompt
evaluation on one loaded artifact carries that error.

Everything below documents the mechanism and the failed attempts.

Why this is needed
------------------
The decode artifact keeps its KV cache on the device across calls, aliased to
the graph's extra outputs (see decode_patches). That is exactly what makes
decode cheap, and it is also why a *second* prompt cannot simply start over:
`run_neuron.py --mode decode` re-enters at `start_pos = 0`, but the cache still
holds the previous prompt's keys and values. Attention would score the new
query against the old context.

Some of that is self-healing and some is not, which is the part worth being
precise about:

  * `Attention.kv_cache` window slots — self-healing. `window_topk_idxs` masks
    slot `j` unless `j <= pos`, so at low positions the stale tail is not
    referenced. It only matters once a *shorter* second prompt leaves slots the
    first one wrote still inside the mask.
  * `Attention.kv_cache` compressed tail and `Indexer.kv_cache` — NOT
    self-healing in the same way. The compressed entries are indexed by
    `pos // ratio` and the indexer's own top-k reads the whole cache with a
    `-inf` bias on the unwritten tail; a stale entry there is a real KV vector
    with a real score, so it can be selected.
  * `Compressor.kv_state` / `score_state` — NOT self-healing. These are the
    ring accumulator for the current compression window, and the overlap
    variant (ratio == 4) carries the previous window in its first half. A
    leftover value there corrupts the first compressed entry of the new prompt.

So a reset has to zero all of them. One detail matters more than it looks:

    score_state is initialised to -inf, not to zero.

HF does `torch.full(..., float("-inf"))` because the state is consumed by
`softmax(dim=1)` over the ring — an unwritten slot must contribute *zero
weight*, and zero would instead make it contribute `exp(0)`, i.e. a uniform
vote for a slot holding no data. Zeroing this one buffer would silently skew
every compression at the start of a prompt. `_INIT_VALUES` records that.

How the reset reaches the device
--------------------------------
`torch_neuronx` keeps the aliased state as parameters in a `states` child of
each rank's traced module, moved to `privateuseone:<rank>` by
`move_trace_to_device`. `NeuronModule.forward` re-reads `self.states` on every
call, so overwriting them from the host is enough — no recompile, no reload of
the 663 GB artifact.

Getting at them has three traps, each of which fails quietly:

  * `module.model` is not the way in. On a loaded artifact that attribute is the
    C++ `torch.classes.neuron.Model` runtime handle, which has no parameters at
    all — `getattr(m.model, "states")` is None, not an error.
  * `states` is a scripted `ParameterList`, so `len()` and plain iteration raise
    `NotImplementedError` on a `RecursiveScriptModule`. `named_parameters()`
    works. See `_rank_state_tensors`.
  * **`fill_` on a device tensor is a silent no-op.** It returns successfully,
    reports the new value if you read the returned tensor, and leaves what the
    graph reads unchanged. The reset then claims to have written all 6016
    tensors while the model's output stays byte-identical to the un-reset run —
    which reads as "the cache is not the problem" and sends you looking in the
    wrong place. Writes must go through `param.data.copy_(host_tensor)`.
    `reset_decode_cache(verify=True)` reads a slot back so this cannot recur.

The state tensors are anonymous at this level: NxD names them by position, in
the order `collect_state_aliases` produced them. We do not need the mapping back
to module names to reset, only to know which ones are `score_state` — and that
is recoverable from the fill value, since it is the only state HF initialises to
-inf. `reset_decode_cache` therefore takes the per-tensor init value from a
recorded template, falling back to detecting -inf when no template is given.
"""

from typing import Dict, List, Optional

import torch


# Buffer name -> value HF initialises it to. Anything not listed is zero.
# score_state is -inf because it is softmax'd over the ring: an unwritten slot
# must contribute no weight, and 0.0 would contribute exp(0) instead.
_INIT_VALUES = {"score_state": float("-inf")}


def state_init_values(slots) -> List[float]:
    """Per-slot init value, in the order `collect_state_aliases` returned.

    `slots` is that function's [(module, buffer_name)] list, so this is the
    bridge from "which buffer is this" to "what does it reset to".
    """
    return [_INIT_VALUES.get(name, 0.0) for _, name in slots]


def _rank_state_tensors(model) -> List[List[torch.Tensor]]:
    """The aliased state tensors of every rank, in graph-output order.

    Two details about where these live, both of which cost a round trip to
    discover:

      * They hang off the rank's *ScriptModule* as a `states` child, not off
        `module.model` — that attribute is the C++ `torch.classes.neuron.Model`
        runtime handle and has no parameters at all.
      * `states` is a scripted `ParameterList`. `len()` and iteration raise
        `NotImplementedError` on a `RecursiveScriptModule`, so the tensors have
        to come from `named_parameters()`, which does work. Sorting by the
        numeric suffix keeps them in the order `collect_state_aliases` produced,
        which is the order the graph's extra outputs alias.

    `move_trace_to_device` reaches the same objects to push them onto
    `privateuseone:<rank>`, so these are live device tensors and an in-place
    write is visible to the next call — no recompile, no artifact reload.
    """
    return [[p for _, p in _sorted_state_items(m)] for m in model.models]


def _sorted_state_items(rank_module):
    """(name, tensor) for one rank's states, in graph-output order.

    Sorted by the numeric suffix of the ParameterList key, which is the order
    `collect_state_aliases` produced and therefore the order the graph's extra
    outputs alias. Lexicographic order would interleave 1, 10, 11, 2 and quietly
    reset the wrong slots.
    """
    named = (list(rank_module.states.named_parameters())
             if hasattr(rank_module, "states") else [])
    if not named:
        raise RuntimeError(
            "traced module exposes no `states` parameters, so there is no "
            "device-resident KV cache to reset. That means the output "
            "aliases were dropped at trace time — see decode_patches on "
            "why state must be an nn.Parameter, not a register_buffer."
        )

    def _order(item):
        digits = "".join(c for c in item[0] if c.isdigit())
        return int(digits) if digits else 0

    return sorted(named, key=_order)


def reset_decode_cache(model, init_values: Optional[List[float]] = None,
                       verify: bool = True) -> int:
    """Zero (or -inf) every aliased KV state on every rank, in place.

    Call between independent prompts. Without it the next prompt attends to the
    previous one's compressed KV, which does not always show up as garbage — it
    shows up as a slightly wrong answer, which is worse.

    `init_values` is the per-slot fill from `state_init_values(slots)`. When
    omitted, every slot is zeroed *except* those whose current value is already
    -inf everywhere, which are refilled with -inf; that heuristic lets a caller
    that never built the slot list still reset correctly, because score_state is
    the only -inf state and it is -inf at the start of any fresh process.

    Returns the number of state tensors written, summed over ranks, so a caller
    can assert the reset actually reached something.

    `verify` reads one slot back per rank and raises if the write did not land.
    It is on by default because the failure mode is silent: the writes appear to
    succeed and the model keeps answering, just with the previous prompt still in
    its cache.
    """
    written = 0
    for rank, m in enumerate(model.models):
        store = m.states._parameters
        for i, (name, param) in enumerate(_sorted_state_items(m)):
            if init_values is not None and i < len(init_values):
                fill = init_values[i]
            else:
                # No template: infer. Only score_state starts at -inf, and it
                # stays -inf in the slots the current prompt has not written, so
                # `any` rather than `all` still identifies it mid-run. Read the
                # check on the host: `isneginf` on a device tensor is another op
                # with no Neuron lowering.
                fill = float("-inf") if bool(
                    torch.isneginf(param.detach().cpu()).any()
                ) else 0.0
            # Build the fresh state on the host and *replace the parameter*,
            # rather than writing into the existing device tensor. In-place
            # mutation does not work here: both `fill_` and `data.copy_` on a
            # privateuseone tensor return successfully and leave what the graph
            # reads unchanged, so the reset reports having written all 6016
            # tensors while the model's output stays byte-identical to the
            # un-reset run — a silent failure that looks like the cache not
            # being the problem.
            #
            # Replacement works because `NeuronModule.forward` does
            # `inputs.extend(self.states)` on every call, so whatever object is
            # in the ParameterList at call time is what the graph gets. This is
            # the same move `torch_neuronx.move_trace_to_device` makes to put
            # these on the device in the first place, which is the proof that
            # host -> device transfer is supported on this path.
            host = torch.full(tuple(param.shape), fill, dtype=param.dtype)
            store[name] = torch.nn.Parameter(
                host.to(param.device), requires_grad=False,
            )
            written += 1

    if verify:
        _verify_reset(model, init_values)
    return written


def _verify_reset(model, init_values: Optional[List[float]]) -> None:
    """Read back one state slot per rank and confirm the write actually landed.

    Cheap on purpose — one tensor per rank, not all 188 — because the point is to
    catch a write path that silently does nothing (see the `copy_` note above),
    and that fails for every slot or none.
    """
    for rank, tensors in enumerate(_rank_state_tensors(model)):
        if not tensors:
            continue
        want = (init_values[0] if init_values else None)
        got = tensors[0].detach().cpu()
        if want is None:
            # Inferred mode: slot 0 is a kv/score state, so it must now be
            # uniform — either all zero or all -inf.
            uniform = bool((got == got.flatten()[0]).all())
        else:
            uniform = bool((got == want).all())
        if not uniform:
            raise RuntimeError(
                f"rank {rank}: KV state did not reset — the write returned "
                f"successfully but the tensor still holds mixed values. On this "
                f"backend `fill_` on a device tensor is a silent no-op; the "
                f"reset must go through `param.data.copy_(host_tensor)`."
            )


def snapshot_state_norms(model, rank: int = 0) -> List[float]:
    """Per-slot L1 norm of rank `rank`'s state, for asserting a reset happened.

    Norms rather than the tensors themselves: these live on the device and are
    large, and the only question a caller has is "is this still holding the
    previous prompt". -inf slots report inf, which is the expected fresh value
    for score_state.

    Everything is copied to the host *first* and reduced there. Reducing on the
    device instead does not work for this diagnostic, in two escalating ways:
    `.float()` on a device tensor raises "Expected self.dtype() == dst.dtype()"
    (the Neuron backend has no cross-dtype copy on this path), and an
    eager-mode reduction over 188 state tensors deadlocks outright — the process
    blocks with CPU time flat while wall clock advances. The state is a few MB
    per slot, so a host copy is cheap and, unlike the device path, terminates.

    This is a debug helper, not something to call per token.
    """
    out = []
    for param in _rank_state_tensors(model)[rank]:
        t = param.detach().cpu().float()
        finite = torch.isfinite(t)
        if not bool(finite.any()):
            out.append(float("inf"))       # fresh score_state: all -inf
            continue
        out.append(float(t[finite].abs().sum()))
    return out
