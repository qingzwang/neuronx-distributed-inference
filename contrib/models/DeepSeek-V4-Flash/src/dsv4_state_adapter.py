# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Let the validated prefill/decode forwards read state from DSV4CacheManager.

Design note: the forwards in `prefill_patches` and `decode_patches` are *not*
being rewritten. They are the part of this port with the strongest evidence behind
them — bit-identical logits and state against the in-place reference, mutation
tested (a +1 ring offset fails at max|d|=6.52 while logits still pass), and
validated on device at 43 layers. Rewriting them to move onto NxDI's model base
would put the one well-tested thing at risk to change plumbing.

So instead this adapts the plumbing. Both forwards already access every piece of
state through a `StateSink` (`sink.get(module, name)` / `sink.put(...)`), which was
introduced because functional `index_copy` cannot write through HF's view aliasing.
That indirection is exactly the seam needed here: point the sink at
`DSV4CacheManager` instead of at the modules' own Parameters, and the same forwards
read and write the framework-owned cache with no change to their arithmetic.

Two things this has to get right:

**Identity mapping.** The forwards address state as `(module_object, buffer_name)`;
the manager addresses it as `(layer_idx, dotted_name)`. The map between them is
built once by walking the model, so a mismatch shows up as a KeyError at build time
rather than as a silently-wrong tensor at run time.

**HF's view aliasing.** `compressor.kv_cache` is a *view* into the enclosing
Attention's cache (`self.kv_cache[:, win:]`), and the Indexer's compressor writes
into the Indexer's own cache from column 0. `StateSink` already models this as an
owner/offset redirect; this preserves those redirects, because the manager holds
one tensor per *owner*, not one per accessor — the same reason the manager
publishes 17 states at 5 layers rather than 22.
"""

from typing import Dict, List, Optional, Tuple

import torch

from decode_patches import StateSink


def build_state_map(model, hf_mod, layer_kinds) -> Dict[Tuple[int, str], object]:
    """`(layer_idx, dotted_name) -> module object` for every published state.

    Walks the real model so the mapping is derived rather than assumed. The dotted
    names must match `LayerKind.state_specs()`, which is what the manager keys on.
    """
    out: Dict[Tuple[int, str], object] = {}
    for kind in layer_kinds:
        attn = model.layers[kind.layer_idx].attn
        out[(kind.layer_idx, "kv_cache")] = attn
        if kind.has_compressor:
            if getattr(attn, "compressor", None) is None:
                raise RuntimeError(
                    f"layer {kind.layer_idx} declares a compressor "
                    f"(ratio={kind.ratio}) but the model has none"
                )
            out[(kind.layer_idx, "compressor.kv_state")] = attn.compressor
            out[(kind.layer_idx, "compressor.score_state")] = attn.compressor
        if kind.has_indexer:
            if getattr(attn, "indexer", None) is None:
                raise RuntimeError(
                    f"layer {kind.layer_idx} declares an indexer "
                    f"(ratio={kind.ratio}) but the model has none"
                )
            out[(kind.layer_idx, "indexer.kv_cache")] = attn.indexer
            out[(kind.layer_idx, "indexer.compressor.kv_state")] = \
                attn.indexer.compressor
            out[(kind.layer_idx, "indexer.compressor.score_state")] = \
                attn.indexer.compressor
    return out


def build_managed_sink(model, hf_mod, layer_kinds, manager) -> StateSink:
    """A StateSink whose owner values come from `manager`, not from the modules.

    The forwards then operate on framework-owned state without knowing it. Also
    re-installs the two redirects and the per-forward `freqs_cis` wiring that
    `decode_patches.build_sink` does, for the same reasons documented there:

      * the compressor's cache is the tail of the enclosing layer's cache, so a
        write through the view has to be spliced back onto the owner;
      * `freqs_cis` must be wired per forward rather than at patch time, because
        only Attention registers it as a buffer and `_apply` does not move plain
        attributes — wiring it early pins a host constant into the graph.
    """
    sink = StateSink()
    state_map = build_state_map(model, hf_mod, layer_kinds)

    # Seed owners from the manager. `sink.get` resolves redirects first, so only
    # real owners need registering.
    for (layer_idx, dotted), mod in state_map.items():
        leaf = dotted.split(".")[-1]
        sink.register_owner(mod, leaf, manager.get(layer_idx, dotted))

    # Re-apply HF's view aliasing as explicit redirects.
    for kind in layer_kinds:
        attn = model.layers[kind.layer_idx].attn
        if not kind.has_compressor:
            continue
        sink.register_redirect(attn.compressor, "kv_cache",
                               attn, "kv_cache", kind.window_size)
        attn.compressor.freqs_cis = attn.freqs_cis
        if kind.has_indexer:
            sink.register_redirect(attn.indexer.compressor, "kv_cache",
                                   attn.indexer, "kv_cache", 0)
            attn.indexer.freqs_cis = attn.freqs_cis
            attn.indexer.compressor.freqs_cis = attn.freqs_cis
    return sink


def collect_outputs(sink, manager, state_map) -> List[torch.Tensor]:
    """Post-forward state, in the manager's published order.

    This is the other half of the aliasing contract: the graph must append its
    state outputs in exactly the order `manager.output_order()` reports, because
    NxD writes output `n_real_outputs + i` back over `past_key_values[i]`. Reading
    the order from the manager rather than re-deriving it here is what keeps the
    producer and the alias map from drifting — a drift that would corrupt the
    cache silently.

    `state_map` comes from `build_state_map` and is passed in rather than rebuilt,
    because this runs inside the traced graph on every forward.
    """
    outs = []
    for layer_idx, dotted in manager.output_order():
        mod = state_map[(layer_idx, dotted)]
        outs.append(sink.get(mod, dotted.split(".")[-1]))
    return outs
