# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cache manager for DeepSeek-V4-Flash's heterogeneous, ragged per-layer state.

Why not `KVCacheManager`
------------------------
NxDI's manager assumes exactly one K and one V tensor per layer, all the same
shape `(batch, kv_heads, max_len, head_dim)`, all zero-initialised, and indexes
them as pairs (`get_cache` walks `range(len(past_key_values) // 2)`). This model
has none of those properties:

  * **three kinds of state, not two** — a sliding-window + compressed-tail
    attention cache, the compressor's `(kv_state, score_state)` ring, and for
    ratio-4 layers a second whole cache plus ring inside the Indexer.
  * **ragged across layers** — `compress_ratios` is `[0, 0, 4, 128, 4, ...]`, so
    layer 0 owns 1 state, layer 3 owns 3, layer 2 owns 6. Different shapes too.
  * **not K/V pairs** — pairing is meaningless here, so `// 2` indexing is wrong.

What is reusable is the *contract*: `past_key_values` is a flat
`nn.ParameterList`, and `DecoderModelInstance.get()` builds the alias map by
enumerating it (model_wrapper.py:1629) with no pairing assumption. So a manager
that publishes a flat, stably-ordered list of Parameters satisfies the framework
even when the states are heterogeneous. That is what this does.

Ordering is the load-bearing property. The alias map is positional: output
`n_real_outputs + i` is written back over `past_key_values[i]`. If the order this
publishes ever disagrees with the order the graph returns its state outputs, the
runtime writes the wrong cache and the failure is silent — a model that answers
plausibly from corrupted context. `LayerKind.state_specs()` owns the per-layer
order; this class owns the concatenation, and `slot_index()` is the single lookup
both the graph and any debugging use, so the two cannot drift.

Initial values
--------------
Everything is zero-initialised, including the compressor score rings that HF
creates as `-inf`. That is safe *only because* `compress_state.ring_slot_mask`
masks unwritten ring slots inside the graph, so results no longer depend on the
fill (verified exactly, 4 init values x 28 ring states, in
test_compress_state_mask.py). Zeroing is not merely convenient: NxD's
`StateInitializer` hardcodes `torch.zeros` with no hook (base_nxd_model.py:31),
so a manager that needed -inf would be fighting the framework.
"""

from typing import Dict, List, Optional, Tuple

import torch
from torch import nn


class DSV4CacheManager(nn.Module):
    """Flat, ordered, heterogeneous state for the whole model.

    Publishes `past_key_values` (the name NxDI's aliasing code looks for) as a
    flat `nn.ParameterList`, plus a name -> index map so callers address state by
    meaning rather than by position.
    """

    def __init__(self, layer_kinds, batch_size, dtype=torch.bfloat16):
        super().__init__()
        self.batch_size = batch_size
        self.dtype = dtype
        self.layer_kinds = layer_kinds

        params: List[nn.Parameter] = []
        self._index: Dict[Tuple[int, str], int] = {}
        self._shapes: List[Tuple[int, ...]] = []

        for kind in layer_kinds:
            for name, shape, init in kind.state_specs():
                full = (batch_size,) + tuple(shape)
                t = torch.zeros(full, dtype=dtype) if init == 0.0 else \
                    torch.full(full, init, dtype=dtype)
                self._index[(kind.layer_idx, name)] = len(params)
                self._shapes.append(full)
                params.append(nn.Parameter(t, requires_grad=False))

        # The attribute name matters: DecoderModelInstance.get() reads
        # `module.kv_mgr.past_key_values`.
        self.past_key_values = nn.ParameterList(params)

    # -- addressing ---------------------------------------------------------

    def slot_index(self, layer_idx: int, name: str) -> int:
        """Position of one state in the flat list == its alias output offset.

        Raising rather than returning -1 on a miss: a typo here would otherwise
        silently alias the wrong tensor, which is the failure mode this class is
        most exposed to.
        """
        key = (layer_idx, name)
        if key not in self._index:
            raise KeyError(
                f"no state {name!r} on layer {layer_idx}; that layer publishes "
                f"{[n for (li, n) in self._index if li == layer_idx]}"
            )
        return self._index[key]

    def get(self, layer_idx: int, name: str) -> torch.Tensor:
        return self.past_key_values[self.slot_index(layer_idx, name)]

    def layer_state_names(self, layer_idx: int) -> List[str]:
        return [n for (li, n) in self._index if li == layer_idx]

    def __len__(self) -> int:
        return len(self.past_key_values)

    # -- the alias map ------------------------------------------------------

    def alias_map(self, n_real_outputs: int) -> Dict[torch.Tensor, int]:
        """`{state_tensor: output_index}`, the form NxD's tracing wants.

        Mirrors `DecoderModelInstance.get()`'s construction
        (model_wrapper.py:1629) so the positional convention is identical:
        graph output `n_real_outputs + i` aliases `past_key_values[i]`.
        """
        return {t: n_real_outputs + i
                for i, t in enumerate(self.past_key_values)}

    def output_order(self) -> List[Tuple[int, str]]:
        """(layer_idx, name) per flat position, for building graph outputs.

        The graph must append its state outputs in exactly this order. Exposing
        it means the producer and the alias map read from one source instead of
        both re-deriving it.
        """
        order: List[Optional[Tuple[int, str]]] = [None] * len(self._index)
        for key, i in self._index.items():
            order[i] = key
        return order  # type: ignore[return-value]

    # -- diagnostics -------------------------------------------------------

    def describe(self) -> str:
        lines = [f"{len(self)} states, batch={self.batch_size}, dtype={self.dtype}"]
        for i, (layer_idx, name) in enumerate(self.output_order()):
            lines.append(f"  [{i:>3}] layer{layer_idx}.{name} "
                         f"{tuple(self._shapes[i])}")
        return "\n".join(lines)

    def total_bytes(self) -> int:
        elem = torch.empty((), dtype=self.dtype).element_size()
        return sum(
            elem * int(torch.tensor(s).prod()) for s in self._shapes
        )
