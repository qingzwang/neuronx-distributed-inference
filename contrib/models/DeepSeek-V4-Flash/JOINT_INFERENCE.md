# Joint prefill + decode: why it does not work yet, and what NxDI does instead

Status: **not implemented.** Prefill and decode are two independently traced
artifacts with two independent KV caches, and nothing hands one to the other. A
decode run therefore re-ingests the prompt one token at a time, which costs ~14x
on TTFT (12.0 s vs 0.85 s for a 128-token prompt at 43 layers, TP=32).

This document records what the NxDI codebase already does about exactly this
problem, because the port is on the wrong API and that is the whole reason both
this and the KV-cache reset are blocked.

## The root cause: two different tracing APIs

| | `parallel_model_trace` (this port) | `ModelBuilder` (NxDI production) |
|---|---|---|
| returns | `TensorParallelNeuronModel` | `NxDModel` |
| models per artifact | exactly one | many, keyed by tag |
| state | one alias dict per traced graph | one shared `states` dict for the collection |
| host access to state | none | `read_from_neuron_buffer` / `write_to_neuron_buffer` |

Verified on this venv:

```python
[m for m in dir(TensorParallelNeuronModel) if 'buffer' in m]
# -> ['get_buffer', 'named_buffers', 'register_buffer']   (all plain nn.Module)

[m for m in dir(NxDModel) if 'buffer' in m]
# -> [..., 'read_from_neuron_buffer', 'write_to_neuron_buffer']
```

So the two things this port cannot do — share a cache between graphs, and reset a
cache from the host — are both *supported features* that live on the class the
port does not use. `cache_reset.py` failed for this reason: there is no supported
host write path on `TensorParallelNeuronModel`, and the three unsupported ones
(`fill_`, `data.copy_`, parameter replacement) all silently no-op because
`forward_v2` binds its device buffers once at load.

## How NxDI shares one KV cache across CTE and TKG

Three pieces, all in the installed package:

**1. Both graphs are registered in one builder**
(`models/application_base.py:174`, tags from `models/model_wrapper.py:38-39`):

```python
for model in self.models:
    self._builder.add(key=model.tag, model_instance=..., example_inputs=...)
```

`model_base.py:3107` / `:3130` build those two entries from one config, differing
only in what makes prefill prefill and decode decode:

```python
enable_context_encoding():   n_active_tokens = max_context_length,  is_prefill_stage = True
enable_token_generation():   n_active_tokens = 1,                   is_prefill_stage = False
```

Same weights, same cache, same `_model_cls` — two shapes of the same model.

**2. The state is allocated once for the whole collection**
(`trace/model_builder.py:1002`):

```python
def build_state_initializer(self):
    source_model_key = list(self.model_collection.keys())[0]   # ANY metaneff
    ...
    for tensor in metaneff.input_tensors:
        if tensor.type is INPUT_STATE:
            shapes[checkpoint_key] = list(tensor.shape)
```

It takes **any one** model's metaneff and keys the state by `checkpoint_key`.
Both graphs declare the same KV cache under the same key, so they resolve to the
same device buffer. `NxDModel.to_neuron` then does it once for everyone
(`nxd_model.py:323`):

```python
self.states = self.state_initializer()
for model in ...:
    model.initialize(self.states, self.weights, self.start_rank)
```

That is the hand-off: not a copy between graphs, but *one allocation both graphs
were compiled against*. Which is why it cannot be bolted onto two artifacts that
were traced separately — the sharing has to happen at build time.

**3. Dispatch picks a graph per call** (`nxd_model.py:451`, `:460`)

`router()` selects by input shape, so a 128-wide input runs CTE and a 1-wide
input runs TKG, against the cache they share. `forward(model_name=...)`
overrides when shapes are ambiguous.

## KV cache layout, for reference

`modules/kvcache/kv_cache_manager.py:106`. Same `nn.ParameterList` of zeros this
port uses, so the port's instinct was right; what differs is ownership.

```python
self.past_key_values = nn.ParameterList([
    nn.Parameter(torch.zeros(k_or_v_shape, dtype=self.cache_dtype), requires_grad=False)
    for _ in range(num_layer) for k_or_v_shape in [self.k_shape, self.v_shape]
])
```

Shape `(batch, num_kv_head_per_rank, max_len, head_dim)`, one K and one V per
layer, with `get_cache`/`update_cache` vending reads and writes. Variants exist
for paged attention (`block_kv_cache_manager.py`) and chunked/sliding-window
attention, which is the closest analogue to this model's window + compressed
tail — worth reading before designing the port's version.

There is also `models/deepseek/modeling_deepseek.py` in-tree. It is V3-family
MLA, not V4's compressed-sparse + indexer attention, so its *cache layout* does
not transfer. Its integration with `NxDModelForCausalLM` does.

## What this means for the port

The port's own attention is not the problem — it is validated at full depth in
both modes. The problem is that `parallel_model_trace` gives one graph per
artifact with no shared state and no host buffer access.

Two ways forward:

**A. Port to `ModelBuilder`.** Register prefill and decode as two keys, let
`build_state_initializer` give them one cache. This is what NxDI does and it
gets joint inference, the cache reset, and bucketing for free. Cost: the port
currently relies on `parallel_model_trace`'s factory + hand-rolled
`shard_loader`, and `ModelBuilder` wants a `BaseModelInstance` plus a
`checkpoint_loader` — but note NxD's sharding still will not recognise HF's
same-named parallel classes (see `shard_loader.py`), so the custom loader has to
be carried over rather than replaced.

**B. One graph, two shapes.** Keep `parallel_model_trace` and trace a single
graph that takes `n_active_tokens` as a bucket dimension. Smaller change, but it
duplicates what `ModelBuilder` already solves and gives up the host buffer API,
so the cache reset stays unsolved.

A is the right answer. It is a substantial refactor of `compile_neuron.py` and
was not attempted here — this document exists so that work starts from the
mechanism rather than rediscovering it.

## Measured baseline to beat

| | TTFT (128-token prompt) | per position |
|---|---|---|
| prefill graph, one call | 0.85 s | 6.6 ms |
| decode graph, 128 serial calls | 12.0 s | 93.7 ms |
| joint (projected) | ~0.85 s + 94 ms | — |
