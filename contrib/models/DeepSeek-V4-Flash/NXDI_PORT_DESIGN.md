# Porting DeepSeek-V4-Flash onto NxDI's model base, keeping sparse attention

Plan of record for the rewrite. Approach **B**: follow GLM-5.2's structure
(subclass NxDI's model base so the framework owns CTE+TKG registration, aliasing
and NEFF loading) but **keep the compressed-sparse attention**, which GLM-5.2
disables.

## Why rewrite at all

The current port patches HF's `inference/model.py` and drives
`parallel_model_trace` directly. That works and is validated: prefill 0.85 s,
decode 94 ms/token, GSM8K 97.3% at 43 layers. What it cannot do is share one KV
cache between the prefill and decode graphs, because `parallel_model_trace`
produces one graph per artifact with private aliased state. Decode therefore
re-ingests the prompt one token at a time — 12.0 s to first token against 0.85 s
for the prefill graph, a 14x penalty.

Driving `ModelBuilder` by hand to fix that produced five framework-integration
bugs (see compile_joint.py's header) and still ends in all-zero logits with no
NEFF loaded. The diagnosis: the mechanism is fine — the same `ModelBuilder` path
runs a toy shared-cache model correctly (9.0 shared vs 5.0 isolated) — but this
port is reimplementing what `NeuronBaseForCausalLM` already does. Models like
GLM-5.2 do not hit any of it because they let the framework do that work.

## What GLM-5.2 does, and where we must differ

GLM-5.2 rewrites the model in NxDI's idiom: `GLM5Attention(NeuronAttentionBase)`,
`NeuronGLM5Model(NeuronBaseModel)`, `NeuronGLM5ForCausalLM(NeuronBaseForCausalLM)`,
plus a `convert_hf_to_neuron_state_dict` static method. `examples/compile.py` then
just calls `model.compile(path)` and the framework registers context-encoding and
token-generation graphs against one shared cache.

It also turns its sparse indexer **off**:

```python
c.dsa_enabled = False   # examples/compile.py:67
```

with the justification that at `seq_len <= index_topk` the DSA indexer is
mathematically a no-op — top-2048 over ≤2048 keys is full attention.

**That justification does not carry over.** For this model:

| layers | condition for the indexer to be a no-op |
|---|---|
| 21 layers with `compress_ratio=4` | `seq_len <= index_topk * 4` = **2048** |
| 21 layers with `compress_ratio=128` | `seq_len <= 65536` |

The model ships a 1M-token context and its stated advantage is 27% of the
single-token FLOPs and 10% of the KV cache of DeepSeek-V3.2 at that length. Going
full-attention would reproduce a generic MLA model and discard the reason this
model exists. So: keep the indexer, keep the window + compressed-tail cache, and
absorb the cost of a custom cache manager.

## The three things the standard cache manager cannot express

`KVCacheManager` assumes one K and one V tensor per layer, shaped
`(batch, kv_heads, max_len, head_dim)`, all zero-initialised. This model needs:

**1. Three heterogeneous states per layer, not two.**

| state | shape | owner |
|---|---|---|
| `kv_cache` | `(B, window_size + max_seq_len // ratio, head_dim)` | Attention |
| `kv_state` / `score_state` | `(B, coff*ratio, coff*head_dim)` | Compressor |
| `kv_cache` | `(B, max_seq_len // ratio, head_dim)` | Indexer (own) |

Layer types differ: `compress_ratio` is `[0, 0, 4, 128, 4, 128, ...]`, layers with
ratio 0 have no compressor at all, and only ratio-4 layers carry an Indexer. So
the state list is *ragged across layers* — which the base class's
`layer_to_cache_size_mapping` partly anticipates but does not cover.

Feasible: `past_key_values` is a flat `nn.ParameterList`
(kv_cache_manager.py:151-164) and `DecoderModelInstance.get()` aliases it by
enumeration (model_wrapper.py:1629), with no pairing assumption. A subclass can
publish any number of states per layer as long as the order is stable.

**2. `score_state` must initialise to `-inf`, not zero.**

```python
register_buffer("score_state", torch.full(..., float("-inf")))   # model.py:304
```

It is consumed by `softmax(dim=1)` over the compression ring, so an unwritten
slot must contribute *zero weight*. Zero-initialising makes it contribute
`exp(0)` — a uniform vote for a slot holding no data. Measured effect of getting
this wrong: **2x magnitude error** on the compressed output (absmean 0.62 vs
0.31), silently, with finite values throughout.

This is the one constraint the framework actively fights: NxD's `StateInitializer`
hardcodes `torch.zeros` (base_nxd_model.py:31) and is what builds device state on
the ModelBuilder path. Options, in order of preference:

  a. Override `_init_kv_shape`/state construction in the subclass so the
     Parameters themselves are created `-inf`, and verify the device state after
     `initialize()` matches (the state initializer derives shape and dtype from
     the metaneff, so if it also re-zeroes, fall back to (b)).
  b. Fold the mask into the graph: keep the buffer zero-initialised but track a
     written-count and apply `where(slot < n_written, score, -inf)` before the
     softmax. Costs one compare per compression step, removes the dependency on
     init values entirely, and is the more robust choice if (a) proves fragile.

**Decision: implement (b).** It is strictly more robust — correctness stops
depending on whether any layer of the stack re-initialises a buffer — and the
cost is negligible against the masked MoE already dominating the graph.

**3. The compressor writes through a *view* into the layer's cache.**

```python
self.compressor.kv_cache = self.kv_cache[:, win:]    # HF, in Attention.forward
```

A functional `index_copy` returns a new tensor, so a write through the view is
invisible to the owner. `decode_patches.StateSink` already solves this with an
owner/offset redirect, and that logic transfers unchanged — it is independent of
which tracing API is used.

## What carries over

Everything numerically validated stays, which is most of the work:

* `xla_ops.py` — the sort-free top-k. Unchanged.
* `shard_loader.py` — must stay. NxD's `shard_children` still will not recognise
  HF's same-named parallel classes, so `convert_hf_to_neuron_state_dict` will
  wrap this rather than delegate to NxD sharding.
* `dequant_checkpoint.py` — FP4/FP8 → bf16. Unchanged.
* The XLA-safety rewrites in `compile_neuron.apply_xla_patches`: RoPE without
  complex tensors, `torch.where` embedding, static-shape MoE, the TP > o_groups
  O-projection split. These are properties of the ops, not of the tracing API.
* `prefill_patches` / `decode_patches` forward bodies — the index math, the
  closed-form window ring, the masked compressor writes. These become the bodies
  of the new attention module's context-encoding and token-generation paths.
* One more fix found during the ModelBuilder work and needed regardless:
  `torch.where` does not lower under `generate_hlo` in `window_topk_idxs`;
  the arithmetic form `keep * j - (1 - keep)` does, and is verified identical.

## Correctness plan — the part that matters

"Results must be right" is the requirement, so every stage gates on a numerical
comparison against something already trusted, not on "it compiles".

1. **Layer parity, CPU.** New attention module vs the validated patched HF
   forward, same weights, per layer type (ratio 0 / 4 / 128, indexer on/off).
   Gate: logits and all state buffers bit-identical, as
   `test_prefill_functional_vs_inplace.py` already does (it currently reports
   rel=0.00e+00 and is mutation-tested: a +1 ring offset makes it fail at
   max|d|=6.52 while logits still pass).
2. **`-inf` masking equivalence.** The (b) rewrite vs HF's `-inf` init, directly:
   compressed output must match to bf16 tolerance for a partially filled ring.
   This is the check that the 2x error cannot slip through.
3. **Full-depth device parity.** 43 layers, TP=32: the joint graph's prefill
   logits vs the existing standalone prefill artifact, which scored `' Paris'` at
   25.562. Gate: same top-1, logit within bf16 tolerance.
4. **Cache hand-off, on device.** Prefill then N decode steps through the shared
   cache vs decode-only ingest of the same prompt (which is already validated to
   match one-shot prefill). Gate: identical tokens. This is the claim the whole
   rewrite exists to establish, and it must be measured, not asserted.
5. **GSM8K regression.** Re-run the 150-problem set. Gate: 97.3% ± noise. Note
   the current number is measured under cache carry-over; with a working reset the
   comparison becomes cleaner, so a *change* here needs explaining either way.

## Order of work

1. `modeling_dsv4.py` skeleton: config + attention subclass, one layer type, no
   MoE. Gate on (1) for ratio-0 layers.
2. Custom cache manager with the three states and the (b) masking. Gate on (2).
3. Remaining layer types, compressor, indexer. Gate on (1) for all five.
4. MoE + head, full model class. Gate on (1) at 5 layers.
5. `NeuronBaseForCausalLM` subclass + `convert_hf_to_neuron_state_dict` wrapping
   `shard_loader`. Compile at TP=32, 5 layers. Gate on (3) at 5 layers.
6. Full depth, then (3), (4), (5).

Steps 1-4 are CPU-only and fast to iterate. Nothing touches the device until the
numerics already match on host, which is the opposite of how the ModelBuilder
attempt went.

## Risks, stated up front

* **Ragged per-layer state** is the least-tested corner of the framework's
  aliasing. If enumeration order turns out to matter in a way the base class does
  not guarantee, this needs a fixed slot layout with unused slots padded.
* **`sparse_attn` is a custom kernel** (`kernel_cpu.py`) that does not fit
  `NeuronAttentionBase`'s flash-attention assumptions. The attention subclass
  will likely have to override `forward` wholesale rather than reuse the base's
  attention core, which reduces how much the framework actually does for us.
* If both risks bite, the honest fallback is A (full attention, `seq_len <= 2048`)
  as a working baseline, with sparse attention as follow-up work. That would be a
  real reduction in capability and would be reported as such, not quietly.
