# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""DeepSeek-V4-Flash in NxDI's idiom, keeping the compressed-sparse attention.

See NXDI_PORT_DESIGN.md for why this exists and what it must preserve. In short:
the HF-patching port works but cannot share a KV cache between its prefill and
decode graphs, which costs 14x on TTFT; driving `ModelBuilder` by hand to fix that
means reimplementing what `NeuronBaseForCausalLM` already does. So the model moves
into the framework's idiom — but unlike GLM-5.2, which disables its sparse indexer
on the grounds that it is a no-op at short sequence lengths, this keeps the
indexer, the sliding window and the compressed tail. For this model the indexer is
only a no-op at seq_len <= 2048, and a 1M-token context is the point of the
architecture.

Build order (each step gated on CPU numerics before the next — see the design doc):
  1. config + attention for ratio-0 layers        <- this file, so far
  2. the compression ring, init-independent        <- compress_state.py, done
  3. compressor + indexer layers
  4. MoE + head + the full model class
  5. NeuronBaseForCausalLM subclass, compile at TP=32

Nothing here touches a device until host numerics match the validated patched-HF
forward bit for bit. That ordering is deliberate: the previous attempt compiled
first and spent a long time debugging a graph whose numerics had never been
checked.
"""

import json
import os
from typing import List, Optional, Tuple, Type

import torch
from torch import nn

from neuronx_distributed_inference.models.config import InferenceConfig, MoENeuronConfig


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

class DSV4InferenceConfig(InferenceConfig):
    """Maps the checkpoint's `inference/config.json` onto NxDI's expectations.

    The checkpoint ships a *ModelArgs*-flavoured config (`dim`, `n_heads`,
    `n_layers`, ...), not an HF-flavoured one (`hidden_size`,
    `num_attention_heads`, ...). NxDI reads the HF names throughout, so the
    mapping happens in `add_derived_config`.

    Ordering trap, learned from GLM-5.2's header: `InferenceConfig.__init__` runs
    load_config -> add_derived_config -> validate_config and *then* returns, so
    anything validation depends on must be set in `add_derived_config`, not in
    this class's `__init__` body.
    """

    @classmethod
    def get_neuron_config_cls(cls) -> Type[MoENeuronConfig]:
        return MoENeuronConfig

    def add_derived_config(self):
        # --- ModelArgs -> HF names that NxDI reads ---
        self.hidden_size = self.dim
        self.num_attention_heads = self.n_heads
        self.num_hidden_layers = self.n_layers
        self.num_key_value_heads = self.n_heads
        self.rms_norm_eps = getattr(self, "norm_eps", 1e-6)
        self.pad_token_id = getattr(self, "pad_token_id", 0)

        # `head_dim` already means the right thing in this checkpoint (512), and
        # it is NOT hidden_size // n_heads (4096 // 64 = 64). NxDI's
        # `_get_hidden_dim_per_head` prefers an explicit head_dim, so leaving it
        # alone is correct — but it is worth stating, because the fallback would
        # silently be 8x too small.
        assert self.head_dim == 512, (
            f"expected head_dim 512 from the checkpoint, got {self.head_dim}; "
            f"hidden_size // n_heads would be "
            f"{self.hidden_size // self.num_attention_heads}, which is not the "
            f"same thing for this model"
        )

        # --- sparse-attention geometry, kept (not disabled as GLM-5.2 does) ---
        # compress_ratios is per layer: [0, 0, 4, 128, 4, 128, ...]. A ratio of 0
        # means pure sliding-window attention with no compressor and no indexer.
        self.compress_ratios = list(self.compress_ratios[: self.num_hidden_layers])
        self.n_compress_layers = sum(1 for r in self.compress_ratios if r)
        self.n_indexer_layers = sum(1 for r in self.compress_ratios if r == 4)

    def get_required_attributes(self) -> List[str]:
        return [
            "hidden_size",
            "num_attention_heads",
            "num_hidden_layers",
            "head_dim",
            "window_size",
            "compress_ratios",
            "index_topk",
        ]

    @classmethod
    def from_checkpoint(cls, config_path, neuron_config, n_layers=None,
                        **overrides):
        """Build from the checkpoint's inference/config.json."""
        with open(config_path) as f:
            raw = json.load(f)
        raw.update(overrides)
        if n_layers is not None:
            raw["n_layers"] = n_layers
            if len(raw.get("compress_ratios", [])) > n_layers:
                raw["compress_ratios"] = raw["compress_ratios"][:n_layers]

        def load_config(self):
            for k, v in raw.items():
                setattr(self, k, v)

        return cls(neuron_config=neuron_config, load_config=load_config)


# ---------------------------------------------------------------------------
# Layer-type description
# ---------------------------------------------------------------------------

class LayerKind:
    """Which state a layer owns, derived from its compress_ratio.

    This is the ragged-state problem from the design doc made explicit. The three
    layer shapes in this model are:

        ratio == 0    window only.        1 state  (kv_cache)
        ratio == 128  window + compress.  3 states (kv_cache, kv_state, score_state)
        ratio == 4    the above + indexer. 6 states (+ indexer kv_cache,
                                            indexer kv_state, score_state)

    The cache manager publishes states in a fixed order per layer so the alias map
    NxDI builds by enumeration stays stable. Keeping that ordering in one place —
    here — is what makes it auditable rather than implicit in a loop somewhere.
    """

    def __init__(self, layer_idx, ratio, window_size, max_seq_len, head_dim,
                 index_head_dim, overlap_head_dim_mult=2):
        self.layer_idx = layer_idx
        self.ratio = ratio
        self.has_compressor = ratio > 0
        self.has_indexer = ratio == 4
        self.window_size = window_size
        self.max_seq_len = max_seq_len
        self.head_dim = head_dim
        self.index_head_dim = index_head_dim
        # HF: overlap is enabled exactly when ratio == 4, and doubles both the
        # ring length and the feature width (Compressor.__init__: coff = 1 + overlap).
        self.overlap = ratio == 4
        self.coff = 2 if self.overlap else 1

    @property
    def n_compressed(self):
        """Compressed KV entries this layer can hold."""
        return (self.max_seq_len // self.ratio) if self.has_compressor else 0

    @property
    def attn_cache_len(self):
        """HF: window_size + (max_seq_len // ratio if ratio else 0)."""
        return self.window_size + self.n_compressed

    def state_specs(self):
        """[(name, shape, init, dtype)] in the fixed order this layer publishes.

        `init` is 0.0 everywhere: the compression ring no longer depends on an
        -inf fill because `compress_state` masks unwritten slots inside the graph
        (design doc step 2, gated by test_compress_state_mask.py). Recording it
        explicitly anyway so a future reader does not have to infer that the
        omission was deliberate.

        `dtype` is per state and NOT uniform. HF allocates the KV caches in the
        model dtype (bf16) but the compressor rings in **float32**, because the
        compression itself runs in fp32 (`Compressor.forward` does `x = x.float()`
        and the ring buffers are `dtype=torch.float32` at model.py:303-304).
        Creating everything bf16 fails at the first ring write:

            RuntimeError: index_copy_(): self and source expected to have the
            same dtype, but got (self) BFloat16 and (source) Float

        which is a useful error; the dangerous version would have been a silent
        downcast of the accumulator.
        """
        f32 = torch.float32
        specs = [("kv_cache", (self.attn_cache_len, self.head_dim), 0.0, None)]
        if self.has_compressor:
            ring = self.coff * self.ratio
            width = self.coff * self.head_dim
            specs += [
                ("compressor.kv_state", (ring, width), 0.0, f32),
                ("compressor.score_state", (ring, width), 0.0, f32),
            ]
        if self.has_indexer:
            iring = self.coff * self.ratio
            iwidth = self.coff * self.index_head_dim
            specs += [
                ("indexer.kv_cache", (self.n_compressed, self.index_head_dim),
                 0.0, None),
                ("indexer.compressor.kv_state", (iring, iwidth), 0.0, f32),
                ("indexer.compressor.score_state", (iring, iwidth), 0.0, f32),
            ]
        return specs


def build_layer_kinds(config) -> List[LayerKind]:
    """One LayerKind per layer, in layer order."""
    return [
        LayerKind(
            layer_idx=i,
            ratio=config.compress_ratios[i],
            window_size=config.window_size,
            max_seq_len=config.neuron_config.seq_len,
            head_dim=config.head_dim,
            index_head_dim=config.index_head_dim,
        )
        for i in range(config.num_hidden_layers)
    ]


# ---------------------------------------------------------------------------
# The model class
# ---------------------------------------------------------------------------

class DSV4Model(torch.nn.Module):
    """The HF Transformer, wired to a framework-owned cache.

    Deliberately *not* a `NeuronBaseModel` subclass, and the reason is worth
    stating because the design doc assumed it would be.

    `NeuronBaseModel.__init__` requires `init_model` to publish
    `embed_tokens`/`layers`/`norm`/`lm_head`, and its `forward` then drives them
    through `NeuronAttentionBase`-shaped layers with a `KVCacheManager`. Neither
    fits here:

      * HF names them `embed` and `head`, and `head` is not a linear — it takes
        five arguments (`x, hc_head_fn, hc_head_scale, hc_head_base, norm`)
        because of the hyper-connection mixing, and returns only the last
        position.
      * `Attention.forward(x, start_pos)` bears no resemblance to
        `NeuronAttentionBase.forward`'s signature, and its body is a custom
        `sparse_attn` kernel over a window + compressed tail.

    Subclassing would mean satisfying a contract by renaming things and then
    overriding every method that uses them, which buys nothing. What the framework
    is actually needed for is the *tracing* side — CTE+TKG registered against one
    shared cache, alias generation, NEFF loading — and that is driven by
    `BaseModelInstance` + the alias map, not by inheriting `NeuronBaseModel`.

    So this holds the HF model, owns a `DSV4CacheManager`, and exposes the two
    things the tracing side needs: `past_key_values` (where the aliasing code
    looks) and a forward returning `(logits, *state)` in the manager's order.
    """

    def __init__(self, config, hf_mod, hf_model, mode):
        super().__init__()
        import dsv4_kv_cache
        import dsv4_state_adapter

        self.config = config
        self.mode = mode
        self._hf = hf_mod
        self.inner = hf_model
        self.layer_kinds = build_layer_kinds(config)

        self.kv_mgr = dsv4_kv_cache.DSV4CacheManager(
            self.layer_kinds,
            batch_size=config.neuron_config.batch_size,
            dtype=config.neuron_config.torch_dtype,
        )
        # The attribute NxD's aliasing code reads. Exposed at this level too so a
        # BaseModelInstance can find it without knowing about kv_mgr.
        self.past_key_values = self.kv_mgr.past_key_values

        self._state_map = dsv4_state_adapter.build_state_map(
            hf_model, hf_mod, self.layer_kinds)

        self._raw_forward = getattr(hf_mod.Transformer.forward, "__wrapped__",
                                    hf_mod.Transformer.forward)

    def alias_map(self, n_real_outputs=1):
        return self.kv_mgr.alias_map(n_real_outputs)

    def forward(self, input_ids, position_ids):
        """(logits, *state) with state in `kv_mgr.output_order()`.

        `position_ids` is per-token `(batch, n_active)`, matching NxDI's own
        convention (model_wrapper.py:248-251). A scalar position cannot be used:
        prefill's forwards hardcode position 0, so a scalar is a dead input in
        prefill's graph, NxD eliminates unused inputs, and because both graphs
        share one input signature decode then receives a zero-element tensor.
        """
        import compile_neuron
        import dsv4_state_adapter

        with torch.no_grad():
            sink = dsv4_state_adapter.build_managed_sink(
                self.inner, self._hf, self.layer_kinds, self.kv_mgr)
            patch_state = self._hf._dsv4_patch_state
            patch_state["sink"] = sink
            patch_state["active"] = self.mode
            # The scalar position each mode needs is NOT the same entry.
            #
            # decode processes one token at absolute position p, so it wants the
            # last (only) entry. prefill processes the whole prompt starting AT
            # ZERO -- its forwards slice freqs_cis[0:seqlen] and derive every
            # compressor/window index from a start of 0. Feeding it
            # position_ids[0, -1] = 127 instead made prefill run as if the prompt
            # began at 127: freqs_cis got the wrong slice and the compressor's
            # freq_idx = (pos + 1 - ratio) was wrong for every entry, which came
            # out as NaN logits on CPU and all-zero logits on device (the device
            # reports non-finite as zero).
            #
            # This was the actual cause of the "graph does not execute" symptom.
            # The graph executed the whole time; it was computing garbage.
            if self.mode == "prefill":
                pos = torch.zeros((), dtype=torch.int32,
                                  device=position_ids.device)
            else:
                pos = position_ids[0, -1].reshape(())
            patch_state["pos"] = pos

            # The index helpers build their position constants on a device read
            # from input_ids by the patched Transformer.forward, which is bypassed
            # here (the raw function is needed to avoid a wrapper loop under
            # inspect.unwrap). Set it explicitly or they emit CPU tensors.
            compile_neuron.set_index_helper_device(input_ids.device)

            logits = self._raw_forward(self.inner, input_ids, pos)

            if self.mode == "prefill":
                # Keep position_ids alive in prefill's graph WITHOUT multiplying by
                # zero.
                #
                # The previous form was `logits + position_ids.sum() * 0`, which is
                # a mathematical no-op but is also exactly the pattern a compiler
                # constant-folds: `x * 0 -> 0`, then `logits + 0 -> logits`, and the
                # input dependency it was supposed to create disappears again. Worse,
                # on device the whole prefill output came back as exact zeros with
                # the aliased state correctly written -- i.e. the graph ran, and only
                # the logits output was dead.
                #
                # Instead make position_ids feed the result through a path that
                # cannot be folded away: subtract its own first element, which is
                # provably 0 for prefill (position_ids = arange(seqlen)) but is not a
                # literal the compiler can prove without evaluating the input.
                zero_from_pos = (position_ids[0, 0] - position_ids[0, 0]).to(
                    logits.dtype)
                logits = logits + zero_from_pos

            # Return logits TWICE, and alias the state from index 2 onward.
            #
            # Diagnostic for the dead output-0: the metaneff shows out[0] declared
            # [1, 129280] float32 and unaliased, outputs 1..17 aliased to the state
            # inputs, and on device the state comes back WRITTEN while out[0] is an
            # untouched buffer. If out[1] (the duplicate) comes back with real
            # values while out[0] stays zero, the problem is specific to position 0
            # of the root tuple. If both are zero, the logits value itself never
            # reaches the root.
            state_outs = dsv4_state_adapter.collect_outputs(
                sink, self.kv_mgr, self._state_map)

            # `logits` may be a view/slice of a tensor that also feeds an aliased
            # state output (ParallelHead.get_logits does F.linear(x[:, -1], w), and
            # `x` descends from the same activations the caches were written from).
            # An aliased output is written IN PLACE over its input buffer, so if
            # output 0 shares storage with any aliased buffer the runtime can
            # clobber it -- which matches the observation exactly: state written,
            # logits an untouched-looking zero buffer.
            #
            # `.contiguous()` alone is not enough: it is a no-op when the tensor is
            # already contiguous, which a linear's output is. Force a fresh
            # allocation the aliasing cannot reach.
            logits = logits.clone()
            return tuple([logits] + state_outs)
