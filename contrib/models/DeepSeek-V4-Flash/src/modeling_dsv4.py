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
        """[(name, shape, init)] in the fixed order this layer publishes.

        `init` is 0.0 everywhere: the compression ring no longer depends on an
        -inf fill because `compress_state` masks unwritten slots inside the graph
        (design doc step 2, gated by test_compress_state_mask.py). Recording it
        explicitly anyway so a future reader does not have to infer that the
        omission was deliberate.
        """
        specs = [("kv_cache", (self.attn_cache_len, self.head_dim), 0.0)]
        if self.has_compressor:
            ring = self.coff * self.ratio
            width = self.coff * self.head_dim
            specs += [
                ("compressor.kv_state", (ring, width), 0.0),
                ("compressor.score_state", (ring, width), 0.0),
            ]
        if self.has_indexer:
            specs += [
                ("indexer.kv_cache", (self.n_compressed, self.index_head_dim),
                 0.0),
                ("indexer.compressor.kv_state",
                 (self.coff * self.ratio, self.coff * self.index_head_dim), 0.0),
                ("indexer.compressor.score_state",
                 (self.coff * self.ratio, self.coff * self.index_head_dim), 0.0),
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
