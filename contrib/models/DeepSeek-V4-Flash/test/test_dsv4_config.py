#!/usr/bin/env python
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Gate for design step 1: config mapping and the ragged per-layer state layout.

Checks the two things that would silently corrupt everything built on top:

  1. The ModelArgs -> HF name mapping is right, in particular that `head_dim`
     stays 512 rather than falling back to hidden_size // n_heads (= 64). An 8x
     wrong head_dim would produce plausible shapes and wrong results.
  2. Every layer's declared state shapes match what HF's own modules allocate,
     derived from the real checkpoint config rather than from my reading of it.
     This is the ragged-state layout the alias map will be built from, so if it
     disagrees with HF the cache is wrong before a single op runs.

CPU only, no device, seconds to run.

    python test/test_dsv4_config.py
"""

import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_SRC = os.path.join(_ROOT, "src")
for p in (_SRC, _ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

import paths  # noqa: E402

failures = []


def check(cond, label, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {label}" + (f"  {detail}" if detail else ""))
    if not cond:
        failures.append(label)
    return cond


def main():
    from neuronx_distributed_inference.models.config import MoENeuronConfig
    import modeling_dsv4 as md

    SEQ = 256
    N_LAYERS = 5  # covers every distinct layer type (see README)

    nc = MoENeuronConfig(tp_degree=32, batch_size=1, seq_len=SEQ,
                         torch_dtype=torch.bfloat16)
    cfg = md.DSV4InferenceConfig.from_checkpoint(
        paths.config_json(), nc, n_layers=N_LAYERS)

    print("=== 1. ModelArgs -> HF name mapping ===")
    check(cfg.hidden_size == 4096, "hidden_size <- dim", f"{cfg.hidden_size}")
    check(cfg.num_attention_heads == 64, "num_attention_heads <- n_heads",
          f"{cfg.num_attention_heads}")
    check(cfg.num_hidden_layers == N_LAYERS, "num_hidden_layers honoured",
          f"{cfg.num_hidden_layers}")
    # The trap: the generic fallback would give 64, which is 8x too small.
    check(cfg.head_dim == 512,
          "head_dim stays 512, NOT hidden_size // n_heads",
          f"head_dim={cfg.head_dim}, fallback would be "
          f"{cfg.hidden_size // cfg.num_attention_heads}")
    check(cfg.compress_ratios == [0, 0, 4, 128, 4],
          "compress_ratios truncated to n_layers",
          f"{cfg.compress_ratios}")
    check(cfg.n_indexer_layers == 2 and cfg.n_compress_layers == 3,
          "layer-type census", f"{cfg.n_compress_layers} compress, "
                               f"{cfg.n_indexer_layers} indexer")

    print("\n=== 2. per-layer state shapes vs what HF allocates ===")
    # Build the real HF modules and compare shapes, rather than trusting my
    # reading of model.py.
    import compile_neuron
    compile_neuron._wire_shims()
    import model as hf
    import json
    with open(paths.config_json()) as f:
        raw = json.load(f)
    raw.update(dtype="bf16", expert_dtype=None, n_mtp_layers=0,
               max_batch_size=1, max_seq_len=SEQ, n_layers=N_LAYERS)
    raw["compress_ratios"] = raw["compress_ratios"][:N_LAYERS]
    torch.set_default_dtype(torch.bfloat16)
    hf_model = hf.Transformer(hf.ModelArgs(**raw)).eval()
    torch.set_default_dtype(torch.float32)

    kinds = md.build_layer_kinds(cfg)
    check(len(kinds) == N_LAYERS, "one LayerKind per layer")

    n_states_total = 0
    for k in kinds:
        attn = hf_model.layers[k.layer_idx].attn
        specs = dict((n, s) for n, s, _ in k.state_specs())
        n_states_total += len(specs)

        # attention cache
        want = tuple(attn.kv_cache.shape[1:])
        got = specs["kv_cache"]
        if not check(got == want, f"layer{k.layer_idx} kv_cache "
                                  f"(ratio={k.ratio})", f"{got} vs HF {want}"):
            continue

        # presence of compressor / indexer must agree with HF
        hf_has_comp = getattr(attn, "compressor", None) is not None
        hf_has_idx = getattr(attn, "indexer", None) is not None
        check(k.has_compressor == hf_has_comp,
              f"layer{k.layer_idx} compressor presence",
              f"{k.has_compressor} vs HF {hf_has_comp}")
        check(k.has_indexer == hf_has_idx,
              f"layer{k.layer_idx} indexer presence",
              f"{k.has_indexer} vs HF {hf_has_idx}")

        if hf_has_comp:
            for nm, buf in (("compressor.kv_state", attn.compressor.kv_state),
                            ("compressor.score_state",
                             attn.compressor.score_state)):
                check(specs[nm] == tuple(buf.shape[1:]),
                      f"layer{k.layer_idx} {nm}",
                      f"{specs[nm]} vs HF {tuple(buf.shape[1:])}")
        if hf_has_idx:
            check(specs["indexer.kv_cache"] == tuple(attn.indexer.kv_cache.shape[1:]),
                  f"layer{k.layer_idx} indexer.kv_cache",
                  f"{specs['indexer.kv_cache']} vs HF "
                  f"{tuple(attn.indexer.kv_cache.shape[1:])}")
            for nm, buf in (("indexer.compressor.kv_state",
                             attn.indexer.compressor.kv_state),
                            ("indexer.compressor.score_state",
                             attn.indexer.compressor.score_state)):
                check(specs[nm] == tuple(buf.shape[1:]),
                      f"layer{k.layer_idx} {nm}",
                      f"{specs[nm]} vs HF {tuple(buf.shape[1:])}")

    print("\n=== 3. total state count matches the validated port ===")
    # decode_patches.collect_state_aliases found 17 states at 5 layers; the new
    # layout must publish exactly the same set or the alias map differs.
    import decode_patches
    slots, _ = decode_patches.collect_state_aliases(hf_model, n_real_outputs=1)
    check(n_states_total == len(slots),
          "state count equals what collect_state_aliases finds",
          f"{n_states_total} declared vs {len(slots)} promoted")

    print("\n" + "=" * 62)
    if failures:
        print(f"FAILED {len(failures)} check(s):")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
