#!/usr/bin/env python
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bisect which op in the decode graph produces the size-0 tensor.

The failure (`size of tensor a (128) must match tensor b (0)`) is reported at
`window_topk_idxs`'s `j <= pos`, but every operand there is provably correct and
the function works standalone on XLA. XLA is lazy, so the error surfaces at the
first sync point rather than where the bad shape was built.

This walks the decode path op group by op group under the *same*
`generate_hlo(cpu_backend=True)` call ModelBuilder uses, materialising after each
stage. The first stage whose materialisation raises is the one that built the bad
shape.

Runs at TP=1, 1 layer, on CPU-backend HLO generation only — no device, no
compile, seconds per iteration instead of ~3 minutes.

Run:
    python test/spike/bisect_decode_hlo.py
"""

import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
_SRC = os.path.join(_ROOT, "src")
for p in (_SRC, _ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

import paths  # noqa: E402

paths.add_hf_inference_to_syspath()

WIN = 128
SEQ = 256


def _hf():
    import compile_neuron
    compile_neuron._wire_shims()
    import model as hf
    compile_neuron.apply_xla_patches(hf)
    import collectives
    collectives.patch_hf_dist(hf)
    return hf, compile_neuron


def stage(name, fn):
    """Generate HLO for `fn` alone and report whether it survives."""
    from torch_neuronx.xla_impl.trace import generate_hlo

    class M(torch.nn.Module):
        def forward(self, ids, pos):
            return fn(ids, pos)

    ex = (torch.zeros(1, 1, dtype=torch.long),
          torch.ones((), dtype=torch.int32))
    try:
        generate_hlo(M(), ex, {}, inline_weights_to_neff=False,
                     return_weights=False, output_aliased_tensor=False,
                     cpu_backend=True, preserve_parameters=False)
        print(f"  [ok  ] {name}")
        return True
    except Exception as e:
        msg = str(e).split("\n")[0][:110]
        print(f"  [FAIL] {name}: {type(e).__name__}: {msg}")
        return False


def main():
    import decode_patches as dp

    print("=== stage 1: the index helpers in isolation ===")
    stage("window_topk_idxs(pos, WIN, 1)",
          lambda ids, pos: dp.window_topk_idxs(pos, WIN, 1))
    stage("compress_topk_idxs(pos, 4, 64, WIN, 1)",
          lambda ids, pos: dp.compress_topk_idxs(pos, 4, 64, WIN, 1))
    stage("cat(window, compress)",
          lambda ids, pos: torch.cat(
              [dp.window_topk_idxs(pos, WIN, 1),
               dp.compress_topk_idxs(pos, 4, 64, WIN, 1)], dim=-1))

    print("\n=== stage 2: _slot / index_copy, the other pos-derived ops ===")
    stage("_slot(pos, WIN)", lambda ids, pos: dp._slot(pos, WIN))
    stage("cache.index_copy via _slot",
          lambda ids, pos: torch.zeros(1, WIN, 8, device=ids.device).index_copy(
              1, dp._slot(pos, WIN),
              torch.ones(1, 1, 8, device=ids.device)))

    print("\n=== stage 3: the embedding, which is where the graph starts ===")
    hf, cn = _hf()
    stage("nothing but ids passthrough", lambda ids, pos: ids + 0)

    print("\n=== stage 4: full one-layer decode Attention ===")
    # Build a 1-layer model and run only its attention, so the failure (if any)
    # is attributable to the decode attention body rather than the whole stack.
    import json
    import prefill_patches  # noqa: F401  (kept symmetric with compile_joint)
    with open(paths.config_json()) as f:
        cfg = json.load(f)
    cfg.update(dtype="bf16", expert_dtype=None, n_mtp_layers=0,
               max_batch_size=1, max_seq_len=SEQ, n_layers=1)
    cfg["compress_ratios"] = cfg["compress_ratios"][:1]
    torch.set_default_dtype(torch.bfloat16)
    m = hf.Transformer(hf.ModelArgs(**cfg)).eval()
    torch.set_default_dtype(torch.float32)
    with torch.no_grad():
        for n, p in m.named_parameters():
            if "norm" in n or n.endswith("attn_sink"):
                p.fill_(1.0)
            elif "ape" in n:
                p.zero_()
            else:
                p.copy_((torch.randn(p.shape) * 0.02).to(p.dtype))

    state = dp.apply_decode_patches(hf)
    slots, _ = dp.collect_state_aliases(m, n_real_outputs=1)
    layer = m.layers[0]

    moved = {"done": False}

    def one_layer(ids, pos):
        if not moved["done"]:
            # generate_hlo traces on an XLA device; the module was built on CPU,
            # so every weight and state buffer has to come along or the first op
            # mixing them raises "Expected XLA tensor".
            m.to(ids.device)
            moved["done"] = True
        sink = dp.build_sink(m, hf)
        state["sink"] = sink
        state["pos"] = pos.reshape(())
        cn.set_index_helper_device(ids.device)
        x = m.embed(ids)
        return layer.attn(layer.attn_norm(x), pos.reshape(()))

    stage("layer0 decode attention (ratio=0)", one_layer)

    print("\n=== stage 5: the compressor/indexer layers ===")
    # compress_ratios = [0, 0, 4, 128, 4, ...]: layer 2 is the first with a
    # compressor AND an indexer, layer 3 the first ratio-128 (compressor only).
    # Layer 0 passing means the bug is almost certainly in one of these.
    import json as _json
    with open(paths.config_json()) as f:
        cfg5 = _json.load(f)
    cfg5.update(dtype="bf16", expert_dtype=None, n_mtp_layers=0,
                max_batch_size=1, max_seq_len=SEQ, n_layers=5)
    cfg5["compress_ratios"] = cfg5["compress_ratios"][:5]
    torch.set_default_dtype(torch.bfloat16)
    m5 = hf.Transformer(hf.ModelArgs(**cfg5)).eval()
    torch.set_default_dtype(torch.float32)
    with torch.no_grad():
        for n, p in m5.named_parameters():
            if "norm" in n or n.endswith("attn_sink"):
                p.fill_(1.0)
            elif "ape" in n:
                p.zero_()
            else:
                p.copy_((torch.randn(p.shape) * 0.02).to(p.dtype))
    dp.collect_state_aliases(m5, n_real_outputs=1)

    moved5 = {"done": False}

    def layer_n(idx):
        def run(ids, pos):
            if not moved5["done"]:
                m5.to(ids.device)
                moved5["done"] = True
            sink = dp.build_sink(m5, hf)
            state["sink"] = sink
            state["pos"] = pos.reshape(())
            cn.set_index_helper_device(ids.device)
            x = m5.embed(ids)
            lay = m5.layers[idx]
            return lay.attn(lay.attn_norm(x), pos.reshape(()))
        return run

    ratios = cfg5["compress_ratios"]
    for idx in range(5):
        lay = m5.layers[idx]
        has_idx = getattr(lay.attn, "indexer", None) is not None
        stage(f"layer{idx} attention (ratio={ratios[idx]}, "
              f"indexer={has_idx})", layer_n(idx))

    print("\n=== stage 6: the full Transformer.forward (what compile_joint runs) ===")
    # Layers all pass individually, so the difference has to be what the stack
    # adds around them: the ParallelHead, the MoE, or the wrapper's state
    # readback through the sink.
    raw5 = getattr(hf.Transformer.forward, "__wrapped__",
                   hf.Transformer.forward)

    def full_stack(ids, pos):
        if not moved5["done"]:
            m5.to(ids.device)
            moved5["done"] = True
        sink = dp.build_sink(m5, hf)
        state["sink"] = sink
        state["pos"] = pos.reshape(())
        cn.set_index_helper_device(ids.device)
        return raw5(m5, ids, pos.reshape(()))

    stage("full Transformer.forward, 5 layers", full_stack)

    def full_stack_with_state(ids, pos):
        if not moved5["done"]:
            m5.to(ids.device)
            moved5["done"] = True
        sink = dp.build_sink(m5, hf)
        state["sink"] = sink
        state["pos"] = pos.reshape(())
        cn.set_index_helper_device(ids.device)
        logits = raw5(m5, ids, pos.reshape(()))
        # This is the part compile_joint adds and nothing else has exercised:
        # reading every state slot back out of the sink as extra graph outputs.
        slots5, _ = dp.collect_state_aliases(m5, n_real_outputs=1)
        outs = [logits]
        for mod, name in slots5:
            outs.append(sink.get(mod, name))
        return tuple(outs)

    stage("full forward + state readback (compile_joint's wrapper)",
          full_stack_with_state)

    print("\nInterpretation: the first FAIL above is the stage that builds the "
          "size-0 tensor.\nAll-ok means the bad shape comes from something "
          "outside these paths (the\nembedding, MoE, or the wrapper's state "
          "readback) and the next bisection\nshould target those.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
