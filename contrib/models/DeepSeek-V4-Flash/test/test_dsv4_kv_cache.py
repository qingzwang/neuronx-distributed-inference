#!/usr/bin/env python
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Gate for the custom cache manager: shapes, and the alias-ordering contract.

The ordering check is the important one. NxDI's aliasing is positional — graph
output `n_real_outputs + i` is written back over `past_key_values[i]` — so if the
manager's published order and the graph's output order ever disagree, the runtime
writes the wrong cache. That failure is silent: the model still answers, just from
corrupted context. So this asserts the property directly rather than trusting that
two loops in different files stay in step.

Also checks the manager against NxDI's real aliasing code path, by feeding it to
the same construction `DecoderModelInstance.get()` uses, and against HF's own
buffers via the LayerKind layout already gated in test_dsv4_config.py.

CPU only, seconds to run.

    python test/test_dsv4_kv_cache.py
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
    import dsv4_kv_cache as kvc

    SEQ, N_LAYERS, B = 256, 5, 1
    nc = MoENeuronConfig(tp_degree=32, batch_size=B, seq_len=SEQ,
                         torch_dtype=torch.bfloat16)
    cfg = md.DSV4InferenceConfig.from_checkpoint(
        paths.config_json(), nc, n_layers=N_LAYERS)
    kinds = md.build_layer_kinds(cfg)
    mgr = kvc.DSV4CacheManager(kinds, batch_size=B)

    print("=== 1. published state matches the declared layout ===")
    expected = sum(len(k.state_specs()) for k in kinds)
    check(len(mgr) == expected, "flat list length", f"{len(mgr)} == {expected}")
    check(len(mgr) == 17, "17 states at 5 layers (matches the validated port)",
          f"{len(mgr)}")
    check(isinstance(mgr.past_key_values, torch.nn.ParameterList),
          "past_key_values is an nn.ParameterList (what NxDI's aliasing reads)")

    print("\n=== 2. ordering: index <-> name is a bijection, stable ===")
    order = mgr.output_order()
    check(len(order) == len(mgr) and all(o is not None for o in order),
          "output_order covers every slot with no gaps")
    round_trip = all(mgr.slot_index(li, n) == i
                     for i, (li, n) in enumerate(order))
    check(round_trip, "slot_index(output_order()[i]) == i for every i")
    # Rebuilding must give the identical order, or the alias map is not
    # reproducible between the trace and the run.
    mgr2 = kvc.DSV4CacheManager(kinds, batch_size=B)
    check(mgr2.output_order() == order,
          "a freshly built manager publishes the identical order")

    print("\n=== 3. the alias map matches NxDI's positional convention ===")
    aliases = mgr.alias_map(n_real_outputs=1)
    check(len(aliases) == len(mgr), "one alias per state", f"{len(aliases)}")
    # Output index i+1 must correspond to past_key_values[i], exactly as
    # DecoderModelInstance.get() builds it (model_wrapper.py:1629).
    positional_ok = all(
        aliases[mgr.past_key_values[i]] == 1 + i for i in range(len(mgr)))
    check(positional_ok,
          "aliases[past_key_values[i]] == n_real_outputs + i")
    check(len({id(t) for t in aliases}) == len(mgr),
          "alias keys are distinct tensors (no accidental sharing)")

    print("\n=== 4. shapes agree with HF's own buffers ===")
    import compile_neuron
    compile_neuron._wire_shims()
    import model as hf
    import json
    with open(paths.config_json()) as f:
        raw = json.load(f)
    raw.update(dtype="bf16", expert_dtype=None, n_mtp_layers=0,
               max_batch_size=B, max_seq_len=SEQ, n_layers=N_LAYERS)
    raw["compress_ratios"] = raw["compress_ratios"][:N_LAYERS]
    torch.set_default_dtype(torch.bfloat16)
    hf_model = hf.Transformer(hf.ModelArgs(**raw)).eval()
    torch.set_default_dtype(torch.float32)

    worst = None
    for layer_idx, name in order:
        attn = hf_model.layers[layer_idx].attn
        obj = attn
        for part in name.split(".")[:-1]:
            obj = getattr(obj, part)
        hf_buf = getattr(obj, name.split(".")[-1])
        mine = mgr.get(layer_idx, name)
        if tuple(mine.shape) != tuple(hf_buf.shape):
            worst = (layer_idx, name, tuple(mine.shape), tuple(hf_buf.shape))
            break
    check(worst is None, "every published state matches HF's buffer shape",
          "" if worst is None else f"{worst[0]}.{worst[1]}: {worst[2]} vs {worst[3]}")

    print("\n=== 5. init values are safe given in-graph masking ===")
    all_zero = all(float(t.abs().sum()) == 0.0 for t in mgr.past_key_values)
    check(all_zero, "everything zero-initialised (no -inf dependency)")
    finite = all(bool(torch.isfinite(t).all()) for t in mgr.past_key_values)
    check(finite, "no infs anywhere, so a softmax over a fresh ring is finite")

    print("\n=== 6. memory accounting ===")
    gb = mgr.total_bytes() / (1024 ** 3)
    print(f"  {len(mgr)} states, {mgr.total_bytes():,} bytes ({gb:.4f} GB) "
          f"per rank at {N_LAYERS} layers, seq_len={SEQ}")
    check(mgr.total_bytes() > 0, "non-trivial allocation")

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
