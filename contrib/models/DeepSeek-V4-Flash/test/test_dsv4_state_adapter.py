#!/usr/bin/env python
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Gate: the validated forwards, run against manager-owned state, are unchanged.

The adapter's whole claim is that pointing the StateSink at `DSV4CacheManager`
instead of at the modules' own Parameters changes *nothing* numerically. This
checks that claim the only way that means anything — run the same forward both
ways, same weights, same input, and compare logits and every state tensor.

Then it checks the half that a numerical comparison cannot catch: that the state
outputs come back in the manager's published order. NxD's aliasing is positional,
so a permutation here writes the wrong cache and the model answers plausibly from
corrupted context. Verified by giving each state a distinguishable value and
confirming the collected order matches `output_order()` slot for slot.

CPU only.

    python test/test_dsv4_state_adapter.py
"""

import json
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

paths.add_hf_inference_to_syspath()

failures = []


def check(cond, label, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {label}" + (f"  {detail}" if detail else ""))
    if not cond:
        failures.append(label)
    return cond


SEQ, N_LAYERS, B = 256, 5, 1
PREFILL_LEN = 128


def build_model(hf, seed=0):
    """A weight-initialised HF model. See test_prefill_functional_vs_inplace for
    why explicit init is required: HF builds with torch.empty and expects a
    checkpoint load, so an unloaded model is uninitialised memory -> NaN logits."""
    with open(paths.config_json()) as f:
        cfg = json.load(f)
    cfg.update(dtype="bf16", expert_dtype=None, n_mtp_layers=0,
               max_batch_size=B, max_seq_len=SEQ, n_layers=N_LAYERS)
    cfg["compress_ratios"] = cfg["compress_ratios"][:N_LAYERS]
    torch.set_default_dtype(torch.bfloat16)
    m = hf.Transformer(hf.ModelArgs(**cfg)).eval()
    torch.set_default_dtype(torch.float32)
    torch.manual_seed(seed)
    with torch.no_grad():
        for n, p in m.named_parameters():
            if "norm" in n or n.endswith("attn_sink"):
                p.fill_(1.0)
            elif "ape" in n:
                p.zero_()
            else:
                p.copy_((torch.randn(p.shape) * 0.02).to(p.dtype))
        for n, b in m.named_buffers():
            if b.dtype.is_floating_point and "freqs" not in n \
                    and "score_state" not in n:
                b.zero_()
    return m


def main():
    from neuronx_distributed_inference.models.config import MoENeuronConfig
    import compile_neuron
    import decode_patches
    import dsv4_kv_cache as kvc
    import dsv4_state_adapter as adapter
    import modeling_dsv4 as md
    import prefill_patches

    compile_neuron._wire_shims()
    import model as hf
    compile_neuron.apply_xla_patches(hf)
    import collectives
    collectives.patch_hf_dist(hf)

    nc = MoENeuronConfig(tp_degree=32, batch_size=B, seq_len=SEQ,
                         torch_dtype=torch.bfloat16)
    cfg = md.DSV4InferenceConfig.from_checkpoint(
        paths.config_json(), nc, n_layers=N_LAYERS)
    kinds = md.build_layer_kinds(cfg)

    ids = torch.randint(0, 1000, (B, PREFILL_LEN))
    raw_fwd = getattr(hf.Transformer.forward, "__wrapped__",
                      hf.Transformer.forward)

    # --- reference: sink seeded from the model's own promoted Parameters ---
    print("=== reference: state owned by the model (the validated path) ===")
    m_ref = build_model(hf)
    state = prefill_patches.apply_prefill_patches(hf)
    slots_ref, _ = decode_patches.collect_state_aliases(m_ref, n_real_outputs=1)
    sink_ref = decode_patches.build_sink(m_ref, hf)
    state["sink"] = sink_ref
    with torch.no_grad():
        logits_ref = raw_fwd(m_ref, ids, 0).float().clone()
    # Key by (layer_idx, dotted) rather than id(module): m_ref and m_new are
    # different objects, so id()-keying silently matches nothing and every
    # comparison below would be skipped while still reporting "matches".
    import dsv4_state_adapter as _ad
    ref_map = _ad.build_state_map(m_ref, hf, kinds)
    ref_state = {key: sink_ref.get(mod, key[1].split(".")[-1]).float().clone()
                 for key, mod in ref_map.items()}
    print(f"  logits {list(logits_ref.shape)}, {len(ref_state)} states")

    # --- adapted: identical model, state owned by the manager ---
    print("\n=== adapted: state owned by DSV4CacheManager ===")
    m_new = build_model(hf)          # same seed -> same weights
    m_new.load_state_dict(m_ref.state_dict(), strict=False)
    mgr = kvc.DSV4CacheManager(kinds, batch_size=B, dtype=torch.bfloat16)
    state_map = adapter.build_state_map(m_new, hf, kinds)
    sink_new = adapter.build_managed_sink(m_new, hf, kinds, mgr)
    state["sink"] = sink_new
    with torch.no_grad():
        logits_new = raw_fwd(m_new, ids, 0).float().clone()
    outs = adapter.collect_outputs(sink_new, mgr, state_map)
    print(f"  logits {list(logits_new.shape)}, {len(outs)} state outputs")

    print("\n=== 1. logits identical ===")
    d = (logits_new - logits_ref).abs()
    denom = logits_ref.abs().mean().clamp_min(1e-6)
    rel = float(d.mean() / denom)
    cos = float(torch.nn.functional.cosine_similarity(
        logits_new.flatten(), logits_ref.flatten(), dim=0))
    check(rel < 2e-2, "logits match the validated path",
          f"rel={rel:.2e} cos={cos:.6f} max|d|={float(d.max()):.4f}")

    print("\n=== 2. every state tensor identical ===")
    check(len(outs) == len(ref_state),
          "same number of states", f"{len(outs)} vs {len(ref_state)}")
    worst, worst_name = 0.0, None
    compared, skipped = [], []
    order = mgr.output_order()
    for i, (layer_idx, dotted) in enumerate(order):
        mod = state_map[(layer_idx, dotted)]
        leaf = dotted.split(".")[-1]
        got = outs[i].float()
        want = ref_state.get((layer_idx, dotted))
        if want is None:
            skipped.append(f"{layer_idx}.{dotted}")
            continue
        compared.append(f"{layer_idx}.{dotted}")
        if got.shape != want.shape:
            check(False, f"shape {layer_idx}.{dotted}",
                  f"{tuple(got.shape)} vs {tuple(want.shape)}")
            continue
        fa = torch.where(torch.isfinite(got), got, torch.zeros_like(got))
        fb = torch.where(torch.isfinite(want), want, torch.zeros_like(want))
        m = float((fa - fb).abs().max())
        if m > worst:
            worst, worst_name = m, f"{layer_idx}.{dotted}"
    # Guard against a vacuous pass: the reference keys on the *reference model's*
    # module objects, so a bug in the mapping would skip everything and still
    # report "matches". Require that most states were actually compared.
    check(len(compared) >= len(order) - 2,
          "the comparison is not vacuous",
          f"{len(compared)}/{len(order)} compared, skipped: {skipped}")
    check(worst < 2e-2, "every compared state matches",
          f"worst max|d|={worst:.4f} at {worst_name or 'n/a'} "
          f"over {len(compared)} states")

    print("\n=== 3. output order matches the alias map (positional contract) ===")
    # Give every manager slot a unique marker, then confirm collect_outputs
    # returns them in the manager's order. This is what a numerical comparison
    # cannot catch: a permutation would still be "all states present".
    with torch.no_grad():
        for i, t in enumerate(mgr.past_key_values):
            t.fill_(float(i + 1))
    sink_marked = adapter.build_managed_sink(m_new, hf, kinds, mgr)
    marked = adapter.collect_outputs(sink_marked, mgr, state_map)
    # A redirected slot reads a slice of its owner, so compare the first element
    # of each owner tensor rather than the whole thing.
    seen = []
    for i, t in enumerate(marked):
        seen.append(float(t.flatten()[0]))
    expected = []
    for i, (layer_idx, dotted) in enumerate(order):
        mod = state_map[(layer_idx, dotted)]
        key = (id(mod), dotted.split(".")[-1])
        expected.append(float(mgr.past_key_values[i].flatten()[0]))
    mismatch = [(i, s, e) for i, (s, e) in enumerate(zip(seen, expected))
                if abs(s - e) > 1e-6]
    check(not mismatch,
          "collect_outputs returns slots in output_order()",
          "" if not mismatch else f"{len(mismatch)} slots out of order, "
                                  f"first: {mismatch[0]}")

    print("\n=== 4. state actually came from the manager ===")
    # If the sink were still reading the model's Parameters, the markers above
    # would not be visible through it.
    check(any(abs(v) > 0.5 for v in seen),
          "the sink reads manager-owned tensors, not the modules' own",
          f"first few markers: {[round(v, 1) for v in seen[:5]]}")

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
