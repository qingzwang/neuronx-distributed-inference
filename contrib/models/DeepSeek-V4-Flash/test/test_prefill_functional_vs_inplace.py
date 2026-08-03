#!/usr/bin/env python
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check the functional prefill rewrite matches the validated in-place one.

`prefill_patches` rewrites prefill's cache writes from in-place slice assignment
to functional `index_copy` + sink, so prefill can share decode's aliased cache
(JOINT_INFERENCE.md). That rewrite must not change any numbers.

Runs on CPU at small n_layers with random weights, comparing:

  1. logits, functional vs in-place — the rewrite is supposed to be a no-op
  2. every state buffer afterwards — this is the part that actually matters for
     joint inference. Prefill's job is not only to produce logits but to leave
     the window ring, the compressed cache, and the compressor's kv_state /
     score_state in the exact configuration a following decode step expects. A
     logits-only check would pass even if the ring were written at the wrong
     offsets, and the bug would only surface as a subtly wrong first decode
     token later.

Random weights are fine here: this compares two implementations of the same
arithmetic, so any weights exercise it. Run on CPU because the point is the
rewrite, not the backend.

Run:
    python test/test_prefill_functional_vs_inplace.py --n-layers 5 --seq-len 128
"""

import argparse
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


def build_model(hf, n_layers, seq_len, seed=0):
    import compile_neuron
    torch.manual_seed(seed)
    cfg_path = paths.config_json()
    import json
    with open(cfg_path) as f:
        cfg = json.load(f)
    cfg.update(dtype="bf16", expert_dtype=None, n_mtp_layers=0,
               max_batch_size=1, max_seq_len=seq_len, n_layers=n_layers)
    if len(cfg.get("compress_ratios", [])) > n_layers:
        cfg["compress_ratios"] = cfg["compress_ratios"][:n_layers]
    torch.set_default_dtype(torch.bfloat16)
    m = hf.Transformer(hf.ModelArgs(**cfg)).eval()
    torch.set_default_dtype(torch.float32)

    # HF builds its Linear/Embedding with torch.empty and expects a checkpoint
    # load to fill them, so an unloaded model is uninitialised memory, not random
    # weights — every logit comes back NaN. Fill everything explicitly, small
    # enough that 43 layers of residual growth cannot overflow bf16.
    torch.manual_seed(seed)
    with torch.no_grad():
        for name, p in m.named_parameters():
            if "norm" in name or name.endswith("attn_sink"):
                p.fill_(1.0)
            elif "ape" in name:
                p.zero_()
            else:
                p.copy_(torch.randn(p.shape, dtype=torch.float32).to(p.dtype)
                        * 0.02)
        for name, b in m.named_buffers():
            if b.dtype.is_floating_point and "freqs" not in name:
                # kv_state/score_state keep their init values (score_state is
                # -inf); only zero the caches.
                if "score_state" not in name:
                    b.zero_()
    return m


def state_snapshot(model, hf):
    """Every persistent decode-relevant state tensor, by qualified name.

    Reads via getattr rather than named_buffers(): `collect_state_aliases`
    promotes these from buffers to Parameters (NxD only scans named_parameters()
    when resolving aliases), so a buffers-only walk sees 22 entries before the
    promotion and 17 after, and the two snapshots would not be comparable.
    """
    out = {}
    for name, mod in model.named_modules():
        for buf in ("kv_cache", "kv_state", "score_state"):
            t = getattr(mod, buf, None)
            if isinstance(t, torch.Tensor):
                out[f"{name}.{buf}"] = t.detach().float().clone()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-layers", type=int, default=5,
                    help="5 covers every distinct layer type (see README).")
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--tol", type=float, default=2e-2,
                    help="bf16 slack on logits.")
    args = ap.parse_args()

    os.environ.setdefault("DSV4_N_LAYERS", str(args.n_layers))
    os.environ.setdefault("DSV4_SEQ_LEN", str(args.seq_len))

    import compile_neuron
    compile_neuron._wire_shims()
    import model as hf

    ids = torch.randint(0, 1000, (1, args.seq_len))

    # --- in-place reference: the validated XLA prefill path ---
    print("=== in-place (reference) ===")
    compile_neuron.apply_xla_patches(hf)
    m_ref = build_model(hf, args.n_layers, args.seq_len)
    with torch.no_grad():
        raw = getattr(hf.Transformer.forward, "__wrapped__", hf.Transformer.forward)
        ref_logits = raw(m_ref, ids, 0).float().clone()
    ref_state = state_snapshot(m_ref, hf)
    print(f"  logits {list(ref_logits.shape)}, "
          f"{len(ref_state)} state buffers")

    # --- functional: same model, sink-based writes ---
    print("\n=== functional (sink-based) ===")
    import decode_patches
    import prefill_patches

    # Rebuild from the same seed so weights match exactly, then re-patch.
    compile_neuron.apply_xla_patches(hf)
    m_fn = build_model(hf, args.n_layers, args.seq_len)
    m_fn.load_state_dict(m_ref.state_dict(), strict=False)

    state = prefill_patches.apply_prefill_patches(hf)
    slots, _ = decode_patches.collect_state_aliases(m_fn, n_real_outputs=1)
    sink = decode_patches.build_sink(m_fn, hf)
    state["sink"] = sink
    with torch.no_grad():
        raw = getattr(hf.Transformer.forward, "__wrapped__", hf.Transformer.forward)
        fn_logits = raw(m_fn, ids, 0).float().clone()
    # The sink holds the post-forward state; write it back so the comparison
    # sees what a real graph would alias out.
    with torch.no_grad():
        for mod, name in slots:
            getattr(mod, name).copy_(sink.get(mod, name).to(
                getattr(mod, name).dtype))
    fn_state = state_snapshot(m_fn, hf)
    print(f"  logits {list(fn_logits.shape)}, {len(fn_state)} state buffers")

    print("\n=== 1. logits ===")
    d = (fn_logits - ref_logits).abs()
    denom = ref_logits.abs().mean().clamp_min(1e-6)
    rel = float(d.mean() / denom)
    cos = float(torch.nn.functional.cosine_similarity(
        fn_logits.flatten(), ref_logits.flatten(), dim=0))
    check(rel < args.tol, "logits match the in-place path",
          f"rel={rel:.2e} cos={cos:.6f} max|d|={float(d.max()):.4f}")

    print("\n=== 2. state buffers (what decode will read) ===")
    # The reference has *more* entries, and that is expected: HF wires
    # `compressor.kv_cache = attn.kv_cache[:, win:]` during forward, so a
    # compressor shows up owning a view of data the attention layer owns. The
    # sink replaces those views with owner/offset redirects (index_copy returns a
    # new tensor, so a write through a view would be invisible to the owner), so
    # the compressor's attribute stays None and the entry does not appear.
    # Comparing the intersection is therefore the correct comparison — the extra
    # reference rows are duplicates of owner data already being compared.
    only_ref = sorted(set(ref_state) - set(fn_state))
    shared = sorted(set(fn_state) & set(ref_state))
    print(f"  comparing {len(shared)} owner buffers; "
          f"{len(only_ref)} reference-only entries are HF's compressor views")
    if only_ref:
        assert all("compressor" in k for k in only_ref), only_ref
    check(len(shared) > 0 and all("compressor" in k for k in only_ref),
          "the only extra reference entries are compressor views",
          f"{len(only_ref)} views, e.g. {only_ref[:2]}")
    worst, worst_name = 0.0, None
    for k in sorted(set(fn_state) & set(ref_state)):
        a, b = fn_state[k], ref_state[k]
        if a.shape != b.shape:
            check(False, f"{k} shape", f"{tuple(a.shape)} vs {tuple(b.shape)}")
            continue
        fa = torch.where(torch.isfinite(a), a, torch.zeros_like(a))
        fb = torch.where(torch.isfinite(b), b, torch.zeros_like(b))
        m = float((fa - fb).abs().max())
        if m > worst:
            worst, worst_name = m, k
    check(worst < args.tol, "every state buffer matches",
          f"worst max|d|={worst:.4f} at {worst_name}")

    # A cache left all-zero would make the comparison vacuous.
    nz = {k: float((v != 0).float().mean()) for k, v in fn_state.items()}
    written = sum(1 for v in nz.values() if v > 0)
    check(written > 0, "prefill actually wrote state",
          f"{written}/{len(nz)} buffers non-zero")

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
