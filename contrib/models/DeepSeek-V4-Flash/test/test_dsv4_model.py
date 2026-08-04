#!/usr/bin/env python
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Gate for design step 4: the assembled model class, both modes.

Checks the whole `DSV4Model.forward` — MoE, head, hyper-connections and all —
against the validated patched-HF forward, in *both* prefill and decode mode, and
verifies the pieces the tracing side depends on:

  * logits match the reference for prefill and for decode
  * the returned tuple is (logits, *state) with state in the manager's order
  * the alias map is consistent with that tuple: output n+1 aliases slot n
  * the mode dispatch actually switches behaviour (a prefill-shaped call and a
    decode-shaped call must not produce identical state writes)

The last one matters because the dispatcher is shared module-level mutable state.
If `active` were ignored, both graphs would silently trace the same branch — the
bug that cost the most time on the previous attempt.

CPU only.

    python test/test_dsv4_model.py
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


SEQ, N_LAYERS, B, PREFILL_LEN = 256, 5, 1, 128


def build_hf_model(hf, seed=0):
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
    import collectives
    import compile_neuron
    import decode_patches
    import dsv4_patches
    import modeling_dsv4 as md
    import prefill_patches

    compile_neuron._wire_shims()
    import model as hf
    compile_neuron.apply_xla_patches(hf)
    collectives.patch_hf_dist(hf)

    nc = MoENeuronConfig(tp_degree=32, batch_size=B, seq_len=SEQ,
                         torch_dtype=torch.bfloat16)
    cfg = md.DSV4InferenceConfig.from_checkpoint(
        paths.config_json(), nc, n_layers=N_LAYERS)

    ids_prefill = torch.randint(0, 1000, (B, PREFILL_LEN))
    pos_prefill = torch.arange(PREFILL_LEN, dtype=torch.int32).unsqueeze(0)
    ids_decode = torch.randint(0, 1000, (B, 1))
    pos_decode = torch.tensor([[PREFILL_LEN]], dtype=torch.int32)

    raw_fwd = getattr(hf.Transformer.forward, "__wrapped__",
                      hf.Transformer.forward)

    # --- reference, prefill ---
    print("=== reference: validated patched-HF prefill ===")
    m_ref = build_hf_model(hf)
    st = prefill_patches.apply_prefill_patches(hf)
    decode_patches.collect_state_aliases(m_ref, n_real_outputs=1)
    st["sink"] = decode_patches.build_sink(m_ref, hf)
    with torch.no_grad():
        ref_prefill = raw_fwd(m_ref, ids_prefill, 0).float().clone()
    print(f"  logits {list(ref_prefill.shape)}")

    # --- reference, decode (fresh model, position PREFILL_LEN) ---
    print("\n=== reference: validated patched-HF decode ===")
    m_ref_d = build_hf_model(hf)
    st_d = decode_patches.apply_decode_patches(hf)
    decode_patches.collect_state_aliases(m_ref_d, n_real_outputs=1)
    st_d["sink"] = decode_patches.build_sink(m_ref_d, hf)
    st_d["pos"] = torch.tensor(PREFILL_LEN, dtype=torch.int32)
    with torch.no_grad():
        ref_decode = raw_fwd(m_ref_d, ids_decode,
                             torch.tensor(PREFILL_LEN)).float().clone()
    print(f"  logits {list(ref_decode.shape)}")

    # --- the model class, both modes ---
    print("\n=== DSV4Model ===")
    dsv4_patches.install(hf)

    m_p = build_hf_model(hf)
    model_p = md.DSV4Model(cfg, hf, m_p, mode="prefill")
    with torch.no_grad():
        out_p = model_p(ids_prefill, pos_prefill)

    m_d = build_hf_model(hf)
    model_d = md.DSV4Model(cfg, hf, m_d, mode="decode")
    with torch.no_grad():
        out_d = model_d(ids_decode, pos_decode)

    print(f"  prefill -> {len(out_p)} outputs, logits {list(out_p[0].shape)}")
    print(f"  decode  -> {len(out_d)} outputs, logits {list(out_d[0].shape)}")

    print("\n=== 1. prefill logits match the reference ===")
    a, b = out_p[0].float(), ref_prefill
    rel = float((a - b).abs().mean() / b.abs().mean().clamp_min(1e-6))
    cos = float(torch.nn.functional.cosine_similarity(
        a.flatten(), b.flatten(), dim=0))
    check(rel < 2e-2, "prefill logits", f"rel={rel:.2e} cos={cos:.6f}")

    print("\n=== 2. decode logits match the reference ===")
    a, b = out_d[0].float(), ref_decode
    rel = float((a - b).abs().mean() / b.abs().mean().clamp_min(1e-6))
    cos = float(torch.nn.functional.cosine_similarity(
        a.flatten(), b.flatten(), dim=0))
    check(rel < 2e-2, "decode logits", f"rel={rel:.2e} cos={cos:.6f}")

    print("\n=== 3. output tuple shape and alias-map consistency ===")
    n_states = len(model_p.kv_mgr)
    check(len(out_p) == 1 + n_states,
          "prefill returns (logits, *state)", f"{len(out_p)} == 1+{n_states}")
    check(len(out_d) == 1 + n_states,
          "decode returns (logits, *state)", f"{len(out_d)} == 1+{n_states}")
    aliases = model_p.alias_map(n_real_outputs=1)
    ok = all(aliases[model_p.kv_mgr.past_key_values[i]] == 1 + i
             for i in range(n_states))
    check(ok, "alias map: output 1+i aliases past_key_values[i]")
    # And the tuple positions line up with the manager's order.
    order = model_p.kv_mgr.output_order()
    shapes_ok = all(
        tuple(out_p[1 + i].shape[1:]) ==
        tuple(model_p.kv_mgr.past_key_values[i].shape[1:])
        or "kv_cache" in order[i][1]     # redirected slices are narrower
        for i in range(n_states))
    check(shapes_ok, "state outputs align with manager slots by position")

    print("\n=== 4. the mode dispatch actually switches branches ===")
    # A prefill call writes `seqlen` window slots; a decode call writes one. If
    # `active` were ignored, both would trace the same branch -- the failure that
    # cost the most time previously, and it is invisible in the logits.
    p_written = float((out_p[1] != 0).to(torch.float32).mean())
    d_written = float((out_d[1] != 0).to(torch.float32).mean())
    print(f"  layer0 kv_cache non-zero fraction: prefill {p_written:.3f}, "
          f"decode {d_written:.3f}")
    check(p_written > d_written * 4,
          "prefill writes many window slots, decode writes ~one",
          f"{p_written:.3f} vs {d_written:.3f}")

    print("\n=== 5. state is finite (a fresh ring must not produce NaN) ===")
    bad = [order[i] for i in range(n_states)
           if not bool(torch.isfinite(out_p[1 + i]).all())]
    check(not bad, "every prefill state output is finite",
          "" if not bad else f"non-finite at {bad[:3]}")

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
