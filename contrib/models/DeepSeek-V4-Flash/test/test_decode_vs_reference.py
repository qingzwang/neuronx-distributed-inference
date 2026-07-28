#!/usr/bin/env python
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check the decode rewrite against HF's own decode, on CPU at tiny depth.

The decode patches replace five uses of `start_pos`-as-a-Python-int with tensor
equivalents (see decode_patches.py). Each is a place where "looks right" and "is
right" diverge quietly: a wrong ring-buffer slot, an off-by-one in the
compressor's write position, or an index masked that should not have been all
produce finite, plausible logits. Only a comparison against the reference
catches them.

So: run HF's *unpatched* model — prefill the prompt, then decode token by token,
exactly as `model.py`'s own __main__ does — and run the patched decode path over
the same weights and the same positions. The logits must agree at every step.

This runs at world_size=1 on CPU with 3 layers, a few GB rather than the 568 GB
a full-depth CPU model would need. Per the standing constraint, no full-scale
CPU experiments: what is under test here is position arithmetic, which is
depth-independent, so small depth is the right place for it.

Run:
    source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
    cd contrib/models/DeepSeek-V4-Flash
    python test/test_decode_vs_reference.py --n-layers 3 --steps 4
"""

import argparse
import importlib
import json
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.abspath(os.path.join(_HERE, "..", "src"))
sys.path.insert(0, _SRC)

import compile_neuron as C  # noqa: E402
import paths  # noqa: E402
import shard_loader as SL  # noqa: E402


def fresh_hf_module():
    """Import a *new* `model` module object.

    The patches mutate classes in place, so the reference model and the patched
    one cannot share a module object: applying the XLA patches for our run would
    retroactively change the reference too. Dropping `model` from sys.modules
    and re-importing gives each its own classes.
    """
    import torch.distributed as dist
    dist.is_initialized = lambda: False
    for name in ("model", "kernel"):
        sys.modules.pop(name, None)
    C._wire_shims()
    paths.add_hf_inference_to_syspath()
    return importlib.import_module("model")


def build(hf, n_layers, seq_len, batch):
    """Instantiate hf.Transformer with the real weights at world_size=1."""
    torch.set_default_dtype(torch.bfloat16)
    cfg = json.load(open(paths.config_json()))
    cfg.update(dtype="bf16", expert_dtype=None, n_mtp_layers=0,
               max_batch_size=batch, max_seq_len=seq_len)
    cfg["n_layers"] = n_layers
    cfg["compress_ratios"] = cfg["compress_ratios"][:n_layers]
    model = hf.Transformer(hf.ModelArgs(**cfg)).eval()
    SL.load_rank_weights(model, paths.model_path(), rank=0, world_size=1)
    return model


def raw_forward(model, ids, pos):
    """Call Transformer.forward past its @torch.inference_mode() decorator."""
    fn = getattr(model.forward, "__wrapped__", None)
    with torch.no_grad():
        if fn is not None:
            return fn(model, ids, pos)
        return model(ids, pos)


def compare(step, ref, got, tol_rel):
    ref_f, got_f = ref.float().flatten(), got.float().flatten()
    rel = ((ref_f - got_f).abs().mean()
           / ref_f.abs().mean().clamp_min(1e-6)).item()
    cos = torch.nn.functional.cosine_similarity(
        ref_f.unsqueeze(0), got_f.unsqueeze(0)).item()
    r1, g1 = int(ref_f.argmax()), int(got_f.argmax())
    ok = (r1 == g1) and rel < tol_rel
    print(f"  step {step}: cos={cos:.6f} rel={rel:.5f} "
          f"top1 ref={r1} got={g1} "
          f"{'MATCH' if r1 == g1 else 'MISMATCH'} "
          f"=> {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-layers", type=int, default=3,
                    help="3 covers compress_ratio 0 (layers 0-1) and ratio 4 "
                         "with the hash gate + Indexer (layer 2). 5 also covers "
                         "the score gate and ratio 128 — slower, but every "
                         "layer type.")
    ap.add_argument("--seq-len", type=int, default=128,
                    help="max_seq_len; must be >= max(compress_ratios) among "
                         "the layers used.")
    ap.add_argument("--prompt-len", type=int, default=8,
                    help="Tokens to prefill before decoding.")
    ap.add_argument("--steps", type=int, default=4, help="Decode steps.")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--tol-rel", type=float, default=0.02,
                    help="bf16 accumulation order differs between the reference "
                         "and the rewrite, so exact equality is not expected.")
    ap.add_argument("--no-prefill", action="store_true",
                    help="Drive the decode path from position 0 for the prompt "
                         "too, instead of prefilling it in one shot. This is "
                         "what the device harness has to do: prefill and decode "
                         "are separate traced artifacts with separate device "
                         "buffers, so prefill's KV cache cannot be handed to the "
                         "decode graph. The reference still prefills, so this "
                         "also checks that token-by-token and one-shot agree.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    vocab = json.load(open(paths.config_json()))["vocab_size"]
    ids = torch.randint(0, vocab, (args.batch, args.prompt_len))
    next_ids = torch.randint(0, vocab, (args.batch, args.steps))

    print("=" * 68)
    print(f"decode vs HF reference: n_layers={args.n_layers} "
          f"prompt_len={args.prompt_len} steps={args.steps}")
    print("=" * 68)

    # --- Reference: HF verbatim, prefill then decode step by step ---
    print("\n[reference] building unpatched HF model ...")
    hf_ref = fresh_hf_module()
    ref_model = build(hf_ref, args.n_layers, args.seq_len, args.batch)
    print("[reference] prefill + decode ...")
    raw_forward(ref_model, ids, 0)
    ref_logits = []
    for s in range(args.steps):
        out = raw_forward(ref_model, next_ids[:, s:s + 1], args.prompt_len + s)
        ref_logits.append(out.detach().clone())
    del ref_model

    # --- Ours: fill the caches, then run the decode path ---
    print("\n[ours] building XLA-patched model ...")
    hf = fresh_hf_module()
    C.apply_xla_patches(hf)
    model = build(hf, args.n_layers, args.seq_len, args.batch)

    if args.no_prefill:
        # Decode from position 0: every prompt token goes through the decode
        # graph, one at a time. Position 0 included, which the closed-form index
        # helpers have to handle even though HF treats it as the prefill branch.
        print("[ours] skipping prefill; decoding the prompt token by token")
    else:
        # Prefill in one shot, then continue with decode on the same live model
        # so the caches carry over — the cheaper path, and what a single-process
        # CPU run would naturally do.
        print("[ours] prefill via the prefill path ...")
        raw_forward(model, ids, 0)

    print("[ours] installing decode patches ...")
    import decode_patches
    decode_state = decode_patches.apply_decode_patches(hf)
    slots, _ = decode_patches.collect_state_aliases(model, n_real_outputs=1)
    print(f"[ours] {len(slots)} state tensors promoted to Parameters")

    def decode_one(token, pos_int):
        """One decode call, then the state write-back NxD's aliasing does."""
        pos = torch.tensor(pos_int, dtype=torch.int32)
        sink = decode_patches.build_sink(model, hf)
        decode_state["pos"] = pos
        decode_state["sink"] = sink
        out = raw_forward(model, token, pos)
        with torch.no_grad():
            for mod, name in slots:
                getattr(mod, name).copy_(sink.get(mod, name))
        return out

    if args.no_prefill:
        print(f"[ours] decoding {args.prompt_len} prompt tokens ...")
        for p in range(args.prompt_len):
            decode_one(ids[:, p:p + 1], p)

    print("[ours] decode ...")
    got_logits = []
    for s in range(args.steps):
        out = decode_one(next_ids[:, s:s + 1], args.prompt_len + s)
        got_logits.append(out.detach().clone())

    print("\n" + "-" * 68)
    print("logits, decode step by step")
    print("-" * 68)
    all_ok = True
    for s, (r, g) in enumerate(zip(ref_logits, got_logits)):
        all_ok &= compare(s, r, g, args.tol_rel)

    print("\n" + "=" * 68)
    print("ALL STEPS PASS" if all_ok else "FAILURES PRESENT")
    print("=" * 68)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
