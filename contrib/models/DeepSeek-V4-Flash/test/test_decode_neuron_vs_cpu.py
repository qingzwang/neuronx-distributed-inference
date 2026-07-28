#!/usr/bin/env python
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compare the compiled *decode* artifact against a CPU run of the same path.

test_decode_vs_reference.py proved the decode rewrite is right on CPU;
run_neuron.py --mode decode proved the compiled graph runs and stays finite.
Neither shows that the device is computing the right numbers, and decode has a
failure mode prefill does not: the KV cache lives on the device and is updated
through NxD's output aliasing. If an alias were dropped — which happens
silently, since a buffer key matches nothing (see decode_patches) — every step
would still produce plausible logits, computed against a cache that reset.

So drive both sides through the same sequence of positions and compare at every
step. A dropped alias or a mis-ordered state output shows up as a divergence
that *grows* with step count, which a single-step check would miss.

The CPU side is world_size=1 (full-size tensors, no sharding); the artifact is
the TP=N build. The gap between them is TP reduction + bf16 + Neuron lowering,
same as the prefill comparison.

Run:
    source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
    cd contrib/models/DeepSeek-V4-Flash
    python test/test_decode_neuron_vs_cpu.py \
        --artifact /mnt/data/artifacts/dsv4_decode_tp32_L5 \
        --n-layers 5 --seq-len 256 --prompt-len 6 --steps 8
"""

import argparse
import json
import os
import sys
import time

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.abspath(os.path.join(_HERE, "..", "src"))
sys.path.insert(0, _SRC)

import compile_neuron as C  # noqa: E402
import paths  # noqa: E402
import shard_loader as SL  # noqa: E402


def build_cpu_decode(n_layers, seq_len, batch):
    """Patched HF Transformer at world_size=1, with the decode forwards installed.

    Returns (model, hf, decode_state, slots) — everything the step function
    needs. Mirrors what _picklable_factory does in decode mode, minus the trace.
    """
    import torch.distributed as dist
    dist.is_initialized = lambda: False

    C._wire_shims()
    paths.add_hf_inference_to_syspath()
    import model as hf
    C.apply_xla_patches(hf)
    torch.set_default_dtype(torch.bfloat16)

    cfg = json.load(open(paths.config_json()))
    cfg.update(dtype="bf16", expert_dtype=None, n_mtp_layers=0,
               max_batch_size=batch, max_seq_len=seq_len)
    cfg["n_layers"] = n_layers
    cfg["compress_ratios"] = cfg["compress_ratios"][:n_layers]
    model = hf.Transformer(hf.ModelArgs(**cfg)).eval()
    SL.load_rank_weights(model, paths.model_path(), rank=0, world_size=1)

    import decode_patches
    decode_state = decode_patches.apply_decode_patches(hf)
    slots, _ = decode_patches.collect_state_aliases(model, n_real_outputs=1)
    return model, hf, decode_state, slots


def cpu_step(model, hf, decode_state, slots, token, pos_int):
    """One CPU decode step, including the state write-back.

    On device NxD's aliasing writes each state output back over its input
    buffer; here the sink holds the new values and we copy them into the
    Parameters by hand, which is the same thing.
    """
    import decode_patches
    pos = torch.tensor(pos_int, dtype=torch.int32)
    sink = decode_patches.build_sink(model, hf)
    decode_state["pos"] = pos
    decode_state["sink"] = sink
    raw = getattr(model.forward, "__wrapped__", None)
    with torch.no_grad():
        out = raw(model, token, pos) if raw else model(token, pos)
        for mod, name in slots:
            getattr(mod, name).copy_(sink.get(mod, name))
    return out


def report(step, ref, got, tol_rel, tol_cos):
    ref_f, got_f = ref.float().flatten(), got.float().flatten()
    rel = ((ref_f - got_f).abs().mean()
           / ref_f.abs().mean().clamp_min(1e-6)).item()
    cos = torch.nn.functional.cosine_similarity(
        ref_f.unsqueeze(0), got_f.unsqueeze(0)).item()
    k = min(5, ref_f.numel())
    ref_top = ref_f.topk(k).indices.tolist()
    got_top = got_f.topk(k).indices.tolist()
    top1 = ref_top[0] == got_top[0]
    # top-1 and cosine, not relative error, decide pass/fail. `rel` divides by
    # mean |logit|, and logits are near-zero-mean, so it is a noisy yardstick
    # here: it swings 8x across positions (0.006 to 0.055) on runs where every
    # top-1 agrees. Cosine over the whole 129k-vocab vector does not.
    ok = top1 and cos > tol_cos and rel < tol_rel
    print(f"  {step:>10}: cos={cos:.6f} rel={rel:.5f} "
          f"top1 cpu={ref_top[0]} neuron={got_top[0]} "
          f"{'MATCH' if top1 else 'MISMATCH'} "
          f"top5={len(set(ref_top) & set(got_top))}/5 "
          f"=> {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", required=True, help="dir with tp_*.pt")
    ap.add_argument("--n-layers", type=int, required=True,
                    help="Must match how the artifact was compiled.")
    ap.add_argument("--seq-len", type=int, required=True,
                    help="Compiled max_seq_len.")
    ap.add_argument("--prompt-len", type=int, default=6,
                    help="Prompt tokens, fed through decode one at a time — the "
                         "decode graph has no prefill to inherit a cache from.")
    ap.add_argument("--steps", type=int, default=8,
                    help="Generated steps after the prompt. More is better here: "
                         "cache corruption compounds, so a divergence that grows "
                         "with step count is the signal to watch for.")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--tol-cos", type=float, default=0.998,
                    help="Minimum cosine similarity on the logits, and the check "
                         "that actually discriminates. Measured floor at "
                         "n_layers=5 tp=32 was 0.9985 across two seeds; most "
                         "positions sit above 0.99997.")
    ap.add_argument("--tol-rel", type=float, default=0.08,
                    help="Allowed mean relative logit diff. Looser than the "
                         "prefill comparison's 0.02: most positions land at "
                         "0.006-0.008, but a few per run reach ~0.05, and which "
                         "ones changes with --seed (measured: gen 10/11 at "
                         "seed 0, gen 7/8 at seed 7). Input-dependent spikes are "
                         "numerics — a wrong index formula would hit the same "
                         "positions every run, since positions do not depend on "
                         "the tokens. Top-1 agreement is the real criterion and "
                         "held at every position in both runs.")
    ap.add_argument("--prompt-ids", default=None,
                    help="Comma-separated ids; default is a fixed pseudo-random "
                         "sequence (tokenizer-independent).")
    ap.add_argument("--seed", type=int, default=0,
                    help="Seed for the default pseudo-random ids. Vary it to "
                         "tell numerics from logic: bf16 noise moves with the "
                         "input, a wrong index formula does not.")
    ap.add_argument("--check-state", action="store_true",
                    help="Also compare the aliased KV caches, not just logits. "
                         "This is the direct test of the decode state machinery: "
                         "logits pass through a softmax and a vocab projection "
                         "that both wash out small differences, whereas a wrong "
                         "ring slot or a dropped alias shows up in the cache "
                         "immediately and unambiguously.")
    ap.add_argument("--tol-state", type=float, default=0.06,
                    help="Allowed mean relative diff on the caches. Measured at "
                         "n_layers=5 tp=32: 0.005-0.006 for most slots, up to "
                         "~0.05 for the deepest layer's compressor, already at "
                         "position 0 with nothing accumulated. The compressor "
                         "softmaxes over a float32 staging buffer, which "
                         "amplifies the TP-reduction difference in its inputs, "
                         "so the floor here is genuinely higher than for logits.")
    ap.add_argument("--tol-state-cos", type=float, default=0.997,
                    help="Minimum cosine similarity on the caches. This is the "
                         "check that actually discriminates: rounding noise "
                         "leaves cosine at ~1 however large the relative error, "
                         "while a wrong write index or a stale cache does not.")
    args = ap.parse_args()

    total = args.prompt_len + args.steps
    if total > args.seq_len:
        raise SystemExit(
            f"prompt_len + steps = {total} exceeds the compiled max_seq_len "
            f"({args.seq_len})")

    if args.prompt_ids:
        ids_list = [int(x) for x in args.prompt_ids.split(",")]
    else:
        g = torch.Generator().manual_seed(args.seed)
        ids_list = torch.randint(0, 129280, (total,), generator=g).tolist()
    ids_list = (ids_list + [0] * total)[:total]
    print(f"[input] {total} positions, ids[:8]={ids_list[:8]}")

    print(f"\n[cpu] building n_layers={args.n_layers} decode path (world_size=1)")
    t0 = time.perf_counter()
    model, hf, decode_state, slots = build_cpu_decode(
        args.n_layers, args.seq_len, args.batch)
    print(f"[cpu] built + loaded in {time.perf_counter() - t0:.1f}s; "
          f"{len(slots)} state tensors")

    # Names for the state slots, so a cache mismatch says which module it is.
    qualified = {}
    for mod_name, mod in model.named_modules():
        for m, name in slots:
            if m is mod:
                qualified[(id(mod), name)] = f"{mod_name}.{name}"

    print("[cpu] running all positions through decode ...")
    t0 = time.perf_counter()
    cpu_logits, cpu_states = [], []
    for p, tid in enumerate(ids_list):
        token = torch.tensor([[tid]] * args.batch, dtype=torch.long)
        cpu_logits.append(
            cpu_step(model, hf, decode_state, slots, token, p).detach().clone())
        if args.check_state:
            # After the write-back the Parameters are the post-step state, which
            # is exactly what the device aliases back over its own buffers.
            # .clone() is required, not decoration: several state slots are
            # already float32 and on CPU, so .cpu() and .to(float32) both
            # return *self* and we would store an alias of the live Parameter.
            # Every position would then hold the final state, and the comparison
            # would fail for exactly the float32 slots while the bf16 ones —
            # which get a real copy out of the dtype conversion — passed.
            cpu_states.append([getattr(m, n).detach().cpu()
                               .to(torch.float32).clone() for m, n in slots])
    state_names = [qualified.get((id(m), n), n) for m, n in slots]
    print(f"[cpu] {total} steps in {time.perf_counter() - t0:.1f}s")
    del model

    print(f"\n[neuron] loading {args.artifact}")
    from neuronx_distributed.trace import parallel_model_load
    t0 = time.perf_counter()
    neuron = parallel_model_load(args.artifact)
    print(f"[neuron] loaded in {time.perf_counter() - t0:.1f}s")

    print("[neuron] running the same positions ...")
    t0 = time.perf_counter()
    neuron_logits, neuron_states = [], []
    for p, tid in enumerate(ids_list):
        token = torch.tensor([[tid]] * args.batch, dtype=torch.long)
        out = neuron(token, torch.tensor(p, dtype=torch.int32))
        if isinstance(out, (tuple, list)):
            # (logits, *states) — the states are the aliased caches, which is
            # what makes this comparison possible at all: the graph hands back
            # its own post-step state, so we can check it without reaching into
            # device memory.
            neuron_logits.append(out[0].detach().clone())
            if args.check_state:
                # .cpu().float() rather than .clone(): the aliased state outputs
                # are not always plain CPU bf16 tensors, and .float() on them
                # fails with "Expected self.dtype() == dst.dtype()". Pull them
                # across explicitly and normalise the dtype here.
                # .cpu() first: these live on the Neuron device
                # (privateuseone:N) and .float() on a device bf16 tensor raises
                # "Expected self.dtype() == dst.dtype()". .clone() for the same
                # aliasing reason as the CPU side.
                neuron_states.append(
                    [s.detach().cpu().to(torch.float32).clone()
                     for s in out[1:]])
        else:
            neuron_logits.append(out.detach().clone())
    dt = time.perf_counter() - t0
    print(f"[neuron] {total} steps in {dt:.1f}s ({dt / total * 1000:.0f} ms/step)")

    bad = [i for i, x in enumerate(neuron_logits)
           if torch.isnan(x).any() or torch.isinf(x).any()]
    if bad:
        print(f"[neuron] FAIL: non-finite logits at steps {bad}")

    print("\n" + "-" * 74)
    print("Neuron TP vs CPU, per position "
          f"(prompt = 0..{args.prompt_len - 1}, generated = "
          f"{args.prompt_len}..{total - 1})")
    print("-" * 74)
    results = []
    for p, (c, n) in enumerate(zip(cpu_logits, neuron_logits)):
        label = f"prompt {p}" if p < args.prompt_len else f"gen {p}"
        results.append(report(label, c, n, args.tol_rel, args.tol_cos))

    if args.check_state and cpu_states and neuron_states:
        print("\n" + "-" * 74)
        print("aliased KV state, worst slot per position")
        print("-" * 74)
        if len(neuron_states[0]) != len(cpu_states[0]):
            print(f"  FAIL: graph returned {len(neuron_states[0])} states, CPU "
                  f"has {len(cpu_states[0])} — the alias map and the wrapper's "
                  f"output order disagree")
            results.append(False)
        else:
            nonfinite, mask_mismatch = [], []
            for p, (cs, ns) in enumerate(zip(cpu_states, neuron_states)):
                worst, worst_i, worst_cos = -1.0, 0, 1.0
                for i, (c, n) in enumerate(zip(cs, ns)):
                    cf, nf = c.flatten(), n.flatten()
                    if cf.shape != nf.shape:
                        print(f"  pos {p}: {state_names[i]} shape "
                              f"{tuple(c.shape)} vs {tuple(n.shape)} — the "
                              f"wrapper's output order does not match the "
                              f"alias map")
                        worst, worst_i = float("inf"), i
                        break
                    # -inf is legitimate here and must not be treated as an
                    # error: HF *initialises* Compressor.score_state to -inf
                    # (model.py "torch.full(..., float('-inf'))") so that
                    # not-yet-written slots soft-max to zero weight. Unwritten
                    # slots therefore stay -inf for as long as the buffer is not
                    # full. Compare the finite entries, and separately require
                    # that both sides mark the *same* slots as masked — a wrong
                    # write position shows up there, and only there.
                    c_masked = torch.isneginf(cf)
                    n_masked = torch.isneginf(nf)
                    if not torch.equal(c_masked, n_masked):
                        mask_mismatch.append((p, state_names[i]))
                        worst, worst_i = float("inf"), i
                        break
                    finite = ~c_masked
                    if not finite.any():
                        continue
                    cf, nf = cf[finite], nf[finite]
                    # Anything non-finite left over is a genuine NaN/+inf.
                    bad_side = []
                    if not torch.isfinite(cf).all():
                        bad_side.append("cpu")
                    if not torch.isfinite(nf).all():
                        bad_side.append("neuron")
                    if bad_side:
                        nonfinite.append(
                            (p, state_names[i], "+".join(bad_side)))
                        continue
                    rel = ((cf - nf).abs().mean()
                           / cf.abs().mean().clamp_min(1e-6)).item()
                    if rel > worst:
                        worst, worst_i = rel, i
                        # Cosine as well as relative error, because they fail
                        # differently: bf16 rounding perturbs magnitudes and
                        # leaves cosine at ~1, whereas a wrong slot or a stale
                        # cache changes *which* vector is there and drops it.
                        worst_cos = torch.nn.functional.cosine_similarity(
                            cf.unsqueeze(0), nf.unsqueeze(0)).item()
                ok = worst < args.tol_state and worst_cos > args.tol_state_cos
                results.append(ok)
                print(f"  pos {p:>3}: worst rel={worst:.5f} cos={worst_cos:.6f} "
                      f"in {state_names[worst_i]} => {'PASS' if ok else 'FAIL'}")
            # A cache that reset would read as near-total disagreement in the
            # populated region, so also report how full it is.
            nz = [float((s != 0).to(torch.float32).mean()) for s in neuron_states[-1]]
            print(f"  final state occupancy: min={min(nz):.3f} "
                  f"max={max(nz):.3f} (0.0 would mean the cache never got "
                  f"written, i.e. a dropped alias)")
            if mask_mismatch:
                print(f"\n  FAIL: {len(mask_mismatch)} slot(s) where the two "
                      f"sides masked different positions — this is a wrong "
                      f"write index, not numerics:")
                for p, name in mask_mismatch[:8]:
                    print(f"    pos {p} {name}")
                results.append(False)
            if nonfinite:
                # NaN or +inf among the *finite* (written) entries. Both sides
                # agreeing still means something is wrong upstream; one side
                # only means the device diverged.
                both = [x for x in nonfinite if x[2] == "cpu+neuron"]
                one = [x for x in nonfinite if x[2] != "cpu+neuron"]
                print(f"\n  FAIL: {len(nonfinite)} slot(s) non-finite outside "
                      f"the -inf mask ({len(both)} both sides, {len(one)} one):")
                for p, name, side in (both[:4] + one[:4]):
                    print(f"    pos {p} {name} ({side})")
                results.append(False)

    print("\n" + "=" * 74)
    ok = all(results) and not bad
    print("ALL POSITIONS PASS" if ok else "FAILURES PRESENT")
    print("=" * 74)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
