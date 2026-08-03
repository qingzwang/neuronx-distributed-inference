#!/usr/bin/env python
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Measure the prefill graph's TTFT, separating warmup from steady state.

Why this exists as its own script: the README used to report "492 s for one
128-token prefill" and call prefill unusable. That was a single measurement, and
this port has a large first-call warmup on *any* graph (compiler cache
population, collective setup), which a one-shot timing cannot separate from real
cost. Measuring it properly settled the question the other way:

    warmup 464 s, then 569 s, then 0.85 s / 0.85 s / 0.85 s / 0.85 s

Steady-state prefill is **0.85 s** for 128 tokens — 6.6 ms/position, against
93.7 ms/position through the decode graph. Prefill is 14x faster to first token,
because a batched call amortizes one pass over the masked MoE's expert weights
across every position in the prompt.

Two lessons this encodes, both of which cost a wrong conclusion first:
  * warmup takes TWO calls here, not one
  * report the median; a mean over a set holding one ~500 s outlier is noise

Method: run the same fixed-width prompt N+1 times. Call 0 is reported as warmup
and excluded; calls 1..N give the steady-state distribution. Nothing here needs
batch > 1 — the question is the cost of one prefill call, repeated.

The prefill graph takes the whole prompt in one call with `start_pos` baked to 0,
so TTFT *is* that single call: no per-token ingest loop, unlike decode. The
prompt must fill `seq_len` exactly (the head emits only the last position), so
`--fill` left-pads with real text the way run_neuron.py does.

Run:
    python src/bench_prefill_ttft.py \
        --artifact /mnt/data/artifacts/dsv4_prefill_tp32_L43_S128 \
        --seq-len 128 --repeats 5
"""

import argparse
import json
import os
import statistics
import sys
import time

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import paths  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", required=True)
    ap.add_argument("--seq-len", type=int, required=True,
                    help="Must equal the compiled graph's seq_len; the prompt "
                         "is padded to exactly this width.")
    ap.add_argument("--repeats", type=int, default=6,
                    help="Timed calls after the warmup call. Keep this >= 3: "
                         "warmup takes TWO calls on this graph, not one, so the "
                         "first timed call is still ~500 s and the mean is "
                         "meaningless below that. Read the median.")
    ap.add_argument("--prompt", default="It is well known that the capital city "
                                        "of France is")
    ap.add_argument("--fill", default="Artificial intelligence research has a "
                                      "long history. ",
                    help="Left-fill text so the prompt ends exactly at seq_len.")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(os.path.join(paths.model_path(), "tokenizer.json"))

    ids = tok.encode(args.prompt).ids
    if len(ids) < args.seq_len:
        filler = tok.encode(args.fill).ids
        need = args.seq_len - len(ids)
        reps = (need // len(filler)) + 1
        ids = (filler * reps)[-need:] + ids
    ids = ids[-args.seq_len:]
    print(f"[input] {len(ids)} tokens (prompt tail: {tok.decode(ids[-12:])!r})")

    from neuronx_distributed.trace import parallel_model_load
    print(f"[neuron] loading {args.artifact}", flush=True)
    t0 = time.perf_counter()
    model = parallel_model_load(args.artifact)
    load_s = time.perf_counter() - t0
    print(f"[neuron] loaded in {load_s:.1f}s", flush=True)

    inp = torch.tensor([ids], dtype=torch.long)
    start_pos = torch.zeros((), dtype=torch.int32)

    def one_call():
        t = time.perf_counter()
        out = model(inp, start_pos)
        logits = out[0] if isinstance(out, (tuple, list)) else out
        if logits.dim() == 3:
            logits = logits[:, -1, :]
        # Force the result to the host: the timing must include getting the
        # logits back, or it measures dispatch rather than completion.
        top = int(logits[0].float().argmax())
        return time.perf_counter() - t, top

    print("\n[warmup] first call (excluded from statistics) ...", flush=True)
    warm_s, top0 = one_call()
    print(f"[warmup] {warm_s:.1f}s  top1={top0} {tok.decode([top0])!r}",
          flush=True)

    times = []
    for i in range(args.repeats):
        dt, top = one_call()
        times.append(dt)
        print(f"  call {i + 1}: {dt:.2f}s  top1={top} {tok.decode([top])!r}",
              flush=True)

    # Warmup takes two calls, so times[0] is still a warmup outlier. Report the
    # median and derive per-position from it; the mean over a set containing a
    # ~500 s outlier describes nothing.
    med = statistics.median(times)
    settled = [t for t in times if t < 10 * med]
    print("\n" + "=" * 62)
    print(f"prefill TTFT, seq_len={args.seq_len}, 43 layers, TP=32, batch=1")
    print(f"  warmup call    {warm_s:.1f} s  (excluded)")
    print(f"  steady state   median {med:.2f} s"
          f"   min {min(times):.2f}  max {max(times):.2f}")
    if len(settled) < len(times):
        print(f"                 {len(times) - len(settled)} of {len(times)} "
              f"timed calls were still warming up (>10x median) and are "
              f"excluded from the mean")
    if settled:
        print(f"                 mean of settled calls {statistics.mean(settled):.2f} s"
              f"  (n={len(settled)})")
    print(f"  per position   {med / args.seq_len * 1000:.1f}"
          f" ms  ({args.seq_len} positions in one call, from the median)")
    print(f"  artifact load  {load_s:.1f} s")

    if args.out:
        with open(args.out, "w") as f:
            json.dump({
                "seq_len": args.seq_len, "warmup_s": warm_s,
                "steady_s": times, "load_s": load_s,
            }, f, indent=2)
        print(f"\n[out] wrote {args.out}")


if __name__ == "__main__":
    main()
