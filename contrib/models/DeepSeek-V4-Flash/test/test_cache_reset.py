#!/usr/bin/env python
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check that the decode cache can be reset, and that it has to be.

Two claims, both measured on device against the same artifact:

  1. Without a reset, running prompt B after prompt A gives a different answer
     to B than running B first. That is the contamination — the device-resident
     cache still holds A's compressed KV, and the indexer can select it.
  2. With `cache_reset.reset_decode_cache`, B-after-A matches B-first exactly.

Claim 1 is the one worth measuring rather than assuming. Part of the stale
window is masked out by `window_topk_idxs` (slot j is only read when j <= pos),
so it is genuinely possible for a short run to look clean by luck. If claim 1
comes back negative for the prompts tried, that is reported as UNPROVEN rather
than PASS: it means these prompts did not expose it, not that the cache is safe.

Run:
    python test/test_cache_reset.py --artifact /path/to/dsv4_decode_tp32_L43 \
        --seq-len 512
"""

import argparse
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(os.path.dirname(_HERE), "src")
for p in (_SRC, _HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import cache_reset  # noqa: E402
import paths  # noqa: E402

failures = []


def check(cond, label, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {label}" + (f"  {detail}" if detail else ""))
    if not cond:
        failures.append(label)
    return cond


def greedy_run(model, ids, n_new, batch=1):
    """Ingest `ids` then greedily emit n_new tokens. Returns (first_logits, toks)."""
    logits = None
    for p, tid in enumerate(ids):
        inp = torch.tensor([[tid]] * batch, dtype=torch.long)
        out = model(inp, torch.tensor(p, dtype=torch.int32))
        logits = out[0] if isinstance(out, (tuple, list)) else out
    first = logits[0].float().clone()

    toks, pos = [], len(ids)
    for _ in range(n_new):
        nxt = int(logits[0].float().argmax())
        toks.append(nxt)
        inp = torch.tensor([[nxt]] * batch, dtype=torch.long)
        out = model(inp, torch.tensor(pos, dtype=torch.int32))
        logits = out[0] if isinstance(out, (tuple, list)) else out
        pos += 1
    return first, toks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", required=True)
    ap.add_argument("--seq-len", type=int, required=True)
    ap.add_argument("--n-new", type=int, default=6)
    ap.add_argument("--check-norms", action="store_true",
                    help="Also compare per-slot state norms before/after. Off "
                         "by default: it copies all 188 state tensors to the "
                         "host, which is slow, and the token-level check below "
                         "already covers what it would tell us.")
    args = ap.parse_args()

    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(os.path.join(paths.model_path(), "tokenizer.json"))

    # A deliberately different, longer prompt, so it writes cache slots that a
    # shorter B would otherwise leave untouched — that is what makes stale state
    # reachable rather than masked.
    a_ids = tok.encode(
        "Beijing is the capital of China and a major cultural center with a "
        "long history spanning many dynasties and imperial eras."
    ).ids
    b_ids = tok.encode("The capital of France is").ids
    print(f"[input] A = {len(a_ids)} tokens, B = {len(b_ids)} tokens")

    from neuronx_distributed.trace import parallel_model_load
    print(f"[neuron] loading {args.artifact}")
    model = parallel_model_load(args.artifact)

    print("\n=== 1. B on a fresh cache (the reference) ===")
    fresh = cache_reset.snapshot_state_norms(model) if args.check_norms else None
    n_states = len(cache_reset._rank_state_tensors(model)[0])
    if fresh is not None:
        print(f"  {n_states} state tensors per rank; "
              f"{sum(1 for v in fresh if v == float('inf'))} start at -inf")
    else:
        print(f"  {n_states} state tensors per rank")
    b_first_logits, b_first_toks = greedy_run(model, b_ids, args.n_new)
    print(f"  B first -> {b_first_toks} {tok.decode(b_first_toks)!r}")

    print("\n=== 2. B after A, no reset (is it contaminated?) ===")
    greedy_run(model, a_ids, args.n_new)
    dirty_logits, dirty_toks = greedy_run(model, b_ids, args.n_new)
    print(f"  B after A -> {dirty_toks} {tok.decode(dirty_toks)!r}")
    logit_delta = float((dirty_logits - b_first_logits).abs().max())
    contaminated = dirty_toks != b_first_toks or logit_delta > 1e-3
    if contaminated:
        check(True, "stale cache changes B's output (a reset is required)",
              f"max|dlogit|={logit_delta:.4f} toks_differ={dirty_toks != b_first_toks}")
    else:
        print(f"  [UNPROVEN] these two prompts did not expose contamination "
              f"(max|dlogit|={logit_delta:.2e}). This does NOT show the cache is "
              f"safe to reuse — window slots above pos are masked, so a clean "
              f"result here can be luck. Reset anyway.")

    print("\n=== 3. B after A, with reset (does the reset fix it?) ===")
    greedy_run(model, a_ids, args.n_new)
    written = cache_reset.reset_decode_cache(
        model, init_values=None,
    )
    check(written == n_states * len(model.models),
          "reset wrote every state tensor on every rank",
          f"wrote={written} expected={n_states * len(model.models)}")
    if fresh is not None:
        after = cache_reset.snapshot_state_norms(model)
        check(after == fresh, "state norms match a fresh process after reset",
              f"n_matching={sum(1 for x, y in zip(after, fresh) if x == y)}/"
              f"{n_states}")

    clean_logits, clean_toks = greedy_run(model, b_ids, args.n_new)
    print(f"  B after A+reset -> {clean_toks} {tok.decode(clean_toks)!r}")
    delta = float((clean_logits - b_first_logits).abs().max())
    check(clean_toks == b_first_toks,
          "reset restores B's tokens exactly", f"{clean_toks} vs {b_first_toks}")
    check(delta < 1e-3, "reset restores B's logits", f"max|dlogit|={delta:.2e}")

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
