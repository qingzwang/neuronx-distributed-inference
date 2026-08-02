#!/usr/bin/env python
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""GSM8K accuracy + TTFT/TPOT on the device-resident decode graph.

What the three numbers mean here, since the decode graph makes them less
obvious than usual:

  accuracy  exact match on the final number. GSM8K answers are the integer
            after `####`; a generation is correct if the last number it
            produces equals it. "Last number" rather than "a number appearing
            anywhere" — the latter scores a model correct for reciting the
            question's operands.
  TTFT      time to first token: the whole prompt ingest plus the first
            generated token. This graph has no chunked prefill, so ingest is
            `prompt_len` separate single-token calls, and TTFT grows linearly
            with prompt length instead of being one batched matmul. That is a
            property of the port, not of the model, and it is why TTFT here is
            seconds rather than milliseconds.
  TPOT      time per output token, steady state: the mean over generated
            tokens *after* the first. Reported separately from TTFT because the
            two differ by ~2 orders of magnitude on this graph.

The first call in a fresh process costs ~1920 s of one-time warmup (compiler
caches, collective setup). It is excluded from every reported statistic and
reported on its own — folding it in would make the first problem look 20x
slower than the rest. `--warmup` does that deliberately before the timed run.

CAVEAT: problems are not independent. The device-resident KV cache cannot be
reset from the host — see the status note at the top of cache_reset.py, and
test_cache_reset.py for the measurement — so problem N+1 starts with problem N's
compressed KV still in the cache. The window slots are largely masked by
position (`window_topk_idxs` only reads slot j when j <= pos), but the compressed
tail and the indexer's cache are not, and the measured effect on a short prompt
is max|dlogit| = 2.36 with different sampled tokens.

The accuracy below is therefore measured under carry-over, not on clean state.
It is reported as-is rather than corrected, because the fix is a recompile with a
reset input, not something the harness can paper over. `--reset-attempt` runs the
(ineffective) host-side reset anyway, for A/B measurement.

Run:
    python src/run_gsm8k.py --artifact /path/to/dsv4_decode_tp32_L43 \
        --seq-len 512 --n 20 --max-new 320
"""

import argparse
import json
import os
import re
import statistics
import sys
import time

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import cache_reset  # noqa: E402
import paths  # noqa: E402

# GSM8K's reference answer is the integer after ####.
_GOLD_RE = re.compile(r"####\s*(-?[\d,]+)")
# Any number in the generation, including negatives, decimals and 1,234 commas.
_NUM_RE = re.compile(r"-?\d[\d,]*\.?\d*")


def gold_answer(answer_text):
    m = _GOLD_RE.search(answer_text)
    if not m:
        return None
    return _norm_num(m.group(1))


def _norm_num(s):
    """'1,234' / '18.0' / '18' -> a comparable canonical form."""
    s = s.replace(",", "").rstrip(".")
    try:
        v = float(s)
    except ValueError:
        return None
    # Integers compare as integers so 18.0 == 18.
    return str(int(v)) if v == int(v) else str(v)


def extract_pred(text):
    """The last number in the generation, GSM8K's standard convention.

    Last rather than first: a chain of thought states intermediate results
    before the final one, and the conclusion is what is being graded.
    """
    nums = _NUM_RE.findall(text)
    for tok in reversed(nums):
        v = _norm_num(tok)
        if v is not None:
            return v
    return None


def build_prompt(question, encoder, thinking_mode):
    """Chat-encode one problem.

    This checkpoint is the *instruct* model, so it wants its own conversation
    encoding (`inference/`-adjacent `encoding/encoding_dsv4.py`) rather than the
    few-shot 'Question:/Answer:' continuation format the Base model is
    benchmarked with. Using the base-style prompt on the instruct weights
    measures the wrong thing, so the encoder is not optional here.

    The instruction is explicit about the answer format because accuracy is
    scored on the last number: without it the model often ends on a restatement
    or a unit, and a correct solve gets marked wrong for formatting.
    """
    content = (
        question.strip()
        + "\n\nSolve this step by step, then give the final numeric answer "
        + "on its own line as: #### <number>"
    )
    return encoder([{"role": "user", "content": content}],
                   thinking_mode=thinking_mode)


def load_encoder():
    """Import the checkpoint's own prompt encoder from `encoding/`."""
    enc_dir = os.path.join(paths.model_path(), "encoding")
    if enc_dir not in sys.path:
        sys.path.insert(0, enc_dir)
    from encoding_dsv4 import encode_messages
    return encode_messages


class DecodeRunner:
    """One-token-per-call greedy generation against the aliased-cache graph."""

    def __init__(self, model, eos_ids, batch=1):
        self.model = model
        self.eos_ids = set(eos_ids)
        self.batch = batch

    def _step(self, token, pos):
        inp = torch.tensor([[token]] * self.batch, dtype=torch.long)
        out = self.model(inp, torch.tensor(pos, dtype=torch.int32))
        logits = out[0] if isinstance(out, (tuple, list)) else out
        if logits.dim() == 3:
            logits = logits[:, -1, :]
        return logits

    def generate(self, prompt_ids, max_new, max_seq_len):
        """Ingest then greedily generate. Returns (tokens, timing dict).

        Timing separates the ingest+first-token cost (TTFT) from the mean of the
        remaining steps (TPOT), because on this graph they differ by orders of
        magnitude and a single mean would describe neither.
        """
        t_start = time.perf_counter()
        logits = None
        for p, tid in enumerate(prompt_ids):
            logits = self._step(tid, p)
        ingest_s = time.perf_counter() - t_start

        toks, step_times = [], []
        pos = len(prompt_ids)
        hit_eos = False
        # Never write past the compiled cache: max_seq_len bounds both the KV
        # ring and the RoPE table.
        budget = min(max_new, max_seq_len - len(prompt_ids))
        for _ in range(budget):
            nxt = int(logits[0].float().argmax())
            if nxt in self.eos_ids:
                hit_eos = True
                break
            toks.append(nxt)
            t0 = time.perf_counter()
            logits = self._step(nxt, pos)
            step_times.append(time.perf_counter() - t0)
            pos += 1

        # TTFT = ingest + the step that produced the first token. The argmax
        # that reads token 0 comes off the last ingest call, so ingest already
        # contains it; the first *generation* step is step_times[0].
        ttft = ingest_s
        # Steady state excludes the first generated step, which still carries
        # some first-touch cost on this graph.
        steady = step_times[1:] if len(step_times) > 1 else step_times
        return toks, {
            "ttft_s": ttft,
            "tpot_ms": (statistics.mean(steady) * 1000) if steady else float("nan"),
            "n_new": len(toks),
            "hit_eos": hit_eos,
            "truncated": not hit_eos and len(toks) >= budget,
            "prompt_len": len(prompt_ids),
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", required=True)
    ap.add_argument("--seq-len", type=int, required=True,
                    help="The compiled max_seq_len. Prompt + generation must "
                         "fit inside it; the KV ring and RoPE table are both "
                         "sized to it.")
    ap.add_argument("--n", type=int, default=20,
                    help="Number of GSM8K test problems.")
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--max-new", type=int, default=320)
    ap.add_argument("--thinking-mode", default="chat",
                    choices=["chat", "thinking"],
                    help="chat closes the think block immediately (fast, "
                         "answer-only). thinking lets the model reason first, "
                         "which needs a much larger --max-new and seq_len.")
    ap.add_argument("--data", default="/mnt/nvme/data/gsm8k/main/"
                                     "test-00000-of-00001.parquet")
    ap.add_argument("--out", default=None, help="Write per-problem JSONL here.")
    ap.add_argument("--warmup", action="store_true",
                    help="Burn the ~1920 s first-call warmup before timing, so "
                         "the reported TTFT/TPOT describe steady state rather "
                         "than one-time graph setup.")
    ap.add_argument("--reset-attempt", action="store_true",
                    help="Call the host-side KV reset between problems. It does "
                         "NOT work (see cache_reset.py) — this exists to A/B "
                         "that, not because it makes the run clean.")
    args = ap.parse_args()

    import pandas as pd
    df = pd.read_parquet(args.data)
    rows = df.iloc[args.offset:args.offset + args.n]
    print(f"[data] {len(rows)} problems from {args.data} (offset {args.offset})")

    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(os.path.join(paths.model_path(), "tokenizer.json"))
    encoder = load_encoder()

    # EOS: generation_config says id 1, and the chat encoding closes an
    # assistant turn with <｜end▁of▁sentence｜>. Take both, plus the
    # end-of-file token the model emitted in earlier runs, so a finished answer
    # actually stops instead of running to the token budget.
    eos_ids = {1}
    for t in ("<｜end▁of▁sentence｜>", "<｜end▁of▁file｜>"):
        tid = tok.token_to_id(t)
        if tid is not None:
            eos_ids.add(tid)
    print(f"[tok] eos ids: {sorted(eos_ids)}")

    from neuronx_distributed.trace import parallel_model_load
    print(f"[neuron] loading {args.artifact}")
    t0 = time.perf_counter()
    model = parallel_model_load(args.artifact)
    print(f"[neuron] loaded in {time.perf_counter() - t0:.1f}s", flush=True)

    runner = DecodeRunner(model, eos_ids)

    warmup_s = None
    if args.warmup:
        print("[warmup] burning the one-time first-call cost ...", flush=True)
        t0 = time.perf_counter()
        runner._step(int(tok.encode("The").ids[0]), 0)
        warmup_s = time.perf_counter() - t0
        print(f"[warmup] first call took {warmup_s:.1f}s "
              f"(excluded from all statistics below)", flush=True)
        # No reset here: it does not work (cache_reset.py). The warmup writes one
        # token at position 0, which the first real problem then overwrites from
        # position 0 anyway, so this particular carry-over is the benign case.

    results = []
    n_correct = 0
    for i, (_, row) in enumerate(rows.iterrows()):
        if args.reset_attempt:
            # Known ineffective on a loaded artifact; see cache_reset.py.
            cache_reset.reset_decode_cache(model, verify=False)

        prompt = build_prompt(row["question"], encoder, args.thinking_mode)
        pids = tok.encode(prompt).ids
        if len(pids) >= args.seq_len:
            print(f"[skip] problem {i}: prompt {len(pids)} >= seq_len "
                  f"{args.seq_len}")
            continue

        toks, timing = runner.generate(pids, args.max_new, args.seq_len)
        text = tok.decode(toks)
        gold = gold_answer(row["answer"])
        pred = extract_pred(text)
        ok = pred is not None and gold is not None and pred == gold
        n_correct += int(ok)

        rec = {
            "idx": args.offset + i, "gold": gold, "pred": pred, "correct": ok,
            "generation": text, **timing,
        }
        results.append(rec)
        print(f"[{i + 1}/{len(rows)}] {'OK ' if ok else 'BAD'} "
              f"gold={gold} pred={pred} "
              f"ttft={timing['ttft_s']:.1f}s tpot={timing['tpot_ms']:.0f}ms "
              f"new={timing['n_new']}{' EOS' if timing['hit_eos'] else ''}"
              f"{' TRUNC' if timing['truncated'] else ''}", flush=True)

    if not results:
        raise SystemExit("no problems ran")

    n = len(results)
    ttfts = [r["ttft_s"] for r in results]
    tpots = [r["tpot_ms"] for r in results if r["tpot_ms"] == r["tpot_ms"]]
    acc = n_correct / n

    print("\n" + "=" * 62)
    print(f"GSM8K  n={n}  thinking_mode={args.thinking_mode}")
    print(f"  accuracy      {acc * 100:.1f}%  ({n_correct}/{n})")
    print(f"  TTFT   mean   {statistics.mean(ttfts):.1f} s"
          f"   median {statistics.median(ttfts):.1f} s"
          f"   min {min(ttfts):.1f}  max {max(ttfts):.1f}")
    if tpots:
        print(f"  TPOT   mean   {statistics.mean(tpots):.1f} ms"
              f"  median {statistics.median(tpots):.1f} ms"
              f"  min {min(tpots):.1f}  max {max(tpots):.1f}")
    n_trunc = sum(r["truncated"] for r in results)
    n_eos = sum(r["hit_eos"] for r in results)
    print(f"  stopped on EOS   {n_eos}/{n}")
    print(f"  hit token budget {n_trunc}/{n}"
          + ("   <- these cannot be scored fairly" if n_trunc else ""))
    print(f"  mean new tokens  {statistics.mean(r['n_new'] for r in results):.0f}")
    print(f"  mean prompt len  {statistics.mean(r['prompt_len'] for r in results):.0f}")
    if warmup_s:
        print(f"  (one-time warmup {warmup_s:.0f} s, excluded above)")
    print("  NOTE: problems share one device-resident KV cache — it cannot be "
          "reset\n        from the host, so this is measured under carry-over. "
          "See cache_reset.py.")

    if args.out:
        with open(args.out, "w") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")
        print(f"\n[out] wrote {args.out}")


if __name__ == "__main__":
    main()
