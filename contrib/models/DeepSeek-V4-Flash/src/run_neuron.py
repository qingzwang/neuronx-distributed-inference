#!/usr/bin/env python
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run a compiled DeepSeek-V4-Flash artifact on Neuron and decode its output.

This is the device-only path: it loads the traced artifact and runs it. There
is no CPU reference here — see test_neuron_vs_cpu.py for that. At full depth a
CPU comparison model would need ~568 GB of host RAM, so past a few layers the
device output has to stand on its own.

What "stand on its own" means without a reference: the checks below are the
ones that do not need a second model to compare against.

  * no NaN / Inf in the logits
  * the logit distribution is sane (finite, not collapsed to one value, a
    plausible max/entropy for a trained LM rather than a random-init one)
  * the argmax token decodes to text that is a plausible continuation

The last one is the real test. A model with mis-sharded weights or a broken
collective still produces finite logits; it does not produce a coherent next
token. `--greedy N` extends this by feeding the prediction back in, which is
where a subtly wrong graph shows up as text that degenerates after a token or
two.

Note this uses the prefill graph only. `start_pos` is baked into the trace, so
each greedy step re-runs the whole prompt (O(n^2) and capped at seq_len) rather
than using the KV cache. Correct, just not how you would serve it. See the
decode task for the real token-generation graph.

Run:
    source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
    cd contrib/models/DeepSeek-V4-Flash
    python src/run_neuron.py --artifact /mnt/data/artifacts/dsv4_tp32_L43 \
        --seq-len 128 --prompt "The capital of France is" --greedy 8
"""

import argparse
import json
import os
import sys
import time

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import paths  # noqa: E402


def load_tokenizer():
    """The checkpoint ships tokenizer.json; fall back to ids-only if absent."""
    tok_path = os.path.join(paths.model_path(), "tokenizer.json")
    if not os.path.exists(tok_path):
        return None
    try:
        from tokenizers import Tokenizer
        return Tokenizer.from_file(tok_path)
    except Exception as e:
        print(f"[warn] tokenizer unavailable ({e}); falling back to raw ids")
        return None


def describe_logits(logits):
    """Sanity stats that need no reference model."""
    x = logits.float().flatten()
    finite = torch.isfinite(x)
    stats = {
        "nan": bool(torch.isnan(x).any()),
        "inf": bool(torch.isinf(x).any()),
        "min": float(x[finite].min()) if finite.any() else float("nan"),
        "max": float(x[finite].max()) if finite.any() else float("nan"),
        "mean": float(x[finite].mean()) if finite.any() else float("nan"),
        "std": float(x[finite].std()) if finite.any() else float("nan"),
    }
    probs = torch.softmax(x, dim=-1)
    stats["entropy"] = float(-(probs * probs.clamp_min(1e-12).log()).sum())
    stats["top1_prob"] = float(probs.max())
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", required=True, help="dir with tp_*.pt")
    ap.add_argument("--seq-len", type=int, required=True,
                    help="Must match the compiled graph's seq_len.")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--prompt-ids", default=None,
                    help="Comma-separated ids, bypassing the tokenizer.")
    ap.add_argument("--greedy", type=int, default=0,
                    help="Generate this many tokens by re-running prefill with "
                         "the prediction appended and the oldest token dropped "
                         "(a sliding window, since the graph's seq_len is "
                         "static). O(n) forwards, each a full prefill.")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--fill", default=None,
                    help="Left-fill the window with this text (repeated as "
                         "needed) so the prompt ends exactly at seq_len. The "
                         "graph's input shape is static and the head reads only "
                         "the last position, so the prompt must fill it exactly; "
                         "filler is real text rather than pad tokens because "
                         "there is no attention mask to hide pads behind.")
    args = ap.parse_args()

    tok = load_tokenizer()
    if args.prompt_ids:
        ids = [int(t) for t in args.prompt_ids.split(",")]
    elif tok is not None:
        ids = tok.encode(args.prompt).ids
        print(f"[input] {args.prompt!r} -> {len(ids)} tokens {ids}")
    else:
        raise SystemExit("no tokenizer available; pass --prompt-ids")

    if args.fill and tok is not None and len(ids) < args.seq_len:
        filler = tok.encode(args.fill).ids
        if not filler:
            raise SystemExit("--fill encoded to zero tokens")
        need = args.seq_len - len(ids)
        reps = (need // len(filler)) + 1
        ids = (filler * reps)[-need:] + ids
        print(f"[input] left-filled to {len(ids)} tokens; the last "
              f"{args.seq_len - need} are the prompt")

    if len(ids) != args.seq_len:
        raise SystemExit(
            f"prompt is {len(ids)} tokens but the graph is compiled for "
            f"seq_len={args.seq_len}. This model's head emits only the last "
            f"position, so padding predicts the token after the padding rather "
            f"than after the prompt. Compile with --seq-len {len(ids)}, or give "
            f"a prompt of exactly {args.seq_len} tokens."
        )

    from neuronx_distributed.trace import parallel_model_load

    print(f"[neuron] loading {args.artifact}")
    t0 = time.perf_counter()
    model = parallel_model_load(args.artifact)
    print(f"[neuron] loaded in {time.perf_counter() - t0:.1f}s")

    def forward(token_ids):
        """Run the prompt. Its length MUST equal the graph's seq_len.

        Right-padding does not work here, and getting this wrong looks like a
        broken port rather than a broken harness. ParallelHead.get_logits does
        `F.linear(x[:, -1], w)`: the head returns logits for the *last*
        position only, shape [b, vocab], with no sequence axis to index. Pad
        to seq_len and the "next token" you read is the one following the
        padding, so a correct model dutifully continues a run of pad tokens
        (observed: 'The capital of France is' -> '########').

        Left-padding would be no better: this model has no attention mask
        input, and the compressor / window-attention index helpers are built
        from absolute positions, so leading pads shift every position.

        So the prompt has to fill the graph exactly. Compile a graph whose
        seq_len matches the prompt you want to run.
        """
        if len(token_ids) != args.seq_len:
            raise SystemExit(
                f"prompt is {len(token_ids)} tokens but the graph is compiled "
                f"for seq_len={args.seq_len}. The head only emits the last "
                f"position, so padding would predict the token after the pads, "
                f"not after the prompt. Compile with "
                f"--seq-len {len(token_ids)}, or pass a prompt of exactly "
                f"{args.seq_len} tokens (--pad-with-text repeats the prompt)."
            )
        inp = torch.tensor([token_ids] * args.batch, dtype=torch.long)
        start_pos = torch.zeros((), dtype=torch.int32)
        out = model(inp, start_pos)
        if out.dim() == 3:          # defensive: [b, s, vocab] -> last position
            out = out[:, -1, :]
        return out

    generated = []
    window = list(ids)
    for step in range(max(1, args.greedy)):
        t0 = time.perf_counter()
        logits = forward(window)
        dt = time.perf_counter() - t0

        row = logits[0].float()
        stats = describe_logits(row)
        nxt = int(row.argmax())

        if step == 0:
            print(f"\n[neuron] forward in {dt:.1f}s  shape={list(logits.shape)}")
            print(f"  NaN={stats['nan']}  Inf={stats['inf']}")
            print(f"  logits: min={stats['min']:.3f} max={stats['max']:.3f} "
                  f"mean={stats['mean']:.3f} std={stats['std']:.3f}")
            print(f"  softmax: entropy={stats['entropy']:.3f} "
                  f"top1_prob={stats['top1_prob']:.4f}")
            topv, topi = row.topk(args.top_k)
            print(f"  top-{args.top_k}:")
            for v, i in zip(topv.tolist(), topi.tolist()):
                piece = tok.decode([i]) if tok else ""
                print(f"    {i:>7}  {v:8.3f}  {piece!r}")
            if stats["nan"] or stats["inf"]:
                raise SystemExit("[FAIL] non-finite logits")
            if stats["std"] < 1e-3:
                raise SystemExit("[FAIL] logits collapsed to a constant")

        generated.append(nxt)
        # Slide the window: the graph's input shape is fixed, so appending the
        # new token means dropping the oldest one.
        window = window[1:] + [nxt]
        if args.greedy:
            piece = tok.decode([nxt]) if tok else str(nxt)
            print(f"  step {step + 1}: {nxt} {piece!r}  ({dt:.1f}s)")

    if args.greedy and tok is not None:
        print(f"\n[prompt]     {tok.decode(ids)!r}")
        print(f"[generated]  {tok.decode(generated)!r}")
        print(f"[full]       {tok.decode(ids + generated)!r}")
    print("\n[ok] device inference completed")


if __name__ == "__main__":
    main()
