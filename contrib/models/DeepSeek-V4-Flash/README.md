# DeepSeek-V4-Flash on Trainium 2

A port of DeepSeek-V4-Flash (283.8 B params, 43 layers) to AWS Trainium 2 via
NeuronX Distributed. Prefill works end to end at full depth and produces correct
output on device. Decode works with a device-resident KV cache — validated at
5 layers, not yet compiled at full depth.

## Status

| | |
|---|---|
| Compile, all 43 layers at TP=32 | works — 32/32 ranks PASS |
| Correct output on device | yes, see below |
| Numerics vs CPU reference | cosine 0.99996+ at 1/4/12 layers, TP=8 and TP=32 |
| Decode (`start_pos > 0`) | works at 5 layers, TP=32: 20 ms/token, cache on device |
| Decode at 43 layers | works — **94 ms/token** steady state, correct output |
| GSM8K, 150 problems | **97.3%** EM, TTFT median 7.4 s, TPOT 93.4 ms (see below) |
| Performance | decode usable; prefill **unusable** — 492 s for one 128-token prefill |
| `seq_len > 2048` | unsupported, raises (needs an NKI top-k kernel) |

Device output for `"It is well known that the capital city of France is"`
(43 layers, TP=32, seq_len 128, real weights):

```
  11111   25.562   ' Paris'     <- top-1, p=0.79
  16235   21.913   ' Rome'
    260   21.781   ' a'
     28   21.574   ':'
   6693   21.231   ' London'
```

The distractors being other capitals is the signal that matters: mis-sharded
weights or a broken collective still yield finite logits, but not that
neighbourhood.

## Setup

```bash
source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
export DSV4_MODEL_PATH=/path/to/DeepSeek-V4-Flash   # the HF checkpoint root
```

The checkpoint's own `inference/model.py` is the reference implementation and
is imported directly — this port patches it at runtime rather than forking it,
so a checkpoint update does not silently diverge from a copied model file. All
paths resolve through `src/paths.py` from that one env var.

Requires a trn2.48xlarge: 16 devices x 4 cores, `logical-neuroncore-config: 2`,
**24 GB per logical core**. At 283.8 B params in bf16 that budget is what
forces TP >= 32 (TP=8 would need 70.9 GB/core).

## Compile and run

```bash
# full depth, ~55 min, 661 GB artifact, ~810 GB peak host RAM
python src/compile_neuron.py --tp 32 --n-layers 43 --seq-len 128 \
    --load-weights --out-dir /mnt/data/artifacts/dsv4_tp32_L43 \
    --compiler-workdir /mnt/data/tmp/ws

# run it
python src/run_neuron.py --artifact /mnt/data/artifacts/dsv4_tp32_L43 \
    --seq-len 128 --greedy 12 \
    --prompt "It is well known that the capital city of France is" \
    --fill "Artificial intelligence research has a long history. "
```

Start smaller when iterating: `--n-layers 5 --tp 32` compiles in ~6 min and
covers every distinct layer type (see below), so almost every bug shows up
there.

### Decode

```bash
# --mode decode: one token per call, start_pos a runtime input.
# --seq-len sets max_seq_len (how far generation can run), not the input width.
python src/compile_neuron.py --mode decode --tp 32 --n-layers 5 --seq-len 256 \
    --load-weights --out /mnt/data/artifacts/dsv4_decode_tp32_L5 \
    --compiler-workdir /mnt/data/tmp/decode_ws

python src/run_neuron.py --mode decode \
    --artifact /mnt/data/artifacts/dsv4_decode_tp32_L5 \
    --seq-len 256 --prompt "The capital of France is" --greedy 8
```

Measured at 5 layers, TP=32: 516 s to compile, 79 GB artifact, **20 ms/token**
against 7.4 s/token for the same work through the prefill graph.

#### At full depth

43 layers, TP=32, `--seq-len 512`, real weights. Compile: 5932 s, 663 GB
artifact, 32/32 ranks PASS. Load onto device: 404 s. Device memory sits at
22.15 GB of tensors per core (32 cores, 720 GB total), which is the whole reason
TP >= 32 — the per-core budget is 24 GB.

```
[input] 'It is well known that the capital city of France is' -> 11 tokens
  11111   25.829   ' Paris'     <- top-1, p=0.785
   2619   22.343   ' **'
    295   21.679   ' in'
   7840   21.667   ' located'
    680   21.403   ' {'
[full]  'It is well known that the capital city of France is Paris.",\n
         "label": 0\n}<｜end▁of▁file｜>\n'
```

Two things worth reading off that. The prefill graph scores `' Paris'` at
25.562 and this one at 25.829 — two separate graphs, separate device buffers,
independently sharded weights, same answer, so the decode rewrite's closed-form
index math agrees with the validated prefill path at full depth. And the
continuation stays structurally coherent for all 12 steps, closing its JSON and
emitting a real EOS rather than degenerating, which is the failure mode a subtly
wrong cache write produces.

**The first call costs ~1920 s; every later call costs 94 ms.** That is a
one-time warmup, not per-token cost, and it dominates any average taken over a
short run: this 11-token ingest reported 175 s/token, which is the warmup
divided by 11 rather than a real rate. Steady-state decode is 94 ms/token,
measured over 11 consecutive generation steps (92-95 ms, no drift). Budget the
warmup once per process and ignore it in throughput numbers.

Decode ingests the prompt one token at a time rather than inheriting prefill's
cache. Prefill and decode are separate traced artifacts with separate device
buffers, and NxD offers no way to hand one graph's aliased state to another. The
numbers are the same either way (`test_decode_vs_reference.py --no-prefill`
checks exactly this), it just costs `prompt_len` cheap calls instead of one big
one.

## GSM8K

```bash
python src/run_gsm8k.py --artifact /mnt/data/artifacts/dsv4_decode_tp32_L43 \
    --seq-len 512 --n 150 --max-new 400 --warmup --out /tmp/gsm8k.jsonl
```

150 problems from `main/test`, 0-shot, greedy, chat-mode encoding:

| | |
|---|---|
| accuracy (EM) | **97.3%** (146/150) — 98.6% over the 148 that finished |
| TTFT | median **7.4 s**, mean 7.9, p95 11.9 (excludes problem 1) |
| TPOT | mean **93.4 ms**, median 93.4, range 92.9-94.5 |
| stopped on EOS | 148/150 |
| hit the 400-token budget | 2/150 — unscoreable, not wrong |
| mean prompt / generation | 84 / 127 tokens |

**TTFT is exactly `prompt_len x 94 ms`.** Measured per problem it tracks prompt
length to within a millisecond per token, which is the missing chunked prefill
stated as a number: ingest is `prompt_len` sequential single-token calls, each
paying a full forward, so there is no batching win over generation. A real
prefill path would collapse this to roughly one forward.

Reported TTFT excludes problem 1. In a fresh process the first call carries the
warmup (`--warmup` moves most of it, and what leaks past still made problem 1
cost 613 s against a 7.4 s median). Mixing that into the mean inflates it to
11.9 s and describes nothing real.

The four misses are worth naming, because none is a port bug:

* two hit the 400-token budget mid-derivation (`--max-new` is the limit, not the
  model)
* one answered `400/11` where gold is `36` — correct to 36.36, and the
  last-number rule then scored the `11` of the fraction
* one is the standard "10 times more than 60" ambiguity (600 vs 660)

Against the model card's 90.8 (8-shot, Base), 97.3 here is not the same
measurement: this is the *instruct* checkpoint, 0-shot, through its own chat
encoding, on 150 of 1319 problems. Treat it as "the port reasons correctly at
full depth", not as a reproduction of the published number.

### The caveat that matters

Problems are **not** independent. The KV cache lives on the device and cannot be
reset from the host, so problem N+1 begins with problem N's compressed KV still
resident. `test_cache_reset.py` measures the effect directly: prompt B after
prompt A diverges from B on a fresh cache, max|dlogit| = 2.36, with different
tokens.

Three host-side write paths were tried — `fill_`, `data.copy_`, and replacing the
`nn.Parameter` outright — and all three leave the graph's behaviour unchanged.
The third is the diagnostic one: the new value reads back correctly from
`named_parameters()` and the model still emits the pre-reset continuation, so
`forward_v2` binds its device buffers once and the Python parameters are only the
seed for that binding. The fix is a `reset` input threaded into the traced graph,
i.e. a recompile; `cache_reset.py` documents this rather than pretending to solve
it.

So 97.3% is measured under carry-over. Since window slots above the current
position are masked out and each problem restarts at position 0, most of the
stale state is unreachable — but "most" is not "all", and the honest statement is
that the number is contaminated by an amount bounded by that 2.36 logit delta.

## Layout

```
src/
  paths.py             checkpoint path resolution (DSV4_MODEL_PATH)
  compile_neuron.py    the port: XLA patches + parallel_model_trace driver
  decode_patches.py    the decode path: tensor start_pos + aliased KV state
  run_neuron.py        device inference, --mode prefill | decode
  run_gsm8k.py         GSM8K accuracy + TTFT/TPOT
  cache_reset.py       KV-cache reset attempt (does not work; see its header)
  shard_loader.py      rank-aware weight loading and sharding
  xla_ops.py           trn2-safe replacements for unsupported ops
  dequant_checkpoint.py  FP4/FP8 -> bf16
  hf_reference/        CPU reference helpers
test/
  test_xla_ops.py        top-k equivalence, + device check
  test_shard_loader.py   sharding round-trip
  test_high_tp_o_proj.py TP > o_groups O-projection vs TP=1
  test_neuron_vs_cpu.py  device logits vs CPU, same weights
  test_decode_vs_reference.py   decode rewrite vs HF's decode, on CPU
  test_decode_neuron_vs_cpu.py  device decode vs CPU, incl. the aliased caches
  test_cache_reset.py    measures the cross-prompt cache carry-over
  test/spike/            exploratory scripts kept for reference
```

## What the port had to change, and why

Each of these is a place where the reference model does something neuronx-cc or
torch-xla rejects. They live in `compile_neuron.apply_xla_patches()`.

**`ParallelEmbedding` bool-mask assignment.** `x[mask] = 0` lowers via
`nonzero()`, whose output size is **frozen at trace time**. `parallel_model_trace`
traces with all-zero ids, so rank 0 records 0 masked elements and the others
record all of them; real ids at runtime then read out of bounds
(`status=1006 Execution Out-Of-Bounds Memory Access`). Rewritten with
`torch.where`.

**`torch.topk`.** Lowers to HLO `sort`:
`[NCC_EVRF029] Operation sort is not supported on trn2`. `src/xla_ops.py`
replaces it with a k-step argmax-and-mask loop, exact for the gate's k=6.
Above k=32 it raises rather than emit a graph that takes minutes to compile —
that is the `seq_len > 2048` limit in the table.

**`as_strided` is unimplemented on the XLA backend.** This rules out
`.repeat()`, `.expand()`, *and* constructing any zero-width tensor. Broadcast-add
a zeros tensor instead. The zero-width case is a real config constraint
(`seqlen // ratio == 0` when `seq_len < max(compress_ratios)`), so
`_build_model_args` rejects it up front with a message naming the fix.

**CPU tensors leaking into the graph.** HF's index helpers build masks with a
bare `torch.arange` (fine when a global CUDA default device is set) and cache
them with an `lru_cache` whose key omits the device. Both are made
device-aware; the cache is dropped.

**TP > o_groups.** HF caps parallelism at `o_groups` (8):
`n_local_groups = o_groups // world_size` hits 0 at TP>=16 and the
O-projection silently contributes nothing. Since 24 GB/core forces TP>=32, the
split has to move to the contraction dim past `o_groups`, with `wo_b` holding
its group's whole `o_lora_rank` block (replicated across group-mates — splitting
it drops the contraction's cross terms, measured at 66-83% relative error).
`test_high_tp_o_proj.py` checks TP=16/32/64 against TP=1.

**Sharding is done by hand.** NxD's `checkpoint_loader_callable` cannot shard
this model: `trace.shard_children()` early-returns unless a module is an
`isinstance` of NxD's own parallel classes, and HF's `model.py` declares its
*own* classes with the same names. NxD matches none of them and shards nothing,
**silently**. See the header of `shard_loader.py`.

## What decode had to change, and why

Prefill can bake `start_pos = 0` into the trace. Decode cannot: a graph per
position is not an option, so every Python-int use of `start_pos` becomes a
tensor op. These live in `src/decode_patches.py`, on top of the XLA patches.

**Variable-length index lists become fixed-length with `-1` padding.** HF's
`get_compress_topk_idxs` returns `arange(0, (p+1) // ratio)`, whose *length*
depends on the position — impossible in a graph. Emit all `max_comp` entries and
mark the unwritten tail `-1`, which `sparse_attn` already treats as masked.

**HF's three-way window branch collapses to one expression.** Ring slot `j`
holds absolute position `p - ((p - j) mod win)`, which is real iff `j <= p`, so
`where(arange(win) <= p, arange(win), -1)` covers every case. For `p >= win-1`
this is a *permutation* of what HF emits, not the same order — which is fine
only because `sparse_attn` gathers the listed slots and softmaxes over them, so
just the set and the mask matter. Verified against both HF branches for
`p = 1..19`.

**Early returns become masked writes.** `Compressor.forward` returns early when
`(p+1) % ratio != 0`. An early return changes the graph, so instead always
compute and always write, but write back `where(should, new, old)`.

**State must be `nn.Parameter`, not `register_buffer`.** This is the trap that
cost the most. NxD resolves output aliases by scanning `named_parameters()` and
matching `.data_ptr()` (`torch_neuronx/xla_impl/hlo_conversion.py`). Buffers are
never scanned, so a buffer key matches nothing and the alias is **silently
dropped** — the graph still compiles and runs, and resets its cache every call,
which reads as a model that forgot its context rather than as a config error.

**Alias keys must stay CPU tensors.** NxD pickles the alias dict back to the
parent, which does `initial_states = tuple(aliases.keys())` and rebuilds each
key. An XLA tensor key makes that `nrt_init` a device the parent does not own:
`[NRT_FAILURE] status_code=1`, then `BrokenProcessPool`. Note `.cpu()` on a CPU
tensor returns *self*, so it does not undo a `module._apply()` that already
moved the tensor — the state has to be a Parameter from the start.

**`index_copy` demands a Long index on XLA.** `start_pos` arrives as int32
because that is what `torch.jit.trace` takes as a scalar input:
`Check failed: index->dtype() == at::ScalarType::Long (Int vs. Long)`.

**HF's view aliasing has to become an explicit redirect.** HF does
`self.compressor.kv_cache = self.kv_cache[:, win:]`, so a compressor write lands
in the enclosing layer's cache. `index_copy` returns a new tensor rather than
mutating, so a write through the view would be invisible to the owner.
`StateSink` replaces the view with an owner/offset redirect and splices writes
back.

**`freqs_cis` must be wired per forward, not at patch time.** Only `Attention`
registers it as a buffer; `Compressor` and `Indexer` initialise theirs to `None`
and HF fills them in on the first `Attention.forward`. Since `_apply` moves
buffers but not plain attributes, wiring before the model reaches the device
would pin the CPU copy into the graph as a constant.

## Two traps worth knowing before you touch this

**Do not let weights be inlined into the NEFF.** NxD's
`parallel_model_trace` defaults to `inline_weights_to_neff=True`, which turns
every weight into an HLO constant that the compiler materializes and copies, in
all TP ranks at once. NxDI's own production path passes `False`
(`neuronx_distributed_inference/models/model_wrapper.py`). Measured at TP=32:

| | inlined | separated |
|---|---|---|
| peak host RAM | ~41 GB/rank | 4.3 GB/rank |
| compiler workdir | 336 GB (12 layers) | 240 MB (4 layers) |
| compile time | 3284 s (12 layers) | 366 s (4 layers) |
| artifact load | 523 s (12 layers) | 33 s (4 layers) |

With inlining on, 43 layers at TP=32 needs ~1841 GB against a 1999 GB box, and
because trn2.48xlarge has **no swap** the kernel livelocks instead of
OOM-killing: SSH dies and the instance needs a power cycle. `compile_neuron.py`
now runs a host-RAM preflight that refuses to start such a run (predicted
721 GB vs 812 GB actual for the full model, so the model is roughly right).

**The head returns only the last position.** `ParallelHead.get_logits` does
`F.linear(x[:, -1], w)` — the output is `[batch, vocab]`, with no sequence axis.
Right-padding a short prompt to `seq_len` therefore predicts the token after the
*padding*: `"The capital of France is"` padded to 128 comes back as `########`,
which reads exactly like a broken port but is a harness bug. Left-padding is no
better — there is no attention-mask input, and the compressor and window-index
helpers are built from absolute positions. The prompt must fill the window
exactly; `run_neuron.py --fill` left-fills with real text to make that easy.

## Layer types

Only three distinct `compress_ratio` values exist across all 43 layers, and
every combination of them with the two gate types appears within the first 5.
`compress_ratios` is `[0, 0, 4, 128, 4, 128, ...]` and `n_hash_layers=3`, so:

| layer | `compress_ratio` | gate | Indexer |
|---|---|---|---|
| 0, 1 | 0 | hash (`tid2eid`) | no |
| 2 | 4 | hash | yes |
| 3 | 128 | score | no |
| 4 | 4 | score | yes |
| 5+ | alternates 128 / 4 | score | on ratio-4 layers |

`--n-layers 4` covers everything except the score-gated Indexer layer, and
`--n-layers 5` covers all of it; past that, depth only adds scale.

## Known gaps

1. **Decode's serving path is still minimal.** The graph now runs at full depth
   (94 ms/token, above), but the prompt-ingest path is `prompt_len` separate
   calls rather than a chunked prefill, so a long prompt pays one call per
   token. Batch > 1 is untraced. There is also no stopping criterion, sampling,
   or MTP head — `run_neuron.py --mode decode` is greedy only, and it does not
   stop on the EOS it correctly emits.
2. **The KV cache cannot be reset without recompiling.** Serving independent
   requests from one loaded artifact is therefore not correct today: every
   request inherits the last one's compressed KV (max|dlogit| = 2.36, measured).
   No host-side write path reaches the runtime's buffers — see `cache_reset.py`
   for the three that were tried and why the third one's readback passing is the
   informative part. Needs a `reset` input in the traced graph.
3. **Prefill performance.** One 128-token prefill takes 492 s. The static-shape
   MoE patch computes all experts and masks, doing ~42.7x the ideal FLOPs
   (2.22 TFLOP/rank/token-batch at 43 layers, TP=32). Fixing this means NxDI's
   blockwise MoE or an NKI kernel. Decode is far less exposed to this — it runs
   one token through the same masked-MoE at 94 ms — so this is a prefill
   problem first.
4. **`seq_len > 2048`** hits the `k > 32` guard in
   `xla_ops.topk_indices_unordered`; needs an NKI top-k kernel.
5. **Artifact size and startup.** 663 GB at full depth, ~400 s to load onto
   device, plus a ~1920 s first-call warmup before the first token comes back.
   That is ~39 min from process start to first token, against 94 ms for every
   token after it. Unblocking this is what stands between the current state and
   anything servable; it was not investigated here beyond measuring it.
