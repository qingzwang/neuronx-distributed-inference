# DeepSeek-V4-Flash on Trainium 2

A port of DeepSeek-V4-Flash (283.8 B params, 43 layers) to AWS Trainium 2 via
NeuronX Distributed. Prefill works end to end at full depth and produces
correct output on device; decode is not implemented yet.

## Status

| | |
|---|---|
| Compile, all 43 layers at TP=32 | works — 32/32 ranks PASS |
| Correct output on device | yes, see below |
| Numerics vs CPU reference | cosine 0.99996+ at 1/4/12 layers, TP=8 and TP=32 |
| Decode (`start_pos > 0`) | **not implemented** — prefill graph only |
| Performance | **unusable** — 492 s for one 128-token prefill |
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

## Layout

```
src/
  paths.py             checkpoint path resolution (DSV4_MODEL_PATH)
  compile_neuron.py    the port: XLA patches + parallel_model_trace driver
  shard_loader.py      rank-aware weight loading and sharding
  xla_ops.py           trn2-safe replacements for unsupported ops
  dequant_checkpoint.py  FP4/FP8 -> bf16
  hf_reference/        CPU reference helpers
test/
  test_xla_ops.py        top-k equivalence, + device check
  test_shard_loader.py   sharding round-trip
  test_high_tp_o_proj.py TP > o_groups O-projection vs TP=1
  test_neuron_vs_cpu.py  device logits vs CPU, same weights
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

1. **Decode is missing.** `_TraceWrapper` bakes `start_pos` in via `.item()`,
   so only the prefill graph exists. This is not just a tracing detail:
   `Compressor.forward` early-returns when `(start_pos + 1) % ratio != 0`, and
   `get_window_topk_idxs` branches three ways on `start_pos`, so one graph
   cannot cover every decode position. It needs either `ratio`-many graphs or a
   rewrite to compute-then-select, plus real KV-cache state carried across
   invocations.
2. **Performance.** One 128-token prefill takes 492 s. The static-shape MoE
   patch computes all experts and masks, doing ~42.7x the ideal FLOPs
   (2.22 TFLOP/rank/token-batch at 43 layers, TP=32). Fixing this means NxDI's
   blockwise MoE or an NKI kernel.
3. **`seq_len > 2048`** hits the `k > 32` guard in
   `xla_ops.topk_indices_unordered`; needs an NKI top-k kernel.
4. **Artifact size.** 661 GB at full depth, and ~390 s to load onto device.
