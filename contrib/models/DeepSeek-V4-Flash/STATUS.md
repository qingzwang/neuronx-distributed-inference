# DeepSeek-V4-Flash on NxDI — where this stands

Work stopped here by decision on 2026-08-05. This file is the honest state of the
port, so nobody has to re-derive it from the log files.

## What works, measured

| thing | number | how |
|---|---|---|
| 43-layer decode | 94 ms/token | standalone artifact |
| 43-layer prefill | 0.85 s TTFT, 128 tokens | standalone artifact |
| GSM8K | 97.3% (150 problems) | decode artifact, cache carry-over |
| joint prefill+decode, 5 layers | cosine 0.999994 vs all-decode, 20/20 top-20 | shared cache on device |
| prompt ingest, 5 layers | 0.29 s joint vs 1.55 s all-decode | 5.3x |
| joint at 43 layers | fits and runs: 20.600 GB/rank, TTFT 1.00 s, TPOT 96 ms | **output wrong, see below** |

The two-artifact configuration (prefill graph + decode graph, private caches) is the
one with validated accuracy. Note its TTFT and TPOT come from **different
artifacts** and cannot be added together.

## The open failure

The 43-layer joint graph runs but produces garbage: `'leirinerineesterday'`,
logits absmax 170.056, where the trusted standalone 43-layer prefill artifact gives
`11111 25.562 ' Paris'` on the same prompt. At 5 layers the same code path is
near-exact (cosine 0.999994), so this is depth- or config-dependent, not a wiring
error.

What was ruled out:

* **prompt** — ids are byte-identical between the good and bad runs (checked in the
  saved logit files).
* **the 42 unshipped weights** — they are exactly 21 ratio-4 layers x
  {`indexer.weights_proj`, `indexer.wq_b`}, which neither graph binds at a 128-token
  prefill. Dropping them cannot change the result.
* **HBM/OOM** — fixed, see below. `initialize()` completes in 24.7 s.

What was NOT ruled out, and is the next thing to test:

* **`seq_len`** — the bad run used 136, the trusted baseline 256, and cache geometry
  genuinely differs: ratio-4 layers get `attn_cache_len` 162 vs 192, ratio-128 get
  129 vs 130. A run at `seq_len=256` was compiling when work stopped; its log is
  `/mnt/nvme/logs/dev43f.log` and the workdir `/mnt/nvme/tmp/joint43f_ws` is intact,
  so it can be resumed.
* **silent miscompiles** at 43 layers under `neuronx-cc 2.26.6360.0`. MiMo-V2.5-Pro
  reports these for this compiler version and says they show up above ~200-token
  prompts. All accuracy evidence here uses <200-token prompts, so this is
  unmeasured either way.

## The HBM lesson, since it cost the most time

The 43-layer joint graph appeared not to fit. My first diagnosis — two NEFFs plus
prefill activations against 1.85 GB of headroom — was wrong, and I acted on it by
shrinking `prefill_len`, which is architecturally pinned at >= 128 anyway (below
that a ratio-128 layer gets a zero-width compressed tensor and XLA lowering fails
on `aten::as_strided`).

The runtime already had the answer. It writes `/tmp/nrt_mem_log_device_*.csv` and
prints a usage table showing `Model Code 3.374MB`, `Scratchpad 0.000B`,
`Tensors 23.892GB`. Both NEFFs are negligible; the graph had not executed so there
were no activations; it was entirely weights.

Real cause: weights were shipped under **both** naming forms (dotted for the
runtime, arrow-separated for the HLO) on the assumption that shared host storage
made this free. `_parallel_load` allocates **per dict key** and does not dedup by
storage — measured on device: 4 keys over one 2 GB host tensor produce 4 device
copies. So 2231 tensors (20.610 GB) were requested as 4462 keys, about 41 GB against
a 24 GB core.

Fix: ship only the names a graph binds, converted to dotted form (runtime binding in
`spmd.initialize` is by dotted name; arrows alone raise `Missing weight tensor with
key inner.head.weight`). 2189 keys, 20.600 GB/rank, fits.

**Method note for next time:** read the runtime's own dump before theorising about
memory. The existing reconciliation had been printing `extra in my dict: 2273` the
whole time.

## Assessment

The architecture-specific work is done and gated on CPU numerics: ragged per-layer
state, the init-independent compression ring, sliding window + compressed tail, the
sparse indexer kept rather than disabled as GLM-5.2 does. The framework
integration also works — two graphs against one shared cache, which is what the
rewrite existed to prove, demonstrated at 5 layers.

What is not established is full-depth correctness through the joint path. Until
that `seq_len=256` comparison runs, the joint path should be treated as unvalidated
at 43 layers, and the two-artifact configuration is the one to use.
