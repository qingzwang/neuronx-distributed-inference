# Qwen3.5-2B VL Benchmark (Neuron)

End-to-end vision-language pipeline on AWS Trainium 2 for the task
**"What is in this image?"** over `0.png` (a Bulbasaur illustration).

Both the vision encoder and the text decoder are compiled to Neuron.
This report covers the **multi-bucket recurrent** DeltaNet CTE kernel
(`USE_NKI=1` + `enable_bucketing=True`), which is the configuration that
solves both the NaN problem and the constant-time-CTE problem.

## TTFT vs L40S (2026-05-11 update)

L40S transformers reference: **~180 ms TTFT for 512x512 image input**.

### Short-prompt regime — trn2 wins by 3x

After adding tiny CTE buckets [32, 64], short-prompt TTFT drops well
below 180 ms:

| Scenario | Tokens | trn2.48xlarge TP=4 LNC=2 | L40S `transformers` | Speedup |
|---|---:|---:|---:|---:|
| Text-only chat | 19 | **52 ms** | ~180 ms | **3.5x** |
| Image (grid=4, 64x64) + text | 24 | **56 ms** | ~180 ms | **3.2x** |
| Image (grid=16, 256x256) + text | 84 | **312 ms** | (~) | (~) |

### 512x512 image regime — trn2 currently 4x slower

| Config | TTFT | CTE | Decode | Status |
|---|---:|---:|---:|---|
| TP=2 LNC=2 NKI recurrent bucket=512 | 738 ms | 691 ms | 125 tok/s | ok |
| **TP=4 LNC=2 NKI recurrent bucket=512** | **710 ms** | **686 ms** | **152 tok/s** | ok (best) |
| TP=8 LNC=2 NKI recurrent bucket=512 | 711 ms | 687 ms | 168 tok/s | ok |
| L40S `transformers` chunk_gated_delta_rule | ~180 ms | - | - | reference |

**Why TP doesn't help CTE.** The current NKI recurrent kernel
(`USE_NKI=1`) walks `bucket` sequence positions per `(B, head)` pair,
launched in a Python loop in `_nki_recurrent_forward`. With 16 v_heads
and TP=4 → 4 heads/rank × 18 layers × 512 steps = 36,864 sequential
kernel calls per CTE. At ~75 µs per launch, that's a fixed 691 ms
regardless of TP because the overhead is per-launch, not per-FLOP.
TP=8 gives 2 heads/rank × 18 × 512 = 18,432 calls at the same
~75 µs each → still ~691 ms (saturated by launch latency).

**Why fused Neumann kernel can't help yet.** The `_fused_chunked_forward`
path processes 128-token chunks in a single SBUF-state-resident kernel
call. A bucket-512 prefill would need 4 chunks instead of 512 sequential
launches — projected CTE under 100 ms, well under L40S 180 ms. But
the bucketed-fused compile in SDK 2.29.5133 produces wrong outputs
(empty / `<|endoftext|>` flood) for **any compiled seq_len in
{384, 512, 1024}**. seq_len=128 (1 chunk) and seq_len=2048 (16 chunks
single-bucket) both produce correct outputs; everything in between
breaks. This looks like a compiler regression in cross-chunk state
propagation. Fix requires either an upstream compiler bug-fix or a
kernel rewrite that batches BH into a single kernel call (eliminating
the launch-overhead bottleneck of the recurrent path).

**FLA-style per-chunk path is also blocked by the same compiler issue.**
Following `fla.ops.gated_delta_rule` design, we split the SSD-style
fused kernel into a per-chunk kernel called in a Python loop with state
passed through as a tensor. The kernel itself is verified correct
against the FLA reference to fp32 noise (1e-7 max abs err on 4-chunk
bucket=512 inputs — see `test/integration/test_chunked_kernel.py`).
With this path, **measured CTE = 60.9 ms** for 512x512 (TTFT 84.3 ms),
beating L40S 180 ms by 2.1×. However when the same per-chunk loop is
embedded inside the full NeuronModel forward and traced/compiled, the
multi-chunk case produces empty output (1-chunk works correctly).

**Investigation summary (2026-05-11).** Spent ~5 hours bisecting:

1. **Kernel math:** verified correct against FLA reference, 1e-7 max
   abs err on 4-chunk × multi-head inputs (5 unit tests passing).
2. **Standalone XLA tracing:** correct at all scales tested — 1 layer
   × 16 BH × 4 chunks = 64 calls, and 4 layers × 16 BH × 4 chunks =
   256 calls all materialize to FLA-matching values.
3. **HLO IR inspection:** dumped HLO from the failing compile and
   verified state-passthrough dataflow. Each chunk's `state_in` is
   correctly tied to the previous chunk's `state_out` via
   `get-tuple-element` idx 1. 1152 chunked calls in entry computation
   for bucket=512 (18 layers × 16 BH × 4 chunks). All chained
   correctly.
4. **Workarounds tried, none helped:** `out_chunk.clone()` /
   `state.clone()`, `out_chunk * 1.0`, pre-allocated `zero_states[bh]`
   indexing, `xm.optimization_barrier_([out_chunk, state])`, reversed
   `(state_out, output)` tuple order, output dtype bf16 vs state_out
   fp32 (different sizes to defeat aliasing), `-O2` compile.
5. **Other paths:** USE_PYTORCH_CHUNK hits the documented
   neuronx-cc ICE (NCC_INLA001). USE_NKI_FUSED at multi-chunk seq_len
   produces same empty output. Single-bucket=2048 fused works
   (16 chunks) but is slow (2.7s CTE).

**Conclusion:** the bug is in NEFF generation specific to NxDI's
`parallel_model_trace` of a Python-loop pattern that calls the same
`@nki.jit` kernel many times with tensor-passed state. HLO is correct;
NEFF (or the schedulers/allocators that produce it) miscompiles. This
is a Neuron compiler issue, not a contrib-model bug. Filed for upstream.

| Kernel | Math | Standalone XLA | Inside compiled NeuronModel |
|---|---|---|---|
| Recurrent (`USE_NKI=1`) | exact | ✓ | ✓ correct, 686 ms CTE for bucket=512 |
| Fused (`USE_NKI_FUSED=1`, default) | log-sum-exp | ✓ at 1, 16 chunks | ✗ empty at 3-8 chunks |
| Per-chunk (`USE_NKI_CHUNKED=1`) | log-sum-exp | ✓ all chunk counts | ✗ empty at >1 chunks |

The per-chunk and fused paths both demonstrate that the kernel-level
work for sub-180-ms 512x512 TTFT is achievable on Neuron; the gap is in
the upstream compiler's handling of multiple sequential NKI kernel
calls with state pass-through inside a traced graph.

### Decode throughput

Decode is unchanged across TP because TKG runs only 1 recurrent step
per token. TP scaling helps the FF/attention parts of TKG slightly:

| TP | Decode tok/s |
|---:|---:|
| 2 | 125 |
| 4 | 152 |
| 8 | 168 |

### Recommended buckets

Default in `examples/compile_text_decoder.py`:

    USE_NKI=1 python3 examples/compile_text_decoder.py \
        --buckets 32 64 128 256 512 1024 2048

Harness: `examples/benchmark_vl.py`. Raw JSON: `/tmp/bench_mb.json`.

## Setup

| | |
|---|---|
| Instance | `trn2.48xlarge` (16 × trainium2, 64 logical cores @ LNC=2) |
| Neuron SDK | 2.29 |
| Vision encoder | Qwen3.5 ViT (24 layers, hidden=1024), BF16, **single NeuronCore** |
| Vision buckets | 16 patches (tiny grids) and 4096 patches (all grids ≤ 64) |
| Text decoder | Qwen3.5-2B, 24 layers hybrid DeltaNet + GQA, BF16, **TP=4** |
| Text decoder buckets | **[128, 512, 1024, 2048]** multi-bucket |
| DeltaNet CTE kernel | `USE_NKI=1` (per-token recurrent NKI) |
| State dtypes | `recurrent_state_buffer` and `conv_state_buffer` **fp32** |
| Gated RMSNorm | Full fp32 (norm + silu(gate)) before final bf16 cast |
| Prompt | `"What is in this image?"` |
| `max_new_tokens` | 40 (greedy) |
| Iterations | 5 timed (2 warm-up) |

## Results: end-to-end per grid

Each row is a different image size. `input_ids` = `20 + grid²` (20 chat
template tokens + one `<|image_pad|>` per raw patch). The runtime picks
the smallest bucket that fits.

| Grid | Image px | Vision tokens | `input_ids` | Bucket | Vision (ms) | CTE (ms) | **TTFT (ms)** | TKG (ms, 39 tok) | Decode (tok/s) | **Total (ms)** | Output |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---|
|  4 |   64² |   4 |   24 |  128 |   3.9 |  304.9 |   **309.3** | 255.5 | 152.6 |   **564.9** | Bulbasaur ✓ |
| 16 |  256² |  64 |   84 |  128 | 198.1 |  305.0 |   **504.1** | 256.9 | 151.8 |   **761.1** | Bulbasaur ✓ |
| 32 |  512² | 256 |  276 |  512 | 201.7 |  687.1 |   **889.9** | 256.1 | 152.3 |  **1146.0** | Bulbasaur ✓ |
| 48 |  768² | 576 |  596 | 1024 | 205.5 | 1366.4 |  **1572.9** | 257.0 | 151.8 |  **1830.0** | Bulbasaur ✓ |
| 62 |  992² | 961 |  981 | 1024 | 214.0 | 1367.0 |  **1583.4** | 255.9 | 152.4 |  **1839.4** | Bulbasaur ✓ |

Notable observations:

- **TTFT scales with the bucket the prompt lands in**, not a fixed cost.
  grid 4 (24 tokens) hits the 128 bucket and starts producing text in
  **309 ms**; grid 62 (981 tokens) uses the 1024 bucket and takes
  **1.58 s**. This is possible because DeltaNet CTE is O(seq_len)
  regardless of the actual prompt fill (the recurrent kernel walks
  every position), so giving it a shorter compiled graph is the fix.

- **TKG decode is ~152 tok/s across all grids** — same per-token kernel.

- **Vision encoder latency depends on the compiled vision bucket**.
  grid 4 hits the 16-patch bucket (~4 ms); grid 16/32/48/62 all use the
  4096-patch bucket (~200 ms) regardless of true patch count. Compiling
  intermediate vision buckets (256, 1024) would help.

  **2026-05-09 update:** intermediate buckets 256 and 1024 are now
  compiled (`compile_vision_encoder.py --buckets 16 256 1024 4096`),
  giving the latencies below. Buckets are auto-discovered from the
  vision directory by `NeuronQwen35VisionModelWrapper.load_compiled`.

  | Grid | Patches | Old vision (ms, 4096-only) | New vision (ms, 16/256/1024/4096) | Speedup |
  |---:|---:|---:|---:|---:|
  |  4 |   16 |   3.9 |   4.2 | ~ |
  | 16 |  256 | 198.1 |   7.6 | **26x** |
  | 32 | 1024 | 201.7 |  24.8 |  **8x** |
  | 48 | 2304 | 205.5 | 207.3 | ~ (no bucket between 1024 and 4096) |
  | 62 | 3844 | 214.0 | 211.3 | ~ |

  Total task time at grid 16 drops from 761 ms to 572 ms (-25 %); grid
  32 drops from 1146 ms to 972 ms (-15 %). HBM peak grows from ~22 GB
  to ~24.9 GB to hold the 4 vision graphs.

- **All grids correctly identify Bulbasaur** with detailed, coherent
  descriptions — the fp32 fixes (state buffer, conv state, gated norm)
  hold up through 961 vision tokens.

## Single-bucket baseline (for context)

Same recurrent kernel but compiled with `seq_len=2048` only, no
bucketing. Every CTE walks the full 2048 steps.

| Grid | TTFT (ms) | Total (ms) |
|---:|---:|---:|
|  4 | 2739 | 2994 |
| 16 | 2934 | 3190 |
| 32 | 2937 | 3192 |
| 48 | 2941 | 3193 |
| 62 | 2948 | 3206 |

**Speedup from multi-bucket**:

| Grid | TTFT speedup | Total speedup |
|---:|---:|---:|
|  4 | **8.9×** | **5.3×** |
| 16 | **5.8×** | **4.2×** |
| 32 | **3.3×** | **2.8×** |
| 48 | **1.9×** | **1.7×** |
| 62 | **1.9×** | **1.7×** |

## Stability (grid=62, 5 iterations)

| Stage | min | max | spread |
|---|---:|---:|---:|
| Vision encode | 212.6 | 215.8 | 3.2 |
| CTE           | 1366.3 | 1367.6 | 1.3 |
| TTFT          | 1581.5 | 1585.2 | 3.7 |
| TKG total     | 254.4 | 256.8 | 2.4 |
| Total         | 1837.0 | 1841.3 | 4.3 |

## HBM usage

Multi-bucket compilation stores 4 CTE graphs + 1 TKG graph. Total HBM
peak is ~22 GB (vs 19 GB for single bucket).

| Core | Role | HBM (GB) |
|---:|---|---:|
| 0 | text decoder TP rank 0 (4 CTE + 1 TKG graph + weights) | ~4.5 |
| 1 | text decoder TP rank 1 | ~4.5 |
| 2 | text decoder TP rank 2 | ~4.5 |
| 3 | text decoder TP rank 3 | ~4.5 |
| 4 | vision encoder (bucket 16 + 4096) | ~2.77 |
| 5 | vision encoder scratch | ~1.42 |
| **process total** | | **~22 GB** |

## Fixes summary

The PR's original contrib code produces NaN on any prefill longer than
~30 tokens. The version benchmarked here has the following changes
(all in `src/modeling_qwen35.py` and `src/nki_kernels/nki_deltanet_fused.py`):

1. **Log-sum-exp decay matrix**: replaced `exp(gc[i]) * exp(-gc[j])`
   with `exp(min(gc[i] - gc[j], 0))`. Same math, no fp32 overflow in
   the masked region.
2. **fp32 `recurrent_state_buffer`**: was bf16 — lost precision after
   ~1000 recurrent steps, causing immediate EOS in TKG for long VL
   prompts.
3. **fp32 `conv_state_buffer`**: same reason, small but same class of bug.
4. **fp32 gated RMSNorm at the DeltaNet output**: matches HF
   `Qwen3_5RMSNormGated`, which keeps norm math and `silu(gate)` in
   fp32 before the final bf16 cast. The PR cast to bf16 before the
   norm.
5. **Per-token recurrent NKI kernel (`USE_NKI=1`)**: swaps the default
   128-step Neumann fused kernel for a straight sequential recurrence.
   ~18× slower per prompt, but no Neumann spectral-radius blow-up, so
   correctly handles up to 981-token prompts.
6. **Multi-bucket compilation**: `[128, 512, 1024, 2048]` with
   `enable_bucketing=True`. CTE now scales with prompt length, not
   constant at the max bucket.

## Remaining limitation

~~`grid_size=64` still produces an empty reply.~~ **Fixed 2026-05-09**.
The earlier hypothesis (state-magnitude blow-up in the NKI kernel) was
wrong. Bisection showed that grid 62 (981 tokens) and grid 64 + a
slightly longer text prompt (1051 tokens) both work, while only the
default grid 64 prompt at exactly 1044 tokens fails. Root cause was a
sentinel-index bug in `Qwen35VLForCausalLM.generate`: trailing zero
vision embeddings were `index_put_`-scattered to position
`pad_limit-1`, which is the LAST REAL INPUT TOKEN, zeroing the
assistant-turn-marker `\n\n` embedding for grid 64 specifically and
making the model emit `\n\n<|im_end|>` immediately. Fix:
`src/modeling_qwen35_vl.py` now uses a sentinel inside the NxDI pad
region (`max_bucket - 1`, clamped by `pad_inputs` to `bucket - 1`).
No recompile needed.

After the fix, grid 64 (1024×1024 image, 1024 vision tokens, 1044-token
prompt) produces detailed coherent descriptions on the existing
bucket-2048 compiled model.

## How to reproduce

```bash
# One-time: compile the text decoder with multi-bucket + recurrent kernel
USE_NKI=1 python3 examples/compile_text_decoder.py \
    --out /home/ubuntu/traced_model/Qwen3.5-2B-s2048-multibucket \
    --buckets 128 512 1024 2048
# ~11 minutes on trn2.48xlarge

# One-time: compile vision-encoder buckets
python3 examples/compile_vision_encoder.py --buckets 16 4096

# Benchmark
python3 examples/benchmark_vl.py \
    --configs 4 16 32 48 62 \
    --iters 5 --warmup 2 \
    --buckets 128 512 1024 2048 \
    --compiled-path /home/ubuntu/traced_model/Qwen3.5-2B-s2048-multibucket \
    --max-new-tokens 40 \
    --out-json /tmp/bench_mb.json

# Interactive usage
python3 examples/describe_image.py \
    --image /home/ubuntu/qwen-omini-on-trn/0.png \
    --grid-size 62 \
    --max-new-tokens 80
```
