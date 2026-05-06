# Qwen3.5-2B VL Benchmark (Neuron)

End-to-end vision-language pipeline on AWS Trainium 2 for the task
**"What is in this image?"** over `0.png` (a Bulbasaur illustration).

Both the vision encoder and the text decoder are compiled to Neuron.
This report covers the **multi-bucket recurrent** DeltaNet CTE kernel
(`USE_NKI=1` + `enable_bucketing=True`), which is the configuration that
solves both the NaN problem and the constant-time-CTE problem.

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

`grid_size=64` (1024×1024 image, 1044-token prompt) still produces an
empty reply. Root cause: vision embedding norms are ~10× text embedding
norms (mean 6.68 vs 0.68), and after 1024 recurrent steps the state
accumulates to a magnitude that bf16 engine intermediates inside the
NKI kernel cannot preserve. grid 62 (961 vision tokens, 992×992 image)
is the practical upper bound.

Fixing this requires either a chunked kernel that matches HF's
`torch_chunk_gated_delta_rule` (blocked on an upstream neuronx-cc ICE,
NCC_INLA001), or a rewrite of the recurrent NKI kernel using fp32
matmul accumulators.

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
