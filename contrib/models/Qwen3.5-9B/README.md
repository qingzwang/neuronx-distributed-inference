# Qwen3.5-9B on NeuronX Distributed Inference (Trn2)

`Qwen/Qwen3.5-9B` is a 9 B-parameter vision-language decoder with the same
hybrid attention stack as the 2B/4B siblings — **[3 gated DeltaNet (linear
attention) + 1 full GQA attention] × 8** = 32 layers,
`head_dim = 256`, `partial_rotary_factor = 0.25`,
`mrope_section = [11, 11, 10]`. Text hidden size steps up to **4096**
(vs 2560 in 4B); the vision tower is **larger** as well (depth 27, hidden
1152, `out_hidden_size = 4096`).

This contrib copies the 2B/4B modeling code (`modeling_qwen35*.py`,
`nki_kernels/*`, `hybrid_apc.py`) verbatim; the architecture switch is
data-driven by `config.json`. Only test-script paths differ. No NxDI-library
modifications are required — runs on the stock
`/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/` DLAMI venv
(`neuronx-cc` 2.26.6360 / `nki` 0.5.0).

**Status:** text-only and vision-language inference are both validated
end-to-end on `trn2.48xlarge`. VL requires the legacy-direct DeltaNet CTE
kernel — set automatically by the runner scripts.

## Architecture diff vs Qwen3.5-4B

| field | 4B | **9B** |
|---|---:|---:|
| `hidden_size` | 2560 | **4096** |
| `intermediate_size` | 9216 | **12288** |
| `num_hidden_layers` | 32 | 32 |
| `num_attention_heads` | 16 | 16 |
| `num_key_value_heads` | 4 | 4 |
| `linear_num_value_heads` | 32 | 32 |
| `head_dim` | 256 | 256 |
| `vocab_size` | 248,320 | 248,320 |
| `tie_word_embeddings` | true | **false** (separate lm_head) |
| Vision `hidden_size` | 1024 | **1152** |
| Vision `depth` | 24 | **27** |
| Vision `intermediate_size` | 4096 | **4304** |
| Vision `out_hidden_size` | 2560 | **4096** |

## Contents

```
Qwen3.5-9B/
├── README.md
├── src/                   — verbatim copy of Qwen3.5-2B/src (parametric)
└── test/integration/      — runner + benchmark scripts, paths updated to 9B
```

## Compatibility

| Component | Version |
|---|---|
| Instance | `trn2.48xlarge` (validated at TP=8) |
| `neuronx-cc` | 2.26.6360.0 |
| `nki` | 0.5.0 |
| `neuronx-distributed` | 0.19.28492 |
| `neuronx-distributed-inference` | 0.10.18399 |
| `torch-neuronx` | 2.9.0.2 (torch 2.9.1) |
| `libneuronxla` | 2.2.17544 |
| Python | 3.12 |
| `transformers` | 4.57.6 (Neuron runtime). HF CPU reference needs ≥ 5.13. |

## Checkpoint

- HuggingFace: [`Qwen/Qwen3.5-9B`](https://huggingface.co/Qwen/Qwen3.5-9B)
- Architecture identifier: `qwen3_5`
- Weights: 4 shards, ~19 GB bfloat16

Download:

```bash
python -c "from huggingface_hub import snapshot_download; \
  snapshot_download('Qwen/Qwen3.5-9B', local_dir='/mnt/nvme/models/Qwen3.5-9B')"
```

## Quick start — text-only

```bash
source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
python contrib/models/Qwen3.5-9B/test/integration/run_text_smoke.py \
    --model-path    /mnt/nvme/models/Qwen3.5-9B \
    --compiled-path /tmp/qwen35_9b_traced \
    --tp 8 --seq-len 512 --max-new-tokens 32 \
    --prompt "The capital of France is"
```

Sample validated on `trn2.48xlarge`, TP=8, bf16, seq_len=512:

```
prompt : 'The capital of France is'
output : 'The capital of France is Paris.\nThe capital of France is Paris.\n...'
TTFT   : 49.6 ms
TPOT   : 6.8 ms  (147.58 tok/s)
```

## Measured text-only performance (TP=8, bf16, seq_len=512)

`run_benchmark.py --prompt-lens 16 64 256 --max-new-tokens 64 --repeats 5`

| prompt tokens | TTFT (ms, median) | TPOT (ms, median) | Throughput (tok/s) |
|---:|---:|---:|---:|
| 16  | **42.9** | 6.89 | 145.0 |
| 64  | 43.0 | 6.91 | 144.8 |
| 256 | 43.2 | 6.95 | 143.9 |

TTFT stays flat across prompt length (DeltaNet dominates 24/32 layers).

**Cross-size comparison (identical 5-prompt suite, TP=8, bf16):**

| model | TTFT (16 tok) | TPOT | Text HF match rate |
|---|---:|---:|---:|
| 2B  | 17.6 ms | 4.00 ms | 66% |
| 4B  | 34.6 ms | 5.68 ms | 36% |
| **9B**  | **42.9 ms** | **6.89 ms** | **91%** |

## Accuracy vs HuggingFace CPU bf16 greedy

Same 5-prompt suite, 16 new tokens each:

| Prompt | Match |
|---|---:|
| "The capital of France is" | **16/16** ✅ |
| "The largest planet in our solar system is" | **16/16** ✅ |
| "Water boils at" | 9/16 |
| "A haiku about autumn leaves:" | **16/16** ✅ |
| "In one sentence, explain photosynthesis." | **16/16** ✅ |

Aggregate: **73/80 = 91 %** exact-token match — the best of the 2B/4B/9B
trio. Output is qualitatively coherent and factually correct on every
prompt.

## Vision (image → text)

Same pipeline as 2B/4B: HF `AutoProcessor` → CPU pre-processing → Neuron ViT
→ scatter into Neuron text decoder. 2×2 tiling activates automatically for
inputs above the largest compiled vision bucket.

```bash
# Compile vision buckets once (~12 min per bucket, larger than 2B/4B)
python contrib/models/Qwen3.5-9B/test/integration/compile_vision_encoder.py \
    --model-path /mnt/nvme/models/Qwen3.5-9B \
    --out-dir    /tmp/qwen35_9b_vl_bench/vision \
    --buckets    1024 4096

# Run the size sweep (compiles text with CTE bucketing on first run)
python contrib/models/Qwen3.5-9B/test/integration/run_vl_benchmark.py \
    --model-path /mnt/nvme/models/Qwen3.5-9B \
    --compiled-path /tmp/qwen35_9b_vl_bench \
    --vision-compiled-dir /tmp/qwen35_9b_vl_bench/vision \
    --tp 8 --images 512 1024 2048 \
    --max-new-tokens 48 --repeats 3
```

Measured on `trn2.48xlarge`, TP=8, bf16, text CTE buckets `[512..8192]`:

| image | vision tokens | TTFT (ms) | TPOT (ms) | tok/s | identification |
|---|---:|---:|---:|---:|---|
| 512×512   |   256 |    **176** | 7.4 | 135 | analyzing body/head features |
| 1024×1024 | 1,024 |    **933** | 7.5 | 133 | **Pallas's cat (Manul)** ✅ |
| 2048×2048 | 4,096 |  **3,763** | 7.5 | 133 | close-up shot (tile boundary loss) |

2048×2048 uses the 2×2 tiled Neuron VE path (four 4,096-patch encodes,
merged outputs re-interleaved).

**9B vs 4B VL:**

| image | 4B TTFT | 9B TTFT | 4B TPOT | 9B TPOT | best identification |
|---|---:|---:|---:|---:|---|
| 512×512   |   162 |   176 | 6.2 | 7.4 | 9B analyzes features, 4B calls "tabby cat" |
| 1024×1024 |   826 |   933 | 6.2 | 7.5 | **9B correctly names "Pallas's cat / Manul"** — 4B says "fluffy cat" |
| 2048×2048 | 3,331 | 3,763 | 6.2 | 7.5 | Both hedge — tile trade-off |

**Accuracy vs HF reference on a Pallas's cat image:** HF (5.13, CPU bf16
greedy) identifies "Pallas's cat" at all 3 sizes. Neuron 9B matches HF at
1024×1024 (best case, uses full 4096-bucket single-shot vision encoder).
Both smaller (512×512) and larger (2048×2048) images incur trade-offs:
small resolution loses feature detail, tiled 2048 loses cross-tile
attention. Text quality remains coherent at all 3 sizes.

## HF reference on CPU

Because `transformers==4.57.6` predates `qwen3_5`, HF-side comparisons need
an isolated venv:

```bash
python3 -m venv /tmp/hf_ref_venv
/tmp/hf_ref_venv/bin/pip install \
    "transformers>=5.13" "torch>=2.6" "safetensors" "sentencepiece" \
    "accelerate" "torchvision" "pillow"

/tmp/hf_ref_venv/bin/python \
    contrib/models/Qwen3.5-9B/test/integration/run_hf_reference.py \
    --model-path /mnt/nvme/models/Qwen3.5-9B \
    --max-new-tokens 16 \
    --out-json /tmp/qwen35_9b_hf_reference.json
```

## Notable configuration choices

- `use_hybrid_cache_manager=False`, `use_hybrid_apc_manager=False` — simplest
  bring-up path; hybrid APC scaffolding is present in `hybrid_apc.py` but
  disabled.
- `tie_word_embeddings=False` — unlike 2B/4B, 9B stores `lm_head.weight`
  separately in safetensors. The tied-weights hook in modeling code is a
  no-op for 9B.
- **`QWEN36_DELTANET_CTE_IMPL=legacy_direct`** and
  `QWEN36_DELTANET_MULTIHEAD_CTE=0` — set automatically by `run_vl_smoke.py`
  and `run_vl_benchmark.py`. The default fused-multihead NKI kernel is
  numerically unstable on structured vision embeddings.

## Known limitations / follow-ups

- Same as 2B/4B: 16,384 patch bucket doesn't fit trn2 single-core HBM.
  2048×2048 images fall through the 2×2 tiled path, which loses cross-tile
  attention. TP-ing the vision encoder or a windowed-attention 16,384
  kernel would preserve full attention.
- Only batch size 1 is validated.
- Speculative decoding is not wired up.

## Maintainer

Contributed alongside the 2B and 4B ports. Modeling code originates from
PR #173 (27B sibling, `qwen3_5` model_type shared), reused with only
test-path adjustments for 9B.
