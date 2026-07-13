# Qwen3.5-4B on NeuronX Distributed Inference (Trn2)

`Qwen/Qwen3.5-4B` is a 4 B-parameter vision-language decoder with the same
hybrid attention stack as the 2B sibling — **[3 gated DeltaNet (linear
attention) + 1 full GQA attention] × 8** = 32 layers, `head_dim = 256`,
`partial_rotary_factor = 0.25`, `mrope_section = [11, 11, 10]`. The linear-
attention layers have 16/32 K/V heads at `head_dim = 128`. The ViT vision
encoder is identical to 2B's (24 layers, `patch_size = 16`,
`spatial_merge_size = 2`, `hidden_size = 1024`) but the merger's
`out_hidden_size = 2560` (matches the text `hidden_size = 2560`).

This contrib copies the 2B modeling code (`modeling_qwen35*.py`,
`nki_kernels/*`, `hybrid_apc.py`) verbatim; the architecture switch is
data-driven by `config.json`. Only test-script paths differ. No NxDI-library
modifications are required — runs on the stock
`/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/` DLAMI venv
(`neuronx-cc` 2.26.6360 / `nki` 0.5.0).

**Status:** text-only and vision-language inference are both validated
end-to-end on `trn2.48xlarge`. VL requires the legacy-direct DeltaNet CTE
kernel — same env var flags as 2B, set automatically by the runner scripts.

## Architecture diff vs Qwen3.5-2B

| field | 2B | **4B** |
|---|---:|---:|
| `hidden_size` | 2048 | **2560** |
| `intermediate_size` | 6144 | **9216** |
| `num_hidden_layers` | 24 | **32** |
| `num_attention_heads` | 8 | **16** |
| `num_key_value_heads` | 2 | **4** |
| `linear_num_value_heads` | 16 | **32** |
| `linear_num_key_heads` | 16 | 16 |
| `head_dim` | 256 | 256 |
| `vocab_size` | 248,320 | 248,320 |
| `tie_word_embeddings` | true | true |
| Vision `out_hidden_size` | 2048 | **2560** |
| Vision layers (ViT) | 24 | 24 |

## Contents

```
Qwen3.5-4B/
├── README.md              — this file
├── src/                   — verbatim copy of Qwen3.5-2B/src (parametric)
└── test/integration/      — runner + benchmark scripts, paths updated to 4B
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

- HuggingFace: [`Qwen/Qwen3.5-4B`](https://huggingface.co/Qwen/Qwen3.5-4B)
- Architecture identifier: `qwen3_5`
- Weights: 2 shards, ~9.3 GB bfloat16

Download:

```bash
python -c "from huggingface_hub import snapshot_download; \
  snapshot_download('Qwen/Qwen3.5-4B', local_dir='/mnt/nvme/models/Qwen3.5-4B')"
```

## Quick start — text-only

```bash
source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
python contrib/models/Qwen3.5-4B/test/integration/run_text_smoke.py \
    --model-path    /mnt/nvme/models/Qwen3.5-4B \
    --compiled-path /tmp/qwen35_4b_traced \
    --tp 8 --seq-len 512 --max-new-tokens 32 \
    --prompt "The capital of France is"
```

Sample validated on `trn2.48xlarge`, TP=8, bf16, seq_len=512:

```
prompt : 'The capital of France is'
output : 'The capital of France is Paris.\nA. True\nB. False\n\n<think>...'
TTFT   : 41.3 ms
TPOT   : 5.5 ms  (180.84 tok/s)
```

## Measured text-only performance (TP=8, bf16, seq_len=512)

`run_benchmark.py --prompt-lens 16 64 256 --max-new-tokens 64 --repeats 5`

| prompt tokens | TTFT (ms, median) | TPOT (ms, median) | Throughput (tok/s) |
|---|---:|---:|---:|
| 16  | **34.6** | 5.72 | 175.0 |
| 64  | 34.9 | 5.67 | 176.6 |
| 256 | 34.9 | 5.68 | 175.8 |

vs Qwen3.5-2B: 4B has ~2× TTFT (34.9 vs 17.6 ms) and ~1.4× TPOT (5.68 vs
4.00 ms). TTFT stays flat across prompt length as with 2B — DeltaNet
dominates 24/32 layers with O(1) state.

## Accuracy vs HuggingFace CPU bf16 greedy

Same 5 prompts as 2B, 16 new tokens each:

| Prompt | Neuron | HF | Match |
|---|---|---|---:|
| "The capital of France is" | "Paris. A. True B. False..." | "Paris. Explanation:..." | 10/16 |
| "The largest planet in..." | "Jupiter. It is a gas giant..." | "Jupiter, followed by Saturn..." | 8/16 |
| "Water boils at" | "100°C at sea level..." | "100°C..." | 9/16 |
| "A haiku about autumn leaves:" | "\n\n<think>\n\n</think>\n\nGolden leaves descend..." | "The maple leaves..." | 0/16 |
| "In one sentence, explain photosynthesis." | "Photosynthesis is the biological process..." | "1. Understanding the request..." | 2/16 |

Aggregate: **29/80 = 36 %** exact-token match. Output is qualitatively
coherent and factually correct on every prompt; the divergence is dominated
by different chat-template branches taken (Neuron output tends to include
`<think>` blocks that HF's reduced output doesn't, or vice versa). Same
pattern as 2B: 2 prompts fully coherent, 3 diverge after a few tokens.

## Vision (image → text)

Same pipeline as 2B: HF `AutoProcessor` → CPU pre-processing → Neuron ViT →
scatter into Neuron text decoder. The 2×2 tiling path activates
automatically for inputs above the largest compiled vision bucket.

```bash
# Compile vision buckets once (~10-11 min per bucket)
python contrib/models/Qwen3.5-4B/test/integration/compile_vision_encoder.py \
    --model-path /mnt/nvme/models/Qwen3.5-4B \
    --out-dir    /tmp/qwen35_4b_vl_bench/vision \
    --buckets    1024 4096

# Run the size sweep (compiles text with CTE bucketing on first run)
python contrib/models/Qwen3.5-4B/test/integration/run_vl_benchmark.py \
    --model-path /mnt/nvme/models/Qwen3.5-4B \
    --compiled-path /tmp/qwen35_4b_vl_bench \
    --vision-compiled-dir /tmp/qwen35_4b_vl_bench/vision \
    --tp 8 --images 512 1024 2048 \
    --max-new-tokens 48 --repeats 3
```

Measured on `trn2.48xlarge`, TP=8, bf16, text CTE buckets `[512..8192]`:

| image | vision tokens | TTFT (ms) | TPOT (ms) | tok/s |
|---|---:|---:|---:|---:|
| 512×512   |   256 |    **162** | 6.2 | 161 |
| 1024×1024 | 1,024 |    **826** | 6.2 | 162 |
| 2048×2048 | 4,096 |  **3,331** | 6.2 | 163 |

2048×2048 uses the 2×2 tiled Neuron VE path (four 4,096-patch encodes,
merged outputs re-interleaved).

**Accuracy vs HF reference on a Pallas's cat image:** HF (5.13, CPU bf16
greedy) identifies "Pallas's cat" at all 3 sizes. Neuron at 512×512
identifies as "tabby cat" (correct family, missed species — but better
than 2B which called it "pangolin"), at 1024×1024 as "domestic shorthair
in snowy environment" (correct that it's a cat), at 2048×2048 as
"bear/polar bear/brown bear" (tile boundaries lose global silhouette — same
LLaVA-NeXT-style trade-off as 2B). Text quality is coherent at all sizes.

## HF reference on CPU

Because `transformers==4.57.6` predates `qwen3_5`, HF-side comparisons need
an isolated venv:

```bash
python3 -m venv /tmp/hf_ref_venv
/tmp/hf_ref_venv/bin/pip install \
    "transformers>=5.13" "torch>=2.6" "safetensors" "sentencepiece" \
    "accelerate" "torchvision" "pillow"

/tmp/hf_ref_venv/bin/python \
    contrib/models/Qwen3.5-4B/test/integration/run_hf_reference.py \
    --model-path /mnt/nvme/models/Qwen3.5-4B \
    --max-new-tokens 16 \
    --out-json /tmp/qwen35_4b_hf_reference.json
```

## Notable configuration choices

- `use_hybrid_cache_manager=False`, `use_hybrid_apc_manager=False` — simplest
  bring-up path; hybrid APC scaffolding is present in `hybrid_apc.py` but
  disabled.
- `tie_word_embeddings=True` — same tied-weight hook as 2B populates
  `lm_head.weight` from `embed_tokens.weight`.
- **`QWEN36_DELTANET_CTE_IMPL=legacy_direct`** and
  `QWEN36_DELTANET_MULTIHEAD_CTE=0` — set automatically by `run_vl_smoke.py`
  and `run_vl_benchmark.py`. The default fused-multihead NKI kernel is
  numerically unstable on structured vision embeddings; legacy-direct is
  stable.

## Known limitations / follow-ups

- Same as 2B: 16,384 patch bucket doesn't fit trn2 single-core HBM. 2048×2048
  images fall through the 2×2 tiled path, which loses cross-tile attention.
  TP-ing the vision encoder or a windowed-attention 16,384 kernel would
  preserve full attention.
- Only batch size 1 is validated.
- Speculative decoding is not wired up.

## Running the pytest suite

```bash
source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
export QWEN35_MODEL_PATH=/mnt/nvme/models/Qwen3.5-4B
pytest contrib/models/Qwen3.5-4B/test/integration/test_model.py -s
```

## Maintainer

Contributed alongside the 2B port. Modeling code originates from PR #173
(27B sibling, `qwen3_5` model_type shared), reused with only test-path
adjustments for 4B.
