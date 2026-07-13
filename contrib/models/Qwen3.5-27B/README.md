# Qwen3.5-27B on NeuronX Distributed Inference (Trn2)

`Qwen/Qwen3.5-27B` is the flagship 27 B-parameter vision-language decoder in
the Qwen3.5 family, matching the model that
[PR #173](https://github.com/aws-neuron/neuronx-distributed-inference/pull/173)
originally targeted. Same hybrid attention stack as the smaller siblings —
**[3 gated DeltaNet + 1 full GQA] × 16 = 64 layers**, `head_dim = 256`,
`partial_rotary_factor = 0.25`, `mrope_section = [11, 11, 10]`. Text hidden
size is **5120** and there are **48 linear-attention value heads** per layer.
The ViT vision tower matches 9B's dims (depth 27, hidden 1152) but the
merger's `out_hidden_size = 5120` (matches text hidden).

This contrib copies the 2B modeling code (`modeling_qwen35*.py`,
`nki_kernels/*`, `hybrid_apc.py`) verbatim — everything is config-driven. No
NxDI-library modifications are required — runs on the stock
`/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/` DLAMI venv
(Neuron SDK 2.29 / NKI 0.3.0).

**Status:** text-only and vision-language inference are both validated
end-to-end on `trn2.48xlarge` at TP=8. VL requires the legacy-direct
DeltaNet CTE kernel — set automatically by the runner scripts.

## Architecture diff vs Qwen3.5-9B

| field | 9B | **27B** |
|---|---:|---:|
| `hidden_size` | 4096 | **5120** |
| `intermediate_size` | 12288 | **17408** |
| `num_hidden_layers` | 32 | **64** |
| `num_attention_heads` | 16 | **24** |
| `num_key_value_heads` | 4 | 4 |
| `linear_num_value_heads` | 32 | **48** |
| `linear_num_key_heads` | 16 | 16 |
| `head_dim` | 256 | 256 |
| `vocab_size` | 248,320 | 248,320 |
| `tie_word_embeddings` | false | false |
| Vision `depth` | 27 | 27 |
| Vision `hidden_size` | 1152 | 1152 |
| Vision `intermediate_size` | 4304 | 4304 |
| Vision `out_hidden_size` | 4096 | **5120** |

## Contents

```
Qwen3.5-27B/
├── README.md
├── src/                   — verbatim copy of Qwen3.5-2B/src (parametric)
└── test/integration/      — runner + benchmark scripts, paths updated to 27B
```

## Compatibility

| Component | Version |
|---|---|
| Instance | `trn2.48xlarge` (validated at TP=8) |
| Neuron SDK | 2.29 (NKI 0.3.0) |
| Python | 3.12 |
| `torch` | 2.9.1 (torch-neuronx 2.9.0.2) |
| `neuronx-distributed-inference` | 0.10.18399 |
| `transformers` | 4.57.6 (Neuron runtime). HF CPU reference needs ≥ 5.13. |

## Checkpoint

- HuggingFace: [`Qwen/Qwen3.5-27B`](https://huggingface.co/Qwen/Qwen3.5-27B)
- Architecture identifier: `qwen3_5`
- Weights: 11 shards, ~55.6 GB bfloat16

Download:

```bash
python -c "from huggingface_hub import snapshot_download; \
  snapshot_download('Qwen/Qwen3.5-27B', local_dir='/mnt/nvme/models/Qwen3.5-27B')"
```

## Quick start — text-only

```bash
source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
python contrib/models/Qwen3.5-27B/test/integration/run_text_smoke.py \
    --model-path    /mnt/nvme/models/Qwen3.5-27B \
    --compiled-path /tmp/qwen35_27b_traced \
    --tp 8 --seq-len 512 --max-new-tokens 32 \
    --prompt "The capital of France is"
```

Sample validated on `trn2.48xlarge`, TP=8, bf16, seq_len=512:

```
prompt : 'The capital of France is'
output : "The capital of France is Paris.\n\n<think>\n\n</think>\n\nThat's correct!
          **Paris** is the capital city of France..."
TTFT   : 125.4 ms
TPOT   : 21.3 ms  (46.96 tok/s)
```

## Measured text-only performance (TP=8, bf16, seq_len=512)

`run_benchmark.py --prompt-lens 16 64 256 --max-new-tokens 64 --repeats 5`

| prompt tokens | TTFT (ms, median) | TPOT (ms, median) | Throughput (tok/s) |
|---:|---:|---:|---:|
| 16  | **118.6** | 21.58 | 46.3 |
| 64  | 118.7 | 21.60 | 46.3 |
| 256 | 118.9 | 21.61 | 46.3 |

TTFT stays flat across prompt length (DeltaNet dominates 48/64 layers).

**Full family comparison at TP=8:**

| model | Params | Layers | TTFT (16 tok, ms) | TPOT (ms) | HF match | 1024×1024 identification |
|---|---:|---:|---:|---:|---:|---|
| 2B  |  2B |  24 | 17.6 | 4.00 | 66% | Pallas's cat ✅ (guess) |
| 4B  |  4B |  32 | 34.6 | 5.68 | 36% | fluffy cat |
| 9B  |  9B |  32 | 42.9 | 6.89 | 91% | Pallas's cat (Manul) ✅ |
| **27B** | **27B** | **64** | **118.6** | **21.58** | **77%** | **Pallas's cat (manul)** ✅ |

## Accuracy vs HuggingFace CPU bf16 greedy

Same 5-prompt suite, 16 new tokens each:

| Prompt | Match |
|---|---:|
| "The capital of France is" | 5/16 |
| "The largest planet in our solar system is" | **16/16** ✅ |
| "Water boils at" | 9/16 |
| "A haiku about autumn leaves:" | **16/16** ✅ |
| "In one sentence, explain photosynthesis." | **16/16** ✅ |

Aggregate: **62/80 = 77.5 %** exact-token match. All outputs are
qualitatively coherent and factually correct. Divergence is dominated by
different chat-template branches (with/without `<think>` block).

## Vision (image → text)

Same pipeline as smaller siblings: HF `AutoProcessor` → CPU pre-processing
→ Neuron ViT → scatter into Neuron text decoder. 2×2 tiling activates
automatically for inputs above the largest compiled vision bucket.

```bash
# Compile vision buckets once (~12-13 min per bucket)
python contrib/models/Qwen3.5-27B/test/integration/compile_vision_encoder.py \
    --model-path /mnt/nvme/models/Qwen3.5-27B \
    --out-dir    /tmp/qwen35_27b_vl_bench/vision \
    --buckets    1024 4096

# Run the size sweep
python contrib/models/Qwen3.5-27B/test/integration/run_vl_benchmark.py \
    --model-path /mnt/nvme/models/Qwen3.5-27B \
    --compiled-path /tmp/qwen35_27b_vl_bench \
    --vision-compiled-dir /tmp/qwen35_27b_vl_bench/vision \
    --tp 8 --images 512 1024 2048 \
    --max-new-tokens 48 --repeats 3
```

Measured on `trn2.48xlarge`, TP=8, bf16, text CTE buckets `[512..8192]`:

| image | vision tokens | TTFT (ms) | TPOT (ms) | tok/s | identification |
|---|---:|---:|---:|---:|---|
| 512×512   |   256 |     **444** | 22.0 | 45.4 | "Pallas's cat (manul)" ✅ |
| 1024×1024 | 1,024 |   **1,974** | 21.9 | 45.5 | "Pallas's cat (manul)" ✅ |
| 2048×2048 | 4,096 |   **7,986** | 22.1 | 45.2 | cat/lynx/bobcat (tile boundary loss) |

2048×2048 uses the 2×2 tiled Neuron VE path (four 4,096-patch encodes,
merged outputs re-interleaved).

**27B is the ONLY size in the family that correctly identifies "Pallas's
cat (manul)" at both 512×512 AND 1024×1024.** 2B/4B/9B all miss at 512×512
(model capacity limit at that resolution). At 2048×2048 27B still hedges
because tile boundaries lose global silhouette — same LLaVA-NeXT-style
trade-off across the family.

**VL comparison across sizes:**

| image | 2B TTFT | 4B TTFT | 9B TTFT | **27B TTFT** | 27B TPOT |
|---|---:|---:|---:|---:|---:|
| 512×512   |    86 |    162 |    176 |   **444** | 22.0 |
| 1024×1024 |   528 |    826 |    933 | **1,974** | 21.9 |
| 2048×2048 | 2,090 |  3,331 |  3,763 | **7,986** | 22.1 |

TTFT scales roughly with model size (27B is ~2.1× 9B) while TPOT is
essentially the same across VL and text — text decode is Neuron-only.

## HF reference on CPU

Because `transformers==4.57.6` predates `qwen3_5`, HF-side comparisons need
an isolated venv:

```bash
python3 -m venv /tmp/hf_ref_venv
/tmp/hf_ref_venv/bin/pip install \
    "transformers>=5.13" "torch>=2.6" "safetensors" "sentencepiece" \
    "accelerate" "torchvision" "pillow"

/tmp/hf_ref_venv/bin/python \
    contrib/models/Qwen3.5-27B/test/integration/run_hf_reference.py \
    --model-path /mnt/nvme/models/Qwen3.5-27B \
    --max-new-tokens 16 \
    --out-json /tmp/qwen35_27b_hf_reference.json
```

## Notable configuration choices

- `use_hybrid_cache_manager=False`, `use_hybrid_apc_manager=False` —
  simplest bring-up path. This is the same config as PR #173's original
  target; hybrid APC scaffolding is present in `hybrid_apc.py` but disabled
  by default.
- `tie_word_embeddings=False` — 27B stores `lm_head.weight` separately in
  safetensors, same as 9B.
- **`QWEN36_DELTANET_CTE_IMPL=legacy_direct`** and
  `QWEN36_DELTANET_MULTIHEAD_CTE=0` — set automatically by `run_vl_smoke.py`
  and `run_vl_benchmark.py`. The default fused-multihead NKI kernel is
  numerically unstable on structured vision embeddings.

## Known limitations / follow-ups

- Same as the smaller siblings: 16,384 patch bucket doesn't fit trn2
  single-core HBM. 2048×2048 images fall through the 2×2 tiled path, which
  loses cross-tile attention. TP-ing the vision encoder or a
  windowed-attention 16,384 kernel would preserve full attention.
- Only batch size 1 is validated.
- Speculative decoding is not wired up.
- Weight loading takes noticeably longer than smaller models (~6 min at
  TP=8 for the initial `.load()` call). Sharded checkpoint reuse cuts this
  after the first run.

## Maintainer

Contributed alongside the 2B/4B/9B ports. Modeling code originates from
PR #173 (the original 27B target); this contrib exercises exactly the
architecture PR #173 was designed for.
