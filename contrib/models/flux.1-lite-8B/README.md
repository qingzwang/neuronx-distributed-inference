# FLUX.1-lite-8B

[FLUX.1-lite-8B](https://huggingface.co/Freepik/flux.1-lite-8B) is Freepik's
distillation of FLUX.1-dev: the same `FluxTransformer2DModel`, pruned from 19
double-stream (MMDiT) blocks to 8 while keeping all 38 single-stream blocks, and
guidance-distilled the same way.

## No modeling code was needed

NxD Inference's FLUX implementation already runs this checkpoint. The backbone
reads `num_layers` and `num_single_layers` from the checkpoint's `config.json`
through `load_diffusers_config`, so an 8-block checkpoint builds an 8-block model,
and the tensor-parallel sharding follows the attention head count, which lite
shares with dev (24 heads x 128 dims). Verified end to end at 1024x1024 with
TP=4 on trn2.

This folder therefore contains configuration, validation, latency measurement and
tests, not a new model:

| File | Contents |
| --- | --- |
| `src/flux_lite.py` | Checkpoint and TP-degree validation, config/application builder, per-stage latency measurement |
| `src/generate.py` | CLI: generate images, sweep step counts, report where the time goes |
| `test/integration/test_model.py` | Validation tests, plus accuracy against a CPU diffusers reference |
| `samples/` | Outputs from trn2 at the settings documented below |

What the validation is for: a wrong checkpoint otherwise fails as a shape error
deep in weight loading. In particular **FLUX.1-schnell is rejected** — it is
guidance-*free*, so it needs a different sampling path rather than just different
weights — and a TP degree that does not divide 24 is rejected with the list of
degrees that do.

## Usage

```bash
pip install "diffusers==0.32.0" accelerate   # the [flux] extra

python contrib/models/flux.1-lite-8B/src/generate.py \
    --checkpoint-dir /path/to/flux.1-lite-8B \
    --prompt "A close-up photo of a red panda wearing tiny round glasses, reading a leather-bound book" \
    --steps 28 --save-image
```

![FLUX.1-lite-8B on trn2: a red panda in round glasses reading a book](samples/flux_lite_1024px_28steps_tp4.png)

*1024x1024, 28 steps, guidance 3.5, seed 42, TP=4 — exactly the command above.
Downscaled for this page; `samples/` has the rest.*

```python
import sys
sys.path.insert(0, "contrib/models/flux.1-lite-8B/src")
from flux_lite import build_application, compile_and_load, generate

app = build_application("/path/to/flux.1-lite-8B", height=1024, width=1024, tp_degree=4)
compile_and_load(app, "/tmp/flux_lite_8b/")

image, latency = generate(app, "a red panda reading a book", num_inference_steps=28, seed=42)
image.save("out.png")
print(latency.as_dict())
```

Components are compiled once into `--compiled-model-path` and reused, so only the
first run pays for it (~4 min at 1024x1024 on a cold cache; ~34 s to reach the
first request afterwards). Within a process the first request pays device warmup,
so `generate.py` discards one before reporting; `--no-warmup` keeps it.

Resolution is fixed at compile time — `height` and `width` are part of the
compiled shapes, so changing them means recompiling.

## Compatibility

| | |
| --- | --- |
| Instance types tested | trn2 (`trn2.3xlarge`, `logical-neuroncore-config: 2`) |
| NxD Inference | 0.9.x (this repo at `4bcdc54`) |
| Neuron SDK | `torch-neuronx` 2.9.0.2.15, `neuronx-distributed` 0.19.28492, **`neuronx-cc` 2.26.6360.0** |
| Frameworks | torch 2.9.1, transformers 4.57.6, diffusers 0.32.0 |
| Precision | BF16 |
| Parallelism | Backbone TP=4 (trn2 default) and TP=2, both measured below; TP=8 on trn1 untested. Context- and CFG-parallel are inherited from the core FLUX path but untested here. |

> `neuronx-cc` must be pinned. `libneuronxla 2.2` requires only `neuronx-cc~=2.0`,
> so pip resolves to 2.27, which fails to compile with
> `[NCC_ISMP902] Simplifier error: is_subset(): incompatible function arguments`.

Checkpoints: [Freepik/flux.1-lite-8B](https://huggingface.co/Freepik/flux.1-lite-8B),
[Freepik/flux.1-lite-8B-alpha](https://huggingface.co/Freepik/flux.1-lite-8B-alpha).
`--allow-any-flux` also accepts other guidance-distilled FLUX checkpoints;
FLUX.1-dev runs at roughly 1.6x the step latency (19 double blocks instead of 8).

## Measured latency

trn2, BF16, batch 1, backbone TP=4, 512-token prompt budget. Median over 2
requests after a discarded warmup request; per-stage figures come from
`measure_stages`, which times the four Neuron submodels the pipeline calls.

| TP | Resolution | Steps | ms/step | Prompt encode | Denoise | VAE decode | **End to end** |
|---|---|---|---|---|---|---|---|
| 4 | 1024x1024 | 4 | 217 | 38 ms | 0.88 s | 0.30 s | **1.25 s** |
| 4 | 1024x1024 | 8 | 217 | 35 ms | 1.75 s | 0.29 s | **2.13 s** |
| 4 | 1024x1024 | 28 | 217 | 35 ms | 6.10 s | 0.29 s | **6.52 s** |
| 4 | 512x512 | 4 | 73 | 40 ms | 0.31 s | 0.07 s | **0.44 s** |
| 2 | 1024x1024 | 4 | 406 | 57 ms | 1.64 s | 0.30 s | **2.02 s** |
| 2 | 1024x1024 | 8 | 406 | 54 ms | 3.26 s | 0.29 s | **3.65 s** |
| 2 | 1024x1024 | 28 | 406 | 54 ms | 11.37 s | 0.29 s | **11.82 s** |

Step latency is flat across step counts — the same graph runs every step on
static shapes. Denoising is 94% of a 28-step request, which is why the backbone
is the only component that is tensor-parallel.

### TP=2 vs TP=4

Doubling the backbone's tensor-parallel degree is worth **1.87x** on the step
(406 -> 217 ms), not 2x; the shortfall is the collectives. VAE decode does not
move at all (0.29 s either way) because it runs at tp_degree=1 regardless.

![Same prompt and seed at TP=2 and TP=4](samples/tp2_vs_tp4_28steps.png)

Quality is unaffected: same prompt, seed and schedule at the two degrees give
PSNR 35.2 dB, mean 0.99/255. The images are not bit-identical because summing
partial products over 2 ranks and over 4 ranks rounds differently in BF16 — the
same reason two GPU batch sizes can differ.

TP=2 leaves two cores of the chip idle, so it is useful for running something
else alongside, not for speed.

FLUX.1-lite degrades gracefully as steps come down — 8 steps still resolves the
subject, materials and lighting, and 4 is a usable preview:

![FLUX.1-lite-8B at 4, 8 and 28 steps](samples/steps_4_8_28_tp4.png)

Reproduce with:

```bash
python contrib/models/flux.1-lite-8B/src/generate.py \
    -c /path/to/flux.1-lite-8B -n 4,8,28 --iterations 2 --json latency.json

# TP=2 needs the visible core set narrowed to match; see below.
NEURON_RT_VISIBLE_CORES=0-1 python contrib/models/flux.1-lite-8B/src/generate.py \
    -c /path/to/flux.1-lite-8B --tp-degree 2 -n 28 --iterations 2
```

### TP below the core count needs NEURON_RT_VISIBLE_CORES

Loading pre-compiled TP=2 artifacts on a process that can see all four cores
fails partway into the first request:

```
NRT has already been setup with a collectives world size of 2 ... but trying to
set up collectives world size of 4
Failed to create global communicator, g_device_id=0, g_device_count=4
```

The artifacts are correct (their `neuron_config.json` records
`tp_degree=2, world_size=2`); the mismatch is that the distributed world is sized
from the visible cores, which is 4. Narrow it to the TP degree —
`NEURON_RT_VISIBLE_CORES=0-1` — and it runs. Compiling and running in one process
does not hit this, which is why it only shows up on a second, load-only run.

## Accuracy

Compared against stock diffusers on CPU, same prompt and schedule, from the same
initial noise. Two references: fp32 measures how accurate the Neuron path is, BF16
shows where another BF16 implementation sits.

| Comparison | vs CPU fp32 | vs CPU BF16 |
| --- | --- | --- |
| CLIP pooled embedding | cos 0.99993 | cos 0.99989 |
| T5 prompt embedding | cos 0.99941 | cos 0.99944 |
| One backbone step (velocity) | **cos 0.99989**, mean abs 0.0129 | cos 0.98837 |
| Latents after 4 steps | **cos 0.99662** | cos 0.97890 |

For scale, CPU BF16 against the same fp32 reference reaches only cos 0.98679
(mean abs 0.1428) on one step and 0.97741 after four. **The Neuron path is about
11x closer to fp32 than CPU BF16 is** — Trainium accumulates matmuls in fp32 where
CPU BF16 accumulates in BF16 — so the Neuron-vs-CPU-BF16 gap is almost entirely
the reference's own error.

T5's largest per-element differences (max 1.69 against a 0.15 reference std) sit
in the **padding** positions, not the prompt: for a 17-token prompt, real tokens
differ by at most 0.41 while padding differs by up to 1.69. Padded positions are
numerically unconstrained, and FLUX feeds all 512 of them to the backbone, so they
contribute to the image but are not evidence of a defect.

### Two measurement traps

Both of these produce numbers that look exactly like a broken model, and both bit
during this work — the first attempt measured cos 0.816 and was wrong:

1. **`diffusers==0.32` takes `torch_dtype`,** and `from_pretrained` accepts
   `**kwargs`, so a stray `dtype=torch.bfloat16` is silently dropped and the
   reference runs in fp32.
2. **`randn_tensor` from a fixed seed draws different numbers at fp32 than at
   BF16** — not the same numbers rounded (max abs difference 4.2 on a unit-normal
   draw). Handing the same seed to two pipelines of different dtype starts them
   from unrelated noise.

Together they meant the two sides began from different latents. The tests build
the initial latents once and pass them to both sides explicitly, and assert the
reference's dtype after loading it.

## Running the tests

```bash
FLUX_LITE_CHECKPOINT=/path/to/flux.1-lite-8B \
    pytest contrib/models/flux.1-lite-8B/test/integration/test_model.py -v -s
```

Tests requiring the checkpoint skip when `FLUX_LITE_CHECKPOINT` is unset. All six
run in ~1.5 minutes against a warm compilation cache. The CPU references run the
full 8B backbone per step, which is why the tight accuracy assertion is on a
single step and only the loose end-to-end check runs a whole schedule.

| Test | Checks |
| --- | --- |
| `test_checkpoint_is_lite_shaped` | 8 + 38 blocks, `guidance_embeds=True` |
| `test_tp_degree_must_divide_head_count` | Valid and invalid TP degrees against 24 heads |
| `test_default_tp_degree_per_instance` | trn2 -> 4, trn1 -> 8, unknown rejected |
| `test_generates_an_image_of_the_requested_size` | Output size, one timed backbone call per step |
| `test_backbone_step_matches_cpu_fp32` | One step within cos 0.999 of fp32, and closer to it than CPU BF16 |
| `test_latents_match_cpu_diffusers` | Four steps within cos 0.95 of a BF16 CPU reference |

## Not covered

- Batch sizes above 1.
- trn1's TP=8 default (only TP=2 and TP=4 were measured, both on trn2).
- Context parallelism and CFG parallelism: available from the core FLUX path
  (`build_application(context_parallel_enabled=...)`) but not tested here.
- img2img, inpainting, ControlNet, IP-Adapter, LoRA. The core FLUX path has
  `control_flux.py` and `inpaint_flux.py` examples that were not exercised with
  this checkpoint.
- trn1.
