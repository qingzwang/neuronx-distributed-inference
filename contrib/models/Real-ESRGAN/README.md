# Contrib Model: Real-ESRGAN

NeuronX Distributed Inference port of [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) —
a practical super-resolution / image restoration model. The reference PyTorch
implementation is compiled for AWS Neuron (Trn/Inf) with `torch_neuronx.trace`.

## Model Information

- **Upstream repo:** `xinntao/Real-ESRGAN`
- **Model Type:** Convolutional generator for image super-resolution (not a transformer / LLM)
- **License:** BSD 3-Clause

## Supported Variants

| Preset name                  | Architecture       | Scale | Official checkpoint                                                                                                 |
|------------------------------|--------------------|:-----:|---------------------------------------------------------------------------------------------------------------------|
| `RealESRGAN_x4plus`          | RRDBNet (23 blocks)|  x4   | https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth                               |
| `RealESRNet_x4plus`          | RRDBNet (23 blocks)|  x4   | https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth                               |
| `RealESRGAN_x4plus_anime_6B` | RRDBNet (6 blocks) |  x4   | https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth                    |
| `RealESRGAN_x2plus`          | RRDBNet (23 blocks)|  x2   | https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth                               |
| `realesr-animevideov3`       | SRVGGNetCompact    |  x4   | https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth                          |
| `realesr-general-x4v3`       | SRVGGNetCompact    |  x4   | https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth                          |

## Architecture Notes

Real-ESRGAN is a fully-convolutional image generator — there is no attention,
KV cache, or autoregressive decoding — so this contrib entry does **not**
subclass `NeuronBaseModel`. Instead it traces the PyTorch generator with
`torch_neuronx.trace` and runs it on a single Neuron core. Key details:

- `nn.LeakyReLU(inplace=True)` is replaced with `inplace=False` because the
  Neuron tracer follows the functional graph and inplace ops complicate the
  pattern-match for activations.
- `pixel_unshuffle` is re-implemented as a pure PyTorch reshape/permute so the
  model has no `basicsr` dependency at inference time.
- Checkpoints published by Real-ESRGAN are loaded transparently — they can
  either be a bare `state_dict` or a dict with `params_ema` / `params` keys.
- Tracing requires a **fixed input shape**, so the helper takes a `tile` size
  and traces for `(1, 3, tile, tile)`. To super-resolve larger images, apply
  the upstream `RealESRGANer.tile_process` loop over fixed-size tiles.

## Validation Results

**Configuration:** `torch_neuronx.trace` with `(1, 3, 256, 256)` input, FP32 weights

### CPU Architecture Smoke Tests

`test/unit/test_architecture.py` runs without Neuron hardware and verifies:

| Test                                     | Result |
|------------------------------------------|:------:|
| RRDBNet x4 output shape                  |   PASS |
| RRDBNet x2 output shape (pixel_unshuffle)|   PASS |
| SRVGGNetCompact output shape             |   PASS |
| preprocess / postprocess roundtrip       |   PASS |
| `build_model` without weights            |   PASS |
| All preset net-scales                    |   PASS |

### Neuron Integration Test

`test/integration/test_model.py` compiles the chosen preset at the requested
tile size, runs one forward pass on an input from the Real-ESRGAN `inputs/`
folder, and compares pixel output against the eager-mode CPU model
(within an 8-bit tolerance that accommodates BF16 rounding if the user opts
into lower precision). Results written as `{stem}_{model_name}_x{scale}_neuron.png`.

## Usage

### 1. Get the checkpoint

```bash
mkdir -p /home/ubuntu/models/real-esrgan/
wget -O /home/ubuntu/models/real-esrgan/RealESRGAN_x4plus.pth \
    https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth
```

### 2. Get a test image

Clone the upstream repo just for the `inputs/` folder:

```bash
git clone --depth 1 https://github.com/xinntao/Real-ESRGAN.git /tmp/Real-ESRGAN
ls /tmp/Real-ESRGAN/inputs/
# 00003.png  0014.jpg  0030.jpg  ADE_val_00000114.jpg  OST_009.png  ...
```

### 3. Run the demo

```bash
cd neuronx-distributed-inference/contrib/models/Real-ESRGAN
python test/integration/test_model.py \
    --model-name RealESRGAN_x4plus \
    --model-path /home/ubuntu/models/real-esrgan/RealESRGAN_x4plus.pth \
    --input /tmp/Real-ESRGAN/inputs/0014.jpg \
    --output-dir ./results \
    --tile 256
```

This compiles the RRDBNet generator for Neuron at `(1, 3, 256, 256)`, runs a
forward pass on a center-cropped `256x256` tile of the input, and saves the
`1024x1024` upscaled tile to `./results/0014_RealESRGAN_x4plus_x4_neuron.png`.

### 4. Use the model from Python

```python
import torch
from src.modeling_real_esrgan import (
    NeuronRealESRGAN, preprocess_image, postprocess_image,
)

neuron = NeuronRealESRGAN(
    model_name="RealESRGAN_x4plus",
    model_path="/home/ubuntu/models/real-esrgan/RealESRGAN_x4plus.pth",
    dtype=torch.float32,
)
neuron.compile(input_shape=(1, 3, 256, 256),
               compiler_workdir="/tmp/real_esrgan_compile/")
neuron.save("/home/ubuntu/neuron_models/real-esrgan/RealESRGAN_x4plus/tile256/")

# Later, on the same instance:
neuron.load("/home/ubuntu/neuron_models/real-esrgan/RealESRGAN_x4plus/tile256/")

import cv2
img_bgr = cv2.imread("/tmp/Real-ESRGAN/inputs/0014.jpg")  # HWC uint8 BGR
# The traced model requires exactly 256x256; production code should tile.
img_tile = img_bgr[:256, :256]
input_tensor = preprocess_image(img_tile)
output_tensor = neuron(input_tensor)
output_bgr = postprocess_image(output_tensor)
cv2.imwrite("upscaled.png", output_bgr)
```

## Demo: 1K → 4K on 8 pinned NeuronCores

End-to-end run through the pinned-replica pipeline: 1024×1024 input is tiled
into 64 non-overlapping 128×128 blocks, each replica forwards its share in
parallel (bf16, LNC=1), and the outputs are stitched back into a single
4096×4096 image.

Run it:

```bash
cd contrib/models/Real-ESRGAN
python test/integration/demo_1k_to_4k.py \
    --input /tmp/Real-ESRGAN/inputs/0014.jpg \
    --iters 5 --warmup 2
```

Results are written to `results/<stem>_input_1024.png` and
`results/<stem>_neuron_4x_4096.png`. Measured on trn2 with the LNC=1 bf16
128×128 NEFF replicated to 8 NeuronCores:

| Phase | Time |
|-------|-----:|
| Read image + center crop + Lanczos resize to 1024×1024 | 119 ms |
| Preprocess (HWC uint8 BGR → CHW bf16) + tile into 64 blocks | 31 ms |
| Load 8 pinned replicas (one-time startup) | 15,351 ms |
| **Neuron forward (64 tiles, 8 cores, median over 5 iters)** | **447 ms** |
| Stitch 64 output tiles + write 4K PNG | 826 ms |
| **Steady-state per image (excludes one-time replica load)** | **~1.4 s** |

Per-tile latency: **6.99 ms/tile**, throughput: **143 tiles/s**. The 5 timed
iterations were `[430, 448, 447, 449, 442] ms` — stable to ±2%. Compared to
the fp32 single-core baseline (48 ms/tile), the pinned 8-core bf16 pipeline
is **~7× faster** on the same hardware; going to 32 cores pushes it to
~1.67 ms/tile (see `BENCHMARK_PINNED_MULTICORE.md`).

### Before / after

Input (1024×1024, `inputs/0014.jpg` resized):

![Input 1024×1024](results/0014_input_1024.jpg)

Output (4096×4096, RealESRGAN_x4plus on Neuron):

![Neuron 4x output](results/0014_neuron_4x_4096.jpg)

> **Seam caveat**: this demo uses non-overlapping tiles so you will see faint
> grid lines at 128-pixel boundaries in the output. Production pipelines should
> use the overlapping tile loop from the upstream `RealESRGANer.tile_process`
> (configurable `tile_pad`, typically 10–16 pixels) — the per-tile latency is
> unchanged, only the number of tiles grows by a small constant.

## Running the Tests

CPU unit tests (no Neuron hardware required):

```bash
cd neuronx-distributed-inference/contrib/models/Real-ESRGAN
python -m pytest test/unit/test_architecture.py --forked
```

Neuron integration test (requires Trn/Inf instance + checkpoint + input image):

```bash
export REAL_ESRGAN_MODEL_PATH=/home/ubuntu/models/real-esrgan/RealESRGAN_x4plus.pth
export REAL_ESRGAN_INPUT_IMAGE=/tmp/Real-ESRGAN/inputs/0014.jpg
python -m pytest test/integration/test_model.py --forked
```

All configuration (model name, checkpoint, input image, tile size, compiled
artifact dir, output dir) can be overridden via env vars or CLI flags — see
the top of `test/integration/test_model.py` for the full list.

## Compatibility Matrix

| Instance/Version | 2.20+         | 2.19 and earlier |
|------------------|---------------|------------------|
| Trn1/Trn2        | Expected to work (traces cleanly with `torch_neuronx.trace`) | Not tested |
| Inf2             | Expected to work | Not tested |

## Maintainer

Community contribution.
