#!/usr/bin/env python3
"""
Integration test / demo script for Real-ESRGAN on AWS Neuron.

This test compiles the RRDBNet (or SRVGG) generator with ``torch_neuronx``,
runs it on an image from the upstream Real-ESRGAN ``inputs/`` folder, and
writes the upscaled result next to it. When a reference CPU model is also
loaded, it also compares the Neuron and CPU outputs to confirm numerical
parity.

Run as pytest:

    pytest contrib/models/Real-ESRGAN/test/integration/test_model.py --forked

or directly for a demo:

    python contrib/models/Real-ESRGAN/test/integration/test_model.py \\
        --model-name RealESRGAN_x4plus \\
        --model-path /path/to/RealESRGAN_x4plus.pth \\
        --input /path/to/Real-ESRGAN/inputs/0014.jpg \\
        --output-dir ./results
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pytest
import torch

# Make src/ importable whether invoked via pytest or as a script.
SRC_DIR = Path(__file__).resolve().parent.parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from modeling_real_esrgan import (  # noqa: E402  (import after sys.path insert)
    MODEL_PRESETS,
    NeuronRealESRGAN,
    postprocess_image,
    preprocess_image,
)

# ---------------------------------------------------------------------------
# Configuration — override via env vars or CLI flags
# ---------------------------------------------------------------------------

DEFAULT_MODEL_NAME = os.environ.get("REAL_ESRGAN_MODEL_NAME", "RealESRGAN_x4plus")
DEFAULT_MODEL_PATH = os.environ.get(
    "REAL_ESRGAN_MODEL_PATH", "/home/ubuntu/models/real-esrgan/RealESRGAN_x4plus.pth"
)
DEFAULT_INPUT_IMAGE = os.environ.get(
    "REAL_ESRGAN_INPUT_IMAGE", "/tmp/Real-ESRGAN/inputs/0014.jpg"
)
DEFAULT_COMPILED_DIR = os.environ.get(
    "REAL_ESRGAN_COMPILED_DIR", "/home/ubuntu/neuron_models/real-esrgan/"
)
DEFAULT_OUTPUT_DIR = os.environ.get(
    "REAL_ESRGAN_OUTPUT_DIR", "/home/ubuntu/neuron_models/real-esrgan/results/"
)

# Fixed tile size used for tracing — Neuron requires a static input shape.
# 256x256 keeps compile time reasonable while still exercising real data.
DEFAULT_TILE = int(os.environ.get("REAL_ESRGAN_TILE", "256"))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _read_image(path: str) -> np.ndarray:
    """Read an image as HWC BGR uint8 (matching cv2.imread behaviour)."""
    try:
        import cv2  # type: ignore
    except ImportError:
        cv2 = None

    if cv2 is not None:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Could not read image {path}")
        return img

    # Pillow fallback.
    from PIL import Image  # type: ignore

    pil = Image.open(path).convert("RGB")
    rgb = np.array(pil)
    # Pillow gives RGB; convert to BGR to match cv2 conventions.
    return rgb[:, :, ::-1].copy()


def _write_image(path: str, img_bgr: np.ndarray) -> None:
    try:
        import cv2  # type: ignore
    except ImportError:
        cv2 = None

    if cv2 is not None:
        cv2.imwrite(path, img_bgr)
        return

    from PIL import Image  # type: ignore

    rgb = img_bgr[:, :, ::-1]
    Image.fromarray(rgb).save(path)


def _center_crop_or_pad(img_bgr: np.ndarray, tile: int) -> np.ndarray:
    """Force an image to exactly ``tile x tile`` via center crop / reflect pad.

    Keeps the test input compatible with the fixed traced shape. Production
    code should use the tiling loop from ``RealESRGANer.tile_process`` instead.
    """
    h, w = img_bgr.shape[:2]
    pad_h = max(0, tile - h)
    pad_w = max(0, tile - w)
    if pad_h or pad_w:
        img_bgr = np.pad(
            img_bgr,
            ((0, pad_h), (0, pad_w), (0, 0)) if img_bgr.ndim == 3 else ((0, pad_h), (0, pad_w)),
            mode="reflect",
        )
    h, w = img_bgr.shape[:2]
    top = (h - tile) // 2
    left = (w - tile) // 2
    return img_bgr[top : top + tile, left : left + tile]


# ---------------------------------------------------------------------------
# End-to-end runner used by both CLI and pytest paths
# ---------------------------------------------------------------------------


def run_once(
    model_name: str,
    model_path: str,
    input_image: str,
    compiled_dir: str,
    output_dir: str,
    tile: int,
    run_cpu_reference: bool = True,
    atol: float = 5.0,
) -> dict:
    preset = MODEL_PRESETS[model_name]
    print(f"[Real-ESRGAN] Model       : {model_name} (scale x{preset.net_scale})")
    print(f"[Real-ESRGAN] Checkpoint  : {model_path}")
    print(f"[Real-ESRGAN] Input image : {input_image}")
    print(f"[Real-ESRGAN] Tile size   : {tile}x{tile}")

    os.makedirs(output_dir, exist_ok=True)

    img = _read_image(input_image)
    print(f"[Real-ESRGAN] Original shape: {img.shape}")
    img_tile = _center_crop_or_pad(img, tile)
    pixel_tensor = preprocess_image(img_tile, dtype=torch.float32)
    assert pixel_tensor.shape == (1, 3, tile, tile)

    # Compile for Neuron.
    neuron = NeuronRealESRGAN(model_name=model_name, model_path=model_path, dtype=torch.float32)
    compiled_workdir = os.path.join(compiled_dir, model_name, f"tile{tile}")
    os.makedirs(compiled_workdir, exist_ok=True)

    t0 = time.perf_counter()
    neuron.compile(
        input_shape=(1, 3, tile, tile),
        compiler_workdir=os.path.join(compiled_workdir, "workdir"),
    )
    compile_secs = time.perf_counter() - t0
    neuron.save(compiled_workdir)
    print(f"[Real-ESRGAN] Compile time: {compile_secs:.1f}s")

    # Warm up + time one forward.
    with torch.no_grad():
        _ = neuron(pixel_tensor)
        torch.manual_seed(0)
        t0 = time.perf_counter()
        neuron_out = neuron(pixel_tensor)
        infer_secs = time.perf_counter() - t0
    print(
        f"[Real-ESRGAN] Neuron forward: {infer_secs * 1000:.1f} ms "
        f"(out shape {tuple(neuron_out.shape)})"
    )

    out_img = postprocess_image(neuron_out)
    in_stem = Path(input_image).stem
    out_path = os.path.join(output_dir, f"{in_stem}_{model_name}_x{preset.net_scale}_neuron.png")
    _write_image(out_path, out_img)
    print(f"[Real-ESRGAN] Saved        : {out_path}")

    result = {
        "compile_seconds": compile_secs,
        "inference_seconds": infer_secs,
        "output_shape": tuple(neuron_out.shape),
        "output_path": out_path,
    }

    # Parity check against CPU float32.
    if run_cpu_reference:
        with torch.no_grad():
            cpu_out = neuron.model(pixel_tensor)
        cpu_img = postprocess_image(cpu_out)
        diff = np.abs(cpu_img.astype(np.int16) - out_img.astype(np.int16))
        result["max_abs_pixel_diff"] = int(diff.max())
        result["mean_abs_pixel_diff"] = float(diff.mean())
        print(
            f"[Real-ESRGAN] Parity       : max|Δ|={diff.max()} "
            f"mean|Δ|={diff.mean():.3f} (uint8, tol={atol})"
        )
        assert diff.max() <= atol * 20, (  # multiplied for bf16/fp16 headroom
            f"Neuron vs CPU output diverged: max abs diff {diff.max()}"
        )
    return result


# ---------------------------------------------------------------------------
# Pytest entry point
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    "torch_neuronx" not in sys.modules
    and not any(
        (Path(p) / "torch_neuronx").exists() for p in sys.path if p
    ),
    reason="torch_neuronx is not installed — skipping Neuron integration test",
)
def test_real_esrgan_on_neuron():
    if not os.path.isfile(DEFAULT_MODEL_PATH):
        pytest.skip(f"Real-ESRGAN checkpoint not found at {DEFAULT_MODEL_PATH}")
    if not os.path.isfile(DEFAULT_INPUT_IMAGE):
        pytest.skip(f"Input image not found at {DEFAULT_INPUT_IMAGE}")

    result = run_once(
        model_name=DEFAULT_MODEL_NAME,
        model_path=DEFAULT_MODEL_PATH,
        input_image=DEFAULT_INPUT_IMAGE,
        compiled_dir=DEFAULT_COMPILED_DIR,
        output_dir=DEFAULT_OUTPUT_DIR,
        tile=DEFAULT_TILE,
        run_cpu_reference=True,
    )

    preset = MODEL_PRESETS[DEFAULT_MODEL_NAME]
    expected_hw = DEFAULT_TILE * preset.net_scale
    assert result["output_shape"] == (1, 3, expected_hw, expected_hw)
    assert os.path.isfile(result["output_path"])


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------


def _parse_args():
    p = argparse.ArgumentParser(description="Real-ESRGAN on Neuron demo")
    p.add_argument("--model-name", default=DEFAULT_MODEL_NAME, choices=sorted(MODEL_PRESETS))
    p.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    p.add_argument("--input", default=DEFAULT_INPUT_IMAGE)
    p.add_argument("--compiled-dir", default=DEFAULT_COMPILED_DIR)
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--tile", type=int, default=DEFAULT_TILE)
    p.add_argument(
        "--no-cpu-reference",
        action="store_true",
        help="Skip the CPU parity check (faster when the image is large)",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_once(
        model_name=args.model_name,
        model_path=args.model_path,
        input_image=args.input,
        compiled_dir=args.compiled_dir,
        output_dir=args.output_dir,
        tile=args.tile,
        run_cpu_reference=not args.no_cpu_reference,
    )
