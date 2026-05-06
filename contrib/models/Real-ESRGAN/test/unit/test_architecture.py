"""CPU-only unit tests for the Real-ESRGAN architecture port.

These validate that the re-implemented ``RRDBNet`` / ``SRVGGNetCompact`` produce
the expected output shapes and consume official checkpoints without key
mismatches. They do not require Neuron hardware.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

SRC_DIR = Path(__file__).resolve().parent.parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from modeling_real_esrgan import (  # noqa: E402
    MODEL_PRESETS,
    RRDBNet,
    SRVGGNetCompact,
    build_model,
    postprocess_image,
    preprocess_image,
)


def test_rrdbnet_x4_output_shape():
    model = RRDBNet(num_in_ch=3, num_out_ch=3, scale=4, num_feat=64, num_block=2, num_grow_ch=32)
    model.eval()
    x = torch.randn(1, 3, 16, 16)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (1, 3, 64, 64)


def test_rrdbnet_x2_output_shape():
    model = RRDBNet(num_in_ch=3, num_out_ch=3, scale=2, num_feat=64, num_block=2, num_grow_ch=32)
    model.eval()
    x = torch.randn(1, 3, 16, 16)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (1, 3, 32, 32)


def test_srvgg_output_shape():
    model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=32, num_conv=2, upscale=4, act_type="prelu")
    model.eval()
    x = torch.randn(1, 3, 16, 16)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (1, 3, 64, 64)


def test_presets_declare_correct_scales():
    # Sanity check — all scale-2 variants should use net_scale=2, etc.
    assert MODEL_PRESETS["RealESRGAN_x2plus"].net_scale == 2
    for name in (
        "RealESRGAN_x4plus",
        "RealESRNet_x4plus",
        "RealESRGAN_x4plus_anime_6B",
        "realesr-animevideov3",
        "realesr-general-x4v3",
    ):
        assert MODEL_PRESETS[name].net_scale == 4


def test_preprocess_postprocess_roundtrip():
    rng = np.random.default_rng(0)
    img_bgr = rng.integers(0, 256, size=(32, 48, 3), dtype=np.uint8)
    tensor = preprocess_image(img_bgr)
    assert tensor.shape == (1, 3, 32, 48)
    assert tensor.dtype == torch.float32
    assert 0.0 <= tensor.min().item() and tensor.max().item() <= 1.0

    # A perfect round-trip (no model) should reproduce the input exactly.
    recovered = postprocess_image(tensor)
    assert recovered.shape == img_bgr.shape
    assert np.array_equal(recovered, img_bgr)


def test_build_model_without_weights():
    # Checkpoint not supplied — build_model should return a randomly-initialised net.
    model, preset = build_model("RealESRGAN_x4plus", model_path=None)
    assert preset.net_scale == 4
    assert isinstance(model, RRDBNet)


@pytest.mark.skipif(
    not (Path("/home/ubuntu/models/real-esrgan/RealESRGAN_x4plus.pth").exists()),
    reason="Real-ESRGAN checkpoint not available locally",
)
def test_build_model_loads_official_checkpoint():
    model, preset = build_model(
        "RealESRGAN_x4plus",
        model_path="/home/ubuntu/models/real-esrgan/RealESRGAN_x4plus.pth",
    )
    assert preset.net_scale == 4
    x = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (1, 3, 128, 128)
