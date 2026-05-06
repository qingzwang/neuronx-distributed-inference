"""Real-ESRGAN modeling code adapted for AWS Neuron (Trainium / Inferentia).

The reference PyTorch implementation lives at https://github.com/xinntao/Real-ESRGAN.
Real-ESRGAN is a purely convolutional super-resolution model, so it fits the
single-device ``torch_neuronx.trace`` compilation path instead of NxD's tensor
parallel ``NeuronBaseModel`` flow (which targets transformer/LLM workloads).

This file therefore re-implements the two architectures used by Real-ESRGAN,
``RRDBNet`` and ``SRVGGNetCompact``, stripped of the ``basicsr`` dependency and
of training-only init code, plus helpers that:

* ``build_model`` — instantiates the correct architecture for a given model
  name and loads an official ``.pth`` checkpoint.
* ``NeuronRealESRGAN`` — a thin wrapper that traces the network for a fixed
  input resolution with ``torch_neuronx.trace`` and exposes ``compile`` /
  ``save`` / ``load`` / ``__call__`` helpers.
* ``preprocess_image`` / ``postprocess_image`` — convert between an HWC uint8
  BGR image (as returned by ``cv2.imread``) and the NCHW float tensor the model
  expects.

The goal is parity with the upstream inference output while keeping the Neuron
entry point self-contained inside ``contrib/models/Real-ESRGAN``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Architectures
# ---------------------------------------------------------------------------


def _pixel_unshuffle(x: torch.Tensor, scale: int) -> torch.Tensor:
    """Inverse of ``nn.PixelShuffle`` — kept local so we don't pull in basicsr."""
    b, c, hh, hw = x.size()
    out_channel = c * (scale ** 2)
    assert hh % scale == 0 and hw % scale == 0, (
        f"pixel_unshuffle requires H/W divisible by scale={scale}, got {hh}x{hw}"
    )
    h = hh // scale
    w = hw // scale
    x_view = x.view(b, c, h, scale, w, scale)
    return x_view.permute(0, 1, 3, 5, 2, 4).reshape(b, out_channel, h, w)


class ResidualDenseBlock(nn.Module):
    def __init__(self, num_feat: int = 64, num_grow_ch: int = 32):
        super().__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)
        # inplace=False keeps the graph compatible with tracing.
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    def __init__(self, num_feat: int, num_grow_ch: int = 32):
        super().__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return out * 0.2 + x


class RRDBNet(nn.Module):
    """RRDB-based generator used by RealESRGAN_x{2,4}plus and the anime 6B variant."""

    def __init__(
        self,
        num_in_ch: int = 3,
        num_out_ch: int = 3,
        scale: int = 4,
        num_feat: int = 64,
        num_block: int = 23,
        num_grow_ch: int = 32,
    ):
        super().__init__()
        self.scale = scale
        if scale == 2:
            num_in_ch = num_in_ch * 4
        elif scale == 1:
            num_in_ch = num_in_ch * 16
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = nn.Sequential(
            *[RRDB(num_feat=num_feat, num_grow_ch=num_grow_ch) for _ in range(num_block)]
        )
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.scale == 2:
            feat = _pixel_unshuffle(x, scale=2)
        elif self.scale == 1:
            feat = _pixel_unshuffle(x, scale=4)
        else:
            feat = x
        feat = self.conv_first(feat)
        body_feat = self.conv_body(self.body(feat))
        feat = feat + body_feat
        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode="nearest")))
        feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode="nearest")))
        out = self.conv_last(self.lrelu(self.conv_hr(feat)))
        return out


class SRVGGNetCompact(nn.Module):
    """Compact VGG-style generator used by realesr-animevideov3 / general-x4v3."""

    def __init__(
        self,
        num_in_ch: int = 3,
        num_out_ch: int = 3,
        num_feat: int = 64,
        num_conv: int = 16,
        upscale: int = 4,
        act_type: str = "prelu",
    ):
        super().__init__()
        self.upscale = upscale

        body = nn.ModuleList()
        body.append(nn.Conv2d(num_in_ch, num_feat, 3, 1, 1))
        body.append(self._make_activation(act_type, num_feat))
        for _ in range(num_conv):
            body.append(nn.Conv2d(num_feat, num_feat, 3, 1, 1))
            body.append(self._make_activation(act_type, num_feat))
        body.append(nn.Conv2d(num_feat, num_out_ch * upscale * upscale, 3, 1, 1))
        self.body = body
        self.upsampler = nn.PixelShuffle(upscale)

    @staticmethod
    def _make_activation(act_type: str, num_feat: int) -> nn.Module:
        if act_type == "relu":
            return nn.ReLU(inplace=False)
        if act_type == "prelu":
            return nn.PReLU(num_parameters=num_feat)
        if act_type == "leakyrelu":
            return nn.LeakyReLU(negative_slope=0.1, inplace=False)
        raise ValueError(f"Unsupported activation {act_type}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        for layer in self.body:
            out = layer(out)
        out = self.upsampler(out)
        base = F.interpolate(x, scale_factor=self.upscale, mode="nearest")
        out = out + base
        return out


# ---------------------------------------------------------------------------
# Model presets: name -> (factory, default checkpoint URL, net scale)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelPreset:
    name: str
    url: str
    net_scale: int
    factory: callable  # type: ignore[type-arg]


def _factory_rrdb_23():
    return RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)


def _factory_rrdb_x2():
    return RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)


def _factory_rrdb_anime6():
    return RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)


def _factory_srvgg_anime():
    return SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4, act_type="prelu")


def _factory_srvgg_general():
    return SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type="prelu")


MODEL_PRESETS = {
    "RealESRGAN_x4plus": ModelPreset(
        "RealESRGAN_x4plus",
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
        4,
        _factory_rrdb_23,
    ),
    "RealESRNet_x4plus": ModelPreset(
        "RealESRNet_x4plus",
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth",
        4,
        _factory_rrdb_23,
    ),
    "RealESRGAN_x4plus_anime_6B": ModelPreset(
        "RealESRGAN_x4plus_anime_6B",
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
        4,
        _factory_rrdb_anime6,
    ),
    "RealESRGAN_x2plus": ModelPreset(
        "RealESRGAN_x2plus",
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
        2,
        _factory_rrdb_x2,
    ),
    "realesr-animevideov3": ModelPreset(
        "realesr-animevideov3",
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth",
        4,
        _factory_srvgg_anime,
    ),
    "realesr-general-x4v3": ModelPreset(
        "realesr-general-x4v3",
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth",
        4,
        _factory_srvgg_general,
    ),
}


def _extract_state_dict(checkpoint) -> dict:
    """Real-ESRGAN checkpoints are either a plain state_dict or a dict with
    ``params_ema`` / ``params`` keys — normalise to a state_dict."""
    if isinstance(checkpoint, dict):
        for key in ("params_ema", "params"):
            if key in checkpoint:
                return checkpoint[key]
    return checkpoint


def build_model(model_name: str, model_path: Optional[str] = None) -> Tuple[nn.Module, ModelPreset]:
    """Instantiate the correct architecture for ``model_name`` and load weights
    from ``model_path`` if supplied."""
    if model_name not in MODEL_PRESETS:
        raise ValueError(
            f"Unknown model_name {model_name!r}. Expected one of {list(MODEL_PRESETS)}"
        )
    preset = MODEL_PRESETS[model_name]
    model = preset.factory()
    if model_path is not None and os.path.isfile(model_path):
        checkpoint = torch.load(model_path, map_location="cpu")
        state_dict = _extract_state_dict(checkpoint)
        model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model, preset


# ---------------------------------------------------------------------------
# Neuron wrapper
# ---------------------------------------------------------------------------


TRACED_FILE_NAME = "model_neuron.pt"


class NeuronRealESRGAN:
    """Wrap a Real-ESRGAN generator so it can be compiled for Neuron with a
    fixed input resolution and then re-used for inference.

    Neuron trace requires a concrete input shape; callers should pick an
    ``(H, W)`` that matches their intended workload (for example the tiles
    produced by the upstream ``RealESRGANer`` tiling loop).
    """

    def __init__(
        self,
        model_name: str,
        model_path: Optional[str] = None,
        dtype: torch.dtype = torch.float32,
    ):
        self.model_name = model_name
        self.dtype = dtype
        self.model, self.preset = build_model(model_name, model_path)
        self.model = self.model.to(dtype=dtype)
        self.traced: Optional[torch.jit.ScriptModule] = None
        self.input_shape: Optional[Tuple[int, int, int, int]] = None

    # -- compilation ------------------------------------------------------

    def compile(
        self,
        input_shape: Tuple[int, int, int, int],
        compiler_workdir: Optional[str] = None,
        compiler_args: Optional[list] = None,
    ) -> torch.jit.ScriptModule:
        """Trace and compile the model for Neuron at ``input_shape``.

        ``input_shape`` is ``(batch, 3, H, W)``. Must satisfy H and W divisible
        by 4 so ``pixel_unshuffle`` works for the x1 variant; for x2/x4 only
        divisibility by 2 is needed.
        """
        import torch_neuronx  # Lazy import — only needed on Neuron hosts.

        example = torch.zeros(input_shape, dtype=self.dtype)
        trace_kwargs = {}
        if compiler_workdir is not None:
            trace_kwargs["compiler_workdir"] = compiler_workdir
        if compiler_args is not None:
            trace_kwargs["compiler_args"] = compiler_args
        self.traced = torch_neuronx.trace(self.model, example, **trace_kwargs)
        self.input_shape = input_shape
        return self.traced

    def save(self, path: str) -> None:
        if self.traced is None:
            raise RuntimeError("Must call .compile() before .save()")
        os.makedirs(path, exist_ok=True)
        torch.jit.save(self.traced, os.path.join(path, TRACED_FILE_NAME))

    def load(self, path: str) -> torch.jit.ScriptModule:
        # Importing torch_neuronx registers the custom ``__torch__.torch.classes.neuron.Model``
        # class that traced NEFF files reference — without it ``torch.jit.load`` raises
        # "Unknown type name" when the artifact was produced on another process.
        import torch_neuronx  # noqa: F401
        artifact = os.path.join(path, TRACED_FILE_NAME)
        if not os.path.isfile(artifact):
            raise FileNotFoundError(f"Traced model not found at {artifact}")
        self.traced = torch.jit.load(artifact)
        return self.traced

    # -- inference --------------------------------------------------------

    def __call__(self, pixel_tensor: torch.Tensor) -> torch.Tensor:
        if self.traced is not None:
            return self.traced(pixel_tensor.to(self.dtype))
        # Fall back to the eager PyTorch model — useful for CPU parity checks.
        with torch.no_grad():
            return self.model(pixel_tensor.to(self.dtype))


# ---------------------------------------------------------------------------
# Pre/post processing
# ---------------------------------------------------------------------------


def preprocess_image(img_bgr: np.ndarray, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Convert an HWC uint8 BGR image (as read by ``cv2.imread``) into an NCHW
    float tensor in ``[0, 1]``, matching the upstream ``RealESRGANer`` logic."""
    if img_bgr.dtype == np.uint16:
        max_range = 65535.0
    else:
        max_range = 255.0
    img = img_bgr.astype(np.float32) / max_range
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    if img.shape[2] == 4:
        # Drop alpha here; callers that need alpha handling should do it
        # separately (see RealESRGANer.enhance for the reference approach).
        img = img[:, :, :3]
    # HWC -> CHW, add batch dim.
    tensor = torch.from_numpy(np.transpose(img, (2, 0, 1))).unsqueeze(0)
    return tensor.to(dtype=dtype)


def postprocess_image(output: torch.Tensor) -> np.ndarray:
    """Convert the model's NCHW float tensor back to an HWC uint8 BGR image."""
    out = output.detach().float().clamp_(0, 1).squeeze(0).cpu().numpy()
    out = np.transpose(out, (1, 2, 0))
    out = (out * 255.0).round().astype(np.uint8)
    return out
