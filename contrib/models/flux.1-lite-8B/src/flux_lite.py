# Copyright Amazon Web Services and its Affiliates. All Rights Reserved.
# ==============================================================================
"""FLUX.1-lite-8B on Neuron, via NxDI's FLUX implementation.

FLUX.1-lite-8B is Freepik's distillation of FLUX.1-dev: the same
``FluxTransformer2DModel``, pruned from 19 double-stream (MMDiT) blocks to 8 while
keeping all 38 single-stream blocks, and guidance-distilled the same way.

**No modeling code is needed for it.** NxDI's FLUX backbone takes ``num_layers``
and ``num_single_layers`` from the checkpoint's ``config.json`` via
``load_diffusers_config``, so an 8-block checkpoint builds an 8-block model and
the tensor-parallel sharding follows the attention head count, which lite shares
with dev (24 heads x 128). Verified end to end at 1024x1024 with TP=4 on trn2.

What this module adds around that:

* checkpoint validation, so a wrong checkpoint fails with a sentence instead of a
  shape error deep in weight loading -- in particular FLUX.1-schnell, which is
  guidance-*free* and needs a different sampling path, not just other weights
* a TP-degree check against the head count, since a degree that does not divide
  24 fails obscurely
* per-stage and per-step latency measurement. The stock example times whole
  images only, which cannot tell prompt encoding from denoising from VAE decode.
"""

import json
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from statistics import fmean, median
from typing import Any, Callable, Optional

import torch

from neuronx_distributed_inference.models.diffusers.flux.application import (
    NeuronFluxApplication,
    create_flux_config,
    get_flux_parallelism_config,
)

# What a FLUX.1-lite-8B transformer config must say. num_layers is the whole
# difference from FLUX.1-dev (19); the rest is shared and is checked because the
# TP sharding and the static shapes depend on it.
LITE_NUM_LAYERS = 8
LITE_NUM_SINGLE_LAYERS = 38
LITE_NUM_ATTENTION_HEADS = 24
LITE_ATTENTION_HEAD_DIM = 128

# The T5 branch is trained up to 512 tokens and NeuronFluxApplication compiles
# for exactly that.
MAX_SEQUENCE_LENGTH = 512

DEFAULT_CHECKPOINT = "Freepik/flux.1-lite-8B"

# Both trn1 and trn2 have 24 attention heads to divide, so these are just the
# core counts that keep a whole device busy.
DEFAULT_TP_DEGREE = {"trn1": 8, "trn2": 4}


@dataclass
class StageLatency:
    """Per-stage wall-clock breakdown of one generation, in milliseconds.

    ``denoise_ms`` is the sum of ``step_ms``; each entry there is one backbone
    invocation, which is what tensor parallelism actually changes.
    """

    encode_ms: float = 0.0
    decode_ms: float = 0.0
    step_ms: list[float] = field(default_factory=list)
    total_ms: float = 0.0

    @property
    def denoise_ms(self) -> float:
        return sum(self.step_ms)

    @property
    def mean_step_ms(self) -> float:
        return fmean(self.step_ms) if self.step_ms else 0.0

    @property
    def median_step_ms(self) -> float:
        return median(self.step_ms) if self.step_ms else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "encode_ms": round(self.encode_ms, 2),
            "denoise_ms": round(self.denoise_ms, 2),
            "decode_ms": round(self.decode_ms, 2),
            "other_ms": round(
                self.total_ms - self.encode_ms - self.denoise_ms - self.decode_ms, 2
            ),
            "total_ms": round(self.total_ms, 2),
            "steps": len(self.step_ms),
            "mean_step_ms": round(self.mean_step_ms, 2),
            "median_step_ms": round(self.median_step_ms, 2),
            "step_ms": [round(s, 2) for s in self.step_ms],
        }


def read_transformer_config(checkpoint_dir: str) -> dict[str, Any]:
    """Load the transformer's ``config.json`` from a diffusers checkpoint."""
    path = os.path.join(checkpoint_dir, "transformer", "config.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. checkpoint_dir must be a diffusers FluxPipeline "
            "folder containing transformer/, text_encoder/, text_encoder_2/ and "
            f"vae/ subfolders; got {checkpoint_dir!r}."
        )
    with open(path) as handle:
        return json.load(handle)


def validate_checkpoint(checkpoint_dir: str, strict: bool = True) -> dict[str, Any]:
    """Check that a checkpoint is a FLUX.1-lite-8B-shaped, guidance-distilled FLUX.

    Args:
        checkpoint_dir: Local path to the diffusers checkpoint.
        strict: Require the lite block counts. Set False to accept any
            guidance-distilled FLUX (FLUX.1-dev works, just slower).

    Returns:
        The transformer config dict.

    Raises:
        FileNotFoundError: If the folder is not a diffusers FLUX checkpoint.
        ValueError: If the checkpoint is guidance-free (FLUX.1-schnell), if its
            attention geometry differs from what the FLUX path assumes, or if
            ``strict`` and the block counts are not lite's.
    """
    config = read_transformer_config(checkpoint_dir)

    if not config.get("guidance_embeds", False):
        raise ValueError(
            "this checkpoint has guidance_embeds=False, i.e. it is a "
            "guidance-free FLUX such as FLUX.1-schnell. Those take no guidance "
            "embedding and are sampled differently, so they need their own "
            "pipeline rather than a different checkpoint here."
        )

    heads = config.get("num_attention_heads")
    head_dim = config.get("attention_head_dim")
    if (heads, head_dim) != (LITE_NUM_ATTENTION_HEADS, LITE_ATTENTION_HEAD_DIM):
        raise ValueError(
            f"expected {LITE_NUM_ATTENTION_HEADS} attention heads of "
            f"{LITE_ATTENTION_HEAD_DIM} dims, got {heads} x {head_dim}. The "
            "tensor-parallel sharding and the compiled shapes are derived from "
            "these, so a different geometry is untested here."
        )

    if strict:
        layers = config.get("num_layers")
        single_layers = config.get("num_single_layers")
        if (layers, single_layers) != (LITE_NUM_LAYERS, LITE_NUM_SINGLE_LAYERS):
            raise ValueError(
                f"expected FLUX.1-lite-8B's {LITE_NUM_LAYERS} double + "
                f"{LITE_NUM_SINGLE_LAYERS} single blocks, got {layers} + "
                f"{single_layers}. This is likely FLUX.1-dev (19 + 38), which "
                "does run through the same path -- pass strict=False to accept "
                "it, and expect roughly 1.6x the step latency."
            )
    return config


def validate_tp_degree(tp_degree: int, num_heads: int = LITE_NUM_ATTENTION_HEADS) -> None:
    """Check that the backbone's attention heads divide evenly across ranks.

    Raises:
        ValueError: If ``tp_degree`` does not divide ``num_heads``, listing the
            degrees that do.
    """
    if tp_degree < 1:
        raise ValueError(f"tp_degree={tp_degree} must be at least 1.")
    if num_heads % tp_degree != 0:
        valid = [d for d in range(1, num_heads + 1) if num_heads % d == 0]
        raise ValueError(
            f"tp_degree={tp_degree} does not divide the backbone's {num_heads} "
            f"attention heads. Valid degrees: {valid}."
        )


def default_tp_degree(instance_type: str) -> int:
    """The TP degree that fills one device of ``instance_type``."""
    if instance_type not in DEFAULT_TP_DEGREE:
        raise ValueError(
            f"unknown instance_type {instance_type!r}; expected one of "
            f"{sorted(DEFAULT_TP_DEGREE)}."
        )
    return DEFAULT_TP_DEGREE[instance_type]


def build_application(
    checkpoint_dir: str = DEFAULT_CHECKPOINT,
    height: int = 1024,
    width: int = 1024,
    tp_degree: Optional[int] = None,
    instance_type: str = "trn2",
    dtype: torch.dtype = torch.bfloat16,
    strict: bool = True,
    context_parallel_enabled: bool = False,
    cfg_parallel_enabled: bool = False,
) -> NeuronFluxApplication:
    """Build a ``NeuronFluxApplication`` for a FLUX.1-lite-8B checkpoint.

    Validates first, so a wrong checkpoint or TP degree is reported before the
    minutes spent compiling. The application still needs ``compile()`` and
    ``load()``; see :func:`compile_and_load`.

    Args:
        checkpoint_dir: Local path to the diffusers checkpoint.
        height: Output height in pixels; fixed at compile time.
        width: Output width in pixels; fixed at compile time.
        tp_degree: Backbone tensor-parallel degree. Defaults to filling the
            device for ``instance_type``.
        instance_type: ``"trn1"`` or ``"trn2"``, only used for that default.
        dtype: Compute dtype.
        strict: Passed to :func:`validate_checkpoint`.
        context_parallel_enabled: Split the image sequence across two data-parallel
            groups; doubles world size.
        cfg_parallel_enabled: Run the conditional and unconditional passes of true
            classifier-free guidance in parallel; doubles world size.

    Returns:
        An unloaded ``NeuronFluxApplication``.
    """
    validate_checkpoint(checkpoint_dir, strict=strict)

    if tp_degree is None:
        tp_degree = default_tp_degree(instance_type)
    validate_tp_degree(tp_degree)

    world_size = get_flux_parallelism_config(
        tp_degree,
        context_parallel_enabled=context_parallel_enabled,
        cfg_parallel_enabled=cfg_parallel_enabled,
    )

    clip_config, t5_config, backbone_config, decoder_config = create_flux_config(
        checkpoint_dir,
        world_size,
        tp_degree,
        dtype,
        height,
        width,
        cfg_parallel_enabled=cfg_parallel_enabled,
        context_parallel_enabled=context_parallel_enabled,
    )

    return NeuronFluxApplication(
        model_path=checkpoint_dir,
        text_encoder_config=clip_config,
        text_encoder2_config=t5_config,
        backbone_config=backbone_config,
        decoder_config=decoder_config,
        height=height,
        width=width,
    )


def compile_and_load(
    app: NeuronFluxApplication, compiled_model_path: str
) -> float:
    """Compile any missing component and load them all onto the device.

    Components already present under ``compiled_model_path`` are reused, so a
    second call is cheap.

    Returns:
        Seconds spent, for reporting.
    """
    start = time.perf_counter()
    app.compile(compiled_model_path)
    app.load(compiled_model_path)
    return time.perf_counter() - start


@contextmanager
def measure_stages(app: NeuronFluxApplication, latency: StageLatency):
    """Time the pipeline's components for the duration of the block.

    Wraps the four Neuron submodels the pipeline calls, rather than editing the
    pipeline: prompt encoding is CLIP plus T5, one backbone call is one denoising
    step, and the VAE decoder runs once. Restores the originals on exit, so a
    failure inside the block cannot leave the application instrumented.

    Each submodel's ``forward`` is what gets replaced, not the attribute holding
    it -- diffusers' ``encode_prompt`` reads ``self.text_encoder.dtype``, so the
    modules have to stay modules.

    Args:
        app: A loaded application.
        latency: Filled in as the block runs.
    """
    pipe = app.pipe

    def timed(fn: Callable, sink: Callable[[float], None]) -> Callable:
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                sink((time.perf_counter() - start) * 1e3)

        return wrapper

    def add_encode(ms: float) -> None:
        latency.encode_ms += ms

    def add_step(ms: float) -> None:
        latency.step_ms.append(ms)

    def add_decode(ms: float) -> None:
        latency.decode_ms += ms

    hooks = [
        (pipe.text_encoder, add_encode),
        (pipe.text_encoder_2, add_encode),
        (pipe.transformer, add_step),
        (pipe.vae.decoder, add_decode),
    ]
    originals = [(module, module.forward) for module, _ in hooks]
    try:
        for module, sink in hooks:
            module.forward = timed(module.forward, sink)
        yield latency
    finally:
        for module, original in originals:
            module.forward = original


def generate(
    app: NeuronFluxApplication,
    prompt: str,
    num_inference_steps: int = 28,
    guidance_scale: float = 3.5,
    seed: Optional[int] = None,
    height: Optional[int] = None,
    width: Optional[int] = None,
    output_type: str = "pil",
    **kwargs: Any,
) -> tuple[Any, StageLatency]:
    """Generate one image and measure where the time went.

    Args:
        app: A loaded application.
        prompt: Text prompt.
        num_inference_steps: Denoising steps.
        guidance_scale: Distilled guidance embedding value. Not classifier-free
            guidance -- there is no negative pass unless ``true_cfg_scale`` is
            passed through ``kwargs`` -- so cost does not depend on it.
        seed: Host RNG seed for the initial latents; ``None`` is nondeterministic.
        height: Overrides the application's height. Must match what was compiled.
        width: Overrides the application's width. Must match what was compiled.
        output_type: ``"pil"``, ``"np"``, ``"pt"`` or ``"latent"``.
        **kwargs: Forwarded to the pipeline (``negative_prompt``,
            ``true_cfg_scale``, ...).

    Returns:
        ``(image, latency)``, where ``image`` is the first output of the batch.
    """
    generator = None
    if seed is not None:
        generator = torch.Generator("cpu").manual_seed(seed)

    latency = StageLatency()
    start = time.perf_counter()
    with measure_stages(app, latency):
        output = app(
            prompt,
            height=height or app.height,
            width=width or app.width,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
            output_type=output_type,
            **kwargs,
        )
    latency.total_ms = (time.perf_counter() - start) * 1e3

    result = output.images if hasattr(output, "images") else output
    return result[0], latency
