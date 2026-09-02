# Copyright Amazon Web Services and its Affiliates. All Rights Reserved.
# ==============================================================================
"""Integration tests for FLUX.1-lite-8B on Neuron.

Run with a checkpoint on disk:

    FLUX_LITE_CHECKPOINT=/path/to/flux.1-lite-8B \
        pytest contrib/models/flux.1-lite-8B/test/integration/test_model.py -v -s

The tests are skipped when that variable is unset, so a collection run on a
machine without the checkpoint stays green.

On what is validated
--------------------
The contrib guide's sample test uses logit matching. A diffusion transformer has
no logits, so the analogue used here is the **denoised latent**: the same prompt
and schedule are run on Neuron and on CPU through stock diffusers, from the same
initial noise, and the two latents are compared.

The threshold is loose (cosine similarity, not equality). Both sides are BF16 and
the flow-matching ODE amplifies any per-step difference across steps, so exact
agreement is not available and the CPU side is not ground truth either. What this
does catch is real breakage -- a tensor-parallel sharding or weight loading bug
puts the similarity near zero, not near one.

Two traps this test has to avoid, both of which look exactly like a model bug:

* ``diffusers==0.32`` takes ``torch_dtype``, and ``from_pretrained`` accepts
  ``**kwargs``, so a stray ``dtype=`` is silently dropped and the reference runs
  in fp32.
* ``randn_tensor`` from a fixed seed draws *different numbers* at fp32 than at
  BF16 -- not the same numbers rounded. Passing the same seed to two pipelines of
  different dtype starts them from unrelated noise, which reads as ~0.8 cosine
  similarity no matter how correct the model is.

So the initial latents are built once here and handed to both sides explicitly.
"""

import os
import sys

import pytest
import torch

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src")
)

from flux_lite import (  # noqa: E402
    LITE_NUM_LAYERS,
    LITE_NUM_SINGLE_LAYERS,
    build_application,
    compile_and_load,
    default_tp_degree,
    generate,
    validate_checkpoint,
    validate_tp_degree,
)

CHECKPOINT = os.environ.get("FLUX_LITE_CHECKPOINT")
COMPILED_PATH = os.environ.get("FLUX_LITE_COMPILED_PATH", "/tmp/flux_lite_8b_test/")

# Small on purpose: the CPU reference runs the full 8B backbone per step, so a
# 512px / 4-step run keeps the test to a few minutes.
HEIGHT = WIDTH = 512
STEPS = 4
SEED = 7
PROMPT = "A close-up photo of a red panda wearing tiny round glasses"

# One backbone step against an fp32 CPU reference. Measured 0.99989 on trn2 with
# TP=4, so this has room for hardware and compiler variation while still failing
# hard on a sharding or weight-loading bug (those land near 0).
MIN_STEP_COSINE = 0.999

# End to end against a BF16 CPU reference, after the flow ODE has amplified four
# steps of BF16 divergence on *both* sides. Measured 0.9789, where CPU BF16 itself
# only reaches 0.9774 against fp32 -- so almost all of the gap is the reference's
# own error, and the threshold cannot be tight.
MIN_LATENT_COSINE = 0.95

requires_checkpoint = pytest.mark.skipif(
    CHECKPOINT is None,
    reason="set FLUX_LITE_CHECKPOINT to a diffusers FLUX.1-lite-8B folder",
)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _initial_latents(pipe, dtype: torch.dtype) -> torch.Tensor:
    """Packed initial latents for this test's shape, in ``dtype``.

    Built once and handed to every side of a comparison. A shared seed is not
    enough: ``randn_tensor`` draws different numbers at fp32 than at BF16, so
    seeding two pipelines of different dtype starts them from unrelated noise.
    """
    channels = pipe.transformer.config.in_channels // 4
    latent_h = 2 * (HEIGHT // (pipe.vae_scale_factor * 2))
    latent_w = 2 * (WIDTH // (pipe.vae_scale_factor * 2))
    noise = torch.randn(
        1,
        channels,
        latent_h,
        latent_w,
        generator=torch.Generator("cpu").manual_seed(SEED),
        dtype=torch.float32,
    ).to(dtype)
    return pipe._pack_latents(noise, 1, channels, latent_h, latent_w)


def _compare(actual: torch.Tensor, expected: torch.Tensor) -> tuple[float, float]:
    """Cosine similarity and mean absolute error, both flattened in fp32."""
    a = actual.detach().float().cpu().flatten()
    b = expected.detach().float().cpu().flatten()
    assert a.shape == b.shape, f"shape mismatch: {tuple(a.shape)} vs {tuple(b.shape)}"
    cosine = torch.nn.functional.cosine_similarity(a, b, dim=0).item()
    return cosine, (a - b).abs().mean().item()


# ----------------------------------------------------------------------
# Validation, no device needed
# ----------------------------------------------------------------------


@requires_checkpoint
def test_checkpoint_is_lite_shaped():
    config = validate_checkpoint(CHECKPOINT)
    assert config["num_layers"] == LITE_NUM_LAYERS
    assert config["num_single_layers"] == LITE_NUM_SINGLE_LAYERS
    assert config["guidance_embeds"] is True


def test_tp_degree_must_divide_head_count():
    for degree in (1, 2, 3, 4, 6, 8, 12, 24):
        validate_tp_degree(degree)
    for degree in (5, 7, 16, 32):
        with pytest.raises(ValueError, match="does not divide"):
            validate_tp_degree(degree)
    with pytest.raises(ValueError, match="at least 1"):
        validate_tp_degree(0)


def test_default_tp_degree_per_instance():
    assert default_tp_degree("trn2") == 4
    assert default_tp_degree("trn1") == 8
    with pytest.raises(ValueError, match="unknown instance_type"):
        default_tp_degree("inf2")


# ----------------------------------------------------------------------
# On device
# ----------------------------------------------------------------------


@pytest.fixture(scope="module")
def app():
    """Compile and load once for every on-device test in this module."""
    application = build_application(
        checkpoint_dir=CHECKPOINT,
        height=HEIGHT,
        width=WIDTH,
        tp_degree=default_tp_degree("trn2"),
    )
    compile_and_load(application, COMPILED_PATH)
    # Discard one request: the first pays device warmup.
    generate(application, PROMPT, num_inference_steps=1, seed=SEED)
    return application


@requires_checkpoint
def test_generates_an_image_of_the_requested_size(app):
    image, latency = generate(
        app, PROMPT, num_inference_steps=STEPS, seed=SEED
    )
    assert image.size == (WIDTH, HEIGHT)
    assert len(latency.step_ms) == STEPS, (
        f"expected one backbone call per step, timed {len(latency.step_ms)}"
    )
    assert latency.encode_ms > 0 and latency.decode_ms > 0
    print(f"\nlatency: {latency.as_dict()}")


@requires_checkpoint
def test_backbone_step_matches_cpu_fp32(app):
    """One denoising step against an fp32 CPU reference.

    This is the tight check, and the cheap one: a single backbone forward rather
    than a whole schedule. It is also the only place a BF16 reference would be
    actively misleading, so the comparison is against fp32 and the BF16 CPU run
    is included only as the baseline to beat.
    """
    from diffusers import FluxPipeline

    ref32 = FluxPipeline.from_pretrained(CHECKPOINT, torch_dtype=torch.float32)
    ref16 = FluxPipeline.from_pretrained(CHECKPOINT, torch_dtype=torch.bfloat16)
    for pipe, dtype in ((ref32, torch.float32), (ref16, torch.bfloat16)):
        pipe.set_progress_bar_config(disable=True)
        assert pipe.transformer.dtype == dtype, (
            f"reference loaded as {pipe.transformer.dtype}, wanted {dtype}; on "
            "diffusers 0.32 the kwarg is torch_dtype and anything else is "
            "silently ignored"
        )

    with torch.no_grad():
        embeds, pooled, _ = ref32.encode_prompt(
            prompt=PROMPT, prompt_2=None, device="cpu", max_sequence_length=512
        )

    latents = _initial_latents(ref32, torch.float32)
    latent_h = 2 * (HEIGHT // (ref32.vae_scale_factor * 2))
    latent_w = 2 * (WIDTH // (ref32.vae_scale_factor * 2))
    img_ids = ref32._prepare_latent_image_ids(
        1, latent_h // 2, latent_w // 2, "cpu", torch.float32
    )
    txt_ids = torch.zeros(embeds.shape[1], 3, dtype=torch.float32)
    timestep = torch.full((1,), 0.7, dtype=torch.float32)
    guidance = torch.full((1,), 3.5, dtype=torch.float32)

    def run_reference(pipe):
        dtype = pipe.transformer.dtype
        with torch.no_grad():
            return pipe.transformer(
                hidden_states=latents.to(dtype),
                timestep=timestep.to(dtype),
                guidance=guidance,
                pooled_projections=pooled.to(dtype),
                encoder_hidden_states=embeds.to(dtype),
                txt_ids=txt_ids,
                img_ids=img_ids,
                return_dict=False,
            )[0]

    with torch.no_grad(), app.pipe.transformer.image_rotary_emb_cache_context():
        neuron = app.pipe.transformer(
            hidden_states=latents.to(torch.bfloat16),
            timestep=timestep,
            guidance=guidance,
            pooled_projections=pooled.to(torch.bfloat16),
            encoder_hidden_states=embeds.to(torch.bfloat16),
            txt_ids=txt_ids,
            img_ids=img_ids,
            return_dict=False,
        )
    neuron = neuron[0] if isinstance(neuron, (tuple, list)) else neuron

    target = run_reference(ref32)
    cpu_bf16 = run_reference(ref16)

    neuron_cos, neuron_err = _compare(neuron, target)
    cpu_cos, cpu_err = _compare(cpu_bf16, target)
    print(
        f"\nvelocity vs fp32: neuron cos={neuron_cos:.6f} mean|d|={neuron_err:.6f} | "
        f"cpu-bf16 cos={cpu_cos:.6f} mean|d|={cpu_err:.6f}"
    )

    assert neuron_cos > MIN_STEP_COSINE, (
        f"backbone step cosine {neuron_cos:.6f} against fp32 is below "
        f"{MIN_STEP_COSINE}; that is a correctness failure, not BF16 noise"
    )
    # Self-calibrating: Trainium accumulates matmuls in fp32, so BF16 on device
    # should land closer to the fp32 reference than BF16 on CPU does. Measured
    # ~11x closer. If that inverts, something regressed in a way an absolute
    # threshold would not necessarily catch.
    assert neuron_err <= cpu_err, (
        f"Neuron BF16 is further from fp32 (mean|d|={neuron_err:.6f}) than CPU "
        f"BF16 is ({cpu_err:.6f})"
    )


@requires_checkpoint
def test_latents_match_cpu_diffusers(app):
    """Full schedule against a BF16 CPU reference, from the same initial noise.

    Loose by necessity -- see MIN_LATENT_COSINE. This is an integration check:
    scheduler, RoPE, packing and both text encoders all participate, and any of
    them being wired up wrongly shows here even though the per-step test passes.
    """
    from diffusers import FluxPipeline

    reference_pipe = FluxPipeline.from_pretrained(
        CHECKPOINT, torch_dtype=torch.bfloat16
    )
    reference_pipe.set_progress_bar_config(disable=True)
    assert reference_pipe.transformer.dtype == torch.bfloat16, (
        "reference pipeline is not BF16; on diffusers 0.32 the kwarg is "
        "torch_dtype and anything else is silently ignored"
    )

    initial_latents = _initial_latents(reference_pipe, torch.bfloat16)

    neuron_latent, _ = generate(
        app,
        PROMPT,
        num_inference_steps=STEPS,
        output_type="latent",
        latents=initial_latents,
    )
    with torch.no_grad():
        reference_latent = reference_pipe(
            prompt=PROMPT,
            height=HEIGHT,
            width=WIDTH,
            num_inference_steps=STEPS,
            guidance_scale=3.5,
            latents=initial_latents,
            output_type="latent",
            return_dict=False,
        )[0]

    cosine, mean_abs = _compare(neuron_latent, reference_latent)
    print(f"\nlatent cosine={cosine:.6f} mean|d|={mean_abs:.6f}")
    assert cosine > MIN_LATENT_COSINE, (
        f"latent cosine similarity {cosine:.6f} is below {MIN_LATENT_COSINE}; "
        "this is a correctness failure, not BF16 noise"
    )
