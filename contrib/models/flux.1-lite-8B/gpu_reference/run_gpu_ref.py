# Copyright Amazon Web Services and its Affiliates. All Rights Reserved.
# ==============================================================================
"""Stock-diffusers GPU reference for FLUX.1-lite-8B, at this model's Neuron settings.

Runs on a CUDA box with no Neuron dependency at all -- the point is to compare
this repository's Neuron path against an unmodified ``diffusers`` pipeline, which
is what someone reporting "the GPU image looks different" is running.

Mirrors ``src/generate.py``'s defaults: 1024x1024, 28 steps,
``guidance_scale=3.5``, no true classifier-free guidance,
``max_sequence_length=512``, seed 42 on a **CPU** generator, BF16, batch 1.

Two things make the comparison mean what it looks like, and both are options here:

* ``--dtype``. BF16 is what the Neuron path uses, but a GPU BF16 kernel
  accumulates matmuls in BF16 while Trainium accumulates in FP32, so GPU BF16 is
  the least accurate of the three precisions and diverges from *everything*
  including a GPU FP32 run of this same script. Use ``--dtype fp32`` for a
  reference to measure against.
* ``--latents-from``. A fixed seed does *not* give the same initial noise at two
  dtypes: ``randn`` draws different numbers, not the same numbers rounded. To
  attribute a difference to arithmetic rather than to the noise, draw the latents
  once and pass them to every run.

Each run writes the image plus an ``.npz`` of the tensors that let a difference be
located: the CLIP pooled embedding, the T5 prompt embedding, the initial latents
and the packed latents handed to the VAE. ``compare_precision.py`` reads those.

Usage:
    python run_gpu_ref.py -c /path/to/flux.1-lite-8B --steps 4,8,28 --tag bf16

    # precision comparison from bit-identical initial latents
    L=gpu_ref_out/bf16_28steps_seed42_g3.5.npz
    for d in bf16 fp16 fp32; do
        python run_gpu_ref.py -c /path/to/flux.1-lite-8B --dtype $d \
            --latents-from $L --tag fix_$d
    done
"""

import argparse
import contextlib
import json
import os
import time

import numpy as np
import torch
from diffusers import FluxPipeline

# Must stay identical to src/generate.py's DEFAULT_PROMPT. It is duplicated rather
# than imported because importing that module pulls in the Neuron stack, which is
# not installed on a GPU box.
DEFAULT_PROMPT = (
    "A close-up photo of a red panda wearing tiny round glasses, reading a "
    "leather-bound book in a cozy library, warm afternoon light, shallow "
    "depth of field"
)

DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-c", "--checkpoint-dir", required=True)
    parser.add_argument("-p", "--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--dtype", default="bf16", choices=list(DTYPES))
    parser.add_argument(
        "--gen-device",
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device of the latents' generator. src/generate.py uses cpu; a cuda "
        "generator draws unrelated noise from the same seed.",
    )
    parser.add_argument("-n", "--steps", default="28", help="Comma-separated to sweep.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("-g", "--guidance-scale", type=float, default=3.5)
    parser.add_argument("-hh", "--height", type=int, default=1024)
    parser.add_argument("-w", "--width", type=int, default=1024)
    parser.add_argument("--max-sequence-length", type=int, default=512)
    parser.add_argument(
        "--attn",
        default="default",
        choices=["default", "math", "flash", "efficient"],
        help="SDPA backend. Changing it alone moves the image by 34-37 dB, which "
        "is the scale of GPU-internal kernel choice.",
    )
    parser.add_argument(
        "--latents-from",
        default=None,
        help="An .npz from an earlier run; its initial_latents are reused so runs "
        "of different dtype start from bit-identical noise.",
    )
    parser.add_argument("--repeat", type=int, default=1, help="Rerun, to check determinism.")
    parser.add_argument("--cpu-offload", action="store_true", help="If FP32 does not fit.")
    parser.add_argument("--tag", default=None)
    parser.add_argument("--out-dir", default="gpu_ref_out")
    return parser.parse_args()


def attn_context(name: str):
    """SDPA backend selection, or a no-op for whatever torch would pick."""
    if name == "default":
        return contextlib.nullcontext()
    from torch.nn.attention import SDPBackend, sdpa_kernel

    return sdpa_kernel(
        {
            "math": SDPBackend.MATH,
            "flash": SDPBackend.FLASH_ATTENTION,
            "efficient": SDPBackend.EFFICIENT_ATTENTION,
        }[name]
    )


def main() -> None:
    args = parse_args()
    dtype = DTYPES[args.dtype]
    os.makedirs(args.out_dir, exist_ok=True)
    tag = args.tag or f"{args.dtype}_gen{args.gen_device}_{args.attn}"

    pipe = FluxPipeline.from_pretrained(args.checkpoint_dir, torch_dtype=dtype)
    # diffusers 0.32 forwards unknown kwargs into **kwargs and drops them, so a
    # misspelled dtype argument leaves the pipeline in FP32 silently. Assert.
    assert pipe.transformer.dtype == dtype, f"transformer is {pipe.transformer.dtype}"
    assert pipe.text_encoder_2.dtype == dtype, f"T5 is {pipe.text_encoder_2.dtype}"
    if args.cpu_offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to("cuda")
    pipe.set_progress_bar_config(disable=True)

    captured: dict = {}
    real_prepare_latents = pipe.prepare_latents
    real_encode_prompt = pipe.encode_prompt

    def prepare_latents(*args_, **kwargs):
        latents, ids = real_prepare_latents(*args_, **kwargs)
        captured["initial_latents"] = latents.detach().float().cpu().numpy()
        return latents, ids

    def encode_prompt(*args_, **kwargs):
        embeds, pooled, text_ids = real_encode_prompt(*args_, **kwargs)
        captured["t5_embeds"] = embeds.detach().float().cpu().numpy()
        captured["clip_pooled"] = pooled.detach().float().cpu().numpy()
        return embeds, pooled, text_ids

    def grab_latents(_pipe, _i, _t, kwargs):
        captured["final_latents"] = kwargs["latents"].detach().float().cpu().numpy()
        return kwargs

    pipe.prepare_latents = prepare_latents
    pipe.encode_prompt = encode_prompt

    fixed_latents = None
    if args.latents_from:
        array = np.load(args.latents_from)["initial_latents"]
        fixed_latents = torch.from_numpy(array).to("cuda", dtype)
        print(f"initial latents {tuple(fixed_latents.shape)} from {args.latents_from}")

    results = []
    for steps in [int(s) for s in args.steps.split(",")]:
        for rep in range(args.repeat):
            generator = torch.Generator(args.gen_device).manual_seed(args.seed)
            torch.cuda.synchronize()
            start = time.perf_counter()
            with attn_context(args.attn):
                output = pipe(
                    args.prompt,
                    height=args.height,
                    width=args.width,
                    guidance_scale=args.guidance_scale,
                    num_inference_steps=steps,
                    generator=generator,
                    max_sequence_length=args.max_sequence_length,
                    latents=fixed_latents,
                    callback_on_step_end=grab_latents,
                    output_type="pil",
                )
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start

            suffix = f"_rep{rep}" if args.repeat > 1 else ""
            base = f"{tag}_{steps}steps_seed{args.seed}_g{args.guidance_scale}{suffix}"
            image = output.images[0]
            image.save(os.path.join(args.out_dir, base + ".png"))
            np.savez_compressed(
                os.path.join(args.out_dir, base + ".npz"),
                image=np.asarray(image),
                **captured,
            )
            print(
                f"{base}: {elapsed:.2f} s ({elapsed / steps * 1e3:.0f} ms/step)",
                flush=True,
            )
            results.append(
                {"tag": tag, "steps": steps, "rep": rep, "seconds": round(elapsed, 3)}
            )

    with open(os.path.join(args.out_dir, f"{tag}_timing.json"), "w") as handle:
        json.dump(results, handle, indent=2)


if __name__ == "__main__":
    main()
