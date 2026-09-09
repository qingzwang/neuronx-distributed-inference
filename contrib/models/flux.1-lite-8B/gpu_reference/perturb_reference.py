# Copyright Amazon Web Services and its Affiliates. All Rights Reserved.
# ==============================================================================
"""How large a numeric difference does it take to move the image by a given amount?

This exists to answer "is the Neuron-vs-GPU difference just rounding?" with a
number rather than an assurance. It injects a controlled perturbation into an
otherwise unmodified GPU run and reports what the image does, so a claimed
component-level accuracy can be checked against an observed image difference.

The useful result: a perturbation of the size ``README.md`` measures for one
backbone step (cos 0.99989) moves the image by only ~43 dB when it is drawn
independently each step, and ~36 dB when its direction is frozen -- while GPU BF16
sits 21.7 dB from GPU FP32. Real low-precision error is correlated with the
signal, so it steers the trajectory far more than injected noise of the same
per-step magnitude. Small injected perturbations therefore *cannot* be used to
argue a difference is benign; run the precisions against each other instead
(``run_gpu_ref.py --latents-from``).

Modes:
    iid     fresh random direction each step, at relative L2 size ``--rel``
    frozen  one random direction, reused every step (coherent, like rounding)
    scale   multiply the backbone output by (1 + rel), a purely systematic bias
    t5pad   perturb only T5's padded positions; ``--rel`` is a per-element std.
            FLUX feeds all 512 positions to the backbone unmasked, and the
            padded ones are where a T5 port differs most.
    t5real  perturb only the real prompt tokens, for contrast with t5pad

Usage:
    python perturb_reference.py -c /path/to/flux.1-lite-8B --mode frozen \
        --rels 0.0015,0.0148
"""

import argparse
import os

import numpy as np
import torch
from diffusers import FluxPipeline

from run_gpu_ref import DEFAULT_PROMPT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-c", "--checkpoint-dir", required=True)
    parser.add_argument("-p", "--prompt", default=DEFAULT_PROMPT)
    parser.add_argument(
        "--mode", required=True, choices=["iid", "frozen", "scale", "t5pad", "t5real"]
    )
    parser.add_argument(
        "--rels",
        default="0.0015,0.0148",
        help="Perturbation sizes to sweep. Relative L2 for the backbone modes "
        "(0.0148 gives cos 0.99989); a per-element std for the T5 modes.",
    )
    parser.add_argument("-n", "--steps", type=int, default=28)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("-g", "--guidance-scale", type=float, default=3.5)
    parser.add_argument("-hh", "--height", type=int, default=1024)
    parser.add_argument("-w", "--width", type=int, default=1024)
    parser.add_argument("--noise-seed", type=int, default=1234)
    parser.add_argument("--out-dir", default="gpu_ref_out")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    backbone_modes = ("iid", "frozen", "scale")

    pipe = FluxPipeline.from_pretrained(args.checkpoint_dir, torch_dtype=torch.bfloat16)
    assert pipe.transformer.dtype == torch.bfloat16
    pipe.to("cuda")
    pipe.set_progress_bar_config(disable=True)

    num_real_tokens = len(
        pipe.tokenizer_2(args.prompt, truncation=True, max_length=512)["input_ids"]
    )
    print(
        f"T5: {num_real_tokens} real tokens (incl. EOS), "
        f"{512 - num_real_tokens} padded positions"
    )

    state: dict = {"rel": 0.0, "direction": None, "cos": []}
    real_forward = pipe.transformer.forward
    real_encode_prompt = pipe.encode_prompt

    def perturbed_forward(*fargs, **fkwargs):
        output = real_forward(*fargs, **fkwargs)
        if args.mode not in backbone_modes or state["rel"] == 0.0:
            return output
        tensor = output[0] if isinstance(output, tuple) else output.sample
        clean = tensor.float()
        if args.mode == "scale":
            dirty = clean * (1.0 + state["rel"])
        else:
            if args.mode == "iid" or state["direction"] is None:
                generator = (
                    None
                    if args.mode == "iid"
                    else torch.Generator("cuda").manual_seed(args.noise_seed)
                )
                noise = torch.randn(
                    clean.shape, generator=generator, device=clean.device
                )
                state["direction"] = noise / noise.norm()
            dirty = clean + state["direction"] * (state["rel"] * clean.norm())
        dirty = dirty.to(tensor.dtype)
        state["cos"].append(
            float(
                clean.ravel() @ dirty.float().ravel() / (clean.norm() * dirty.float().norm())
            )
        )
        if isinstance(output, tuple):
            return (dirty,) + output[1:]
        output.sample = dirty
        return output

    def perturbed_encode_prompt(*eargs, **ekwargs):
        embeds, pooled, text_ids = real_encode_prompt(*eargs, **ekwargs)
        if args.mode in backbone_modes or state["rel"] == 0.0:
            return embeds, pooled, text_ids
        generator = torch.Generator("cuda").manual_seed(args.noise_seed)
        noise = torch.randn(
            embeds.shape, generator=generator, device=embeds.device, dtype=torch.float32
        )
        mask = torch.zeros_like(noise)
        if args.mode == "t5pad":
            mask[:, num_real_tokens:, :] = 1.0
        else:
            mask[:, :num_real_tokens, :] = 1.0
        perturbed = embeds.float() + noise * mask * state["rel"]
        delta = (perturbed - embeds.float()).abs()
        print(
            f"  T5 perturbation: max|d| {delta.max():.3f}, mean|d| {delta.mean():.4f}, "
            f"embedding std {embeds.float().std():.4f}, cos "
            f"{float(perturbed.ravel() @ embeds.float().ravel() / (perturbed.norm() * embeds.float().norm())):.6f}"
        )
        return perturbed.to(embeds.dtype), pooled, text_ids

    pipe.transformer.forward = perturbed_forward
    pipe.encode_prompt = perturbed_encode_prompt

    for rel in [float(r) for r in args.rels.split(",")]:
        state.update(rel=rel, direction=None, cos=[])
        generator = torch.Generator("cpu").manual_seed(args.seed)
        output = pipe(
            args.prompt,
            height=args.height,
            width=args.width,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.steps,
            generator=generator,
            max_sequence_length=512,
        )
        tag = f"perturb_{args.mode}_{rel}_{args.steps}steps"
        output.images[0].save(os.path.join(args.out_dir, tag + ".png"))
        np.savez_compressed(
            os.path.join(args.out_dir, tag + ".npz"), image=np.asarray(output.images[0])
        )
        measured = f"{np.mean(state['cos']):.6f}" if state["cos"] else "n/a"
        print(f"{tag}: per-step cos {measured}", flush=True)


if __name__ == "__main__":
    main()
