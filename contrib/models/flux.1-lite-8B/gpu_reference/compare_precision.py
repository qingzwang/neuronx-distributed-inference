# Copyright Amazon Web Services and its Affiliates. All Rights Reserved.
# ==============================================================================
"""Compare images and intermediate tensors from several runs, pairwise.

Takes ``label=path`` pairs, where ``path`` is either an ``.npz`` written by
``run_gpu_ref.py`` (image plus embeddings and latents) or a plain image, and
prints every pair's PSNR and mean absolute difference. Images of different sizes
are compared at ``--size``, since the committed Neuron samples are downscaled;
the resampling filter moves the numbers by under 0.5 dB, so it changes no
ranking, but pass ``--filter`` to check that on your own data.

Usage:
    python compare_precision.py \
        "GPU FP32=gpu_ref_out/fix_fp32_28steps_seed42_g3.5.npz" \
        "GPU FP16=gpu_ref_out/fix_fp16_28steps_seed42_g3.5.npz" \
        "GPU BF16=gpu_ref_out/fix_bf16_28steps_seed42_g3.5.npz" \
        "Neuron TP=4=../samples/flux_lite_1024px_28steps_tp4.png" \
        --figure cmp.png
"""

import argparse
import itertools
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont

TENSOR_KEYS = ["clip_pooled", "t5_embeds", "initial_latents", "final_latents"]
FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "runs",
        nargs="+",
        help="label=path pairs. path is an .npz from run_gpu_ref.py, or an image. "
        "A label may contain '=' as long as the path does not.",
    )
    parser.add_argument("--size", type=int, default=576, help="Common comparison size.")
    parser.add_argument(
        "--filter", default="LANCZOS", choices=["LANCZOS", "BICUBIC", "BILINEAR", "BOX"]
    )
    parser.add_argument("--figure", default=None, help="Write a labelled strip here.")
    parser.add_argument("--figure-tile", type=int, default=376)
    return parser.parse_args()


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return float("inf") if mse == 0 else 10 * np.log10(255.0**2 / mse)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a.ravel().astype(np.float64)
    b = b.ravel().astype(np.float64)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def load(path: str) -> dict:
    if path.endswith(".npz"):
        data = np.load(path)
        return {k: data[k] for k in data.files}
    return {"image": np.asarray(Image.open(path).convert("RGB"))}


def resize(image: np.ndarray, size: int, filt: str) -> np.ndarray:
    if image.shape[0] == size and image.shape[1] == size:
        return image
    return np.asarray(Image.fromarray(image).resize((size, size), getattr(Image, filt)))


def main() -> None:
    args = parse_args()
    runs = {}
    for spec in args.runs:
        label, _, path = spec.rpartition("=")
        assert label and path, f"expected label=path, got {spec!r}"
        runs[label] = load(path)

    labels = list(runs)
    width = max(len(label) for label in labels) + 2

    sizes = {r["image"].shape[:2] for r in runs.values()}
    if len(sizes) > 1:
        print(
            f"images differ in size ({sorted(sizes)}), comparing at "
            f"{args.size}x{args.size} with {args.filter}\n"
        )
    images = {
        label: resize(run["image"], args.size, args.filter) for label, run in runs.items()
    }

    print("Image difference, PSNR dB / mean absolute difference per 255\n")
    print(" " * width + "".join(f"{label:>22s}" for label in labels))
    for a in labels:
        row = f"{a:{width}s}"
        for b in labels:
            if a == b:
                row += f"{'-':>22s}"
            else:
                mean = np.abs(images[a].astype(float) - images[b].astype(float)).mean()
                row += f"{psnr(images[a], images[b]):>15.1f} /{mean:5.1f}"
        print(row)

    tensor_pairs = [
        (a, b)
        for a, b in itertools.combinations(labels, 2)
        if all(k in runs[a] and k in runs[b] for k in TENSOR_KEYS)
    ]
    if tensor_pairs:
        print("\nIntermediate tensors, cosine similarity\n")
        print(" " * (width * 2 + 3) + "".join(f"{k:>18s}" for k in TENSOR_KEYS))
        for a, b in tensor_pairs:
            row = f"{a:{width}s} v {b:{width}s}"
            for key in TENSOR_KEYS:
                row += f"{cosine(runs[a][key], runs[b][key]):>18.6f}"
            print(row)

    if args.figure:
        side, gap, band = args.figure_tile, 8, 28
        font = ImageFont.truetype(FONT_PATH, 13) if os.path.exists(FONT_PATH) else None
        canvas = Image.new(
            "RGB", (side * len(labels) + gap * (len(labels) - 1), side + band), "white"
        )
        draw = ImageDraw.Draw(canvas)
        for i, label in enumerate(labels):
            tile = Image.fromarray(runs[label]["image"]).resize(
                (side, side), getattr(Image, args.filter)
            )
            canvas.paste(tile, (i * (side + gap), band))
            text_width = draw.textlength(label, font=font)
            draw.text(
                (i * (side + gap) + (side - text_width) / 2, 7),
                label,
                fill=(32, 32, 32),
                font=font,
            )
        canvas.save(args.figure)
        print(f"\nwrote {args.figure}")


if __name__ == "__main__":
    main()
