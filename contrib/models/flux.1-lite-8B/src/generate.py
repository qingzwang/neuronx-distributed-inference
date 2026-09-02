# Copyright Amazon Web Services and its Affiliates. All Rights Reserved.
# ==============================================================================
"""Generate images with FLUX.1-lite-8B on Neuron, and report where time goes.

Usage:
    python contrib/models/flux.1-lite-8B/src/generate.py \
        --checkpoint-dir /path/to/flux.1-lite-8B \
        --prompt "a red panda reading a book in a cozy library" \
        --steps 28 --save-image

    # Sweep step counts, 3 measured requests each, JSON out
    python contrib/models/flux.1-lite-8B/src/generate.py \
        --checkpoint-dir /path/to/flux.1-lite-8B \
        --steps 4,8,28 --iterations 3 --json latency.json

The first request of a process is discarded by default: it pays one-off device
warmup that would otherwise be charged to the image you asked for. Use
``--no-warmup`` to see that cost instead.
"""

import argparse
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flux_lite import (  # noqa: E402
    DEFAULT_CHECKPOINT,
    build_application,
    compile_and_load,
    default_tp_degree,
    generate,
)

DEFAULT_PROMPT = (
    "A close-up photo of a red panda wearing tiny round glasses, reading a "
    "leather-bound book in a cozy library, warm afternoon light, shallow "
    "depth of field"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-c", "--checkpoint-dir", default=DEFAULT_CHECKPOINT)
    parser.add_argument("-p", "--prompt", default=DEFAULT_PROMPT)
    parser.add_argument(
        "--compiled-model-path",
        default="/tmp/flux_lite_8b/",
        help="Where compiled components live. Reused if already present.",
    )
    parser.add_argument("-hh", "--height", type=int, default=1024)
    parser.add_argument("-w", "--width", type=int, default=1024)
    parser.add_argument(
        "-n",
        "--steps",
        default="28",
        help="Denoising steps; comma-separated to sweep.",
    )
    parser.add_argument("-g", "--guidance-scale", type=float, default=3.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--tp-degree",
        type=int,
        default=None,
        help="Backbone tensor-parallel degree. Must divide 24 attention heads. "
        "Defaults to filling the device for --instance-type.",
    )
    parser.add_argument("-i", "--instance-type", default="trn2", choices=["trn1", "trn2"])
    parser.add_argument(
        "--iterations",
        type=int,
        default=1,
        help="Measured requests per step count, after warmup.",
    )
    parser.add_argument("--no-warmup", action="store_true")
    parser.add_argument("--save-image", action="store_true")
    parser.add_argument("--output-dir", default=".")
    parser.add_argument(
        "--allow-any-flux",
        action="store_true",
        help="Accept any guidance-distilled FLUX, not just lite's 8+38 blocks.",
    )
    parser.add_argument("--json", default=None, help="Write raw latencies here.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    step_counts = [int(s) for s in args.steps.split(",")]
    tp_degree = args.tp_degree or default_tp_degree(args.instance_type)

    app = build_application(
        checkpoint_dir=args.checkpoint_dir,
        height=args.height,
        width=args.width,
        tp_degree=tp_degree,
        instance_type=args.instance_type,
        strict=not args.allow_any_flux,
    )
    setup_s = compile_and_load(app, args.compiled_model_path)
    print(
        f"Ready in {setup_s:.1f} s "
        f"({args.height}x{args.width}, tp_degree={tp_degree})",
        flush=True,
    )

    if not args.no_warmup:
        print("Warming up...", flush=True)
        generate(app, args.prompt, num_inference_steps=step_counts[0], seed=args.seed)

    results = []
    for steps in step_counts:
        runs = []
        image = None
        for i in range(args.iterations):
            image, latency = generate(
                app,
                args.prompt,
                num_inference_steps=steps,
                guidance_scale=args.guidance_scale,
                seed=args.seed + i,
            )
            runs.append(latency)
            print(
                f"  [{steps} steps] iter {i}: {latency.total_ms / 1e3:.2f} s total, "
                f"{latency.median_step_ms:.0f} ms/step",
                flush=True,
            )

        if args.save_image and image is not None:
            os.makedirs(args.output_dir, exist_ok=True)
            path = os.path.join(
                args.output_dir, f"flux_lite_{args.height}px_{steps}steps.png"
            )
            image.save(path)
            print(f"  wrote {path}", flush=True)

        all_steps = [ms for r in runs for ms in r.step_ms]
        results.append(
            {
                "height": args.height,
                "width": args.width,
                "steps": steps,
                "tp_degree": tp_degree,
                "iterations": len(runs),
                "total_s_p50": round(
                    statistics.median([r.total_ms for r in runs]) / 1e3, 3
                ),
                "step_ms_p50": round(statistics.median(all_steps), 2),
                "encode_ms": round(statistics.fmean([r.encode_ms for r in runs]), 2),
                "denoise_ms": round(statistics.fmean([r.denoise_ms for r in runs]), 2),
                "decode_ms": round(statistics.fmean([r.decode_ms for r in runs]), 2),
                "runs": [r.as_dict() for r in runs],
            }
        )

    print(f"\n{'steps':>6} {'ms/step':>9} {'encode':>8} {'decode':>8} {'total s':>9}")
    for r in results:
        print(
            f"{r['steps']:>6} {r['step_ms_p50']:>9.1f} {r['encode_ms']:>8.1f} "
            f"{r['decode_ms']:>8.1f} {r['total_s_p50']:>9.2f}"
        )

    if args.json:
        with open(args.json, "w") as handle:
            json.dump(results, handle, indent=2)
        print(f"\nWrote {args.json}")


if __name__ == "__main__":
    main()
