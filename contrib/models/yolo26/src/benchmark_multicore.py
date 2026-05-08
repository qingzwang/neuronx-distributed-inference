"""Multi-core throughput benchmark for the compiled YOLO26 module.

The single-core latency is already established in `benchmark.py`; this
script replicates the compiled module across N NeuronCores with
`torch_neuronx.DataParallel` and measures:

 - images/sec (throughput) at a sustained batch
 - per-image latency at batch=N (wall clock / images)

Each core runs the same NEFF; DataParallel scatters along dim 0 and
gathers results.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch_neuronx

from yolo26_common import (
    ASSETS_DIR,
    BENCHMARK_DIR,
    COMPILED_DIR,
    DEFAULT_IMGSZ,
    preprocess_image,
)


def _build_batch(image_paths: List[Path], batch_size: int, imgsz: int) -> torch.Tensor:
    """Build a (batch_size, 3, imgsz, imgsz) tensor by cycling through images."""
    tensors = []
    for i in range(batch_size):
        p = image_paths[i % len(image_paths)]
        t, _, _ = preprocess_image(str(p), imgsz=imgsz)
        tensors.append(t)
    return torch.cat(tensors, dim=0)


def bench(
    compiled_path: str,
    num_cores: int,
    image_paths: List[Path],
    iters: int,
    warmup: int,
    imgsz: int,
) -> dict:
    """Place one NEFF copy per NeuronCore and dispatch concurrently.

    `torch_neuronx.DataParallel` often routes every call through the core that
    first loaded the module unless each replica is placed explicitly. Loading
    under `neuron_cores_context(start_nc=i, nc_count=1)` pins replica i to
    core i; a shared `ThreadPoolExecutor` then fires one traced call per core
    in parallel. This matches the placement used by the reference AWS Neuron
    YOLO26 benchmark and scales much closer to linear.
    """
    from concurrent.futures import ThreadPoolExecutor

    modules = []
    for i in range(num_cores):
        with torch_neuronx.experimental.placement.neuron_cores_context(
            start_nc=i, nc_count=1,
        ):
            modules.append(torch.jit.load(compiled_path))
    pool = ThreadPoolExecutor(max_workers=max(2 * num_cores, 4))

    # Each core processes one image per step (per-core batch is baked into the
    # NEFF). Total images per step = num_cores.
    t_single, _, _ = preprocess_image(str(image_paths[0]), imgsz=imgsz)

    def step() -> None:
        futs = [pool.submit(m, t_single) for m in modules]
        for f in futs:
            f.result()

    for _ in range(warmup):
        step()

    timings: List[float] = []
    for _ in range(iters):
        t0 = time.perf_counter()
        step()
        timings.append((time.perf_counter() - t0) * 1000.0)

    arr = np.asarray(timings)
    batch_mean_ms = float(arr.mean())
    batch_p50_ms = float(np.median(arr))
    per_image_ms = batch_mean_ms / num_cores
    throughput = num_cores / (batch_mean_ms / 1000.0)

    return {
        "num_cores": num_cores,
        "device_ids": [f"nc:{i}" for i in range(num_cores)],
        "batch_size": num_cores,
        "batch_mean_ms": batch_mean_ms,
        "batch_p50_ms": batch_p50_ms,
        "per_image_ms": per_image_ms,
        "throughput_img_per_sec": throughput,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--compiled", default=str(COMPILED_DIR / "yolo26n_neuron_fp32.pt"),
    )
    parser.add_argument(
        "--core-counts",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8, 16, 32],
        help="NeuronCore counts to sweep",
    )
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ)
    parser.add_argument(
        "--out-json",
        default=str(BENCHMARK_DIR / "benchmark_multicore.json"),
    )
    parser.add_argument(
        "--out-md",
        default=str(BENCHMARK_DIR / "benchmark_multicore.md"),
    )
    args = parser.parse_args()

    image_paths = sorted(ASSETS_DIR.glob("*.jpg"))
    assert image_paths, "no images in assets/"

    rows = []
    for n in args.core_counts:
        print(f"[mc] benchmarking {n} NeuronCore(s) ...")
        row = bench(
            args.compiled, n, image_paths,
            iters=args.iters, warmup=args.warmup, imgsz=args.imgsz,
        )
        rows.append(row)
        print(
            f"  batch={row['batch_size']}  "
            f"batch mean={row['batch_mean_ms']:.2f} ms  "
            f"per-image={row['per_image_ms']:.2f} ms  "
            f"throughput={row['throughput_img_per_sec']:.1f} img/s"
        )

    summary = {
        "model": "yolo26n",
        "imgsz": args.imgsz,
        "iters": args.iters,
        "warmup": args.warmup,
        "compiled": args.compiled,
        "instance": "trn2.48xlarge",
        "rows": rows,
    }

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"[mc] wrote {out_json}")

    # Markdown report
    md = ["# YOLO26-nano multi-core throughput", ""]
    md.append(f"- Compiled artifact: `{summary['compiled']}`")
    md.append(f"- Input size: {summary['imgsz']}x{summary['imgsz']}")
    md.append(f"- Instance: {summary['instance']}")
    md.append(f"- Iters/warmup: {summary['iters']}/{summary['warmup']}")
    md.append("")
    md.append("Each step feeds one image per NeuronCore (batch = num_cores) through "
              "`torch_neuronx.DataParallel`. Per-image latency is the wall-clock "
              "duration of a step divided by the batch size; throughput is cores / step time.")
    md.append("")
    md.append("| cores | batch | step mean (ms) | step p50 (ms) | per-image (ms) | throughput (img/s) |")
    md.append("|------:|------:|---------------:|--------------:|---------------:|-------------------:|")
    baseline = rows[0]["throughput_img_per_sec"] if rows else 1.0
    for r in rows:
        md.append(
            f"| {r['num_cores']} | {r['batch_size']} | "
            f"{r['batch_mean_ms']:.2f} | {r['batch_p50_ms']:.2f} | "
            f"{r['per_image_ms']:.2f} | {r['throughput_img_per_sec']:.1f} |"
        )
    md.append("")
    md.append("| cores | throughput vs 1 core |")
    md.append("|------:|---------------------:|")
    for r in rows:
        md.append(f"| {r['num_cores']} | {r['throughput_img_per_sec'] / baseline:.2f}x |")

    Path(args.out_md).write_text("\n".join(md) + "\n")
    print(f"[mc] wrote {args.out_md}")


if __name__ == "__main__":
    main()
