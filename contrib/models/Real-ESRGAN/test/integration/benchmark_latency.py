#!/usr/bin/env python3
"""Benchmark Real-ESRGAN latency on AWS Neuron across batch sizes.

Compiles the RRDBNet x4 generator at a fixed tile size for each batch size in
a user-supplied list, then measures wall-clock latency per forward pass
(warmup runs, then ``iters`` timed runs). Writes a markdown report next to
this file with per-batch compile time and latency percentiles, plus
per-image throughput (images/s) for comparison.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

import torch

SRC_DIR = Path(__file__).resolve().parent.parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from modeling_real_esrgan import MODEL_PRESETS, NeuronRealESRGAN  # noqa: E402


def percentile(xs, p: float) -> float:
    if not xs:
        return float("nan")
    xs = sorted(xs)
    # Linear interp like numpy default.
    k = (len(xs) - 1) * p
    f = int(k)
    c = min(f + 1, len(xs) - 1)
    if f == c:
        return xs[f]
    return xs[f] + (xs[c] - xs[f]) * (k - f)


def benchmark_one(
    model_name: str,
    model_path: str,
    batch_size: int,
    tile: int,
    compiled_root: str,
    iters: int,
    warmup: int,
) -> dict:
    compiled_dir = os.path.join(compiled_root, model_name, f"bs{batch_size}_tile{tile}")
    os.makedirs(compiled_dir, exist_ok=True)

    neuron = NeuronRealESRGAN(model_name=model_name, model_path=model_path, dtype=torch.float32)
    shape = (batch_size, 3, tile, tile)

    t0 = time.perf_counter()
    neuron.compile(input_shape=shape, compiler_workdir=os.path.join(compiled_dir, "workdir"))
    compile_secs = time.perf_counter() - t0
    neuron.save(compiled_dir)

    example = torch.zeros(shape, dtype=torch.float32)

    for _ in range(warmup):
        neuron(example)

    latencies_ms = []
    for _ in range(iters):
        t0 = time.perf_counter()
        out = neuron(example)
        # Force materialisation — torch_neuronx returns a CPU tensor already,
        # but we touch a value to be safe against any lazy behavior.
        _ = out[0, 0, 0, 0].item()
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)

    preset = MODEL_PRESETS[model_name]
    mean_ms = statistics.fmean(latencies_ms)
    median_ms = statistics.median(latencies_ms)
    p90_ms = percentile(latencies_ms, 0.90)
    p99_ms = percentile(latencies_ms, 0.99)
    stdev_ms = statistics.pstdev(latencies_ms) if len(latencies_ms) > 1 else 0.0

    return {
        "model_name": model_name,
        "batch_size": batch_size,
        "tile": tile,
        "output_h": tile * preset.net_scale,
        "output_w": tile * preset.net_scale,
        "compile_seconds": compile_secs,
        "iters": iters,
        "warmup": warmup,
        "latency_ms": {
            "mean": mean_ms,
            "median": median_ms,
            "p90": p90_ms,
            "p99": p99_ms,
            "min": min(latencies_ms),
            "max": max(latencies_ms),
            "stdev": stdev_ms,
        },
        "throughput_images_per_s": (batch_size * 1000.0 / mean_ms) if mean_ms > 0 else float("nan"),
        "ms_per_image": mean_ms / batch_size,
        "raw_latencies_ms": latencies_ms,
    }


def format_markdown(results: list[dict], env_info: dict) -> str:
    lines = []
    lines.append("# Real-ESRGAN Neuron Latency Benchmark")
    lines.append("")
    lines.append(f"- **Model:** `{results[0]['model_name']}` (scale x{MODEL_PRESETS[results[0]['model_name']].net_scale})")
    lines.append(f"- **Tile (input) size:** {results[0]['tile']}x{results[0]['tile']}")
    lines.append(f"- **Output size per image:** {results[0]['output_h']}x{results[0]['output_w']}")
    lines.append(f"- **dtype:** fp32 weights / fp32 activations (as traced)")
    lines.append(f"- **Warmup runs:** {results[0]['warmup']}")
    lines.append(f"- **Timed iters:** {results[0]['iters']}")
    lines.append(f"- **torch_neuronx:** {env_info.get('torch_neuronx', 'unknown')}")
    lines.append(f"- **torch:** {env_info.get('torch', 'unknown')}")
    lines.append(f"- **instance:** {env_info.get('instance', 'unknown')}")
    lines.append(f"- **Benchmark date:** {env_info.get('date', 'unknown')}")
    lines.append("")
    lines.append("## Per-batch-size latency")
    lines.append("")
    lines.append("| Batch | Compile (s) | Mean (ms) | Median (ms) | p90 (ms) | p99 (ms) | Min (ms) | Max (ms) | Stdev (ms) | ms/image | Throughput (img/s) |")
    lines.append("|------:|------------:|----------:|------------:|---------:|---------:|---------:|---------:|-----------:|---------:|-------------------:|")
    for r in results:
        lat = r["latency_ms"]
        lines.append(
            "| {bs} | {comp:.1f} | {mean:.2f} | {med:.2f} | {p90:.2f} | {p99:.2f} | {mn:.2f} | {mx:.2f} | {sd:.2f} | {mpi:.2f} | {tp:.2f} |".format(
                bs=r["batch_size"],
                comp=r["compile_seconds"],
                mean=lat["mean"],
                med=lat["median"],
                p90=lat["p90"],
                p99=lat["p99"],
                mn=lat["min"],
                mx=lat["max"],
                sd=lat["stdev"],
                mpi=r["ms_per_image"],
                tp=r["throughput_images_per_s"],
            )
        )
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- Real-ESRGAN's RRDBNet is a pure convolutional generator, so batch-size support comes")
    lines.append("  \"for free\" from `torch_neuronx.trace` — we simply trace with the desired batch dim")
    lines.append("  and the compiler folds the batch into the NEFF. Each batch size produces a distinct")
    lines.append("  NEFF artifact and must be compiled separately.")
    lines.append("- Latencies are wall-clock around the `__call__` into the traced NEFF, including")
    lines.append("  any CPU-side copies. Warmup runs are discarded.")
    lines.append("- Throughput = `batch_size * 1000 / mean_ms`. On a convolutional network, you expect")
    lines.append("  per-image latency to be roughly flat across batch sizes once the compute fills the")
    lines.append("  device — crossover below that point is where batching pays off.")
    lines.append("- Compile time scales with batch size because the traced graph grows; NEFFs are")
    lines.append("  cached under `compiled_dir/<model>/bs<N>_tile<T>/`, so subsequent runs reuse them.")
    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="RealESRGAN_x4plus")
    parser.add_argument("--model-path", default="/home/ubuntu/models/real-esrgan/RealESRGAN_x4plus.pth")
    parser.add_argument("--tile", type=int, default=128)
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[1, 2, 4],
        help="Batch sizes to benchmark (each triggers a fresh compile)",
    )
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument(
        "--compiled-root",
        default="/home/ubuntu/neuron_models/real-esrgan/bench/",
    )
    parser.add_argument(
        "--output-md",
        default=str(Path(__file__).resolve().parent.parent.parent / "BENCHMARK.md"),
    )
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args()

    try:
        import torch_neuronx

        tnx_ver = torch_neuronx.__version__
    except Exception:
        tnx_ver = "unknown"

    env_info = {
        "torch": torch.__version__,
        "torch_neuronx": tnx_ver,
        "instance": os.environ.get("NEURON_INSTANCE_TYPE", os.uname().nodename),
        "date": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
    }

    results = []
    for bs in args.batch_sizes:
        print(f"\n=== Benchmarking batch_size={bs} ===", flush=True)
        r = benchmark_one(
            model_name=args.model_name,
            model_path=args.model_path,
            batch_size=bs,
            tile=args.tile,
            compiled_root=args.compiled_root,
            iters=args.iters,
            warmup=args.warmup,
        )
        results.append(r)
        lat = r["latency_ms"]
        print(
            f"[bs={bs}] compile={r['compile_seconds']:.1f}s  "
            f"mean={lat['mean']:.2f}ms  p90={lat['p90']:.2f}ms  p99={lat['p99']:.2f}ms  "
            f"ms/image={r['ms_per_image']:.2f}  throughput={r['throughput_images_per_s']:.2f} img/s",
            flush=True,
        )

    md = format_markdown(results, env_info)
    Path(args.output_md).write_text(md)
    print(f"\nWrote markdown report to {args.output_md}")

    if args.output_json:
        Path(args.output_json).write_text(json.dumps({"env": env_info, "results": results}, indent=2))
        print(f"Wrote JSON results to {args.output_json}")


if __name__ == "__main__":
    main()
