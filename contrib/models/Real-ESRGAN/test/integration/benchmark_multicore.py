#!/usr/bin/env python3
"""Multi-core + bf16 latency benchmark for Real-ESRGAN on Neuron.

Real-ESRGAN tiles are independent, so data-parallelism across NeuronCores is
"free": we replicate the same bs=1 NEFF on N cores and dispatch tiles round-robin
via ``torch_neuronx.DataParallel``. Orthogonal to that, compiling in bf16
instead of fp32 roughly halves per-tile latency on the same core.

This script produces ``BENCHMARK_MULTICORE.md`` comparing:

* fp32 single core (baseline, from the earlier resolution benchmark)
* fp32 across ``N`` cores for several N
* bf16 single core
* bf16 across ``N`` cores for several N

using the same 1K / 2K / 4K tiled-image workflow.
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


RESOLUTIONS = {"1K": 1024, "2K": 2048, "4K": 4096}


def synth_image(resolution: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.rand((1, 3, resolution, resolution), generator=g, dtype=torch.float32)


def build_tile_batch(image: torch.Tensor, tile: int) -> torch.Tensor:
    """Slice ``image`` into non-overlapping tile x tile chunks and stack on
    the batch dim, returning a ``(n_tiles, 3, tile, tile)`` tensor."""
    b, c, h, w = image.shape
    assert b == 1 and h % tile == 0 and w % tile == 0
    tiles = []
    for y in range(0, h, tile):
        for x in range(0, w, tile):
            tiles.append(image[0, :, y : y + tile, x : x + tile])
    return torch.stack(tiles, dim=0)


def run_dataparallel(
    dp_model,
    tile_batch: torch.Tensor,
    chunk: int,
    dtype: torch.dtype,
) -> float:
    """Feed ``tile_batch`` through ``dp_model`` in chunks of ``chunk`` tiles
    (chunk ~= num_cores so DataParallel can fan each chunk out across the
    replicas). Returns wall time in ms."""
    n = tile_batch.shape[0]
    t0 = time.perf_counter()
    for i in range(0, n, chunk):
        batch = tile_batch[i : i + chunk].to(dtype)
        out = dp_model(batch)
        _ = out[-1, 0, 0, 0].item()
    return (time.perf_counter() - t0) * 1000.0


def get_or_compile(
    model_name: str,
    model_path: str,
    tile: int,
    dtype: torch.dtype,
    compiled_root: str,
    force: bool = False,
) -> tuple[NeuronRealESRGAN, float]:
    tag = "fp32" if dtype is torch.float32 else "bf16"
    out_dir = os.path.join(compiled_root, model_name, f"bs1_tile{tile}_{tag}")
    traced = os.path.join(out_dir, "model_neuron.pt")
    neuron = NeuronRealESRGAN(model_name=model_name, model_path=model_path, dtype=dtype)
    if os.path.isfile(traced) and not force:
        print(f"  [load] cached {tag} NEFF from {out_dir}", flush=True)
        neuron.load(out_dir)
        compile_secs = 0.0
    else:
        os.makedirs(out_dir, exist_ok=True)
        print(f"  [compile] {tag} NEFF (this takes several minutes)...", flush=True)
        t0 = time.perf_counter()
        neuron.compile(
            input_shape=(1, 3, tile, tile),
            compiler_workdir=os.path.join(out_dir, "workdir"),
        )
        compile_secs = time.perf_counter() - t0
        neuron.save(out_dir)
        print(f"  [compile] {tag} took {compile_secs:.1f}s", flush=True)
    return neuron, compile_secs


def make_dataparallel(traced_model, device_ids: list[int]):
    import torch_neuronx

    dp = torch_neuronx.DataParallel(traced_model, device_ids=device_ids)
    return dp


def bench_single_core(neuron: NeuronRealESRGAN, tile_batch: torch.Tensor, dtype: torch.dtype) -> float:
    t0 = time.perf_counter()
    for i in range(tile_batch.shape[0]):
        chunk = tile_batch[i : i + 1].to(dtype)
        out = neuron(chunk)
        _ = out[-1, 0, 0, 0].item()
    return (time.perf_counter() - t0) * 1000.0


def bench_cfg(
    neuron: NeuronRealESRGAN,
    resolution: int,
    tile: int,
    dtype: torch.dtype,
    n_cores: int,
    iters: int,
) -> dict:
    image = synth_image(resolution)
    tile_batch = build_tile_batch(image, tile)
    expected = (resolution // tile) ** 2
    assert tile_batch.shape[0] == expected

    if n_cores == 1:
        model = neuron
        runner = lambda: bench_single_core(neuron, tile_batch, dtype)  # noqa: E731
    else:
        device_ids = list(range(n_cores))
        dp = make_dataparallel(neuron.traced, device_ids)
        model = dp
        chunk = n_cores
        runner = lambda: run_dataparallel(dp, tile_batch, chunk, dtype)  # noqa: E731

    # Warmup.
    runner()

    latencies = []
    for _ in range(iters):
        latencies.append(runner())

    return {
        "resolution": resolution,
        "n_tiles": expected,
        "dtype": "fp32" if dtype is torch.float32 else "bf16",
        "n_cores": n_cores,
        "iters": iters,
        "total_ms_mean": statistics.fmean(latencies),
        "total_ms_median": statistics.median(latencies),
        "total_ms_min": min(latencies),
        "total_ms_max": max(latencies),
        "ms_per_tile_mean": statistics.fmean(latencies) / expected,
    }


def format_markdown(all_results: list[dict], compile_info: dict, env_info: dict) -> str:
    lines = []
    lines.append("# Real-ESRGAN Neuron Multi-core + bf16 Benchmark")
    lines.append("")
    lines.append(f"- **Model:** `RealESRGAN_x4plus` (RRDBNet, scale x4)")
    lines.append(f"- **Tile size:** 128x128 input (512x512 output)")
    lines.append(f"- **Instance:** {env_info.get('instance')}")
    lines.append(f"- **torch:** {env_info.get('torch')}")
    lines.append(f"- **torch_neuronx:** {env_info.get('torch_neuronx')}")
    lines.append(f"- **Date:** {env_info.get('date')}")
    lines.append("")
    lines.append("## Compile times")
    lines.append("")
    lines.append("| dtype | compile seconds |")
    lines.append("|:------|----------------:|")
    for tag, secs in compile_info.items():
        if secs > 0:
            lines.append(f"| {tag} | {secs:.1f} |")
        else:
            lines.append(f"| {tag} | (cached) |")
    lines.append("")
    lines.append("## End-to-end tiled-image latency")
    lines.append("")
    lines.append("Each row runs one whole image by dispatching all tiles to `n_cores`")
    lines.append("NeuronCores via `torch_neuronx.DataParallel`. `ms/tile` is the effective")
    lines.append("per-tile latency (`total / n_tiles`) — for a perfectly parallel workload")
    lines.append("this drops linearly with `n_cores` until host overhead dominates.")
    lines.append("")
    lines.append("| Resolution | Tiles | dtype | Cores | Median total (s) | Mean (s) | ms/tile | Speedup vs fp32 1-core |")
    lines.append("|-----------:|------:|:------|------:|-----------------:|---------:|--------:|-----------------------:|")
    # Build baseline lookup.
    baseline = {}
    for r in all_results:
        if r["dtype"] == "fp32" and r["n_cores"] == 1:
            baseline[r["resolution"]] = r["total_ms_median"]
    for r in all_results:
        base = baseline.get(r["resolution"])
        speedup = (base / r["total_ms_median"]) if base else float("nan")
        lines.append(
            "| {res}x{res} | {n} | {dt} | {c} | {med:.2f} | {mean:.2f} | {mpt:.2f} | {sp:.2f}x |".format(
                res=r["resolution"],
                n=r["n_tiles"],
                dt=r["dtype"],
                c=r["n_cores"],
                med=r["total_ms_median"] / 1000.0,
                mean=r["total_ms_mean"] / 1000.0,
                mpt=r["ms_per_tile_mean"],
                sp=speedup,
            )
        )
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- **Data parallelism is free for this workload** — tiles are independent,")
    lines.append("  so `DataParallel` replicates the NEFF across the requested NeuronCores")
    lines.append("  and round-robins tile batches onto them. Each replica holds its own")
    lines.append("  ~64MB of weights, so VRAM scales with `n_cores`.")
    lines.append("- **bf16 vs fp32**: the generator has no attention / normalisation layers")
    lines.append("  that are sensitive to dtype, so bf16 gives a large raw speedup with")
    lines.append("  negligible pixel-level quality change (previous parity test showed")
    lines.append("  max |Δ| = 1/255 against CPU fp32). Produce a bf16 NEFF once and keep it.")
    lines.append("- **Host overhead ceiling**: after enough cores the per-tile latency is")
    lines.append("  dominated by Python-side slicing, `to(dtype)`, and the DataParallel")
    lines.append("  dispatch thread pool. If you see speedup plateau, the next lever is a")
    lines.append("  bigger tile size or pre-batching tiles on the host into a larger NEFF.")
    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="RealESRGAN_x4plus")
    parser.add_argument("--model-path", default="/home/ubuntu/models/real-esrgan/RealESRGAN_x4plus.pth")
    parser.add_argument("--tile", type=int, default=128)
    parser.add_argument("--resolutions", nargs="+", default=["1K", "2K", "4K"])
    parser.add_argument("--cores", nargs="+", type=int, default=[1, 2, 4, 8, 16])
    parser.add_argument("--iters", type=int, default=2)
    parser.add_argument("--compiled-root", default="/home/ubuntu/neuron_models/real-esrgan/bench/")
    parser.add_argument(
        "--output-md",
        default=str(Path(__file__).resolve().parent.parent.parent / "BENCHMARK_MULTICORE.md"),
    )
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--skip-fp32-multicore", action="store_true",
                        help="Only run fp32 at 1 core (reuse earlier baseline); full sweep in bf16")
    parser.add_argument("--skip-bf16", action="store_true")
    args = parser.parse_args()

    resolutions = [RESOLUTIONS[r] if r in RESOLUTIONS else int(r) for r in args.resolutions]

    # Compile / load both dtypes.
    compile_info = {}
    print("== Preparing fp32 NEFF ==", flush=True)
    neuron_fp32, t = get_or_compile(
        args.model_name, args.model_path, args.tile, torch.float32, args.compiled_root
    )
    compile_info["fp32"] = t

    neuron_bf16 = None
    if not args.skip_bf16:
        print("== Preparing bf16 NEFF ==", flush=True)
        neuron_bf16, t = get_or_compile(
            args.model_name, args.model_path, args.tile, torch.bfloat16, args.compiled_root
        )
        compile_info["bf16"] = t

    # Build work matrix.
    configs = []
    for res in resolutions:
        for c in args.cores:
            if not args.skip_fp32_multicore or c == 1:
                configs.append((res, torch.float32, c, neuron_fp32))
        if neuron_bf16 is not None:
            for c in args.cores:
                configs.append((res, torch.bfloat16, c, neuron_bf16))

    results = []
    for res, dtype, n_cores, neuron in configs:
        tag = "fp32" if dtype is torch.float32 else "bf16"
        print(f"\n=== {res}x{res}  dtype={tag}  cores={n_cores} ===", flush=True)
        r = bench_cfg(neuron, res, args.tile, dtype, n_cores, args.iters)
        results.append(r)
        print(
            f"  median={r['total_ms_median']/1000:.2f}s  "
            f"mean={r['total_ms_mean']/1000:.2f}s  "
            f"ms/tile={r['ms_per_tile_mean']:.2f}",
            flush=True,
        )

    # Env info.
    try:
        import torch_neuronx
        tnx = torch_neuronx.__version__
    except Exception:
        tnx = "unknown"
    env_info = {
        "torch": torch.__version__,
        "torch_neuronx": tnx,
        "instance": os.environ.get("NEURON_INSTANCE_TYPE", os.uname().nodename),
        "date": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
    }

    md = format_markdown(results, compile_info, env_info)
    Path(args.output_md).write_text(md)
    print(f"\nWrote {args.output_md}")

    if args.output_json:
        Path(args.output_json).write_text(
            json.dumps({"env": env_info, "compile": compile_info, "results": results}, indent=2)
        )
        print(f"Wrote {args.output_json}")


if __name__ == "__main__":
    main()
