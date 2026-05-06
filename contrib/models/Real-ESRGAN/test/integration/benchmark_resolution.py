#!/usr/bin/env python3
"""Benchmark Real-ESRGAN end-to-end latency at 1K / 2K / 4K input resolutions
and compare Neuron vs. CPU eager PyTorch.

Real-ESRGAN on Neuron traces a fixed input shape, so we always run the same
bs=1 tile=TILE NEFF and tile the big image on the host side (same strategy
as the upstream ``RealESRGANer.tile_process``). The CPU baseline runs the
same tiling pipeline through ``torch.nn.Module.__call__`` in eager mode, so
the comparison is apples-to-apples (identical tile count, identical padding,
identical weights).

Outputs a markdown report (``BENCHMARK_RESOLUTION.md``) that includes
per-resolution wall-clock latency, tile count, per-tile latency, and the
Neuron / CPU speedup.
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


RESOLUTIONS = {
    "1K": 1024,
    "2K": 2048,
    "4K": 4096,
}


def synth_image(resolution: int, seed: int = 0) -> torch.Tensor:
    """Deterministic synthetic image — content does not affect conv latency."""
    g = torch.Generator().manual_seed(seed)
    return torch.rand((1, 3, resolution, resolution), generator=g, dtype=torch.float32)


def tile_forward(
    forward_fn,
    image: torch.Tensor,
    tile: int,
    scale: int,
) -> tuple[torch.Tensor, int, list[float]]:
    """Run ``forward_fn`` on ``tile x tile`` chunks of ``image`` and stitch the
    x``scale`` output back together. Returns the stitched output, the number
    of tiles processed, and the per-tile latencies in ms.

    Tiles are non-overlapping and assumed to land on tile boundaries — the
    caller guarantees this by picking input resolutions divisible by ``tile``.
    """
    b, c, h, w = image.shape
    assert b == 1 and h % tile == 0 and w % tile == 0, (b, h, w, tile)
    out_h, out_w = h * scale, w * scale
    out = torch.zeros((1, c, out_h, out_w), dtype=torch.float32)

    per_tile_ms = []
    n_tiles = 0
    for y in range(0, h, tile):
        for x in range(0, w, tile):
            chunk = image[:, :, y : y + tile, x : x + tile]
            t0 = time.perf_counter()
            out_chunk = forward_fn(chunk)
            # Force realisation so timing includes any lazy copy.
            _ = out_chunk[0, 0, 0, 0].item()
            per_tile_ms.append((time.perf_counter() - t0) * 1000.0)
            oy, ox = y * scale, x * scale
            out[:, :, oy : oy + tile * scale, ox : ox + tile * scale] = out_chunk
            n_tiles += 1
    return out, n_tiles, per_tile_ms


def warmup_neuron(neuron: NeuronRealESRGAN, tile: int, n: int = 3) -> None:
    dummy = torch.zeros((1, 3, tile, tile), dtype=torch.float32)
    for _ in range(n):
        _ = neuron(dummy)


def _run_n_tiles(
    forward_fn,
    image: torch.Tensor,
    tile: int,
    scale: int,
    max_tiles: int,
    cpu_like: bool = False,
) -> tuple[float, list[float], int]:
    """Run at most ``max_tiles`` of the input through ``forward_fn``. Returns
    ``(total_ms, per_tile_ms, tiles_run)`` — total is the wall-clock over the
    tiles actually executed (not extrapolated)."""
    b, c, h, w = image.shape
    per_tile_ms: list[float] = []
    t_total0 = time.perf_counter()
    tiles_run = 0
    for y in range(0, h, tile):
        for x in range(0, w, tile):
            if tiles_run >= max_tiles:
                break
            chunk = image[:, :, y : y + tile, x : x + tile]
            t0 = time.perf_counter()
            if cpu_like:
                with torch.no_grad():
                    out_chunk = forward_fn(chunk)
            else:
                out_chunk = forward_fn(chunk)
            _ = out_chunk[0, 0, 0, 0].item()
            per_tile_ms.append((time.perf_counter() - t0) * 1000.0)
            tiles_run += 1
        if tiles_run >= max_tiles:
            break
    total_ms = (time.perf_counter() - t_total0) * 1000.0
    return total_ms, per_tile_ms, tiles_run


def run_resolution(
    neuron: NeuronRealESRGAN,
    cpu_model,
    resolution: int,
    tile: int,
    scale: int,
    neuron_iters: int,
    cpu_sample_tiles: int,
) -> dict:
    image = synth_image(resolution)
    expected_tiles = (resolution // tile) ** 2
    print(f"  resolution={resolution}x{resolution}  tiles={expected_tiles}", flush=True)

    # --- Neuron: full sweep, multiple iters ---
    neuron_totals: list[float] = []
    neuron_per_tile_all: list[float] = []
    for i in range(neuron_iters):
        t0 = time.perf_counter()
        _, n_tiles, per_tile = tile_forward(neuron, image, tile, scale)
        total_ms = (time.perf_counter() - t0) * 1000.0
        neuron_totals.append(total_ms)
        neuron_per_tile_all.extend(per_tile)
        assert n_tiles == expected_tiles
        print(
            f"    [neuron iter {i+1}/{neuron_iters}] "
            f"total={total_ms:.1f}ms ({total_ms/expected_tiles:.2f}ms/tile)",
            flush=True,
        )

    # --- CPU: sample up to cpu_sample_tiles, then extrapolate to expected_tiles ---
    sample = min(cpu_sample_tiles, expected_tiles)
    print(f"    [cpu] sampling {sample}/{expected_tiles} tiles...", flush=True)
    cpu_sample_total_ms, cpu_per_tile, cpu_tiles_run = _run_n_tiles(
        cpu_model, image, tile, scale, max_tiles=sample, cpu_like=True
    )
    cpu_per_tile_median = statistics.median(cpu_per_tile)
    cpu_per_tile_mean = statistics.fmean(cpu_per_tile)
    # Extrapolate whole-image CPU latency as n_tiles * median per-tile.
    cpu_total_est = expected_tiles * cpu_per_tile_median
    print(
        f"    [cpu] sampled {cpu_tiles_run} tiles  "
        f"per-tile median={cpu_per_tile_median:.1f}ms  "
        f"extrapolated whole-image={cpu_total_est/1000:.1f}s",
        flush=True,
    )

    neuron_median = statistics.median(neuron_totals)
    neuron_per_tile_median = statistics.median(neuron_per_tile_all)
    speedup = cpu_total_est / neuron_median if neuron_median > 0 else float("nan")

    return {
        "resolution": resolution,
        "tile": tile,
        "scale": scale,
        "n_tiles": expected_tiles,
        "neuron": {
            "iters": neuron_iters,
            "total_ms_median": neuron_median,
            "total_ms_mean": statistics.fmean(neuron_totals),
            "total_ms_min": min(neuron_totals),
            "total_ms_max": max(neuron_totals),
            "per_tile_ms_median": neuron_per_tile_median,
            "per_tile_ms_mean": statistics.fmean(neuron_per_tile_all),
        },
        "cpu": {
            "sampled_tiles": cpu_tiles_run,
            "sample_total_ms": cpu_sample_total_ms,
            "total_ms_extrapolated": cpu_total_est,
            "per_tile_ms_median": cpu_per_tile_median,
            "per_tile_ms_mean": cpu_per_tile_mean,
            "per_tile_ms_min": min(cpu_per_tile),
            "per_tile_ms_max": max(cpu_per_tile),
        },
        "speedup_cpu_over_neuron": speedup,
    }


def format_markdown(
    results: list[dict],
    env_info: dict,
    model_name: str,
    tile: int,
) -> str:
    preset = MODEL_PRESETS[model_name]
    lines = []
    lines.append("# Real-ESRGAN Input-Resolution Latency: Neuron vs CPU")
    lines.append("")
    lines.append(f"- **Model:** `{model_name}` (RRDBNet, scale x{preset.net_scale})")
    lines.append(f"- **Tile size (fixed NEFF input):** {tile}x{tile}")
    lines.append(f"- **Input resolutions tested:** {', '.join(f'{r['resolution']}x{r['resolution']}' for r in results)}")
    lines.append(f"- **Input dtype:** fp32, synthetic random pixels")
    lines.append(f"- **torch:** {env_info.get('torch')}")
    lines.append(f"- **torch_neuronx:** {env_info.get('torch_neuronx')}")
    lines.append(f"- **CPU:** {env_info.get('cpu_model')} x {env_info.get('cpu_count')} threads used by PyTorch")
    lines.append(f"- **Instance:** {env_info.get('instance')}")
    lines.append(f"- **Date:** {env_info.get('date')}")
    lines.append("")
    lines.append("## Methodology")
    lines.append("")
    lines.append(f"Both Neuron and CPU run the **same tiling pipeline**: the input image is split into")
    lines.append(f"non-overlapping {tile}x{tile} tiles and each tile is super-resolved x{preset.net_scale} by")
    lines.append("the RRDBNet generator. The outputs are stitched back into a single image of size")
    lines.append(f"`resolution * {preset.net_scale}` on each side. The only difference between the two")
    lines.append("runs is where the tile's forward pass happens:")
    lines.append("")
    lines.append("- **Neuron**: traced NEFF via `torch_neuronx.trace` (one Trainium / Inferentia core)")
    lines.append("- **CPU**: the same `nn.Module` in eager PyTorch on the host CPU, fp32")
    lines.append("")
    lines.append("Warmup runs are executed before timing. Per-tile latencies are measured inside")
    lines.append("`time.perf_counter`, per-resolution totals wrap the whole tiled sweep. Reported")
    lines.append("numbers are the **median** across iterations to filter outliers.")
    lines.append("")
    lines.append("## End-to-end latency (whole image)")
    lines.append("")
    lines.append("Neuron is measured over a full sweep (every tile actually runs). CPU whole-image")
    lines.append("latency is **extrapolated** as `n_tiles * CPU-per-tile-median` because running 1024")
    lines.append("tiles on CPU for the 4K row would take >10 minutes per iteration.")
    lines.append("")
    lines.append("| Resolution | Tiles | Neuron median (s) | CPU extrapolated (s) | Speedup (CPU / Neuron) |")
    lines.append("|-----------:|------:|------------------:|---------------------:|-----------------------:|")
    for r in results:
        lines.append(
            "| {res}x{res} | {t} | {n:.2f} | {c:.2f} | {s:.1f}x |".format(
                res=r["resolution"],
                t=r["n_tiles"],
                n=r["neuron"]["total_ms_median"] / 1000.0,
                c=r["cpu"]["total_ms_extrapolated"] / 1000.0,
                s=r["speedup_cpu_over_neuron"],
            )
        )
    lines.append("")
    lines.append("## Per-tile latency (measured)")
    lines.append("")
    lines.append("| Resolution | Neuron per-tile median (ms) | CPU per-tile median (ms) | Per-tile speedup |")
    lines.append("|-----------:|----------------------------:|-------------------------:|-----------------:|")
    for r in results:
        n_pt = r["neuron"]["per_tile_ms_median"]
        c_pt = r["cpu"]["per_tile_ms_median"]
        spd = c_pt / n_pt if n_pt > 0 else float("nan")
        lines.append(
            "| {res}x{res} | {n:.2f} | {c:.2f} | {s:.1f}x |".format(
                res=r["resolution"], n=n_pt, c=c_pt, s=spd
            )
        )
    lines.append("")
    lines.append("## Neuron whole-image latency distribution (ms)")
    lines.append("")
    lines.append("| Resolution | Iters | Mean | Median | Min | Max |")
    lines.append("|-----------:|------:|-----:|-------:|----:|----:|")
    for r in results:
        s = r["neuron"]
        lines.append(
            "| {res}x{res} | {it} | {m:.1f} | {med:.1f} | {mn:.1f} | {mx:.1f} |".format(
                res=r["resolution"],
                it=s["iters"],
                m=s["total_ms_mean"],
                med=s["total_ms_median"],
                mn=s["total_ms_min"],
                mx=s["total_ms_max"],
            )
        )
    lines.append("")
    lines.append("## CPU per-tile sample")
    lines.append("")
    lines.append("| Resolution | Sampled tiles | Sample wall time (s) | Per-tile mean (ms) | Per-tile median (ms) | Per-tile min (ms) | Per-tile max (ms) |")
    lines.append("|-----------:|--------------:|---------------------:|-------------------:|---------------------:|------------------:|------------------:|")
    for r in results:
        s = r["cpu"]
        lines.append(
            "| {res}x{res} | {st} | {w:.1f} | {m:.1f} | {med:.1f} | {mn:.1f} | {mx:.1f} |".format(
                res=r["resolution"],
                st=s["sampled_tiles"],
                w=s["sample_total_ms"] / 1000.0,
                m=s["per_tile_ms_mean"],
                med=s["per_tile_ms_median"],
                mn=s["per_tile_ms_min"],
                mx=s["per_tile_ms_max"],
            )
        )
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- Tiles are non-overlapping — production pipelines should use the overlapping")
    lines.append("  tile loop from the upstream `RealESRGANer.tile_process` to avoid seam artifacts.")
    lines.append("  Latency per tile is unchanged; only the tile count increases by the overlap factor.")
    lines.append("- Neuron latency per tile should match the bs=1 number from `BENCHMARK.md`; any")
    lines.append("  per-tile overhead above that comes from host-side slicing and the `out[...]` copy.")
    lines.append("- CPU is single-process eager PyTorch (fp32) with the default thread pool. Whole-image")
    lines.append("  CPU latency for 1K/2K/4K inputs is extrapolated from a per-tile sample; the first")
    lines.append("  few tiles include one-time torch overheads and are kept in the sample on purpose.")
    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="RealESRGAN_x4plus")
    parser.add_argument("--model-path", default="/home/ubuntu/models/real-esrgan/RealESRGAN_x4plus.pth")
    parser.add_argument("--tile", type=int, default=128)
    parser.add_argument(
        "--resolutions",
        nargs="+",
        default=["1K", "2K", "4K"],
        help="Resolutions to test: 1K/2K/4K or explicit ints like 1024",
    )
    parser.add_argument(
        "--neuron-iters",
        type=int,
        default=2,
        help="Iterations per resolution on Neuron (4K has 1024 tiles, so keep this small)",
    )
    parser.add_argument(
        "--cpu-sample-tiles",
        type=int,
        default=8,
        help=(
            "How many tiles to actually run through the CPU eager model per resolution. "
            "Whole-image CPU latency is extrapolated from the median of this sample."
        ),
    )
    parser.add_argument("--compiled-root", default="/home/ubuntu/neuron_models/real-esrgan/bench/")
    parser.add_argument(
        "--output-md",
        default=str(Path(__file__).resolve().parent.parent.parent / "BENCHMARK_RESOLUTION.md"),
    )
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args()

    # Parse resolutions.
    resolutions = []
    for r in args.resolutions:
        if r in RESOLUTIONS:
            resolutions.append(RESOLUTIONS[r])
        else:
            resolutions.append(int(r))
    for res in resolutions:
        if res % args.tile != 0:
            raise ValueError(f"Resolution {res} must be divisible by tile {args.tile}")

    print(f"Benchmarking resolutions {resolutions} with tile={args.tile}", flush=True)

    # --- Compile / load a single bs=1 tile NEFF; reuse for every resolution ---
    compiled_dir = os.path.join(
        args.compiled_root, args.model_name, f"bs1_tile{args.tile}"
    )
    traced_path = os.path.join(compiled_dir, "model_neuron.pt")
    neuron = NeuronRealESRGAN(
        model_name=args.model_name, model_path=args.model_path, dtype=torch.float32
    )
    if os.path.isfile(traced_path):
        print(f"Loading cached NEFF from {compiled_dir}", flush=True)
        neuron.load(compiled_dir)
    else:
        print(f"Compiling bs=1 tile={args.tile} NEFF (this takes ~7min)...", flush=True)
        os.makedirs(compiled_dir, exist_ok=True)
        t0 = time.perf_counter()
        neuron.compile(
            input_shape=(1, 3, args.tile, args.tile),
            compiler_workdir=os.path.join(compiled_dir, "workdir"),
        )
        print(f"Compile took {time.perf_counter()-t0:.1f}s", flush=True)
        neuron.save(compiled_dir)

    # CPU baseline uses the same eager module.
    cpu_model = neuron.model
    cpu_model.eval()

    print("Warming up Neuron...", flush=True)
    warmup_neuron(neuron, args.tile, n=3)

    print("Warming up CPU...", flush=True)
    dummy = torch.zeros((1, 3, args.tile, args.tile), dtype=torch.float32)
    with torch.no_grad():
        for _ in range(2):
            _ = cpu_model(dummy)

    preset = MODEL_PRESETS[args.model_name]
    results = []
    for res in resolutions:
        print(f"\n=== Resolution {res}x{res} ===", flush=True)
        r = run_resolution(
            neuron=neuron,
            cpu_model=cpu_model,
            resolution=res,
            tile=args.tile,
            scale=preset.net_scale,
            neuron_iters=args.neuron_iters,
            cpu_sample_tiles=args.cpu_sample_tiles,
        )
        results.append(r)
        print(
            f"  median neuron={r['neuron']['total_ms_median']:.1f}ms  "
            f"cpu(extrap)={r['cpu']['total_ms_extrapolated']:.1f}ms  "
            f"speedup={r['speedup_cpu_over_neuron']:.1f}x",
            flush=True,
        )

    # Collect env info.
    try:
        import torch_neuronx
        tnx = torch_neuronx.__version__
    except Exception:
        tnx = "unknown"

    cpu_model_name = "unknown"
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    cpu_model_name = line.split(":", 1)[1].strip()
                    break
    except Exception:
        pass

    env_info = {
        "torch": torch.__version__,
        "torch_neuronx": tnx,
        "cpu_model": cpu_model_name,
        "cpu_count": torch.get_num_threads(),
        "instance": os.environ.get("NEURON_INSTANCE_TYPE", os.uname().nodename),
        "date": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
    }

    md = format_markdown(results, env_info, args.model_name, args.tile)
    Path(args.output_md).write_text(md)
    print(f"\nWrote {args.output_md}")

    if args.output_json:
        Path(args.output_json).write_text(
            json.dumps({"env": env_info, "results": results}, indent=2)
        )
        print(f"Wrote {args.output_json}")


if __name__ == "__main__":
    main()
