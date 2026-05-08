#!/usr/bin/env python3
"""Compile Real-ESRGAN with LNC=1 (each physical NeuronCore is its own logical
core) and benchmark the tile-based latency, compared against the existing
LNC=2 bf16 baseline.

On trn2, the default is LNC=2 (two physical cores fused into one logical core).
LNC=1 doubles the number of logical cores but halves the per-core compute
capacity, so it only wins when the workload has poor per-core utilization OR
when you need more DataParallel replicas than LNC=2 can provide.

This script:

1. Compiles bf16 + fp32 NEFFs at `(1, 3, 128, 128)` with `--logical-nc-config 1`
   (cached under `bs1_tile128_bf16_lnc1/` and `bs1_tile128_fp32_lnc1/`).
2. Runs the same tile sweep as benchmark_multicore.py over cores ∈ {1,2,4,8,16,32}
   for 1K / 2K / 4K inputs.
3. Writes BENCHMARK_LNC1.md with LNC=1 vs LNC=2 comparison.
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

from modeling_real_esrgan import NeuronRealESRGAN  # noqa: E402

RESOLUTIONS = {"1K": 1024, "2K": 2048, "4K": 4096}


def synth_image(resolution: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.rand((1, 3, resolution, resolution), generator=g, dtype=torch.float32)


def build_tile_batch(image: torch.Tensor, tile: int) -> torch.Tensor:
    b, c, h, w = image.shape
    tiles = []
    for y in range(0, h, tile):
        for x in range(0, w, tile):
            tiles.append(image[0, :, y : y + tile, x : x + tile])
    return torch.stack(tiles, dim=0)


def get_or_compile_lnc1(
    model_path: str,
    tile: int,
    dtype: torch.dtype,
    compiled_root: str,
) -> tuple[NeuronRealESRGAN, float]:
    tag = "fp32" if dtype is torch.float32 else "bf16"
    out_dir = os.path.join(compiled_root, "RealESRGAN_x4plus", f"bs1_tile{tile}_{tag}_lnc1")
    traced = os.path.join(out_dir, "model_neuron.pt")
    neuron = NeuronRealESRGAN(model_name="RealESRGAN_x4plus", model_path=model_path, dtype=dtype)
    if os.path.isfile(traced):
        print(f"  [load] cached LNC=1 {tag} NEFF from {out_dir}", flush=True)
        neuron.load(out_dir)
        return neuron, 0.0
    os.makedirs(out_dir, exist_ok=True)
    print(f"  [compile] LNC=1 {tag} NEFF (several minutes)...", flush=True)
    t0 = time.perf_counter()
    neuron.compile(
        input_shape=(1, 3, tile, tile),
        compiler_workdir=os.path.join(out_dir, "workdir"),
        compiler_args=["--logical-nc-config", "1"],
    )
    secs = time.perf_counter() - t0
    neuron.save(out_dir)
    print(f"  [compile] took {secs:.1f}s", flush=True)
    return neuron, secs


def bench_single_core(neuron: NeuronRealESRGAN, tile_batch: torch.Tensor, dtype: torch.dtype) -> float:
    t0 = time.perf_counter()
    for i in range(tile_batch.shape[0]):
        chunk = tile_batch[i : i + 1].to(dtype)
        out = neuron(chunk)
        _ = out[-1, 0, 0, 0].item()
    return (time.perf_counter() - t0) * 1000.0


def bench_dp(dp_model, tile_batch: torch.Tensor, chunk: int, dtype: torch.dtype) -> float:
    t0 = time.perf_counter()
    for i in range(0, tile_batch.shape[0], chunk):
        batch = tile_batch[i : i + chunk].to(dtype)
        out = dp_model(batch)
        _ = out[-1, 0, 0, 0].item()
    return (time.perf_counter() - t0) * 1000.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="/home/ubuntu/models/real-esrgan/RealESRGAN_x4plus.pth")
    parser.add_argument("--tile", type=int, default=128)
    parser.add_argument("--resolutions", nargs="+", default=["1K", "2K", "4K"])
    parser.add_argument("--cores", nargs="+", type=int, default=[1, 2, 4, 8, 16, 32])
    parser.add_argument("--iters", type=int, default=2)
    parser.add_argument("--compiled-root", default="/home/ubuntu/neuron_models/real-esrgan/bench/")
    parser.add_argument(
        "--output-md",
        default=str(Path(__file__).resolve().parent.parent.parent / "BENCHMARK_LNC1.md"),
    )
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--skip-fp32", action="store_true")
    args = parser.parse_args()

    resolutions = [RESOLUTIONS[r] if r in RESOLUTIONS else int(r) for r in args.resolutions]

    import torch_neuronx

    compile_info = {}
    results = []

    dtypes = [torch.bfloat16] if args.skip_fp32 else [torch.float32, torch.bfloat16]

    for dtype in dtypes:
        tag = "fp32" if dtype is torch.float32 else "bf16"
        print(f"\n== Preparing LNC=1 {tag} NEFF ==", flush=True)
        neuron, secs = get_or_compile_lnc1(args.model_path, args.tile, dtype, args.compiled_root)
        compile_info[tag] = secs

        for res in resolutions:
            image = synth_image(res)
            tile_batch = build_tile_batch(image, args.tile)
            n_tiles = tile_batch.shape[0]
            for n_cores in args.cores:
                print(f"\n=== LNC=1 {tag} {res}x{res} tiles={n_tiles} cores={n_cores} ===", flush=True)
                try:
                    if n_cores == 1:
                        runner = lambda: bench_single_core(neuron, tile_batch, dtype)  # noqa: E731
                    else:
                        dp = torch_neuronx.DataParallel(
                            neuron.traced, device_ids=list(range(n_cores))
                        )
                        runner = lambda: bench_dp(dp, tile_batch, n_cores, dtype)  # noqa: E731
                    runner()  # warmup
                    lat = [runner() for _ in range(args.iters)]
                except Exception as e:
                    print(f"  ERROR: {e}", flush=True)
                    continue
                med = statistics.median(lat)
                results.append({
                    "lnc": 1,
                    "dtype": tag,
                    "resolution": res,
                    "n_tiles": n_tiles,
                    "n_cores": n_cores,
                    "total_ms_median": med,
                    "total_ms_mean": statistics.fmean(lat),
                    "ms_per_tile": med / n_tiles,
                })
                print(
                    f"  median={med/1000:.2f}s ms/tile={med/n_tiles:.2f}",
                    flush=True,
                )

    # Env info.
    try:
        tnx = torch_neuronx.__version__
    except Exception:
        tnx = "unknown"
    env_info = {
        "torch": torch.__version__,
        "torch_neuronx": tnx,
        "instance": os.uname().nodename,
        "date": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
    }

    # LNC=2 reference values (from BENCHMARK_MULTICORE.md) for side-by-side table.
    LNC2_REF = {
        # (dtype, res, cores) -> ms/tile
        ("fp32", 1024, 1): 199.52, ("fp32", 1024, 2): 100.44, ("fp32", 1024, 4): 100.19, ("fp32", 1024, 8): 100.09, ("fp32", 1024, 16): 100.55,
        ("bf16", 1024, 1): 48.97, ("bf16", 1024, 2): 24.69, ("bf16", 1024, 4): 24.32, ("bf16", 1024, 8): 24.23, ("bf16", 1024, 16): 24.19,
        ("fp32", 2048, 1): 199.36, ("fp32", 2048, 2): 100.37, ("fp32", 2048, 4): 100.07, ("fp32", 2048, 8): 99.95, ("fp32", 2048, 16): 100.49,
        ("bf16", 2048, 1): 48.43, ("bf16", 2048, 2): 24.62, ("bf16", 2048, 4): 24.36, ("bf16", 2048, 8): 24.22, ("bf16", 2048, 16): 24.39,
        ("fp32", 4096, 1): 199.40, ("fp32", 4096, 2): 100.32, ("fp32", 4096, 4): 100.06, ("fp32", 4096, 8): 100.32, ("fp32", 4096, 16): 100.55,
        ("bf16", 4096, 1): 48.39, ("bf16", 4096, 2): 24.63, ("bf16", 4096, 4): 24.35, ("bf16", 4096, 8): 24.25, ("bf16", 4096, 16): 24.40,
    }

    lines = []
    lines.append("# Real-ESRGAN LNC=1 vs LNC=2 Benchmark")
    lines.append("")
    lines.append(f"- **Model:** `RealESRGAN_x4plus` (RRDBNet, scale x4)")
    lines.append(f"- **Tile size:** {args.tile}x{args.tile}")
    lines.append(f"- **Instance:** trn2.48xlarge")
    lines.append(f"- **torch_neuronx:** {tnx}")
    lines.append(f"- **Date:** {env_info['date']}")
    lines.append("")
    lines.append("## Compile times (LNC=1)")
    lines.append("")
    lines.append("| dtype | compile seconds |")
    lines.append("|:------|----------------:|")
    for tag, s in compile_info.items():
        lines.append(f"| {tag} | {'(cached)' if s == 0 else f'{s:.1f}'} |")
    lines.append("")
    lines.append("## Per-tile latency: LNC=1 vs LNC=2")
    lines.append("")
    lines.append(
        "LNC=2 baseline numbers are copied from BENCHMARK_MULTICORE.md. "
        "Lower ms/tile is better; speedup = LNC2 / LNC1."
    )
    lines.append("")
    lines.append("| Resolution | dtype | Cores | LNC=1 ms/tile | LNC=2 ms/tile | Speedup (LNC=2 / LNC=1) |")
    lines.append("|-----------:|:------|------:|--------------:|--------------:|------------------------:|")
    for r in results:
        key = (r["dtype"], r["resolution"], r["n_cores"])
        lnc2 = LNC2_REF.get(key)
        sp = (lnc2 / r["ms_per_tile"]) if lnc2 else float("nan")
        lines.append(
            "| {res}x{res} | {dt} | {c} | {a:.2f} | {b} | {s} |".format(
                res=r["resolution"], dt=r["dtype"], c=r["n_cores"],
                a=r["ms_per_tile"],
                b=f"{lnc2:.2f}" if lnc2 else "n/a",
                s=f"{sp:.2f}x" if lnc2 else "n/a",
            )
        )
    lines.append("")
    lines.append("## Full LNC=1 whole-image latency")
    lines.append("")
    lines.append("| Resolution | dtype | Cores | Median (s) | Mean (s) |")
    lines.append("|-----------:|:------|------:|-----------:|---------:|")
    for r in results:
        lines.append(
            "| {res}x{res} | {dt} | {c} | {m:.2f} | {mn:.2f} |".format(
                res=r["resolution"], dt=r["dtype"], c=r["n_cores"],
                m=r["total_ms_median"] / 1000, mn=r["total_ms_mean"] / 1000,
            )
        )
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- LNC=1 gives twice as many logical NeuronCores but each has half the compute")
    lines.append("  of an LNC=2 logical core. For a pure conv network like Real-ESRGAN whose per-core")
    lines.append("  utilisation is already high, we expect LNC=1 per-tile latency to roughly double")
    lines.append("  at the same `cores` number (because one LNC=1 core is half the hardware).")
    lines.append("- LNC=1 is interesting when you want to run more DataParallel replicas to get past")
    lines.append("  the host-side dispatch ceiling — but the host bottleneck, not the core count, is")
    lines.append("  what limits multi-core scaling on this model (see the probe_workers test).")
    lines.append("")

    Path(args.output_md).write_text("\n".join(lines))
    print(f"\nWrote {args.output_md}")

    if args.output_json:
        Path(args.output_json).write_text(json.dumps({
            "env": env_info, "compile": compile_info, "results": results,
        }, indent=2))
        print(f"Wrote {args.output_json}")


if __name__ == "__main__":
    main()
