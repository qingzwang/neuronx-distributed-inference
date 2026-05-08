#!/usr/bin/env python3
"""Benchmark Real-ESRGAN on Neuron with varying tile shapes to see whether the
~24 ms/tile ceiling we hit under DataParallel is a fixed per-dispatch overhead
(→ bigger tiles amortise it) or scales with tile pixel count (→ it's device
compute and dispatch is cheap).

All compiles use LNC=1 bf16 (matching BENCHMARK_LNC1.md). For each tile shape
we sweep ``cores ∈ {1, 2, 4, 8, 16}`` running a full 1024×1024 sweep and
compute ms/tile, ms/megapixel and images/s.

Writes ``BENCHMARK_TILE_SIZES.md`` alongside the existing reports.
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

# Tile shapes to benchmark. Each must evenly divide 1024 along both dims.
TILE_SHAPES = [
    (128, 128),
    (256, 256),
    (128, 256),
    (128, 512),
]


def synth_image(resolution: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.rand((1, 3, resolution, resolution), generator=g, dtype=torch.float32)


def build_tile_batch(image: torch.Tensor, tile_h: int, tile_w: int) -> torch.Tensor:
    b, c, h, w = image.shape
    assert h % tile_h == 0 and w % tile_w == 0, (h, w, tile_h, tile_w)
    tiles = []
    for y in range(0, h, tile_h):
        for x in range(0, w, tile_w):
            tiles.append(image[0, :, y : y + tile_h, x : x + tile_w])
    return torch.stack(tiles, dim=0)


def get_or_compile_lnc1(
    model_path: str,
    tile_h: int,
    tile_w: int,
    dtype: torch.dtype,
    compiled_root: str,
) -> tuple[NeuronRealESRGAN, float]:
    tag = "bf16" if dtype is torch.bfloat16 else "fp32"
    out_dir = os.path.join(
        compiled_root,
        "RealESRGAN_x4plus",
        f"bs1_tile{tile_h}x{tile_w}_{tag}_lnc1",
    )
    traced = os.path.join(out_dir, "model_neuron.pt")
    neuron = NeuronRealESRGAN(
        model_name="RealESRGAN_x4plus", model_path=model_path, dtype=dtype
    )
    if os.path.isfile(traced):
        print(f"  [load] cached NEFF {tile_h}x{tile_w} from {out_dir}", flush=True)
        neuron.load(out_dir)
        return neuron, 0.0
    os.makedirs(out_dir, exist_ok=True)
    print(f"  [compile] LNC=1 {tag} NEFF {tile_h}x{tile_w} (several minutes)...", flush=True)
    t0 = time.perf_counter()
    neuron.compile(
        input_shape=(1, 3, tile_h, tile_w),
        compiler_workdir=os.path.join(out_dir, "workdir"),
        compiler_args=["--logical-nc-config", "1"],
    )
    secs = time.perf_counter() - t0
    neuron.save(out_dir)
    print(f"  [compile] took {secs:.1f}s", flush=True)
    return neuron, secs


def bench_single_core(
    neuron: NeuronRealESRGAN, tile_batch: torch.Tensor, dtype: torch.dtype
) -> float:
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
    parser.add_argument(
        "--model-path",
        default="/home/ubuntu/models/real-esrgan/RealESRGAN_x4plus.pth",
    )
    parser.add_argument("--resolution", type=int, default=1024,
                        help="Image resolution (must be divisible by every tile dim)")
    parser.add_argument("--cores", nargs="+", type=int, default=[1, 2, 4, 8, 16])
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument(
        "--compiled-root", default="/home/ubuntu/neuron_models/real-esrgan/bench/"
    )
    parser.add_argument(
        "--output-md",
        default=str(
            Path(__file__).resolve().parent.parent.parent / "BENCHMARK_TILE_SIZES.md"
        ),
    )
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args()

    import torch_neuronx

    compile_info = {}
    results = []
    dtype = torch.bfloat16

    image = synth_image(args.resolution)

    for tile_h, tile_w in TILE_SHAPES:
        assert args.resolution % tile_h == 0 and args.resolution % tile_w == 0, (
            f"resolution {args.resolution} not divisible by tile {tile_h}x{tile_w}"
        )
        print(f"\n== Preparing LNC=1 bf16 NEFF for tile {tile_h}x{tile_w} ==", flush=True)
        neuron, secs = get_or_compile_lnc1(
            args.model_path, tile_h, tile_w, dtype, args.compiled_root
        )
        compile_info[f"{tile_h}x{tile_w}"] = secs

        tile_batch = build_tile_batch(image, tile_h, tile_w)
        n_tiles = tile_batch.shape[0]
        tile_pixels = tile_h * tile_w
        for n_cores in args.cores:
            print(
                f"\n=== tile={tile_h}x{tile_w} tiles={n_tiles} cores={n_cores} ===",
                flush=True,
            )
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
            ms_per_tile = med / n_tiles
            # Input megapixels per second.
            total_input_px = n_tiles * tile_pixels
            mpx_per_s = (total_input_px / 1_000_000) * 1000.0 / med
            results.append({
                "tile_h": tile_h,
                "tile_w": tile_w,
                "tile_pixels": tile_pixels,
                "n_tiles": n_tiles,
                "n_cores": n_cores,
                "total_ms_median": med,
                "total_ms_mean": statistics.fmean(lat),
                "ms_per_tile": ms_per_tile,
                "ms_per_megapixel": ms_per_tile / (tile_pixels / 1_000_000),
                "megapixels_per_s": mpx_per_s,
            })
            print(
                f"  median={med/1000:.2f}s ms/tile={ms_per_tile:.2f} "
                f"MPx/s={mpx_per_s:.2f}",
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
        "resolution": args.resolution,
    }

    # --- Markdown report ---
    lines = []
    lines.append("# Real-ESRGAN Tile-Shape Benchmark (LNC=1, bf16)")
    lines.append("")
    lines.append(f"- **Model:** `RealESRGAN_x4plus` (RRDBNet, scale x4)")
    lines.append(f"- **Image:** synthetic {args.resolution}x{args.resolution} input")
    lines.append(f"- **LNC:** 1 (each physical NeuronCore is its own logical core)")
    lines.append(f"- **dtype:** bf16")
    lines.append(f"- **torch_neuronx:** {tnx}")
    lines.append(f"- **Date:** {env_info['date']}")
    lines.append("")
    lines.append("## Compile times")
    lines.append("")
    lines.append("| Tile | compile seconds |")
    lines.append("|:-----|----------------:|")
    for key, s in compile_info.items():
        lines.append(f"| {key} | {'(cached)' if s == 0 else f'{s:.1f}'} |")
    lines.append("")
    lines.append("## Summary: best config per tile shape")
    lines.append("")
    lines.append(
        "`MPx/s` is the sustained input megapixels throughput; higher is better. "
        "`ms/megapixel` normalises per-tile latency by tile area — if the "
        "DataParallel ceiling were fixed dispatch overhead, bigger tiles would "
        "have lower ms/megapixel."
    )
    lines.append("")
    lines.append("| Tile | Pixels | Tiles/image | Best cores | Best total (s) | ms/tile | ms/megapixel | MPx/s |")
    lines.append("|:-----|-------:|------------:|-----------:|---------------:|--------:|-------------:|------:|")
    # Group by tile, pick min total time.
    by_tile: dict[tuple[int, int], dict] = {}
    for r in results:
        key = (r["tile_h"], r["tile_w"])
        if key not in by_tile or r["total_ms_median"] < by_tile[key]["total_ms_median"]:
            by_tile[key] = r
    for (h, w), r in sorted(by_tile.items(), key=lambda kv: kv[0][0] * kv[0][1]):
        lines.append(
            "| {h}x{w} | {px} | {nt} | {c} | {t:.2f} | {mt:.2f} | {mm:.2f} | {mpx:.2f} |".format(
                h=h, w=w, px=r["tile_pixels"], nt=r["n_tiles"], c=r["n_cores"],
                t=r["total_ms_median"] / 1000, mt=r["ms_per_tile"],
                mm=r["ms_per_megapixel"], mpx=r["megapixels_per_s"],
            )
        )
    lines.append("")
    lines.append("## Full sweep")
    lines.append("")
    lines.append(
        "| Tile | Tiles | Cores | Total median (s) | ms/tile | ms/megapixel | MPx/s |"
    )
    lines.append("|:-----|------:|------:|-----------------:|--------:|-------------:|------:|")
    for r in results:
        lines.append(
            "| {h}x{w} | {nt} | {c} | {t:.2f} | {mt:.2f} | {mm:.2f} | {mpx:.2f} |".format(
                h=r["tile_h"], w=r["tile_w"], nt=r["n_tiles"], c=r["n_cores"],
                t=r["total_ms_median"] / 1000, mt=r["ms_per_tile"],
                mm=r["ms_per_megapixel"], mpx=r["megapixels_per_s"],
            )
        )
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- If `ms/megapixel` is roughly constant across tile shapes, host dispatch")
    lines.append("  overhead is negligible and the ceiling is device compute — in that case")
    lines.append("  larger tiles give the same MPx/s.")
    lines.append("- If `ms/megapixel` **drops** with bigger tiles, there's fixed per-dispatch")
    lines.append("  overhead being amortised — the bigger the tile, the fewer dispatches per image,")
    lines.append("  so `MPx/s` rises.")
    lines.append("- If `ms/megapixel` **rises** with bigger tiles (worst case), device")
    lines.append("  utilisation drops with tile size — unlikely for conv nets, but possible if")
    lines.append("  HBM bandwidth saturates or intermediate feature maps exceed some threshold.")
    lines.append("")

    Path(args.output_md).write_text("\n".join(lines))
    print(f"\nWrote {args.output_md}")

    if args.output_json:
        Path(args.output_json).write_text(
            json.dumps({"env": env_info, "compile": compile_info, "results": results}, indent=2)
        )
        print(f"Wrote {args.output_json}")


if __name__ == "__main__":
    main()
