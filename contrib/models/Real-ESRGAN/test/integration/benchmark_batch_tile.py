#!/usr/bin/env python3
"""Benchmark Real-ESRGAN on Neuron with different (batch_size, tile_size) combos.

Uses the yolo26 pinned multi-core pattern: 8 replicas of a
``(batch_size, 3, tile, tile)`` NEFF, each pinned to its own NeuronCore,
dispatched in parallel via ``ThreadPoolExecutor``. Input is always a 4096x4096
image tiled into ``tile x tile`` chunks.

For tile=128 the image has 1024 tiles, for tile=256 it has 256 tiles. With
N_CORES pinned replicas each running `batch_size` tiles per forward, a single
sweep requires ``ceil(n_tiles / (N_CORES * batch_size))`` dispatch waves.

Expected timeline:
 - NEFF compile per (bs, tile) combo: ~3-15 minutes (bigger combos = longer)
 - Benchmark per combo: < 30s (4K tile sweep on 8 cores)

Results land in ``BENCHMARK_BATCH_TILE.md`` / ``.json`` next to the other reports.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch

SRC_DIR = Path(__file__).resolve().parent.parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from modeling_real_esrgan import NeuronRealESRGAN  # noqa: E402


N_CORES = 8
RESOLUTION = 4096


def synth_image(resolution: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.rand((1, 3, resolution, resolution), generator=g, dtype=torch.float32)


def build_tile_batch(image: torch.Tensor, tile: int) -> torch.Tensor:
    b, c, h, w = image.shape
    assert h % tile == 0 and w % tile == 0
    tiles = []
    for y in range(0, h, tile):
        for x in range(0, w, tile):
            tiles.append(image[0, :, y : y + tile, x : x + tile])
    return torch.stack(tiles, dim=0)


def traced_path_for(bs: int, tile: int, compiled_root: str) -> str:
    return os.path.join(
        compiled_root,
        "RealESRGAN_x4plus",
        f"bs{bs}_tile{tile}x{tile}_bf16_lnc1",
        "model_neuron.pt",
    )


def compile_if_missing(
    bs: int, tile: int, model_path: str, compiled_root: str
) -> tuple[str, float]:
    out_dir = os.path.dirname(traced_path_for(bs, tile, compiled_root))
    traced = os.path.join(out_dir, "model_neuron.pt")
    if os.path.isfile(traced):
        print(f"  [load] cached NEFF bs={bs} tile={tile}x{tile}", flush=True)
        return traced, 0.0
    os.makedirs(out_dir, exist_ok=True)
    print(f"  [compile] bs={bs} tile={tile}x{tile} LNC=1 bf16 (several minutes)...", flush=True)
    neuron = NeuronRealESRGAN(
        model_name="RealESRGAN_x4plus", model_path=model_path, dtype=torch.bfloat16
    )
    t0 = time.perf_counter()
    neuron.compile(
        input_shape=(bs, 3, tile, tile),
        compiler_workdir=os.path.join(out_dir, "workdir"),
        compiler_args=["--logical-nc-config", "1"],
    )
    secs = time.perf_counter() - t0
    neuron.save(out_dir)
    print(f"  [compile] took {secs:.1f}s", flush=True)
    # Free the eager model — we only need the traced artifact on disk.
    del neuron
    return traced, secs


def load_pinned_replicas(traced_path: str, num_cores: int) -> list:
    import torch_neuronx  # noqa: F401
    from torch_neuronx.experimental.placement import neuron_cores_context

    replicas = []
    for i in range(num_cores):
        with neuron_cores_context(start_nc=i, nc_count=1):
            replicas.append(torch.jit.load(traced_path))
    return replicas


def bench_combo(
    traced_path: str,
    tile: int,
    bs: int,
    iters: int,
    warmup: int,
) -> dict:
    image = synth_image(RESOLUTION)
    tile_batch = build_tile_batch(image, tile).to(torch.bfloat16)
    n_tiles = tile_batch.shape[0]

    # bs must divide n_tiles for this benchmark (keeps the loop trivial).
    usable_tiles = (n_tiles // bs) * bs
    tile_batch = tile_batch[:usable_tiles]
    n_tiles = usable_tiles
    n_batches = n_tiles // bs
    # Reshape into (n_batches, bs, 3, tile, tile) for easy dispatch.
    batches = tile_batch.view(n_batches, bs, 3, tile, tile)

    replicas = load_pinned_replicas(traced_path, N_CORES)
    pool = ThreadPoolExecutor(max_workers=2 * N_CORES)

    def step() -> None:
        # Send n_batches forwards out across N_CORES replicas, N_CORES at a time.
        for wave_start in range(0, n_batches, N_CORES):
            futs = []
            for core_idx in range(N_CORES):
                batch_idx = wave_start + core_idx
                if batch_idx >= n_batches:
                    break
                futs.append(pool.submit(replicas[core_idx], batches[batch_idx]))
            for f in futs:
                out = f.result()
        _ = out[-1, 0, 0, 0].item()

    # Warmup.
    for _ in range(warmup):
        step()
    lat = []
    for _ in range(iters):
        t0 = time.perf_counter()
        step()
        lat.append((time.perf_counter() - t0) * 1000.0)
    pool.shutdown(wait=True)

    median = statistics.median(lat)
    return {
        "tile": tile,
        "batch_size": bs,
        "n_cores": N_CORES,
        "n_tiles": n_tiles,
        "n_batches": n_batches,
        "dispatch_waves": (n_batches + N_CORES - 1) // N_CORES,
        "iters": iters,
        "total_ms_median": median,
        "total_ms_mean": statistics.fmean(lat),
        "total_ms_min": min(lat),
        "total_ms_max": max(lat),
        "ms_per_tile": median / n_tiles,
        "tiles_per_s": n_tiles * 1000.0 / median,
        "megapixels_per_s": (n_tiles * tile * tile / 1_000_000) * 1000.0 / median,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="/home/ubuntu/models/real-esrgan/RealESRGAN_x4plus.pth")
    parser.add_argument("--compiled-root", default="/home/ubuntu/neuron_models/real-esrgan/bench/")
    parser.add_argument("--tiles", type=int, nargs="+", default=[128, 256])
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument(
        "--output-md",
        default=str(Path(__file__).resolve().parent.parent.parent / "BENCHMARK_BATCH_TILE.md"),
    )
    parser.add_argument("--output-json", default=None)
    parser.add_argument(
        "--skip-oom",
        action="store_true",
        help="If a combo fails at compile or load, continue to the next one.",
    )
    args = parser.parse_args()

    combos = []
    for tile in args.tiles:
        for bs in args.batch_sizes:
            combos.append((tile, bs))

    compile_info = {}
    results = []
    errors = []

    for tile, bs in combos:
        key = f"bs{bs}_tile{tile}x{tile}"
        print(f"\n== {key} ==", flush=True)
        try:
            traced, secs = compile_if_missing(bs, tile, args.model_path, args.compiled_root)
        except Exception as e:
            msg = f"compile-failed: {e}"
            print(f"  ERROR: {msg}", flush=True)
            errors.append({"key": key, "stage": "compile", "error": str(e)})
            if args.skip_oom:
                continue
            raise
        compile_info[key] = secs

        try:
            r = bench_combo(traced, tile, bs, args.iters, args.warmup)
        except Exception as e:
            msg = f"bench-failed: {e}"
            print(f"  ERROR: {msg}", flush=True)
            errors.append({"key": key, "stage": "bench", "error": str(e)})
            if args.skip_oom:
                continue
            raise
        results.append(r)
        print(
            f"  median={r['total_ms_median']/1000:.2f}s  "
            f"ms/tile={r['ms_per_tile']:.3f}  "
            f"throughput={r['tiles_per_s']:.1f} tiles/s  "
            f"{r['megapixels_per_s']:.2f} MPx/s",
            flush=True,
        )

    # --- Markdown report ---
    import torch_neuronx
    env = {
        "torch": torch.__version__,
        "torch_neuronx": torch_neuronx.__version__,
        "date": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
    }

    lines = []
    lines.append("# Real-ESRGAN batch_size x tile_size Benchmark (8 pinned cores, 4K input)")
    lines.append("")
    lines.append(f"- **Model:** `RealESRGAN_x4plus` (RRDBNet, scale x4)")
    lines.append(f"- **Image:** synthetic 4096x4096 (so tile=128 → 1024 tiles, tile=256 → 256 tiles)")
    lines.append(f"- **LNC:** 1, **dtype:** bf16, **cores:** {N_CORES} (pinned)")
    lines.append(f"- **Dispatch:** `torch_neuronx.experimental.placement.neuron_cores_context` + `ThreadPoolExecutor`")
    lines.append(f"- **torch_neuronx:** {env['torch_neuronx']}")
    lines.append(f"- **Date:** {env['date']}")
    lines.append("")
    lines.append("## Compile times")
    lines.append("")
    lines.append("| Combo | Compile seconds |")
    lines.append("|:------|----------------:|")
    for k, s in compile_info.items():
        lines.append(f"| {k} | {'(cached)' if s == 0 else f'{s:.1f}'} |")
    lines.append("")

    lines.append("## Throughput matrix (median ms over 4K sweep)")
    lines.append("")
    lines.append("| Tile | bs=1 | bs=2 | bs=4 | bs=8 | bs=16 | bs=32 |")
    lines.append("|:-----|-----:|-----:|-----:|-----:|------:|------:|")
    by_tile = {}
    for r in results:
        by_tile.setdefault(r["tile"], {})[r["batch_size"]] = r["total_ms_median"]
    for tile in args.tiles:
        row = f"| {tile}x{tile} |"
        for bs in args.batch_sizes:
            v = by_tile.get(tile, {}).get(bs)
            row += f" {v/1000:.2f}s |" if v is not None else " n/a |"
        lines.append(row)
    lines.append("")

    lines.append("## ms/tile matrix")
    lines.append("")
    lines.append("| Tile | bs=1 | bs=2 | bs=4 | bs=8 | bs=16 | bs=32 |")
    lines.append("|:-----|-----:|-----:|-----:|-----:|------:|------:|")
    by_tile_mpt: dict[int, dict[int, float]] = {}
    for r in results:
        by_tile_mpt.setdefault(r["tile"], {})[r["batch_size"]] = r["ms_per_tile"]
    for tile in args.tiles:
        row = f"| {tile}x{tile} |"
        for bs in args.batch_sizes:
            v = by_tile_mpt.get(tile, {}).get(bs)
            row += f" {v:.2f} |" if v is not None else " n/a |"
        lines.append(row)
    lines.append("")

    lines.append("## Megapixels / second")
    lines.append("")
    lines.append("| Tile | bs=1 | bs=2 | bs=4 | bs=8 | bs=16 | bs=32 |")
    lines.append("|:-----|-----:|-----:|-----:|-----:|------:|------:|")
    by_tile_mpx: dict[int, dict[int, float]] = {}
    for r in results:
        by_tile_mpx.setdefault(r["tile"], {})[r["batch_size"]] = r["megapixels_per_s"]
    for tile in args.tiles:
        row = f"| {tile}x{tile} |"
        for bs in args.batch_sizes:
            v = by_tile_mpx.get(tile, {}).get(bs)
            row += f" {v:.2f} |" if v is not None else " n/a |"
        lines.append(row)
    lines.append("")

    lines.append("## Full rows")
    lines.append("")
    lines.append(
        "| Tile | bs | Tiles | Dispatch waves | Median (s) | ms/tile | tiles/s | MPx/s |"
    )
    lines.append(
        "|:-----|---:|------:|---------------:|-----------:|--------:|--------:|------:|"
    )
    for r in results:
        lines.append(
            "| {t}x{t} | {b} | {n} | {w} | {m:.2f} | {mt:.3f} | {tp:.1f} | {mpx:.2f} |".format(
                t=r["tile"], b=r["batch_size"], n=r["n_tiles"], w=r["dispatch_waves"],
                m=r["total_ms_median"] / 1000, mt=r["ms_per_tile"],
                tp=r["tiles_per_s"], mpx=r["megapixels_per_s"],
            )
        )
    lines.append("")
    if errors:
        lines.append("## Skipped combos")
        lines.append("")
        for e in errors:
            lines.append(f"- `{e['key']}` ({e['stage']}): `{e['error']}`")
        lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append(f"- All 8 replicas share one traced NEFF and are loaded via")
    lines.append(f"  `neuron_cores_context(start_nc=i, nc_count=1)`; each occupies one physical")
    lines.append(f"  NeuronCore. Memory per combo scales with the batch size (activations) plus")
    lines.append(f"  ~64MB weights per replica.")
    lines.append(f"- Larger batch sizes reduce the number of dispatch waves: a 4K sweep has")
    lines.append(f"  `n_tiles / (8 * bs)` waves, so bs=32 tile=128 only needs 4 waves total.")
    lines.append(f"- Throughput tends to improve with `bs` until the activation memory saturates")
    lines.append(f"  on-chip SRAM and the compiler has to spill — beyond that, bigger batches")
    lines.append(f"  hurt ms/tile. If a combo OOMs at compile time it will appear under")
    lines.append(f"  'Skipped combos' above.")
    lines.append("")

    Path(args.output_md).write_text("\n".join(lines))
    print(f"\nWrote {args.output_md}")

    if args.output_json:
        Path(args.output_json).write_text(
            json.dumps(
                {"env": env, "compile": compile_info, "results": results, "errors": errors},
                indent=2,
            )
        )
        print(f"Wrote {args.output_json}")


if __name__ == "__main__":
    main()
