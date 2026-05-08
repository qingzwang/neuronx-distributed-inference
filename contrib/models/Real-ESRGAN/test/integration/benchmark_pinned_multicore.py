#!/usr/bin/env python3
"""Multi-core benchmark for Real-ESRGAN using the yolo26 optimisation pattern.

The ``DataParallel`` path we previously benchmarked plateaued at ~2x no matter
how many cores we added. The yolo26 contrib (see
``contrib/models/yolo26/src/benchmark_multicore.py``) found that the real
DataParallel API on trn2 tends to funnel every call through the replica loaded
first, so adding cores doesn't help. The fix is to:

1. Load each replica under its own ``neuron_cores_context(start_nc=i, nc_count=1)``
   so the NEFF is physically pinned to core *i*.
2. Dispatch with a plain Python ``ThreadPoolExecutor`` so the replicas run in
   parallel — one Python thread per NeuronCore.

Yolo26 went from 1x → 16.35x on 32 cores this way. This script tries the same
pattern on the Real-ESRGAN RRDBNet tile NEFF (LNC=1 bf16, 128x128), then writes
``BENCHMARK_PINNED_MULTICORE.md`` comparing it side-by-side with the existing
``DataParallel`` numbers.
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


# DataParallel reference numbers (from BENCHMARK_LNC1.md) — bf16, 1K (64 tiles).
DP_REFERENCE_MS_PER_TILE = {
    1: 48.82, 2: 25.05, 4: 24.99, 8: 24.92, 16: 24.94,
}


def build_tile_batch(resolution: int, tile: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    image = torch.rand((1, 3, resolution, resolution), generator=g, dtype=torch.float32)
    assert resolution % tile == 0
    tiles = []
    for y in range(0, resolution, tile):
        for x in range(0, resolution, tile):
            tiles.append(image[0, :, y : y + tile, x : x + tile])
    return torch.stack(tiles, dim=0)


def load_pinned_replicas(traced_path: str, num_cores: int) -> list:
    """Load the traced NEFF once per NeuronCore, pinned to that core.

    Uses ``torch_neuronx.experimental.placement.neuron_cores_context`` — this is
    the key API yolo26 discovered. Without it, ``torch.jit.load`` will place
    the module on whatever default core the runtime picked, and subsequent
    forward() calls all execute on that one core.
    """
    import torch_neuronx  # noqa: F401  (must be imported before torch.jit.load)
    from torch_neuronx.experimental.placement import neuron_cores_context

    replicas = []
    for i in range(num_cores):
        with neuron_cores_context(start_nc=i, nc_count=1):
            replicas.append(torch.jit.load(traced_path))
    return replicas


def bench_pinned(
    replicas: list,
    tile_batch: torch.Tensor,
    dtype: torch.dtype,
    iters: int,
    warmup: int,
) -> dict:
    """Dispatch tiles across pinned replicas via ThreadPoolExecutor.

    Each "step" submits num_cores tiles to num_cores threads — one tile per
    core. We repeat steps until every tile has been processed, then return
    the median wall-clock of the whole sweep.
    """
    n_cores = len(replicas)
    n_tiles = tile_batch.shape[0]
    pool = ThreadPoolExecutor(max_workers=max(2 * n_cores, 4))

    tiles_typed = tile_batch.to(dtype)

    def step() -> None:
        # Dispatch one batch of n_cores tiles — one tile per replica per step.
        # Loop until all tiles processed.
        for base in range(0, n_tiles, n_cores):
            futures = []
            for core_idx in range(n_cores):
                tile_idx = base + core_idx
                if tile_idx >= n_tiles:
                    break
                chunk = tiles_typed[tile_idx : tile_idx + 1]
                futures.append(pool.submit(replicas[core_idx], chunk))
            # Wait for this wave before submitting the next one.
            for f in futures:
                out = f.result()
            _ = out[-1, 0, 0, 0].item()

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
        "n_cores": n_cores,
        "n_tiles": n_tiles,
        "iters": iters,
        "total_ms_median": median,
        "total_ms_mean": statistics.fmean(lat),
        "total_ms_min": min(lat),
        "total_ms_max": max(lat),
        "ms_per_tile": median / n_tiles,
        "throughput_images_per_s": n_tiles * 1000.0 / median,
    }


def bench_single_core(
    traced_path: str, tile_batch: torch.Tensor, dtype: torch.dtype, iters: int, warmup: int
) -> dict:
    """Baseline: one replica, no threading."""
    import torch_neuronx  # noqa: F401

    m = torch.jit.load(traced_path)
    tiles_typed = tile_batch.to(dtype)
    n_tiles = tile_batch.shape[0]

    def step() -> None:
        for i in range(n_tiles):
            out = m(tiles_typed[i : i + 1])
        _ = out[-1, 0, 0, 0].item()

    for _ in range(warmup):
        step()
    lat = []
    for _ in range(iters):
        t0 = time.perf_counter()
        step()
        lat.append((time.perf_counter() - t0) * 1000.0)
    median = statistics.median(lat)
    return {
        "n_cores": 1,
        "n_tiles": n_tiles,
        "iters": iters,
        "total_ms_median": median,
        "total_ms_mean": statistics.fmean(lat),
        "total_ms_min": min(lat),
        "total_ms_max": max(lat),
        "ms_per_tile": median / n_tiles,
        "throughput_images_per_s": n_tiles * 1000.0 / median,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--traced-path",
        default="/home/ubuntu/neuron_models/real-esrgan/bench/RealESRGAN_x4plus/bs1_tile128_bf16_lnc1/model_neuron.pt",
        help="Existing LNC=1 bf16 tile128 NEFF to replicate across cores",
    )
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--tile", type=int, default=128)
    parser.add_argument("--cores", nargs="+", type=int, default=[1, 2, 4, 8, 16, 32])
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument(
        "--output-md",
        default=str(
            Path(__file__).resolve().parent.parent.parent / "BENCHMARK_PINNED_MULTICORE.md"
        ),
    )
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args()

    if not os.path.isfile(args.traced_path):
        raise FileNotFoundError(
            f"Traced model not found at {args.traced_path}. "
            "Run benchmark_lnc1.py first to produce the LNC=1 bf16 NEFF."
        )

    dtype = torch.bfloat16
    tile_batch = build_tile_batch(args.resolution, args.tile)
    print(
        f"Input: {args.resolution}x{args.resolution}, "
        f"tile {args.tile}x{args.tile}, "
        f"n_tiles={tile_batch.shape[0]}",
        flush=True,
    )

    results = []

    for n_cores in args.cores:
        print(f"\n=== cores={n_cores} (pinned + ThreadPool) ===", flush=True)
        try:
            if n_cores == 1:
                r = bench_single_core(args.traced_path, tile_batch, dtype, args.iters, args.warmup)
            else:
                replicas = load_pinned_replicas(args.traced_path, n_cores)
                r = bench_pinned(replicas, tile_batch, dtype, args.iters, args.warmup)
        except Exception as e:
            print(f"  ERROR: {e}", flush=True)
            continue
        results.append(r)
        print(
            f"  median={r['total_ms_median']/1000:.2f}s  "
            f"ms/tile={r['ms_per_tile']:.2f}  "
            f"throughput={r['throughput_images_per_s']:.1f} tiles/s",
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
    lines.append("# Real-ESRGAN Pinned Multi-core Benchmark (yolo26 pattern)")
    lines.append("")
    lines.append(f"- **Model:** `RealESRGAN_x4plus` (RRDBNet, scale x4)")
    lines.append(f"- **Tile:** {args.tile}x{args.tile}, **LNC:** 1, **dtype:** bf16")
    lines.append(f"- **Image:** {args.resolution}x{args.resolution} ({tile_batch.shape[0]} tiles)")
    lines.append(f"- **torch_neuronx:** {env['torch_neuronx']}")
    lines.append(f"- **Date:** {env['date']}")
    lines.append("")
    lines.append("## Method")
    lines.append("")
    lines.append("Borrowed from `contrib/models/yolo26/src/benchmark_multicore.py`:")
    lines.append("")
    lines.append("1. Load the traced NEFF **once per NeuronCore**, each load wrapped in")
    lines.append("   `torch_neuronx.experimental.placement.neuron_cores_context(start_nc=i, nc_count=1)`.")
    lines.append("   This pins replica *i* to physical core *i* — otherwise the runtime routes")
    lines.append("   every inference through whichever core loaded first.")
    lines.append("2. Fan out tile dispatch via `concurrent.futures.ThreadPoolExecutor`, so each")
    lines.append("   core gets its own Python thread and the Neuron runtime can overlap them.")
    lines.append("")
    lines.append("## Results: pinned vs. DataParallel")
    lines.append("")
    lines.append("`DataParallel ms/tile` values are copied from `BENCHMARK_LNC1.md`. Lower is better.")
    lines.append("")
    lines.append("| Cores | Pinned ms/tile | DataParallel ms/tile | Pinned speedup vs DP | Pinned vs 1-core pinned |")
    lines.append("|------:|---------------:|---------------------:|---------------------:|------------------------:|")
    base_ms = results[0]["ms_per_tile"] if results else 1.0
    for r in results:
        dp = DP_REFERENCE_MS_PER_TILE.get(r["n_cores"])
        dp_str = f"{dp:.2f}" if dp is not None else "n/a"
        dp_sp = f"{dp / r['ms_per_tile']:.2f}x" if dp else "n/a"
        lines.append(
            "| {c} | {p:.2f} | {dp_s} | {dps} | {scale:.2f}x |".format(
                c=r["n_cores"], p=r["ms_per_tile"], dp_s=dp_str, dps=dp_sp,
                scale=base_ms / r["ms_per_tile"],
            )
        )
    lines.append("")
    lines.append("## Full pinned numbers")
    lines.append("")
    lines.append(
        "| Cores | n_tiles | total median (s) | total mean (s) | ms/tile | throughput (tiles/s) |"
    )
    lines.append(
        "|------:|--------:|-----------------:|---------------:|--------:|---------------------:|"
    )
    for r in results:
        lines.append(
            "| {c} | {nt} | {m:.2f} | {mn:.2f} | {mt:.2f} | {tp:.1f} |".format(
                c=r["n_cores"], nt=r["n_tiles"],
                m=r["total_ms_median"] / 1000.0, mn=r["total_ms_mean"] / 1000.0,
                mt=r["ms_per_tile"], tp=r["throughput_images_per_s"],
            )
        )
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- Each replica holds its own ~64 MB of weights — memory scales with `cores`.")
    lines.append("- The traced NEFF was compiled with `(1, 3, 128, 128)` input at LNC=1 bf16.")
    lines.append("- No recompile is needed to change the core count; only the number of")
    lines.append("  replicas loaded into memory.")
    lines.append("- If a given core count fails with a placement or OOM error, lower `cores`")
    lines.append("  or check that other Neuron processes are not using the same cores.")
    lines.append("")

    Path(args.output_md).write_text("\n".join(lines))
    print(f"\nWrote {args.output_md}")

    if args.output_json:
        Path(args.output_json).write_text(json.dumps({"env": env, "results": results}, indent=2))
        print(f"Wrote {args.output_json}")


if __name__ == "__main__":
    main()
