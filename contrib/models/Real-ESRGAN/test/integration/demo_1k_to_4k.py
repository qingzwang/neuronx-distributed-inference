#!/usr/bin/env python3
"""End-to-end 1K -> 4K demo using the pinned multi-core pattern.

Takes a real image, resizes/crops to 1024x1024, tiles it into 128x128 blocks,
and runs the bf16 LNC=1 RRDBNet NEFF on 8 NeuronCores pinned with
``torch_neuronx.experimental.placement.neuron_cores_context``, dispatching
through a ``ThreadPoolExecutor``. Stitches the outputs into a single 4096x4096
image and writes it next to the input.

Reports:

* raw Neuron forward time (excludes pre/post-processing)
* total wall-clock including host-side I/O and stitching

Used by `README.md` to show before/after alongside the measured latency.
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

SRC_DIR = Path(__file__).resolve().parent.parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from modeling_real_esrgan import preprocess_image, postprocess_image  # noqa: E402


def read_image_bgr(path: str) -> np.ndarray:
    """HWC uint8 BGR, matching cv2.imread output. Falls back to Pillow."""
    try:
        import cv2  # type: ignore
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(path)
        return img
    except ImportError:
        from PIL import Image  # type: ignore
        rgb = np.array(Image.open(path).convert("RGB"))
        return rgb[:, :, ::-1].copy()


def write_image_bgr(path: str, img_bgr: np.ndarray) -> None:
    try:
        import cv2  # type: ignore
        cv2.imwrite(path, img_bgr)
    except ImportError:
        from PIL import Image  # type: ignore
        rgb = img_bgr[:, :, ::-1]
        Image.fromarray(rgb).save(path)


def resize_to_square(img_bgr: np.ndarray, side: int) -> np.ndarray:
    """Center-crop to square, then resize to side x side (Lanczos if cv2 avail)."""
    h, w = img_bgr.shape[:2]
    m = min(h, w)
    top = (h - m) // 2
    left = (w - m) // 2
    square = img_bgr[top : top + m, left : left + m]
    try:
        import cv2  # type: ignore
        return cv2.resize(square, (side, side), interpolation=cv2.INTER_LANCZOS4)
    except ImportError:
        from PIL import Image  # type: ignore
        rgb = square[:, :, ::-1]
        pil = Image.fromarray(rgb).resize((side, side), Image.LANCZOS)
        return np.array(pil)[:, :, ::-1].copy()


def tile_image(tensor: torch.Tensor, tile: int) -> tuple[torch.Tensor, list[tuple[int, int]]]:
    """Split a (1,3,H,W) tensor into (N, 3, tile, tile) + list of (y,x) offsets."""
    _, _, h, w = tensor.shape
    assert h % tile == 0 and w % tile == 0
    chunks = []
    positions = []
    for y in range(0, h, tile):
        for x in range(0, w, tile):
            chunks.append(tensor[0, :, y : y + tile, x : x + tile])
            positions.append((y, x))
    return torch.stack(chunks, dim=0), positions


def stitch_output(
    out_chunks: torch.Tensor, positions: list[tuple[int, int]], scale: int, tile: int
) -> torch.Tensor:
    n = out_chunks.shape[0]
    max_y = max(y for y, _ in positions) + tile
    max_x = max(x for _, x in positions) + tile
    out = torch.zeros((1, 3, max_y * scale, max_x * scale), dtype=torch.float32)
    for i in range(n):
        y, x = positions[i]
        out[:, :, y * scale : y * scale + tile * scale, x * scale : x * scale + tile * scale] = (
            out_chunks[i : i + 1]
        )
    return out


def run(
    input_path: str,
    output_dir: str,
    traced_path: str,
    tile: int,
    num_cores: int,
    iters: int,
    warmup: int,
) -> dict:
    import torch_neuronx  # noqa: F401
    from torch_neuronx.experimental.placement import neuron_cores_context

    os.makedirs(output_dir, exist_ok=True)

    # 1) Load + resize input to 1024x1024 BGR.
    t0 = time.perf_counter()
    img_orig = read_image_bgr(input_path)
    input_1k = resize_to_square(img_orig, 1024)
    stem = Path(input_path).stem
    input_path_saved = os.path.join(output_dir, f"{stem}_input_1024.png")
    write_image_bgr(input_path_saved, input_1k)
    t_input = (time.perf_counter() - t0) * 1000.0

    # 2) Preprocess to CHW float32 -> bf16 tile batch.
    t0 = time.perf_counter()
    pixel = preprocess_image(input_1k, dtype=torch.float32)  # (1,3,1024,1024) fp32
    tiles_fp32, positions = tile_image(pixel, tile)
    tiles_bf16 = tiles_fp32.to(torch.bfloat16)
    n_tiles = tiles_bf16.shape[0]
    t_prep = (time.perf_counter() - t0) * 1000.0

    # 3) Load N pinned replicas.
    t0 = time.perf_counter()
    replicas = []
    for i in range(num_cores):
        with neuron_cores_context(start_nc=i, nc_count=1):
            replicas.append(torch.jit.load(traced_path))
    t_load = (time.perf_counter() - t0) * 1000.0

    # 4) Dispatch all tiles through the pool, repeated `iters` times.
    pool = ThreadPoolExecutor(max_workers=2 * num_cores)

    def forward_all() -> torch.Tensor:
        outs = [None] * n_tiles
        for wave_start in range(0, n_tiles, num_cores):
            futs = []
            for core_idx in range(num_cores):
                idx = wave_start + core_idx
                if idx >= n_tiles:
                    break
                chunk = tiles_bf16[idx : idx + 1]
                futs.append((idx, pool.submit(replicas[core_idx], chunk)))
            for idx, f in futs:
                outs[idx] = f.result()
        return torch.cat(outs, dim=0).to(torch.float32)  # (N, 3, 4*tile, 4*tile)

    for _ in range(warmup):
        _ = forward_all()

    fwd_ms = []
    for _ in range(iters):
        t0 = time.perf_counter()
        out_tiles = forward_all()
        fwd_ms.append((time.perf_counter() - t0) * 1000.0)

    pool.shutdown(wait=True)
    fwd_median = statistics.median(fwd_ms)

    # 5) Stitch + postprocess (using the last iter's output).
    t0 = time.perf_counter()
    stitched = stitch_output(out_tiles, positions, scale=4, tile=tile)
    output_4k = postprocess_image(stitched)
    output_path = os.path.join(output_dir, f"{stem}_neuron_4x_4096.png")
    write_image_bgr(output_path, output_4k)
    t_post = (time.perf_counter() - t0) * 1000.0

    total_wall = t_input + t_prep + t_load + fwd_median + t_post

    return {
        "input_path": input_path_saved,
        "output_path": output_path,
        "tile": tile,
        "num_cores": num_cores,
        "n_tiles": n_tiles,
        "input_size": 1024,
        "output_size": 4096,
        "iters": iters,
        "timings_ms": {
            "read_and_resize": t_input,
            "preprocess_and_tile": t_prep,
            "load_replicas": t_load,
            "neuron_forward_median": fwd_median,
            "neuron_forward_all": fwd_ms,
            "stitch_and_save": t_post,
            "total_wall": total_wall,
        },
        "ms_per_tile": fwd_median / n_tiles,
        "tiles_per_s": n_tiles * 1000.0 / fwd_median,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to input image")
    parser.add_argument(
        "--traced-path",
        default="/home/ubuntu/neuron_models/real-esrgan/bench/RealESRGAN_x4plus/bs1_tile128_bf16_lnc1/model_neuron.pt",
    )
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parent.parent.parent / "results"),
    )
    parser.add_argument("--tile", type=int, default=128)
    parser.add_argument("--num-cores", type=int, default=8)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    args = parser.parse_args()

    result = run(
        args.input, args.output_dir, args.traced_path, args.tile,
        args.num_cores, args.iters, args.warmup,
    )

    print("\n=== Real-ESRGAN 1K -> 4K demo ===")
    print(f"Input (1024x1024) : {result['input_path']}")
    print(f"Output (4096x4096): {result['output_path']}")
    print(f"Cores             : {result['num_cores']} pinned")
    print(f"Tile              : {result['tile']}x{result['tile']}  "
          f"({result['n_tiles']} tiles)")
    print(f"Iterations        : {result['iters']} (+{args.warmup} warmup)")
    print()
    t = result["timings_ms"]
    print(f"read + resize input : {t['read_and_resize']:.1f} ms")
    print(f"preprocess + tile   : {t['preprocess_and_tile']:.1f} ms")
    print(f"load 8 replicas     : {t['load_replicas']:.1f} ms")
    print(f"Neuron forward      : {t['neuron_forward_median']:.1f} ms (median)")
    print(f"                       raw iters: {[f'{x:.0f}' for x in t['neuron_forward_all']]} ms")
    print(f"stitch + save PNG   : {t['stitch_and_save']:.1f} ms")
    print(f"ms/tile             : {result['ms_per_tile']:.2f}")
    print(f"throughput          : {result['tiles_per_s']:.1f} tiles/s")
    print(f"TOTAL wall-clock    : {t['total_wall']/1000:.2f} s")


if __name__ == "__main__":
    main()
