"""Benchmark aligned with the AWS Neuron reference table for YOLO26.

The reference peak-throughput table uses per-variant dtype and a large
per-core batch size (so DMA + Python dispatch overhead is amortised):

  variant   dtype   BS/core
  yolo26n   fp32    1
  yolo26s   fp32    32
  yolo26m   bf16    32
  yolo26l   bf16    32
  yolo26x   bf16    16

All variants: imgsz=640, DP=8 NeuronCores, LNC=1.
This script compiles each variant with its prescribed batch/dtype and
measures throughput exactly as the reference does.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch_neuronx
from ultralytics import YOLO

from compile_neuron import YOLO26Wrapper, _patch_detect_head
from neuron_patches import patch_attention_modules
from yolo26_common import (
    ASSETS_DIR,
    BENCHMARK_DIR,
    COMPILED_DIR,
    DEFAULT_IMGSZ,
    PROJECT_DIR,
    preprocess_image,
)


@dataclass
class VariantSpec:
    size: str
    dtype: str
    batch_size: int


REFERENCE_SPECS: List[VariantSpec] = [
    # Exactly matches the AWS Neuron reference table: per-variant dtype and
    # BS/core, imgsz=640, LNC=1, DP=8.
    VariantSpec("n", "fp32", 1),
    VariantSpec("s", "fp32", 32),
    VariantSpec("m", "bf16", 32),
    VariantSpec("l", "bf16", 32),
    VariantSpec("x", "bf16", 16),
]


def _weights(size: str) -> Path:
    return PROJECT_DIR / f"yolo26{size}.pt"


def _compiled_name(spec: VariantSpec, imgsz: int) -> Path:
    return COMPILED_DIR / f"yolo26{spec.size}_neuron_{spec.dtype}_bs{spec.batch_size}_sz{imgsz}.pt"


def _pre_fuse_param_count(size: str) -> int:
    """Parameter count before ultralytics' fuse() runs.

    This matches the number reported in the upstream yolo26.yaml config.
    """
    yolo = YOLO(str(_weights(size)))
    return sum(p.numel() for p in yolo.model.parameters())


class _BF16WeightsWrapper(torch.nn.Module):
    """Cast weights to bf16 and I/O to/from fp32 at the boundary.

    `--auto-cast=matmult --auto-cast-type=bf16` spills activations for the
    larger yolo26 variants (m/l/x) at 640x640 on a single NeuronCore.
    Casting the weights to bf16 directly halves the weight footprint before
    compile sees them, which avoids the spill and still runs with bf16
    matmuls. Input is fp32 (what the preprocessing produces); output is
    cast back to fp32 so downstream post-processing is unchanged.
    """

    def __init__(self, inner: torch.nn.Module) -> None:
        super().__init__()
        self.inner = inner

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.inner(x.to(torch.bfloat16))
        if isinstance(out, (list, tuple)):
            out = out[0]
        return out.to(torch.float32)


def _compile(spec: VariantSpec, imgsz: int, force: bool) -> dict:
    out = _compiled_name(spec, imgsz)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and not force:
        return {"path": str(out), "compile_seconds": None, "cached": True,
                "neff_size_mb": out.stat().st_size / 1024 / 1024}

    print(f"[compile] yolo26{spec.size} dtype={spec.dtype} bs={spec.batch_size} imgsz={imgsz} -> {out.name}")
    yolo = YOLO(str(_weights(spec.size)))
    model = yolo.model.eval().to(torch.float32).cpu().fuse()
    _patch_detect_head(model)
    patch_attention_modules(model)

    if spec.dtype == "bf16":
        model_bf = model.to(torch.bfloat16)
        wrapper = _BF16WeightsWrapper(model_bf).eval()
        compiler_args = ["--target=trn2", "--logical-nc-config=1"]
    elif spec.dtype == "fp16":
        wrapper = YOLO26Wrapper(model).eval()
        compiler_args = [
            "--target=trn2", "--logical-nc-config=1", "--enable-saturate-infinity",
            "--auto-cast=matmult", "--auto-cast-type=fp16",
        ]
    elif spec.dtype == "fp32":
        wrapper = YOLO26Wrapper(model).eval()
        compiler_args = [
            "--target=trn2", "--logical-nc-config=1", "--enable-saturate-infinity",
            "--auto-cast=none", "--optlevel=2",
        ]
    else:
        raise ValueError(spec.dtype)

    example = torch.zeros(spec.batch_size, 3, imgsz, imgsz, dtype=torch.float32)
    t0 = time.perf_counter()
    m = torch_neuronx.trace(wrapper, example, compiler_args=compiler_args)
    dt = time.perf_counter() - t0
    torch.jit.save(m, str(out))
    return {"path": str(out), "compile_seconds": dt, "cached": False,
            "neff_size_mb": out.stat().st_size / 1024 / 1024}


def _throughput(
    compiled_path: Path, spec: VariantSpec, imgsz: int, dp: int, iters: int, warmup: int,
) -> dict:
    """Run DP with `dp` cores, each processing `spec.batch_size` images per step.

    Uses explicit per-core placement (one NEFF replica per NeuronCore) plus a
    thread pool — closer to linear scaling than `torch_neuronx.DataParallel`,
    which otherwise tends to serialise calls through the first-loaded core.
    """
    from concurrent.futures import ThreadPoolExecutor

    modules = []
    for i in range(dp):
        with torch_neuronx.experimental.placement.neuron_cores_context(
            start_nc=i, nc_count=1,
        ):
            modules.append(torch.jit.load(str(compiled_path)))
    pool = ThreadPoolExecutor(max_workers=max(2 * dp, 4))

    t_single, _, _ = preprocess_image(str(ASSETS_DIR / "bus.jpg"), imgsz=imgsz)
    per_core_batch = t_single.repeat(spec.batch_size, 1, 1, 1).contiguous()
    images_per_step = dp * spec.batch_size

    def step():
        futs = [pool.submit(m, per_core_batch) for m in modules]
        for f in futs:
            f.result()

    for _ in range(warmup):
        step()

    timings = []
    for _ in range(iters):
        t0 = time.perf_counter()
        step()
        timings.append((time.perf_counter() - t0) * 1000.0)
    arr = np.asarray(timings)
    step_mean = float(arr.mean())
    return {
        "dp": dp,
        "batch_per_core": spec.batch_size,
        "images_per_step": images_per_step,
        "step_mean_ms": step_mean,
        "step_p50_ms": float(np.median(arr)),
        "per_image_ms": step_mean / images_per_step,
        "throughput_img_per_sec": images_per_step / (step_mean / 1000.0),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ)
    parser.add_argument("--dp", type=int, default=8, help="NeuronCores for DataParallel")
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--force-recompile", action="store_true")
    parser.add_argument("--out-json", default=str(BENCHMARK_DIR / "benchmark_aligned.json"))
    parser.add_argument("--out-md", default=str(BENCHMARK_DIR / "benchmark_aligned.md"))
    args = parser.parse_args()

    results: Dict[str, dict] = {}
    for spec in REFERENCE_SPECS:
        if not _weights(spec.size).exists():
            print(f"[skip] {spec.size}: weights file missing"); continue
        print(f"\n=== yolo26{spec.size} {spec.dtype} BS/core={spec.batch_size} imgsz={args.imgsz} ===")
        row: Dict = {
            "dtype": spec.dtype,
            "batch_per_core": spec.batch_size,
            "pre_fuse_params": _pre_fuse_param_count(spec.size),
        }
        try:
            row["compile"] = _compile(spec, args.imgsz, args.force_recompile)
        except Exception as exc:
            print(f"[{spec.size}] compile FAILED: {exc}")
            row["compile"] = {"error": str(exc)}
            results[spec.size] = row
            continue

        row["throughput"] = _throughput(
            Path(row["compile"]["path"]), spec, args.imgsz,
            args.dp, args.iters, args.warmup,
        )
        t = row["throughput"]
        print(
            f"  params(pre-fuse)={row['pre_fuse_params'] / 1e6:.1f}M  "
            f"NEFF={row['compile']['neff_size_mb']:.1f}MB  "
            f"DP={t['dp']} BS/core={t['batch_per_core']}  "
            f"throughput={t['throughput_img_per_sec']:.0f} img/s  "
            f"per-image={t['per_image_ms']:.2f} ms"
        )
        results[spec.size] = row

    summary = {
        "imgsz": args.imgsz,
        "dp": args.dp,
        "iters": args.iters,
        "warmup": args.warmup,
        "instance": "trn2.48xlarge",
        "lnc": 1,
        "reference_specs": [
            {"size": s.size, "dtype": s.dtype, "batch_per_core": s.batch_size}
            for s in REFERENCE_SPECS
        ],
        "results": results,
    }
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {out_json}")

    # Markdown
    lines = [f"# YOLO26 aligned peak-throughput (LNC=1, DP={args.dp}, imgsz={args.imgsz})", ""]
    lines.append("Per-variant config matches the AWS Neuron reference table: dtype and BS/core vary per size.")
    lines.append("")
    lines.append("| variant | params (pre-fuse) | dtype | NEFF (MB) | BS/core | throughput (img/s) | per-image (ms) |")
    lines.append("|---------|------------------:|-------|----------:|--------:|-------------------:|---------------:|")
    for size, row in results.items():
        if "throughput" not in row:
            lines.append(f"| yolo26{size} | {row['pre_fuse_params']/1e6:.1f}M | {row['dtype']} | — | {row['batch_per_core']} | FAILED | — |")
            continue
        t = row["throughput"]
        c = row["compile"]
        lines.append(
            f"| yolo26{size} | {row['pre_fuse_params']/1e6:.1f}M | "
            f"{row['dtype'].upper()} | {c['neff_size_mb']:.1f} | "
            f"{row['batch_per_core']} | {t['throughput_img_per_sec']:.0f} | "
            f"{t['per_image_ms']:.2f} |"
        )
    Path(args.out_md).write_text("\n".join(lines) + "\n")
    print(f"wrote {args.out_md}")


if __name__ == "__main__":
    main()
