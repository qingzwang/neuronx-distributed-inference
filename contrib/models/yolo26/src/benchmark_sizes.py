"""Sweep yolo26 {n,s,m,l,x} on CPU, Neuron fp32 single-core, Neuron fp16
single-core, and Neuron fp32 multi-core data-parallel.

Produces:
  benchmark/benchmark_sizes.json   # raw numbers
  benchmark/benchmark_sizes.md     # tables
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch_neuronx
from ultralytics import YOLO

from neuron_patches import patch_attention_modules
from yolo26_common import (
    ASSETS_DIR,
    BENCHMARK_DIR,
    COMPILED_DIR,
    DEFAULT_IMGSZ,
    PROJECT_DIR,
    neuron_topk,
    postprocess,
    preprocess_image,
    time_runs,
)

# Re-use the compile logic from compile_neuron.py without shelling out.
from compile_neuron import YOLO26Wrapper, _patch_detect_head


SIZES = ["n", "s", "m", "l", "x"]
DTYPES = ["fp32", "fp16"]


def _weights_path(size: str) -> Path:
    return PROJECT_DIR / f"yolo26{size}.pt"


def _compiled_path(size: str, dtype: str, imgsz: int) -> Path:
    suffix = "" if imgsz == DEFAULT_IMGSZ else f"_sz{imgsz}"
    return COMPILED_DIR / f"yolo26{size}_neuron_{dtype}{suffix}.pt"


def _cpu_bench(size: str, image: Path, iters: int, warmup: int, conf: float, imgsz: int) -> dict:
    yolo = YOLO(str(_weights_path(size)))
    model = yolo.model.eval().to(torch.float32).cpu()

    tensor, info, _ = preprocess_image(str(image), imgsz=imgsz)

    def fwd():
        with torch.inference_mode():
            out = model(tensor)
        return out[0] if isinstance(out, (list, tuple)) else out

    # warmup + correctness
    raw = fwd()
    dets = postprocess(raw, info, conf_thres=conf)
    mean_ms, p50_ms, _ = time_runs(fwd, warmup=warmup, iters=iters)
    return {
        "model_forward_mean_ms": mean_ms,
        "model_forward_p50_ms": p50_ms,
        "num_detections": len(dets),
    }


def _compile(size: str, dtype: str, imgsz: int, force: bool) -> dict:
    out = _compiled_path(size, dtype, imgsz)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and not force:
        return {"path": str(out), "compile_seconds": None, "cached": True}

    print(f"[compile] yolo26{size} dtype={dtype} -> {out}")
    yolo = YOLO(str(_weights_path(size)))
    model = yolo.model.eval().to(torch.float32).cpu().fuse()
    _patch_detect_head(model)
    patch_attention_modules(model)
    wrapper = YOLO26Wrapper(model).eval()
    example = torch.zeros(1, 3, imgsz, imgsz, dtype=torch.float32)

    common = ["--target=trn2", "--logical-nc-config=1", "--enable-saturate-infinity"]
    if dtype == "fp32":
        compiler_args = common + ["--auto-cast=none", "--optlevel=2"]
    elif dtype == "fp16":
        compiler_args = common + ["--auto-cast=matmult", "--auto-cast-type=fp16"]
    else:
        raise ValueError(dtype)

    t0 = time.perf_counter()
    neuron_model = torch_neuronx.trace(wrapper, example, compiler_args=compiler_args)
    dt = time.perf_counter() - t0
    torch.jit.save(neuron_model, str(out))
    return {"path": str(out), "compile_seconds": dt, "cached": False}


def _neuron_single_core(
    compiled_path: Path, image: Path, iters: int, warmup: int, conf: float, imgsz: int,
) -> dict:
    mod = torch.jit.load(str(compiled_path))
    tensor, info, _ = preprocess_image(str(image), imgsz=imgsz)

    def fwd():
        with torch.inference_mode():
            return mod(tensor)

    raw = fwd()
    topk = neuron_topk(raw, max_det=300, nc=raw.shape[-1] - 4)
    dets = postprocess(topk, info, conf_thres=conf)
    mean_ms, p50_ms, _ = time_runs(fwd, warmup=warmup, iters=iters)
    return {
        "model_forward_mean_ms": mean_ms,
        "model_forward_p50_ms": p50_ms,
        "num_detections": len(dets),
    }


def _parity_check(size: str, compiled_path: Path, image: Path, conf: float, imgsz: int) -> dict:
    """Compare Neuron detections vs CPU eager on the same image."""
    yolo = YOLO(str(_weights_path(size)))
    cpu = yolo.model.eval().to(torch.float32).cpu()
    tensor, info, _ = preprocess_image(str(image), imgsz=imgsz)
    with torch.inference_mode():
        cpu_out = cpu(tensor)
    cpu_raw = cpu_out[0] if isinstance(cpu_out, (list, tuple)) else cpu_out
    cpu_dets = postprocess(cpu_raw, info, conf_thres=conf)

    mod = torch.jit.load(str(compiled_path))
    with torch.inference_mode():
        neu_raw = mod(tensor)
    neu_topk = neuron_topk(neu_raw, max_det=300, nc=neu_raw.shape[-1] - 4)
    neu_dets = postprocess(neu_topk, info, conf_thres=conf)

    cpu_sorted = sorted(cpu_dets, key=lambda d: -d.score)
    neu_sorted = sorted(neu_dets, key=lambda d: -d.score)
    matched = min(len(cpu_sorted), len(neu_sorted))
    score_deltas = []
    iou_list = []
    for c, n in zip(cpu_sorted[:matched], neu_sorted[:matched]):
        if c.cls == n.cls:
            score_deltas.append(abs(c.score - n.score))
            # IoU
            ix1 = max(c.x1, n.x1); iy1 = max(c.y1, n.y1)
            ix2 = min(c.x2, n.x2); iy2 = min(c.y2, n.y2)
            iw = max(0.0, ix2 - ix1); ih = max(0.0, iy2 - iy1)
            inter = iw * ih
            a = max(0.0, c.x2 - c.x1) * max(0.0, c.y2 - c.y1)
            b = max(0.0, n.x2 - n.x1) * max(0.0, n.y2 - n.y1)
            iou_list.append(inter / (a + b - inter) if (a + b - inter) > 0 else 0.0)
    return {
        "cpu_detections": len(cpu_dets),
        "neuron_detections": len(neu_dets),
        "max_score_delta": float(max(score_deltas)) if score_deltas else 0.0,
        "mean_score_delta": float(np.mean(score_deltas)) if score_deltas else 0.0,
        "min_iou": float(min(iou_list)) if iou_list else 1.0,
    }


def _multicore(
    compiled_path: Path,
    image: Path,
    core_counts: List[int],
    iters: int,
    warmup: int,
    imgsz: int,
) -> List[dict]:
    # Build a tensor once; batch is num_cores, same image replicated.
    base_t, _, _ = preprocess_image(str(image), imgsz=imgsz)
    rows = []
    for n in core_counts:
        mod = torch.jit.load(str(compiled_path))
        device_ids = [f"nc:{i}" for i in range(n)]
        runner = torch_neuronx.DataParallel(mod, device_ids=device_ids)
        runner.num_workers = max(2 * n, 4)

        batch = base_t.repeat(n, 1, 1, 1).contiguous()
        for _ in range(warmup):
            _ = runner(batch)
        timings = []
        for _ in range(iters):
            t0 = time.perf_counter()
            _ = runner(batch)
            timings.append((time.perf_counter() - t0) * 1000.0)
        arr = np.asarray(timings)
        step_mean = float(arr.mean())
        rows.append({
            "num_cores": n,
            "batch": n,
            "step_mean_ms": step_mean,
            "step_p50_ms": float(np.median(arr)),
            "per_image_ms": step_mean / n,
            "throughput_img_per_sec": n / (step_mean / 1000.0),
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", nargs="+", default=SIZES)
    parser.add_argument("--dtypes", nargs="+", default=DTYPES)
    parser.add_argument("--image", default=str(ASSETS_DIR / "bus.jpg"))
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument(
        "--core-counts", type=int, nargs="+", default=[1, 8, 32],
        help="NeuronCore counts to sweep for multi-core throughput",
    )
    parser.add_argument("--force-recompile", action="store_true")
    parser.add_argument("--out-json", default=None)
    parser.add_argument("--out-md", default=None)
    args = parser.parse_args()

    suffix = "" if args.imgsz == DEFAULT_IMGSZ else f"_sz{args.imgsz}"
    if args.out_json is None:
        args.out_json = str(BENCHMARK_DIR / f"benchmark_sizes{suffix}.json")
    if args.out_md is None:
        args.out_md = str(BENCHMARK_DIR / f"benchmark_sizes{suffix}.md")

    image_path = Path(args.image)
    results: Dict[str, dict] = {}

    for size in args.sizes:
        w = _weights_path(size)
        if not w.exists():
            print(f"[skip] {w} not found"); continue
        print(f"\n=== yolo26{size} ===")
        r: Dict = {"weights_size_mb": w.stat().st_size / 1024 / 1024}

        print(f"[{size}] CPU ...")
        r["cpu"] = _cpu_bench(size, image_path, args.iters, args.warmup, args.conf, args.imgsz)

        compiled_ok: Dict[str, Path] = {}
        for dtype in args.dtypes:
            print(f"[{size}] compile {dtype} ...")
            try:
                comp = _compile(size, dtype, args.imgsz, args.force_recompile)
            except Exception as exc:
                print(f"[{size}] compile {dtype} FAILED: {exc}")
                r[f"compile_{dtype}"] = {"error": str(exc)}
                continue
            r[f"compile_{dtype}"] = comp
            compiled_ok[dtype] = Path(comp["path"])

            print(f"[{size}] neuron {dtype} single-core ...")
            r[f"neuron_{dtype}_single"] = _neuron_single_core(
                Path(comp["path"]), image_path, args.iters, args.warmup, args.conf, args.imgsz,
            )
            print(f"[{size}] parity {dtype} vs CPU ...")
            r[f"neuron_{dtype}_parity"] = _parity_check(
                size, Path(comp["path"]), image_path, args.conf, args.imgsz,
            )

        # Multi-core on the best available artifact: fp32 first, else fp16.
        mc_dtype = "fp32" if "fp32" in compiled_ok else ("fp16" if "fp16" in compiled_ok else None)
        if mc_dtype:
            print(f"[{size}] multicore {mc_dtype} cores={args.core_counts} ...")
            r[f"multicore_{mc_dtype}"] = _multicore(
                compiled_ok[mc_dtype], image_path,
                args.core_counts, args.iters, args.warmup, args.imgsz,
            )
            r["multicore_dtype"] = mc_dtype

        results[size] = r
        # Per-row summary
        cpu_ms = r["cpu"]["model_forward_mean_ms"]
        fp32_ms = r.get("neuron_fp32_single", {}).get("model_forward_mean_ms")
        fp16_ms = r.get("neuron_fp16_single", {}).get("model_forward_mean_ms")
        msg = f"[{size}] cpu={cpu_ms:.1f} ms"
        if fp32_ms:
            msg += f"  neuron fp32={fp32_ms:.1f} ms"
        if fp16_ms:
            msg += f"  neuron fp16={fp16_ms:.1f} ms"
        if "multicore_fp32" in r:
            peak = max(row["throughput_img_per_sec"] for row in r["multicore_fp32"])
            msg += f"  peak throughput={peak:.1f} img/s"
        print(msg)

    summary = {
        "image": str(image_path),
        "imgsz": args.imgsz,
        "iters": args.iters,
        "warmup": args.warmup,
        "conf": args.conf,
        "core_counts": args.core_counts,
        "results": results,
    }

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {out_json}")

    _render_md(summary, Path(args.out_md))
    print(f"wrote {args.out_md}")


def _render_md(summary: dict, out_md: Path) -> None:
    lines = ["# YOLO26 size sweep — CPU vs Neuron (trn2)", ""]
    lines.append(f"- Image: `{Path(summary['image']).name}`  ({summary['imgsz']}x{summary['imgsz']})")
    lines.append(f"- Iters/warmup: {summary['iters']}/{summary['warmup']}")
    lines.append(f"- Multi-core sweep: cores={summary['core_counts']}")
    lines.append("")
    lines.append("## Single-image latency (model forward only, mean ms)")
    lines.append("")
    lines.append("| variant | weights (MB) | CPU | Neuron fp32 | fp32 speedup | Neuron fp16 | fp16 speedup |")
    lines.append("|---------|-------------:|----:|------------:|-------------:|------------:|-------------:|")
    for size, r in summary["results"].items():
        cpu_ms = r["cpu"]["model_forward_mean_ms"]
        fp32 = r.get("neuron_fp32_single", {}).get("model_forward_mean_ms")
        fp16 = r.get("neuron_fp16_single", {}).get("model_forward_mean_ms")
        fp32_s = f"{fp32:.2f}" if fp32 else "—"
        fp16_s = f"{fp16:.2f}" if fp16 else "—"
        fp32_sp = f"{cpu_ms / fp32:.2f}x" if fp32 else "—"
        fp16_sp = f"{cpu_ms / fp16:.2f}x" if fp16 else "—"
        lines.append(
            f"| yolo26{size} | {r['weights_size_mb']:.1f} | {cpu_ms:.2f} | "
            f"{fp32_s} | {fp32_sp} | {fp16_s} | {fp16_sp} |"
        )
    lines.append("")
    lines.append("## Accuracy (Neuron vs CPU, same image, same conf=0.25)")
    lines.append("")
    lines.append("| variant | cpu dets | fp32 dets | fp32 max Δ | fp32 min IoU | fp16 dets | fp16 max Δ | fp16 min IoU |")
    lines.append("|---------|--------:|----------:|-----------:|-------------:|----------:|-----------:|-------------:|")
    for size, r in summary["results"].items():
        p32 = r.get("neuron_fp32_parity", {})
        p16 = r.get("neuron_fp16_parity", {})
        lines.append(
            f"| yolo26{size} | {p32.get('cpu_detections', '—')} | "
            f"{p32.get('neuron_detections', '—')} | "
            f"{p32.get('max_score_delta', 0):.2e} | {p32.get('min_iou', 1):.4f} | "
            f"{p16.get('neuron_detections', '—')} | "
            f"{p16.get('max_score_delta', 0):.2e} | {p16.get('min_iou', 1):.4f} |"
        )
    lines.append("")
    lines.append("## Multi-core throughput (Neuron data-parallel)")
    lines.append("")
    header = ["variant", "dtype"] + [f"{c} cores (img/s)" for c in summary["core_counts"]] + [
        f"{summary['core_counts'][-1]} cores per-img (ms)"
    ]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---" for _ in header]) + "|")
    for size, r in summary["results"].items():
        mc_dtype = r.get("multicore_dtype")
        rows = r.get(f"multicore_{mc_dtype}", []) if mc_dtype else []
        by_cores = {row["num_cores"]: row for row in rows}
        cells = [f"yolo26{size}", mc_dtype or "—"]
        for c in summary["core_counts"]:
            row = by_cores.get(c)
            cells.append(f"{row['throughput_img_per_sec']:.1f}" if row else "—")
        top = by_cores.get(summary["core_counts"][-1])
        cells.append(f"{top['per_image_ms']:.2f}" if top else "—")
        lines.append("| " + " | ".join(cells) + " |")
    out_md.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
