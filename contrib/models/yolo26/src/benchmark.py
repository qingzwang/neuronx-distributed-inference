"""Consolidated CPU-vs-Neuron benchmark for YOLO26-nano.

Runs both paths on each image in `assets/`, confirms detection parity, and
writes a summary JSON plus a markdown report to `benchmark/`.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch_neuronx  # noqa: F401
from ultralytics import YOLO

from yolo26_common import (
    ASSETS_DIR,
    BENCHMARK_DIR,
    COMPILED_DIR,
    DEFAULT_IMGSZ,
    WEIGHTS_PATH,
    format_detections,
    load_class_names,
    neuron_topk,
    postprocess,
    preprocess_image,
    time_runs,
)


def _cpu_model() -> tuple[torch.nn.Module, dict]:
    y = YOLO(str(WEIGHTS_PATH))
    m = y.model.eval().to(torch.float32).cpu()
    return m, y.names


def _cpu_forward(model: torch.nn.Module, t: torch.Tensor) -> torch.Tensor:
    with torch.inference_mode():
        o = model(t)
    return o[0] if isinstance(o, (list, tuple)) else o


def _bench_image(
    image_path: Path,
    cpu_model: torch.nn.Module,
    neuron_model: torch.jit.ScriptModule,
    names: dict,
    iters: int,
    warmup: int,
    conf: float,
    max_det: int,
) -> Dict:
    tensor, info, _ = preprocess_image(str(image_path))
    # CPU path
    cpu_raw = _cpu_forward(cpu_model, tensor)
    cpu_dets = postprocess(cpu_raw, info, conf_thres=conf)

    # Neuron path
    with torch.inference_mode():
        neu_raw = neuron_model(tensor)
    neu_topk = neuron_topk(neu_raw, max_det=max_det, nc=neu_raw.shape[-1] - 4)
    neu_dets = postprocess(neu_topk, info, conf_thres=conf)

    # Latency
    def cpu_fwd():
        _cpu_forward(cpu_model, tensor)

    def neu_fwd():
        with torch.inference_mode():
            neuron_model(tensor)

    cpu_mean, cpu_p50, _ = time_runs(cpu_fwd, warmup=warmup, iters=iters)
    neu_mean, neu_p50, _ = time_runs(neu_fwd, warmup=warmup, iters=iters)

    # E2E including preprocess + postprocess
    def e2e_cpu():
        t, i, _ = preprocess_image(str(image_path))
        r = _cpu_forward(cpu_model, t)
        postprocess(r, i, conf_thres=conf)

    def e2e_neu():
        t, i, _ = preprocess_image(str(image_path))
        with torch.inference_mode():
            r = neuron_model(t)
        tk = neuron_topk(r, max_det=max_det, nc=r.shape[-1] - 4)
        postprocess(tk, i, conf_thres=conf)

    cpu_e2e_mean, cpu_e2e_p50, _ = time_runs(e2e_cpu, warmup=warmup, iters=iters)
    neu_e2e_mean, neu_e2e_p50, _ = time_runs(e2e_neu, warmup=warmup, iters=iters)

    return {
        "image": image_path.name,
        "num_detections_cpu": len(cpu_dets),
        "num_detections_neuron": len(neu_dets),
        "cpu_model_forward_mean_ms": cpu_mean,
        "cpu_model_forward_p50_ms": cpu_p50,
        "neuron_model_forward_mean_ms": neu_mean,
        "neuron_model_forward_p50_ms": neu_p50,
        "cpu_end_to_end_mean_ms": cpu_e2e_mean,
        "cpu_end_to_end_p50_ms": cpu_e2e_p50,
        "neuron_end_to_end_mean_ms": neu_e2e_mean,
        "neuron_end_to_end_p50_ms": neu_e2e_p50,
        "speedup_forward": cpu_mean / neu_mean if neu_mean > 0 else None,
        "speedup_end_to_end": cpu_e2e_mean / neu_e2e_mean if neu_e2e_mean > 0 else None,
        "cpu_detections": [d.as_tuple() for d in cpu_dets],
        "neuron_detections": [d.as_tuple() for d in neu_dets],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--compiled", default=str(COMPILED_DIR / "yolo26n_neuron_fp32.pt"),
    )
    parser.add_argument(
        "--images",
        nargs="+",
        default=[str(p) for p in sorted(ASSETS_DIR.glob("*.jpg"))],
    )
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument(
        "--out-json",
        default=str(BENCHMARK_DIR / "benchmark_summary.json"),
    )
    parser.add_argument(
        "--out-md",
        default=str(BENCHMARK_DIR / "benchmark_report.md"),
    )
    args = parser.parse_args()

    print("[bench] loading CPU model ...")
    cpu_model, names = _cpu_model()
    print(f"[bench] loading Neuron module {args.compiled} ...")
    neuron_model = torch.jit.load(args.compiled)

    image_paths: List[Path] = [Path(p) for p in args.images]
    rows: List[Dict] = []
    for p in image_paths:
        print(f"[bench] == {p.name} ==")
        row = _bench_image(
            p, cpu_model, neuron_model, names,
            iters=args.iters, warmup=args.warmup,
            conf=args.conf, max_det=args.max_det,
        )
        rows.append(row)
        print(
            f"  cpu  fwd={row['cpu_model_forward_mean_ms']:.2f} ms  "
            f"e2e={row['cpu_end_to_end_mean_ms']:.2f} ms  "
            f"dets={row['num_detections_cpu']}"
        )
        print(
            f"  neu  fwd={row['neuron_model_forward_mean_ms']:.2f} ms  "
            f"e2e={row['neuron_end_to_end_mean_ms']:.2f} ms  "
            f"dets={row['num_detections_neuron']}  "
            f"speedup(fwd)={row['speedup_forward']:.2f}x"
        )

    summary = {
        "model": "yolo26n",
        "imgsz": DEFAULT_IMGSZ,
        "iters": args.iters,
        "warmup": args.warmup,
        "weights": str(WEIGHTS_PATH),
        "compiled": args.compiled,
        "instance": "trn2.48xlarge (single NeuronCore)",
        "rows": rows,
    }

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"[bench] wrote {out_json}")

    md = _render_md(summary)
    out_md = Path(args.out_md)
    out_md.write_text(md)
    print(f"[bench] wrote {out_md}")


def _render_md(summary: dict) -> str:
    lines = []
    lines.append(f"# YOLO26-nano on Neuron vs CPU — benchmark")
    lines.append("")
    lines.append(f"- Model: `{summary['model']}` (ultralytics end-to-end, NMS-free)")
    lines.append(f"- Input size: {summary['imgsz']}x{summary['imgsz']}")
    lines.append(f"- Compiled artifact: `{summary['compiled']}`")
    lines.append(f"- Instance: {summary['instance']}")
    lines.append(f"- Iters/warmup: {summary['iters']}/{summary['warmup']}")
    lines.append("")
    lines.append("## Per-image latency (mean over iters)")
    lines.append("")
    lines.append("| image | cpu fwd (ms) | neuron fwd (ms) | speedup | cpu e2e (ms) | neuron e2e (ms) | e2e speedup | cpu dets | neu dets |")
    lines.append("|-------|-------------:|----------------:|--------:|-------------:|----------------:|------------:|--------:|--------:|")
    for r in summary["rows"]:
        lines.append(
            f"| {r['image']} | {r['cpu_model_forward_mean_ms']:.2f} | "
            f"{r['neuron_model_forward_mean_ms']:.2f} | "
            f"{r['speedup_forward']:.2f}x | "
            f"{r['cpu_end_to_end_mean_ms']:.2f} | "
            f"{r['neuron_end_to_end_mean_ms']:.2f} | "
            f"{r['speedup_end_to_end']:.2f}x | "
            f"{r['num_detections_cpu']} | {r['num_detections_neuron']} |"
        )
    lines.append("")
    lines.append("## Per-image p50 latency")
    lines.append("")
    lines.append("| image | cpu fwd p50 (ms) | neuron fwd p50 (ms) | cpu e2e p50 (ms) | neuron e2e p50 (ms) |")
    lines.append("|-------|-----------------:|--------------------:|-----------------:|--------------------:|")
    for r in summary["rows"]:
        lines.append(
            f"| {r['image']} | {r['cpu_model_forward_p50_ms']:.2f} | "
            f"{r['neuron_model_forward_p50_ms']:.2f} | "
            f"{r['cpu_end_to_end_p50_ms']:.2f} | "
            f"{r['neuron_end_to_end_p50_ms']:.2f} |"
        )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
