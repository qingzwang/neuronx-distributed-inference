"""Run YOLO26-nano on CPU with letterbox pre/postprocessing, print latency."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from ultralytics import YOLO

from yolo26_common import (
    ASSETS_DIR,
    BENCHMARK_DIR,
    DEFAULT_IMGSZ,
    WEIGHTS_PATH,
    format_detections,
    postprocess,
    preprocess_image,
    time_runs,
)


def load_model(weights_path: Path) -> torch.nn.Module:
    yolo = YOLO(str(weights_path))
    model = yolo.model
    model.eval()
    # Ultralytics picks dtype/device on first forward; pin to cpu/float32 explicitly.
    model = model.to(torch.float32).cpu()
    return model, yolo.names


def run_single(model: torch.nn.Module, tensor: torch.Tensor) -> torch.Tensor:
    with torch.inference_mode():
        out = model(tensor)
    # end-to-end head returns (dets, aux_dict); we want the decoded detections only.
    if isinstance(out, (list, tuple)):
        out = out[0]
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default=str(ASSETS_DIR / "bus.jpg"))
    parser.add_argument("--weights", default=str(WEIGHTS_PATH))
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument(
        "--out-json",
        default=str(BENCHMARK_DIR / "cpu_latency.json"),
        help="where to dump the benchmark summary (set empty string to skip)",
    )
    args = parser.parse_args()

    print(f"[cpu] loading {args.weights}")
    model, names = load_model(Path(args.weights))

    tensor, info, _ = preprocess_image(args.image, imgsz=args.imgsz)
    print(f"[cpu] input tensor shape={tuple(tensor.shape)} dtype={tensor.dtype}")

    # Warm / confirm correctness
    raw = run_single(model, tensor)
    dets = postprocess(raw, info, conf_thres=args.conf)
    print(f"[cpu] {len(dets)} detections on {args.image}:")
    print(format_detections(dets, names))

    # Pure model forward (what the Neuron compiled module will run)
    mean_ms, p50_ms, _ = time_runs(run_single, model, tensor, warmup=args.warmup, iters=args.iters)
    print(f"[cpu] model forward only: mean={mean_ms:.2f} ms  p50={p50_ms:.2f} ms (iters={args.iters})")

    # End-to-end timing including preprocess + postprocess
    def end_to_end(path: str) -> int:
        t, i, _ = preprocess_image(path, imgsz=args.imgsz)
        r = run_single(model, t)
        d = postprocess(r, i, conf_thres=args.conf)
        return len(d)

    e2e_mean_ms, e2e_p50_ms, _ = time_runs(
        end_to_end, args.image, warmup=args.warmup, iters=args.iters
    )
    print(f"[cpu] end-to-end:         mean={e2e_mean_ms:.2f} ms  p50={e2e_p50_ms:.2f} ms")

    summary = {
        "device": "cpu",
        "weights": str(args.weights),
        "image": str(args.image),
        "imgsz": args.imgsz,
        "iters": args.iters,
        "warmup": args.warmup,
        "model_forward_mean_ms": mean_ms,
        "model_forward_p50_ms": p50_ms,
        "end_to_end_mean_ms": e2e_mean_ms,
        "end_to_end_p50_ms": e2e_p50_ms,
        "num_detections": len(dets),
        "detections": [d.as_tuple() for d in dets],
    }
    if args.out_json:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2))
        print(f"[cpu] wrote summary to {out_path}")


if __name__ == "__main__":
    main()
