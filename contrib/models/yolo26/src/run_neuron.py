"""Run the compiled YOLO26 on a Neuron core, verify vs CPU, benchmark latency."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch_neuronx  # noqa: F401 — required for torch.jit.load of a neuron artifact
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


def run_cpu_reference(
    weights: str, tensor: torch.Tensor
) -> torch.Tensor:
    """Eager-mode CPU forward used as ground truth for output matching."""
    yolo = YOLO(weights)
    model = yolo.model.eval().to(torch.float32).cpu()
    with torch.inference_mode():
        out = model(tensor)
    if isinstance(out, (list, tuple)):
        out = out[0]
    return out  # (B, 300, 6)


def compare_detections(cpu: torch.Tensor, neuron: torch.Tensor, conf: float) -> dict:
    """Match detections by (class, IoU>=0.9) and report stats."""
    cpu_np = cpu[0].cpu().numpy()
    neu_np = neuron[0].cpu().numpy()
    cpu_keep = cpu_np[cpu_np[:, 4] >= conf]
    neu_keep = neu_np[neu_np[:, 4] >= conf]

    def iou(a, b):
        ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
        ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
        iw = max(0.0, ix2 - ix1); ih = max(0.0, iy2 - iy1)
        inter = iw * ih
        area_a = max(0.0, (a[2] - a[0])) * max(0.0, (a[3] - a[1]))
        area_b = max(0.0, (b[2] - b[0])) * max(0.0, (b[3] - b[1]))
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

    matches = []
    unmatched_cpu = list(range(len(cpu_keep)))
    for j, nd in enumerate(neu_keep):
        best_i, best_iou = -1, 0.0
        for i in unmatched_cpu:
            cd = cpu_keep[i]
            if int(cd[5]) != int(nd[5]):
                continue
            v = iou(cd, nd)
            if v > best_iou:
                best_iou, best_i = v, i
        if best_i >= 0 and best_iou >= 0.9:
            matches.append((best_i, j, best_iou,
                            float(cpu_keep[best_i, 4]), float(nd[4])))
            unmatched_cpu.remove(best_i)

    score_deltas = [abs(m[3] - m[4]) for m in matches]
    return {
        "cpu_detections": int(len(cpu_keep)),
        "neuron_detections": int(len(neu_keep)),
        "matched": int(len(matches)),
        "max_score_delta": float(max(score_deltas)) if score_deltas else 0.0,
        "mean_score_delta": float(np.mean(score_deltas)) if score_deltas else 0.0,
        "min_iou_among_matches": float(min((m[2] for m in matches), default=1.0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default=str(ASSETS_DIR / "bus.jpg"))
    parser.add_argument("--weights", default=str(WEIGHTS_PATH))
    parser.add_argument(
        "--compiled", default=str(COMPILED_DIR / "yolo26n_neuron.pt"),
    )
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument(
        "--out-json", default=str(BENCHMARK_DIR / "neuron_latency.json"),
    )
    args = parser.parse_args()

    print(f"[neuron] loading compiled module from {args.compiled}")
    neuron_model = torch.jit.load(args.compiled)

    tensor, info, _ = preprocess_image(args.image, imgsz=args.imgsz)
    print(f"[neuron] input tensor shape={tuple(tensor.shape)} dtype={tensor.dtype}")

    # Neuron forward -> raw preds, then CPU top-k + letterbox-unmap
    with torch.inference_mode():
        raw = neuron_model(tensor)
    print(f"[neuron] raw model output shape={tuple(raw.shape)} dtype={raw.dtype}")

    topk = neuron_topk(raw, max_det=args.max_det, nc=raw.shape[-1] - 4)
    dets = postprocess(topk, info, conf_thres=args.conf)
    names = load_class_names(args.weights)
    print(f"[neuron] {len(dets)} detections on {args.image}:")
    print(format_detections(dets, names))

    # Correctness vs. CPU eager forward
    print("[neuron] running CPU reference forward for output matching ...")
    cpu_raw = run_cpu_reference(args.weights, tensor)
    accuracy = compare_detections(cpu_raw, topk, conf=args.conf)
    print(f"[neuron] detection match vs CPU: {accuracy}")

    # Latency: compiled forward only
    def forward_only(t: torch.Tensor) -> torch.Tensor:
        with torch.inference_mode():
            return neuron_model(t)

    mean_ms, p50_ms, _ = time_runs(forward_only, tensor, warmup=args.warmup, iters=args.iters)
    print(f"[neuron] model forward only: mean={mean_ms:.2f} ms  p50={p50_ms:.2f} ms (iters={args.iters})")

    # End-to-end (preprocess + neuron forward + cpu topk + unmap)
    def end_to_end(path: str) -> int:
        t, i, _ = preprocess_image(path, imgsz=args.imgsz)
        with torch.inference_mode():
            r = neuron_model(t)
        tk = neuron_topk(r, max_det=args.max_det, nc=r.shape[-1] - 4)
        d = postprocess(tk, i, conf_thres=args.conf)
        return len(d)

    e2e_mean_ms, e2e_p50_ms, _ = time_runs(
        end_to_end, args.image, warmup=args.warmup, iters=args.iters
    )
    print(f"[neuron] end-to-end:         mean={e2e_mean_ms:.2f} ms  p50={e2e_p50_ms:.2f} ms")

    summary = {
        "device": "neuron",
        "weights": str(args.weights),
        "compiled": str(args.compiled),
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
        "accuracy_vs_cpu": accuracy,
    }
    if args.out_json:
        out = Path(args.out_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2))
        print(f"[neuron] wrote summary to {out}")


if __name__ == "__main__":
    main()
