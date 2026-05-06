"""Shared helpers for running YOLO26 inference on CPU or Neuron.

YOLO26 is end-to-end: the model emits a fixed-shape `(B, max_det, 6)` tensor
with already-decoded boxes (xyxy in model input coordinates), scores and class
ids. No NMS or grid decoding is needed on the host side - only letterbox
preprocessing and coordinate rescaling back to the original image.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import cv2
import numpy as np
import torch


DEFAULT_IMGSZ = 640


@dataclass
class LetterboxInfo:
    """Describes the letterbox transform applied to a source image."""

    scale: float
    pad_x: float
    pad_y: float
    orig_h: int
    orig_w: int


def letterbox(
    image_bgr: np.ndarray,
    imgsz: int = DEFAULT_IMGSZ,
    color: Tuple[int, int, int] = (114, 114, 114),
) -> Tuple[np.ndarray, LetterboxInfo]:
    """Resize + pad a BGR image to a square `imgsz x imgsz` canvas."""
    h0, w0 = image_bgr.shape[:2]
    scale = min(imgsz / h0, imgsz / w0)
    new_w, new_h = int(round(w0 * scale)), int(round(h0 * scale))
    resized = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    pad_x = (imgsz - new_w) / 2
    pad_y = (imgsz - new_h) / 2
    top, bottom = int(round(pad_y - 0.1)), int(round(pad_y + 0.1))
    left, right = int(round(pad_x - 0.1)), int(round(pad_x + 0.1))
    canvas = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    info = LetterboxInfo(scale=scale, pad_x=pad_x, pad_y=pad_y, orig_h=h0, orig_w=w0)
    return canvas, info


def preprocess_image(
    image_path: str,
    imgsz: int = DEFAULT_IMGSZ,
) -> Tuple[torch.Tensor, LetterboxInfo, np.ndarray]:
    """Load an image and return the model-ready tensor plus letterbox metadata."""
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise FileNotFoundError(f"cannot read image {image_path}")
    canvas, info = letterbox(image_bgr, imgsz=imgsz)
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    arr = rgb.astype(np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))  # CHW
    tensor = torch.from_numpy(arr).unsqueeze(0).contiguous()
    return tensor, info, image_bgr


@dataclass
class Detection:
    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    cls: int

    def as_tuple(self) -> Tuple[float, float, float, float, float, int]:
        return (self.x1, self.y1, self.x2, self.y2, self.score, self.cls)


def neuron_topk(preds: torch.Tensor, max_det: int = 300, nc: int = 80) -> torch.Tensor:
    """CPU-side counterpart of the ultralytics end-to-end `Detect.postprocess`.

    Input:  `(B, num_anchors, 4 + nc)` with xyxy boxes and sigmoid class scores.
    Output: `(B, max_det, 6)` `[x1, y1, x2, y2, score, cls]`, matching the
    tensor shape the eager model produces end-to-end.
    """
    assert preds.shape[-1] == 4 + nc, f"unexpected last dim {preds.shape[-1]}"
    boxes, scores = preds.split([4, nc], dim=-1)
    batch_size, anchors, _ = scores.shape
    k = min(max_det, anchors)
    # Replicates ultralytics' class-aware top-k: first pick top-k anchors by
    # their best-class score, then flatten (k, nc) and take top-k again.
    ori_index = scores.max(dim=-1)[0].topk(k, dim=-1)[1].unsqueeze(-1)  # (B, k, 1)
    gathered_scores = scores.gather(dim=1, index=ori_index.repeat(1, 1, nc))  # (B, k, nc)
    flat_scores, index = gathered_scores.flatten(1).topk(k, dim=-1)  # (B, k)
    batch_arange = torch.arange(batch_size, device=preds.device)[..., None]
    idx = ori_index[batch_arange, index // nc]  # (B, k, 1)
    cls = (index % nc)[..., None].to(preds.dtype)  # (B, k, 1)
    out_boxes = boxes.gather(dim=1, index=idx.repeat(1, 1, 4))  # (B, k, 4)
    return torch.cat([out_boxes, flat_scores[..., None], cls], dim=-1)


def postprocess(
    raw: torch.Tensor,
    info: LetterboxInfo,
    conf_thres: float = 0.25,
) -> List[Detection]:
    """Convert model output back to original image coordinates.

    `raw` is shaped `(max_det, 6)` with columns `[x1, y1, x2, y2, score, cls]`
    in letterboxed input space.
    """
    if raw.ndim == 3:
        raw = raw[0]
    arr = raw.detach().to(torch.float32).cpu().numpy()
    keep = arr[:, 4] >= conf_thres
    arr = arr[keep]
    if arr.size == 0:
        return []
    # undo letterbox: first remove padding, then unscale
    arr[:, [0, 2]] -= info.pad_x
    arr[:, [1, 3]] -= info.pad_y
    arr[:, :4] /= info.scale
    arr[:, [0, 2]] = np.clip(arr[:, [0, 2]], 0, info.orig_w - 1)
    arr[:, [1, 3]] = np.clip(arr[:, [1, 3]], 0, info.orig_h - 1)
    dets: List[Detection] = []
    for row in arr:
        dets.append(Detection(
            x1=float(row[0]), y1=float(row[1]),
            x2=float(row[2]), y2=float(row[3]),
            score=float(row[4]), cls=int(row[5]),
        ))
    return dets


def time_runs(
    fn,
    *args,
    warmup: int = 3,
    iters: int = 20,
) -> Tuple[float, float, List[float]]:
    """Run `fn(*args)` with warmup and return (mean_ms, p50_ms, all_ms)."""
    for _ in range(warmup):
        fn(*args)
    timings: List[float] = []
    for _ in range(iters):
        start = time.perf_counter()
        fn(*args)
        timings.append((time.perf_counter() - start) * 1000.0)
    arr = np.asarray(timings)
    return float(arr.mean()), float(np.median(arr)), timings


def format_detections(dets: Sequence[Detection], names: dict) -> str:
    lines = []
    for d in dets:
        name = names.get(d.cls, str(d.cls))
        lines.append(
            f"  {name:<14s} score={d.score:.3f} "
            f"box=({d.x1:.1f},{d.y1:.1f},{d.x2:.1f},{d.y2:.1f})"
        )
    return "\n".join(lines) if lines else "  (no detections above threshold)"


def load_class_names(weights_path: str) -> dict:
    """Extract the class-name dict from a YOLO26 checkpoint without a forward pass."""
    ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
    model = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    names = getattr(model, "names", None)
    if names is None:
        names = {i: str(i) for i in range(80)}
    return dict(names)


PROJECT_DIR = Path(__file__).resolve().parent.parent
WEIGHTS_PATH = PROJECT_DIR / "yolo26n.pt"
ASSETS_DIR = PROJECT_DIR / "assets"
COMPILED_DIR = PROJECT_DIR / "compiled"
BENCHMARK_DIR = PROJECT_DIR / "benchmark"
