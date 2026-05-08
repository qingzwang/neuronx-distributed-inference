"""Run compiled YOLO26 on each sample image and render bounding boxes + labels.

Outputs `assets/<name>_pred.jpg` for every `assets/*.jpg`, using the same
deterministic COCO palette as ultralytics. Useful as a quick visual sanity
check and to embed directly in the README.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Sequence

import cv2
import numpy as np
import torch
import torch_neuronx  # noqa: F401 — needed to deserialise the traced NEFF

from yolo26_common import (
    ASSETS_DIR,
    COMPILED_DIR,
    DEFAULT_IMGSZ,
    Detection,
    load_class_names,
    neuron_topk,
    postprocess,
    preprocess_image,
)


# Deterministic per-class colour: hashes the class index to a BGR triple,
# matching the palette ultralytics uses so the two outputs look familiar.
_PALETTE = np.array(
    [
        [255, 56, 56], [255, 157, 151], [255, 112, 31], [255, 178, 29],
        [207, 210, 49], [72, 249, 10], [146, 204, 23], [61, 219, 134],
        [26, 147, 52], [0, 212, 187], [44, 153, 168], [0, 194, 255],
        [52, 69, 147], [100, 115, 255], [0, 24, 236], [132, 56, 255],
        [82, 0, 133], [203, 56, 255], [255, 149, 200], [255, 55, 199],
    ],
    dtype=np.uint8,
)


def _colour_for(cls: int) -> tuple[int, int, int]:
    rgb = _PALETTE[cls % len(_PALETTE)]
    return int(rgb[2]), int(rgb[1]), int(rgb[0])  # BGR for OpenCV


def draw_detections(
    image_bgr: np.ndarray,
    detections: Sequence[Detection],
    class_names: dict,
    line_thickness: int | None = None,
) -> np.ndarray:
    """Draw xyxy boxes + `"class score"` labels on top of the source image."""
    out = image_bgr.copy()
    h, w = out.shape[:2]
    if line_thickness is None:
        line_thickness = max(2, round(0.002 * max(h, w)))
    font_scale = max(0.5, line_thickness / 3)

    for d in detections:
        colour = _colour_for(d.cls)
        x1, y1, x2, y2 = map(int, (d.x1, d.y1, d.x2, d.y2))
        cv2.rectangle(out, (x1, y1), (x2, y2), colour, thickness=line_thickness)

        label = f"{class_names.get(d.cls, d.cls)} {d.score:.2f}"
        (tw, th), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, max(1, line_thickness - 1),
        )
        # Draw the label background just above the box; if it would go
        # off-screen, put it inside the top edge instead.
        ty_top = y1 - th - baseline - 2
        if ty_top < 0:
            bg_tl = (x1, y1)
            bg_br = (x1 + tw + 2, y1 + th + baseline + 2)
            text_org = (x1 + 1, y1 + th + 1)
        else:
            bg_tl = (x1, ty_top)
            bg_br = (x1 + tw + 2, y1)
            text_org = (x1 + 1, y1 - baseline - 1)
        cv2.rectangle(out, bg_tl, bg_br, colour, thickness=cv2.FILLED)
        text_colour = (0, 0, 0) if sum(colour) > 380 else (255, 255, 255)
        cv2.putText(
            out, label, text_org, cv2.FONT_HERSHEY_SIMPLEX,
            font_scale, text_colour, max(1, line_thickness - 1), cv2.LINE_AA,
        )
    return out


def _run_neuron(
    compiled_path: Path, tensor: torch.Tensor, info, conf: float,
) -> List[Detection]:
    mod = torch.jit.load(str(compiled_path))
    with torch.inference_mode():
        raw = mod(tensor)
    topk = neuron_topk(raw, max_det=300, nc=raw.shape[-1] - 4)
    return postprocess(topk, info, conf_thres=conf)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--compiled", default=str(COMPILED_DIR / "yolo26n_neuron_fp32.pt"),
        help="Path to a compiled Neuron NEFF",
    )
    parser.add_argument(
        "--weights", default=None,
        help="Weights file; only used to read class names. Defaults to ../yolo26n.pt",
    )
    parser.add_argument(
        "--images", nargs="+", default=None,
        help="Image paths; defaults to assets/*.jpg",
    )
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--suffix", default="_pred", help="Suffix appended to output filename")
    args = parser.parse_args()

    if args.weights is None:
        args.weights = str(ASSETS_DIR.parent / "yolo26n.pt")
    names = load_class_names(args.weights)

    if args.images is None:
        args.images = sorted(str(p) for p in ASSETS_DIR.glob("*.jpg")
                             if args.suffix not in p.stem)

    for path in args.images:
        src = Path(path)
        tensor, info, image_bgr = preprocess_image(str(src), imgsz=args.imgsz)
        dets = _run_neuron(Path(args.compiled), tensor, info, args.conf)
        annotated = draw_detections(image_bgr, dets, names)

        out_path = src.with_name(f"{src.stem}{args.suffix}.jpg")
        cv2.imwrite(str(out_path), annotated)
        print(f"{src.name}: {len(dets)} detections -> {out_path.name}")
        for d in dets:
            print(f"  {names.get(d.cls, d.cls):<12s} score={d.score:.3f} "
                  f"box=({d.x1:.0f},{d.y1:.0f},{d.x2:.0f},{d.y2:.0f})")


if __name__ == "__main__":
    main()
