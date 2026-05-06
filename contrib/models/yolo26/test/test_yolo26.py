"""Lightweight parity check: Neuron detections must match CPU within tight tolerance."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))

from yolo26_common import (  # noqa: E402
    ASSETS_DIR,
    COMPILED_DIR,
    WEIGHTS_PATH,
    neuron_topk,
    postprocess,
    preprocess_image,
)


def _neuron_available() -> bool:
    compiled = COMPILED_DIR / "yolo26n_neuron_fp32.pt"
    if not compiled.exists():
        return False
    try:
        import torch_neuronx  # noqa: F401
    except ImportError:
        return False
    return True


@pytest.mark.skipif(not _neuron_available(),
                    reason="requires Neuron + compiled yolo26n_neuron_fp32.pt")
@pytest.mark.parametrize("image_name", ["bus.jpg", "zidane.jpg"])
def test_neuron_matches_cpu(image_name: str) -> None:
    import torch_neuronx  # noqa: F401
    from ultralytics import YOLO

    image_path = ASSETS_DIR / image_name
    tensor, info, _ = preprocess_image(str(image_path))

    y = YOLO(str(WEIGHTS_PATH))
    cpu = y.model.eval().to(torch.float32).cpu()
    with torch.inference_mode():
        cpu_out = cpu(tensor)
    cpu_raw = cpu_out[0] if isinstance(cpu_out, (list, tuple)) else cpu_out
    cpu_dets = postprocess(cpu_raw, info, conf_thres=0.25)

    neuron = torch.jit.load(str(COMPILED_DIR / "yolo26n_neuron_fp32.pt"))
    with torch.inference_mode():
        neu_raw = neuron(tensor)
    neu_topk = neuron_topk(neu_raw, max_det=300, nc=neu_raw.shape[-1] - 4)
    neu_dets = postprocess(neu_topk, info, conf_thres=0.25)

    # Same count
    assert len(cpu_dets) == len(neu_dets), (
        f"count mismatch cpu={len(cpu_dets)} neuron={len(neu_dets)}"
    )

    # Rank detections by score descending for a stable pairing; both paths
    # use identical top-k logic so the order must align.
    cpu_sorted = sorted(cpu_dets, key=lambda d: -d.score)
    neu_sorted = sorted(neu_dets, key=lambda d: -d.score)
    for c, n in zip(cpu_sorted, neu_sorted):
        assert c.cls == n.cls, f"class mismatch {c} vs {n}"
        assert abs(c.score - n.score) < 1e-3, f"score delta too large: {c} vs {n}"
        assert abs(c.x1 - n.x1) < 1.0 and abs(c.y1 - n.y1) < 1.0, f"box drift: {c} vs {n}"
        assert abs(c.x2 - n.x2) < 1.0 and abs(c.y2 - n.y2) < 1.0, f"box drift: {c} vs {n}"
