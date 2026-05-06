"""Trace YOLO26-nano to a Neuron-compiled TorchScript module.

Ultralytics' NMS-free head ends in a `topk` call over the 8400 anchors. XLA
lowers that to a `sort` op, which neuronx-cc does not support on trn2 — so
the compiled module stops one step short: it returns the raw per-anchor
predictions `(B, num_anchors, 4+nc)` with xyxy boxes and sigmoid class
scores, and the cheap top-k / confidence filter runs on CPU (see
`yolo26_common.neuron_postprocess`).
"""

from __future__ import annotations

import argparse
import time
import types
from pathlib import Path

import torch
import torch.nn as nn
import torch_neuronx
from ultralytics import YOLO

from neuron_patches import patch_attention_modules
from yolo26_common import COMPILED_DIR, DEFAULT_IMGSZ, WEIGHTS_PATH


def _passthrough_postprocess(self, preds: torch.Tensor) -> torch.Tensor:
    # preds is (B, num_anchors, 4 + nc) with xyxy boxes and sigmoid class scores.
    return preds


class YOLO26Wrapper(nn.Module):
    """Return the pre-topk per-anchor tensor from YOLO26's detection head."""

    def __init__(self, inner: nn.Module) -> None:
        super().__init__()
        self.inner = inner

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.inner(x)
        # With postprocess neutralised, the detect head returns (preds, aux_dict).
        if isinstance(out, (list, tuple)):
            return out[0]
        return out


def _patch_detect_head(model: nn.Module) -> None:
    """Replace the head's topk postprocess with a passthrough, enable export mode."""
    detect = model.model[-1]
    # The YOLO26 end-to-end head is enabled by default; keep it on so we get
    # the one2one predictions, but skip the trailing topk.
    detect.postprocess = types.MethodType(_passthrough_postprocess, detect)
    # `export=True` makes the head return just the decoded tensor instead of
    # `(y, aux_dict)`, so the traced graph has a single tensor output.
    detect.export = True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default=str(WEIGHTS_PATH))
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ)
    parser.add_argument("--out", default=str(COMPILED_DIR / "yolo26n_neuron.pt"))
    parser.add_argument(
        "--dtype",
        choices=["fp32", "bf16", "fp16"],
        default="fp32",
        help="auto-cast dtype during Neuron compilation",
    )
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[trace] loading {args.weights}")
    yolo = YOLO(args.weights)
    model = yolo.model.eval().to(torch.float32).cpu()
    # Fold BN into conv — smaller graph, faster compile and execution.
    model = model.fuse()
    _patch_detect_head(model)
    n_attn = patch_attention_modules(model)
    print(f"[trace] patched {n_attn} attention modules to avoid torch.split on Neuron")
    wrapper = YOLO26Wrapper(model).eval()

    example = torch.zeros(1, 3, args.imgsz, args.imgsz, dtype=torch.float32)

    with torch.inference_mode():
        ref = wrapper(example)
    print(f"[trace] wrapper output shape={tuple(ref.shape)} dtype={ref.dtype}")

    # --logical-nc-config=1 uses a single neuron core per LNC (trn2 default is 2, which shards matmul).
    # --enable-saturate-infinity avoids the attention softmax producing NaNs when q@k overflows.
    common = ["--target=trn2", "--logical-nc-config=1", "--enable-saturate-infinity"]
    if args.dtype == "bf16":
        compiler_args = common + ["--auto-cast=matmult", "--auto-cast-type=bf16"]
    elif args.dtype == "fp16":
        compiler_args = common + ["--auto-cast=matmult", "--auto-cast-type=fp16"]
    else:
        compiler_args = common + ["--auto-cast=none", "--optlevel=2"]
    print(f"[trace] compiling with args={compiler_args}")
    t0 = time.perf_counter()
    neuron_model = torch_neuronx.trace(
        wrapper,
        example,
        compiler_args=compiler_args,
    )
    compile_secs = time.perf_counter() - t0
    print(f"[trace] compile took {compile_secs:.1f}s")

    torch.jit.save(neuron_model, str(out_path))
    print(f"[trace] saved compiled module to {out_path}")


if __name__ == "__main__":
    main()
