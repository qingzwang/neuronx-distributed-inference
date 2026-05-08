"""VERBATIM COPY from jimburtoft/neuronx-distributed-inference contrib/yolo26 branch
(contrib/models/YOLO26/src/modeling_yolo26.py) — do not modify. Used to reproduce
the upstream AWS Neuron YOLO26 throughput benchmark on our trn2 instance.
"""

import os
import time

import torch
import torch.nn as nn

try:
    import torch_neuronx
except ImportError:
    torch_neuronx = None

from ultralytics import YOLO
from ultralytics.nn.modules.block import C2f


VARIANT_DTYPES = {
    "n": torch.float32,
    "s": torch.float32,
    "m": torch.bfloat16,
    "l": torch.bfloat16,
    "x": torch.bfloat16,
}

INPUT_SHAPE = (3, 640, 640)

DEFAULT_COMPILER_ARGS = []

COSINE_SIM_THRESHOLDS = {
    "n": 0.99,
    "s": 0.99,
    "m": 0.98,
    "l": 0.99,
    "x": 0.99,
}


def prepare_yolo26(weight_path: str, dtype: torch.dtype = torch.float32) -> nn.Module:
    model = YOLO(weight_path)
    pytorch_model = model.model.eval()

    detect = pytorch_model.model[-1]
    detect.end2end = False

    pytorch_model = pytorch_model.fuse(verbose=False)

    for m in pytorch_model.modules():
        if hasattr(m, "export"):
            m.export = True
        if hasattr(m, "dynamic"):
            m.dynamic = False
        if hasattr(m, "format"):
            m.format = "torchscript"
        if hasattr(m, "shape"):
            m.shape = None
        if isinstance(m, C2f):
            m.forward = m.forward_split

    if dtype != torch.float32:
        pytorch_model = pytorch_model.to(dtype)

    return pytorch_model


def get_variant_dtype(variant: str) -> torch.dtype:
    return VARIANT_DTYPES[variant]


def compile_yolo26(
    weight_path: str,
    batch_size: int = 1,
    dtype: torch.dtype = None,
    save_path: str = None,
    lnc: int = None,
    compiler_args=None,
) -> torch.jit.ScriptModule:
    if torch_neuronx is None:
        raise RuntimeError("torch_neuronx is not installed. Run on a Neuron instance.")

    variant = _infer_variant(weight_path)
    if dtype is None:
        dtype = get_variant_dtype(variant) if variant else torch.float32

    model = prepare_yolo26(weight_path, dtype=dtype)

    dummy = torch.randn(batch_size, *INPUT_SHAPE, dtype=dtype)
    with torch.no_grad():
        _ = model(dummy)

    args = list(compiler_args) if compiler_args else list(DEFAULT_COMPILER_ARGS)
    if lnc is not None:
        args.extend(["--lnc", str(lnc)])

    traced = torch_neuronx.trace(model, dummy, compiler_args=args)

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        torch.jit.save(traced, save_path)

    return traced


class YOLO26NeuronModel:
    def __init__(
        self,
        variant: str,
        batch_size: int = 1,
        cache_dir: str = "compiled",
        lnc: int = None,
        num_cores: int = 1,
    ):
        self.variant = variant
        self.batch_size = batch_size
        self.dtype = get_variant_dtype(variant)
        self.dtype_name = "bf16" if self.dtype == torch.bfloat16 else "fp32"
        self.num_cores = num_cores

        if lnc is None:
            lnc = int(os.environ.get("NEURON_LOGICAL_NC_CONFIG", "2"))
        self.lnc = lnc

        weight_path = f"yolo26{variant}.pt"
        neff_name = f"yolo26{variant}_{self.dtype_name}_bs{batch_size}_lnc{lnc}.pt"
        save_path = os.path.join(cache_dir, neff_name)

        if os.path.exists(save_path):
            self._model = torch.jit.load(save_path)
        else:
            self._model = compile_yolo26(
                weight_path,
                batch_size=batch_size,
                dtype=self.dtype,
                save_path=save_path,
                lnc=lnc,
            )

        if num_cores > 1 and torch_neuronx is not None:
            self._model = torch_neuronx.DataParallel(
                self._model,
                device_ids=list(range(num_cores)),
                dim=0,
            )

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self._model(x.to(self.dtype))

    def benchmark(self, warmup: int = 10, iterations: int = 50) -> dict:
        import numpy as np

        total_bs = self.batch_size * self.num_cores
        dummy = torch.randn(total_bs, *INPUT_SHAPE, dtype=self.dtype)

        for _ in range(warmup):
            self(dummy)

        latencies = []
        for _ in range(iterations):
            t0 = time.time()
            self(dummy)
            latencies.append((time.time() - t0) * 1000)

        lat = np.array(sorted(latencies))
        p50 = float(np.percentile(lat, 50))
        return {
            "p50_ms": round(p50, 2),
            "p95_ms": round(float(np.percentile(lat, 95)), 2),
            "p99_ms": round(float(np.percentile(lat, 99)), 2),
            "throughput_img_s": round(total_bs / (p50 / 1000), 1),
        }


def validate_accuracy(
    weight_path: str,
    neuron_model,
    dtype: torch.dtype = None,
    seed: int = 42,
) -> dict:
    variant = _infer_variant(weight_path)
    if dtype is None:
        dtype = get_variant_dtype(variant) if variant else torch.float32

    torch.manual_seed(seed)
    dummy = torch.randn(1, *INPUT_SHAPE, dtype=dtype)

    cpu_model = prepare_yolo26(weight_path, dtype=dtype)
    with torch.no_grad():
        cpu_out = cpu_model(dummy)

    if isinstance(neuron_model, YOLO26NeuronModel):
        nrn_out = neuron_model(dummy)
    else:
        with torch.no_grad():
            nrn_out = neuron_model(dummy)

    cpu_flat = cpu_out.flatten().float()
    nrn_flat = nrn_out.flatten().float()

    cossim = torch.nn.functional.cosine_similarity(
        cpu_flat.unsqueeze(0), nrn_flat.unsqueeze(0)
    ).item()

    diff = (cpu_flat - nrn_flat).abs()

    return {
        "cosine_similarity": round(cossim, 6),
        "max_error": round(diff.max().item(), 6),
        "mean_error": round(diff.mean().item(), 6),
        "has_nan": bool(torch.isnan(nrn_out).any().item()),
    }


def _infer_variant(weight_path: str):
    base = os.path.basename(weight_path).lower()
    for v in ("n", "s", "m", "l", "x"):
        if f"yolo26{v}" in base:
            return v
    return None
