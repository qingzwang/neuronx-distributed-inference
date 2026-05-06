"""Trace the Qwen3.5 ViT vision encoder onto Neuron.

The contrib PR references this script but does not ship it. This builds
the CPU ViT module (CPUVisionModel), loads HF weights into it, and traces
it with torch_neuronx.trace for one or more sequence-length buckets. The
resulting vision_encoder_{bucket}.pt files are what NeuronQwen35VisionModelWrapper
loads via load_compiled().

Trace is single-core (no TP) -- the vision encoder is ~600M params, fits
in HBM on a single Neuron core, and runs in ~100ms per image on BF16.

Usage:
    python3 examples/compile_vision_encoder.py \\
        --model-path /home/ubuntu/models/Qwen3.5-2B \\
        --out-dir    /home/ubuntu/traced_model/Qwen3.5-2B/vision \\
        --buckets    256 1024 4096
"""

import argparse
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch
import torch_neuronx

_CONTRIB_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _CONTRIB_ROOT not in sys.path:
    sys.path.insert(0, _CONTRIB_ROOT)

from src.modeling_qwen35_vision import CPUVisionModel
from safetensors import safe_open


def build_cpu_model(model_path: str, vision_config: dict) -> torch.nn.Module:
    """Build the CPU reference ViT with HF weights loaded.

    Mirrors NeuronQwen35VisionModelWrapper.load_cpu_model() but returns
    the module directly.
    """
    cfg = SimpleNamespace(**vision_config)
    model = CPUVisionModel(cfg)

    key_map = {}
    for i in range(cfg.depth):
        hf_pre = f"model.visual.blocks.{i}"
        loc_pre = f"blocks.{i}"
        for suffix in [
            "attn.qkv.weight", "attn.qkv.bias",
            "attn.proj.weight", "attn.proj.bias",
            "mlp.linear_fc1.weight", "mlp.linear_fc1.bias",
            "mlp.linear_fc2.weight", "mlp.linear_fc2.bias",
            "norm1.weight", "norm1.bias",
            "norm2.weight", "norm2.bias",
        ]:
            key_map[f"{hf_pre}.{suffix}"] = f"{loc_pre}.{suffix}"

    key_map.update({
        "model.visual.merger.norm.weight":       "merger_norm.weight",
        "model.visual.merger.norm.bias":         "merger_norm.bias",
        "model.visual.merger.linear_fc1.weight": "merger_fc1.weight",
        "model.visual.merger.linear_fc1.bias":   "merger_fc1.bias",
        "model.visual.merger.linear_fc2.weight": "merger_fc2.weight",
        "model.visual.merger.linear_fc2.bias":   "merger_fc2.bias",
    })

    state = model.state_dict()
    loaded = 0
    for sf in sorted(Path(model_path).glob("model*.safetensors")):
        with safe_open(str(sf), framework="pt") as f:
            for k in f.keys():
                local = key_map.get(k)
                if local and local in state:
                    state[local].copy_(f.get_tensor(k))
                    loaded += 1
    model.load_state_dict(state)
    print(f"[weights] loaded {loaded} ViT tensors from HF safetensors")
    return model.to(torch.bfloat16).eval()


def trace_bucket(model: torch.nn.Module, cfg: SimpleNamespace, bucket: int,
                 out_dir: str, workdir_base: str):
    """Trace the vision encoder for a single bucket size."""
    head_dim = cfg.hidden_size // cfg.num_heads
    # CPUVisionModel.forward(hidden_states, attention_mask, cos, sin)
    example = (
        torch.zeros(bucket, cfg.hidden_size, dtype=torch.bfloat16),
        torch.zeros(1, 1, bucket, bucket, dtype=torch.bfloat16),
        torch.zeros(bucket, head_dim, dtype=torch.bfloat16),
        torch.zeros(bucket, head_dim, dtype=torch.bfloat16),
    )

    workdir = os.path.join(workdir_base, f"bucket_{bucket}")
    os.makedirs(workdir, exist_ok=True)

    print(f"[trace] bucket={bucket}  compiling (workdir={workdir})...")
    t0 = time.perf_counter()
    traced = torch_neuronx.trace(
        model,
        example,
        compiler_workdir=workdir,
        compiler_args=[
            "--auto-cast", "none",
            "--enable-saturate-infinity",
            "-O1",
        ],
    )
    elapsed = time.perf_counter() - t0
    print(f"[trace] bucket={bucket} done in {elapsed:.1f}s")

    out_path = os.path.join(out_dir, f"vision_encoder_{bucket}.pt")
    torch.jit.save(traced, out_path)
    sz_mb = os.path.getsize(out_path) / 1024 / 1024
    print(f"[save ] wrote {out_path} ({sz_mb:.1f} MB)")


def main():
    import json

    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default="/home/ubuntu/models/Qwen3.5-2B")
    ap.add_argument("--out-dir",
                    default="/home/ubuntu/traced_model/Qwen3.5-2B/vision")
    ap.add_argument("--buckets", type=int, nargs="+", default=[16])
    args = ap.parse_args()

    with open(os.path.join(args.model_path, "config.json")) as f:
        full_cfg = json.load(f)
    vision_config = full_cfg["vision_config"]
    cfg = SimpleNamespace(**vision_config)

    os.makedirs(args.out_dir, exist_ok=True)
    workdir_base = os.path.join(args.out_dir, "_workdir")

    print(f"[config] depth={cfg.depth}  hidden={cfg.hidden_size}  "
          f"heads={cfg.num_heads}  patch={cfg.patch_size}  "
          f"merge={cfg.spatial_merge_size}  out_hidden={cfg.out_hidden_size}")
    print(f"[config] buckets: {args.buckets}")

    model = build_cpu_model(args.model_path, vision_config)

    for bucket in args.buckets:
        trace_bucket(model, cfg, bucket, args.out_dir, workdir_base)


if __name__ == "__main__":
    main()
