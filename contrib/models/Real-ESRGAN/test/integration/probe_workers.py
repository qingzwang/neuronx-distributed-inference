#!/usr/bin/env python3
"""Probe whether NEURON_RT_NUM_WORKERS is the DataParallel scaling ceiling.

Runs 1K tile sweep through N cores at several NEURON_RT_NUM_WORKERS values,
then prints per-tile ms. Must set the env var BEFORE importing torch_neuronx —
we do that here explicitly.
"""
import os
import sys
import statistics
import subprocess
import time
from pathlib import Path

# Re-exec self with the requested NEURON_RT_NUM_WORKERS value if given as arg[1].
# This is necessary because torch_neuronx caches the value at import time.
SRC_DIR = Path(__file__).resolve().parent.parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

if len(sys.argv) > 1 and sys.argv[1].isdigit():
    workers = sys.argv[1]
    cores = sys.argv[2]
    os.environ["NEURON_RT_NUM_WORKERS"] = workers
    os.environ["NRT_NUM_WORKERS"] = workers

    import torch
    from modeling_real_esrgan import NeuronRealESRGAN
    import torch_neuronx

    n_cores = int(cores)
    neuron = NeuronRealESRGAN(
        model_name="RealESRGAN_x4plus",
        model_path="/home/ubuntu/models/real-esrgan/RealESRGAN_x4plus.pth",
        dtype=torch.bfloat16,
    )
    neuron.load(
        "/home/ubuntu/neuron_models/real-esrgan/bench/RealESRGAN_x4plus/bs1_tile128_bf16"
    )
    dp = torch_neuronx.DataParallel(neuron.traced, device_ids=list(range(n_cores)))

    # 64 tiles (a 1K image)
    n_tiles = 64
    batch = torch.zeros(n_tiles, 3, 128, 128, dtype=torch.bfloat16)

    # warmup
    _ = dp(batch)
    _ = dp(batch)

    times = []
    for _ in range(5):
        t0 = time.perf_counter()
        out = dp(batch)
        _ = out[-1, 0, 0, 0].item()
        times.append((time.perf_counter() - t0) * 1000.0)
    median = statistics.median(times)
    print(
        f"workers={workers} cores={cores} median={median:.1f}ms "
        f"ms/tile={median/n_tiles:.2f}  raw={[f'{t:.0f}' for t in times]}",
        flush=True,
    )
    sys.exit(0)


# Driver — fork a subprocess per config
configs = [
    # (workers, cores)
    (None, 1),   # baseline single core
    (2, 2),
    (2, 4),
    (2, 8),
    (2, 16),
    (8, 4),
    (8, 8),
    (8, 16),
    (16, 8),
    (16, 16),
    (32, 16),
]

for workers, cores in configs:
    if workers is None:
        # Use the default
        env = dict(os.environ)
        env.pop("NEURON_RT_NUM_WORKERS", None)
        env.pop("NRT_NUM_WORKERS", None)
        args_workers = "2"  # default is probably 2 but we don't force it — re-exec ignores
    else:
        args_workers = str(workers)
    proc = subprocess.run(
        [sys.executable, __file__, args_workers, str(cores)],
        capture_output=True,
        text=True,
    )
    print(proc.stdout.strip() or proc.stderr[-500:])
