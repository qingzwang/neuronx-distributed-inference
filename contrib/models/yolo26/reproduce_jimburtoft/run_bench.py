"""Run jimburtoft's YOLO26NeuronModel.benchmark() for each variant at the
same config documented in their README: LNC=1, DP=8, with per-variant
BS/core. The weights must be in the project root (yolo26{n,s,m,l,x}.pt).

Expected (from upstream README, trn2.3xlarge SDK 2.28/2.29):
  n  fp32 BS=1   272  img/s
  s  fp32 BS=32  1523 img/s
  m  bf16 BS=32  1267 img/s
  l  bf16 BS=32  1093 img/s
  x  bf16 BS=16   876 img/s
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

# Make the verbatim jimburtoft module importable
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

# Need to chdir so their class can find yolo26*.pt (it uses a relative path)
os.chdir(HERE.parent)  # project dir that holds yolo26n.pt, yolo26s.pt, ...
CACHE_DIR = HERE / "jb_cache"
CACHE_DIR.mkdir(exist_ok=True)

from modeling_yolo26 import YOLO26NeuronModel  # noqa: E402


CONFIG = [
    # (variant, batch_per_core, expected img/s on ref trn2.3xlarge)
    ("n", 1, 272),
    ("s", 32, 1523),
    ("m", 32, 1267),
    ("l", 32, 1093),
    ("x", 16, 876),
]


def main() -> None:
    dp = 8
    results = []
    for variant, bs, expected in CONFIG:
        print(f"\n=== yolo26{variant} BS/core={bs} DP={dp} LNC=1 ===", flush=True)
        t0 = time.perf_counter()
        model = YOLO26NeuronModel(
            variant=variant,
            batch_size=bs,
            cache_dir=str(CACHE_DIR),
            lnc=1,
            num_cores=dp,
        )
        setup_s = time.perf_counter() - t0
        print(f"  setup (compile + load) took {setup_s:.1f}s", flush=True)

        bench = model.benchmark(warmup=10, iterations=50)
        thr = bench["throughput_img_s"]
        print(
            f"  p50={bench['p50_ms']} ms  p95={bench['p95_ms']} ms  "
            f"p99={bench['p99_ms']} ms  throughput={thr} img/s  "
            f"(reference {expected} img/s, ratio {thr/expected:.2f}x)",
            flush=True,
        )
        results.append({
            "variant": variant,
            "batch_per_core": bs,
            "dp": dp,
            "lnc": 1,
            "expected_img_s": expected,
            **bench,
            "ratio_vs_reference": thr / expected,
            "setup_seconds": setup_s,
        })

    out = HERE / "reproduce_jimburtoft.json"
    out.write_text(json.dumps({"results": results}, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
