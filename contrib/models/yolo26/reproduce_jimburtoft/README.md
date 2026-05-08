# reproduce_jimburtoft/

Self-contained reproduction of the upstream AWS Neuron YOLO26 benchmark.

## Source

- `modeling_yolo26.py` — copied **verbatim** from
  [`jimburtoft/neuronx-distributed-inference`](https://github.com/jimburtoft/neuronx-distributed-inference/tree/contrib/yolo26/contrib/models/YOLO26),
  branch `contrib/yolo26`, path `contrib/models/YOLO26/src/modeling_yolo26.py`.
- `yolo26_neuron_notebook.ipynb` — fetched unchanged from the same path.

## Runs

- `run_bench.py` — script form of the notebook's DP=8 peak-throughput cells.
  Wraps `YOLO26NeuronModel.benchmark()` for each (variant, BS/core) pair.
- `reproduce_jimburtoft.json` — results from running the script verbatim on
  `trn2.48xlarge`, `NEURON_LOGICAL_NC_CONFIG=1`:

  | variant | BS/core | my run | reference | ratio |
  |---------|--------:|-------:|----------:|------:|
  | n | 1  |  67.7 img/s |   272 img/s | 0.25× |
  | s | 32 | 226.6 img/s | 1 523 img/s | 0.15× |
  | m | 32 | 279.2 img/s | 1 267 img/s | 0.22× |
  | l | 32 | 244.3 img/s | 1 093 img/s | 0.22× |
  | x | 16 | 201.6 img/s |   876 img/s | 0.23× |

- `dispatch_ab_test.json` — same NEFF, same batch, three dispatch paths.
  Shows that bumping `DataParallel.num_workers` from its default `2` to
  `2 × DP = 16` already takes s BS=32 from 229 → 562 img/s (2.45× speedup),
  and explicit per-core placement adds another +18%.

## How to re-run

```bash
cd reproduce_jimburtoft
source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
# Copy or symlink weights next to this README:
ln -sf ../yolo26{n,s,m,l,x}.pt .
env NEURON_LOGICAL_NC_CONFIG=1 python run_bench.py

# Or execute the notebook end-to-end:
jupyter nbconvert --to notebook --execute yolo26_neuron_notebook.ipynb \
  --output yolo26_neuron_notebook_executed.ipynb \
  --ExecutePreprocessor.timeout=3600
```

## Conclusion

- **Code is identical** to upstream (byte-for-byte copy).
- **Accuracy matches** (CosSim ≥ 0.988 across all 5 variants).
- **SDK is one point release newer** than upstream (2.29.1 vs 2.28/2.29).
- **Throughput is 4–7× below upstream.**

The reference was validated on `trn2.3xlarge` (single Trainium2 chip,
8 cores on-die). Our box is `trn2.48xlarge` where DP=8 spans two chips
and pays PCIe/cross-chip latency on every step. Our own top-level
benchmarks (`../benchmark_aligned.py`, `../benchmark_multicore.py`) claw
back most of the gap by **explicit per-core NEFF placement + a wide
thread pool**, but even that plateaus at 40–50% of the upstream number,
which is consistent with the trn2.48xlarge multi-chip penalty.
