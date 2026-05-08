# Real-ESRGAN Pinned Multi-core Benchmark (yolo26 pattern)

- **Model:** `RealESRGAN_x4plus` (RRDBNet, scale x4)
- **Tile:** 128x128, **LNC:** 1, **dtype:** bf16
- **Image:** 1024x1024 (64 tiles)
- **torch_neuronx:** 2.9.0.2.13.24727+8e870898
- **Date:** 2026-05-08 08:54:06 UTC

## Method

Borrowed from `contrib/models/yolo26/src/benchmark_multicore.py`:

1. Load the traced NEFF **once per NeuronCore**, each load wrapped in
   `torch_neuronx.experimental.placement.neuron_cores_context(start_nc=i, nc_count=1)`.
   This pins replica *i* to physical core *i* — otherwise the runtime routes
   every inference through whichever core loaded first.
2. Fan out tile dispatch via `concurrent.futures.ThreadPoolExecutor`, so each
   core gets its own Python thread and the Neuron runtime can overlap them.

## Results: pinned vs. DataParallel

`DataParallel ms/tile` values are copied from `BENCHMARK_LNC1.md`. Lower is better.

| Cores | Pinned ms/tile | DataParallel ms/tile | Pinned speedup vs DP | Pinned vs 1-core pinned |
|------:|---------------:|---------------------:|---------------------:|------------------------:|
| 1 | 48.00 | 48.82 | 1.02x | 1.00x |
| 2 | 24.22 | 25.05 | 1.03x | 1.98x |
| 4 | 12.20 | 24.99 | 2.05x | 3.93x |
| 8 | 6.17 | 24.92 | 4.04x | 7.78x |
| 16 | 3.16 | 24.94 | 7.90x | 15.21x |
| 32 | 1.67 | n/a | n/a | 28.75x |

## Full pinned numbers

| Cores | n_tiles | total median (s) | total mean (s) | ms/tile | throughput (tiles/s) |
|------:|--------:|-----------------:|---------------:|--------:|---------------------:|
| 1 | 64 | 3.07 | 3.07 | 48.00 | 20.8 |
| 2 | 64 | 1.55 | 1.55 | 24.22 | 41.3 |
| 4 | 64 | 0.78 | 0.78 | 12.20 | 82.0 |
| 8 | 64 | 0.39 | 0.39 | 6.17 | 162.1 |
| 16 | 64 | 0.20 | 0.20 | 3.16 | 316.8 |
| 32 | 64 | 0.11 | 0.11 | 1.67 | 598.9 |

## Notes

- Each replica holds its own ~64 MB of weights — memory scales with `cores`.
- The traced NEFF was compiled with `(1, 3, 128, 128)` input at LNC=1 bf16.
- No recompile is needed to change the core count; only the number of
  replicas loaded into memory.
- If a given core count fails with a placement or OOM error, lower `cores`
  or check that other Neuron processes are not using the same cores.
