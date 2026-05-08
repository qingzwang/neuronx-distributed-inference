# Real-ESRGAN LNC=1 vs LNC=2 Benchmark

- **Model:** `RealESRGAN_x4plus` (RRDBNet, scale x4)
- **Tile size:** 128x128
- **Instance:** trn2.48xlarge
- **torch_neuronx:** 2.9.0.2.13.24727+8e870898
- **Date:** 2026-05-08 08:01:57 UTC

## Compile times (LNC=1)

| dtype | compile seconds |
|:------|----------------:|
| bf16 | 168.4 |

## Per-tile latency: LNC=1 vs LNC=2

LNC=2 baseline numbers are copied from BENCHMARK_MULTICORE.md. Lower ms/tile is better; speedup = LNC2 / LNC1.

| Resolution | dtype | Cores | LNC=1 ms/tile | LNC=2 ms/tile | Speedup (LNC=2 / LNC=1) |
|-----------:|:------|------:|--------------:|--------------:|------------------------:|
| 1024x1024 | bf16 | 1 | 48.34 | 48.97 | 1.01x |
| 1024x1024 | bf16 | 2 | 24.52 | 24.69 | 1.01x |
| 1024x1024 | bf16 | 4 | 24.31 | 24.32 | 1.00x |
| 1024x1024 | bf16 | 8 | 24.21 | 24.23 | 1.00x |
| 1024x1024 | bf16 | 16 | 24.30 | 24.19 | 1.00x |
| 1024x1024 | bf16 | 32 | 24.40 | n/a | n/a |
| 2048x2048 | bf16 | 1 | 48.40 | 48.43 | 1.00x |
| 2048x2048 | bf16 | 2 | 24.61 | 24.62 | 1.00x |
| 2048x2048 | bf16 | 4 | 24.35 | 24.36 | 1.00x |
| 2048x2048 | bf16 | 8 | 24.17 | 24.22 | 1.00x |
| 2048x2048 | bf16 | 16 | 24.29 | 24.39 | 1.00x |
| 2048x2048 | bf16 | 32 | 24.42 | n/a | n/a |
| 4096x4096 | bf16 | 1 | 48.47 | 48.39 | 1.00x |
| 4096x4096 | bf16 | 2 | 24.60 | 24.63 | 1.00x |
| 4096x4096 | bf16 | 4 | 24.31 | 24.35 | 1.00x |
| 4096x4096 | bf16 | 8 | 24.22 | 24.25 | 1.00x |
| 4096x4096 | bf16 | 16 | 24.20 | 24.40 | 1.01x |
| 4096x4096 | bf16 | 32 | 24.49 | n/a | n/a |

## Full LNC=1 whole-image latency

| Resolution | dtype | Cores | Median (s) | Mean (s) |
|-----------:|:------|------:|-----------:|---------:|
| 1024x1024 | bf16 | 1 | 3.09 | 3.09 |
| 1024x1024 | bf16 | 2 | 1.57 | 1.57 |
| 1024x1024 | bf16 | 4 | 1.56 | 1.56 |
| 1024x1024 | bf16 | 8 | 1.55 | 1.55 |
| 1024x1024 | bf16 | 16 | 1.56 | 1.56 |
| 1024x1024 | bf16 | 32 | 1.56 | 1.56 |
| 2048x2048 | bf16 | 1 | 12.39 | 12.39 |
| 2048x2048 | bf16 | 2 | 6.30 | 6.30 |
| 2048x2048 | bf16 | 4 | 6.23 | 6.23 |
| 2048x2048 | bf16 | 8 | 6.19 | 6.19 |
| 2048x2048 | bf16 | 16 | 6.22 | 6.22 |
| 2048x2048 | bf16 | 32 | 6.25 | 6.25 |
| 4096x4096 | bf16 | 1 | 49.63 | 49.63 |
| 4096x4096 | bf16 | 2 | 25.19 | 25.19 |
| 4096x4096 | bf16 | 4 | 24.90 | 24.90 |
| 4096x4096 | bf16 | 8 | 24.80 | 24.80 |
| 4096x4096 | bf16 | 16 | 24.78 | 24.78 |
| 4096x4096 | bf16 | 32 | 25.07 | 25.07 |

## Notes

- LNC=1 gives twice as many logical NeuronCores but each has half the compute
  of an LNC=2 logical core. For a pure conv network like Real-ESRGAN whose per-core
  utilisation is already high, we expect LNC=1 per-tile latency to roughly double
  at the same `cores` number (because one LNC=1 core is half the hardware).
- LNC=1 is interesting when you want to run more DataParallel replicas to get past
  the host-side dispatch ceiling — but the host bottleneck, not the core count, is
  what limits multi-core scaling on this model (see the probe_workers test).
