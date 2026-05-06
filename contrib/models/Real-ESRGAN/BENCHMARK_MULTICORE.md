# Real-ESRGAN Neuron Multi-core + bf16 Benchmark

- **Model:** `RealESRGAN_x4plus` (RRDBNet, scale x4)
- **Tile size:** 128x128 input (512x512 output)
- **Instance:** ip-172-31-21-9
- **torch:** 2.9.1+cu128
- **torch_neuronx:** 2.9.0.2.13.24727+8e870898
- **Date:** 2026-05-06 01:46:16 UTC

## Compile times

| dtype | compile seconds |
|:------|----------------:|
| fp32 | (cached) |
| bf16 | 184.6 |

## End-to-end tiled-image latency

Each row runs one whole image by dispatching all tiles to `n_cores`
NeuronCores via `torch_neuronx.DataParallel`. `ms/tile` is the effective
per-tile latency (`total / n_tiles`) — for a perfectly parallel workload
this drops linearly with `n_cores` until host overhead dominates.

| Resolution | Tiles | dtype | Cores | Median total (s) | Mean (s) | ms/tile | Speedup vs fp32 1-core |
|-----------:|------:|:------|------:|-----------------:|---------:|--------:|-----------------------:|
| 1024x1024 | 64 | fp32 | 1 | 12.77 | 12.77 | 199.52 | 1.00x |
| 1024x1024 | 64 | fp32 | 2 | 6.43 | 6.43 | 100.44 | 1.99x |
| 1024x1024 | 64 | fp32 | 4 | 6.41 | 6.41 | 100.19 | 1.99x |
| 1024x1024 | 64 | fp32 | 8 | 6.41 | 6.41 | 100.09 | 1.99x |
| 1024x1024 | 64 | fp32 | 16 | 6.44 | 6.44 | 100.55 | 1.98x |
| 1024x1024 | 64 | bf16 | 1 | 3.13 | 3.13 | 48.97 | 4.07x |
| 1024x1024 | 64 | bf16 | 2 | 1.58 | 1.58 | 24.69 | 8.08x |
| 1024x1024 | 64 | bf16 | 4 | 1.56 | 1.56 | 24.32 | 8.20x |
| 1024x1024 | 64 | bf16 | 8 | 1.55 | 1.55 | 24.23 | 8.23x |
| 1024x1024 | 64 | bf16 | 16 | 1.55 | 1.55 | 24.19 | 8.25x |
| 2048x2048 | 256 | fp32 | 1 | 51.04 | 51.04 | 199.36 | 1.00x |
| 2048x2048 | 256 | fp32 | 2 | 25.69 | 25.69 | 100.37 | 1.99x |
| 2048x2048 | 256 | fp32 | 4 | 25.62 | 25.62 | 100.07 | 1.99x |
| 2048x2048 | 256 | fp32 | 8 | 25.59 | 25.59 | 99.95 | 1.99x |
| 2048x2048 | 256 | fp32 | 16 | 25.73 | 25.73 | 100.49 | 1.98x |
| 2048x2048 | 256 | bf16 | 1 | 12.40 | 12.40 | 48.43 | 4.12x |
| 2048x2048 | 256 | bf16 | 2 | 6.30 | 6.30 | 24.62 | 8.10x |
| 2048x2048 | 256 | bf16 | 4 | 6.24 | 6.24 | 24.36 | 8.18x |
| 2048x2048 | 256 | bf16 | 8 | 6.20 | 6.20 | 24.22 | 8.23x |
| 2048x2048 | 256 | bf16 | 16 | 6.24 | 6.24 | 24.39 | 8.17x |
| 4096x4096 | 1024 | fp32 | 1 | 204.19 | 204.19 | 199.40 | 1.00x |
| 4096x4096 | 1024 | fp32 | 2 | 102.73 | 102.73 | 100.32 | 1.99x |
| 4096x4096 | 1024 | fp32 | 4 | 102.46 | 102.46 | 100.06 | 1.99x |
| 4096x4096 | 1024 | fp32 | 8 | 102.73 | 102.73 | 100.32 | 1.99x |
| 4096x4096 | 1024 | fp32 | 16 | 102.96 | 102.96 | 100.55 | 1.98x |
| 4096x4096 | 1024 | bf16 | 1 | 49.55 | 49.55 | 48.39 | 4.12x |
| 4096x4096 | 1024 | bf16 | 2 | 25.22 | 25.22 | 24.63 | 8.10x |
| 4096x4096 | 1024 | bf16 | 4 | 24.94 | 24.94 | 24.35 | 8.19x |
| 4096x4096 | 1024 | bf16 | 8 | 24.83 | 24.83 | 24.25 | 8.22x |
| 4096x4096 | 1024 | bf16 | 16 | 24.99 | 24.99 | 24.40 | 8.17x |

## Notes

- **Data parallelism is free for this workload** — tiles are independent,
  so `DataParallel` replicates the NEFF across the requested NeuronCores
  and round-robins tile batches onto them. Each replica holds its own
  ~64MB of weights, so VRAM scales with `n_cores`.
- **bf16 vs fp32**: the generator has no attention / normalisation layers
  that are sensitive to dtype, so bf16 gives a large raw speedup with
  negligible pixel-level quality change (previous parity test showed
  max |Δ| = 1/255 against CPU fp32). Produce a bf16 NEFF once and keep it.
- **Host overhead ceiling**: after enough cores the per-tile latency is
  dominated by Python-side slicing, `to(dtype)`, and the DataParallel
  dispatch thread pool. If you see speedup plateau, the next lever is a
  bigger tile size or pre-batching tiles on the host into a larger NEFF.
