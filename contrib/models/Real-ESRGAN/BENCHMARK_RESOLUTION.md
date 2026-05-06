# Real-ESRGAN Input-Resolution Latency: Neuron vs CPU

- **Model:** `RealESRGAN_x4plus` (RRDBNet, scale x4)
- **Tile size (fixed NEFF input):** 128x128
- **Input resolutions tested:** 1024x1024, 2048x2048, 4096x4096
- **Input dtype:** fp32, synthetic random pixels
- **torch:** 2.9.1+cu128
- **torch_neuronx:** 2.9.0.2.13.24727+8e870898
- **CPU:** Intel(R) Xeon(R) Platinum 8488C x 96 threads used by PyTorch
- **Instance:** ip-172-31-21-9
- **Date:** 2026-05-05 16:31:55 UTC

## Methodology

Both Neuron and CPU run the **same tiling pipeline**: the input image is split into
non-overlapping 128x128 tiles and each tile is super-resolved x4 by
the RRDBNet generator. The outputs are stitched back into a single image of size
`resolution * 4` on each side. The only difference between the two
runs is where the tile's forward pass happens:

- **Neuron**: traced NEFF via `torch_neuronx.trace` (one Trainium / Inferentia core)
- **CPU**: the same `nn.Module` in eager PyTorch on the host CPU, fp32

Warmup runs are executed before timing. Per-tile latencies are measured inside
`time.perf_counter`, per-resolution totals wrap the whole tiled sweep. Reported
numbers are the **median** across iterations to filter outliers.

## End-to-end latency (whole image)

Neuron is measured over a full sweep (every tile actually runs). CPU whole-image
latency is **extrapolated** as `n_tiles * CPU-per-tile-median` because running 1024
tiles on CPU for the 4K row would take >10 minutes per iteration.

| Resolution | Tiles | Neuron median (s) | CPU extrapolated (s) | Speedup (CPU / Neuron) |
|-----------:|------:|------------------:|---------------------:|-----------------------:|
| 1024x1024 | 64 | 12.81 | 22.08 | 1.7x |
| 2048x2048 | 256 | 51.23 | 103.97 | 2.0x |
| 4096x4096 | 1024 | 204.85 | 417.91 | 2.0x |

## Per-tile latency (measured)

| Resolution | Neuron per-tile median (ms) | CPU per-tile median (ms) | Per-tile speedup |
|-----------:|----------------------------:|-------------------------:|-----------------:|
| 1024x1024 | 199.49 | 344.95 | 1.7x |
| 2048x2048 | 199.47 | 406.13 | 2.0x |
| 4096x4096 | 199.42 | 408.12 | 2.0x |

## Neuron whole-image latency distribution (ms)

| Resolution | Iters | Mean | Median | Min | Max |
|-----------:|------:|-----:|-------:|----:|----:|
| 1024x1024 | 2 | 12812.6 | 12812.6 | 12804.2 | 12821.0 |
| 2048x2048 | 2 | 51233.3 | 51233.3 | 51199.8 | 51266.7 |
| 4096x4096 | 2 | 204850.7 | 204850.7 | 204797.6 | 204903.8 |

## CPU per-tile sample

| Resolution | Sampled tiles | Sample wall time (s) | Per-tile mean (ms) | Per-tile median (ms) | Per-tile min (ms) | Per-tile max (ms) |
|-----------:|--------------:|---------------------:|-------------------:|---------------------:|------------------:|------------------:|
| 1024x1024 | 8 | 3.2 | 399.4 | 345.0 | 337.6 | 764.7 |
| 2048x2048 | 8 | 3.2 | 399.6 | 406.1 | 377.0 | 411.2 |
| 4096x4096 | 8 | 3.6 | 448.3 | 408.1 | 403.9 | 731.6 |

## Notes

- Tiles are non-overlapping — production pipelines should use the overlapping
  tile loop from the upstream `RealESRGANer.tile_process` to avoid seam artifacts.
  Latency per tile is unchanged; only the tile count increases by the overlap factor.
- Neuron latency per tile should match the bs=1 number from `BENCHMARK.md`; any
  per-tile overhead above that comes from host-side slicing and the `out[...]` copy.
- CPU is single-process eager PyTorch (fp32) with the default thread pool. Whole-image
  CPU latency for 1K/2K/4K inputs is extrapolated from a per-tile sample; the first
  few tiles include one-time torch overheads and are kept in the sample on purpose.
