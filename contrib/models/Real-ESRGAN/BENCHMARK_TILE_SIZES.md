# Real-ESRGAN Tile-Shape Benchmark (LNC=1, bf16)

- **Model:** `RealESRGAN_x4plus` (RRDBNet, scale x4)
- **Image:** synthetic 1024x1024 input
- **LNC:** 1 (each physical NeuronCore is its own logical core)
- **dtype:** bf16
- **torch_neuronx:** 2.9.0.2.13.24727+8e870898
- **Date:** 2026-05-08 08:40:38 UTC

## Compile times

| Tile | compile seconds |
|:-----|----------------:|
| 128x128 | (cached) |
| 256x256 | 587.0 |
| 128x256 | 273.7 |
| 128x512 | 485.4 |

## Summary: best config per tile shape

`MPx/s` is the sustained input megapixels throughput; higher is better. `ms/megapixel` normalises per-tile latency by tile area — if the DataParallel ceiling were fixed dispatch overhead, bigger tiles would have lower ms/megapixel.

| Tile | Pixels | Tiles/image | Best cores | Best total (s) | ms/tile | ms/megapixel | MPx/s |
|:-----|-------:|------------:|-----------:|---------------:|--------:|-------------:|------:|
| 128x128 | 16384 | 64 | 8 | 1.55 | 24.26 | 1480.49 | 0.68 |
| 128x256 | 32768 | 32 | 8 | 1.62 | 50.73 | 1548.13 | 0.65 |
| 256x256 | 65536 | 16 | 8 | 2.06 | 128.87 | 1966.45 | 0.51 |
| 128x512 | 65536 | 16 | 8 | 1.57 | 98.32 | 1500.28 | 0.67 |

## Full sweep

| Tile | Tiles | Cores | Total median (s) | ms/tile | ms/megapixel | MPx/s |
|:-----|------:|------:|-----------------:|--------:|-------------:|------:|
| 128x128 | 64 | 1 | 3.12 | 48.71 | 2972.80 | 0.34 |
| 128x128 | 64 | 2 | 1.57 | 24.60 | 1501.66 | 0.67 |
| 128x128 | 64 | 4 | 1.56 | 24.32 | 1484.20 | 0.67 |
| 128x128 | 64 | 8 | 1.55 | 24.26 | 1480.49 | 0.68 |
| 128x128 | 64 | 16 | 1.56 | 24.34 | 1485.87 | 0.67 |
| 256x256 | 16 | 1 | 4.12 | 257.24 | 3925.13 | 0.25 |
| 256x256 | 16 | 2 | 2.07 | 129.36 | 1973.95 | 0.51 |
| 256x256 | 16 | 4 | 2.07 | 129.16 | 1970.87 | 0.51 |
| 256x256 | 16 | 8 | 2.06 | 128.87 | 1966.45 | 0.51 |
| 256x256 | 16 | 16 | 2.07 | 129.61 | 1977.67 | 0.51 |
| 128x256 | 32 | 1 | 3.25 | 101.57 | 3099.72 | 0.32 |
| 128x256 | 32 | 2 | 1.64 | 51.22 | 1563.13 | 0.64 |
| 128x256 | 32 | 4 | 1.63 | 50.87 | 1552.58 | 0.64 |
| 128x256 | 32 | 8 | 1.62 | 50.73 | 1548.13 | 0.65 |
| 128x256 | 32 | 16 | 1.62 | 50.74 | 1548.40 | 0.65 |
| 128x512 | 16 | 1 | 3.14 | 196.24 | 2994.35 | 0.33 |
| 128x512 | 16 | 2 | 1.58 | 98.62 | 1504.85 | 0.66 |
| 128x512 | 16 | 4 | 1.58 | 98.49 | 1502.81 | 0.67 |
| 128x512 | 16 | 8 | 1.57 | 98.32 | 1500.28 | 0.67 |
| 128x512 | 16 | 16 | 1.58 | 98.82 | 1507.89 | 0.66 |

## Notes

- If `ms/megapixel` is roughly constant across tile shapes, host dispatch
  overhead is negligible and the ceiling is device compute — in that case
  larger tiles give the same MPx/s.
- If `ms/megapixel` **drops** with bigger tiles, there's fixed per-dispatch
  overhead being amortised — the bigger the tile, the fewer dispatches per image,
  so `MPx/s` rises.
- If `ms/megapixel` **rises** with bigger tiles (worst case), device
  utilisation drops with tile size — unlikely for conv nets, but possible if
  HBM bandwidth saturates or intermediate feature maps exceed some threshold.
