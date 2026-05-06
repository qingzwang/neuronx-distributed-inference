# YOLO26 size sweep — CPU vs Neuron (trn2)

- Image: `bus.jpg`  (480x480)
- Iters/warmup: 30/5
- Multi-core sweep: cores=[1, 8, 32]

## Single-image latency (model forward only, mean ms)

| variant | weights (MB) | CPU | Neuron fp32 | fp32 speedup | Neuron fp16 | fp16 speedup |
|---------|-------------:|----:|------------:|-------------:|------------:|-------------:|
| yolo26n | 5.3 | 38.66 | 9.97 | 3.88x | 18.44 | 2.10x |
| yolo26s | 19.5 | 46.72 | 7.56 | 6.18x | 8.98 | 5.20x |
| yolo26m | 42.2 | 65.45 | 9.12 | 7.18x | 6.93 | 9.44x |
| yolo26l | 50.7 | 89.69 | 12.20 | 7.35x | 8.25 | 10.87x |
| yolo26x | 113.2 | 108.55 | 16.62 | 6.53x | 9.73 | 11.15x |

## Accuracy (Neuron vs CPU, same image, same conf=0.25)

| variant | cpu dets | fp32 dets | fp32 max Δ | fp32 min IoU | fp16 dets | fp16 max Δ | fp16 min IoU |
|---------|--------:|----------:|-----------:|-------------:|----------:|-----------:|-------------:|
| yolo26n | 6 | 6 | 6.02e-06 | 1.0000 | 6 | 4.60e-03 | 0.9995 |
| yolo26s | 5 | 5 | 4.17e-07 | 1.0000 | 5 | 3.90e-04 | 0.9996 |
| yolo26m | 5 | 5 | 7.15e-07 | 1.0000 | 5 | 3.71e-04 | 0.9997 |
| yolo26l | 6 | 6 | 5.34e-05 | 1.0000 | 5 | 1.37e-02 | 0.9997 |
| yolo26x | 5 | 5 | 4.17e-07 | 1.0000 | 5 | 7.44e-05 | 0.9998 |

## Multi-core throughput (Neuron data-parallel)

| variant | dtype | 1 cores (img/s) | 8 cores (img/s) | 32 cores (img/s) | 32 cores per-img (ms) |
|---|---|---|---|---|---|
| yolo26n | fp32 | 93.5 | 651.7 | 1344.6 | 0.74 |
| yolo26s | fp32 | 124.7 | 786.8 | 1857.9 | 0.54 |
| yolo26m | fp32 | 101.3 | 667.0 | 1650.4 | 0.61 |
| yolo26l | fp32 | 78.0 | 514.5 | 1323.3 | 0.76 |
| yolo26x | fp32 | 58.1 | 406.0 | 1066.2 | 0.94 |
