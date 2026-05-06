# YOLO26 size sweep — CPU vs Neuron (trn2)

- Image: `bus.jpg`  (640x640)
- Iters/warmup: 30/5
- Multi-core sweep: cores=[1, 8, 32]

## Single-image latency (model forward only, mean ms)

| variant | weights (MB) | CPU | Neuron fp32 | fp32 speedup | Neuron fp16 | fp16 speedup |
|---------|-------------:|----:|------------:|-------------:|------------:|-------------:|
| yolo26n | 5.3 | 47.98 | 17.52 | 2.74x | 32.39 | 1.48x |
| yolo26s | 19.5 | 54.53 | 13.20 | 4.13x | 16.54 | 3.30x |

## Accuracy (Neuron vs CPU, same image, same conf=0.25)

| variant | cpu dets | fp32 dets | fp32 max Δ | fp32 min IoU | fp16 dets | fp16 max Δ | fp16 min IoU |
|---------|--------:|----------:|-----------:|-------------:|----------:|-----------:|-------------:|
| yolo26n | 5 | 5 | 6.56e-07 | 1.0000 | 5 | 1.65e-04 | 0.9995 |
| yolo26s | 5 | 5 | 1.19e-07 | 1.0000 | 5 | 3.23e-03 | 0.9998 |

## Multi-core throughput (Neuron data-parallel)

| variant | dtype | 1 cores (img/s) | 8 cores (img/s) | 32 cores (img/s) | 32 cores per-img (ms) |
|---|---|---|---|---|---|
| yolo26n | fp32 | 54.5 | 383.4 | 826.8 | 1.21 |
| yolo26s | fp32 | 71.0 | 488.4 | 933.4 | 1.07 |
