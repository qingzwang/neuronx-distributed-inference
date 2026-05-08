# YOLO26 aligned peak-throughput (LNC=1, DP=8, imgsz=640)

Per-variant config matches the AWS Neuron reference table: dtype and BS/core vary per size.

| variant | params (pre-fuse) | dtype | NEFF (MB) | BS/core | throughput (img/s) | per-image (ms) |
|---------|------------------:|-------|----------:|--------:|-------------------:|---------------:|
| yolo26n | 2.6M | FP32 | 10.2 | 1 | 419 | 2.38 |
| yolo26s | 10.0M | FP32 | 75.7 | 32 | 539 | 1.85 |
| yolo26m | 21.9M | BF16 | 78.9 | 32 | 493 | 2.03 |
| yolo26l | 26.3M | BF16 | 97.4 | 32 | 449 | 2.23 |
| yolo26x | 59.0M | BF16 | 127.5 | 16 | 380 | 2.63 |
