# YOLO26-nano multi-core throughput

- Compiled artifact: `/home/ubuntu/qwen-omini-on-trn/neuronx-distributed-inference/contrib/models/yolo26/compiled/yolo26n_neuron_fp16.pt`
- Input size: 640x640
- Instance: trn2.48xlarge
- Iters/warmup: 50/5

Each step feeds one image per NeuronCore (batch = num_cores) through `torch_neuronx.DataParallel`. Per-image latency is the wall-clock duration of a step divided by the batch size; throughput is cores / step time.

| cores | batch | step mean (ms) | step p50 (ms) | per-image (ms) | throughput (img/s) |
|------:|------:|---------------:|--------------:|---------------:|-------------------:|
| 1 | 1 | 33.27 | 33.27 | 33.27 | 30.1 |
| 2 | 2 | 33.67 | 33.66 | 16.84 | 59.4 |
| 4 | 4 | 34.67 | 34.68 | 8.67 | 115.4 |
| 8 | 8 | 36.21 | 36.18 | 4.53 | 220.9 |
| 16 | 16 | 42.06 | 42.06 | 2.63 | 380.4 |
| 32 | 32 | 50.98 | 50.69 | 1.59 | 627.7 |

| cores | throughput vs 1 core |
|------:|---------------------:|
| 1 | 1.00x |
| 2 | 1.98x |
| 4 | 3.84x |
| 8 | 7.35x |
| 16 | 12.66x |
| 32 | 20.88x |
