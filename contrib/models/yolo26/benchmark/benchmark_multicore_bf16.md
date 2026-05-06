# YOLO26-nano multi-core throughput

- Compiled artifact: `/home/ubuntu/qwen-omini-on-trn/neuronx-distributed-inference/contrib/models/yolo26/compiled/yolo26n_neuron_bf16.pt`
- Input size: 640x640
- Instance: trn2.48xlarge
- Iters/warmup: 50/5

Each step feeds one image per NeuronCore (batch = num_cores) through `torch_neuronx.DataParallel`. Per-image latency is the wall-clock duration of a step divided by the batch size; throughput is cores / step time.

| cores | batch | step mean (ms) | step p50 (ms) | per-image (ms) | throughput (img/s) |
|------:|------:|---------------:|--------------:|---------------:|-------------------:|
| 1 | 1 | 33.25 | 33.25 | 33.25 | 30.1 |
| 2 | 2 | 33.75 | 33.70 | 16.87 | 59.3 |
| 4 | 4 | 34.62 | 34.64 | 8.65 | 115.6 |
| 8 | 8 | 35.56 | 35.45 | 4.45 | 225.0 |
| 16 | 16 | 42.72 | 42.70 | 2.67 | 374.5 |
| 32 | 32 | 49.44 | 49.29 | 1.55 | 647.2 |

| cores | throughput vs 1 core |
|------:|---------------------:|
| 1 | 1.00x |
| 2 | 1.97x |
| 4 | 3.84x |
| 8 | 7.48x |
| 16 | 12.45x |
| 32 | 21.52x |
