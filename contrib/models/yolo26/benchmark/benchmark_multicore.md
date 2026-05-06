# YOLO26-nano multi-core throughput

- Compiled artifact: `/home/ubuntu/qwen-omini-on-trn/neuronx-distributed-inference/contrib/models/yolo26/compiled/yolo26n_neuron_fp32.pt`
- Input size: 640x640
- Instance: trn2.48xlarge
- Iters/warmup: 50/5

Each step feeds one image per NeuronCore (batch = num_cores) through `torch_neuronx.DataParallel`. Per-image latency is the wall-clock duration of a step divided by the batch size; throughput is cores / step time.

| cores | batch | step mean (ms) | step p50 (ms) | per-image (ms) | throughput (img/s) |
|------:|------:|---------------:|--------------:|---------------:|-------------------:|
| 1 | 1 | 18.38 | 18.36 | 18.38 | 54.4 |
| 2 | 2 | 19.07 | 19.03 | 9.54 | 104.9 |
| 4 | 4 | 19.78 | 19.75 | 4.95 | 202.2 |
| 8 | 8 | 21.50 | 21.50 | 2.69 | 372.1 |
| 16 | 16 | 27.89 | 27.80 | 1.74 | 573.6 |
| 32 | 32 | 35.97 | 36.03 | 1.12 | 889.6 |

| cores | throughput vs 1 core |
|------:|---------------------:|
| 1 | 1.00x |
| 2 | 1.93x |
| 4 | 3.72x |
| 8 | 6.84x |
| 16 | 10.55x |
| 32 | 16.35x |
