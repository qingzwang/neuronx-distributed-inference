# YOLO26-nano on Neuron vs CPU — benchmark

- Model: `yolo26n` (ultralytics end-to-end, NMS-free)
- Input size: 640x640
- Compiled artifact: `/home/ubuntu/qwen-omini-on-trn/neuronx-distributed-inference/contrib/models/yolo26/compiled/yolo26n_neuron_fp32.pt`
- Instance: trn2.48xlarge (single NeuronCore)
- Iters/warmup: 50/5

## Per-image latency (mean over iters)

| image | cpu fwd (ms) | neuron fwd (ms) | speedup | cpu e2e (ms) | neuron e2e (ms) | e2e speedup | cpu dets | neu dets |
|-------|-------------:|----------------:|--------:|-------------:|----------------:|------------:|--------:|--------:|
| bus.jpg | 38.21 | 17.56 | 2.18x | 44.33 | 23.92 | 1.85x | 5 | 5 |
| zidane.jpg | 38.85 | 17.55 | 2.21x | 43.81 | 23.11 | 1.90x | 3 | 3 |

## Per-image p50 latency

| image | cpu fwd p50 (ms) | neuron fwd p50 (ms) | cpu e2e p50 (ms) | neuron e2e p50 (ms) |
|-------|-----------------:|--------------------:|-----------------:|--------------------:|
| bus.jpg | 38.21 | 17.56 | 44.29 | 23.89 |
| zidane.jpg | 38.87 | 17.54 | 43.79 | 23.02 |
