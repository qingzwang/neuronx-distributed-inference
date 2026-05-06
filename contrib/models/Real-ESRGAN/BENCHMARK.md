# Real-ESRGAN Neuron Latency Benchmark

- **Model:** `RealESRGAN_x4plus` (scale x4)
- **Tile (input) size:** 128x128
- **Output size per image:** 512x512
- **dtype:** fp32 weights / fp32 activations (as traced)
- **Warmup runs:** 5
- **Timed iters:** 30
- **torch_neuronx:** 2.9.0.2.13.24727+8e870898
- **torch:** 2.9.1+cu128
- **instance:** ip-172-31-21-9
- **Benchmark date:** 2026-05-05 15:40:09 UTC

## Per-batch-size latency

| Batch | Compile (s) | Mean (ms) | Median (ms) | p90 (ms) | p99 (ms) | Min (ms) | Max (ms) | Stdev (ms) | ms/image | Throughput (img/s) |
|------:|------------:|----------:|------------:|---------:|---------:|---------:|---------:|-----------:|---------:|-------------------:|
| 1 | 454.0 | 199.29 | 199.25 | 199.40 | 199.66 | 199.18 | 199.70 | 0.11 | 199.29 | 5.02 |
| 2 | 680.4 | 240.57 | 240.56 | 241.03 | 241.32 | 240.06 | 241.35 | 0.31 | 120.29 | 8.31 |
| 4 | 1057.5 | 483.25 | 483.25 | 483.52 | 483.94 | 482.55 | 484.02 | 0.28 | 120.81 | 8.28 |

## Notes

- Real-ESRGAN's RRDBNet is a pure convolutional generator, so batch-size support comes
  "for free" from `torch_neuronx.trace` — we simply trace with the desired batch dim
  and the compiler folds the batch into the NEFF. Each batch size produces a distinct
  NEFF artifact and must be compiled separately.
- Latencies are wall-clock around the `__call__` into the traced NEFF, including
  any CPU-side copies. Warmup runs are discarded.
- Throughput = `batch_size * 1000 / mean_ms`. On a convolutional network, you expect
  per-image latency to be roughly flat across batch sizes once the compute fills the
  device — crossover below that point is where batching pays off.
- Compile time scales with batch size because the traced graph grows; NEFFs are
  cached under `compiled_dir/<model>/bs<N>_tile<T>/`, so subsequent runs reuse them.
