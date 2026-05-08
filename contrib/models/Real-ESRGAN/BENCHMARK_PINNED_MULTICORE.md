# Real-ESRGAN Pinned Multi-core Benchmark (yolo26 pattern)

- **Model:** `RealESRGAN_x4plus` (RRDBNet, scale x4)
- **Tile:** 128x128, **LNC:** 1, **dtype:** bf16
- **Image:** 1024x1024 (64 tiles)
- **torch_neuronx:** 2.9.0.2.13.24727+8e870898
- **Date:** 2026-05-08 08:54:06 UTC

## Method

Borrowed from `contrib/models/yolo26/src/benchmark_multicore.py`:

1. Load the traced NEFF **once per NeuronCore**, each load wrapped in
   `torch_neuronx.experimental.placement.neuron_cores_context(start_nc=i, nc_count=1)`.
   This pins replica *i* to physical core *i* — otherwise the runtime routes
   every inference through whichever core loaded first.
2. Fan out tile dispatch via `concurrent.futures.ThreadPoolExecutor`, so each
   core gets its own Python thread and the Neuron runtime can overlap them.

## Results: pinned vs. DataParallel

`DataParallel ms/tile` values are copied from `BENCHMARK_LNC1.md`. Lower is better.

| Cores | Pinned ms/tile | DataParallel ms/tile | Pinned speedup vs DP | Pinned vs 1-core pinned |
|------:|---------------:|---------------------:|---------------------:|------------------------:|
| 1 | 48.00 | 48.82 | 1.02x | 1.00x |
| 2 | 24.22 | 25.05 | 1.03x | 1.98x |
| 4 | 12.20 | 24.99 | 2.05x | 3.93x |
| 8 | 6.17 | 24.92 | 4.04x | 7.78x |
| 16 | 3.16 | 24.94 | 7.90x | 15.21x |
| 32 | 1.67 | n/a | n/a | 28.75x |

## Full pinned numbers

| Cores | n_tiles | total median (s) | total mean (s) | ms/tile | throughput (tiles/s) |
|------:|--------:|-----------------:|---------------:|--------:|---------------------:|
| 1 | 64 | 3.07 | 3.07 | 48.00 | 20.8 |
| 2 | 64 | 1.55 | 1.55 | 24.22 | 41.3 |
| 4 | 64 | 0.78 | 0.78 | 12.20 | 82.0 |
| 8 | 64 | 0.39 | 0.39 | 6.17 | 162.1 |
| 16 | 64 | 0.20 | 0.20 | 3.16 | 316.8 |
| 32 | 64 | 0.11 | 0.11 | 1.67 | 598.9 |

## Notes

- Each replica holds its own ~64 MB of weights — memory scales with `cores`.
- The traced NEFF was compiled with `(1, 3, 128, 128)` input at LNC=1 bf16.
- No recompile is needed to change the core count; only the number of
  replicas loaded into memory.
- If a given core count fails with a placement or OOM error, lower `cores`
  or check that other Neuron processes are not using the same cores.

## Why does pinning unlock ~15x more throughput than DataParallel?

Short answer: `torch_neuronx.DataParallel` does **not** actually duplicate the
NEFF across multiple NeuronCores on this runtime version — pinned replicas do.

### What DataParallel is really doing

When you do:

```python
traced = torch.jit.load("model_neuron.pt")
dp = torch_neuronx.DataParallel(traced, device_ids=[0, 1, ..., 31])
```

1. `traced` wraps a C++ `__torch__.torch.classes.neuron.Model` object whose
   NEFF is already loaded onto **one** physical NeuronCore (the core the runtime
   picked at `torch.jit.load` time — typically core 0).
2. `DataParallel` is a Python wrapper around *that one* object. `device_ids=[...]`
   does not cause the runtime to load additional copies of the NEFF onto the
   listed cores. Every `dp(x)` call ultimately runs on the single core the NEFF
   originally landed on.

So "32-core DataParallel" is really **32 Python threads fighting for one core**.

### Why DataParallel still shows 2x

The Neuron runtime does hardware-level double-buffering: while one inference is
computing on a core, the next request can overlap its host→device DMA. That lets
the same core carry ~2 in-flight requests without serialising. Adding a third
request has nowhere to go, so:

- 1 call in flight → 48 ms/tile (bf16, tile=128)
- 2 calls in flight → 24 ms/tile (the exact 2x we measured)
- 3+ calls in flight → still 24 ms/tile (hard ceiling)

That's exactly the DataParallel plateau from `BENCHMARK_LNC1.md`.

### What pinning does differently

```python
from torch_neuronx.experimental.placement import neuron_cores_context
for i in range(num_cores):
    with neuron_cores_context(start_nc=i, nc_count=1):
        replicas.append(torch.jit.load(traced_path))
```

Each `torch.jit.load` inside its own `neuron_cores_context` tells the runtime
to place the NEFF on physical core `i`. After the loop we have N genuinely
independent NEFFs, each sitting on its own core. `replicas[5](x)` runs on core 5
only; `replicas[12](x)` runs on core 12 only — they don't share compute.

A `ThreadPoolExecutor` then issues N forwards concurrently from Python, one
per replica, and the Neuron runtime launches them on their respective cores in
parallel. That's why scaling is close to linear:

| Cores | Theoretical (48 / N) | Measured ms/tile | Efficiency |
|------:|---------------------:|-----------------:|-----------:|
| 1     | 48.0                 | 48.00            | 100%       |
| 2     | 24.0                 | 24.22            | 99%        |
| 4     | 12.0                 | 12.20            | 98%        |
| 8     | 6.0                  | 6.17             | 97%        |
| 16    | 3.0                  | 3.16             | 95%        |
| 32    | 1.5                  | 1.67             | 90%        |

The 10% tax at 32 cores is Python thread scheduling + `Future.result()`
round-trip, not anything in the Neuron runtime.

### How to verify

Run the benchmark in one terminal and `neuron-top` in another:

- **DataParallel**: a single core sits at ~100% utilisation, the rest at 0%.
- **Pinned + ThreadPool**: all N cores sit near 100% simultaneously.

That visual is the quickest confirmation that pinning is what actually puts
every core to work.

### One-line takeaway

**DataParallel gets you the runtime's built-in 2x double-buffering; pinned
replicas get you true N-core parallelism.**
