# Real-ESRGAN batch_size Benchmark at tile=128 (8 pinned cores, 4K input)

- **Model:** `RealESRGAN_x4plus` (RRDBNet, scale x4)
- **Image:** synthetic 4096x4096 (1024 non-overlapping tiles at 128x128)
- **Compile:** LNC=1, bf16, per-(bs, tile) NEFF via `torch_neuronx.trace(..., compiler_args=["--logical-nc-config", "1"])`
- **Runtime:** 8 NeuronCores pinned via `torch_neuronx.experimental.placement.neuron_cores_context`, dispatched through `ThreadPoolExecutor` (yolo26 pattern)
- **dtype:** bf16
- **Date:** 2026-05-08

## What we measured

Each row compiles a dedicated `(bs, 3, 128, 128)` NEFF, replicates it onto 8 pinned
NeuronCores, and runs a full 4K sweep: 1024 tiles are split into `1024 / bs` batches,
dispatched 8 at a time across the replicas. Wall-clock around the full sweep is
reported; ms/tile = total / 1024.

## Results

| Tile | bs | Tiles | Dispatch waves | Median (s) | ms/tile | tiles/s | MPx/s | Compile (s) |
|:-----|---:|------:|---------------:|-----------:|--------:|--------:|------:|------------:|
| 128x128 |  1 | 1024 | 128 | 6.32 | 6.17 | 162.1 | 2.66 | (cached) |
| 128x128 |  2 | 1024 |  64 | 6.37 | 6.22 | 160.7 | 2.63 | 292.5 |
| 128x128 |  4 | 1024 |  32 | 6.37 | 6.22 | 160.7 | 2.63 | 522.4 |
| 128x128 |  8 | 1024 |  16 | 6.42 | 6.27 | 159.5 | 2.61 | 1043.2 |

## What the numbers say

- **Batching does not help when compute is already the bottleneck.** Each LNC=1
  NeuronCore running RRDBNet on a 128x128 tile is fully compute-bound: doubling
  the per-replica batch size just makes each forward twice as long, the total
  sweep time stays flat. bs=1 and bs=8 land within 2% of each other (6.17 vs 6.27 ms/tile).
- **Compile time scales super-linearly with batch size.** 293s → 522s → 1043s for
  bs=2,4,8. bs=16 did not finish within 60 minutes of compile time and was aborted —
  at that point the compiler is spending most of its time on layout / spill decisions
  for the RRDB feature maps at that size.
- **The correct knob for throughput is *replicas*, not *batch*.** The earlier pinned
  multi-core sweep (see `BENCHMARK_PINNED_MULTICORE.md`) shows near-linear scaling
  with core count — 1024 tiles in 1.7s at 32 cores, vs 6.3s at 8 cores. Spending
  cores on more replicas beats spending them on bigger per-replica batches.

## Skipped combos

- **tile=128 @ bs=16, 32** — aborted: compile time > 1 hour with no indication it
  would terminate, and the flat ms/tile from bs=1,2,4,8 gave us no reason to expect
  different behaviour at larger bs.
- **tile=256 series (bs=1..32)** — removed from the sweep. The tile-size benchmark
  (`BENCHMARK_TILE_SIZES.md`) already showed that 256x256 tiles have ~30% worse
  throughput than 128x128 because the RRDB intermediate feature map
  (`(1, 192, 256, 256)` bf16 ≈ 24 MB) is close to the on-chip SRAM limit; larger
  batches at tile=256 would only make spilling worse.

## How to reproduce

```bash
cd contrib/models/Real-ESRGAN
python test/integration/benchmark_batch_tile.py \
  --tiles 128 \
  --batch-sizes 1 2 4 8 \
  --iters 3 \
  --output-md BENCHMARK_BATCH_TILE.md
```

Each run reuses cached NEFFs at
`/home/ubuntu/neuron_models/real-esrgan/bench/RealESRGAN_x4plus/bs{N}_tile128x128_bf16_lnc1/`;
delete that directory to force recompilation. The benchmark uses the same
pinned-replica ThreadPool pattern as `benchmark_pinned_multicore.py`.
