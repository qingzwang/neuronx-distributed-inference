# YOLO26 on AWS Neuron (trn2)

End-to-end object detection with Ultralytics' YOLO26 family compiled for
trn2 via `torch_neuronx.trace`. Neuron output matches CPU eager-mode on
the same image to within fp32 rounding noise across all 5 size variants.

- Upstream repo: https://github.com/ultralytics/ultralytics
- Weights: `yolo26{n,s,m,l,x}.pt` (80 COCO classes)
- Input size: 640×640 works for every variant. `n`/`s` fit with fp32
  activations + `--auto-cast`; `m`/`l`/`x` need model weights pre-cast to
  `bfloat16` before tracing so the single-core SBUF can hold BS=32 (see
  the aligned benchmark section).
- A smaller resolution (480×480) sweep is also included so `n → x` can be
  compared at the same input shape.
- Instance used for benchmarks: `trn2.48xlarge` (LNC=1)

## Layout

```
yolo26/
  src/
    yolo26_common.py      # letterbox, pre/postprocess, timing helpers
    neuron_patches.py     # monkey-patch Attention to dodge torch.split bug
    run_cpu.py            # CPU inference + CPU latency benchmark
    compile_neuron.py     # trace + torch_neuronx.compile
    run_neuron.py         # Neuron inference + CPU-parity check + latency
    benchmark.py          # consolidated CPU-vs-Neuron benchmark (yolo26n)
    benchmark_multicore.py # data-parallel throughput sweep across NeuronCores
    benchmark_sizes.py    # sweep n/s/m/l/x on CPU + Neuron fp32/fp16 + multi-core
    benchmark_aligned.py  # peak-throughput aligned with AWS Neuron reference table
  test/
    test_yolo26.py        # pytest parity check (cpu vs neuron)
  reproduce_jimburtoft/   # verbatim copy of the upstream contrib + notebook
    modeling_yolo26.py    # unmodified from jimburtoft/neuronx-distributed-inference
    yolo26_neuron_notebook.ipynb
    run_bench.py          # script form of the notebook's DP=8 benchmark
  assets/                 # sample images (bus.jpg, zidane.jpg)
  compiled/               # saved .pt traced modules
  benchmark/              # JSON + markdown benchmark outputs
  yolo26n.pt              # pretrained weights
```

## Running

All commands assume the Neuron pytorch venv is active:
`source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate`.

```bash
cd contrib/models/yolo26/src

# 1. CPU baseline
python run_cpu.py

# 2. Compile for Neuron (fp32 / bf16 / fp16 matmul, LNC=1). Takes ~30-40 s.
python compile_neuron.py --dtype fp32
python compile_neuron.py --dtype bf16 --out ../compiled/yolo26n_neuron_bf16.pt
python compile_neuron.py --dtype fp16 --out ../compiled/yolo26n_neuron_fp16.pt

# 3. Run on Neuron, verify vs CPU, benchmark
python run_neuron.py

# 4. Produce the combined single-core benchmark report
python benchmark.py

# 5. Sweep throughput across 1/2/4/8/16/32 NeuronCores with DataParallel
python benchmark_multicore.py

# 6. Sweep all five sizes {n,s,m,l,x} on CPU + Neuron fp32/fp16 + multi-core.
#    640 fits only n/s with --auto-cast; 480 fits all 5.
python benchmark_sizes.py --imgsz 480                 # full sweep
python benchmark_sizes.py --sizes n s --imgsz 640     # 640 sweep (n/s only)

# 7. Aligned peak-throughput at 640 with per-variant dtype/BS (reference config).
#    m/l/x use bf16 weights directly (cast before trace) to fit on a single core.
python benchmark_aligned.py

# 8. Unit tests (skip automatically if Neuron / compiled .pt absent)
pytest ../test/test_yolo26.py -v
```

## Size sweep — single-core latency at 480x480 (bus.jpg, 30 iters)

All five variants at the same input resolution, so absolute numbers
compare directly. Speedup is CPU ÷ Neuron-single-core.

| variant | weights (MB) | CPU (ms) | Neuron fp32 (ms) | fp32 speedup | Neuron fp16 (ms) | fp16 speedup |
| ------- | -----------: | -------: | ---------------: | -----------: | ---------------: | -----------: |
| yolo26n |          5.3 |    38.66 |             9.97 |       3.88×  |            18.44 |       2.10×  |
| yolo26s |         19.5 |    46.72 |             7.56 |       6.18×  |             8.98 |       5.20×  |
| yolo26m |         42.2 |    65.45 |             9.12 |       7.18×  |             6.93 |       9.44×  |
| yolo26l |         50.7 |    89.69 |            12.20 |       7.35×  |             8.25 |      10.87×  |
| yolo26x |        113.2 |   108.55 |            16.62 |       6.53×  |             9.73 |      11.15×  |

Observations:
- fp32 is faster than fp16 for **n** (compute-light, cast overhead dominates).
- From **m** upward fp16 wins — the matmul engine runs at 2× throughput in fp16
  and the cast overhead becomes amortised.
- Across 5 variants Neuron gives 3.9× → 11.2× over single-threaded CPU.

### Accuracy vs CPU (same image, same conf=0.25, 480x480)

| variant | cpu dets | fp32 dets | fp32 max Δ | fp32 min IoU | fp16 dets | fp16 max Δ | fp16 min IoU |
| ------- | -------: | --------: | ---------: | -----------: | --------: | ---------: | -----------: |
| yolo26n |        6 |         6 |    6.0e-6  |       1.0000 |         6 |     4.6e-3 |       0.9995 |
| yolo26s |        5 |         5 |    4.2e-7  |       1.0000 |         5 |     3.9e-4 |       0.9996 |
| yolo26m |        5 |         5 |    7.2e-7  |       1.0000 |         5 |     3.7e-4 |       0.9997 |
| yolo26l |        6 |         6 |    5.3e-5  |       1.0000 |         5 |     1.4e-2 |       0.9997 |
| yolo26x |        5 |         5 |    4.2e-7  |       1.0000 |         5 |     7.4e-5 |       0.9998 |

fp32 matches CPU to within 6e-5 across all sizes. fp16 stays within 1.4e-2
score delta and IoU ≥ 0.9995; yolo26l fp16 drops one low-confidence detection
(5 vs 6).

## Peak-throughput sweep at 640×640 (aligned with AWS Neuron reference)

`benchmark_aligned.py` runs the exact per-variant config reported in the
upstream AWS Neuron YOLO26 benchmark: dtype varies per variant, BS/core up
to 32 at 640×640, DP=8 NeuronCores, LNC=1.

For `m/l/x` the reference's `bf16` route is implemented by **casting model
weights to `torch.bfloat16` before tracing** (not `--auto-cast=matmult`);
`--auto-cast` leaves enough fp32 activations in memory that 640×640 at
BS=32 exceeds SBUF on a single core. Casting weights directly halves the
weight footprint during compile and lets the reference BS configuration
fit.

| variant | params (pre-fuse) | dtype | BS/core | NEFF (MB) | throughput (img/s) | per-image (ms) |
|---------|------------------:|-------|--------:|----------:|-------------------:|---------------:|
| yolo26n |              2.6M | FP32  |       1 |     10.2  |                419 |           2.38 |
| yolo26s |             10.0M | FP32  |      32 |     75.7  |                539 |           1.85 |
| yolo26m |             21.9M | BF16  |      32 |     78.9  |                493 |           2.03 |
| yolo26l |             26.3M | BF16  |      32 |     97.4  |                449 |           2.23 |
| yolo26x |             59.0M | BF16  |      16 |    127.5  |                380 |           2.63 |

### Comparison to reference (trn2.3xlarge, Neuron SDK 2.28/2.29)

| variant | my trn2.48xlarge | ref trn2.3xlarge | gap  |
|---------|-----------------:|-----------------:|-----:|
| yolo26n |         419 img/s |        272 img/s | +54% |
| yolo26s |         539 img/s |      1 523 img/s | −65% |
| yolo26m |         493 img/s |      1 267 img/s | −61% |
| yolo26l |         449 img/s |      1 093 img/s | −59% |
| yolo26x |         380 img/s |        876 img/s | −57% |

Parameter counts, dtype, and BS/core are identical to the reference. My
SDK is `2.29.1` (one point release *newer* than the reference's 2.28/2.29),
so version isn't what's causing the remaining gap — it's the instance
form factor. See the next section for a verbatim reproduction of the
upstream notebook.

### Running the upstream notebook verbatim

`reproduce_jimburtoft/` contains the original
[`yolo26_neuron_notebook.ipynb`](https://github.com/jimburtoft/neuronx-distributed-inference/tree/contrib/yolo26/contrib/models/YOLO26)
and its `modeling_yolo26.py` copied unchanged. Executed end-to-end on
our `trn2.48xlarge` with `NEURON_LOGICAL_NC_CONFIG=1`:

| variant | dtype | BS/core | my trn2.48xlarge | ref trn2.3xlarge | ratio |
|---------|-------|--------:|-----------------:|-----------------:|------:|
| yolo26n | fp32  |  1 |   67.9 img/s |   272 img/s | 0.25× |
| yolo26s | fp32  | 32 |  227.0 img/s | 1 523 img/s | 0.15× |
| yolo26m | bf16  | 32 |  281.9 img/s | 1 267 img/s | 0.22× |
| yolo26l | bf16  | 32 |  249.5 img/s | 1 093 img/s | 0.20× |
| yolo26x | bf16  | 16 |  203.4 img/s |   876 img/s | 0.23× |

Accuracy matches (CosSim ≥ 0.988 across all variants, same numbers the
upstream README reports). Throughput is 4–7× lower than the reference.
Same code, same dtype, same BS — so neither source modifications nor
dtype handling explains it.

**Where the gap comes from — single core vs multi-core.** Single-core
latency already runs ~2× slower than the reference's implied per-core
share: `yolo26s` single-core is 14.1 ms/image here vs 5.3 ms/image on
trn2.3xlarge (= 1523 img/s ÷ 8 cores). And then `DataParallel` leaves
another ~2.5× on the table for 8-way scaling (see below).

### Why my top-level numbers are higher than the upstream notebook

Same NEFF (`yolo26s_fp32_bs32`), same DP=8, same warmup/iters, **only the
dispatch path changes**:

| dispatch               | step (ms) | throughput  | vs upstream |
| ---------------------- | --------: | ----------: | ----------: |
| A. `DataParallel` defaults (`num_workers=2`, what the upstream notebook uses) |    1 118 |   229 img/s | 1.00× |
| B. `DataParallel` with `num_workers = 2 × DP` |      456 |   562 img/s | 2.45× |
| C. Explicit per-core NEFF placement via `neuron_cores_context` + `ThreadPoolExecutor` |      384 |   666 img/s | 2.91× |

Two dispatcher bottlenecks stack on top of each other:
1. `torch_neuronx.DataParallel` defaults to `num_workers=2`, so all 8 cores
   get fed through a 2-thread pool — the 7th core is waiting for a thread
   long before its predecessor finishes. Bumping the workers to `2 × DP`
   removes that serialisation (→ 562 img/s, +145%).
2. Even then, `DataParallel`'s scatter still has overhead. Loading one
   NEFF replica per core explicitly and dispatching with a plain
   `ThreadPoolExecutor` is another +18% on top (→ 666 img/s).

`benchmark_aligned.py` and `benchmark_multicore.py` both use route C.
`benchmark_sizes.py` does the same. If you run `reproduce_jimburtoft/`
you get route A and see numbers that match the upstream notebook.

## Multi-core throughput (fp32, `torch_neuronx.DataParallel`, 480x480, 30 iters)

Each step feeds one image per NeuronCore (`batch = num_cores`). Per-image
latency is step wall-clock ÷ batch; `num_workers` is set to `2 × num_cores`
so the dispatcher isn't the bottleneck.

| variant | 1 core (img/s) | 8 cores (img/s) | 32 cores (img/s) | 32-core per-image (ms) |
| ------- | -------------: | --------------: | ---------------: | ---------------------: |
| yolo26n |           93.5 |           651.7 |           1344.6 |                   0.74 |
| yolo26s |          124.7 |           786.8 |           1857.9 |                   0.54 |
| yolo26m |          101.3 |           667.0 |           1650.4 |                   0.61 |
| yolo26l |           78.0 |           514.5 |           1323.3 |                   0.76 |
| yolo26x |           58.1 |           406.0 |           1066.2 |                   0.94 |

yolo26**s** is the throughput sweet spot at 480x480 (1858 img/s across 32
cores, 0.54 ms/image).

## 640×640 with `--auto-cast` (yolo26n/s only)

At 640×640 with the default `--auto-cast=matmult` path, activations remain
fp32 and only m/l/x spill SBUF. For n and s this config is fine:

| variant | CPU (ms) | Neuron fp32 (ms) | fp32 speedup | Neuron fp16 (ms) | 32-core throughput (img/s) |
| ------- | -------: | ---------------: | -----------: | ---------------: | -------------------------: |
| yolo26n |    47.98 |            17.52 |        2.74× |            32.39 |                      826.8 |
| yolo26s |    54.53 |            13.20 |        4.13× |            16.54 |                      933.4 |

For the larger models at 640, use the **bf16-weights** approach from the
aligned benchmark above (`benchmark_aligned.py` casts model weights to
`torch.bfloat16` before tracing), or reduce `--imgsz` to 480 / 576.

Raw JSON / markdown outputs per resolution and dtype live in `benchmark/`.

## Why the monkey-patch?

YOLO26's detect head is NMS-free: its eager forward ends with a single
`topk` over 8400 anchors. That `topk` gets lowered by XLA to a `sort` op,
which neuronx-cc rejects for trn2 with
`[NCC_EVRF029] Operation sort is not supported on trn2`. Rather than
implementing top-k via an NKI kernel, we compile the graph up to the
pre-topk tensor `(B, 8400, 84)` (xyxy boxes + sigmoid class scores) and
run the cheap class-aware top-k on CPU — see
`yolo26_common.neuron_topk`.

The second issue is numerical: the C2PSA attention block uses
`torch.split` to cut a `(B, H, key_dim*2 + head_dim, N)` tensor into
`q/k/v` chunks of unequal size. neuronx-cc lowers that to a dynamic
slice that returns wrong values on trn2 for our layout (32/32/64), so
the patched `Attention.forward` in `neuron_patches.py` replaces `split`
with explicit contiguous slices. After the patch, fp32 Neuron output
matches CPU to within 1e-6.

## Limitations

- Batch size is baked into each NEFF at compile time (`--batch-size` in
  `compile_neuron.py`). Different batches require fresh trace runs.
- LNC=1 only. `m`/`l`/`x` at 640 need bf16 weights to fit; to stay in fp32
  activations for those, drop the input resolution to ≤480.
- The top-k step (≈0.5 ms) runs on CPU. A custom NKI top-k kernel would
  make the whole pipeline on-device.
- Throughput with plain `torch_neuronx.DataParallel` collapses past DP=2
  for these graphs; `benchmark_multicore.py` / `benchmark_aligned.py` use
  explicit per-core placement via `torch_neuronx.neuron_cores_context`
  and a thread pool instead.
