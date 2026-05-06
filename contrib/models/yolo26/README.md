# YOLO26 on AWS Neuron (trn2)

End-to-end object detection with Ultralytics' YOLO26 family compiled for
trn2 via `torch_neuronx.trace`. Neuron output matches CPU eager-mode on
the same image to within fp32 rounding noise across all 5 size variants.

- Upstream repo: https://github.com/ultralytics/ultralytics
- Weights: `yolo26{n,s,m,l,x}.pt` (80 COCO classes)
- Input size: 640x640 fits `n` and `s`; `m`/`l`/`x` require ≤ ~576x576
  because the full model doesn't fit in a single NeuronCore's SBUF at 640.
  The size sweep below uses 480x480 for apples-to-apples comparison.
- Instance used for benchmarks: `trn2.48xlarge`

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
  test/
    test_yolo26.py        # pytest parity check (cpu vs neuron)
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
#    640 fits only n/s on trn2; 480 fits all 5.
python benchmark_sizes.py --imgsz 480                 # full sweep
python benchmark_sizes.py --sizes n s --imgsz 640     # 640 sweep (n/s only)

# 7. Unit tests (skip automatically if Neuron / compiled .pt absent)
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

## 640x640 (yolo26n/s only — larger models don't fit)

At 640x640 the full model + activations exceed the SBUF on one NeuronCore
for m/l/x and the compiler bails with `NCC_IGCA030`. For n and s at 640:

| variant | CPU (ms) | Neuron fp32 (ms) | fp32 speedup | Neuron fp16 (ms) | 32-core throughput (img/s) |
| ------- | -------: | ---------------: | -----------: | ---------------: | -------------------------: |
| yolo26n |    47.98 |            17.52 |        2.74× |            32.39 |                      826.8 |
| yolo26s |    54.53 |            13.20 |        4.13× |            16.54 |                      933.4 |

For the larger models at 640, options are: (a) reduce `--imgsz` to ≤ ~576,
(b) shard with tensor parallelism over multiple NeuronCores via `neuronx_distributed`.

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

- Batch size is fixed to 1 at compile time. `benchmark_multicore.py` scales
  throughput by running one NEFF per NeuronCore in parallel — change batch
  by recompiling with a different example tensor.
- LNC=1 means each traced graph fits on a single NeuronCore. This works up
  through yolo26x at ≤480×480. For 640×640 m/l/x, either drop resolution or
  shard with tensor parallelism via `neuronx_distributed`.
- The top-k step (≈0.5 ms) runs on CPU. A custom NKI top-k kernel would
  make the whole pipeline on-device.
