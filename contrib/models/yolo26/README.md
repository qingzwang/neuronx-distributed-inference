# YOLO26 on AWS Neuron (trn2)

End-to-end object detection with Ultralytics' YOLO26-nano compiled for
trn2 via `torch_neuronx.trace`. Neuron output matches CPU eager-mode on
the same image to within fp32 rounding noise.

- Upstream repo: https://github.com/ultralytics/ultralytics
- Weights: `yolo26n.pt` (nano, 80 COCO classes, 640x640 input)
- Instance used for benchmarks: `trn2.48xlarge`, single NeuronCore

## Layout

```
yolo26/
  src/
    yolo26_common.py      # letterbox, pre/postprocess, timing helpers
    neuron_patches.py     # monkey-patch Attention to dodge torch.split bug
    run_cpu.py            # CPU inference + CPU latency benchmark
    compile_neuron.py     # trace + torch_neuronx.compile
    run_neuron.py         # Neuron inference + CPU-parity check + latency
    benchmark.py          # consolidated CPU-vs-Neuron benchmark
    benchmark_multicore.py # data-parallel throughput sweep across NeuronCores
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

# 6. Unit tests (skip automatically if Neuron / compiled .pt absent)
pytest ../test/test_yolo26.py -v
```

## Single-core latency (bus.jpg, 640x640, 50 iters, 5 warmup)

| path          | model forward mean | end-to-end mean | max score Δ vs CPU | min IoU vs CPU |
| ------------- | -----------------: | --------------: | -----------------: | -------------: |
| CPU (x86)     | **54.06 ms**       | 60.25 ms        | —                  | —              |
| Neuron (fp32) | **17.59 ms**       | 24.16 ms        | 6.6e-7             | 0.9999         |
| Neuron (fp16) | 32.42 ms           | 39.05 ms        | 1.6e-4             | 0.9993         |
| Neuron (bf16) | 32.44 ms           | 38.84 ms        | 1.7e-3             | 0.9967         |

Note: on a model this small (~2.4 M params, 5.4 GFLOPs), `--auto-cast=none`
(fp32) actually *beats* fp16/bf16 on trn2 because the nano graph is
compute-light and matmul auto-cast inserts cast ops whose overhead dominates.
fp16/bf16 are still available for mixing with larger models.

Full per-image tables are in `benchmark/benchmark_report.md`.

### Accuracy vs CPU

```
bus.jpg    : 5 detections CPU vs 5 Neuron, max score delta 6.6e-07, min IoU 0.9999
zidane.jpg : 3 detections CPU vs 3 Neuron, max score delta 7.2e-07, min IoU 0.9999
```

## Multi-core throughput (fp32, `torch_neuronx.DataParallel`, 50 iters, 5 warmup)

Each step feeds one image per NeuronCore (`batch = num_cores`). Per-image
latency is step wall-clock ÷ batch; `num_workers` is set to `2 * num_cores`
so the dispatcher does not become the bottleneck.

| cores | batch | per-image (ms) | throughput (img/s) | scaling |
|------:|------:|---------------:|-------------------:|--------:|
| 1     | 1     | 18.38          | 54.4               | 1.00x   |
| 2     | 2     |  9.54          | 104.9              | 1.93x   |
| 4     | 4     |  4.95          | 202.2              | 3.72x   |
| 8     | 8     |  2.69          | 372.1              | 6.84x   |
| 16    | 16    |  1.74          | 573.6              | 10.55x  |
| 32    | 32    |  1.12          | 889.6              | 16.35x  |

Raw JSON / markdown outputs live in `benchmark/benchmark_multicore*.{json,md}`
for each dtype (fp32 is the default artifact).

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

- Only YOLO26-nano is wired in; other sizes need a fresh trace.
- Batch size is fixed to 1 at compile time (change via `--imgsz` etc.
  and recompile).
- LNC=1 is used so the graph fits on a single NeuronCore; no tensor
  parallelism is needed for this model size.
- The top-k step (~0.5 ms) runs on CPU. If you need fully on-device
  postprocessing, replace it with a custom NKI top-k kernel.
