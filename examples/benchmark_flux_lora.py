"""Measure what a LoRA costs per request, per cache tier, on one compiled FLUX model.

The number that matters for sizing a deployment is not "how long does selecting an
adapter take" -- selection is a graph input, so it takes nothing -- but "is the
adapter I need resident on device". This script measures both sides of that:

    --max-loras N (default: all of them)  every adapter resident, so an alternating
                                          request pattern never moves weights
    --max-loras 1                         one device slot, so alternating forces a
                                          host -> device swap every request

Three patterns are timed against each other, on the backbone step alone with
synthetic inputs, because that is where an adapter is applied and it keeps the
measurement away from the encoders and the VAE:

    repeating one adapter   a device hit
    alternating adapters    a device hit at max_loras=N, a swap at max_loras=1
    no adapter              the base model in slot 0

  python examples/benchmark_flux_lora.py \
      -c /shared/flux/FLUX.1-dev/ --compile_workdir /tmp/flux-lora/ \
      --lora xlabs=/adapters/xlabs-realism \
      --lora kohya=/adapters/super-realism.safetensors \
      --max-lora-rank 64

Changing --max-loras or --max-lora-rank changes the graph, so give each
configuration its own --compile_workdir.

See examples/flux_lora.md for what these numbers came out as on a trn2.3xlarge.
"""

import argparse
import statistics
import time

import torch

from neuronx_distributed_inference.models.diffusers.flux.application import (
    NeuronFluxApplication,
    create_flux_config,
    get_flux_parallelism_config,
)
from neuronx_distributed_inference.models.diffusers.flux.lora import build_flux_lora_config

DEFAULT_COMPILE_WORK_DIR = "/tmp/flux_lora_benchmark/compiler_workdir/"
DEFAULT_CKPT_DIR = "/shared/flux/FLUX.1-dev/"


def parse_adapters(values):
    """Turn ["name=path", ...] into {"name": "path", ...}."""
    adapters = {}
    for value in values or []:
        if "=" not in value:
            raise ValueError(f"Expected NAME=PATH, got {value!r}")
        name, path = value.split("=", 1)
        adapters[name.strip()] = path.strip()
    return adapters


def synthetic_inputs(size, seq_len, dtype=torch.bfloat16):
    """One backbone step's worth of inputs, shaped for ``size`` x ``size``."""
    num_patches = size * size // (16 * 16)
    return dict(
        hidden_states=torch.randn(1, num_patches, 64, dtype=dtype),
        encoder_hidden_states=torch.randn(1, seq_len, 4096, dtype=dtype),
        pooled_projections=torch.randn(1, 768, dtype=dtype),
        timestep=torch.full((1,), 0.7),
        guidance=torch.full((1,), 3.5),
        txt_ids=torch.zeros(seq_len, 3),
        img_ids=torch.zeros(num_patches, 3),
        return_dict=False,
    )


def show(label, samples):
    print(
        f"  {label:<40} median {statistics.median(samples):7.1f} ms  "
        f"min {min(samples):7.1f}  max {max(samples):7.1f}",
        flush=True,
    )
    return statistics.median(samples)


def run(args):
    adapters = parse_adapters(args.lora)
    if len(adapters) < 2:
        raise ValueError(
            "Pass at least two --lora adapters: the point of the measurement is what "
            "alternating between them costs."
        )

    max_loras = args.max_loras or len(adapters)
    # An adapter declared at build time is resident by definition: LoraServingConfig
    # raises max_loras to cover every declared checkpoint. So measuring fewer slots
    # than adapters means declaring only the ones that fit and adding the rest at
    # runtime -- which is also what a working set larger than HBM looks like.
    declared = dict(list(adapters.items())[:max_loras])
    dynamic = {k: v for k, v in adapters.items() if k not in declared}
    lora_config = build_flux_lora_config(
        max_loras=max_loras,
        max_lora_rank=args.max_lora_rank,
        # Room in host memory for all of them, so a miss on the device tier is a
        # host -> device swap rather than a re-read from disk.
        max_cpu_loras=max(args.max_cpu_loras, len(adapters)),
        dynamic_multi_lora=True,
        lora_ckpt_paths=declared,
    )
    world_size = get_flux_parallelism_config(args.backbone_tp_degree)
    clip_config, t5_config, backbone_config, decoder_config = create_flux_config(
        args.checkpoint_dir,
        world_size,
        args.backbone_tp_degree,
        torch.bfloat16,
        args.size,
        args.size,
        lora_config=lora_config,
    )
    flux_app = NeuronFluxApplication(
        model_path=args.checkpoint_dir,
        text_encoder_config=clip_config,
        text_encoder2_config=t5_config,
        backbone_config=backbone_config,
        decoder_config=decoder_config,
        height=args.size,
        width=args.size,
    )
    start = time.perf_counter()
    flux_app.compile(args.compile_workdir)
    print(f"compiled (or cache hit) in {time.perf_counter() - start:.0f} s", flush=True)
    flux_app.load(args.compile_workdir)

    for name, path in dynamic.items():
        start = time.perf_counter()
        flux_app.add_lora_adapter(name, path)
        print(f"added {name} to the host tier in {time.perf_counter() - start:.2f} s",
              flush=True)

    names = list(adapters)
    resident = max_loras >= len(names)
    print(
        f"\nmax_loras={max_loras}, rank {args.max_lora_rank}, "
        f"{len(names)} adapters: {names} -- "
        f"{'all resident' if resident else f'{len(declared)} resident, {len(dynamic)} in host memory'}",
        flush=True,
    )

    inputs = synthetic_inputs(args.size, args.max_sequence_length)

    def call(adapter):
        start = time.perf_counter()
        with torch.no_grad():
            flux_app.pipe.transformer(
                adapter_ids=[adapter] if adapter else None, **inputs
            )
        return (time.perf_counter() - start) * 1e3

    # Every adapter through the machinery once, so nothing below pays a first-touch
    # cost that belongs to startup rather than to a request.
    for name in names * 2:
        call(name)

    repeat = [call(names[0]) for _ in range(args.reps)]
    alternate = [call(names[i % len(names)]) for i in range(args.reps)]
    base = [call(None) for _ in range(args.reps)]

    print(
        f"\n=== {args.size}px, one backbone step, TP={args.backbone_tp_degree}, "
        f"rank {args.max_lora_rank}, max_loras={max_loras} ===",
        flush=True,
    )
    r = show(f"repeating one adapter ({names[0]})", repeat)
    a = show(f"alternating {len(names)} adapters", alternate)
    b = show("no adapter (base model, slot 0)", base)

    print(
        f"\n  alternating vs repeating = {a - r:+.1f} ms per call"
        f"{' -- selection only, every adapter resident' if resident else ' -- a host -> device swap'}",
        flush=True,
    )
    print(f"  any adapter vs the base model = {r - b:+.1f} ms per call", flush=True)
    if resident:
        print(
            "\n  With every adapter resident, alternating costs what repeating costs:\n"
            "  the slot index is a graph input. Re-run with --max-loras 1 to see what\n"
            "  the same pattern costs when the adapter has to be brought in.",
            flush=True,
        )
    else:
        print(
            f"\n  {a - r:.0f} ms of every request goes to moving weights. Raise "
            f"--max-loras to\n  {len(names)} (at {args.max_lora_rank} rank, per the memory "
            "footprint logged above) to remove it.",
            flush=True,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--checkpoint_dir", type=str, default=DEFAULT_CKPT_DIR)
    parser.add_argument("--compile_workdir", type=str, default=DEFAULT_COMPILE_WORK_DIR)
    parser.add_argument("--lora", action="append", metavar="NAME=PATH",
                        help="Adapter declared at build time; repeatable, at least twice")
    parser.add_argument("--size", type=int, default=256,
                        help="Square resolution. The adapter cost does not depend on it, "
                             "so the default is small to keep the run short.")
    parser.add_argument("--max_sequence_length", type=int, default=512,
                        help="T5 prompt budget the backbone was built for")
    parser.add_argument("--backbone_tp_degree", type=int, default=4)
    parser.add_argument("--max-loras", type=int, default=None,
                        help="Adapters resident on device, on top of the base slot. "
                             "Defaults to all of them; pass 1 to measure the swap. "
                             "Adapters beyond this many are added at runtime rather "
                             "than declared, since a declared adapter is always "
                             "resident.")
    parser.add_argument("--max-lora-rank", type=int, default=64,
                        help="Slot width. Memory and swap cost scale with it.")
    parser.add_argument("--max-cpu-loras", type=int, default=4,
                        help="Adapters held in host memory")
    parser.add_argument("--reps", type=int, default=10,
                        help="Timed calls per pattern")
    run(parser.parse_args())
