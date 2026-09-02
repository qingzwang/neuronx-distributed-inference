"""Generate one image per LoRA adapter from a single compiled FLUX model.

Adapters passed with --lora are declared at build time and are resident on device
after load. Adapters passed with --dynamic-lora are loaded after the model has been
compiled and loaded, which needs --max-cpu-loras (dynamic_multi_lora is turned on
automatically). Either way the model is compiled once.

The per-image timings show what an adapter costs: a request whose adapter is
already in a device slot runs at base-model speed, while one that has to be
swapped in from host memory pays for the copy, once per request.

See examples/flux_lora.md for the full picture.

  python examples/generate_flux_lora.py \
      -c /shared/flux/FLUX.1-dev/ --compile_workdir /tmp/flux-lora/ \
      --lora realism=/adapters/xlabs-realism \
      --dynamic-lora superreal=/adapters/super-realism.safetensors \
      --max-lora-rank 64 --max-cpu-loras 4 --save_image
"""

import argparse
import time

import torch

from neuronx_distributed_inference.models.diffusers.flux.application import (
    NeuronFluxApplication,
    create_flux_config,
    get_flux_parallelism_config,
)
from neuronx_distributed_inference.models.diffusers.flux.lora import build_flux_lora_config
from neuronx_distributed_inference.utils.random import set_random_seed

DEFAULT_COMPILE_WORK_DIR = "/tmp/flux_lora/compiler_workdir/"
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


def run(args):
    set_random_seed(args.seed)
    declared = parse_adapters(args.lora)
    dynamic = parse_adapters(args.dynamic_lora)
    if not declared and not dynamic:
        raise ValueError("Pass at least one --lora or --dynamic-lora")

    lora_config = build_flux_lora_config(
        # Room for every adapter at once unless the caller asked for less, so the
        # default run does not spend its time swapping.
        max_loras=args.max_loras or len(declared) + len(dynamic),
        max_lora_rank=args.max_lora_rank,
        max_cpu_loras=args.max_cpu_loras,
        dynamic_multi_lora=bool(dynamic),
        lora_ckpt_paths=declared or None,
    )

    world_size = get_flux_parallelism_config(args.backbone_tp_degree)
    clip_config, t5_config, backbone_config, decoder_config = create_flux_config(
        args.checkpoint_dir,
        world_size,
        args.backbone_tp_degree,
        torch.bfloat16,
        args.height,
        args.width,
        lora_config=lora_config,
    )
    flux_app = NeuronFluxApplication(
        model_path=args.checkpoint_dir,
        text_encoder_config=clip_config,
        text_encoder2_config=t5_config,
        backbone_config=backbone_config,
        decoder_config=decoder_config,
        height=args.height,
        width=args.width,
    )
    flux_app.compile(args.compile_workdir)
    flux_app.load(args.compile_workdir)

    for name, path in dynamic.items():
        start = time.perf_counter()
        flux_app.add_lora_adapter(name, path)
        print(f"loaded {name} from {path} in {time.perf_counter() - start:.2f} s", flush=True)

    print(f"adapters available: {sorted(flux_app.list_lora_adapters())}")

    # None first: the base model still occupies slot 0 and is reachable at any time.
    for name in [None] + list(declared) + list(dynamic):
        flux_app.set_lora_adapters(name)
        start = time.perf_counter()
        with flux_app.pipe.transformer.image_rotary_emb_cache_context():
            image = flux_app(
                args.prompt,
                height=args.height,
                width=args.width,
                guidance_scale=args.guidance_scale,
                num_inference_steps=args.num_inference_steps,
                generator=torch.Generator("cpu").manual_seed(args.seed),
            ).images[0]
        elapsed = time.perf_counter() - start
        label = name or "base"
        print(f"{label:<24} {elapsed:6.2f} s")
        if args.save_image:
            image.save(f"output_{label}.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--prompt", type=str,
                        default="a close-up portrait photograph of an elderly fisherman mending a net")
    parser.add_argument("-hh", "--height", type=int, default=1024)
    parser.add_argument("-w", "--width", type=int, default=1024)
    parser.add_argument("-n", "--num_inference_steps", type=int, default=20)
    parser.add_argument("-g", "--guidance_scale", type=float, default=3.5)
    parser.add_argument("-c", "--checkpoint_dir", type=str, default=DEFAULT_CKPT_DIR)
    parser.add_argument("--compile_workdir", type=str, default=DEFAULT_COMPILE_WORK_DIR)
    parser.add_argument("--backbone_tp_degree", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save_image", action="store_true")
    parser.add_argument("--lora", action="append", metavar="NAME=PATH",
                        help="Adapter declared at build time; repeatable")
    parser.add_argument("--dynamic-lora", action="append", metavar="NAME=PATH",
                        help="Adapter loaded after the model is loaded; repeatable")
    parser.add_argument("--max-loras", type=int, default=None,
                        help="Adapters resident on device, on top of the base slot. "
                             "Defaults to all of them, so nothing has to be swapped.")
    parser.add_argument("--max-lora-rank", type=int, default=64,
                        help="Largest adapter rank to size the slots for. Memory and "
                             "swap cost scale with this, so keep it tight.")
    parser.add_argument("--max-cpu-loras", type=int, default=4,
                        help="Adapters held in host memory; only used with --dynamic-lora")
    run(parser.parse_args())
