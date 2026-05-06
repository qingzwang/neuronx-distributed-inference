"""Describe an image with Qwen3.5-2B, end-to-end on Neuron.

- Vision encoder (ViT, 24 layers) runs on Neuron as a standalone traced
  module (see compile_vision_encoder.py).
- Text decoder (2B parameters, hybrid DeltaNet + GQA) runs on Neuron
  with tensor parallelism.

The text decoder's NKI DeltaNet kernel is numerically unstable for long
or highly-repetitive input sequences, so we shrink the image to a 4x4
patch grid (= 4 merged vision tokens after spatial merge) by constraining
the HF processor's pixel budget. That keeps total prompt length well
inside the validated seq_len=128 bucket.

Setup (one-time):
    # 1. compile the text decoder
    pytest test/integration/test_model.py  (or any integration script)
    # 2. compile the vision encoder
    python3 examples/compile_vision_encoder.py --buckets 16

Usage:
    python3 examples/describe_image.py --image /path/to/img.png \\
        --prompt "Describe this image."
"""

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

import torch
from PIL import Image

_CONTRIB_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _CONTRIB_ROOT not in sys.path:
    sys.path.insert(0, _CONTRIB_ROOT)

from src.modeling_qwen35 import (
    NeuronQwen35ForCausalLM,
    Qwen35InferenceConfig,
)
from src.modeling_qwen35_vl import (
    NeuronQwen35VLForCausalLM,
    Qwen35VLInferenceConfig,
)
from neuronx_distributed_inference.models.config import (
    NeuronConfig,
    OnDeviceSamplingConfig,
)


def build_text_model(args):
    """Load the multi-bucket Neuron text decoder.

    Config must match the compile-time config in examples/compile_text_decoder.py
    (bucket list + enable_bucketing) or the runtime won't pick the short
    buckets for short prompts and CTE will always run at the max bucket
    length.
    """
    with open(os.path.join(args.model_path, "config.json")) as f:
        full_cfg = json.load(f)
    text_cfg = dict(full_cfg["text_config"])
    text_cfg["pad_token_id"] = text_cfg.get("eos_token_id", 248044)
    text_cfg.setdefault("tie_word_embeddings", full_cfg.get("tie_word_embeddings", True))
    if "rope_parameters" in text_cfg:
        text_cfg["rope_theta"] = text_cfg["rope_parameters"].get("rope_theta", 10000000)

    buckets = args.buckets or [args.seq_len]
    neuron_config = NeuronConfig(
        tp_degree=args.tp_degree,
        batch_size=1,
        ctx_batch_size=1,
        tkg_batch_size=1,
        seq_len=max(buckets),
        torch_dtype=torch.bfloat16,
        on_device_sampling_config=OnDeviceSamplingConfig(top_k=1),
        enable_bucketing=len(buckets) > 1,
        buckets=buckets,
        context_encoding_buckets=buckets,
        flash_decoding_enabled=False,
        logical_nc_config=2,
        save_sharded_checkpoint=True,
    )
    text_inference_cfg = Qwen35InferenceConfig(
        neuron_config=neuron_config, **text_cfg
    )

    neff = os.path.join(args.compiled_path, "model.pt")
    if not os.path.exists(neff):
        raise FileNotFoundError(
            f"No compiled text model at {args.compiled_path}. Run once: "
            f"USE_NKI=1 python3 examples/compile_text_decoder.py "
            f"--out {args.compiled_path} --buckets {' '.join(map(str, buckets))}"
        )

    return full_cfg, text_inference_cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="/home/ubuntu/models/Qwen3.5-2B")
    parser.add_argument("--compiled-path",
                        default="/home/ubuntu/traced_model/Qwen3.5-2B-s2048-multibucket",
                        help="Directory with the compiled text decoder. "
                             "Produced by examples/compile_text_decoder.py.")
    parser.add_argument("--buckets", type=int, nargs="+",
                        default=[128, 512, 1024, 2048],
                        help="CTE buckets the text decoder was compiled with. "
                             "Must match compile-time config.")
    parser.add_argument("--vision-compiled-path",
                        default="/home/ubuntu/traced_model/Qwen3.5-2B/vision",
                        help="Directory with vision_encoder_{bucket}.pt files")
    parser.add_argument("--image", required=True)
    parser.add_argument("--prompt", default="What is in this image?")
    parser.add_argument("--grid-size", type=int, default=62,
                        help="Vision token grid size (N x N patches -> "
                             "(N/2) x (N/2) merged tokens). Default 62 "
                             "(~992x992 image, 961 merged vision tokens) "
                             "is the largest supported resolution.")
    parser.add_argument("--seq-len", type=int, default=2048,
                        help="Max compiled bucket (for backwards compat; "
                             "use --buckets for the full list).")
    parser.add_argument("--tp-degree", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=80)
    args = parser.parse_args()

    # ---- Build text model & VL wrapper --------------------------------------
    full_cfg, text_inference_cfg = build_text_model(args)
    vl_cfg = Qwen35VLInferenceConfig(
        text_config=text_inference_cfg,
        vision_config=full_cfg["vision_config"],
        image_token_id=full_cfg.get("image_token_id", 248056),
        video_token_id=full_cfg.get("video_token_id", 248057),
        vision_start_token_id=full_cfg.get("vision_start_token_id", 248053),
        vision_end_token_id=full_cfg.get("vision_end_token_id", 248054),
        spatial_merge_size=full_cfg["vision_config"].get("spatial_merge_size", 2),
    )

    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(args.model_path)

    vl_model = NeuronQwen35VLForCausalLM(
        model_path=args.model_path,
        text_config=text_inference_cfg,
        vision_config=vl_cfg,
        processor=processor,
    )
    print("[neuron] loading text decoder on Neuron...")
    vl_model.text_model.load(args.compiled_path)
    print("[neuron] loading vision encoder on Neuron...")
    vl_model.vision_model_wrapper.load_vision_weights_from_hf(args.model_path)
    vl_model.vision_model_wrapper.load_compiled(args.vision_compiled_path)

    # ---- Prepare inputs -----------------------------------------------------
    patch = full_cfg["vision_config"]["patch_size"]
    merge = full_cfg["vision_config"]["spatial_merge_size"]
    # Constrain HF processor to emit a grid_size x grid_size patch grid so the
    # total prompt length stays small.
    pixel_budget = (args.grid_size * patch) ** 2
    processor.image_processor.min_pixels = pixel_budget
    processor.image_processor.max_pixels = pixel_budget

    img = Image.open(args.image).convert("RGB")
    print(f"[input] image original size: {img.size}")

    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": img},
            {"type": "text", "text": args.prompt},
        ],
    }]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )
    input_ids = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask", torch.ones_like(input_ids))
    pixel_values = inputs["pixel_values"]
    image_grid_thw = inputs["image_grid_thw"]
    print(f"[input] input_ids: {input_ids.shape}  "
          f"grid_thw: {image_grid_thw.tolist()}  "
          f"merged_tokens: "
          f"{(image_grid_thw[0,1] // merge) * (image_grid_thw[0,2] // merge)}")

    orig_len = input_ids.shape[1]
    max_bucket = max(args.buckets) if args.buckets else args.seq_len
    if orig_len + args.max_new_tokens > max_bucket:
        print(f"[warn] input ({orig_len}) + max_new ({args.max_new_tokens}) "
              f"> largest bucket ({max_bucket}); lower --grid-size or "
              f"--max-new-tokens, or recompile with a larger bucket.")

    # ---- Generate -----------------------------------------------------------
    print("\n[neuron] generating...")
    t0 = time.perf_counter()
    out_ids = vl_model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        pixel_values=pixel_values,
        image_grid_thw=image_grid_thw,
        max_new_tokens=args.max_new_tokens,
        temperature=0.0,
    )
    elapsed = time.perf_counter() - t0

    gen_tokens = out_ids[0, orig_len:]
    text = processor.tokenizer.decode(gen_tokens, skip_special_tokens=True)
    print(f"\n=== Reply ({len(gen_tokens)} tokens in {elapsed:.2f}s) ===")
    print(text)


if __name__ == "__main__":
    main()
