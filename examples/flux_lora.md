# LoRA adapters for FLUX

The FLUX backbone supports multi-LoRA serving: several adapters live on device at
once, each request picks one by name, and with `dynamic_multi_lora=True` adapters
can be loaded after the model has been compiled and loaded. The base model stays
available throughout — it occupies slot 0 and is what a request with no adapter
gets.

Community FLUX adapters ship in at least three key conventions (diffusers/PEFT,
kohya, XLabs). All three work: the file is handed to
`FluxPipeline.lora_state_dict`, so diffusers' own converters do the format work,
and the result is mapped onto NxDI's module names from there.

Never used Trainium before? [flux_lora_handson_zh.md](flux_lora_handson_zh.md) walks
through the whole thing in Chinese, from an empty machine to generated images.

A runnable version of everything below is `examples/generate_flux_lora.py`:

```bash
python examples/generate_flux_lora.py \
    -c /shared/flux/FLUX.1-dev/ --compile_workdir /tmp/flux-lora/ \
    --lora realism=/adapters/xlabs-realism \
    --dynamic-lora superreal=/adapters/super-realism.safetensors \
    --max-lora-rank 64 --save_image
```

## Declaring adapters at build time

```python
import torch
from neuronx_distributed_inference.models.diffusers.flux.application import (
    NeuronFluxApplication, create_flux_config, get_flux_parallelism_config,
)
from neuronx_distributed_inference.models.diffusers.flux.lora import build_flux_lora_config

lora_config = build_flux_lora_config(
    max_loras=2,          # adapters resident on device, on top of the base slot
    max_lora_rank=64,     # the largest rank the slots are sized for
    lora_ckpt_paths={
        "realism": "/adapters/xlabs-realism",       # a directory, or
        "superreal": "/adapters/super-realism.safetensors",   # a single file
    },
)

tp = 4
world_size = get_flux_parallelism_config(tp)
clip_c, t5_c, backbone_c, vae_c = create_flux_config(
    "/models/FLUX.1-dev", world_size, tp, torch.bfloat16, 1024, 1024,
    lora_config=lora_config,
)
app = NeuronFluxApplication(
    model_path="/models/FLUX.1-dev",
    text_encoder_config=clip_c, text_encoder2_config=t5_c,
    backbone_config=backbone_c, decoder_config=vae_c,
    height=1024, width=1024,
)
app.compile("/compiled/flux-lora")
app.load("/compiled/flux-lora")

app.set_lora_adapters("realism")
image = app(prompt="a portrait of a fisherman", num_inference_steps=20).images[0]

app.set_lora_adapters(None)        # back to the base model
```

Every adapter declared here is resident on device, whether or not `max_loras` says so:
`LoraServingConfig` raises `max_loras` to cover the declared checkpoints and logs
`Setting the number of LoRA adapters in HBM to N`. So `max_loras` is a floor for
declared adapters, not a cap — to serve more adapters than fit on device, declare the
ones that fit and add the rest with `add_lora_adapter()`.

Only the backbone is adapted. LoRA weights aimed at the text encoders are ignored
with a warning; the CLIP and T5 graphs are unchanged.

## Loading adapters at runtime

Set `dynamic_multi_lora=True` and adapters no longer have to be known when the
model is built. `max_cpu_loras` then sizes a host-memory tier that backs the
device slots:

```python
lora_config = build_flux_lora_config(
    max_loras=1,                  # one adapter on device at a time
    max_cpu_loras=4,              # four kept ready in host memory -- see Memory below
    max_lora_rank=64,
    dynamic_multi_lora=True,
    lora_ckpt_paths={"realism": "/adapters/xlabs-realism"},   # optional
)
...
app.load("/compiled/flux-lora")

app.add_lora_adapter("superreal", "/adapters/super-realism.safetensors")
app.set_lora_adapters("superreal")     # swapped into the device slot on first use
image = app(prompt=..., num_inference_steps=20).images[0]

app.list_lora_adapters()               # {'realism', 'superreal'}
```

Nothing is recompiled and nothing is reloaded: the graph takes the slot index as
an input, so a new adapter only has to be read, sharded, and copied into a slot.
An adapter whose rank exceeds `max_lora_rank` is rejected rather than truncated —
the slots were compiled at that width.

## How a call selects its adapter

Three ways:

| | how | when to use |
|---|---|---|
| `adapter_ids=["name"]` | per call, one name per batch item | calling the backbone directly |
| `set_lora_adapters("name")` | sticky, used by calls that pass nothing | full pipeline runs — `FluxPipeline.__call__` has no adapter argument and does not forward extra kwargs |
| `adapter_ids=torch.tensor([1])` | an already-resolved device slot | tests, or a caller managing slots itself; bounds-checked against `max_loras` |

A single name is broadcast across the batch, so CFG parallel (batch 2, one
request) needs only one name.

## What a LoRA costs at request time

Three tiers, measured on trn2 with TP=4, `max_lora_rank=64`, one 256px backbone
step on FLUX.1-dev:

| | cost | when |
|---|---|---|
| device hit | free — 75.7 ms/step against the base model's 74.1 ms, within the run-to-run spread of both | the adapter is already in a slot, whether or not the previous request used a different one |
| host → device swap | **+1.84 s, once per request** | the adapter is in host memory but not in a slot |
| disk → host | +1.1 s, once per adapter | `add_lora_adapter()`, or a request for an adapter the host tier has evicted |

A device hit is genuinely free: the slot index is a graph input, so selecting an
adapter costs nothing at all. Everything else is data movement.

Those three tiers are measured at `max_loras=1`, where only one adapter can be on
device at a time — so the swap row is the cost of an adapter *not being resident*,
not the cost of choosing between adapters. With `max_loras=2` and both declared at
build time, alternating between them costs nothing either (same 256px backbone step,
10 calls each):

| | `max_loras=2`, both resident | `max_loras=1`, one slot |
|---|---|---|
| repeating one adapter | 74.2 ms | 75.1 ms |
| alternating two adapters | 73.1 ms | 1844.8 ms |
| no adapter (slot 0) | 74.2 ms | 75.2 ms |
| **alternating − repeating** | **−1.1 ms** | **+1769.8 ms** |

With both resident, the difference between alternating and repeating is about a
millisecond on a 74 ms step and it changes sign between runs (a second run gave
+0.4 ms), which is what "free" looks like when it is measured rather than asserted: at
ten calls per pattern there is nothing there to resolve. Whether *any* adapter is
active is equally invisible (+0.1 ms here, −0.2 ms at `max_loras=1`). With one slot the
same request pattern spends 1.77 s per call moving weights.

Reproduce with:

```bash
python examples/benchmark_flux_lora.py \
    -c /shared/flux/FLUX.1-dev/ --compile_workdir /tmp/flux-lora-2/ \
    --lora xlabs=/adapters/xlabs-realism \
    --lora kohya=/adapters/super-realism.safetensors \
    --max-lora-rank 64
```

The same command with `--max-loras 1` (and its own `--compile_workdir`, since that
changes the graph) measures the right-hand column: one device slot, so the alternating
pattern has to swap every request. The script declares only the adapters that fit and
adds the rest with `add_lora_adapter()`, because a declared adapter is always
resident.

So the 1.84 s swap is a configuration outcome, not a floor: raise `max_loras` until the
hot adapters stay resident and per-request adapter selection disappears from the
latency, at 634 MB per slot per core.

The swap is expensive, and worth understanding before sizing a deployment. It
rewrites the whole slot — 634 MB per core, 2.5 GB across four cores — as ~4300
separate small copies, one per adapted module per rank. Two things follow:

- **The cost follows `max_lora_rank`, not the adapter's own rank.** A 22 MB
  rank-16 adapter and a 585 MB rank-64 one swap in for the same 1.84 s, because
  both are padded into a rank-64 slot and the whole slot is rewritten. Lowering the
  slot width helps, but less than the memory saving suggests — the same adapter,
  with only the compiled slot width changed:

  | `max_lora_rank` | memory per slot per core | swap |
  |---|---|---|
  | 16 | 158.5 MB | 940 ms |
  | 32 | 317 MB | 1322 ms |
  | 64 | 634 MB | 1754 ms |

  Memory is exactly linear in the rank; the swap is not, because roughly 0.7 s of
  it is the fixed cost of issuing ~4300 copies regardless of how big each one is.
  So rank 16 instead of 64 buys 4× the memory but only ~1.9× the swap.
- **It is paid once per request, not once per denoising step**, so its share of a
  request falls as the request gets longer while its absolute cost stays put.
  Full 1024px requests, medians, `max_loras=1` so a different adapter each request
  always misses:

  | steps | base model | adapter resident | adapter swapped in | swap | swap's share |
  |---|---|---|---|---|---|
  | 4 | 1.58 s | 1.57 s | 3.37 s | 1.80 s | 53% |
  | 20 | 6.39 s | 6.38 s | 8.15 s | 1.77 s | 22% |
  | 28 | 8.80 s | 8.81 s | 10.58 s | 1.77 s | 17% |

  Note the second and third columns: an adapter that is already in a slot costs
  nothing measurable against the base model, at any step count. Raising
  `max_loras` so the hot adapters stay resident is therefore the whole
  optimisation, at 634 MB of HBM per core per slot.

Falling back to disk on top of that re-reads and re-shards the adapter, so keep
`max_cpu_loras` large enough for the working set. Eviction is LRU by default
(`eviction_policy="lfu"` is also available).

## Memory

Slots are allocated at `max_lora_rank` regardless of what the adapters actually
use, and the cost is per device slot per core. At rank 64 on FLUX.1-dev it is
634 MB per slot per Neuron core, plus one slot's worth for the staging buffer;
the runtime logs the total at build time (below: one adapter and the base slot, so
three slots' worth):

```
WARNING The memory footprint for LoRA adapters on each Neuron core is 1902.375 MB
```

It is exactly linear in both `max_loras` and `max_lora_rank`: 158.5 MB per slot at
rank 16, 317 MB at 32, 634 MB at 64. The host tier costs the same per adapter, in
host memory, times `tp_degree`.

## Samples

One compiled model, one seed per prompt, one device slot; `kohya` was added at
runtime after the model was loaded, and every row swapped adapters through that one
slot. 1024px, 20 steps, FLUX.1-dev at TP=4, shown here at 384px.

| | base model | [XLabs realism](https://huggingface.co/XLabs-AI/flux-RealismLora) (r=16) | [kohya super-realism](https://huggingface.co/strangerzonehf/Flux-Super-Realism-LoRA) (r=64) |
|---|---|---|---|
| *fisherman* | ![](flux_lora_samples/fisherman_base.png) | ![](flux_lora_samples/fisherman_xlabs.png) | ![](flux_lora_samples/fisherman_kohya.png) |
| *cafe* | ![](flux_lora_samples/cafe_base.png) | ![](flux_lora_samples/cafe_xlabs.png) | ![](flux_lora_samples/cafe_kohya.png) |
| *workshop* | ![](flux_lora_samples/workshop_base.png) | ![](flux_lora_samples/workshop_xlabs.png) | ![](flux_lora_samples/workshop_kohya.png) |

<sub>*fisherman*: a close-up portrait photograph of an elderly fisherman mending a
net, weathered hands, overcast harbour light — *cafe*: a photograph of a woman
reading a paperback in a busy cafe window, afternoon sun, shallow depth of field —
*workshop*: a photograph of a cluttered watchmaker's workbench, brass tools and
loupe, single desk lamp</sub>

Each adapter has a consistent effect across prompts, which is the point of the
comparison: it is the adapter showing up, not prompt-to-prompt variation.

The images are also reproducible across swaps. Generating the base image again
after both adapters had passed through the device slot reproduced the first one
byte for byte, as did re-running an adapter — including in a separate process with
a different load and swap history.

## Accuracy

Each adapter is checked against CPU diffusers with the same adapter loaded, by
comparing one backbone step (velocity) on identical inputs:

| slot | vs CPU base | vs CPU xlabs | vs CPU kohya |
|---|---|---|---|
| base | **0.99923** | 0.98942 | 0.97726 |
| xlabs adapter | 0.98918 | **0.99949** | 0.97403 |
| kohya adapter | 0.97893 | 0.97529 | **0.99960** |

Cosine similarity; the reference is float32 on CPU, so ~0.9992 is the bfloat16
noise floor and every slot sits at it against its own adapter. An adapter is
bit-identical after being evicted from the device cache and swapped back in.

Run the tests with:

```bash
pytest test/unit/models/flux/test_flux_lora.py
FLUX_LORA_TEST_CHECKPOINT=/models/FLUX.1-dev pytest test/unit/models/flux/test_flux_lora.py
```

The second form adds the tests that need a checkpoint; neither needs a device.

## Limits

- The text encoders are not adapted.
- One `max_lora_rank` for all slots: a rank-8 adapter is zero-padded to the slot
  width and costs the same memory as a rank-64 one.
- `lora_shard_linear_layer` must stay `False`. The single-stream block's
  `proj_out` halves defer their all-reduce to the block, which is only exact when
  `lora_B` is replicated across ranks.
- diffusers' single `proj_out` is split across NxDI's `proj_out_attn` and
  `proj_out_mlp`. The split is exact, not an approximation: `lora_A` is cut along
  its input dimension and `lora_B` is used by both halves, so their sum
  reproduces the original product.
