# Control experiment: does NxDI work at all on this box and SDK?

The NxDI-native port compiles cleanly and binds correctly but returns all-zero
logits with no NEFF on any core. Six hypotheses had been ruled out by direct test
without finding the cause, so the question worth settling was whether the fault is
in this port's integration or in the environment.

## Method

Run NxDI's own `inference_demo` on a small model with a native NxDI
implementation, on the same box, same venv, same compiler
(`neuronx-cc 2.26.6360.0+6f180f47`). Nothing from this port is involved.

```bash
inference_demo --model-type qwen3 --task-type causal-lm run \
  --model-path /mnt/nvme/models/Qwen3-1.7B \
  --compiled-model-path /mnt/nvme/artifacts/qwen3_ctrl \
  --torch-dtype bfloat16 --tp-degree 8 --batch-size 1 --seq-len 512 \
  --pad-token-id 151643 --top-k 1 \
  --prompt "It is well known that the capital city of France is"
```

## Result: the environment is fine

```
Output 0: It is well known that the capital city of France is Paris.
          The capital city of the United States is Washington, D.C.  The ca...
```

Correct output, coherent continuation. Load timings:

```
Sharding weights on load...
Sharding weights for ranks: 0...7
Done Sharding weights in 0.82 s
Finished weights loading in 12.04 s
Warming up the model.
Warmup completed in 0.30 s
Total model loading time: 12.60 s
```

So on this box and this SDK, NxDI's own end-to-end path works. Two consequences:

1. **The all-zero failure is this port's integration, not the environment.** Every
   remaining hypothesis has to be about what this port does differently.
2. **`neuronx-cc 2.26.6360` is not categorically broken.** MiMo-V2.5-Pro reports it
   silently miscompiling *their* model, and that finding stands, but it does not
   generalise: Qwen3-1.7B compiles and runs correctly on it here. So 2.26 is not a
   blanket explanation for this port's problems either.

## What the working path does that this port's driver does not

`NeuronBaseForCausalLM.load()` (application_base.py:318) is a sequence, not a
single call:

```
set_env_vars(neuron_config)
traced_model = torch.jit.load(path + COMPILED_MODEL_FILE_NAME)
load_weights(path, ...)          # -> weights = builder.shard_checkpoint()
                                 #    then nxd_model.initialize(weights, rank)
self.to(torch_dtype)
for model_wrapper in self.models: model_wrapper.model = self.traced_model
warmup()
```

The driver in `run_dsv4_device.py` calls `nxd_model.initialize()` with a
hand-assembled per-rank dict and then invokes the model. Differences still
untested, in the order they are worth trying:

1. **`weights` comes from `builder.shard_checkpoint()`, not a hand-built dict.**
   That path runs `preprocess_checkpoint`, `cast_weights`, and — importantly —
   the weight *layout transform* (`_prepare_weight_layout_transform_model`,
   model_builder.py:1090), which `initialize()` then applies via
   `self.weight_loader.forward(checkpoint, False)` rather than
   `torch.ops.neuron._parallel_load`. A dict that never went through the layout
   transform may bind by name and still be in the wrong physical layout for the
   NEFF. This is the most likely remaining cause and it is consistent with every
   observation: names reconcile, no error is raised, and the graph produces
   nothing.
2. `model_wrapper.model = traced_model` is never set here (this driver has no
   ModelWrapper objects at all).
3. `warmup()` is never called.

## Cost note

Qwen3-1.7B: 3.8 GB download, ~6 min to compile and run at TP=8. Cheap enough that
this should have been the *first* diagnostic rather than the seventh — it answers
"is it me or the box" in one run, which six model-side experiments could not.
