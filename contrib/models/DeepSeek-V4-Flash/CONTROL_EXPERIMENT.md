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


## Follow-up: static comparison of the two artifacts

Compared the working Qwen3 artifact against ours without occupying the device.
Both are `NxDModel`s with the same structure; every dispatch component in ours
checks out individually:

| | Qwen3 (works) | ours |
|---|---|---|
| `models` keys | `context_encoding_model`, `token_generation_model` | `prefill`, `decode` |
| `flattener_map` keys | `..._0` for each | `prefill_0`, `decode_0` |
| `input_shape_map` | 7 inputs per graph | 2 inputs per graph |
| `state_initializer` | present | present |
| `weight_loader` | **present** | **None** |

Verified directly on our artifact:

* `router([ids, pos])` returns `('prefill', 0)` — routing is correct. (An earlier
  Python-level string comparison suggested the shape keys could not match; that was
  wrong. TorchScript renders `tensor.shape` as a plain list, so they do match.)
* `flattener_map['prefill_0']([ids, pos])` returns the 2 expected tensors with the
  right shapes and dtypes.
* `models` keys equal what the router returns, so `NxDModel.forward`'s
  match-by-name loops both fire.
* A **freshly loaded** artifact raises "not initialized" on forward, so that guard
  works; after our `initialize()` it does not raise, i.e. the SPMD models really do
  report initialised.

So: routing, flattening, naming, state binding, weight shape and weight naming are
all correct, the guard confirms initialisation happened, and the NEFF still
produces nothing. `weight_loader` is the one structural difference left, and it is
None here only because no `priority_model_idx` is set — which also means
`initialize()` takes the plain `_parallel_load` path rather than the layout-aware
one.

Two inputs differences also stand out and are cheap to try next:
`input_ids` is int32 in NxDI's own input generator (ours is int64), and NxDI passes
seven inputs including `seq_ids` and `sampling_params`. Neither should matter given
routing succeeds, but "should not matter" has been wrong twice already here.


## Quantitative evidence that initialize() does not transfer

The strongest signal yet, and it is arithmetic rather than another hypothesis.
Comparing our `initialize()` against the working control's weight load:

| | per-rank weights | ranks | time |
|---|---|---|---|
| Qwen3-1.7B (works) | ~0.42 GB | 8 | **12.0 s** |
| DSV4 5 layers (zeros) | ~2.06 GB | 32 | **5.2 s** |

We hand over 4.9x the data per rank and it completes in 43% of the time — an ~11x
apparent transfer-rate difference. Combined with `neuron-monitor` reporting zero
runtimes during the subsequent call, the conclusion is that our `initialize()`
returns without moving weights to the device, despite:

* accepting the dict without error
* validating shapes (proven: a wrong-shape dict from `shard_checkpoint()` raises
  "Incorrect tensor shape at ... received 129280 4096, expected 4040 4096")
* reconciling all 207 graph weight names (0 missing)

So the weights are correctly shaped and correctly named, the call validates them,
and it still does not put them on the device. That is a narrower and more testable
statement than "the graph does not execute", and it is where the next session
should start.

Also tested and ruled out this round: `int32` vs `int64` `input_ids`, matching
NxDI's own input generator (model_wrapper.py:245). No change.


## Breakthrough: it was never a loading problem — the graph computes NaN

The decisive experiment was finally the cheap one: run **the same graph on CPU**
with the same weights and inputs the device gets.

```
CPU logits: absmax nan  std nan  finite False
```

So `DSV4Model.forward` produces **NaN**, not zeros. The device reports non-finite
as zero, which is why every device-side symptom looked like "the NEFF never ran".
It ran the whole time and computed garbage.

That also retires the entire loading-boundary investigation. For the record, all of
these were verified *correct* and none was the cause:

* weights on device (`privateuseone:0`, shape (4040, 4096) i.e. properly sharded)
* state on device (32 ranks x 17 states, keyed `kv_mgr.past_key_values.N`)
* `is_initialized()` == **True** after our `initialize()` (False on a fresh load,
  and `forward` correctly raises there — so the guard works)
* all 207 graph weight names reconcile, 0 missing
* router returns `('prefill', 0)`; flattener returns the 2 expected tensors
* `_parallel_load` moves tensors to device for any key form

Two of my earlier conclusions were wrong and are corrected here:

* **"initialize() never transfers weights."** It does. The 5.2 s vs Qwen3's 12.0 s
  timing argument was not evidence of a missing transfer — inspecting
  `nxd_model.weights` after the call shows real tensors on `privateuseone:0`.
* **"the graph does not execute."** It executes.

## Where the NaN is, so far

Narrowed by bisection at 3 layers, real weights, TP=32, under `mock_distributed`:

| probe | result |
|---|---|
| embed | finite |
| layers 0,1,2 individually, in sequence | finite (absmax 2.06 / 3.19 / 2.92) |
| `hc_head`, `norm`, `get_logits` called step by step | finite (absmax 3.11 / 2.52) |
| the head's `all_gather` + concat, reproduced by hand | finite, shape (1, 129280) |
| **stack + head in one pass** (`inner.head(...)`, raw `Transformer.forward`, and `DSV4Model.forward`) | **NaN, all three** |

So every component is finite in isolation and the composition is not, which points
at state left behind by one pass being consumed by the next — the managed sink is
rebuilt per call, and a probe that rebuilds it between the stack and the head gets
a different (clean) ring than a single pass does. That is the next thing to test:
whether the head is reading ring state the stack has already rolled.


## Correction: the CPU NaN was a simulation artifact, not a bug

The previous section claimed "the graph computes NaN" and treated that as the root
cause. **That conclusion was wrong**, and the experiment that shows it is a TP
sweep on CPU with everything else held fixed:

| | logits | ids in rank 0's vocab shard |
|---|---|---|
| **TP=1** | **finite**, absmax 6.05 | 128/128 |
| TP=32 | NaN | **5/128** |

Under `mock_distributed` every collective is a no-op. At TP=32 rank 0 owns 1/32 of
the vocab and 1/32 of every column-parallel weight, so 123 of 128 embedding rows
are legitimately zero on that rank and every `all_reduce` that should have summed
32 partial results contributes only one. RMSNorm then normalises by a near-zero
mean, which is where the inf/NaN comes from.

On the real device the collectives are real, so this specific NaN cannot occur
there. Running the model at TP=1 on CPU is finite, which is the correct
single-process check and is what the five CPU gates already do.

Two things follow:

1. **The graph math is not the problem.** TP=1 CPU is finite; the five CPU gates
   are bit-identical to the validated patched-HF path. Nothing points at the model.
2. **The device zeros still need an explanation**, and "the graph computes NaN" is
   no longer it. What remains unexplained: `is_initialized()` is True, weights and
   state are verified on device with correct shapes, routing and flattening are
   correct, and the call returns exact zeros in 0.03 s with zero neuron-monitor
   runtimes.

Also worth recording as a methodological note, since it cost time twice: a
single-process CPU run of a TP>1 model under `mock_distributed` is **not** a valid
correctness reference. It was used here to "prove" NaN and earlier to reason about
transfer timings. Any CPU parity check for this port has to run at TP=1.


## The graph DOES execute — only the logits output is dead

Characterising the device output properly settled what "all-zero logits" means:

```
raw dtype=float32  n_zero=129280/129280  n_nan=0  n_inf=0  unique=1
state kv_mgr.past_key_values.0: nonzero=50688/65536
state kv_mgr.past_key_values.1: nonzero=64512/65536
-> state WAS written by the graph
```

So the NEFF ran, did real work, and wrote the aliased KV state. Exactly one thing
is wrong: output 0 (the logits) comes back as exact zeros — one unique value, no
NaN, no inf, i.e. an untouched output buffer rather than a computed-then-broken
value.

This reframes everything. It is not a loading problem (weights and state verified
on device), not a dispatch problem (router/flattener verified), not a NaN problem
(that was a TP=32-on-one-rank artifact), and not "the graph does not execute". It is
specifically: **the logits output is not written by the compiled graph.**

Consistent with that reading:

* `nxd_model.forward` returns a single `Tensor`, not a tuple. That is correct —
  `ModelBuilder` passes `output_aliased_tensor=False`, so `return_aliases=False`
  and the runtime strips all 17 aliased state outputs, returning only output 0.
* the alias map is verified to claim indices 1..17 only, leaving output 0 free.

So output 0 is declared, unaliased, and never written.

### Ruled out for the dead output

* **The prefill keep-alive multiplied by zero.** `logits + position_ids.sum() * 0`
  is a no-op mathematically but is also exactly what a compiler constant-folds
  (`x*0 -> 0`, `logits+0 -> logits`), which would both defeat the input-liveness
  trick it existed for and plausibly kill the output. Replaced with
  `position_ids[0,0] - position_ids[0,0]`, which is provably zero for
  `arange(seqlen)` but not a literal the compiler can fold without evaluating the
  input. **No change** — still all zeros. So this was a real latent bug worth
  fixing, but not the cause.
* TP=1 as a way to remove collectives from the picture: not possible, one core
  cannot hold the unsharded model (`Failed to allocate nrt tensor`).

### Next

The remaining candidates all concern how output 0 is produced rather than
consumed: whether `DSV4Model.forward` returning `tuple(outs)` with 18 entries maps
output 0 to the logits the packer expects, and whether the packer built at trace
time (from the *prefill* graph's example run) agrees with what decode returns. Both
are inspectable in the HLO's output signature without another device run.


## Metaneff confirms output 0 is correctly declared

Static inspection of `prefill/_tp0_bk0/metaneff.pb`, no device time:

```
return_aliases: False
n input_tensors:  226
n output_tensors:  18
output_aliases_to: {1: 2, 2: 3, ..., 17: 18}

out[0]  output0  dims=[1, 129280]  type=1 (float32)
out[1]  output1  dims=[1, 128, 512] type=14 (bf16)
...
out[17] output17 dims=[1, 8, 256]  type=1
```

Everything here is right. Output 0 is the logits, correct shape and dtype, and it
is **not** in `output_aliases_to` — the 17 aliases map outputs 1..17 to inputs
2..18, which is exactly what `hlo_conversion.py:490` generates
(`output_idx -> i + n_user_inputs`, with 2 user inputs). `return_aliases: False`
explains why `forward` returns a bare Tensor rather than a tuple: the runtime
strips the aliased outputs and hands back only output 0.

So the contract is correct end to end and output 0 is simply never written.

Two more hypotheses tested and eliminated:

* **Output slot 0 being special.** Returned `logits` twice with the aliases shifted
  to start at 2. The compiler rejected it outright — "Trying to set up alias at
  {18}", i.e. out of range for 19 outputs. Useful negative result: aliasing is
  strictly index-checked, so the normal 1..17 configuration is definitely valid.
* **`logits` sharing storage with an aliased buffer.** An aliased output is written
  in place over its input, so if output 0 aliased the same storage as a cache the
  runtime could clobber it — which would match "state written, logits untouched"
  precisely. Added `logits.clone()` to force a fresh allocation (`.contiguous()`
  would be a no-op here). **No change.**

### Status

Everything checkable is correct: weights and state on device, `is_initialized()`
True, router and flattener verified, alias map verified against the generator, the
metaneff signature verified, the graph demonstrably executing and writing 17/17
state buffers. Output 0 alone comes back as an untouched buffer, and the cause is
not any of: loading, dispatch, aliasing indices, alias storage overlap, output
position, NaN, dead-code elimination via `*0`, weight naming, weight layout,
example-input degeneracy, compiler version, or the compile cache.

The next thing I would do is dump the HLO text for the prefill graph and read what
instruction feeds root tuple element 0 — that is the one remaining place the answer
can be, and it is static.
