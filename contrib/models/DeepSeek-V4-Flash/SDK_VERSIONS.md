# Neuron SDK / compiler versions for this port

Prompted by the MiMo-V2.5-Pro contrib branch, which documents a **silent
miscompile** in exactly the compiler this box ships. Worth checking before
blaming model code, and worth recording either way.

## Installed here

```
neuronx-cc                    2.26.6360.0+6f180f47      <- the version MiMo warns about
libneuronxla                  2.2.17544.0+fb9962bf
neuronx-distributed           0.19.28492+435aae2b
neuronx-distributed-inference 0.10.18399+ed62453e
torch                         2.9.1
torch-neuronx                 2.9.0.2.15.32035+de43f57c
```

All five venvs under `/opt` carry the same compiler build, so there is nothing to
A/B locally without installing one.

## What MiMo-V2.5-Pro reports

`whn09/neuronx-distributed-inference`, branch `contrib/MiMo-V2.5-Pro`:

> **`neuronx-cc 2.26.6360.0+6f180f47` (SDK 2.31.0) silently produces a numerically
> wrong NEFF for this model. Generated text is garbage. Compilation succeeds with
> exit 0 and no warning.** Use `2.25.3371.0+f524f7f8` (SDK 2.30.0) or
> `2.24.8799.0+6f62ff7c`.

Their evidence is a byte-identical HLO compiled by three compilers: 2.24 and 2.25
both produce correct text, 2.26 produces `'., the the,,,,1.1.1.1.  the the the'`.
They ruled out the narwhal backend and the allreduce-buffer default; the remaining
2.26-only difference they can see is a force-enabled
`--internal-disable-fma-on-ios`, which they call a lead rather than a diagnosis.

Two of their notes matter for *this* port regardless of the miscompile:

* **Short prompts do not discriminate.** Their broken build still answers
  `"The capital of France is"` plausibly; ≥200-token prompts are needed to expose
  the corruption. Our own accuracy evidence is a 128-token prefill and GSM8K
  prompts averaging 84 tokens — both below that threshold. So a 2.26 miscompile of
  the same kind would not necessarily have shown up in what we have measured.
* **The compiler cache will hand back a bad NEFF.** Any version experiment has to
  clear `/var/tmp/neuron-compile-cache` first.

## Tested here: does 2.25 help?

Installed 2.25 into a shadow venv and pointed `PYTHONPATH` at it, per MiMo's
recipe (do **not** prepend its `bin` to `PATH` — `neuronx-cc` is invoked in-process
by `libneuronxla`, and that venv has no torch):

```bash
python3 -m venv --system-site-packages /mnt/nvme/venv_cc225
/mnt/nvme/venv_cc225/bin/pip install --no-deps 'neuronx-cc==2.25.3371.0+f524f7f8'
export PYTHONPATH=/mnt/nvme/venv_cc225/lib/python3.12/site-packages:$PYTHONPATH
python3 -c "import neuronxcc; print(neuronxcc.__version__)"   # 2.25.3371.0+f524f7f8
rm -rf /var/tmp/neuron-compile-cache/*
```

Result: **2.25 is not a fix for this port — it trades one failure for another.**

| | 2.26.6360 (stock) | 2.25.3371 |
|---|---|---|
| joint HLO generation, both graphs | ok | ok |
| prefill NEFF, 43-layer-shaped graph | ok | **internal compiler error** |

2.25 fails during Tensorizer on the prefill graph:

```
[INTERNAL_ERROR] [NCC_IBCG901] BIRCodeGenLoop assertion
  on _JointWrapper/Transformer/Block[.2]/aten.div.Tensor/aten__div_divide.3146
  Transformation error ... Assertion failed: False
```

which is a hard error, not a silent wrong answer. The traced op sits in
`prefill_patches.py:241`'s neighbourhood (`sparse_attn` over the concatenated
window + compressed KV), i.e. the sparse path MiMo does not exercise.

So: the two compilers fail differently on this model, and neither is known-good
for it yet. 2.24 is untested here and is the obvious next experiment if the
current path stalls again.

## Correction to an earlier claim of mine

I previously concluded that `torch.where` "does not lower inside ModelBuilder's
`generate_hlo`" and replaced it with an arithmetic select. **That scope was too
broad.** Isolated `generate_hlo` reproducers all compile fine:

* `where(j <= pos, j, full_like(j, -1))` alone — ok
* the same plus `_batch`'s broadcast-add — ok
* the same plus an aliased state Parameter as a second output — ok

But reverting the workaround in the real joint graph *does* reproduce
`size of tensor a (128) must match tensor b (0)`. So something about the full
graph triggers it that the minimal cases do not, and the actual trigger is still
unidentified. The workaround stays because it is verified equivalent
(`keep * j - (1 - keep)`, checked against `where` for pos in 0/1/5/127/200) and it
makes the graph compile — but it is a workaround for an *unlocated* problem, not
for a general `where` defect, and the comment in `decode_patches.py` now says so.

## Standing guidance

1. Do not treat a plausible short-prompt answer as evidence of a correct build.
   Any accuracy gate for this model should use a ≥200-token prompt, per MiMo's
   detection note. The existing GSM8K harness averages 84-token prompts.
2. Clear the compile cache before any version comparison.
3. When a graph-level failure resists explanation, check the compiler version
   against MiMo's table before assuming the model code is at fault — that would
   have saved time here.
