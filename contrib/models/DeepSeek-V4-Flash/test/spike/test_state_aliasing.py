#!/usr/bin/env python
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Probe device-resident state across calls, on a toy model, before wiring decode.

Decode needs the KV cache to survive between invocations *on the device*. If it
does not, every step has to ship the whole cache host->device->host, which for
this model is 43 layers x (window + compressed) x 512 dims per step.

NxD's mechanism is the second element of the tuple the trace factory returns:
`(model, aliases)`, where aliases maps a *buffer tensor* to the index of the
output that carries its new value (NxDI does exactly this in
model_wrapper.py: `aliases[model.kv_mgr.past_key_values[i]] = n_out + i`).
The runtime then writes that output straight back over the input buffer.

Two things that are not obvious from the docs and that this probe pins down:
  1. Does an in-place mutation of a registered buffer (`buf.index_copy_(...)`)
     survive tracing, i.e. does reading `self.buf` after the mutation yield the
     updated value rather than the original parameter?
  2. Does the alias actually persist across two separate calls to the traced
     model, so that call N+1 sees what call N wrote?

Toy model, 1 layer, tp=2, a few KB of state — cheap enough to iterate on.

Run:
    source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
    cd contrib/models/DeepSeek-V4-Flash
    python test/spike/test_state_aliasing.py
"""

import os
import shutil
import sys
import traceback

import torch

WIN = 8
DIM = 4
TP = 2
WORKDIR = "/mnt/data/tmp/alias_probe_ws"
OUTDIR = "/mnt/data/tmp/alias_probe_out"


class Counter(torch.nn.Module):
    """Writes `value` into ring slot (pos % WIN) and returns the whole state.

    Deliberately mirrors what Attention.forward does in decode:
      self.kv_cache[:, start_pos % win] = kv.squeeze(1)
    with start_pos a runtime tensor rather than a Python int.

    The state is an nn.Parameter, not a register_buffer, and that is load
    bearing — see factory() below.
    """

    def __init__(self):
        super().__init__()
        self.cache = torch.nn.Parameter(torch.zeros(1, WIN, DIM),
                                        requires_grad=False)

    def forward(self, value: torch.Tensor, pos: torch.Tensor):
        # index_copy demands a Long index on XLA even though start_pos is
        # int32 (that is what jit.trace takes as a scalar input).
        slot = (pos % WIN).reshape(1).long()
        new_cache = self.cache.index_copy(1, slot, value)
        # Sum so the probe has a cheap scalar to eyeball, plus the new state as
        # the aliased output.
        return new_cache.sum().reshape(1), new_cache


def factory():
    m = Counter().eval()
    # aliases: {state tensor -> index of the output carrying its new value}.
    # Output 0 is the sum, output 1 is the new cache.
    #
    # Two non-obvious requirements, both found the hard way:
    #
    # 1. The state must be an nn.Parameter, NOT a register_buffer. torch_neuronx
    #    resolves alias keys by scanning `func.named_parameters()` and matching
    #    on `.data_ptr()` (hlo_conversion.py: "for name, parameter in
    #    func.named_parameters(): for inp_param in input_output_aliases: if
    #    inp_param.data_ptr() == parameter.data_ptr()"). Buffers are never
    #    scanned, so a buffer key silently matches nothing and the alias is
    #    dropped — no error, just a graph whose state resets every call.
    #
    # 2. The key must stay a CPU tensor. NxD pickles this dict back to the
    #    parent through mp_q, and torch_neuronx then does
    #    `initial_states = tuple(input_output_aliases.keys())`, so the parent
    #    rebuilds every key. An XLA tensor key makes that rebuild call
    #    _rebuild_device_tensor_from_cpu_tensor in a process that owns no
    #    device:  RuntimeError: Init: ... [NRT_FAILURE] status_code=1, and the
    #    whole rank pool dies with BrokenProcessPool. Since the key is only
    #    ever compared by data_ptr, a CPU tensor is all that is needed — and
    #    the factory runs before the model is moved to device, so
    #    m.cache is still on CPU here.
    return m, {m.cache: 1}


def example_inputs():
    return (torch.zeros(1, 1, DIM), torch.zeros((), dtype=torch.int32))


def main():
    for d in (WORKDIR, OUTDIR):
        shutil.rmtree(d, ignore_errors=True)
        os.makedirs(d, exist_ok=True)

    from neuronx_distributed.trace import parallel_model_trace

    print("tracing toy model with aliased state ...")
    try:
        model = parallel_model_trace(
            factory,
            example_inputs(),
            tp_degree=TP,
            compiler_workdir=WORKDIR,
            inline_weights_to_neff=False,
        )
    except Exception as e:
        print(f"[FAIL] trace failed: {type(e).__name__}: {e}")
        traceback.print_exc(limit=6)
        return 1
    print("traced ok")

    print("\n" + "-" * 66)
    print("calling repeatedly; each call writes 1.0 into a new ring slot")
    print("if state is device-resident, the sum grows: 4, 8, 12, ...")
    print("if it resets every call, the sum stays 4")
    print("-" * 66)

    sums = []
    for step in range(5):
        val = torch.ones(1, 1, DIM)
        pos = torch.tensor(step, dtype=torch.int32)
        out = model(val, pos)
        s = out[0] if isinstance(out, (tuple, list)) else out
        total = float(s.flatten()[0])
        sums.append(total)
        print(f"  step {step}: pos={step}  sum={total}")

    expected_persist = [DIM * (i + 1) for i in range(5)]
    expected_reset = [DIM] * 5
    print()
    if sums == expected_persist:
        print(f"PERSISTENT: sums {sums} == {expected_persist}")
        print("state survives across calls on device; decode can alias its KV cache.")
        rc = 0
    elif sums == expected_reset:
        print(f"NOT PERSISTENT: sums {sums} == {expected_reset}")
        print("the buffer is re-initialised every call; decode would have to")
        print("thread the cache through as an explicit input/output instead.")
        rc = 1
    else:
        print(f"UNEXPECTED: sums {sums}")
        print(f"  persist would be {expected_persist}, reset would be {expected_reset}")
        rc = 1

    print("\nalso checking the returned state tensor itself:")
    out = model(torch.ones(1, 1, DIM), torch.tensor(0, dtype=torch.int32))
    if isinstance(out, (tuple, list)) and len(out) > 1:
        print(f"  output[1] shape={tuple(out[1].shape)} sum={float(out[1].sum())}")
    else:
        print(f"  single output only: {type(out)}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
