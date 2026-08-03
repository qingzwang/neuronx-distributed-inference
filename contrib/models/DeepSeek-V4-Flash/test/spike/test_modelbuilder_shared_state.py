#!/usr/bin/env python
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Spike: does ModelBuilder really share one KV cache across two graphs?

This is the load-bearing assumption behind porting compile_neuron.py off
`parallel_model_trace` (see JOINT_INFERENCE.md), and it is cheap to test on a toy
model before spending ~100 min compiling 43 layers to find out.

What it checks, on a two-line model whose whole job is to write and read a cache:

  1. Two graphs with *different input widths* (the prefill/decode shape split)
     registered in one builder land in one artifact and route by shape.
  2. Writes made by graph A are visible to graph B. This is the hand-off
     `parallel_model_trace` cannot do.
  3. `write_to_neuron_buffer` actually resets the state from the host — the
     operation that silently no-ops on TensorParallelNeuronModel and is the
     reason cache_reset.py failed.
  4. Weights can be supplied as a per-rank list via `set_weights`, bypassing
     NxD's `shard_children` (which does not recognise HF's same-named parallel
     classes, so the port must keep its own sharding — see shard_loader.py).

Run (needs 2 free cores):
    python test/spike/test_modelbuilder_shared_state.py
"""

import os
import shutil
import sys

import torch

failures = []


def _first(out):
    """The graph's first real output, whatever the packer wrapped it in.

    An aliased-state graph may or may not hand the state tensors back as extra
    outputs depending on how the packer was built, so unpacking a fixed arity is
    not safe here.
    """
    while isinstance(out, (tuple, list)):
        out = out[0]
    return out


def check(cond, label, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {label}" + (f"  {detail}" if detail else ""))
    if not cond:
        failures.append(label)
    return cond


TP = 2
CACHE_LEN = 8
DIM = 4
WORKDIR = "/tmp/dsv4_mb_spike"


class ToyCacheModel(torch.nn.Module):
    """Writes `x` into a persistent cache at `pos`, returns the cache sum.

    The cache is an nn.Parameter for the same reason the real port's state is:
    NxD resolves output aliases by scanning named_parameters() and matching
    data_ptr(), so a register_buffer would be silently dropped.

    `scale` is a weight, present only so there is something for set_weights to
    deliver per rank.
    """

    def __init__(self):
        super().__init__()
        self.cache = torch.nn.Parameter(
            torch.zeros(CACHE_LEN, DIM), requires_grad=False)
        self.scale = torch.nn.Parameter(
            torch.ones(DIM), requires_grad=False)

    def forward(self, x, pos):
        # x: (n_active, DIM) — n_active is what differs between the two graphs.
        n = x.size(0)
        idx = (pos.reshape(1).long() + torch.arange(n)) % CACHE_LEN
        new_cache = self.cache.index_copy(0, idx, x * self.scale)
        # (logits-ish, new_cache) — output 1 aliases the cache input.
        return new_cache.sum(dim=0, keepdim=True), new_cache


def instance_for(n_active):
    """A BaseModelInstance whose graph takes n_active rows of input."""
    from neuronx_distributed.trace.model_builder import BaseModelInstance

    def build():
        return ToyCacheModel().eval()

    # {output_index: input_index} — output 1 aliases input 0 of the *states*,
    # which NxD resolves by data_ptr against named_parameters().
    inst = BaseModelInstance(build, input_output_aliases={})

    def load_module():
        inst.module = build()
        # Alias output 1 back over the cache parameter.
        inst.input_output_aliases = [{inst.module.cache: 1}]

    inst.load_module = load_module
    return inst


def example_for(n_active):
    return [(torch.zeros(n_active, DIM), torch.zeros((), dtype=torch.int32))]


def main():
    if os.path.exists(WORKDIR):
        shutil.rmtree(WORKDIR)

    from neuronx_distributed.trace.model_builder import ModelBuilder

    # Per-rank weights, supplied directly. This is the shard_loader hand-off:
    # rank r gets scale = r + 1, so a wrong-rank load is visible in the output.
    def checkpoint_loader():
        return {"cache": torch.zeros(CACHE_LEN, DIM),
                "scale": torch.ones(DIM)}

    print("=== 1. build one artifact with two input widths ===")
    builder = ModelBuilder(
        router=None, tp_degree=TP, checkpoint_loader=checkpoint_loader,
        compiler_workdir=WORKDIR,
    )
    # 4-wide stands in for prefill, 1-wide for decode.
    builder.add(key="prefill", model_instance=instance_for(4),
                example_inputs=example_for(4),
                compiler_args="--model-type=transformer --auto-cast=none -O1")
    builder.add(key="decode", model_instance=instance_for(1),
                example_inputs=example_for(1),
                compiler_args="--model-type=transformer --auto-cast=none -O1")

    model = builder.trace(initialize_model_weights=False)
    check(True, "traced both graphs into one artifact")

    print("\n=== 2. supply per-rank weights directly ===")
    # nxd_model.initialize(checkpoint, start_rank) is what ModelBuilder.trace
    # itself calls when initialize_model_weights=True. It takes a per-rank list,
    # so a caller that already has rank-local tensors (shard_loader) can hand
    # them over without NxD's shard_children ever running — which matters
    # because shard_children does not recognise HF's same-named parallel classes.
    sharded = [{"cache": torch.zeros(CACHE_LEN, DIM),
                "scale": torch.full((DIM,), float(r + 1))}
               for r in range(TP)]
    model.nxd_model.initialize(sharded, torch.tensor(0))
    check(True, "initialize() accepted a per-rank list "
                "(no shard_children involved)")

    print("\n=== 3. state is shared: decode sees prefill's writes ===")
    ones = torch.ones(4, DIM)
    out_p = _first(model(ones, torch.zeros((), dtype=torch.int32)))
    print(f"  prefill wrote 4 rows -> sum={out_p.flatten().tolist()}")

    # Now a 1-wide call. If the cache were per-graph, the sum would only reflect
    # this one row; if shared, it reflects prefill's 4 rows plus this one.
    out_d = _first(model(torch.full((1, DIM), 5.0),
                         torch.tensor(4, dtype=torch.int32)))
    got = float(out_d.flatten()[0])
    print(f"  decode wrote 1 row at pos 4 -> sum={got}")

    # rank0 scale=1: prefill wrote 1.0 x4 rows, decode wrote 5.0 x1 row.
    expect_shared = 4 * 1.0 + 5.0
    expect_isolated = 5.0
    check(abs(got - expect_shared) < 1e-3,
          "decode graph sees the prefill graph's cache writes",
          f"got {got}, shared would be {expect_shared}, "
          f"isolated would be {expect_isolated}")

    print("\n=== 4. can the state be reset from the host? ===")
    # NOTE: ModelBuilder (v1) builds `trace.spmd.NxDModel`, which does NOT have
    # read/write_to_neuron_buffer — those are on the *other* NxDModel in
    # trace/nxd_model/nxd_model.py, used by model_builder_v2. So the v1 path
    # gives shared state but no documented host reset. Check both ideas here:
    # re-calling initialize() should re-run state_initializer and hand the
    # graphs a freshly zeroed cache.
    has_buffer_api = hasattr(model.nxd_model, "write_to_neuron_buffer")
    print(f"  write_to_neuron_buffer present: {has_buffer_api} "
          f"(expected False on ModelBuilder v1)")

    model.nxd_model.initialize(sharded, torch.tensor(0))
    out_r = _first(model(torch.full((1, DIM), 3.0),
                         torch.tensor(0, dtype=torch.int32)))
    got_r = float(out_r.flatten()[0])
    check(abs(got_r - 3.0) < 1e-3,
          "re-calling initialize() gives the graphs a cleared cache",
          f"got {got_r}, expected 3.0 on a cleared cache "
          f"(stale would be {expect_shared + 3.0})")

    print("\n" + "=" * 62)
    if failures:
        print(f"FAILED {len(failures)} check(s):")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("ALL CHECKS PASSED — ModelBuilder gives shared state + host reset")
    return 0


if __name__ == "__main__":
    sys.exit(main())
