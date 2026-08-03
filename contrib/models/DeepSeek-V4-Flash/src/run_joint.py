#!/usr/bin/env python
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the joint prefill+decode artifact: one batched prefill, then O(1) decode.

STATUS: NOT WORKING. The graphs compile; execution produces all-zero logits.
-----------------------------------------------------------------------------
What works: compile_joint builds both graphs at TP=32 with real weights
(2/2 Compiler status PASS, ~102 s), this script loads all 32 rank weight sets
(260/272 tensors non-zero, verified), `nxd_model.initialize()` accepts them and
takes ~14 s of real work, and dispatch routes correctly by shape
(input_shape_map = {[[1,128],[1,128]]: prefill, [[1,1],[1,1]]: decode}). The
graph returns a correctly shaped (1, 129280) logits tensor.

What does not: every logit is exactly 0.0, TTFT is 0.03 s (impossibly fast for
5 layers), and `neuron-monitor` reports **zero runtimes** during the call — no
NEFF is loaded on any core. So the traced module returns an untouched output
buffer without executing.

Ruled out:
  * weights (260/272 non-zero at handover; the probe below asserts this)
  * key naming (the graph wants an `inner.` prefix from _JointWrapper; without
    it initialize() raises "Missing weight tensor with key
    inner.layers.0.attn.wkv.weight", so the keys do bind)
  * weight/state accounting (272 params = 255 checkpoint weights + 17 aliased
    states, matching shard_loader's copied=255 exactly)
  * jit.save round-tripping a mocked-init flag — compiling in-process with
    --compile-and-run behaves identically, so serialisation is not the cause

Next step: find what actually loads NEFFs to the cores on this path.
`initialize()` calls `torch.ops.neuron._parallel_load(checkpoint)` and then
`initialize_spmd_models(...)`, but something equivalent to
`TensorParallelNeuronModel._load()`'s `_load_collectives_neuron` +
`move_trace_to_device` (trace.py:75-105) appears not to run here. Compare
against how NxDI's own application_base/model_wrapper drives an NxDModel; that
is the reference this port has not yet matched.


This is the point of the whole ModelBuilder port. The decode-only path has to
ingest the prompt one token at a time because it cannot inherit prefill's cache
(12.0 s for a 128-token prompt at 43 layers). Here both graphs share one KV cache
allocation, so the prompt goes through the prefill graph in a single call (0.85 s
measured) and generation continues from that cache at ~94 ms/token.

Dispatch is automatic and by shape: `NxDModel.router` looks the input shape up in
`input_shape_map`, so a (1, prefill_len) input runs the prefill graph and a (1, 1)
input runs decode. Nothing here selects a graph explicitly.

Weights are NOT in the artifact. `compile_joint` deliberately traces with
`initialize_model_weights=False` because NxD's own sharding cannot handle this
model (it does not recognise HF's same-named parallel classes — see
shard_loader.py), so this script loads each rank's slice on the host and hands
the whole per-rank list to `nxd_model.initialize()`.

Run:
    python src/run_joint.py --artifact /mnt/data/artifacts/dsv4_joint_tp32_L5 \\
        --prompt "It is well known that the capital city of France is" \\
        --greedy 8
"""

import argparse
import json
import os
import statistics
import sys
import time

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_CONTRIB_DIR = os.path.abspath(os.path.join(_HERE, ".."))
for p in (_HERE, _CONTRIB_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

import paths  # noqa: E402

_HF_INFERENCE_DIR = paths.hf_inference_dir()
if _HF_INFERENCE_DIR not in sys.path:
    sys.path.insert(0, _HF_INFERENCE_DIR)


def load_rank_weights(cfg, tp, model_path, n_layers, seq_len, max_batch_size):
    """Build each rank's state dict on the host, in rank order.

    Mirrors what compile_joint's `_make_instance` does at trace time, but keeps
    only the tensors: `nxd_model.initialize()` wants
    `List[Dict[str, Tensor]]`, one entry per rank.

    This is the expensive part of startup — it reads and dequantizes the
    checkpoint `tp` times, once per rank slice — and it is why the artifact is
    only 19 MB while the run still needs the full 159 GB checkpoint present.
    """
    import compile_neuron
    import shard_loader
    from neuronx_distributed.trace.mock_torchdist import mock_distributed
    from neuronx_distributed.parallel_layers import parallel_state

    per_rank = []
    with mock_distributed(world_size=tp):
        torch.distributed.init_process_group("xla", rank=0, world_size=tp)
        parallel_state.initialize_model_parallel(
            tp, 1, 1, skip_collective_init=True)
        for rank in range(tp):
            import compile_joint
            compile_joint._N_LAYERS = n_layers
            compile_joint._SEQ_LEN = seq_len
            compile_joint._MAX_BATCH_SIZE = max_batch_size
            compile_joint._MODEL_PATH = model_path
            m, hf, _ = compile_joint._build_hf_model()
            # HF's world_size comes from the mocked group; the rank does not,
            # because mock_distributed always reports rank 0. Set it explicitly
            # so shard_loader slices the right piece.
            hf.rank = rank
            shard_loader.load_rank_weights(
                m, model_path, rank=rank, world_size=tp, verbose=(rank == 0),
            )
            import decode_patches
            decode_patches.collect_state_aliases(m, n_real_outputs=1)
            # Keys must match the traced module's parameter names, and the traced
            # module is _JointWrapper, which holds the Transformer as `self.inner`.
            # Passing the bare model's state_dict fails with
            # "Missing weight tensor with key inner.layers.0.attn.wkv.weight".
            per_rank.append({f"inner.{k}": v.detach()
                             for k, v in m.state_dict().items()})
            del m
        parallel_state.destroy_model_parallel()
        torch.distributed.destroy_process_group()
    return per_rank


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", required=True)
    ap.add_argument("--prompt", default="It is well known that the capital city "
                                       "of France is")
    ap.add_argument("--fill", default="Artificial intelligence research has a "
                                      "long history. ",
                    help="Left-fill so the prompt fills prefill_len exactly; the "
                         "head emits only the last position.")
    ap.add_argument("--greedy", type=int, default=8)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--compile-and-run", action="store_true",
                    help="Compile in this process instead of loading the saved "
                         "artifact. Required for correct output today.")
    ap.add_argument("--compiler-workdir", default="/tmp/dsv4_joint_run_ws")
    ap.add_argument("--skip-weights", action="store_true",
                    help="Load the graph but not the checkpoint. Only useful for "
                         "checking that dispatch and the shared cache work; the "
                         "logits will be garbage.")
    args = ap.parse_args()

    with open(os.path.join(args.artifact, "config.json")) as f:
        acfg = json.load(f)
    tp = acfg["tp"]
    prefill_len = acfg["prefill_len"]
    seq_len = acfg["seq_len"]
    n_layers = acfg["n_layers"]
    print(f"[joint] artifact: tp={tp} n_layers={n_layers} "
          f"prefill_len={prefill_len} max_seq_len={seq_len}")

    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(os.path.join(paths.model_path(), "tokenizer.json"))
    ids = tok.encode(args.prompt).ids
    if len(ids) < prefill_len:
        filler = tok.encode(args.fill).ids
        need = prefill_len - len(ids)
        ids = (filler * (need // len(filler) + 1))[-need:] + ids
    ids = ids[-prefill_len:]
    print(f"[joint] prompt padded to {len(ids)} tokens; tail = "
          f"{tok.decode(ids[-12:])!r}")

    if args.compile_and_run:
        # Build in this process and use the returned object directly. This is the
        # only path that currently works end to end: the flag ModelBuilder clears
        # on its return value is not serialised, so a reloaded artifact runs as
        # mock-initialised and produces zeros (see below).
        print("[joint] compiling in-process (bypasses the jit.save limitation)")
        import compile_joint
        compile_joint._N_LAYERS = n_layers
        compile_joint._SEQ_LEN = seq_len
        compile_joint._PREFILL_LEN = prefill_len
        compile_joint._MAX_BATCH_SIZE = acfg.get("max_batch_size", 1)
        compile_joint._MODEL_PATH = paths.model_path()
        compile_joint._LOAD_WEIGHTS = False
        from neuronx_distributed.trace.model_builder import ModelBuilder
        builder = ModelBuilder(
            router=None, tp_degree=tp, checkpoint_loader=lambda: {},
            compiler_workdir=args.compiler_workdir, logical_nc_config=2,
        )
        cargs = ("--model-type=transformer --auto-cast=none -O1 "
                 "--enable-mixed-precision-accumulation")
        for key, mode in ((compile_joint.PREFILL_KEY, "prefill"),
                          (compile_joint.DECODE_KEY, "decode")):
            builder.add(key=key,
                        model_instance=compile_joint._make_instance(mode),
                        example_inputs=compile_joint._example_inputs(mode),
                        compiler_args=cargs)
        t0 = time.perf_counter()
        model = builder.trace(initialize_model_weights=False)
        print(f"[joint] compiled in {time.perf_counter() - t0:.1f}s")
    else:
        model = None

    # Register the Neuron TorchScript classes before jit.load. The artifact
    # references __torch__.torch.classes.neuron.SPMDModel, which only exists once
    # libtorchneuron is loaded; without this the load fails with
    # "Unknown type name '__torch__.torch.classes.neuron.SPMDModel'".
    import torch_neuronx
    from torch_neuronx import libtorchneuron
    libtorchneuron.load()

    if model is None:
        print(f"[joint] loading graph")
        t0 = time.perf_counter()
        model = torch.jit.load(os.path.join(args.artifact, "joint_model.pt"))
        print(f"[joint] graph loaded in {time.perf_counter() - t0:.1f}s")
        print("[joint] WARNING: a reloaded artifact runs mock-initialised and "
              "returns zeros; use --compile-and-run")


    if args.skip_weights:
        print("[joint] --skip-weights: initializing with zeros")
        nxd = model.nxd_model
        # Still needs one dict per rank so initialize() can bind buffers.
        per_rank = [{} for _ in range(tp)]
    else:
        print(f"[joint] loading weights for {tp} ranks (this is the slow part)")
        t0 = time.perf_counter()
        per_rank = load_rank_weights(
            acfg, tp, paths.model_path(), n_layers, seq_len,
            acfg.get("max_batch_size", 1))
        print(f"[joint] weights ready in {time.perf_counter() - t0:.1f}s")

    # Sanity-check what is about to be handed over: an all-zero or wrongly keyed
    # dict initialises fine and then produces all-zero logits, which is
    # indistinguishable from a broken graph unless checked here.
    if per_rank and per_rank[0]:
        d0 = per_rank[0]
        nz = sum(1 for v in d0.values()
                 if v.numel() and float(v.float().abs().sum()) > 0)
        print(f"[joint] rank0 weights: {len(d0)} tensors, {nz} non-zero")
        probe = "inner.layers.0.attn.wkv.weight"
        if probe in d0:
            print(f"[joint]   {probe} absmean="
                  f"{float(d0[probe].float().abs().mean()):.6f}")
        if nz == 0:
            raise SystemExit("[FAIL] every weight is zero; the load did not work")

    t0 = time.perf_counter()
    # start_rank_tensor is shape (1,), matching NxDI's own call
    # (application_base.py:414: torch.tensor([start_rank_id], dtype=torch.int32,
    # device="cpu")). It is consumed by torch.ops.aten.Int(), so a 0-d tensor is
    # not interchangeable here.
    start_rank_tensor = torch.tensor([0], dtype=torch.int32, device="cpu")
    model.nxd_model.initialize(per_rank, start_rank_tensor)
    print(f"[joint] initialize() (weights + shared state onto device) in "
          f"{time.perf_counter() - t0:.1f}s", flush=True)

    def call(token_ids, positions):
        """One graph call. The shape of `token_ids` selects prefill vs decode."""
        inp = torch.tensor([token_ids], dtype=torch.long)
        pos = torch.tensor([positions], dtype=torch.int32)
        out = model(inp, pos)
        if os.environ.get("DSV4_JOINT_DEBUG") == "1":
            def shape_of(o):
                if isinstance(o, (tuple, list)):
                    return [shape_of(x) for x in o]
                return tuple(o.shape) if hasattr(o, "shape") else type(o).__name__
            print(f"[out] structure = {shape_of(out)}", flush=True)
        while isinstance(out, (tuple, list)):
            out = out[0]
        return out

    # --- the prefill call: whole prompt, ONE forward ---
    print(f"\n[joint] prefill: {len(ids)} tokens in one call ...", flush=True)
    t0 = time.perf_counter()
    logits = call(ids, list(range(len(ids))))
    ttft = time.perf_counter() - t0
    print(f"[joint] TTFT = {ttft:.2f}s  (decode-only ingest of the same prompt "
          f"costs prompt_len x ~94 ms)")

    row = logits[0].float()
    topv, topi = row.topk(args.top_k)
    print(f"  top-{args.top_k}:")
    for v, i in zip(topv.tolist(), topi.tolist()):
        print(f"    {i:>7}  {v:8.3f}  {tok.decode([i])!r}")
    if not bool(torch.isfinite(row).all()):
        raise SystemExit("[FAIL] non-finite logits from the prefill graph")

    # --- decode: continues from the cache prefill just wrote ---
    # This is the claim under test. If the cache were NOT shared, decode would
    # start from an empty cache and the continuation would be incoherent.
    generated, step_ms = [], []
    pos = len(ids)
    print(f"\n[joint] decode from position {pos}, reusing prefill's cache")
    for step in range(args.greedy):
        nxt = int(logits[0].float().argmax())
        generated.append(nxt)
        print(f"  step {step + 1}: {nxt} {tok.decode([nxt])!r}"
              + (f"  ({step_ms[-1]:.0f} ms)" if step_ms else ""), flush=True)
        if step + 1 >= args.greedy or pos >= seq_len - 1:
            break
        t0 = time.perf_counter()
        logits = call([nxt], [pos])
        step_ms.append((time.perf_counter() - t0) * 1000)
        pos += 1

    print("\n" + "=" * 62)
    print(f"[prompt]     {tok.decode(ids[-len(tok.encode(args.prompt).ids):])!r}")
    print(f"[generated]  {tok.decode(generated)!r}")
    print(f"  TTFT            {ttft:.2f} s  (one prefill call)")
    if step_ms:
        print(f"  TPOT            {statistics.mean(step_ms):.0f} ms  "
              f"(median {statistics.median(step_ms):.0f}, n={len(step_ms)})")
    print("\n[ok] joint inference completed — decode continued from prefill's "
          "cache")


if __name__ == "__main__":
    main()
