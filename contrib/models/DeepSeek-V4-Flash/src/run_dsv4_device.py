#!/usr/bin/env python
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compile DSV4Model's two graphs against one shared cache, and verify on device.

STATUS: compiles and binds correctly; execution returns all-zero logits.
-----------------------------------------------------------------------
Working, with evidence:
  * both graphs compile at TP=32, 2/2 Compiler status PASS, ~114 s
  * state binding correct: state_initializer yields 32 ranks x 17 states, keyed
    kv_mgr.past_key_values.N
  * weight binding correct: 0 of 207 graph weights missing; reconciliation printed
  * initialize() returns without error
  * all five CPU gates upstream are bit-identical to the validated patched-HF path

Not working: TTFT 0.03 s (impossible for 5 layers), every logit exactly 0.0, and
neuron-monitor reports ZERO runtimes during the call -- no NEFF on any core, while
the C++ is_initialized() returns True so the "not initialized" guard never fires.

Ruled out, each by direct experiment:
  * weight key naming -- was a real bug (HLO uses '->', runtime uses '.'); fixed,
    both forms now supplied, 0 missing confirmed
  * degenerate example ids -- was a real bug (all-zero ids put every token in rank
    0's vocab shard); fixed, was surfacing as status=1006
  * a poisoned compiler cache from the neuronx-cc 2.25 experiment (cache hits with
    only graph.neff, no model.MODULE_*.neff); cleared and recompiled
  * a stale mock_initialization flag; cleared explicitly
  * calling the traced NxDModelExecutor instead of nxd_model.forward directly
    (which is what NxDI's warmup does); switched, no change
  * save / del / jit.load, i.e. NxDI's own compile-then-load sequence rather than
    running the in-process traced object; adopted, no change
  * the ModelBuilder path itself: the toy shared-cache spike
    (test/spike/test_modelbuilder_shared_state.py) runs correctly on this exact
    path and SDK, producing real values and a genuinely shared cache

So the mechanism works and the numerics work; something specific to this model's
graph is not executing. The remaining untested difference against NxDI's flow is
that GLM-5.2 goes through NeuronBaseForCausalLM.load(), which additionally calls
set_env_vars() and warmup() and populates model_wrapper.model for each registered
model -- this driver constructs none of that scaffolding.

Two follow-ups, in order of expected value:
  1. Compare against a working NxDI model (llama) end-to-end on this box and SDK.
     One run settles whether this is our integration or the environment; the
     diff in its initialize/load sequence is then the answer.
  2. TP=2 for fast iteration is currently blocked by a compiler capacity error,
     [NCC_INLA001] Allocated memory out of bound {concatenate}@SB(505x512), where
     505 is close to index_topk=512 -- pointing at the sparse_attn topk
     concatenation width. Worth understanding regardless.
"""

import argparse
import json
import os
import statistics
import sys
import time

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_CONTRIB = os.path.abspath(os.path.join(_HERE, ".."))
for p in (_HERE, _CONTRIB):
    if p not in sys.path:
        sys.path.insert(0, p)

import paths  # noqa: E402

_HF_DIR = paths.hf_inference_dir()
if _HF_DIR not in sys.path:
    sys.path.insert(0, _HF_DIR)

# Published via env because ModelBuilder's instances are built in contexts that
# re-import this module.
_N_LAYERS = int(os.environ.get("DSV4_N_LAYERS", "5"))
_SEQ_LEN = int(os.environ.get("DSV4_SEQ_LEN", "256"))
_PREFILL_LEN = int(os.environ.get("DSV4_PREFILL_LEN", "128"))
_TP = int(os.environ.get("DSV4_TP", "32"))
_BATCH = 1
_LOAD_WEIGHTS = os.environ.get("DSV4_LOAD_WEIGHTS", "1") == "1"

PREFILL_KEY, DECODE_KEY = "prefill", "decode"
_SERIAL = [0]


def _build(mode):
    """(DSV4Model, hf_module) with every patch applied, on the host."""
    import importlib.util

    import collectives
    import compile_neuron
    import dsv4_patches
    import modeling_dsv4 as md
    from neuronx_distributed_inference.models.config import MoENeuronConfig

    compile_neuron._wire_shims()

    # A private copy of HF's model module per graph: HF keeps forwards on the
    # class and world_size/rank at module scope, so two graphs sharing one module
    # object fight over the same patch site.
    _SERIAL[0] += 1
    spec = importlib.util.spec_from_file_location(
        "dsv4_hf_%d" % _SERIAL[0], os.path.join(_HF_DIR, "model.py"))
    hf = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = hf
    spec.loader.exec_module(hf)

    # world_size/rank are read at module-exec time from a torch.distributed that
    # mock_distributed swaps out wholesale, so the private copy records 1. Re-read
    # after the body has run.
    import torch.distributed as _live
    if _live.is_initialized():
        hf.world_size = _live.get_world_size()
        hf.rank = _live.get_rank()

    compile_neuron.apply_xla_patches(hf)
    collectives.patch_hf_dist(hf)
    helper_state = compile_neuron.index_helper_state()
    dsv4_patches.install(hf)

    torch.set_default_dtype(torch.bfloat16)
    nc = MoENeuronConfig(tp_degree=_TP, batch_size=_BATCH, seq_len=_SEQ_LEN,
                         torch_dtype=torch.bfloat16)
    cfg = md.DSV4InferenceConfig.from_checkpoint(
        paths.config_json(), nc, n_layers=_N_LAYERS,
        max_batch_size=_BATCH, max_seq_len=_SEQ_LEN, dtype="bf16",
        expert_dtype=None, n_mtp_layers=0)
    hf_model = hf.Transformer(hf.ModelArgs(
        **{k: v for k, v in _model_args(cfg).items()})).eval()
    torch.set_default_dtype(torch.float32)

    if _LOAD_WEIGHTS:
        import shard_loader
        shard_loader.load_rank_weights(
            hf_model, paths.model_path(), rank=hf.rank,
            world_size=hf.world_size, verbose=(hf.rank == 0))

    model = md.DSV4Model(cfg, hf, hf_model, mode=mode)
    model._helper_state = helper_state
    return model, hf


def _model_args(cfg):
    """The ModelArgs kwargs HF's Transformer wants, from our config."""
    with open(paths.config_json()) as f:
        raw = json.load(f)
    raw.update(dtype="bf16", expert_dtype=None, n_mtp_layers=0,
               max_batch_size=_BATCH, max_seq_len=_SEQ_LEN,
               n_layers=_N_LAYERS)
    if len(raw.get("compress_ratios", [])) > _N_LAYERS:
        raw["compress_ratios"] = raw["compress_ratios"][:_N_LAYERS]
    return raw


def _instance(mode):
    from neuronx_distributed.trace.model_builder import BaseModelInstance

    inst = BaseModelInstance(module_cls=None, input_output_aliases={})

    def load_module():
        import compile_neuron
        model, hf = _build(mode)
        # The index helpers read their device from a dict compile_neuron keeps as
        # a module global holding only the newest patch, so each graph must capture
        # its own or the first one reads a dict nobody writes.
        inst.module = model
        inst.input_output_aliases = [model.alias_map(n_real_outputs=1)]

    inst.load_module = load_module
    return inst


def _example_inputs(mode):
    n_active = 1 if mode == "decode" else _PREFILL_LEN
    # NOT torch.zeros. Token id 0 lands in rank 0's vocab shard, so tracing with
    # all-zero ids records a degenerate case for every other rank -- the same
    # trap the README documents for ParallelEmbedding's bool-mask assignment,
    # which surfaces at runtime as
    #   status=1006 Execution Out-Of-Bounds Memory Access
    # Spread the example ids across the vocab so every rank traces the general
    # case. The values do not matter otherwise; only the shapes are recorded.
    ids = (torch.arange(n_active, dtype=torch.long) * 977 + 101) % 129280
    ids = ids.unsqueeze(0).repeat(_BATCH, 1)
    if mode == "decode":
        pos = torch.ones(_BATCH, 1, dtype=torch.int32)
    else:
        pos = torch.arange(_PREFILL_LEN, dtype=torch.int32).unsqueeze(0).repeat(
            _BATCH, 1)
    return [(ids, pos)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tp", type=int, default=32)
    ap.add_argument("--n-layers", type=int, default=5)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--prefill-len", type=int, default=128)
    ap.add_argument("--greedy", type=int, default=8)
    ap.add_argument("--prompt", default="It is well known that the capital city "
                                        "of France is")
    ap.add_argument("--fill", default="Artificial intelligence research has a "
                                      "long history. ")
    ap.add_argument("--no-weights", action="store_true")
    ap.add_argument("--compiler-workdir", default="/mnt/nvme/tmp/dsv4_dev_ws")
    args = ap.parse_args()

    os.environ["DSV4_N_LAYERS"] = str(args.n_layers)
    os.environ["DSV4_SEQ_LEN"] = str(args.seq_len)
    os.environ["DSV4_PREFILL_LEN"] = str(args.prefill_len)
    os.environ["DSV4_TP"] = str(args.tp)
    os.environ["DSV4_LOAD_WEIGHTS"] = "0" if args.no_weights else "1"
    global _N_LAYERS, _SEQ_LEN, _PREFILL_LEN, _TP, _LOAD_WEIGHTS
    _N_LAYERS, _SEQ_LEN = args.n_layers, args.seq_len
    _PREFILL_LEN, _TP = args.prefill_len, args.tp
    _LOAD_WEIGHTS = not args.no_weights

    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(os.path.join(paths.model_path(), "tokenizer.json"))
    ids = tok.encode(args.prompt).ids
    if len(ids) < args.prefill_len:
        filler = tok.encode(args.fill).ids
        need = args.prefill_len - len(ids)
        ids = (filler * (need // len(filler) + 1))[-need:] + ids
    ids = ids[-args.prefill_len:]
    print(f"[dev] tp={args.tp} n_layers={args.n_layers} "
          f"prefill_len={args.prefill_len} seq_len={args.seq_len} "
          f"weights={'real' if _LOAD_WEIGHTS else 'random'}")
    print(f"[dev] prompt tail: {tok.decode(ids[-12:])!r}", flush=True)

    from neuronx_distributed.trace.model_builder import ModelBuilder

    os.makedirs(args.compiler_workdir, exist_ok=True)
    # checkpoint_loader returns {} deliberately.
    #
    # Tried routing weights through builder.shard_checkpoint() instead, since
    # NxDI's own load path does that and the control experiment showed that path
    # works on this box. It does NOT work for this model, and the error is the
    # proof of why shard_loader.py has to exist:
    #
    #   Incorrect tensor shape at inner.embed.weight: received 129280 4096,
    #                                                 expected 4040 4096
    #   ... attn_sink: received 64, expected 2
    #
    # 129280/32 = 4040 and 64/32 = 2, i.e. shard_checkpoint handed back the FULL
    # tensors having sharded nothing -- exactly the silent no-op shard_loader's
    # header documents, because NxD's shard_children early-returns unless a module
    # is an instance of its own parallel classes and HF's model.py declares
    # same-named ones of its own.
    #
    # The useful part: that error also shows the runtime DOES validate weight
    # shapes on initialize(). Our hand-sharded dict passes that validation, so the
    # weights reaching the device are correctly shaped and correctly named -- which
    # rules weights out as the cause of the all-zero logits.
    builder = ModelBuilder(
        router=None, tp_degree=args.tp, checkpoint_loader=lambda: {},
        compiler_workdir=args.compiler_workdir, logical_nc_config=2,
    )
    cargs = ("--model-type=transformer --auto-cast=none -O1 "
             "--enable-mixed-precision-accumulation")
    for key, mode in ((PREFILL_KEY, "prefill"), (DECODE_KEY, "decode")):
        builder.add(key=key, model_instance=_instance(mode),
                    example_inputs=_example_inputs(mode), compiler_args=cargs)
        print(f"[dev] registered {key} (width "
              f"{_example_inputs(mode)[0][0].shape[1]})", flush=True)

    t0 = time.perf_counter()
    traced = builder.trace(initialize_model_weights=False)
    print(f"[dev] compiled both graphs in {time.perf_counter() - t0:.1f}s",
          flush=True)

    # Save, drop, reload -- NxDI's own sequence (application_base.compile does
    # jit.save then `del traced_model`; load() then does a fresh jit.load). Worth
    # trying because the object ModelBuilder returns is what jit.trace produced,
    # and every all-zero result so far came from running that in-process object.
    save_path = os.path.join(args.compiler_workdir, "model.pt")
    torch.jit.save(traced, save_path)
    del traced
    from torch_neuronx import libtorchneuron
    libtorchneuron.load()
    traced = torch.jit.load(save_path)
    print(f"[dev] saved, dropped and reloaded from {save_path}", flush=True)

    # Weights via builder.shard_checkpoint(), which is what NxDI's own load path
    # uses (application_base.load_weights -> get_builder().shard_checkpoint()).
    # It runs preprocess_checkpoint (which drops keys absent from
    # model.state_dict() and renames to the graph's convention), cast_weights, and
    # shard_weights_with_cache per rank. The previous hand-assembled dict skipped
    # all of that; the control experiment (CONTROL_EXPERIMENT.md) showed NxDI's
    # path works on this box, so matching it is the next thing to try.
    #
    # The checkpoint_loader passed to ModelBuilder returns the FULL checkpoint
    # here rather than {} -- shard_checkpoint calls it once and slices per rank.
    print(f"[dev] building rank-sharded weights for {args.tp} ranks ...",
          flush=True)
    t0 = time.perf_counter()
    per_rank = _rank_weights(args.tp)
    print(f"[dev] weights ready in {time.perf_counter() - t0:.1f}s", flush=True)
    nz = sum(1 for v in per_rank[0].values()
             if v.numel() and float(v.float().abs().sum()) > 0)
    print(f"[dev] rank0: {len(per_rank[0])} tensors, {nz} non-zero")
    if _LOAD_WEIGHTS and nz == 0:
        raise SystemExit("[FAIL] all weights zero; the load did not work")

    # Diagnose the binding before running: a graph with no state_initializer has
    # nothing to bind the aliased cache to, and a weights dict whose keys do not
    # match the traced parameter names binds nothing -- both surface as all-zero
    # logits rather than as an error.
    nxd = traced.nxd_model
    si = getattr(nxd, "state_initializer", None)
    print(f"[dev] state_initializer present: {si is not None}")
    if si is not None:
        st = si()
        print(f"[dev] state_initializer -> {len(st)} ranks, "
              f"{len(st[0]) if st else 0} states/rank")
        if st and st[0]:
            k0 = sorted(st[0].keys())[:3]
            print(f"[dev] state keys sample: {k0}")
    print(f"[dev] weight keys sample: {sorted(per_rank[0].keys())[:3]}")
    # What does the GRAPH think its weights are called? A mismatch binds nothing
    # and initialize() returns quickly with the model still uninitialised.
    try:
        import neuronx_distributed.trace.model_builder as _mb
        mc = builder.model_collection[PREFILL_KEY]
        art = mc.hlo_artifact_collection[0]
        gnames = sorted(art.weight_name_to_idx.keys())
        print(f"[dev] graph expects {len(gnames)} weights, sample: {gnames[:3]}")
        mine = set(per_rank[0].keys())
        missing = [g for g in gnames if g not in mine]
        extra = [m for m in sorted(mine) if m not in set(gnames)]
        print(f"[dev] missing from my dict: {len(missing)} {missing[:3]}")
        print(f"[dev] extra in my dict:     {len(extra)} {extra[:3]}")
    except Exception as e:
        print(f"[dev] could not read graph weight names: {type(e).__name__}: {e}")

    # ModelBuilder sets mock_initialization(True) so jit.trace can run without a
    # device, then clears it on the object it returns. Clear it again defensively:
    # if it were still set, is_initialized() returns True, the "not initialized"
    # guard is skipped, and forward() returns untouched output buffers -- all-zero
    # logits in 0.03 s with no NEFF ever loaded.
    try:
        traced.nxd_model.mock_initialization(False)
        print("[dev] mock_initialization(False) applied")
    except Exception as e:
        print(f"[dev] mock_initialization not callable: {type(e).__name__}")

    t0 = time.perf_counter()
    traced.nxd_model.initialize(
        per_rank, torch.tensor([0], dtype=torch.int32, device="cpu"))
    print(f"[dev] initialize() in {time.perf_counter() - t0:.1f}s", flush=True)

    # Is the SPMD model actually initialised after our initialize()? A fresh
    # jit.load raises "not initialized" on forward, so the guard does work -- if it
    # reports True here and still returns zeros, the NEFF is being invoked and
    # producing nothing, which is a different problem from never being invoked.
    try:
        sm = traced.nxd_model.models["prefill"]
        print(f"[dev] prefill SPMD is_initialized: "
              f"{[mm.is_initialized() for mm in sm.models]}")
    except Exception as e:
        print(f"[dev] could not query is_initialized: {type(e).__name__}: {e}")

    def call(token_ids, positions):
        inp = torch.tensor([token_ids], dtype=torch.long)
        pos = torch.tensor([positions], dtype=torch.int32)
        # Call nxd_model directly, NOT the traced NxDModelExecutor wrapper.
        # `traced` came out of torch.jit.trace, which recorded the ops of ONE
        # example shape; replaying it does not re-run NxDModel.router, so it
        # neither dispatches by shape nor executes the NEFF -- it returns the
        # recorded (zero) outputs in ~0.03 s. NxDI's own warmup goes straight to
        # `model.model.nxd_model.forward(example)` for the same reason
        # (application_base.py:363).
        out = traced.nxd_model.forward([inp, pos])
        while isinstance(out, (tuple, list)):
            out = out[0]
        return out

    print(f"\n[dev] prefill: {len(ids)} tokens, one call", flush=True)
    t0 = time.perf_counter()
    logits = call(ids, list(range(len(ids))))
    ttft = time.perf_counter() - t0
    row = logits[0].float()
    print(f"[dev] TTFT {ttft:.2f}s")
    print(f"[dev] logits: finite={bool(torch.isfinite(row).all())} "
          f"absmax={float(row.abs().max()):.3f} std={float(row.std()):.3f}")
    if float(row.abs().max()) == 0.0:
        raise SystemExit("[FAIL] all-zero logits: the graph did not execute")
    topv, topi = row.topk(5)
    for v, i in zip(topv.tolist(), topi.tolist()):
        print(f"    {i:>7}  {v:8.3f}  {tok.decode([i])!r}")

    gen, step_ms = [], []
    pos = len(ids)
    print(f"\n[dev] decode from position {pos} (prefill's cache)", flush=True)
    for step in range(args.greedy):
        nxt = int(logits[0].float().argmax())
        gen.append(nxt)
        print(f"  step {step + 1}: {nxt} {tok.decode([nxt])!r}"
              + (f"  ({step_ms[-1]:.0f} ms)" if step_ms else ""), flush=True)
        if step + 1 >= args.greedy or pos >= args.seq_len - 1:
            break
        t0 = time.perf_counter()
        logits = call([nxt], [pos])
        step_ms.append((time.perf_counter() - t0) * 1000)
        pos += 1

    print("\n" + "=" * 62)
    print(f"[generated] {tok.decode(gen)!r}")
    print(f"  TTFT  {ttft:.2f} s   (one prefill call, {len(ids)} tokens)")
    if step_ms:
        print(f"  TPOT  {statistics.mean(step_ms):.0f} ms  "
              f"(median {statistics.median(step_ms):.0f}, n={len(step_ms)})")


def _rank_weights(tp):
    """One state dict per rank, keyed as the traced module names them."""
    import shard_loader
    from neuronx_distributed.parallel_layers import parallel_state
    from neuronx_distributed.trace.mock_torchdist import mock_distributed

    out = []
    with mock_distributed(world_size=tp):
        torch.distributed.init_process_group("xla", rank=0, world_size=tp)
        parallel_state.initialize_model_parallel(tp, 1, 1,
                                                 skip_collective_init=True)
        for rank in range(tp):
            model, hf = _build("prefill")
            hf.rank = rank
            if _LOAD_WEIGHTS:
                shard_loader.load_rank_weights(
                    model.inner, paths.model_path(), rank=rank, world_size=tp,
                    verbose=False)
            # DSV4Model holds the Transformer as `inner`, and the cache manager's
            # Parameters are aliased state, not weights -- initialize() supplies
            # only the latter.
            # Supply BOTH naming forms. The HLO's weight_name_to_idx uses '->'
            # separators ('inner->layers->0->attn->wkv->weight') while the runtime
            # looks the same tensor up by its dotted name
            # ('inner.layers.0.attn.wkv.weight'). Providing only dots leaves all
            # 207 graph weights unbound -- the NEFF then runs against empty
            # buffers and returns all-zero logits in 0.03 s with no error, which
            # is exactly the failure that stalled the previous attempt.
            # Providing only arrows raises "Missing weight tensor with key
            # inner.layers.0.attn.wkv.weight". The two views share storage, so
            # this costs nothing.
            sd = {}
            for k, v in model.inner.state_dict().items():
                t = v.detach()
                sd[f"inner.{k}"] = t
                sd["inner->" + k.replace(".", "->")] = t
            out.append(sd)
            del model
        parallel_state.destroy_model_parallel()
        torch.distributed.destroy_process_group()
    return out


if __name__ == "__main__":
    main()
