#!/usr/bin/env python
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compile prefill and decode into ONE artifact with a shared KV cache.

The problem this solves
-----------------------
`compile_neuron.py` uses `parallel_model_trace`, which produces exactly one graph
per artifact with its own private aliased state. Prefill and decode therefore end
up in two artifacts with two caches and no way to pass one to the other, so a
decode run has to re-ingest the prompt one token at a time: 12.0 s to first token
against 0.85 s for the prefill graph on the same 128-token prompt (~14x).

`ModelBuilder` fixes this structurally. Both graphs are registered as keys in one
builder, and `build_state_initializer` allocates the state **once** for the whole
collection, keyed by `checkpoint_key` off any one model's metaneff. Both graphs
declare the same cache under the same key, so they resolve to the same device
buffer — the hand-off is not a copy between graphs, it is one allocation both
graphs were compiled against. Validated on a toy model first in
test/spike/test_modelbuilder_shared_state.py, which is worth reading before this
file.

Dispatch is by input shape: `input_shape_map` routes a (1, seq_len) input to the
prefill graph and a (1, 1) input to decode. That is why the two graphs must
differ in width — same weights, same cache, two shapes of one model.

Two things this does NOT do NxDI's way
--------------------------------------
**Sharding stays hand-rolled.** `ModelBuilder(checkpoint_loader=...)` expects to
shard the returned checkpoint itself via `shard_children`, which early-returns
unless a module is an instance of NxD's own parallel classes. HF's model.py
declares its own same-named classes, so NxD would match nothing and shard nothing,
silently (see shard_loader.py's header). Instead the checkpoint_loader returns an
empty dict and rank-local weights go in through
`nxd_model.initialize(per_rank_list, start_rank)`, which is the same entry point
`ModelBuilder.trace(initialize_model_weights=True)` uses internally.

**Weights are loaded per rank on the host, not by NxD.** Same reason. This is the
existing `shard_loader.load_rank_weights` path, unchanged.

STATUS: prefill traces, decode does not. Work in progress.
---------------------------------------------------------
Confirmed working:
  * both graphs register in one builder and are recognised as separate keys
  * prefill generates HLO cleanly at TP=2 and TP=32 (1.5 s)
  * the toy-model spike proves the shared-cache mechanism itself
    (test/spike/test_modelbuilder_shared_state.py)

Blocked: the decode graph fails during HLO generation with

    RuntimeError: The size of tensor a (128) must match the size of
                  tensor b (0) at non-singleton dimension 0

attributed to `decode_patches.window_topk_idxs`'s `j <= pos`. What is known:

  * every operand there is provably correct at that point. An unconditional
    probe prints `win=128 j=(128,)/128 pos=()/1 dev=xla:0`, and the comparison
    itself returns `(128,)` when evaluated separately.
  * `window_topk_idxs` compiles and runs correctly on XLA in isolation with the
    same inputs (verified standalone: returns (1, 1, 128), materialises to -125).
  * decode fails the same way when registered ALONE, so this is not
    cross-key interference from prefill's trace.
  * it is not the 0-d `start_pos` (passing shape (1,) changes nothing), not
    device mixing (both xla:0), and not meta-device init (that raises a
    different error).

So the failing op is almost certainly NOT the one in the traceback: XLA is lazy,
and the error surfaces at the first sync point rather than where the bad shape
was created. `XLA_USE_EAGER_DEBUG_MODE=1` would localise it but segfaults on this
model. The next step is to bisect decode_patches by materialising intermediates
(`.cpu()`) one at a time under ModelBuilder until the real origin appears.

Everything else in this file is validated and reusable; the four ModelBuilder
integration fixes below were each needed to get this far.

Usage:
    python src/compile_joint.py --tp 32 --n-layers 5 --seq-len 256 \\
        --prefill-len 128 --load-weights \\
        --out-dir /mnt/data/artifacts/dsv4_joint_tp32_L5
"""

import argparse
import json
import os
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


# Env-published config, because ModelBuilder's instances are constructed in
# worker contexts that re-import this module (same constraint compile_neuron has).
_MODEL_PATH = paths.model_path()
_CONFIG_JSON = paths.config_json()
_N_LAYERS = int(os.environ.get("DSV4_N_LAYERS", "43"))
_SEQ_LEN = int(os.environ.get("DSV4_SEQ_LEN", "256"))
_PREFILL_LEN = int(os.environ.get("DSV4_PREFILL_LEN", "128"))
_MAX_BATCH_SIZE = int(os.environ.get("DSV4_MAX_BATCH_SIZE", "1"))
_LOAD_WEIGHTS = os.environ.get("DSV4_LOAD_WEIGHTS", "0") == "1"

PREFILL_KEY = "prefill"
DECODE_KEY = "decode"

_MODULE_COUNTER = [0]


def _module_serial():
    # Unique suffix so each private `model` module gets its own sys.modules key.
    _MODULE_COUNTER[0] += 1
    return _MODULE_COUNTER[0]


def _build_hf_model():
    """Build the HF Transformer with every patch applied, on the host.

    Returns (model, hf_module). Patches are applied in the order
    compile_neuron establishes: XLA safety first (RoPE, MoE, embedding, top-k,
    o_proj — all position-independent), then the per-mode forwards on top.
    """
    import importlib.util
    import compile_neuron

    compile_neuron._wire_shims()

    # Load a PRIVATE copy of HF's model module per call, rather than sharing
    # sys.modules["model"].
    #
    # HF keeps its forwards on the class and its world_size/rank at module scope,
    # so two registered graphs patching one module object fight over the same
    # patch site: the second install wins for both, and the first graph traces
    # with the other mode's code. A private module gives each graph its own
    # classes to patch and its own closures, which is what makes the two traces
    # independent while still declaring identical state (build_state_initializer
    # keys that on shape and order, not on object identity).
    spec = importlib.util.spec_from_file_location(
        "dsv4_model_%d" % _module_serial(),
        os.path.join(_HF_INFERENCE_DIR, "model.py"),
    )
    hf = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = hf
    spec.loader.exec_module(hf)

    compile_neuron.apply_xla_patches(hf)
    # ModelBuilder traces rank 0 under mock_distributed, which leaves all_reduce
    # real while never creating a default group. Point HF's `dist` at a shim that
    # pins collectives to the TP group, the way NxD's own layers do.
    import collectives
    collectives.patch_hf_dist(hf)
    torch.set_default_dtype(torch.bfloat16)

    with open(_CONFIG_JSON) as f:
        cfg = json.load(f)
    cfg.update(dtype="bf16", expert_dtype=None, n_mtp_layers=0,
               max_batch_size=_MAX_BATCH_SIZE, max_seq_len=_SEQ_LEN)
    if _N_LAYERS < cfg["n_layers"]:
        cfg["n_layers"] = _N_LAYERS
        if len(cfg.get("compress_ratios", [])) > _N_LAYERS:
            cfg["compress_ratios"] = cfg["compress_ratios"][:_N_LAYERS]

    m = hf.Transformer(hf.ModelArgs(**cfg)).eval()
    return m, hf


def _install_dual_patches(hf):
    """Install both modes' forwards behind one call-time dispatch.

    Returns a state dict shared by both: {"active": "prefill"|"decode",
    "sink": StateSink, "pos": tensor}. The wrapper sets "active" before invoking
    forward, so a single set of module-level patches serves both traces.

    This exists because the two modes cannot each own the patch site. HF keeps
    Attention/Compressor/Indexer forwards on the class, ModelBuilder loads every
    registered model before tracing any, and the second load would otherwise
    replace the first one's forwards — running decode's code inside prefill's
    trace.
    """
    import decode_patches
    import prefill_patches

    # One dispatcher per hf module, reused across load_module() calls.
    #
    # ModelBuilder calls load_module() once per registered key, each of which
    # re-imports the same `model` module object — so a fresh install per call
    # re-patches the class and leaves the *earlier* wrapper holding an orphaned
    # state dict. The second install wins the patch site, its closure reads the
    # second dict, and the first wrapper's writes go nowhere: `pos` stays None
    # and window_topk_idxs gets a zero-width tensor ("size of tensor a (128) must
    # match tensor b (0)"). Caching on the module keeps one dict for one patch
    # site, which is what the wrappers assume.
    # A fresh state dict and a fresh install per call. Both are required:
    #
    #   * fresh install, because _build_hf_model runs apply_xla_patches every
    #     time, which puts HF's single-mode Attention.forward back over this
    #     dispatcher.
    #   * fresh dict, because each registered key gets its own model instance
    #     (see _make_instance) and therefore its own closure. Sharing one dict
    #     across models is what made decode read prefill's `pos`.
    state = {"active": "prefill", "sink": None, "pos": None}

    p_compressor, p_indexer, p_attention = (
        prefill_patches.make_prefill_forwards(hf))
    d_compressor, d_indexer, d_attention = (
        decode_patches.make_decode_forwards(hf))

    def attention_forward(self, x, start_pos):
        if state["active"] == "prefill":
            return p_attention(self, x, state["sink"])
        return d_attention(self, x, state["pos"], state["sink"])

    def compressor_forward(self, x, pos):
        if state["active"] == "prefill":
            return p_compressor(self, x, state["sink"])
        return d_compressor(self, x, state["pos"], state["sink"])

    def indexer_forward(self, x, qr, pos, offset):
        if state["active"] == "prefill":
            return p_indexer(self, x, qr, offset, state["sink"])
        return d_indexer(self, x, qr, state["pos"], offset, state["sink"])

    hf.Attention.forward = attention_forward
    hf.Compressor.forward = compressor_forward
    hf.Indexer.forward = indexer_forward
    return state


class _JointWrapper(torch.nn.Module):
    """Runs one branch of the model and returns (logits, *state).

    One class for both graphs, differing only in `mode`, so the two traces are
    guaranteed to declare their state in the same order — which is what makes
    `build_state_initializer` hand them the same buffers. If the orders differed,
    the graphs would silently alias each other's caches to the wrong slots.
    """

    def __init__(self, inner, hf_mod, mode, slots, state, raw_forward):
        super().__init__()
        self.inner = inner
        self._hf = hf_mod
        self._mode = mode
        self._slots = slots
        # The mode's patch state dict, installed once at build time. Patching
        # inside forward() does not work: apply_*_patches is called while
        # ModelBuilder is tracing under mock_distributed, and the patched
        # forwards close over module-level `world_size`, which HF sets in
        # Transformer.__init__ from the live process group. Re-entering the patch
        # machinery mid-trace re-resolves collectives against a group that is not
        # the default one, and the all_reduce in ParallelEmbedding raises
        # "Default process group has not been initialized".
        self._state = state
        self._raw_forward = raw_forward

    def forward(self, input_ids, start_pos):
        import decode_patches

        with torch.no_grad():
            sink = decode_patches.build_sink(self.inner, self._hf)
            self._state["active"] = self._mode
            self._state["sink"] = sink
            self._state["pos"] = start_pos.reshape(())

            # The index helpers build position constants on a device recorded
            # from input_ids by the patched Transformer.forward. We call the
            # unwrapped forward (a wrapper loop otherwise breaks inspect.unwrap),
            # so that recording never happens — set it directly or the helpers
            # emit CPU tensors that fail to cat with on-device ones.
            import compile_neuron
            compile_neuron.set_index_helper_device(input_ids.device)

            logits = self._raw_forward(self.inner, input_ids,
                                       start_pos.reshape(()))

            outs = [logits]
            for mod, name in self._slots:
                outs.append(sink.get(mod, name))
            return tuple(outs)


def _make_instance(mode):
    """A BaseModelInstance for one mode, with its output/state aliases."""
    from neuronx_distributed.trace.model_builder import BaseModelInstance

    inst = BaseModelInstance(module_cls=None, input_output_aliases={})

    def load_module():
        import decode_patches
        import prefill_patches
        import shard_loader

        m, hf = _build_hf_model()
        if _LOAD_WEIGHTS:
            shard_loader.load_rank_weights(
                m, _MODEL_PATH, rank=hf.rank, world_size=hf.world_size,
            )
        # Install a dispatcher that picks the mode at *call* time.
        #
        # Both modes patch the same module-level hf.Attention.forward, and
        # ModelBuilder calls load_module() for every key before tracing any of
        # them, so a per-mode install has the second one silently win for both:
        # decode's forwards would run inside prefill's trace, reading a `pos`
        # that prefill never sets (None -> a zero-width index tensor, surfacing
        # as "size of tensor a (128) must match tensor b (0)").
        #
        # So install once and route on `active`, which the wrapper sets before
        # calling forward. Both patch state dicts stay live; only one is used per
        # trace.
        state = _install_dual_patches(hf)
        fwd = hf.Transformer.forward
        raw_forward = getattr(fwd, "__wrapped__", fwd)

        slots, aliases = decode_patches.collect_state_aliases(
            m, n_real_outputs=1)
        inst.module = _JointWrapper(m, hf, mode, slots, state, raw_forward)
        inst.input_output_aliases = [aliases]

    inst.load_module = load_module
    return inst


def _example_inputs(mode):
    n_active = 1 if mode == "decode" else _PREFILL_LEN
    ids = torch.zeros(_MAX_BATCH_SIZE, n_active, dtype=torch.long)
    # start_pos is shape (1,), NOT 0-d. ModelBuilder keys its shape router on
    # `list(tensor.shape)`, which is `[]` for a 0-d tensor, and its flattener
    # does not round-trip that reliably — the value arrives in the graph as a
    # 0-*element* tensor, so `j <= pos` broadcasts 128 against 0 and raises
    # "size of tensor a (128) must match the size of tensor b (0)".
    # parallel_model_trace tolerated 0-d; this path does not. The wrapper
    # reshapes to () before use, so nothing downstream changes.
    #
    # Decode traces at position 1, not 0: position 0 would exercise the
    # degenerate case of any pos-derived mask, and decode is only ever entered
    # at pos >= 1.
    pos = torch.tensor([1 if mode == "decode" else 0], dtype=torch.int32)
    return [(ids, pos)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tp", type=int, default=32)
    ap.add_argument("--n-layers", type=int, default=5)
    ap.add_argument("--seq-len", type=int, default=256,
                    help="max_seq_len: how far generation can run. Sizes the KV "
                         "cache and the RoPE table.")
    ap.add_argument("--prefill-len", type=int, default=128,
                    help="Width of the prefill graph's input. The prompt must "
                         "fill it exactly (the head emits only the last "
                         "position).")
    ap.add_argument("--max-batch-size", type=int, default=1)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--model-path", default=paths.model_path())
    ap.add_argument("--config", default=paths.config_json())
    ap.add_argument("--load-weights", action="store_true")
    ap.add_argument("--opt-level", default="1", choices=["1", "2", "3"])
    ap.add_argument("--compiler-workdir", default=None)
    args = ap.parse_args()

    if args.prefill_len > args.seq_len:
        raise SystemExit(
            f"--prefill-len {args.prefill_len} exceeds --seq-len {args.seq_len}; "
            f"the prompt has to fit in the cache it is written into."
        )

    os.environ["DSV4_MODEL_PATH"] = args.model_path
    os.environ["DSV4_CONFIG"] = args.config
    os.environ["DSV4_N_LAYERS"] = str(args.n_layers)
    os.environ["DSV4_SEQ_LEN"] = str(args.seq_len)
    os.environ["DSV4_PREFILL_LEN"] = str(args.prefill_len)
    os.environ["DSV4_MAX_BATCH_SIZE"] = str(args.max_batch_size)
    os.environ["DSV4_LOAD_WEIGHTS"] = "1" if args.load_weights else "0"

    global _MODEL_PATH, _CONFIG_JSON, _N_LAYERS, _SEQ_LEN, _PREFILL_LEN
    global _MAX_BATCH_SIZE, _LOAD_WEIGHTS
    _MODEL_PATH, _CONFIG_JSON = args.model_path, args.config
    _N_LAYERS, _SEQ_LEN = args.n_layers, args.seq_len
    _PREFILL_LEN, _MAX_BATCH_SIZE = args.prefill_len, args.max_batch_size
    _LOAD_WEIGHTS = args.load_weights

    from neuronx_distributed.trace.model_builder import ModelBuilder

    import compile_neuron
    with open(args.config) as f:
        compile_neuron.preflight_host_ram(
            json.load(f), args.n_layers, args.tp, None, inline_weights=False)

    workdir = args.compiler_workdir or f"/tmp/dsv4_joint_ws_tp{args.tp}"
    os.makedirs(workdir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)

    compiler_args = (f"--model-type=transformer --auto-cast=none "
                     f"-O{args.opt_level} --enable-mixed-precision-accumulation")
    print(f"[joint] tp={args.tp} n_layers={args.n_layers} "
          f"seq_len={args.seq_len} prefill_len={args.prefill_len} "
          f"weights={'real' if args.load_weights else 'random-init'}")
    print(f"[joint] compiler_args={compiler_args}")
    print(f"[joint] workdir={workdir}")

    # An empty checkpoint: NxD's own sharding cannot handle this model (it does
    # not recognise HF's same-named parallel classes), so weights go in per rank
    # via nxd_model.initialize() after tracing. See the module docstring.
    def checkpoint_loader():
        return {}

    builder = ModelBuilder(
        router=None, tp_degree=args.tp, checkpoint_loader=checkpoint_loader,
        compiler_workdir=workdir, logical_nc_config=2,
    )
    for key, mode in ((PREFILL_KEY, "prefill"), (DECODE_KEY, "decode")):
        builder.add(key=key, model_instance=_make_instance(mode),
                    example_inputs=_example_inputs(mode),
                    compiler_args=compiler_args)
        print(f"[joint] registered {key}: "
              f"input width {_example_inputs(mode)[0][0].shape[1]}")

    t0 = time.perf_counter()
    model = builder.trace(initialize_model_weights=False)
    print(f"[joint] traced both graphs in {time.perf_counter() - t0:.1f}s")

    torch.jit.save(model, os.path.join(args.out_dir, "joint_model.pt"))
    with open(os.path.join(args.out_dir, "config.json"), "w") as f:
        json.dump({
            "tp": args.tp, "n_layers": args.n_layers, "seq_len": args.seq_len,
            "prefill_len": args.prefill_len,
            "max_batch_size": args.max_batch_size,
            "load_weights": args.load_weights,
        }, f, indent=2)
    print(f"[joint] saved -> {args.out_dir}")
    print("[joint] NOTE: weights are NOT in this artifact; run_joint.py loads "
          "them per rank via shard_loader + nxd_model.initialize()")


if __name__ == "__main__":
    main()
