"""Call compile_joint's OWN _make_instance/_example_inputs under mock_distributed.

If this fails, the bug is in those functions. If it passes, the bug is in how
ModelBuilder drives them (ordering, parameter freeing, shared signature).
"""
import os, sys, torch
sys.path.insert(0,'src'); sys.path.insert(0,'.')
os.environ.setdefault("DSV4_N_LAYERS","5")
os.environ.setdefault("DSV4_SEQ_LEN","256")
os.environ.setdefault("DSV4_PREFILL_LEN","128")
from neuronx_distributed.trace.mock_torchdist import mock_distributed
from neuronx_distributed.parallel_layers import parallel_state
from torch_neuronx.xla_impl.trace import generate_hlo
import compile_joint as cj
cj._N_LAYERS, cj._SEQ_LEN, cj._PREFILL_LEN = 5, 256, 128

WS = 2
with mock_distributed(world_size=WS):
    torch.distributed.init_process_group("xla", rank=0, world_size=WS)
    parallel_state.initialize_model_parallel(WS,1,1,skip_collective_init=True)
    parallel_state.set_aot_mode(True)
    # ModelBuilder's ACTUAL order: load every module first, then trace.
    insts = {m: cj._make_instance(m) for m in ("prefill", "decode")}
    for m_ in ("prefill", "decode"):
        insts[m_].load_module()
    for mode in ("prefill", "decode"):
        inst = insts[mode]
        func, aliases = inst.get(0)
        ex = cj._example_inputs(mode)[0]
        print(f"[{mode}] ids={tuple(ex[0].shape)} pos={tuple(ex[1].shape)} "
              f"n_aliases={len(aliases)}", flush=True)
        try:
            generate_hlo(func, ex, aliases, inline_weights_to_neff=False,
                         return_weights=False, output_aliased_tensor=False,
                         cpu_backend=True, preserve_parameters=False)
            print(f"  [ok  ] {mode}")
        except Exception as e:
            print(f"  [FAIL] {mode}: {type(e).__name__}: "
                  f"{str(e).split(chr(10))[0][:100]}")
