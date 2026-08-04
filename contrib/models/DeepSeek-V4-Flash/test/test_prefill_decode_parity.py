"""Prefill vs decode-stepped-to-the-same-position, CPU, TP=1.

The gap in the existing gates: test_prefill_functional_vs_inplace compares the two
PREFILL formulations, and test_dsv4_model compares each mode against its own
reference. Nothing compares prefill(seqlen=N) against N sequential decode steps,
which is exactly the equivalence the joint path depends on.

TP=1 so collectives are identity and CPU numerics are meaningful (see the
mock_distributed note in CONTROL_EXPERIMENT.md).
"""
import torch, sys
sys.path.insert(0, "src")
from neuronx_distributed.trace.mock_torchdist import mock_distributed
from neuronx_distributed.parallel_layers import parallel_state
import run_dsv4_device as rd

N_LAY, SEQ, PLEN = 3, 256, 16      # short prompt: fast, still exercises the ring
rd._N_LAYERS, rd._SEQ_LEN, rd._PREFILL_LEN, rd._TP = N_LAY, SEQ, PLEN, 1
rd._LOAD_WEIGHTS = False

def init(m, seed=0):
    torch.manual_seed(seed)
    with torch.no_grad():
        for n, p in m.named_parameters():
            if "norm" in n or n.endswith("attn_sink"): p.fill_(1.0)
            elif "ape" in n: p.zero_()
            else: p.copy_((torch.randn(p.shape)*0.02).to(p.dtype))

with mock_distributed(world_size=1):
    torch.distributed.init_process_group("xla", rank=0, world_size=1)
    parallel_state.initialize_model_parallel(1, 1, 1, skip_collective_init=True)
    ids = ((torch.arange(PLEN, dtype=torch.int32)*977+101) % 129280).unsqueeze(0)

    # A: prefill in one call
    mp, hfp = rd._build("prefill"); init(mp.inner)
    with torch.no_grad():
        pos = torch.arange(PLEN, dtype=torch.int32).unsqueeze(0)
        a = mp(ids, pos)[0].float().flatten()
    print(f"prefill: finite={bool(torch.isfinite(a).all())} top1={int(a.argmax())} "
          f"max={float(a.max()):.3f}", flush=True)

    # B: decode, PLEN sequential steps, same weights
    rd._PREFILL_LEN = PLEN
    md, hfd = rd._build("decode"); init(md.inner)
    # Emulate NxD's output aliasing. On device, graph output n+i is written back
    # over past_key_values[i] after every call; on CPU nothing does that, so a
    # decode loop would restart from a fresh cache each step. Without this the
    # comparison measures a harness gap rather than the model.
    with torch.no_grad():
        for p in range(PLEN):
            outs = md(ids[:, p:p+1], torch.tensor([[p]], dtype=torch.int32))
            b = outs[0]
            for i, t in enumerate(outs[1:]):
                md.kv_mgr.past_key_values[i].copy_(
                    t.to(md.kv_mgr.past_key_values[i].dtype))
        b = b.float().flatten()
    print(f"decode : finite={bool(torch.isfinite(b).all())} top1={int(b.argmax())} "
          f"max={float(b.max()):.3f}", flush=True)

    cos = float(torch.nn.functional.cosine_similarity(a, b, dim=0))
    rel = float((a-b).abs().mean()/b.abs().mean().clamp_min(1e-6))
    ka, kb = set(a.topk(20).indices.tolist()), set(b.topk(20).indices.tolist())
    print(f"cosine={cos:.6f} rel={rel:.4f} top20_overlap={len(ka&kb)}/20", flush=True)
