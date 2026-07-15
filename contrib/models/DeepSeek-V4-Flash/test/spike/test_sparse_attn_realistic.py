"""Spike 2: realistic shapes matching DeepSeek-V4-Flash attention.
Also test hc_split_sinkhorn."""
import sys, os
sys.path.insert(0, 'src')
sys.path.insert(0, 'src/hf_reference')

import torch
import torch_neuronx
from hf_reference.kernel_cpu import sparse_attn, hc_split_sinkhorn

# --- sparse_attn realistic ---
# From DeepSeek-V4 config: n_heads=64 (TP=8 -> 8 local), head_dim=512
# For prefill seq_len=64 with window=128, KV cache = 64 (all in window, no compress)
# topk_idxs = window indices = up to 64 valid
B = 1
M = 64     # query positions
H = 8      # local heads (n_heads // tp)
D = 512    # head_dim
N = 128    # KV cache size (window + compressed slots)
TOPK = 64  # for CTE seq_len=64 with window=128, only window active

def make_inputs(seed=0):
    torch.manual_seed(seed)
    q = torch.randn(B, M, H, D, dtype=torch.bfloat16)
    kv = torch.randn(B, N, D, dtype=torch.bfloat16)
    attn_sink = torch.randn(H, dtype=torch.float32)
    topk_idxs = torch.randint(0, N, (B, M, TOPK), dtype=torch.int32)
    return q, kv, attn_sink, topk_idxs

def fn(q, kv, attn_sink, topk_idxs):
    return sparse_attn(q, kv, attn_sink, topk_idxs, 1.0 / D**0.5)

inputs = make_inputs()
ref = fn(*inputs)
print(f'[sparse_attn realistic] output shape: {list(ref.shape)}')

import time
t = time.perf_counter()
traced = torch_neuronx.trace(fn, inputs, compiler_args=["--model-type=transformer", "-O1", "--auto-cast=none"])
print(f'[sparse_attn] compile: {time.perf_counter()-t:.1f}s')
out = traced(*inputs)
diff = (ref.float() - out.float()).abs()
print(f'[sparse_attn] max diff={diff.max().item():.4f} mean={diff.mean().item():.4f} ref |mean|={ref.float().abs().mean().item():.4f}')

# --- hc_split_sinkhorn ---
S = 64       # tokens
HC = 4       # hc_mult
MIX_HC = (2 + HC) * HC  # 24
def make_hc_inputs():
    torch.manual_seed(1)
    mixes = torch.randn(1, S, MIX_HC, dtype=torch.float32)
    scale = torch.randn(3, dtype=torch.float32) * 0.1
    base = torch.randn(MIX_HC, dtype=torch.float32) * 0.1
    return mixes, scale, base

hc_inputs = make_hc_inputs()
def hc_fn(mixes, scale, base):
    return hc_split_sinkhorn(mixes, scale, base, 4, 20, 1e-6)

pre_ref, post_ref, comb_ref = hc_fn(*hc_inputs)
print(f'\n[hc_split_sinkhorn] pre={list(pre_ref.shape)} post={list(post_ref.shape)} comb={list(comb_ref.shape)}')

t = time.perf_counter()
try:
    hc_traced = torch_neuronx.trace(hc_fn, hc_inputs, compiler_args=["--model-type=transformer", "-O1", "--auto-cast=none"])
    print(f'[hc_split_sinkhorn] compile: {time.perf_counter()-t:.1f}s')
    p_out, po_out, c_out = hc_traced(*hc_inputs)
    p_diff = (pre_ref - p_out).abs()
    po_diff = (post_ref - po_out).abs()
    c_diff = (comb_ref - c_out).abs()
    print(f'[hc] pre max={p_diff.max().item():.6f}, post max={po_diff.max().item():.6f}, comb max={c_diff.max().item():.6f}')
except Exception as e:
    import traceback; traceback.print_exc()
    print(f'HC ERROR: {e}')
