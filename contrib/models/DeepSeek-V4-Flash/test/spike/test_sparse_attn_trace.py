"""Spike: does torch_neuronx.trace compile our CPU sparse_attn implementation?"""
import sys, os
sys.path.insert(0, 'src')
sys.path.insert(0, 'src/hf_reference')
os.environ["NEURON_CC_FLAGS"] = "--verbose=INFO -O1 --target=trn2 --auto-cast=none"

import torch
import torch_neuronx
from hf_reference.kernel_cpu import sparse_attn

# Simplified inputs (single head-group, no batch)
B = 1
M = 8      # query positions
H = 4      # heads
D = 32     # head_dim
N = 32     # KV cache size
TOPK = 16  # top-k gather size (padded, some may be -1)

def make_inputs(seed=0):
    torch.manual_seed(seed)
    q = torch.randn(B, M, H, D, dtype=torch.bfloat16)
    kv = torch.randn(B, N, D, dtype=torch.bfloat16)
    attn_sink = torch.randn(H, dtype=torch.float32)
    # topk indices: valid = [0, N-1], -1 = padded
    topk_idxs = torch.randint(0, N, (B, M, TOPK), dtype=torch.int32)
    # mask ~25% as -1
    topk_idxs[torch.rand(B, M, TOPK) < 0.25] = -1
    return q, kv, attn_sink, topk_idxs

def fn(q, kv, attn_sink, topk_idxs):
    return sparse_attn(q, kv, attn_sink, topk_idxs, 1.0 / D**0.5)

# CPU reference
inputs = make_inputs()
ref = fn(*inputs)
print(f'CPU ref output shape: {ref.shape}, dtype: {ref.dtype}')
print(f'CPU ref mean={ref.float().mean().item():.4f} std={ref.float().std().item():.4f}')

# Trace to Neuron
print('\n[trace] compiling...')
import time
t = time.perf_counter()
try:
    traced = torch_neuronx.trace(
        fn, inputs,
        compiler_args=["--model-type=transformer", "-O1", "--auto-cast=none"],
    )
    print(f'[trace] done in {time.perf_counter()-t:.1f}s')
    out = traced(*inputs)
    print(f'Neuron out shape: {out.shape}, dtype: {out.dtype}')
    diff = (ref.float() - out.float()).abs()
    print(f'|neuron - ref| max={diff.max().item():.4f} mean={diff.mean().item():.4f} '
          f'ref abs mean={ref.float().abs().mean().item():.4f}')
except Exception as e:
    import traceback; traceback.print_exc()
    print(f'ERROR: {e}')
