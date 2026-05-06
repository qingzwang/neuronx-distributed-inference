"""Monkey-patches that make the ultralytics YOLO26 graph Neuron-compilable.

Neuron's XLA lowering for `torch.Tensor.split` with unequal chunk sizes emits
an incorrect dynamic slice on trn2 (observed: split of a `(B, H, 128, N)` tensor
into `[32, 32, 64]` along dim=2 returns numerically wrong values, with
>30x error vs. CPU). Replacing those calls with explicit contiguous slices
produces results that match CPU to within fp32 rounding noise.

We patch only the minimum surface: `Attention.forward` inside the YOLO26 head
block (`C2PSA -> PSABlock -> Attention`). Everything else stays untouched.
"""

from __future__ import annotations

import types

import torch
from ultralytics.nn.modules.block import Attention


def _neuron_safe_attention_forward(self, x: torch.Tensor) -> torch.Tensor:
    """Drop-in replacement for `Attention.forward` that avoids `torch.split`."""
    B, C, H, W = x.shape
    N = H * W
    qkv = self.qkv(x)
    v_qkv = qkv.view(B, self.num_heads, self.key_dim * 2 + self.head_dim, N)
    kd = self.key_dim
    hd = self.head_dim
    q = v_qkv[:, :, 0:kd, :].contiguous()
    k = v_qkv[:, :, kd : 2 * kd, :].contiguous()
    v = v_qkv[:, :, 2 * kd : 2 * kd + hd, :].contiguous()

    attn = (q.transpose(-2, -1) @ k) * self.scale
    attn = attn.softmax(dim=-1)
    x = (v @ attn.transpose(-2, -1)).view(B, C, H, W) + self.pe(v.reshape(B, C, H, W))
    x = self.proj(x)
    return x


def patch_attention_modules(model: torch.nn.Module) -> int:
    """Bind the safe forward to every `Attention` instance inside `model`."""
    count = 0
    for module in model.modules():
        if isinstance(module, Attention):
            module.forward = types.MethodType(_neuron_safe_attention_forward, module)
            count += 1
    return count
