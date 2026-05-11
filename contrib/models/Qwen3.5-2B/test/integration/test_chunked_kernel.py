# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Numerical agreement of NKI chunked kernel with FLA reference.

The NKI per-chunk kernel `deltanet_chunk_step` should produce the same
output and final state as `fla.ops.gated_delta_rule.chunk_gated_delta_rule`
to within fp32 noise (~1e-6 max abs err).

This test exercises the kernel directly via torch_xla (no NxDI compile),
so it's cheaper than the full model integration test. It catches kernel
regressions without needing a multi-minute model recompile.
"""

import os
import sys
import unittest

import torch
import torch.nn.functional as F

_CONTRIB_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _CONTRIB_ROOT not in sys.path:
    sys.path.insert(0, _CONTRIB_ROOT)

from src.fla_reference import chunk_gated_delta_rule


class TestChunkedNKIKernel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch_xla.core.xla_model as xm
            cls.xla_dev = xm.xla_device()
        except Exception as e:
            raise unittest.SkipTest(f"torch_xla unavailable: {e}")
        try:
            from src.nki_kernels.nki_deltanet_chunked import deltanet_chunk_step
            cls.nki_chunk_step = deltanet_chunk_step
        except Exception as e:
            raise unittest.SkipTest(f"NKI chunked kernel import failed: {e}")

    def _run_nki_chunked(self, q, k, v, g, beta):
        """Run the NKI chunked kernel via per-chunk Python loop on XLA."""
        B, H, S, K = q.shape
        V = v.shape[-1]
        chunk_size = 128
        BH = B * H
        assert S % chunk_size == 0, "test inputs must be chunk-aligned"
        num_chunks = S // chunk_size

        q_n = F.normalize(q.float(), p=2, dim=-1) * (1.0 / K ** 0.5)
        k_n = F.normalize(k.float(), p=2, dim=-1)

        g_cs = g.reshape(B, H, num_chunks, chunk_size).cumsum(dim=-1)
        gc_chunks = g_cs.unsqueeze(-1).expand(-1, -1, -1, -1, V) \
            .reshape(BH, num_chunks, chunk_size, V).contiguous()
        gl_chunks = g_cs[:, :, :, -1:].expand(-1, -1, -1, chunk_size) \
            .unsqueeze(-1).expand(-1, -1, -1, -1, V) \
            .reshape(BH, num_chunks, chunk_size, V).contiguous()
        q_chunks = q_n.reshape(BH, num_chunks, chunk_size, K).contiguous()
        k_chunks = k_n.reshape(BH, num_chunks, chunk_size, K).contiguous()
        v_chunks = v.float().reshape(BH, num_chunks, chunk_size, V).contiguous()
        beta_chunks = beta.reshape(BH, num_chunks, chunk_size).unsqueeze(-1) \
            .expand(-1, -1, -1, V) \
            .reshape(BH, num_chunks, chunk_size, V).contiguous()

        dev = self.xla_dev
        lower_mask = torch.tril(torch.ones(chunk_size, chunk_size, dtype=torch.float32, device=dev), diagonal=-1)
        identity = torch.eye(chunk_size, dtype=torch.float32, device=dev)
        lower_mask_diag = torch.tril(torch.ones(chunk_size, chunk_size, dtype=torch.float32, device=dev), diagonal=0)

        q_chunks = q_chunks.to(dev); k_chunks = k_chunks.to(dev); v_chunks = v_chunks.to(dev)
        beta_chunks = beta_chunks.to(dev); gc_chunks = gc_chunks.to(dev); gl_chunks = gl_chunks.to(dev)

        out_per_bh = []
        state_per_bh = []
        for bh in range(BH):
            state = torch.zeros(K, V, dtype=torch.float32, device=dev)
            out_chunks_list = []
            for c_idx in range(num_chunks):
                out_chunk, state = self.nki_chunk_step(
                    q_chunks[bh, c_idx], k_chunks[bh, c_idx], v_chunks[bh, c_idx],
                    beta_chunks[bh, c_idx], gc_chunks[bh, c_idx], gl_chunks[bh, c_idx],
                    state, lower_mask, identity, lower_mask_diag,
                )
                out_chunks_list.append(out_chunk.cpu())
            out_per_bh.append(torch.cat(out_chunks_list, dim=0))
            state_per_bh.append(state.cpu())

        out = torch.stack(out_per_bh, dim=0).reshape(B, H, S, V)
        final_state = torch.stack(state_per_bh, dim=0).reshape(B, H, K, V)
        return out, final_state

    def _check(self, B, H, S, K=128, V=128, atol=1e-5, seed=0):
        gen = torch.Generator().manual_seed(seed)
        q = torch.randn(B, H, S, K, generator=gen) * 0.5
        k = torch.randn(B, H, S, K, generator=gen) * 0.5
        v = torch.randn(B, H, S, V, generator=gen) * 0.5
        g = -F.softplus(torch.randn(B, H, S, generator=gen))
        beta = torch.sigmoid(torch.randn(B, H, S, generator=gen))

        o_ref, s_ref = chunk_gated_delta_rule(
            q, k, v, g, beta, chunk_size=64,
            use_qk_l2norm_in_kernel=True, output_final_state=True,
        )
        o_nki, s_nki = self._run_nki_chunked(q, k, v, g, beta)

        self.assertLess((o_nki - o_ref).abs().max().item(), atol,
                        f"output mismatch B={B} H={H} S={S}")
        self.assertLess((s_nki - s_ref).abs().max().item(), atol,
                        f"final_state mismatch B={B} H={H} S={S}")

    def test_single_chunk_single_head(self):
        self._check(1, 1, 128)

    def test_single_chunk_multi_head(self):
        self._check(1, 4, 128)

    def test_two_chunks_single_head(self):
        self._check(1, 1, 256)

    def test_two_chunks_multi_head(self):
        self._check(1, 4, 256)

    def test_four_chunks_multi_head(self):
        # This exercises the bucket=512 case (4 chunks × 16 heads in production).
        self._check(1, 2, 512, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
