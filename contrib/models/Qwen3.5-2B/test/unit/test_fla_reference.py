# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU unit tests for the local flash-linear-attention reference impl.

These do not require Neuron and validate that:
  * the chunked and recurrent gated-delta-rule paths agree on small
    inputs (the recurrent path is the spec — chunked must match)
  * the chunked path is exact across chunk boundaries (a regression test
    for the per-chunk Neumann-resolved correction matrix)
  * initial-state carry-over works (chunk 0 with state == manual prefix)
"""

import os
import sys
import unittest

import torch
import torch.nn.functional as F

_CONTRIB_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _CONTRIB_ROOT not in sys.path:
    sys.path.insert(0, _CONTRIB_ROOT)

from src.fla_reference import (
    chunk_gated_delta_rule,
    recurrent_gated_delta_rule,
)


def _random_inputs(B=1, H=2, S=192, K=16, V=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(B, H, S, K, generator=g)
    k = torch.randn(B, H, S, K, generator=g)
    v = torch.randn(B, H, S, V, generator=g)
    g_log = -F.softplus(torch.randn(B, H, S, generator=g))   # negative log-decay
    beta = torch.sigmoid(torch.randn(B, H, S, generator=g))
    return q, k, v, g_log, beta


class TestFLAReference(unittest.TestCase):
    def test_chunked_matches_recurrent_short(self):
        q, k, v, g, beta = _random_inputs(S=128)
        o_rec, s_rec = recurrent_gated_delta_rule(q, k, v, g, beta)
        o_ch, s_ch = chunk_gated_delta_rule(q, k, v, g, beta, chunk_size=64)
        self.assertLess((o_rec - o_ch).abs().max().item(), 1e-4)
        self.assertLess((s_rec - s_ch).abs().max().item(), 1e-4)

    def test_chunked_matches_recurrent_long(self):
        q, k, v, g, beta = _random_inputs(S=512)
        o_rec, s_rec = recurrent_gated_delta_rule(q, k, v, g, beta)
        o_ch, s_ch = chunk_gated_delta_rule(q, k, v, g, beta, chunk_size=64)
        self.assertLess((o_rec - o_ch).abs().max().item(), 1e-3)

    def test_chunked_handles_unaligned_seq_len(self):
        q, k, v, g, beta = _random_inputs(S=200)  # not a multiple of chunk_size
        o_rec, _ = recurrent_gated_delta_rule(q, k, v, g, beta)
        o_ch, _ = chunk_gated_delta_rule(q, k, v, g, beta, chunk_size=64)
        self.assertEqual(o_ch.shape, o_rec.shape)
        self.assertLess((o_rec - o_ch).abs().max().item(), 1e-3)

    def test_initial_state_carry_over(self):
        q, k, v, g, beta = _random_inputs(S=256)
        S = q.shape[2]
        # full pass
        o_full, s_full = recurrent_gated_delta_rule(q, k, v, g, beta)
        # split-then-resume
        S1 = 128
        o1, s1 = recurrent_gated_delta_rule(q[:, :, :S1], k[:, :, :S1],
                                            v[:, :, :S1], g[:, :, :S1],
                                            beta[:, :, :S1])
        o2, s2 = recurrent_gated_delta_rule(q[:, :, S1:], k[:, :, S1:],
                                            v[:, :, S1:], g[:, :, S1:],
                                            beta[:, :, S1:],
                                            initial_state=s1)
        self.assertLess((s_full - s2).abs().max().item(), 1e-4)
        self.assertLess((o_full[:, :, :S1] - o1).abs().max().item(), 1e-4)
        self.assertLess((o_full[:, :, S1:] - o2).abs().max().item(), 1e-4)


if __name__ == "__main__":
    unittest.main()
