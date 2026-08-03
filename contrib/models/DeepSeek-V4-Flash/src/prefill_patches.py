# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Functional (sink-based) prefill forwards, so prefill can share decode's cache.

Why this exists
---------------
The existing prefill path in `compile_neuron.apply_xla_patches` mutates the KV
cache in place:

    self.kv_cache[:bsz, :seqlen] = kv                 # Attention
    self.kv_state[:bsz, :ratio] = kv[:, cutoff-ratio:] # Compressor

That is fine when prefill owns a private cache that nothing else reads, which is
how it works today: two separately traced artifacts, two caches, no sharing.

It stops being fine the moment prefill and decode share one aliased cache
(JOINT_INFERENCE.md). Shared state has to be *returned as a graph output* so NxD
can alias it back over the input, and an in-place slice assignment produces no
value to return — the write happens to a tensor the graph treats as read-only
input, and the update is invisible to the next call. So prefill's writes have to
become functional, exactly as `decode_patches` already did for the decode branch.

This module is the `start_pos == 0` counterpart of `decode_patches`, and it
deliberately reuses that module's `StateSink`: same owner/offset redirect for
HF's `compressor.kv_cache = self.kv_cache[:, win:]` view aliasing, same
`put`/`get` protocol, so a joint graph can hand one sink to both branches.

What differs from decode, and why
---------------------------------
Prefill is *easier* than decode in the one way that dominated decode's design:
`start_pos` is a compile-time 0, so nothing here needs a tensor-valued position.
All the index math stays Python-level and shape-static. What it needs instead is
handling of the whole-sequence writes, which decode never does:

  * **Attention window.** Decode writes one ring slot. Prefill writes `seqlen`
    positions, and when `seqlen > win` only the last `win` of them survive, split
    across the ring boundary at `seqlen % win`. Expressed as one `index_copy`
    over precomputed absolute slots rather than HF's two-way slice assignment.

  * **Compressor.** Decode compresses at most one entry per call. Prefill
    compresses `seqlen // ratio` at once, and separately has to seed `kv_state` /
    `score_state` with the *tail* that did not fill a whole compression block, so
    a following decode step continues the ring correctly. That seeding is the
    part that matters for joint inference: get it wrong and prefill looks fine in
    isolation while the first decode token after it is subtly wrong.

  * **Indexer.** Same as decode apart from the causal mask, which prefill needs
    (a query at position p may only see compressed blocks that closed at or
    before p) and decode does not.

`should_compress` is a Python bool here, not a tensor: it depends only on
`seqlen` and `ratio`, both known at trace time. No masked-write trick needed.
"""

import torch

from decode_patches import StateSink, _slot  # noqa: F401  (shared protocol)


def make_prefill_forwards(hf_mod):
    """Return (compressor, indexer, attention) prefill forwards, sink-based.

    Mirrors `decode_patches.make_prefill_forwards`'s shape so
    `apply_joint_patches` can install either set behind one dispatch.
    """
    import torch as _torch
    import xla_ops

    def compressor_prefill(self, x, sink):
        """HF Compressor.forward, start_pos == 0 branch, functional writes.

        Returns the compressed entries (or None when the prompt is shorter than
        one compression block, matching HF's early return), and leaves both ring
        buffers seeded for the decode steps that follow.
        """
        bsz, seqlen, _ = x.size()
        ratio, overlap = self.compress_ratio, self.overlap
        d, rd = self.head_dim, self.rope_head_dim
        dtype = x.dtype
        x = x.float()
        kv = self.wkv(x)
        score = self.wgate(x)

        should_compress = seqlen >= ratio
        remainder = seqlen % ratio
        cutoff = seqlen - remainder
        offset = ratio if overlap else 0

        kv_state = sink.get(self, "kv_state")
        score_state = sink.get(self, "score_state")

        # Seed the overlap half with the last full block before the cutoff, so a
        # subsequent decode step's overlap_transform has the right history.
        if overlap and cutoff >= ratio:
            head = _torch.arange(ratio, device=kv.device).long()
            kv_state = kv_state.index_copy(
                1, head, kv[:, cutoff - ratio:cutoff].to(kv_state.dtype))
            score_state = score_state.index_copy(
                1, head, (score[:, cutoff - ratio:cutoff]
                          + self.ape).to(score_state.dtype))

        # The ragged tail (positions past the last whole block) is not compressed
        # now; it becomes the live window a following decode step appends to.
        if remainder > 0:
            tail_slots = _torch.arange(
                offset, offset + remainder, device=kv.device).long()
            kv_state = kv_state.index_copy(
                1, tail_slots, kv[:, cutoff:].to(kv_state.dtype))
            score_state = score_state.index_copy(
                1, tail_slots,
                (score[:, cutoff:] + self.ape[:remainder]).to(score_state.dtype))
            kv = kv[:, :cutoff]
            score = score[:, :cutoff]

        sink.put(self, "kv_state", kv_state)
        sink.put(self, "score_state", score_state)

        if not should_compress:
            # HF returns None here; the caller skips the concat.
            return None

        kv = kv.unflatten(1, (-1, ratio))
        score = score.unflatten(1, (-1, ratio)) + self.ape
        if overlap:
            kv = self.overlap_transform(kv, 0)
            score = self.overlap_transform(score, float("-inf"))
        kv = (kv * score.softmax(dim=2)).sum(dim=2)

        kvc = self.norm(kv.to(dtype))
        freqs_cis = self.freqs_cis[:cutoff:ratio]
        hf_mod.apply_rotary_emb(kvc[..., -rd:], freqs_cis)
        if self.rotate:
            kvc = hf_mod.rotate_activation(kvc)
            hf_mod.fp4_act_quant(kvc, hf_mod.fp4_block_size, True)
        else:
            hf_mod.act_quant(kvc[..., :-rd], 64, hf_mod.scale_fmt,
                             hf_mod.scale_dtype, True)

        n_comp = cutoff // ratio
        cache = sink.get(self, "kv_cache")
        slots = _torch.arange(n_comp, device=cache.device).long()
        sink.put(self, "kv_cache",
                 cache.index_copy(1, slots, kvc[:, :n_comp].to(cache.dtype)))
        return kvc

    def indexer_prefill(self, self_x, qr, offset, sink):
        """HF Indexer.forward, start_pos == 0 branch."""
        x = self_x
        bsz, seqlen, _ = x.size()
        ratio, rd = self.compress_ratio, self.rope_head_dim
        freqs_cis = self.freqs_cis[:seqlen]

        q = self.wq_b(qr).unflatten(-1, (self.n_local_heads, self.head_dim))
        hf_mod.apply_rotary_emb(q[..., -rd:], freqs_cis)
        q = hf_mod.rotate_activation(q)
        hf_mod.fp4_act_quant(q, hf_mod.fp4_block_size, True)

        compressor_prefill(self.compressor, x, sink)

        weights = self.weights_proj(x) * (
            self.softmax_scale * self.n_heads ** -0.5)

        # Score against this layer's whole compressed cache and mask the part
        # that is not causally visible, rather than slicing to end_pos // ratio:
        # a runtime-length slice is not shape-static, and after the compressor
        # write above the tail is zeros anyway.
        cache = sink.get(self, "kv_cache")
        index_score = _torch.einsum("bshd,btd->bsht", q, cache[:bsz])
        index_score = (index_score.relu() * weights.unsqueeze(-1)).sum(dim=2)
        if hf_mod.world_size > 1:
            import collectives
            collectives.all_reduce(index_score)

        dev = index_score.device
        # Block j closed at position (j+1)*ratio - 1, so query p sees it iff
        # j < (p+1) // ratio. Same expression HF builds with repeat().
        causal_limit = (
            _torch.arange(1, seqlen + 1, device=dev).unsqueeze(1) // ratio
        )
        cols = _torch.arange(cache.size(1), device=dev).unsqueeze(0)
        mask = cols + _torch.zeros(seqlen, 1, dtype=cols.dtype,
                                   device=dev) >= causal_limit
        index_score = index_score + _torch.where(
            mask, float("-inf"), 0.0).to(index_score.dtype)

        k = min(self.index_topk, cache.size(1))
        topk_idxs = xla_ops.topk_indices_unordered(index_score, k)
        return _torch.where(topk_idxs >= causal_limit, -1, topk_idxs + offset)

    def attention_prefill(self, x, sink):
        """HF Attention.forward, start_pos == 0 branch, functional writes."""
        bsz, seqlen, _ = x.size()
        win, ratio = self.window_size, self.compress_ratio
        rd = self.rope_head_dim
        freqs_cis = self.freqs_cis[:seqlen]

        qr = q = self.q_norm(self.wq_a(x))
        q = self.wq_b(q).unflatten(-1, (self.n_local_heads, self.head_dim))
        q = q * _torch.rsqrt(q.square().mean(-1, keepdim=True) + self.eps)
        hf_mod.apply_rotary_emb(q[..., -rd:], freqs_cis)

        kv = self.kv_norm(self.wkv(x))
        hf_mod.apply_rotary_emb(kv[..., -rd:], freqs_cis)
        hf_mod.act_quant(kv[..., :-rd], 64, hf_mod.scale_fmt,
                         hf_mod.scale_dtype, True)

        topk_idxs = hf_mod.get_window_topk_idxs(win, bsz, seqlen, 0)
        if ratio:
            # In prefill the compressed entries are appended after the live kv,
            # so the offset is kv's own length, matching HF's
            # `offset = kv.size(1) if start_pos == 0 else win`.
            if self.indexer is not None:
                comp_idxs = indexer_prefill(self.indexer, x, qr, seqlen, sink)
            else:
                comp_idxs = hf_mod.get_compress_topk_idxs(
                    ratio, bsz, seqlen, 0, seqlen)
            topk_idxs = _torch.cat([topk_idxs, comp_idxs], dim=-1)
        topk_idxs = topk_idxs.int()

        # Window write. HF does a two-way slice assignment when seqlen > win;
        # express it as one index_copy over the absolute ring slots each of the
        # last `win` positions lands in. For seqlen <= win this is just 0..seqlen.
        cache = sink.get(self, "kv_cache")
        keep = min(seqlen, win)
        src = kv[:, seqlen - keep:]
        slots = ((_torch.arange(seqlen - keep, seqlen, device=cache.device))
                 % win).long()
        cache = cache.index_copy(1, slots, src.to(cache.dtype))
        sink.put(self, "kv_cache", cache)

        # Attention itself reads the *uncompressed* kv for this prompt, with the
        # compressed entries concatenated on — not the ring buffer. The ring is
        # only what a later decode step will read.
        attn_kv = kv
        if ratio:
            kv_compress = compressor_prefill(self.compressor, x, sink)
            if kv_compress is not None:
                attn_kv = _torch.cat([kv, kv_compress], dim=1)

        o = hf_mod.sparse_attn(q, attn_kv, self.attn_sink, topk_idxs,
                               self.softmax_scale)
        hf_mod.apply_rotary_emb(o[..., -rd:], freqs_cis, True)

        if self.n_local_groups >= 1:
            o = o.view(bsz, seqlen, self.n_local_groups, -1)
            wo_a = self.wo_a.weight.view(self.n_local_groups, self.o_lora_rank, -1)
            o = _torch.einsum("bsgd,grd->bsgr", o, wo_a).flatten(2)
        else:
            o = _torch.nn.functional.linear(
                o.reshape(bsz, seqlen, -1), self.wo_a.weight)
        return self.wo_b(o)

    return compressor_prefill, indexer_prefill, attention_prefill


def apply_prefill_patches(hf_mod):
    """Install the functional prefill forwards. Returns the shared state dict.

    Same contract as `decode_patches.apply_decode_patches`: the sink is stashed
    per forward on a module-level dict rather than threaded through
    `Block.forward`, because that signature is fixed by HF's stack and rewriting
    it would fork far more of the reference model than necessary.
    """
    compressor_prefill, indexer_prefill, attention_prefill = (
        make_prefill_forwards(hf_mod))

    state = {"sink": None}

    def attention_forward(self, x, start_pos):
        return attention_prefill(self, x, state["sink"])

    hf_mod.Attention.forward = attention_forward
    hf_mod.Compressor.forward = lambda self, x, pos: compressor_prefill(
        self, x, state["sink"])
    hf_mod.Indexer.forward = lambda self, x, qr, pos, offset: indexer_prefill(
        self, x, qr, offset, state["sink"])
    return state
