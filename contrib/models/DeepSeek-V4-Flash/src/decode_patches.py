#!/usr/bin/env python
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Make DeepSeek-V4-Flash's decode step (`start_pos > 0`) traceable for Neuron.

Prefill only ever runs at `start_pos == 0`, so `compile_neuron.py` can bake the
position into the graph with `.item()`. Decode cannot: `start_pos` advances every
step, and one graph per position would mean `max_seq_len` NEFFs.

So every place the reference model uses `start_pos` as a *Python int* has to
become a tensor op. There are five, and each needed its own trick:

  1. `freqs_cis[p : p+1]`            -> index_select(freqs, 0, p)
  2. `kv_cache[:, p % win] = kv`     -> index_copy with a Long index
  3. `arange(0, (p+1) // ratio)`     -> fixed max length, -1 for the unused tail
  4. window index three-way branch   -> one closed form (see window_topk_idxs)
  5. `if (p+1) % ratio: return`      -> always write, but write back the old
                                        value when the condition is false

...plus two structural changes:

  * The KV cache must live on the device across calls, which means it has to be
    an `nn.Parameter` (not a buffer) and be returned as a graph output so NxD
    can alias it. See `promote_state_to_parameters` / `collect_state_aliases`.

  * State updates must be *functional* (`index_copy`, not `index_copy_`) so the
    new value can be returned as that output. HF instead mutates in place and
    hands the Compressor a *view* into Attention's cache
    (`self.compressor.kv_cache = self.kv_cache[:, win:]`), which a functional
    update cannot express — writing to a view produces a new tensor that the
    owner never sees. `StateSink` replaces the view with an explicit
    (owner, offset) redirect.

Every primitive here was validated before being used:
test/spike/test_decode_primitives.py checks each one lowers on trn2 and matches
CPU (and that the closed forms in 3 and 4 agree with HF's branches for
p = 1..19); test/spike/test_state_aliasing.py checks state really does persist
across calls once promoted to a Parameter.

Applied on top of compile_neuron.apply_xla_patches, not instead of it: the XLA
fixes (top-k, RoPE, MoE, embedding, o_proj) are position-independent and still
needed.
"""

import torch


# The state that has to survive between decode steps. `freqs_cis` is also a
# non-persistent buffer, but it is a *constant* — it must stay a buffer, or it
# becomes an aliased output and gets pointlessly copied every step.
_STATE_BUFFERS = ("kv_cache", "kv_state", "score_state")


# ---------------------------------------------------------------------------
# Index helpers: the closed forms that replace HF's branches on start_pos
# ---------------------------------------------------------------------------

def window_topk_idxs(pos, window_size, bsz):
    """Ring-buffer slot indices for the sliding window, `pos` a scalar tensor.

    HF branches three ways on start_pos. Its two decode branches are:

        if p >= win - 1:                     # buffer full
            pm = p % win
            cat([arange(pm+1, win), arange(0, pm+1)])
        elif p > 0:                          # buffer partly filled
            pad(arange(p+1), (0, win-p-1), value=-1)

    Slot j holds absolute position p - ((p - j) mod win), which is a real
    position iff it is >= 0, i.e. iff j <= p. So

        where(arange(win) <= p, arange(win), -1)

    covers both branches at once. For p >= win-1 it yields all of arange(win),
    which is a *permutation* of HF's rotated list rather than the same order —
    and that is fine, because sparse_attn gathers the listed slots and softmaxes
    over them, so only the set and the -1 mask affect the result. Checked
    against both HF branches for p = 1..19 in test_decode_primitives.py.
    """
    j = torch.arange(window_size, device=pos.device)
    return _batch(torch.where(j <= pos, j, torch.full_like(j, -1)), bsz)


def compress_topk_idxs(pos, ratio, max_comp, offset, bsz):
    """Compressed-KV indices, `pos` a scalar tensor.

    HF emits `arange(0, (p+1) // ratio) + offset`, whose *length* depends on p.
    A traced graph has a fixed output shape, so emit `max_comp` entries and set
    the not-yet-written tail to -1, which sparse_attn treats as masked.
    """
    n_valid = (pos + 1) // ratio
    j = torch.arange(max_comp, device=pos.device)
    return _batch(
        torch.where(j < n_valid, j + offset, torch.full_like(j, -1)), bsz)


def _batch(row, bsz):
    """A flat index list -> `(bsz, 1, n)`, which is what sparse_attn wants.

    HF writes `matrix.unsqueeze(0).expand(bsz, -1, -1)`, and in decode `matrix`
    is 1-D, so `expand` *prepends* the batch dim: the result is (bsz, 1, n), the
    middle axis being the single query position. sparse_attn unpacks exactly
    three dims (`_, _, topk = topk_idxs.shape`), so the query axis has to be
    there.

    Built by broadcast-adding zeros rather than with `expand`/`repeat`, because
    both lower to `as_strided`, which torch-xla does not implement ("View
    operators don't support since the tensor's storage cannot be shared across
    devices").
    """
    zeros = torch.zeros(bsz, 1, 1, dtype=row.dtype, device=row.device)
    return row.reshape(1, 1, -1) + zeros


def _slot(index, size):
    """`index % size` as the Long index that index_copy insists on.

    XLA rejects an Int index outright: "Copy index is expected to be of scalar
    type Long, but it is Int". start_pos arrives as int32 because that is what
    torch.jit.trace accepts as a scalar input, so cast at the boundary.
    """
    return (index % size).reshape(1).long()


# ---------------------------------------------------------------------------
# State plumbing
# ---------------------------------------------------------------------------

class StateSink:
    """Holds the current value of every state tensor during one forward pass.

    Exists because decode state updates have to be functional. HF mutates its
    caches in place, and crucially it aliases them by *view*:

        self.compressor.kv_cache = self.kv_cache[:, win:]

    so a compressor write lands in the enclosing Attention's cache. `index_copy`
    returns a new tensor instead of mutating, so a write through the view would
    be invisible to the owner. The sink replaces the view with an explicit
    redirect: the compressor's writes are rebased onto the owner's tensor at a
    fixed column offset.

    `register` declares who owns what; `get`/`put` read and write the live value.
    """

    def __init__(self):
        self._values = {}     # id(module) -> {name: tensor}
        self._redirect = {}   # (id(module), name) -> (owner_module, name, offset)

    def register_owner(self, module, name, tensor):
        self._values.setdefault(id(module), {})[name] = tensor

    def register_redirect(self, module, name, owner, owner_name, offset):
        self._redirect[(id(module), name)] = (owner, owner_name, offset)

    def get(self, module, name):
        key = (id(module), name)
        if key in self._redirect:
            owner, owner_name, offset = self._redirect[key]
            return self.get(owner, owner_name)[:, offset:]
        return self._values[id(module)][name]

    def put(self, module, name, value):
        key = (id(module), name)
        if key in self._redirect:
            # Rebasing a slice write onto the owner: the caller handed us the
            # slice's new value, so splice it back at the offset.
            owner, owner_name, offset = self._redirect[key]
            full = self.get(owner, owner_name)
            head = full[:, :offset]
            self.put(owner, owner_name, torch.cat([head, value], dim=1))
            return
        self._values[id(module)][name] = value

    def slot_offset(self, module, name):
        """Column offset of `module.name` inside whatever tensor really holds it."""
        key = (id(module), name)
        if key in self._redirect:
            _, _, offset = self._redirect[key]
            return offset
        return 0


def promote_state_to_parameters(model):
    """Turn every KV/compressor state buffer into an nn.Parameter.

    NxD resolves graph-state aliases by scanning `named_parameters()` and
    matching on `.data_ptr()` (torch_neuronx/xla_impl/hlo_conversion.py:
    "for name, parameter in func.named_parameters(): for inp_param in
    input_output_aliases: if inp_param.data_ptr() == parameter.data_ptr()").
    Buffers are never scanned, so a `register_buffer` state key matches nothing
    and the alias is silently dropped — the graph still compiles and runs, but
    resets its cache every call, which reads as a model that forgot its context
    rather than as a configuration error.

    Returns [(module, buffer_name, qualified_name, tensor)] in a deterministic
    order so the caller can map them to output indices.
    """
    promoted = []
    for mod_name, mod in model.named_modules():
        # Snapshot the keys: we mutate the buffer registry while iterating.
        for buf_name in list(mod._buffers.keys()):
            if buf_name not in _STATE_BUFFERS:
                continue
            tensor = mod._buffers[buf_name]
            if tensor is None:
                continue
            del mod._buffers[buf_name]
            param = torch.nn.Parameter(tensor, requires_grad=False)
            setattr(mod, buf_name, param)
            qualified = f"{mod_name}.{buf_name}" if mod_name else buf_name
            promoted.append((mod, buf_name, qualified, param))
    return promoted


def collect_state_aliases(model, n_real_outputs):
    """Build NxD's `{state_tensor: output_index}` alias map.

    The traced model returns (logits, *states); NxD writes output i back over
    the input it is aliased to, in place on the device, so the next call sees
    the updated cache with no host round trip.

    Keys must be CPU tensors: NxD pickles this dict back to the parent process,
    where `initial_states = tuple(aliases.keys())` rebuilds each one. An XLA
    tensor key makes that rebuild try to nrt_init a device the parent does not
    own (NRT_FAILURE status_code=1 -> BrokenProcessPool). The factory runs
    before the model moves to device, so these are still CPU tensors here.

    Returns (slots, aliases) where `slots` is the ordered [(module, name)] the
    trace wrapper reads back out of its StateSink — output n_real_outputs+i must
    carry slots[i], or the runtime writes the wrong cache back.
    """
    promoted = promote_state_to_parameters(model)
    slots = [(mod, name) for mod, name, _, _ in promoted]
    aliases = {t: n_real_outputs + i
               for i, (_, _, _, t) in enumerate(promoted)}
    return slots, aliases


def build_sink(model, hf_mod):
    """Seed a StateSink from the model's promoted state Parameters.

    Also reproduces the lazy wiring HF does on its first Attention.forward:

        if self.compress_ratio and self.compressor.kv_cache is None:
            self.compressor.kv_cache = self.kv_cache[:, win:]   # a view
            self.compressor.freqs_cis = self.freqs_cis
            if self.indexer is not None:
                self.indexer.freqs_cis = self.freqs_cis

    The cache half becomes a redirect: the compressor's slice is the tail of the
    enclosing layer's cache (window first, compressed entries after), while the
    Indexer owns a whole separate cache its own compressor writes from column 0.

    The freqs_cis half is just an attribute copy, but it has to happen *here*,
    per forward, rather than once at patch time. Only Attention registers
    freqs_cis as a buffer; Compressor and Indexer initialise theirs to None.
    Since `_apply` moves buffers and parameters but not plain attributes, wiring
    before the model reaches the device would pin the CPU copy and the graph
    would trace with a host constant. HF gets this right by accident, wiring
    inside forward; do the same.
    """
    sink = StateSink()
    for _, mod in model.named_modules():
        for name in _STATE_BUFFERS:
            tensor = getattr(mod, name, None)
            if isinstance(tensor, torch.nn.Parameter):
                sink.register_owner(mod, name, tensor)

    for _, mod in model.named_modules():
        if not isinstance(mod, hf_mod.Attention):
            continue
        if not getattr(mod, "compress_ratio", 0):
            continue
        # Attention's own compressor writes into the tail of Attention's cache.
        sink.register_redirect(mod.compressor, "kv_cache",
                               mod, "kv_cache", mod.window_size)
        mod.compressor.freqs_cis = mod.freqs_cis
        if getattr(mod, "indexer", None) is not None:
            # The Indexer's compressor writes into the Indexer's own cache,
            # from column 0 — no window in front of it.
            sink.register_redirect(mod.indexer.compressor, "kv_cache",
                                   mod.indexer, "kv_cache", 0)
            mod.indexer.freqs_cis = mod.freqs_cis
            # HF never sets this one: its Indexer.forward assigns
            # compressor.freqs_cis from self.freqs_cis, which is the same
            # tensor, so wire it directly.
            mod.indexer.compressor.freqs_cis = mod.freqs_cis
    return sink


# ---------------------------------------------------------------------------
# The decode forwards
# ---------------------------------------------------------------------------
#
# Each mirrors the reference decode branch (`start_pos > 0`) exactly, with the
# Python-int uses of start_pos replaced per the table at the top of this file.
# The prefill branches are deliberately not reimplemented: a decode graph is
# only ever entered with seqlen == 1 and pos >= 1, so keeping the two paths in
# separate graphs keeps each small and lets prefill stay exactly as validated.


def make_decode_forwards(hf_mod):
    """Return the three decode forwards, closed over the patched hf module."""
    import torch as _torch
    import xla_ops

    def compressor_decode(self, x, pos, sink):
        """Reference Compressor.forward, decode branch, position as a tensor.

        Structural changes vs HF:
          * `if not should_compress: return` becomes an unconditional write of
            `where(should, new, old)`. An early return would make the graph
            depend on the position.
          * the ring rotation at the end of the overlap branch
            (`kv_state[:, :ratio] = kv_state[:, ratio:]`) is folded into the
            same masked functional update.
        """
        bsz, _, _ = x.size()
        ratio, overlap = self.compress_ratio, self.overlap
        d, rd = self.head_dim, self.rope_head_dim
        dtype = x.dtype
        x = x.float()
        kv = self.wkv(x)
        score = self.wgate(x)

        # HF: score += self.ape[start_pos % ratio] — a runtime row of a small
        # constant table.
        score = score + _torch.index_select(self.ape, 0, _slot(pos, ratio))

        should = ((pos + 1) % ratio) == 0
        kv_state = sink.get(self, "kv_state")
        score_state = sink.get(self, "score_state")

        if overlap:
            # State layout is [overlap window | current window], each `ratio`
            # long, so the live write goes into the second half.
            slot = (ratio + pos % ratio).reshape(1).long()
            kv_state = kv_state.index_copy(1, slot, kv)
            score_state = score_state.index_copy(1, slot, score)
            # Compression computed unconditionally; `should` decides if it is
            # kept. First half contributes its low dims, second half its high
            # dims — the overlap trick from Compressor.overlap_transform.
            merged_kv = _torch.cat(
                [kv_state[:bsz, :ratio, :d], kv_state[:bsz, ratio:, d:]], dim=1)
            merged_score = _torch.cat(
                [score_state[:bsz, :ratio, :d], score_state[:bsz, ratio:, d:]],
                dim=1)
            compressed = (merged_kv * merged_score.softmax(dim=1)).sum(
                dim=1, keepdim=True)
            # Rotate current window -> overlap window, on compression steps only.
            head = _torch.arange(ratio, device=kv_state.device).long()
            kv_state = _torch.where(
                should, kv_state.index_copy(1, head, kv_state[:, ratio:]),
                kv_state)
            score_state = _torch.where(
                should, score_state.index_copy(1, head, score_state[:, ratio:]),
                score_state)
        else:
            slot = _slot(pos, ratio)
            kv_state = kv_state.index_copy(1, slot, kv)
            score_state = score_state.index_copy(1, slot, score)
            compressed = (kv_state[:bsz] * score_state[:bsz].softmax(dim=1)).sum(
                dim=1, keepdim=True)

        sink.put(self, "kv_state", kv_state)
        sink.put(self, "score_state", score_state)

        # RoPE for the compressed entry: HF uses freqs_cis[p + 1 - ratio].
        kvc = self.norm(compressed.to(dtype))
        freq_idx = (pos + 1 - ratio).clamp(min=0).reshape(1).long()
        hf_mod.apply_rotary_emb(
            kvc[..., -rd:], _torch.index_select(self.freqs_cis, 0, freq_idx))
        if self.rotate:
            kvc = hf_mod.rotate_activation(kvc)
            hf_mod.fp4_act_quant(kvc, hf_mod.fp4_block_size, True)
        else:
            hf_mod.act_quant(kvc[..., :-rd], 64, hf_mod.scale_fmt,
                             hf_mod.scale_dtype, True)

        # HF: kv_cache[:, p // ratio] = kv, but only on a compression step. Read
        # the target slot back and select, so a non-compression step rewrites
        # the identical value instead of branching.
        cache = sink.get(self, "kv_cache")
        cslot = (pos // ratio).clamp(max=cache.size(1) - 1).reshape(1).long()
        old = _torch.index_select(cache, 1, cslot)
        merged = _torch.where(should, kvc.to(cache.dtype), old)
        sink.put(self, "kv_cache", cache.index_copy(1, cslot, merged))
        return kvc

    def indexer_decode(self, x, qr, pos, offset, sink):
        """Reference Indexer.forward, decode branch, position as a tensor."""
        ratio, rd = self.compress_ratio, self.rope_head_dim
        bsz, _, _ = x.size()
        freqs_cis = _torch.index_select(self.freqs_cis, 0, pos.reshape(1).long())

        q = self.wq_b(qr).unflatten(-1, (self.n_local_heads, self.head_dim))
        hf_mod.apply_rotary_emb(q[..., -rd:], freqs_cis)
        q = hf_mod.rotate_activation(q)
        hf_mod.fp4_act_quant(q, hf_mod.fp4_block_size, True)

        compressor_decode(self.compressor, x, pos, sink)

        weights = self.weights_proj(x) * (
            self.softmax_scale * self.n_heads ** -0.5)

        # HF slices kv_cache[:, :end_pos // ratio] — a runtime length. Score the
        # whole cache and mask the unwritten tail to -inf instead: equivalent,
        # and shape-static.
        cache = sink.get(self, "kv_cache")
        index_score = _torch.einsum("bshd,btd->bsht", q, cache[:bsz])
        index_score = (index_score.relu() * weights.unsqueeze(-1)).sum(dim=2)
        if hf_mod.world_size > 1:
            import torch.distributed as _dist
            _dist.all_reduce(index_score)

        n_valid = (pos + 1) // ratio
        slots = _torch.arange(cache.size(1), device=index_score.device)
        index_score = index_score + _torch.where(
            (slots >= n_valid).view(1, 1, -1), float("-inf"), 0.0,
        ).to(index_score.dtype)

        topk_idxs = xla_ops.topk_indices_unordered(
            index_score, min(self.index_topk, cache.size(1)))
        # Mask picks that landed in the unwritten tail. The -inf bias makes them
        # last resort, but when fewer than k slots are valid some get selected
        # anyway, and they must not become real KV references.
        return _torch.where(topk_idxs >= n_valid, -1, topk_idxs + offset)

    def attention_decode(self, x, pos, sink):
        """Reference Attention.forward, decode branch, position as a tensor."""
        bsz, seqlen, _ = x.size()
        win, ratio = self.window_size, self.compress_ratio
        rd = self.rope_head_dim
        freqs_cis = _torch.index_select(self.freqs_cis, 0, pos.reshape(1).long())

        qr = q = self.q_norm(self.wq_a(x))
        q = self.wq_b(q).unflatten(-1, (self.n_local_heads, self.head_dim))
        q = q * _torch.rsqrt(q.square().mean(-1, keepdim=True) + self.eps)
        hf_mod.apply_rotary_emb(q[..., -rd:], freqs_cis)

        kv = self.kv_norm(self.wkv(x))
        hf_mod.apply_rotary_emb(kv[..., -rd:], freqs_cis)
        hf_mod.act_quant(kv[..., :-rd], 64, hf_mod.scale_fmt,
                         hf_mod.scale_dtype, True)

        topk_idxs = window_topk_idxs(pos, win, bsz)
        if ratio:
            # offset = win in decode: within this layer's single cache the
            # window comes first, compressed entries after it.
            n_comp = sink.get(self, "kv_cache").size(1) - win
            if self.indexer is not None:
                comp_idxs = indexer_decode(self.indexer, x, qr, pos, win, sink)
            else:
                comp_idxs = compress_topk_idxs(pos, ratio, n_comp, win, bsz)
            topk_idxs = _torch.cat([topk_idxs, comp_idxs], dim=-1)
        topk_idxs = topk_idxs.int()

        # Write this step's KV into the window ring...
        cache = sink.get(self, "kv_cache")
        sink.put(self, "kv_cache",
                 cache.index_copy(1, _slot(pos, win), kv.to(cache.dtype)))
        if ratio:
            # ...then let the compressor write into the tail of the same cache,
            # and re-read so attention sees both writes.
            compressor_decode(self.compressor, x, pos, sink)
        cache = sink.get(self, "kv_cache")

        o = hf_mod.sparse_attn(q, cache[:bsz], self.attn_sink, topk_idxs,
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

    return compressor_decode, indexer_decode, attention_decode


def apply_decode_patches(hf_mod):
    """Install the decode forwards so `Transformer.forward` runs in decode mode.

    Only `Attention.forward` is replaced. Block/Transformer keep HF's shape,
    which means the sink has to reach Attention without an extra argument: the
    signature `attn(x, start_pos)` is fixed by Block.forward, and rewriting the
    whole stack to thread a sink through would fork far more of the reference
    model than necessary. So the sink is stashed on the module for the duration
    of one forward, which is safe here because a traced graph is built by a
    single-threaded pass.

    Call after compile_neuron.apply_xla_patches(hf_mod): this deliberately
    overwrites that module's Attention.forward, and relies on its other patches
    (RoPE, MoE, embedding, top-k) staying in place.
    """
    compressor_decode, indexer_decode, attention_decode = make_decode_forwards(
        hf_mod)

    state = {"sink": None, "pos": None}

    def attention_forward(self, x, start_pos):
        # start_pos here is HF's positional argument, ignored: the real position
        # is the tensor set by the wrapper. Keeping the parameter means
        # Block.forward needs no change.
        return attention_decode(self, x, state["pos"], state["sink"])

    hf_mod.Attention.forward = attention_forward
    hf_mod.Compressor.forward = lambda self, x, pos: compressor_decode(
        self, x, state["pos"], state["sink"])
    hf_mod.Indexer.forward = lambda self, x, qr, pos, offset: indexer_decode(
        self, x, qr, state["pos"], offset, state["sink"])
    return state

