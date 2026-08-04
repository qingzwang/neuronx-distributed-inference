# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Install both modes' forwards behind one call-time dispatch.

Why a dispatcher rather than two installs
-----------------------------------------
HF keeps Attention/Compressor/Indexer forwards on the *class*, and NxD's
`ModelBuilder` calls `load_module()` for every registered graph before tracing any
of them. So installing prefill's forwards for one graph and decode's for another
does not work: whichever loads second wins the patch site for both, and the first
graph then traces with the other mode's code. That failure is not obvious — it
surfaced as decode reading a `pos` that prefill never set (None -> a zero-width
index tensor -> "size of tensor a (128) must match tensor b (0)").

So both sets are installed once and routed on `state["active"]`, which the model
wrapper sets before each forward.

This lives in its own module rather than in a compile driver because two drivers
now need it (`compile_joint` and the NxDI-native path), and duplicating the
install is exactly how the two would drift.
"""


def install(hf_mod):
    """Install the dispatcher on `hf_mod`. Returns the shared state dict.

    The dict is `{"active": "prefill"|"decode", "sink": StateSink, "pos": tensor}`
    and is also stashed as `hf_mod._dsv4_patch_state` so a model wrapper can reach
    it without threading it through constructors.

    Re-installs unconditionally: `apply_xla_patches` runs on every model build and
    puts HF's single-mode `Attention.forward` back over this dispatcher, so
    returning early on a second call would leave that one in place. The state dict
    is reused across calls when present, because the wrappers hold a reference to
    it and a fresh dict would orphan them.
    """
    import decode_patches
    import prefill_patches

    state = getattr(hf_mod, "_dsv4_patch_state", None)
    if state is None:
        state = {"active": "prefill", "sink": None, "pos": None}

    p_compressor, p_indexer, p_attention = (
        prefill_patches.make_prefill_forwards(hf_mod))
    d_compressor, d_indexer, d_attention = (
        decode_patches.make_decode_forwards(hf_mod))

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

    hf_mod.Attention.forward = attention_forward
    hf_mod.Compressor.forward = compressor_forward
    hf_mod.Indexer.forward = indexer_forward
    hf_mod._dsv4_patch_state = state
    return state
