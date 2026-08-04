# Cross-check against vLLM's DeepSeek-V4 implementation

Source: vllm-project.github.io `_posts/2026-04-24-deepseek-v4.md`. Read to check
this port's understanding of the architecture against an independent
implementation, after the NxDI-native path stalled on all-zero logits.

## Where this port agrees with vLLM

Every structural claim checks out against both the HF reference and this port's
layout, which is reassuring — the model structure is not where the bug is.

| | vLLM | this port |
|---|---|---|
| compressed group | "weighted sum of **8** uncompressed tokens, **stride 4**" (c4a) | `overlap_transform` builds `2*ratio = 8` from stride-`ratio` groups (model.py:307-314) |
| compressor residual | "8 tokens for c4a, 128 for c128a" | ring length `coff*ratio` = 8 and 128 (`LayerKind.state_specs`) |
| sliding window | "128-token uncompressed local window" | `window_size = 128` from config |
| sparse indexer | "top-k compressed, k=512 for c4a" | `index_topk = 512` from config |
| shared K/V + inverse RoPE | "we apply an inverse RoPE to the attention output" | `apply_rotary_emb(o[..., -rd:], freqs_cis, True)` in both forwards |
| three cache kinds | main KV, indexer KV, compressor state | exactly the three `LayerKind` publishes |

The decode ring roll is also present and correct here
(`decode_patches.compressor_decode` rotates `kv_state[:, :ratio] <-
kv_state[:, ratio:]` on compression steps only), matching the rolling-residual
semantics vLLM describes.

## The design idea worth borrowing

vLLM does **not** treat the compressor ring as private model state. It registers it
*as a KV cache*:

> Rather than a separate side buffer, vLLM registers this state "under the
> sliding-window KV cache spec, with `sliding_window = coff * compress_ratio`."

The payoff they cite is that prefix caching and disaggregated prefill then work on
the ring for free — the state at a block boundary "is already the correct
resumption point", with no residual-specific transfer path.

That is the same insight this port needs for the prefill -> decode hand-off, and it
argues the current approach is structurally right: publish the ring through the
same cache mechanism as the KV, which is what `DSV4CacheManager` does (one flat
`past_key_values` holding all three kinds).

## Two things vLLM does that this port does not, and their relevance

1. **Uniform block unit.** Every layer allocates in "256 native token positions"
   regardless of compression ratio, so a c4a block holds 64 compressed entries and
   a c128a block holds 2. This exists to stop the scheduler from branching per
   layer. This port has no paged allocator and one contiguous cache per layer, so
   there is nothing to unify — not applicable, but worth recording as the reason
   our ragged per-layer shapes are acceptable here and would not be in vLLM.

2. **Different dtypes per phase.** "Prefill uses bfloat16 KV cache; decode uses
   partially token-wise fp8." This port uses bf16 for both. Not a correctness
   issue, and adopting fp8 decode would be a performance follow-up, not a fix.

Neither explains the all-zero logits.

## What this rules out

The all-zero failure is not a misunderstanding of the architecture. Ring size,
stride, window, indexer k, the inverse RoPE, the three cache kinds and the decode
ring roll all match an independent implementation. Combined with the five CPU gates
being bit-identical to the validated patched-HF path, the model math is not in
question — the failure is confined to the NxDI loading boundary, where
`initialize()` reports success and the NEFF still does not run.
