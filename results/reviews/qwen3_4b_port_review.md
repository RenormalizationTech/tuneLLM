# Qwen3-4B GCAD port review (`scripts/qwen3_gcad_port.py`)

Reviewer: Claude advisor, read-only. I made no model, GPU or judge calls, ran
no tests and changed no files.

**How I compared:**

- I extracted the three author definitions by syntax tree from the pinned tree
  (`core/generate_vec.py`, `scripts/generate_vec_prompt_attn_k.py`,
  `core/gated_steerer.py` at `6c879f4…`) and diffed them against the port.
- I read the installed Transformers 4.52.3 `modeling_qwen3.py`: the
  attention forward, `Qwen3RMSNorm`, the eager attention, the rotary
  embedding and the model forward.

## Verdict

**The arithmetic is correct and the diff is minimal.** I found no defect in
the geometry, the norm, RoPE, the system decomposition, response pooling or
the gate. There are **two required robustness fixes**, both against silent
failure (D1, D2), and one runner-side requirement (D3). Exactness itself
stays **provisional until the CPU tests pass**.

## What matches (verified by reading)

**The diff against the author code is exactly the three accepted changes:**

- `head_dim` comes from `getattr(config, "head_dim", hidden // heads)`, in all
  three definitions;
- `q_norm` and `k_norm` are applied to the `(…, heads, 128)` view before
  RoPE: in the key extractor (Q and K) and in the steerer's Q capture;
- `repeat_kv` and `apply_rotary_pos_emb` are imported from `qwen3`.

No other line differs. The gate, the update, the per-head mean, `/√head_dim`,
`inverse`, the dtype casts and the hook sites are the author's.

**Geometry and norms.**

- Qwen3 computes `q_norm(q_proj(h).view(*, -1, head_dim))`, then transposes,
  then applies RoPE. The port does the same: view to `(B, T, nh, hd)`, norm
  over the last dimension, then transpose and RoPE. So the RMSNorm is applied
  over the same axis to the same BF16 tensor.
- The value path has no norm in Qwen3, and the delta extractor correctly
  leaves V unnormed.

**RoPE.**

- **In the steerer**, it uses the model's own `position_embeddings` kwarg.
  `Qwen3DecoderLayer` passes it as a keyword, so the `with_kwargs` pre-hook
  receives it.
- **In the key extractor**, `model.model.rotary_emb(k, arange(T))` matches
  the model for batch 1 without padding. `rope_scaling` is null, so
  `attention_scaling` is 1.

**System decomposition.**

- Eager Qwen3 returns post-softmax weights in the query dtype, cast from an
  FP32 softmax. The port takes the weights over system keys, multiplies by
  the GQA-repeated system V and applies `o_proj`. `o_proj` has no bias
  (`attention_bias=false`), so the split into system and non-system
  contributions is exactly linear.
- Under SDPA with `output_attentions=True`, Transformers falls back to eager,
  but the plan's registered eager load is still the right choice.

**Response pooling** is the author's: the mean over `prompt_len:` per sample.
The difference between the positive and negative means is formed by the
caller.

**K_pos and D̄** are the author's token-weighted system-K mean and mean-Q·K
demand. Because the demand is linear in Q, this equals the mean per-token
demand.

**The gate.** The gate formula is the author's: c = 2b·σ(±s(d − D̄)), and
`attn_out + c·Δ` at the `self_attn` output, before the residual add, in
dimension 2560.

**Prefill, decode and cleanup.**

- In decode, q_proj sees T = 1, and `position_embeddings` holds the current
  position, so the demand is correct with the cache.
- The per-layer state is popped after use and cleared on `__exit__`. Handles
  are removed on exit, including when an exception occurs inside `with`.

**The Qwen2 path** is unchanged by construction: there is no
`q_norm`/`k_norm` on Qwen2, so `_norm` is the identity, and `head_dim` falls
back to `hidden // heads`.

## Required fixes (no arithmetic change)

**D1. The norm can be skipped silently.** `_norm` returns its input whenever
the attribute is missing. The `q_norm`/`k_norm` bug was exactly the silent
mismatch this port exists to prevent, so the port should fail closed on
architecture.

- **Qwen3** (`config.model_type == "qwen3"`): require `q_norm` and `k_norm` to
  be present, with weight shape `(head_dim,)`.
- **Qwen2:** require both to be absent.
- **Any other model type:** raise.
- **Also assert, once, at construction and extraction entry:**
  - `q_proj.out_features == nh·hd`;
  - `k_proj` and `v_proj` `out_features == nkv·hd`;
  - `o_proj.in_features == nh·hd`;
  - `o_proj.out_features == hidden`;
  - `o_proj.bias is None` (this makes the decomposition exact);
  - `rope_scaling is None`;
  - no sliding window active (`use_sliding_window` false).

**D2. The steerer can pass through silently.** `_make_attn_hook` returns the
output **unsteered** when the captured Q or the position embeddings are
missing. This is the author's behaviour. On 4B, any kwarg or ordering drift
would then produce baseline-like arms and a spurious null. The mechanics
"nonzero change" check would miss a partial skip, such as one in decode only.

- **Add counters** per layer and phase (prefill or decode): steering applied
  versus skipped. This is instrumentation, not arithmetic.
- **The runner fails closed** if any hooked call is skipped in mechanics or
  evaluation, for every arm with an active steerer, including b = 0 and the
  random arms.
- **Record the counts** in the shift audit.

**D3 (runner). Range and boundary assertions.** The port doesn't check these,
so the runner must:

- assert that `layer_indices == [12…25]` and that the bank holds all 36
  layers;
- never nest one steerer instance;
- pass prompts rendered with `enable_thinking=False`, where the first
  `<|im_end|>\n` is the system end;
- record, per item, whether `tokens(prompt)` is an exact prefix of
  `tokens(prompt + response)`.
  - Pooling uses the author's `prompt_len` from `encode(prompt)`. The Qwen3
    prompt ends in `</think>\n\n`, so an answer that begins with whitespace
    could merge across the boundary. The 7B prompts ended in
    `assistant\n`, which carries the same risk.
  - Count mismatches, as the plan requires. If the count is nonzero, report
    it at R2 before evaluation.

## Provisional, pending the CPU tests

These are not defects. The plan's tests decide them:

1. **Qwen2 regression:** bit-identical delta, K_pos, D̄ and steered output
   against the AST-loaded author definitions.
2. **Qwen3 exactness:** softmax(q̃k̃ᵀ·scaling + mask), built from the port's
   normed post-RoPE Q and K, equals the eager attention weights.
   - Include a **decode step with a cache** (T = 1, offset position).
   - Include a **left-padded batch**, whose position IDs come from the
     attention mask, matching the impolite sub-batches.
3. **Decomposition:** the system contribution plus the non-system
   contribution equals the actual `o_proj` output, on tiny Qwen3.
4. **Steerer equivalence:** with b > 0 on tiny Qwen3, the hooked output equals
   the manual a + c·Δ, using the model's own internal `query_states`.
5. **Kwarg pickup:** in prefill and cached decode, with the D2 counters
   showing zero skips.
6. **Zero parity:** b = 0 gives token and logit identity.
7. **D1 negatives:** removing `q_norm`, or mismatching `head_dim`, raises.

## Notes (no action)

- The duplicate `repeat_kv` import inside the delta function mirrors the
  author code. It is harmless.
- The demand and gate are computed in BF16, as on 7B. A random control or a
  zero coefficient doesn't change this.
- The license header records the derivation and the changes. The release must
  also carry the author's NOTICE file.

**Status:** accepted subject to D1 and D2, plus D3 in the runner, and to the
seven CPU tests passing. No scientific choice changes.
