# Qwen3.8-27B sarcasm GCAD: design and port review 01

Advisor review, 2026-11-18. Read-only review of `docs/qwen38_sarcasm_plan.md`
and the worker's uncommitted `scripts/qwen38_gcad_port.py`, checked against the
pinned `transformers==5.23.0` `modeling_qwen3_5.py` in
`outputs/qwen38-science-01/env`. No model, judge or GPU call was made. The only
command run was a local sha256 of the installed Claude binaries.

## Verdict

The port's arithmetic is correct where it matters: Q/gate de-interleave, norm
then partial RoPE, GQA head mapping, gated crop before the bias-free `o_proj`,
token-weighted K_pos, and demand centring. **No blocking arithmetic defect.**
Before R0 the plan needs: the corrected token budget (F1), an explicit
precision statement (F2), a new judge runtime pin (F3), synthetic-bank
pre-extraction mechanics (F4), and real-model guards and tests (F5–F8).

## Port arithmetic checked line by line

| Item | Port | Native / author | Status |
| --- | --- | --- | --- |
| Q/gate split | `q_proj` out reshaped `(…,H,2D)`, `chunk(2,-1)[1]`, flatten, sigmoid (L66–67) | `view(*in,-1,2D)`, chunk, `reshape(*in,-1)`, `sigmoid` in model dtype | exact |
| Q, K | captured `q_norm`/`k_norm` outputs `(B,T,H,D)` → transpose → `apply_rotary_pos_emb` with captured `position_embeddings` (L64) | same functions, same dtype, same order | exact |
| GQA mapping | `repeat_kv` in extraction; `repeat_interleave(groups,0)` on K_pos in bank/steer | `repeat_kv` gives head h → kv h // groups; `repeat_interleave` matches | exact |
| Gated crop | `(w[..., :S] @ v[..., :S, :])` → `(1,T,H·D)` × sigmoid gate → `o_proj` (L108–110) | `o_proj(attn · sigmoid(gate))`, no bias | correct; linear in the crop, so sys + nonsys = full |
| Δ | per-sample response-token mean, mean over pairs of (pos − neg) (L111, L125) | E₊ mean − E₋ mean; equal to mean of differences for matched pairs | correct |
| K_pos | Σ system keys / Σ system-token counts over positive samples (L112–113, L126) | system-token-count weighted | correct |
| D̄ | mean_h (q̄_h · k̄_g(h))/√D with q̄ = token-weighted positive response mean (L127–129) | mean over positive response tokens of d_t; by linearity in q for fixed k̄ this equals d(q̄) | correct up to rounding |
| Steer | demand from current captured q; `2b·σ(1.5(d−D̄))·Δ` added to `self_attn` output[0] (L150–158) | registered port choice | correct |
| b = 0 | `0·σ(…)·Δ = 0`, `x + 0 = x` | — | exact parity expected (finite σ) |

## Findings

**F1. Token budget arithmetic (plan error, fix before R0).** The maximum
generated tokens are 200·256 + 128·160 + 128·160 + 6·32 = **92,352**, not ≈81k.
With the F4 synthetic phase (6·32 more) the total is **92,544**. At an assumed
8–15 tok/s this is 1.7–3.2 GPU-h. The 2–3.5 GPU-h estimate and the 8 GPU-h hard
stop stand.

**F2. Pooling precision (decide now, disclose as a port difference).** The port
casts the projected crop to FP32 before the response-token mean (L111), and
sums K and Q in FP32 (L112–114). The 4B port and the author code take each
side's response mean in BF16 and cast afterwards. **Recommendation: keep the
worker's FP32 accumulation and register it** as "o_proj output in model dtype;
response, pair, key and query means accumulated in FP32". This is a declared
port difference, not a claim of equivalence with 7B/4B banks. The plan is
amended accordingly. Attention scores remain in model dtype, with FP32 softmax
(L97, L107), as in native eager.

**F3. Judge runtime pin changed (not a study null).** The current `claude` is
2.1.327, sha256 `343c744b…6ac9`. The 27B pin in `scripts/qwen38_judge_runtime.py`
is 2.1.326, `b6b3647d…cb37`, audited in `outputs/qwen38-judge-runtime-01`.
`identity()` would therefore refuse, correctly. The pinned 2.1.326 binary still
exists in `~/.local/share/claude/versions/` with a matching hash, but the
updater may prune it.
**Recommendation:** register a new, study-specific isolated runtime audit for
2.1.327, with its own audit directory and binding constants and one audit call.
Preserve the old record and wrapper unchanged. Fallback only if PM prefers:
copy 2.1.326 into the run directory and pin that absolute path; this still
needs a fresh audit because the path changes. Judge calls start only after the
new audit passes.

**F4. Synthetic-bank mechanics before elicitation (adopt).** This phase tests
both extraction and steering on the real NF4 model before any of the 200
elicitation generations. It runs in one process on one GPU, after R0.

- *Synthetic bank:* 2 teacher-forced extraction forwards on one fixed pair.
  The pair uses extraction instruction pair 0, positive and negative system
  prompts, and extraction question 0, with one fixed response text frozen in
  the config before the run (the same response bytes on both sides). The
  resulting Δ is a mechanical artifact, not a sarcasm direction. Pooling with
  one pair is allowed for this phase only.
- *Probes (cap 32, the two frozen mechanics prompts):* 2 ordinary, 2 b=0,
  2 b=1, giving **6 generations**. Logit forwards: ordinary, b=0, b=1, then
  ordinary after cleanup, giving **4**.
- *Exact criteria. Any miss is `port_or_mechanics_failure` and stops the run;
  no elicitation follows.*
  1. Ordinary vs b=0: identical generated token IDs, and `torch.equal` on
     last-position logits.
  2. Ordinary vs post-cleanup ordinary: `torch.equal` logits. The count of
     registered forward and pre-forward hooks on every module equals the count
     before entry.
  3. b=1: max |Δlogit| > 0. Final-layer hidden state differs at some position.
     All bank tensors and diagnostics are finite. Per-site ‖Δ‖ > 0.
  4. Coverage per generate call, for each of the 6 sites: prefill = 1 and
     decode = (generated length − 1). There are zero hooks on every
     `linear_attn` module and on the 10 out-of-window `self_attn` modules,
     checked by inspecting hook dictionaries during steering.
  5. Decomposition, at each site on both TF forwards:
     ‖o_proj((w·v)⊙σ(g)) − native self_attn output‖ / ‖native‖ ≤ **0.02**.
     The same reconstruction with the gate omitted must have relative error
     ≥ **0.2** and ≥ 10× the gated error. The tolerance is non-zero because
     the native path is SDPA in BF16 with NF4 projections. ‖o_proj(sys) +
     o_proj(nonsys) − o_proj(full)‖ / ‖o_proj(full)‖ ≤ **0.01**.
- *Report but do not gate:* cache/no-cache agreement, tok/s at cap 32, peak
  allocated memory, and Δ norms relative to mean attention-output norms.
- *Real-bank mechanics (after extraction, unchanged from the plan):* 6 probes
  and 4 logit forwards with criteria 1–4 on the real bank. A fresh process
  reruns its own ordinary references. Exact parity is never compared across
  processes or GPUs.
- *Budget change:* +6 generations, +4 logit forwards, +2 teacher-forced
  forwards. That gives 468 generations, 8 logit forwards and ≤ 302 TF
  forwards (cap 300 for the real bank plus 2 synthetic).

**F5. Boundaries must be stated, not inherited implicitly.** Follow the 4B
`boundary()` convention. The system span runs from position 0 through the
first `<|im_end|>\n`, inclusive. The prompt is the full
`apply_chat_template(…, add_generation_prompt=True, enable_thinking=False)`
string, including any empty `<think>\n\n</think>\n\n` block the Qwen3.8
template emits. The response mask is every position at or after the prompt
length in prompt + output, with no appended end token. Assert that the
rendered prompt starts with `<|im_start|>system\n`, that the system and prompt
token prefixes are equal under joint tokenization, and that there is a
nonempty response. Record the per-item check counts.

**F6. Real-load geometry check.** `validate_geometry` requires
`config.model_type == 'qwen3_5_text'` and reads `in_features`/`out_features`.
Confirm both on the actual `Qwen3_5ForCausalLM` NF4 load. On that load the
config may be the composite `qwen3_5` type, and the projections are
bitsandbytes `Linear4bit`. Do this in the synthetic phase; there is no
separate GPU call. If the type check fails, the fix is a config-path change,
not a relaxation of the geometry assertions.

**F7. Audit counter robustness (minor).** The prefill/decode classification
uses `T > 1` (L156), so a one-token prompt would be misclassified. This is
harmless for the fixed prompts. Also count calls where the captured state is
missing, which is currently a `KeyError`, as skipped calls. Raise rather than
skip silently.

**F8. CPU tests still required (plan §4).** None are present in the working
tree yet. Required: de-interleave bitwise; port weights vs eager within FP32
tolerance; gated decomposition exact in FP32, with a gate-omitted variant that
fails; position-embedding pickup in prefill and decode; b=0 identity; hooks
only on window `self_attn`; cleanup; Qwen2/Qwen3 port regression unchanged.

## Plan amendments made

`docs/qwen38_sarcasm_plan.md` now records F1 (92,352 / 92,544 tokens), F2
(FP32 accumulation), F3 (new judge audit), F4 (synthetic phase and criteria)
and F5 (boundaries). Scientific thresholds, gates, strengths {1, 2},
fixtures and arms are unchanged.
