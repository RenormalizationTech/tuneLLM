# Qwen3.8-27B sarcasm GCAD: prospective design

Status: **advisor design proposal, 2026-11-18. Not registered.** No model or
judge call may start until PM approves and registers this design (R0), then
mechanics (R1) and selection (R2). Historical studies stay byte-identical.
Advisor owns this file and later review files; the worker owns code, config and
tests. A null at any stage is a valid completion.

## 1. Question and scope

Does a fresh GCAD sarcasm bank, extracted on Qwen3.8-27B and applied only at
full-attention blocks, raise sarcasm on ordinary replies by the same fixed
criteria as the retained 7B controller, without exceeding the coherence
allowance? One bank, one window, a small strength comparison on development
questions, one locked final evaluation. No sweep, layer search, SAPS arm,
re-extraction or rescue.

Evidence carried in (not reused as tensors or answers):

- 7B low-strength study: b=2 at 11 blocks retained; held-out SARC 63.78,
  coherence 89.59→70.72, margin 1.125 on the relative gate.
- 4B fixed transfer: b=2 at 14 blocks induced SARC 56.13 but failed only the
  relative coherence gate by 3.09. Strength/quality is the binding constraint.

## 2. Model, representation and runtime

- `Qwen/Qwen3.8-27B` revision `37561816e66b5b9153b1b64c10968c18c1e6f4a0`,
  local snapshot, text-only `Qwen3_5ForCausalLM`, thinking off.
- **Representation: the saved project NF4 checkpoint**
  `artifacts/qwen38-27b-nf4-01` (nf4, double quant, BF16 compute; the settings
  of `configs/qwen38_27b_nf4.json`), bound by
  `results/qwen38_27b_nf4_artifact_01.json` hashes. One A6000 per
  process, ~18 GB. Bank and steering are defined in the NF4 model's activation
  coordinates; this is an NF4-model controller, not a BF16 claim. BF16 would
  need both GPUs and is not run.
- Existing isolated 27B environment (`outputs/qwen38-science-01/env`: Python
  3.13, torch 2.12.0+cu130, Transformers 5.23.0, bitsandbytes 0.49.2). Hash
  weights, tokenizer, `config.json`, `generation_config.json`,
  `chat_template.jinja` and the installed `modeling_qwen3_5.py` at R0.
- SDPA for generation. Known limitation: the NF4 smoke failed strict
  cached-vs-full-prefix logit equality (max 0.125). Therefore every compared
  arm uses the identical cached `generate` path, batch 1; cache/no-cache
  agreement is reported, never gated.

## 3. Architecture facts (verified in the pinned modeling code)

| Property | Qwen3.8-27B text config | Consequence |
| --- | --- | --- |
| Layers | 64; `layer_types` = 3 × `linear_attention` then 1 × `full_attention`, repeating | Only indices 3, 7, …, 63 (16) have softmax attention |
| Linear blocks | Gated DeltaNet, recurrent state, no system-token attention weights | **Never hooked, never modified**; GCAD cropping is undefined there |
| Full attention | 24 Q heads, 4 KV heads (group 6), `head_dim` 256, hidden 5120 | K_pos `[sites,4,256]`, Δ `[sites,5120]` |
| `q_proj` | outputs `heads × 2·head_dim`, viewed `(…, heads, 512)` then chunked → per-head interleaved Q and gate | Port must de-interleave exactly as native (`view(...,-1,2*hd)`, `chunk(2,-1)`), not split the flat vector in halves |
| Norms / RoPE | `q_norm`/`k_norm` (zero-centred RMSNorm, `1+w`) before partial RoPE on the first 64 of 256 dims; text-only M-RoPE with identical t/h/w positions | Use the `position_embeddings` the layer receives; never recompute RoPE |
| Output | `o_proj(attn · sigmoid(gate))`, no bias | Gate is elementwise and post-softmax, so cropping stays linear (below) |

**Cropped system contribution (extraction):**

a_sys(l,t) = W_O,l [ sigmoid(g_l,t) ⊙ concat_h( Σ_{j∈S} α^{l,h}_{tj} v^{l,g(h)}_j ) ]

with α the full-prefix softmax weights (not renormalized over S), S the system
tokens, g the native gate for query position t. Because W_O has no bias and the
gate multiplies the head output elementwise, a_sys + a_nonsys equals the native
`o_proj` output exactly (up to dtype); this is a CPU test. Δ_l = E₊ mean_resp
a_sys − E₋ mean_resp a_sys; K_pos = system-token-count-weighted mean of
post-norm, post-RoPE positive-system keys; D̄ = positive-response-token-weighted
mean demand; demand d = mean_h q̃·k̄₊/√256 with q̃ post-norm post-RoPE.

**Steering (explicit port choice):** a′ = a + 2b·σ(1.5(d − D̄))·Δ, added to the
`self_attn` module output (after native gate and `o_proj`), before the residual
add. Gate scale 1.5, `inverse=False`, raw unnormalized Δ, model-dtype
arithmetic. This matches the author insertion point (post-`o_proj` attention
output); placing Δ inside the sigmoid-gated pre-`o_proj` space would be a
different method and is not run.

**Window (prospective depth formula, as 4B):** start = floor(9·64/28 + 0.5) =
21, end = floor(20·64/28 + 0.5) = 46 (exclusive). Full-attention sites in
[21, 46): **23, 27, 31, 35, 39, 43 (6 sites)**. 7B used 11 blocks; fewer sites
means b is again a fixed setting, not a matched strength. No other window.

## 4. Port and CPU tests (worker, before R0 registration)

New file (e.g. `scripts/qwen38_gcad_port.py`) with the author NOTICE; the
existing Qwen2/Qwen3 port and author files are untouched. Tiny in-memory
Qwen3.5 config with mixed layer types, no download:

1. Q/gate de-interleave equals native `q_proj` chunking bit-for-bit.
2. Port softmax(q̃k̃ᵀ/16 + mask) equals eager attention weights.
3. a_sys + a_nonsys == native `o_proj` output (gate included); a variant
   omitting the gate must fail the test (guards the silent error).
4. Hook receives `position_embeddings` in prefill and cached decode.
5. b = 0 → logits identical to unhooked; hook registered only on window
   full-attention `self_attn`; zero calls on linear blocks and on
   out-of-window full-attention blocks.
6. Cleanup: after removal no hooks remain on any module and an ordinary
   forward reproduces pre-hook logits exactly.
7. Existing Qwen2/Qwen3 port tests still pass unchanged.

## 5. Inputs and rendering (reused bytes, hash-verified)

| Role | File | Notes |
| --- | --- | --- |
| Extraction | `evaluation/sarcasm_method_transfer/extract.json` | 5 instruction pairs × 20 questions = 100 pairs; positive "helpful sarcastic", negative "helpful sincere" |
| Development (strength comparison) | `evaluation/author_profanity_revision/calibration.json` (32) | used for 7B selection; never a final test |
| Final (locked) | `evaluation/author_profanity_revision/report.json` (32) | previously the final set for 7B and 4B |
| Rubrics | existing sarcasm, GCAD coherence, impoliteness rubrics; 11-word lexicon | unchanged bytes |

- Render with the native chat template, `add_generation_prompt=True,
  enable_thinking=False`; prompt token IDs are frozen at R0 (`rendered.json`)
  and every generation asserts equality with them.
- **Raw-trajectory boundaries (registered port difference; no retokenization
  of generated text).** Teacher forcing uses the exact saved prompt
  `input_ids` followed by the exact saved generated `output_ids`, with one
  terminal EOS token (configured EOS list) removed if present; no EOS is
  appended. System span = prompt positions 0 through the first `<|im_end|>\n`
  inclusive, checked as a token prefix of the saved prompt IDs; the prompt
  includes any empty `<think>\n\n</think>\n\n` block. Response mask = positions
  ≥ prompt length, **excluding any special-token positions**; the mask must be
  nonempty. The decode of the same raw output IDs (`skip_special_tokens=True`)
  must equal the text the judge graded. The 4B port instead re-tokenized
  decoded text; this is a declared difference, not historical equivalence.
  The synthetic-phase response IDs are predetermined and frozen at R0.
  Record per-item check counts.
- Evaluation baseline/GCAD/random: native rendering, **no system message**
  (as 4B; 7B had a default system). Prompted arm: positive instruction pair
  index 0 as the system message.
- R0 recomputes NFKC/casefold/whitespace exact overlap of extraction vs
  development vs final; any nonzero stops for PM review. Known topical
  overlap x18 ↔ final H14: descriptive H14-excluded figure only.

## 6. Phases and fixed decoding

**6.1 Elicitation.** 100 pairs × 2 sides = 200 replies; temperature 1, top-p
0.8, top-k 20, repetition penalty 1.05, cap 256, per-item seeds from plan seed
9840. Judge each reply for sarcasm and coherence in separate blinded journals.
**Eligible pair (frozen 7B rule):** positive SARC ≥ 50, negative SARC < 50,
coherence ≥ 50 on both sides. **Fewer than 10 eligible pairs → `extraction_null`,
close.** Bank: ≤ 3 teacher-forced forwards per eligible pair-side role, cap 300;
save Δ `[6,5120]`, K_pos `[6,4,256]`, D̄ `[6]` FP32 safetensors with hashes and
per-site Δ norms. **Precision (registered port difference):** attention scores
in model dtype with FP32 softmax; gated crop and `o_proj` in model dtype;
response, pair, key and query means accumulated in FP32. The 4B/author code
took side means in BF16; this is a declared difference, not historical
equivalence.

**6.0 Synthetic-bank mechanics (after R0, before any elicitation).** One
process, one GPU: 2 teacher-forced extraction forwards on one frozen pair
(instruction pair 0 pos/neg systems, extraction question 0, one fixed response
text frozen in config) build a mechanical bank; then 6 cap-32 probes (2
ordinary, 2 b=0, 2 b=1) and 4 logit forwards (ordinary, b0, b1, post-cleanup).
Exact criteria and decomposition tolerances (gated reconstruction relative
error ≤ 0.02; gate-omitted ≥ 0.2 and ≥ 10× gated; sys+nonsys linearity ≤ 0.01)
are in `results/reviews/qwen38_sarcasm_design_review_01.md` F4. Also confirms
real-load geometry. Any miss → `port_or_mechanics_failure`; no elicitation.

**6.2 Real-bank mechanics (R1).** Cap-32 probes on the two mechanics prompts
(development questions index 0 and 1, identical in 6.0; mechanics outputs are
never graded or used for selection):
2 ordinary, 2 b=0, 2 b=1. Logit forwards: ordinary, b=0, b=1, ordinary
after cleanup (4). Require exact token **and** logit parity ordinary=b0 and
ordinary=post-cleanup; nonzero hidden/logit change at b=1; all 6 sites hooked
at prefill and every decode step (decode count = suffix length − 1), zero
skipped calls, zero calls on linear and out-of-window blocks; finite
diagnostics. Failure → `port_or_mechanics_failure`, an implementation result,
not a method null. No retry of an uncertain GPU phase.

**6.3 Development (R2 after).** Fresh on dev32, greedy, repetition penalty
1.05, cap 160, batch 1, seed 42: baseline, prompted, GCAD b ∈ **{1, 2}**
(128 answers). Judge sarcasm and coherence, blinded fully packed batches of 8,
same-question conditions separated, plans frozen before generation.

*Strength grid (PM decision):* two strengths. With 6 sites versus 11 on 7B,
both may be under-strength; a `development_null` is then the honest outcome
and does not license b > 2 in this study.

**Selection:** lowest b meeting all of: SARC mean ≥ baseline + 20; ≥ 8/32
answers with SARC ≥ 50; coherence ≥ 50 and ≥ baseline − 20; invalid ≤ 3/32
(empty or distinct-bigram ratio < 0.6). None → `development_null`, close, no
final generation. Prompted gain < 20 is a reported validity concern only.
Register selected b and random-control tensors before final generation.
The final-analysis code (frozen `final_decision` mapping, H14-excluded
figures, descriptive metrics and terminal statuses) must be hashed into a
registration **before final registration**; it need not exist at the
synthetic smoke. No scientific stage (elicitation onward) runs until the
review-02 corrections C1–C4 are implemented and CPU-tested.

**6.4 Final (locked).** Fresh on final32: baseline, prompted, selected GCAD,
random GCAD (128 answers), same decoding. Random Δ: per site, CPU generator
seeded `seed + 1000·layer_index` (absolute layer index), seeds
**[9831, 9832, 9833]** by question index mod 3, `randn(5120)` normalized and
scaled to that site's raw Δ norm; same K_pos, D̄, gate, scale, b, sites. Store
tensors and hashes. Realized shift norms reported; no energy-matching claim.

**Retain** iff the selected arm meets all development gates on final32 **and**
SARC mean ≥ random + 20. Otherwise `final_failure`. No fallback strength.

## 7. Metrics

| Metric | Role |
| --- | --- |
| SARC mean, count ≥ 50 | gate |
| GCAD coherence mean | gate (absolute 50, relative −20) |
| Invalid (empty / bigram < .6) | gate (≤ 3) |
| Impoliteness mean (final only, after primary streams; prompted arm is required at dev and final as a descriptive reference) | descriptive coupling; within-arm Pearson, pooled with confounding caveat |
| Cap hits /32, EOS rate, mean tokens | descriptive; coherence is judged on capped text and this is stated |
| Lexicon hits, H14-excluded figures, cache/no-cache agreement, realized shift norms | descriptive |

Grading order on final: sarcasm, coherence, then impoliteness. A fatal stop
during impoliteness yields `primary_complete_descriptive_incomplete`; any
earlier fatal stop is a private `incomplete` record.

## 8. Judge runtime

Tools-off native Claude CLI judge, `claude-opus-5-5`, medium effort, fresh
isolated contexts, payloads with opaque IDs only. The installed CLI is now
2.1.327 (sha256 `343c744b…6ac9`), not the audited 27B pin 2.1.326. This study
registers a **new isolated runtime audit for 2.1.327** (own directory and
binding, one audit call) before any judge call; the old runtime record stays
unchanged. A failed audit is a blocker for PM, not a study null.

**Runtime snapshot (preservation, PM requirement).** Before the audit, copy
the chosen binary (2.1.327) into the new run directory (e.g.
`outputs/qwen38-sarcasm-gcad-01/judge-runtime/claude-2.1.327`), record source
path, size, sha256 and `--version` of the copy, and pin that **absolute
snapshot path** in the binding. Every audit and judge call invokes only the
snapshot; identity is re-checked (hash + version) before and after each call.
Background updates of `~/.local/bin/claude` then cannot change the study
runtime. The snapshot is a local runtime copy, never published. If the
snapshot stops working (e.g. server-side deprecation), that is an
infrastructure stop for PM, not a study null and not a silent switch.
Advisor/PM/worker agent contexts are never used as judge contexts; every judge
call is a fresh isolated tools-off process with opaque-ID payloads.

**Judge payload difference (prospectively disclosed).** Rubric text is the
exact historical bytes (trait = `extract.json` `eval_prompt`, GCAD coherence
definition, author impoliteness prompt), one rubric per payload, with question
and answer as context. The surrounding JSON envelope and rating schema differ
from the 7B/4B studies (integer score, or the string `REFUSAL` for trait; no
`reason`/`refusal` fields or expected-ID list). Within-study comparisons are
unaffected; cross-model score comparisons carry this caveat. ≤ 2 identical structural attempts per batch;
auth/runtime/model/identity/uncertain-start errors are fatal. Rubrics are not
re-calibrated (previously calibrated and reused unchanged); disclosed.

## 9. Budgets (maxima, prospectively fixed)

| Item | Count |
| --- | --- |
| Generations | 6 synthetic mechanics + 200 elicitation + 6 real mechanics + 128 dev + 128 final = **468** |
| Max generated tokens | 6·32 + 200·256 + 6·32 + 128·160 + 128·160 = **92,544** (92,352 without the synthetic phase) |
| Teacher-forced forwards | 2 synthetic + ≤ 300 real bank = **≤ 302** |
| Logit forwards | 4 synthetic + 4 real = **8** |
| First-valid judge batches | 50 elicitation + 32 dev + 48 final = **130**; ≤ 260 starts; plus 1 runtime audit call |
| GPU | one A6000 (NF4) for mechanics, dev and final; the second may run a fixed even/odd elicitation shard in its own process. No exact-parity or arm comparison crosses processes or GPUs. Allocation through PM. |

**Runtime estimate.** ≤ 92,544 generated tokens. Batch-1 NF4 27B decode is
assumed 8–15 tok/s (measured in synthetic mechanics and reported), giving
**≈ 1.7–3.2 GPU-h** generation plus < 0.5 h loading/teacher forcing; hard stop
at 8 GPU-h. Judge ≈ 130 sequential batches, ≈ 2.5–4.5 h wall. End-to-end
≈ 6–9 h wall including reviews, versus the 24.44 GPU-h whole 27B science
track. Exceeding a count budget is fatal, not extendable.

## 9a. Implementation failures versus scientific outcomes

A deterministic implementation bug (wrong boundary, shape, hook site, budget
accounting, crash, failed parity) is a `port_or_mechanics_failure` or
implementation stop, **never** a scientific null. A fix requires: (1) the
failed run, inputs and logs preserved unchanged; (2) a written diagnosis; (3) a
fresh prospective registration of the changed source hashes by PM before any
further data. Data produced by the buggy source is not mixed into later gates.
Changes are allowed only for correctness against this registered design; no
change may be motivated by observed sarcasm, coherence or other quality
scores, and thresholds, strengths, fixtures and arms never change.

## 10. Disclosed limitations

- Development and final questions are reused project material. All 32 final
  items were the final test for 7B and 4B: this is an explicitly reused
  cross-model evaluation, not new project-held-out data. No 27B final answer
  is generated before strength selection is registered. The 27B has never been tuned on it, but the
  designers know its behavior from prior models. Exact-overlap checks are
  lexical only; x18↔H14 topical overlap is known.
- Extraction text, fixtures and judge are within one Claude family.
- NF4 representation; cached/full-prefix nonequivalence; greedy cap 160
  truncation; coherence is not factual accuracy; n = 32 per arm.
- GCAD acts on 6 of 64 blocks; linear-attention blocks are untouched by
  design, so this tests GCAD on the full-attention subset of a hybrid model,
  not a full-model analogue of the 7B controller.
- Outcomes are model-specific; no size trend is inferred.

## 11. After the study

Stop. Pass: retain the selected b as an NF4 27B sarcasm control for this
bounded task, beside 7B/4B results. Null or failure: close with the measured
outcome. Any publication is a separate decision.
