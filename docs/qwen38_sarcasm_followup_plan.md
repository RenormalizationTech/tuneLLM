# Qwen3.8-27B sarcasm follow-up: instruction-activation diagnostics and stronger GCAD

Advisor design, 2026-11-19. **Status: FINAL design for PM registration.**

The PM chose the **+ random at b=16** variant: 224 Part B outputs. The worker
implements from this file. It changes only through a recorded amendment
before registration, and it is frozen at registration.

The user explicitly authorized this follow-up. It is a new, bounded,
exploratory study. The closed study `qwen38-sarcasm-gcad-01` stays frozen,
and its `development_null` outcome is unchanged. Nothing here re-opens,
re-analyses or re-labels that result.

No model or judge call happens until the PM approves the completed
registration. After that single approval, one audited launcher runs every
stage under the guards fixed below, with no further stage prompts.

## 1. Questions and claims

- **Part A (post-hoc mechanistic diagnostic, observational).** When the
  *answer tokens are held fixed*, how does an explicit sarcasm instruction
  change the model's internal activations compared with a literal (sincere)
  instruction? How large is that change, and how is it oriented relative to
  the frozen GCAD bank Δ? The analysis also separates instruction effects
  from content effects (sarcastic vs literal answer text) and from
  system-presence or position effects.
- **Part B (exploratory dose comparison).** Does the frozen bank, applied at
  larger strengths b ∈ {2, 4, 8, 16}, produce coherent sarcasm on 32 fresh
  development questions? The actual steering coefficients and shift
  magnitudes are recorded.

**Claims allowed.** Descriptive and observational statements about these
fixtures, this NF4 model and this bank only.

**Claims not allowed.**
- No causal claim from Part A. Activation differences under fixed tokens are
  correlational and context-conditioned.
- No validated-controller claim and no general specificity claim from
  Part B.
  - There is no independent final set.
  - The b=16-vs-random comparison is the **only** dose-matched direction
    control.
  - A pass at b = 2, 4 or 8 has no matched random arm and stays exploratory
    and not direction-specific.
  - A qualifying strength is at most a *candidate* for a separately
    registered confirmation study.
- Part A must not influence Part B. Part B's arms are fixed here and are not
  chosen from Part A results; both are registered before either runs.

## 2. Fixed inputs (all hash-bound at registration)

| Input | Source | Note |
| --- | --- | --- |
| Model | `artifacts/qwen38-27b-nf4-01`, same manifest hashes as the old R0 | NF4, BF16 compute, SDPA, batch 1, cached `generate` |
| Bank | `outputs/qwen38-sarcasm-gcad-01/extraction/bank.safetensors`, sha256 `4fd9f86b…45de` | Read-only; copied into the new run with a hash check |
| Sites | full-attention layers 23, 27, 31, 35, 39, 43 | Same rule `a' = a + 2b·σ(1.5(d − D̄))·Δ`, inverse = False, scale 1.5 |
| Contexts for Part A | sarcastic system S: the old prompted string (pair-0 positive). Literal system L: the pair-0 negative string. No system N: the old baseline rendering | The old synthetic-smoke boundaries recorded a 45-token system span for both pair-0 strings on extraction question 0. That is supporting evidence, not proof for these prompts. The launcher must **measure**, with the pinned tokenizer, S and L prompt token lengths for every question, and assert that they are equal. If they are unequal for any question, prepare refuses: an `implementation_stop` before any GPU call, with no padding or substitution. |
| Part A continuations | old development rows: for each C01–C32, `baseline` output IDs (content B) and `prompted` output IDs (content P), all exactly 160 tokens (all capped) | Saved raw IDs, never re-tokenized. Explicitly post-hoc, because their scores are known. |
| Part B questions | `evaluation/qwen38_sarcasm_followup/development.json`, F01–F32, sha256 `3da540a3…add4b`, PM-authored with `provenance.json` | Fixture review in §8 |
| Rubrics | the old run's `rubrics.json` (trait, coherence, impolite), byte-identical, hash-checked | Same rubric text as the closed study |
| Judge | a fresh snapshot copy of `claude-2.1.327` (sha256 `343c744b…6ac9`) into the new run directory, mode 0500, plus a fresh isolated audit | Same flags as the old binding |
| Decoding (B) | greedy, repetition penalty 1.05, cap 160, model-default overrides as before, seed 42 | Identical to the old development stage |

Steering, extraction and mechanics in the *frozen* modules
(`qwen38_gcad_port.py`, `qwen38_sarcasm.py`) are imported read-only and never
edited. New behaviour lives in new modules (§9).

## 2a. Post-hoc context: saved-bank geometry (descriptive, not a criterion)

The PM computed this on CPU from saved tensors in
`outputs/qwen38-sarcasm-followup-pm-01/saved-bank-geometry-01.json`, with
source sha256 `b0a75622…`. The advisor recomputed it independently and the
values agree.

- Across the 90 pairs, at all 6 sites, **no** pair difference has a negative
  cosine to the mean.
- Median pair-to-mean cosine: 0.84–0.94.
- ‖mean‖ / mean‖pair‖: 0.82–0.93.

**Reading.** Gross cancellation across extraction pairs is not supported:
Δ is not small *because* pairs cancel.

**Caveat.** The cropped quantity is attention to the *system tokens*. Within
each of the 5 instruction pairs, the sarcastic and sincere system strings are
fixed across all 20 questions. High pair consistency may therefore partly
reflect that fixed lexical difference, not a sarcasm-specific direction.

**Limits.** These numbers do not show that Δ is sarcasm-specific, and they say
nothing about adequate realized dose. They do not select or change anything:
strengths stay {2, 4, 8, 16}. Part A's A1 (instruction vs content alignment
with Δ) and Part B's recorded actual shift address those two questions
directly.

**Norm-unit correction (applies to all closed-study interpretations).**
`port.extract`'s `attention_output_norm` is the Frobenius norm over the whole
sequence tensor. It is not a per-token norm.

For the synthetic run, that sequence was 89 tokens, so a uniform per-token
norm would be about √89 ≈ 9.4 times smaller. The advisor's ratios of ‖Δ‖ to
that number are therefore unit-inconsistent:
- review 04: 1.0–3.2%;
- review 09: 0.6–1.5%;
- review 10: the qualitative "small Δ".

They must not be used. PM record:
`results/reviews/qwen38_sarcasm_norm_interpretation_correction_01.md`.

Gates, reconstruction errors (where numerator and denominator are both
Frobenius), parity and the `development_null` decision are unaffected. This
follow-up uses only matched per-token norms: §A2 and the §4 telemetry.
"Insufficient dose" and "direction/site mismatch" both remain untested
hypotheses. Neither is concluded from closed-study numbers.

## 3. Part A design: fixed-token context replay

**Cells.** For each old development question q (32), there are
3 contexts {S, L, N} × 2 contents {B, P}, giving 6 teacher-forced passes.
- Input: the rendered prompt of the context, followed by the saved 160
  content IDs. No EOS is appended (all were capped).
- Response mask: exactly those 160 positions, with special-token positions
  excluded. If a content ID is special, it is excluded in every context, so
  the masks stay identical.
- **Total: 192 passes.**

**Assertions per question (an implementation stop if violated):**
- S and L prompts have equal token length;
- the response positions are identical across S and L;
- the content IDs are identical across all three contexts;
- the old dev rows' hashes match the old run's receipts;
- the user-turn tokens are equal across contexts.

**Captured per pass.** Observational hooks only: no output is modified, and
the capture is shown to be bit-neutral in mechanics.
1. **Residual stream.** The outputs of all 64 decoder blocks
   (`hidden_states[1..64]`). For each, store the response-token mean (FP32,
   5120) and the mean per-token L2 norm.
2. **The 6 GCAD sites.**
   - Native `self_attn` output after the gated `o_proj`: its response-token
     mean and mean per-token norm.
   - For S and L, the system-cropped gated contribution: its response-token
     mean, using the frozen `port.extract` arithmetic. This is the quantity
     the bank was built from. N has no system span, so no crop is taken.
   - The per-token demand `d_t` against the frozen bank K_pos, and the gate
     value the bank would apply, `g_t = σ(1.5(d_t − D̄))`. Store the mean,
     the standard deviation and the 5/50/95 percentiles.
3. **Continuation NLL.** The mean token negative log-likelihood of the fixed
   continuation under the context. It is computed on GPU from logits at the
   response positions, and only scalars are stored.

**Storage estimate.** About 192 × 64 × 5120 × 4 B ≈ 250 MB for the residual
means (pooled means only; no full residual trajectories). Add about 1.3 GB
of S/L per-token attention outputs at the 6 sites (§A2), or their reduced
paired statistics. All raw arrays live in ignored local outputs, and only
the summaries enter tracked results.

**Contrasts.** All are per question, then averaged over the 32 questions. X is
any captured mean vector.
- Instruction effect at fixed content: `I_c = X(S,c) − X(L,c)`, for
  c ∈ {B, P}. This is the **primary** contrast: positions are matched and
  tokens are identical.
- Content effect at fixed context: `C_x = X(x,P) − X(x,B)`, for
  x ∈ {S, L, N}. Content lengths are equal (160), but the tokens differ.
- Interaction: `I_P − I_B`.
- System presence, confounded with position: `X(L,c) − X(N,c)`. This is
  secondary and labelled position-confounded. N responses are about 45
  positions earlier, so this contrast cannot separate "instruction present"
  from "position shift". No attempt is made to disentangle them.

**Prespecified summaries.** These are the only summaries that may appear in
the headline. Everything else is a supplementary table.

- **A1. Alignment with the bank, at the 6 sites.**
  - Report cos(mean I_P, Δ) and cos(mean I_B, Δ) for the native attention
    output.
  - Report the same cosines for the cropped system contribution.
  - Report cos(mean cropped(S,P) − cropped(L,B), Δ). This is a bank-analogue
    on the development questions, using pair 0 only.
  - Each cosine gets a 95% percentile bootstrap CI (10,000 resamples, seed
    20261119).
  - **The bootstrap unit is the question: 32 clusters.** Each resample draws
    32 question IDs with replacement and carries all 6 cells of each drawn
    question together. Both saved continuations (B, P) under S, L and N stay
    paired. The statistic is then recomputed on the resampled per-question
    vectors. Tokens, the 64 trajectories and layers are **never** resampled
    as independent units.
  - The intervals are descriptive, not hypothesis tests.
  - **Headline:** the actual **signed** cosines, with their
    question-bootstrap CIs.
  - Scale references, CPU-only, from saved tensors, with no model call:
    - 1/√5120 ≈ 0.014 is the SD / RMS scale of the cosine between a fixed
      vector and an isotropic random direction in 5120 dimensions. The
      expected signed cosine is 0. It is a scale reference, not a
      confidence bound.
    - The signed cosines of the same contrast with the 3 stored random
      directions (`random-983{1,2,3}`) are reported as 3 individual
      reference values.

    Neither reference defines a confidence range or a binary
    direction test.
- **A2. Magnitude at the 6 sites.** Native gated attention output, S vs L,
  at fixed content c ∈ {B, P}. Two registered measures, because the norm of a
  pooled mean can cancel changes that vary from token to token:
  - **(a) Per-token paired norm.** For each question q and response token t:
    - n_qt = ‖a_S,qt − a_L,qt‖, computed in FP32 from native-dtype tensors;
    - relative norm r_qt = n_qt / ‖a_L,qt‖, where the denominator is the L
      per-token native norm at the same token.

    Average over t within each question, then over questions. Report the
    mean n and mean r, with question-cluster bootstrap CIs.
  - **(b) Norm of the pooled mean.** ‖mean_q mean_t (a_S − a_L)‖. Its
    denominator is the mean over q and t of ‖a_L,qt‖, stated explicitly.
    Report it with a question-cluster bootstrap CI. The per-question version,
    ‖mean_t (a_S,qt − a_L,qt)‖, is also reported.
  - **Comparison (descriptive only).** Put (a) next to Part B's per-token
    actual / native ratio at each b, with prefill and decode separate. Also
    show the algebraic coef·‖Δ‖ / native and the old b=1, 2 bound 2b‖Δ‖.
  - **Conditioning differs.** Part A holds the continuation fixed: the same
    tokens under S and L. Part B's ratio comes from free generation, where
    the steered trajectory diverges, so the native output it is divided by
    belongs to a different token sequence. The two ratios are therefore not
    the same quantity.
  - **Not an explanation of the old null.** No statement may conclude that
    dose caused the old null. The comparison describes scale only.
  - **Retention.** To compute (a) and its bootstrap, the per-token S and L
    attention outputs are kept at the 6 sites, for both contents, in native
    dtype. That is 128 cells × 160 tokens × 6 × 5120 × 2 B ≈ 1.3 GB, held in
    ignored local run outputs and never committed. A paired temporary capture
    that stores only the per-token n_qt, r_qt and per-question means is also
    acceptable. No extra model calls are made: these come from the same 192
    passes.
- **A3. Residual stream, all 64 layers.**
  - Report ‖mean I_c‖ / mean residual token norm.
  - Report ‖mean C_S‖ / the same norm.
  - Report consistency: the mean of cos(I_c,q, leave-one-out mean I_c) across
    questions.
  - Linear and full-attention layers are marked. Residual readings are
    observational only; no layer is hooked for steering.
- **A4. Gate under each context.** Report the mean `g_t` under S vs L vs N
  for each content: does the instruction open the GCAD gate? Paired
  differences S − L get bootstrap CIs, using the same question-cluster
  bootstrap.
- **A5. Likelihood.** Report NLL(P | S) − NLL(P | L) and
  NLL(B | S) − NLL(B | L), paired by question, with question-cluster CIs.

**Scope of the S−L contrast.** S and L differ in their entire fixed wording
(sarcastic vs sincere persona, and different instruction sentences), not in
a single trait token. The S−L contrast is therefore "this sarcastic
instruction vs this sincere instruction". It **cannot isolate pure trait
specificity** from other lexical or semantic differences between the two
strings.

**Interpretation rules, fixed in advance.**
- **Direction.** Report the signed cosine, its CI and the 3 reference
  values. Describe them with these words:
  - **positively aligned** if the CI lies entirely above 0;
  - **anti-aligned** if the CI lies entirely below 0;
  - **no clear alignment** if the CI includes 0.

  Add the magnitude in words, for example "weak, cos ≈ 0.05", relative to
  the 0.014 scale. A value outside the 3 reference values is never by itself
  called "aligned"; a negative cosine is anti-aligned. No statement is
  causal, and none is a hypothesis test.
- **Magnitude.** Report the measured ratios only. "Dose too small" may be
  stated only as "the prompt-induced change is N× larger in norm than the
  injected shift", never as an explanation of the old null.
- **Post-hoc status.** The old continuations and their scores were seen
  before this design. Part A is labelled post-hoc and mechanistic throughout.

## 4. Part B design: stronger GCAD on fresh32

**Arms.** The 7 arms are:
- baseline;
- prompted (the S system, exactly as in the old prompted arm);
- GCAD at b = 2, 4, 8 and 16, using the frozen bank, sites, scale and the
  same cached decoding;
- **random at b = 16**, using the stored `random-9831.safetensors`. It is
  copied with a hash check, has per-site ‖Δ‖ equal to the bank's, and has
  identical K_pos, D̄, gate, scale and sites. It goes through the same
  recording steerer.

Counts:
- 7 arms × 32 = **224 outputs**, cap 160, at most **35,840 generated
  tokens**.
- Arm order within each question is fixed in the config. Generation is
  sequential and batch 1.
- b = 16 is the largest strength tested. There is no "strongest qualifying"
  search and no extrapolation beyond 16.

**Recording steerer** (new module, arithmetic-identical to the frozen
`Steerer`). For each generate call and each site, separately for prefill and
decode:
- token count;
- the coefficient `coef_t = 2b·g_t`: mean, minimum, maximum and percentiles;
- the demand mean;
- **requested norm**: the per-token ‖(coef_t·Δ)‖, computed in the model
  dtype exactly as added (the BF16 product tensor), with its norm accumulated
  in FP32. Report the mean and maximum;
- **actual norm**: the per-token ‖steered − original‖, taken in FP32 from
  the BF16 output tensors after the add. This is the shift actually applied
  after BF16 product and add rounding. Report the mean and maximum;
- **algebraic** coef_t·‖Δ‖, reported separately and labelled algebraic;
- the native attention-output per-token norm, before the shift: mean;
- the ratios actual / native and requested / native: mean and maximum;
- the bound `2b‖Δ‖` and its ratio to the native norm.

The actual norm is **not** forced to equal the requested or algebraic norm.
BF16 rounding makes them differ, and the differences are reported.

Decode records are kept per step: 159 steps × 6 sites, which is small. The
same records are produced for b = 0 in mechanics. This makes the actual
applied shift **measured** in this follow-up, unlike the closed study.

**Fixed gates.** Each b is compared with baseline using the unchanged
historical rule, through the frozen `sarcasm_method_transfer_common.summarize`
and `gates`:
- SARC ≥ baseline + 20;
- at least 8/32 answers with SARC ≥ 50;
- coherence ≥ 50 and ≥ baseline − 20;
- invalid (empty, or distinct-bigram < 0.6) ≤ 3.

**Exploratory outcome.**
- If one or more strengths pass, report `exploratory_candidate = lowest
  passing b`, with every strength's gate table shown.
- If none passes, report `exploratory_null_through_b16`.

**Dose-matched direction check (b=16 only, reported, not a selection
gate).** `b16_direction_check = SARC(b16) ≥ SARC(random b16) + 20`. The
random arm's own gate table is reported descriptively.
- If b16 passes all gates **and** the direction check, write: "at b=16 the
  bank direction exceeds a norm-matched random direction on these 32
  questions". This is exploratory and is not a general specificity claim.
- If b16 passes the gates but fails the direction check, write: "the b=16
  effect is not distinguishable from a norm-matched random perturbation".
- Lower-strength passes are never compared with random.

**Coherent-response checks (descriptive).** For each arm:
- the count of answers with SARC ≥ 50 **and** coherence ≥ 50
  ("coherent sarcastic");
- the count with coherence < 50;
- invalid, empty, mostly-not-letters, caps and EOS;
- mean tokens;
- lexicon hits;
- impoliteness mean, within-arm and pooled SARC–impoliteness Pearson, with
  the confounding caveat.

A strength that passes the SARC gates mainly through incoherent or
impolite text is flagged in the narrative. The gates themselves are not
changed.

**Dose summary (descriptive).** Report SARC, coherence and coherent-sarcastic
counts against b, alongside the actual shift / native norm. Prefill and
decode are reported separately. No curve
fitting and no monotonicity test.

**Prompted reference.** Required and reported. It is not a gate. If prompted
SARC < baseline + 20, report a judge or fixture validity concern, as in the
old `choose`.

## 5. Mechanics (registered, real model, before any Part A or B data)

All criteria are exact. Any miss is an `implementation_stop`: preserve the
evidence, run no later GPU stage, and record no scientific outcome.

Use the two old mechanics probes C01 and C02 (cap 32). This keeps the fresh
questions untouched.
1. Ordinary vs recording-steerer b=0: identical token IDs, and `torch.equal`
   on last-position logits.
2. Recording steerer at b=2 vs frozen `Steerer` at b=2: `torch.equal` on the
   logits, and identical tokens (2 generations at cap 32). This proves the
   recording adds no arithmetic change.
3. Post-cleanup ordinary logits are `torch.equal` to the first ordinary
   logits. The hook signatures on every module are restored.
4. Coverage per site per call: prefill 1, decode = length − 1, errors 0.
   There are no hooks on linear or out-of-window attention modules.
5. Recording consistency, per site:
   - every recorded `coef_t` satisfies 0 ≤ coef_t ≤ 2b;
   - on one probe call, the logged requested and actual norms equal a
     **manual recomputation** from the captured tensors: ‖(coef·Δ) in model
     dtype‖ and ‖(steered − original) in FP32‖, within FP32 tolerance
     (relative 1e-5). Logged values are not compared with the algebraic
     coef·‖Δ‖;
   - at b=0 the requested and actual norms are **exactly 0**;
   - instrumented outputs are bit-equal to the frozen `Steerer` outputs
     (criterion 2).
6. Part A capture neutrality. On one Part A cell (C01, context S, content P),
   the logits with all capture hooks attached are `torch.equal` to the logits
   without them. The cropped reconstruction error is ≤ 0.02, the gate-omitted
   error is ≥ 0.2 and ≥ 10× the gated error, and the split error is ≤ 0.01.
   These are the old F4 thresholds.
7. All bank tensors are finite, and the bank hash equals `4fd9f86b…`.

**Mechanics counts.**
- Generations: ordinary ×2, b0 ×2, frozen b2 ×2, recording b2 ×2 = 8, at cap
  32, so at most 256 tokens.
- Logit forwards: ordinary, b0, frozen b2, recording b2, post-cleanup
  ordinary = 5.
- Teacher-forced passes: capture on and capture off = 2.

## 6. Judge plan

- **Audit.** A fresh isolated audit with the new snapshot: 1 call, which must
  return `{"project":"UNKNOWN"}` and report only `claude-opus-5-5`. If it
  fails, the launcher still completes all GPU stages, then stops before any
  judging with `judge_audit_failure`. No scores result. This avoids re-running
  the GPU if the PM later fixes the judge under a new registration.
- **Streams.** trait (sarcasm), coherence, then impoliteness. Impoliteness
  starts only after both primary streams are complete.
- **Batches.** 224 outputs ÷ 8 per batch = 28 batches per metric, so **84
  batches**. Each batch holds 8 distinct questions with arms mixed. The order
  uses fixed seeds 20261120 + metric index (trait 0, coherence 1, impolite 2) and is frozen in `plans.json` at prepare.
  Items carry opaque IDs only.
- **Attempts.** At most 2 per batch, and the second is allowed only for an
  invalid structural submission. A model mismatch or identity change is
  fatal. The hard cap is **168 study starts plus 1 audit**.
- **Primary-complete.** A fatal stop during impoliteness gives
  `primary_complete_descriptive_incomplete`, using the same rule as the
  closed study.

## 7. Methodological priorities and known limits

1. **Random control only at b=16 (adopted).** At b=8–16, any large
   perturbation may change style or tone. The single norm-matched random arm
   (seed 9831) controls direction at b=16 only. A pass at b = 2, 4 or 8 is
   reported as "not direction-tested". One random seed on one fixture does
   not support a general specificity claim.
2. **Order and independence.** Part A runs before Part B for runtime reasons
   only. No Part A output is read by the launcher when it runs Part B.
3. **On-policy confound in Part A.** Content B was generated under N, and
   content P under S. Each content is off-policy in the other contexts. A5
   (NLL) quantifies this. The S-vs-L contrast at fixed content is still
   clean in tokens and positions.
4. **Pair-0 only.** Part A uses one instruction pair (the prompted arm's
   strings). The bank pooled 5 pairs. The bank-analogue cosine in A1 is
   therefore indicative, not a reproduction.
5. **Caps.** All Part A contents are 160 tokens, capped. Part B is judged on
   text capped at 160 tokens, as before.
6. **NF4 and cache.** These are carried over from the closed study. The
   cached path is identical across all arms.
7. **Fixture.** The PM authored it with AI assistance after seeing the old
   topics. It is new to the project, with no claim about pretraining
   exposure (§8).
8. **Multiplicity.** Part B has 4 strengths × 5 gates on a single 32-question
   set, with no independent final. Any pass is exploratory.

## 8. Fixture review (PM fresh32, advisor read-only)

- **Exact and near-paraphrase overlap with the old 84 questions: none
  found.** These are the closest topical neighbours, and none is a
  paraphrase:
  - F23 (induction cooktop) and final "microwave oven";
  - F22 (station echo) and final "noise-canceling headphones" or "quieter
    home office";
  - F07 (label adhesive) and the dev or final stain and rust removals;
  - F21 (unlabeled fan switch) and dev "tripped circuit breaker";
  - F13 (static cling) and final "wrinkles without an iron".

  The final fixture is not used in this follow-up, so these neighbours do
  not affect Part B.
- **Neutrality: acceptable.** No item requests a tone or persona, or invites
  insult. A few frustration topics may make the baseline slightly wry: F04
  (elevator close-door button, a well-known placebo topic), F06, F16, F18 and
  F28. This is acceptable and similar to the old extraction set. Note it in
  the closeout.
- **Task mix.**
  - About 17 explanatory items (why or how-does), about 14 practical how-to
    or organization items, and 1 troubleshooting item. That is close to the
    old development set's balance.
  - There is no creative-writing item; the old development set had 1.
  - The domain skews toward household objects and mechanisms, with no
    cooking, nature or biology items.
  - Questions are about twice as long as the old development items.

  Comparisons with the old development numbers are therefore not
  like-for-like.
- **Within-fixture redundancy (minor).**
  - F05 (whiteboard: current vs stale) and F31 (checklist: repeated vs
    one-time) are near-duplicates in intent.
  - F03, F09, F24, F31 and F05 form an organizing and labelling cluster.

  This slightly reduces the effective n. It is not a reason to rewrite: the
  PM owns the fixture, and it should be frozen as is.

## 9. Implementation (worker; new files only)

**New files.** Suggested names, all new; the old sources stay immutable:
- `scripts/qwen38_sarcasm_followup.py`: the launcher, prepare and stages;
- `scripts/qwen38_gcad_record.py`: the recording steerer and Part A capture;
- `scripts/qwen38_sarcasm_followup_analyze.py`: the closeout analyzer;
- `configs/qwen38_sarcasm_followup.json`;
- `tests/test_qwen38_sarcasm_followup*.py`.

**Run root.** `outputs/qwen38-sarcasm-followup-01`.

**CPU tests required before registration.**
- The recording steerer is `torch.equal` to the frozen `Steerer` in output,
  on the tiny Qwen3.5 fixture, at b ∈ {0, 2, 16}. Its records satisfy
  0 ≤ coef ≤ 2b. Requested and actual norms equal a manual recomputation
  within FP32 tolerance. b=0 gives exact zeros. The algebraic value is
  reported separately and not asserted equal.
- Prefill and decode are recorded separately, and the decode step count is
  correct.
- Part A capture is bit-neutral. The residual means equal a manual
  computation from `hidden_states`. The cropped mean equals `port.extract`.
- Part A assertions: S and L length equality, identical masks, and the
  special-token exclusion path.
- Contrast and bootstrap arithmetic on synthetic tensors, with a fixed seed
  and deterministic output.
- Judge plan: 28 batches per metric (7 arms), 8 distinct questions per batch, every
  item exactly once per metric.
- Gates go through the unchanged frozen helpers, and the outcome labels
  follow §4.
- Launcher guards: budget refusal, resume without re-running completed rows,
  no retry of started generation rows, and the stop paths in §10.

## 10. Single launcher: stages, guards and budgets

**One command after one PM approval.** The approval binds the registration
JSON hash, which covers:
- the source, test and config hashes;
- the old artifacts used (bank, old development rows and receipts, rubrics,
  model manifest);
- the fixture hash;
- the snapshot hash;
- this plan's hash.

The launcher verifies all of them at start. It never prompts, and it runs
these stages in order:

| # | Stage | Count (maximum) | Guard / on failure |
| --- | --- | --- | --- |
| 0 | Preflight: hashes, GPU identity (GPU0), free memory, run root empty or resumable | – | any mismatch: refuse to start |
| 1 | Judge audit | 1 judge call | failure: mark `judge_audit_failure`; run GPU stages 2–4, skip 5, closeout |
| 2 | Mechanics (§5) | 8 gen (≤ 256 tok), 5 logit, 2 TF | any miss: `implementation_stop`, halt |
| 3 | Part A replay | 192 TF | assertion or implementation error: `implementation_stop`, halt (Part B not run) |
| 4 | Part B generation | 224 gen, ≤ 35,840 tok | error: `implementation_stop`, halt |
| 5 | Judging | 84 batches, ≤ 168 starts | structural failure after 2 attempts, or a fatal error: stop; primary-complete rule applies |
| 6 | Closeout analyzer | – | writes a fresh immutable `results/qwen38_sarcasm_followup_01.json` |

**Hard caps, enforced by counters before each call.**

| Resource | Cap |
| --- | --- |
| Generations | 232 |
| Generated-token reservation | 36,096 |
| Teacher-forced passes | 194 |
| Logit forwards | 5 |
| Judge starts | 168 study + 1 audit |
| GPU process wall time | **4 h hard stop**, including loads |

**Final counts (PM decision: + random at b=16).**

| Part B outputs | Generations (incl. 8 mechanics) | Token reservation | TF | Logit | Judge batches | Judge starts | GPU estimate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 224 | 232 | 36,096 | 194 | 5 | 84 | ≤ 168 study + 1 audit | about 1.7–2.3 h (4 h hard stop) |

The judge plan has 224 / 8 = 28 batches per metric, each with 8 distinct
questions. Random tensors are the stored `random-9831.safetensors`, hash
checked; their per-site ‖Δ‖, K_pos and D̄ match the bank.

**Stage-entry budget check.** Before stages 3 and 4, the launcher checks that
the remaining GPU time covers the stage's worst case, using the measured
throughput from the old run: 5.6 tok/s, with 600 s allowed per model load.
If it does not, the launcher stops with `budget_stop` and does not start a
partial stage. Part A results stand alone if Part B is budget-stopped.

**Estimated runtime.** On GPU0:

| Item | Estimate |
| --- | --- |
| Model load | about 1–2 min |
| Mechanics | about 2 min |
| Part A (192 TF passes of about 230 tokens) | about 3–5 min, from the old 180 passes in 110 s, plus capture overhead |
| Part B (35,840 tokens at 5.6–7 tok/s) | about 1.4–1.8 h |
| **GPU total** | **about 1.7–2.3 h**, inside the 4 h stop |
| Judging (84 batches at the old pace) | about 1.8–3.5 h |
| **End to end** | **about 3.5–6 h** |

**Implementation failures.** Plan §9a of the closed study applies unchanged.
A deterministic bug is never a scientific null. Preserve the failure, and
fix it under a new registration, never in response to scores. Completed
rows are reused only if their receipts hash-match and the bound source is
unchanged.

**Closeout content.**
- Part A: the A1–A5 tables, all labelled post-hoc and observational.
- Part B: the gate table for each b, the exploratory label, the
  coherent-response checks, the recorded actual, requested and algebraic shift
  ratios (prefill and decode separately), and impoliteness descriptives.
- The ledger.
- The §7 limitations.
- An explicit statement that the old `development_null` is unchanged.

**Additional CPU tests for the finalized variant.**
- The random arm loads `random-9831.safetensors` with its hash check. Its
  per-site ‖Δ‖ equals the bank's, and K_pos and D̄ are identical.
- The b16 direction check and the outcome labels follow §4.
- The bootstrap resamples whole questions, carrying all 6 Part A cells
  together. A test shows that resampling tokens or trajectories is
  impossible in the API.

## 11. Worker hand-off

This file is the specification. The worker implements only the new files in
§9, and the old files are immutable.

**Data-access rule.** The worker reads old artifacts (bank, development rows,
rubrics, random tensors) only through hash-checked copies or reads. The
worker makes no model or judge calls.

**Requested deliverables, in order:**
1. The recording steerer and Part A capture, with CPU tests.
2. The launcher with all stage guards and counters.
3. The analyzer.
4. The config with every count in §10.

Each one is reviewed as it lands. New findings go to
`results/reviews/qwen38_sarcasm_followup_review_NN.md`.
