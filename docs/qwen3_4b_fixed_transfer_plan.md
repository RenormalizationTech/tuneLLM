# Qwen3-4B fixed-setting transfer: plan

Status: **accepted by root for implementation and CPU preparation on
2026-11-16. Not yet registered for model or judge calls.** Advisor review:
`results/reviews/qwen3_4b_fixed_transfer_design_review.md`. The worker is
implementing this plan; R1 must precede model calls and R2 must precede
evaluation.

## 1. Scope (minimal, per the user)

- **7B refusal is closed.** The user closed the 7B political study's coverage
  (C) branch **for priority reasons, before any fit or steering output**. It is
  recorded as "C branch available, not executed". It is **not** a method null.
  The frozen headroom result is unchanged.
- **4B refusal: the existing record is carried forward, with no new
  compute** (§9).
- **All new 4B compute goes to three GCAD style controls:** profanity (PF),
  impoliteness and sarcasm (SARC). Each uses its retained 7B setting unchanged,
  with a **fresh Qwen3-4B bank**. There is no grid, no development selection
  and no tuning.
- **Every outcome is reported.** Outcomes are model-specific and imply no
  effect of model size.

## 2. Model and runtime

- `Qwen/Qwen3-4B`, revision `1cfa9a7208912126459214e8b04321603b3df60c`. It is
  already cached and was used by every historical 4B study. **No download.**
- Native BF16. SDPA for generation; a separately registered eager load for
  extraction only.
- The existing isolated GCAD environment (Python 3.11.18, torch 2.6.0+cu124,
  Transformers 4.52.3 with `qwen3`, Accelerate 1.7.0) and the same GPU 0
  guards.
- Bind the weights, tokenizer, `config.json` and `generation_config.json` by
  hash before any call.

## 3. Architecture port (`scripts/qwen3_gcad_port.py`)

The pinned author hook and extractors are Qwen2-only.

| Author assumption | Qwen3-4B | If unported |
| --- | --- | --- |
| `head_dim = hidden // heads` (= 80) | `config.head_dim = 128` (q 32 × 128, kv 8 × 128, hidden 2560) | The reshape raises (fails loudly) |
| Projection, then RoPE | per-head `q_norm` / `k_norm` **before** RoPE | wrong demand and K_pos (**silent**) |
| `qwen2` imports | the `qwen3` module | — |

**The port copies the three author definitions with exactly three changes:**

1. Take `head_dim` from `config.head_dim`.
2. Apply `attn.q_norm` and `attn.k_norm` to the per-head views before RoPE.
3. Import `apply_rotary_pos_emb` and `repeat_kv` from Qwen3.

**Everything else stays unchanged:**

- the gate, `c = 2b·σ(s(d − D̄))`, and the update, `a′ = a + cΔ`;
- the per-head mean and the `/√head_dim` scaling;
- response pooling, the system boundary and `inverse=False`.

The author files stay untouched. The port carries the author's Apache-2.0
NOTICE.

**CPU tests before registration** (tiny in-memory configs, no download):

- **Qwen2 regression:** bit-identical to the author functions and hook.
- **Qwen3 exactness:** softmax(q̃k̃ᵀ/√d + mask), built from the port's Q and
  K, equals the eager `output_attentions` weights.
- **Decomposition:** the system-attention contribution plus the non-system
  contribution equals the actual `o_proj` output.
- **Kwarg pickup:** the hook receives `position_embeddings` in prefill and in
  cached decode.
- **Zero coefficient:** b = 0 gives logits identical to the unhooked model.

## 4. Rendering

- Every Qwen3 prompt is rendered with
  `apply_chat_template(..., add_generation_prompt=True, enable_thinking=False)`.
- **Teacher forcing:**
  - the response is appended to that generation prompt, using the author's
    prompt/response split;
  - the system boundary is the author's first `<|im_end|>\n`;
  - check every item's token boundary, and record the count.
- **Evaluation** uses the native 4B rendering: Qwen3 has **no** default system
  message, whereas the 7B runs had Qwen2.5's default helpful system. This is
  disclosed.

## 5. Decoding (the 7B-resolved values, passed explicitly)

The 7B runs' `model_defaults.json` records repetition penalty 1.05, top-p .8
and top-k 20. Qwen3's defaults differ, so pass these values explicitly:

| Use | Settings |
| --- | --- |
| PF / SARC evaluation | greedy, repetition penalty 1.05, cap 160, batch 1, seed 42 |
| Elicitation | temperature 1, top-p .8, top-k 20, repetition penalty 1.05, cap 256 |
| Impolite first reply | temperature 1, top-p .8, top-k 20, repetition penalty 1.05, cap 500, left padding with EOS as PAD; batches per §8.3 |

Record each resolved `GenerationConfig`. Disclose that these are the 7B
values, not Qwen3's recommended defaults.

## 6. Block mapping (predeclared)

```
start = floor(9·L/28 + 0.5) = 12;  end = floor(20·L/28 + 0.5) = 26 (exclusive), L = 36
blocks 12…25 (14 blocks; relative depth .32–.71, as on 7B's 9…19 of 28)
```

- `base_coef` and the gate scale of 1.5 are carried over unchanged. They are
  **fixed settings, not matched strengths**, because Δ, K_pos and D̄ are
  fresh.
- No other window or coefficient is run.

## 7. Per-trait inputs (reused bytes, verified by hash)

| Trait | b | Extraction text | Rubrics | Evaluation |
| --- | --- | --- | --- | --- |
| PF | 5 | `evaluation/persona_reference/profanity/extract.json` (5 pairs, 20 questions); `persona_reference_common.system` | profanity rubric; GCAD coherence rubric; 11-word lexicon | `evaluation/author_profanity_revision/report.json` (32; the 7B held-out set) |
| SARC | 2 | `evaluation/sarcasm_method_transfer/extract.json` (5 pairs, 20 questions); same system rendering | sarcasm rubric; GCAD coherence rubric; impolite rubric (descriptive) | the same 32 |
| Impolite | 3.5 | the author's `trait_data_extract/impolite.json` (5 pairs, 20 questions); the author's system texts | the author's `eval_prompt`; GCAD coherence rubric | the author's first-reply task: `trait_data_eval/impolite.json`, the unchanged `sample_question_groups` (seed 42), 20 groups × 3 = 60 |

**Overlap rule: no silent removal.**

- Recompute the exact normalized overlap (NFKC, case-fold, whitespace) between
  each extraction set and its evaluation set at R1. The advisor's read-only
  check found **0/20 for PF, 0/20 for SARC and 0/20 for impolite**.
- Any nonzero exact overlap **stops preparation before registration** for root
  review. Nothing is dropped automatically.
- The known topical SARC overlap (extraction x18 against held-out H14) is
  **reported only**: a descriptive H14-excluded figure, as on 7B. It is never
  removed.

**Provenance:**

- The PF and SARC texts are project-reconstructed and partly
  Claude-generated, as disclosed on 7B.
- The 7B impolite bank was the author's released bank. The 4B bank is built
  by our recipe from the author's text, because no 4B author bank exists.
  This is disclosed.
- No 7B tensor, response, CSV or score enters any 4B bank.

## 8. Phases

### 8.1 Extraction (per trait, the same recipe)

1. **Elicit.** One fresh 4B reply per question × pair × side: **200** per
   trait, with the persona system message.
2. **Filter.** Claude grades the trait and coherence in separate blinded
   contexts. Keep a pair if positive trait ≥ 50, negative trait < 50 and
   coherence ≥ 50 on both sides. This is the frozen
   `profanity_method_transfer_common` eligibility rule, which is the author's
   mask.
3. **Eligibility.** Fewer than **10** eligible pairs means `extraction_null`
   for that trait, with no rescue.
4. **Bank.**
   - Run 3E ≤ 300 teacher-forced forwards per trait, using the eager load.
   - Save Δ [36, 2560], K_pos [36, 8, 128] and D̄ [36] in FP32, as
     **safetensors** with a hash map.

### 8.2 Mechanics (first available bank)

**The anchor bank** is the **first available bank in the fixed order PF,
impolite, SARC.** "Available" means extraction succeeded.

- **If there is no bank:** skip mechanics and all evaluation, report each
  trait `extraction_null`, and close.
- **Probes** (cap 32, the 7B mechanics prompts verified by hash):
  - 2 ordinary;
  - 2 with b = 0 on the anchor bank;
  - 2 at fixed b for **each** available bank;
  - 2 with `use_cache=False` on the anchor bank at its fixed b, as a
    diagnostic.

  That is at most **12** probes. Logit forwards: ordinary, anchor zero, and
  one per available bank, so at most **5**.
- **Requirements:**
  - exact token **and** logit parity between ordinary and zero;
  - nonzero hidden and logit changes for each available bank;
  - prefill and every cached-decode step hooked;
  - complete cleanup.

  The cache/no-cache agreement is reported, not gated.
- **On failure:** `port_or_mechanics_failure` for every trait. It is an
  implementation result, not a method null.

### 8.3 Evaluation arms and random controls

**PF and SARC** (greedy, batch 1): one shared baseline (32) is generated if
either trait has a bank. Then:

- PF: GCAD b = 5 (32) and random (32);
- SARC: GCAD b = 2 (32) and random (32).

**Impolite**: baseline, GCAD b = 3.5 and random, with 60 rows per arm.

- All three arms are generated in the **same three fixed sub-batches**: rows
  with conversation index mod 3 = 0, 1 and 2, of 20 each.
- Set torch seed 42 before each sub-batch.
- The reason: the random arm needs one realized delta per row seed, and the
  author hook takes one delta per layer per batch. Using the same sub-batches
  for every arm keeps batch composition identical across arms.
- This replaces the 7B batch of 64, which is an execution detail. It is
  disclosed.

There are no prompted or residual arms.

**Random controls: the exact frozen 7B recipe**
(`profanity_method_transfer.random_controls`, reused by the sarcasm
low-strength study):

- For evaluation row index i, in fixture order: question seed =
  `random_seeds[i % 3]`.
- For each hooked block ℓ:
  - `torch.Generator(device='cpu').manual_seed(seed + 1000·ℓ)`;
  - draw `randn(hidden=2560, float32)`, normalize, and scale by
    `delta[ℓ].norm()` of that trait's FP32 4B Δ;
  - unhooked blocks are zero;
  - K_pos, D̄, gate, scale, b and blocks are as in the trait's GCAD arm.
- **Seeds:** PF **[9801, 9802, 9803]**, SARC **[9811, 9812, 9813]**, impolite
  **[9821, 9822, 9823]**. For impolite, row i is the conversation index, so
  sub-batch k uses seed index k.
- Save controls as **safetensors** (a serialization change from 7B),
  with per-row and per-layer hashes, norms and seeds, using the frozen
  `controls_metadata` format.
- **Register them at R2, before any evaluation generation.** No additional
  seeds or grid.

### 8.4 Gates (frozen helpers and field names)

**PF** (n = 32): `profanity_method_transfer_common.summarize` and `gates(base,
value, random)`. Pass is `all_pass` over:

- `trait_gain`: `trait_conservative` ≥ baseline + 20;
- `trait_random_gain`: ≥ random `trait_conservative` + 20;
- `lexicon_absolute`: `lexicon_hits` ≥ 8;
- `lexicon_gain`: ≥ baseline `lexicon_hits` + 8;
- `coherence_absolute`: `coherence` ≥ 50;
- `coherence_relative`: ≥ baseline `coherence` − 20;
- `invalid`: `invalid` ≤ 3.

**SARC** (n = 32): `sarcasm_method_transfer_common.summarize` and `gates`.
Pass is `all_pass` over:

- `trait_gain`;
- `individual_ge50`: `individual_trait_ge50` ≥ 8;
- `coherence_absolute` and `coherence_relative`;
- `invalid` ≤ 3;
- `trait_random_gain`.

Impoliteness and H14-excluded figures are descriptive.

**Impolite** (n = 60). Two results are reported **separately**:

1. **Author effect (the 7B rule, unchanged):** `gcad_trait_gain30`. This is
   the frozen `author_gcad_impolite_common.metrics` contrast, an exact
   cross-multiplied numeric-mean difference ≥ 30 against baseline, which
   fails if either numeric count is 0. The residual contrasts are not
   applicable (no residual arm).
2. **4B controller qualification (new; predeclared here, before any output):**
   - `trait_conservative_mean` ≥ random + 20;
   - `coherence_mean` ≥ 50 **and** ≥ baseline `coherence_mean` − 20;
   - effective invalid ≤ **6/60** (§8.5).

**Labels:**

- The trait is a "working 4B controller", and its bank may be published, only
  if **both** results pass.
- `author_effect_only` means result 1 passes and result 2 fails.
- Result 2 is **not** retroactive to 7B, which had no random arm and no such
  guard, and it is not tuning.
- The frozen 7B fields `degenerate` (4-gram excess > .20), `empty`, `caps`
  and `eos` are also reported, descriptively.

**Status values:** `passed`, `author_effect_only` (impolite only),
`heldout_failure` (listing the failed gate keys and their margins),
`extraction_null`, `port_or_mechanics_failure` or `not_run`.

### 8.5 Invalid output: the exact definition

- The definition comes from the frozen **7B style helpers**:
  `author_profanity_method_common.text_checks` / `repetition`.
  - **Whitespace-word bigrams** on the decoded answer text: `text.split()`,
    with distinct pairs over all pairs, and 1.0 if fewer than 3 words.
  - `invalid = empty or distinct_bigram < .6`, where `empty = not
    text.strip()`.
- This is **not** the political study's token-ID bigram metric, which is
  not used here.
- **One added 4B guard, disclosed:** a decoded answer containing `<think>` or
  `</think>` is also invalid. Effective invalid is the **union** of frozen
  invalid rows and think-tag rows, counting each row once. Both component
  counts are reported separately.
- For PF and SARC, the frozen `summarize` field `invalid` is used, plus the
  think-tag count. That changes the ≤ 3 gate only if a think tag ever appears.

## 9. Refusal row (carried forward; no new compute)

From `docs/refusal_reduction_execution.md` and `docs/qwen3_4b_closeout.md`,
unchanged:

| Method | Measured | Status |
| --- | --- | --- |
| Raw subtraction, block 23 output, h′ = h − 0.5v, all positions (selected) | Final, source-unsafe 30: refusals 29 → 22, valid full answers 1 → 7. Benign usefulness 29/42 → 23/42 | Partial effect; **quality rule failed**. No controller is retained |
| All-block projection, h′ = h − u(uᵀh) | Development: valid 6/30 (baseline 2), refusals 24/30, benign useful 13/30 | Missed the valid threshold; not selected |

- No new unsafe steering, no new benign pool and no boundary-removal claim.
- No 4B refusal vector is packaged, so no reload check is needed.

## 10. Budget (maxima, all branches)

| Item | Count |
| --- | --- |
| Elicitation generations | 600 |
| Mechanics generations | 12 |
| Evaluation generations | 160 greedy (32 shared baseline + 4 × 32) + 180 impolite = 340 |
| **Total generations** | **952** |
| Teacher-forced forwards | ≤ 900 |
| Logit forwards | ≤ 5 |

**First-valid native batches (at most 8 items, blinded, frozen before
output):**

| Plan | Items | Batches |
| --- | --- | --- |
| Elicitation trait + coherence, per trait | 200 + 200 | 25 + 25, all full; ×3 traits = **150** |
| PF trait | 96 | 12 |
| SARC trait | 96 | 12 |
| SARC impoliteness (descriptive) | 96 | 12 |
| **Shared** greedy coherence (baseline + 4 arms) | 160 | 20 |
| Impolite trait | 180 | 22 full + **1 batch of 4** = 23 |
| Impolite coherence | 180 | 22 full + **1 batch of 4** = 23 |
| **Total** | | **252** (≤ 504 starts at 2 structural attempts each) |

**Batch rules:**

- Every batch is fully packed (8 items) except the final 4-item batch of
  each impolite plan.
- Different arms of a question are never in the same batch.
- Elicitation pairs (positive and negative) are kept apart.
- **Branch plans are frozen at R1** for every pattern of extraction nulls.
  If only one of PF and SARC has a bank, the shared coherence plan becomes
  that trait's 96 items in 12 batches. A missing trait removes its own plans.

**Judge:** the same pinned native Claude runtime and tools-off isolation, with
the identity recorded and any change from 7B disclosed. **No new calibration
stage.** Structural retries only. Authentication, identity, hash or
uncertain-start errors are fatal. No GPU regeneration to repair grading.

## 11. Implementation and stages

- **One config** (`configs/qwen3_4b_fixed_transfer.json`), **one runner**
  (`scripts/qwen3_4b_fixed_transfer.py`) and the **port**.
- **The runner imports the frozen helpers unchanged:**
  - `profanity_method_transfer_common`: `summarize`, `gates`, eligibility,
    `observe_shift`, `hook_cleanup` and `position_weighted_norm`;
  - `profanity_method_transfer`: `random_controls` logic and
    `controls_metadata`;
  - `sarcasm_method_transfer_common`: `summarize` and `gates`;
  - `author_gcad_impolite(_common)`: `sample_question_groups`, the proxy and
    `metrics`;
  - `author_profanity_method_common.text_checks`;
  - `batched_generation`, `tone_judge` and the plan and journal helpers.

  A test asserts identity with the frozen objects. The random-control logic
  may need a thin parameterization: for hidden size, blocks and seeds. If so,
  a test proves it gives the same output as the frozen function on a 7B-shaped
  input.
- Run root: `outputs/qwen3-4b-fixed-transfer-01`. Aggregate:
  `results/qwen3_4b_fixed_transfer_01.json`.

**Registrations:**

1. **R1, before any 4B call:** hashes, rendered prompts and token IDs, port
   tests, overlap counts, the block formula, decoding, all branch grading
   plans, and seeds.
2. **R2, before evaluation:** eligibility counts, safetensors bank hashes,
   realized random controls, and mechanics passing.
3. **Result and closeout.** No selection registration.

## 12. Reporting

- **One table:** the PF, impolite, SARC and refusal rows, with the 7B status
  beside the 4B status and each study's gate keys, margins, n and setting.
- **For each trait, state what was carried over and what was fresh:**
  - carried over: b, scale, the window by formula, decoding, rubrics and the
    task;
  - fresh: Δ, K_pos, D̄, elicitation and filtering.
- **Report impolite's author effect and its 4B qualification separately.**
- Required sentence: "Outcomes are model-specific. Differences between
  checkpoints do not establish an effect of model size, and a 4B null does not
  show that the method fails on small models."
- **Nulls stay in the release.** A 4B bank is published only for a passing
  trait, in a separate 4B folder.
- **The earlier 4B PV profanity and rudeness nulls are unchanged** and are not
  retested.
- **Limits:**
  - one Claude judge family;
  - Claude-generated PF/SARC inputs;
  - a different impolite bank provenance than on 7B;
  - the system-message difference;
  - decoding pinned to the 7B values;
  - impolite sub-batching;
  - truncation at the cap;
  - n = 32/32/60.
