# Qwen3-4B fixed-setting transfer: design review (DRAFT)

Reviewer: Claude advisor, read-only. I made no model, judge, GPU or download
calls and changed no code.

**What I checked:**

- the cached Qwen3-4B `config.json` and `generation_config.json`, at revision
  `1cfa9a72…`;
- the pinned author `gated_steerer.py` and `generate_vec.py`, plus the
  author's key script;
- the project's extraction loader in `profanity_method_transfer_common.py`;
- the retained PF and SARC profiles;
- the profanity-transfer, sarcasm low-strength and author-impolite protocols;
- the resolved 7B decoding in the runs' `model_defaults.json`;
- the closed 4B refusal record.

This review accompanies `docs/qwen3_4b_fixed_transfer_plan.md`.

## Verdict

**Acceptable as the minimal scope, subject to the port tests (plan §3)
passing before registration.** The user's minimal reading is scientifically
sound and is the recommended scope:

- **Refusal: carry forward the tested 4B conventional record. No rerun.** The
  record already holds what the user asked for: fixed subtraction, projection
  removal, an honest partial effect and a failed quality rule. A new benign
  4B refusal pool would be new research. It would also probably repeat the 7B
  outcome of no headroom: on XSTest-safe, the 4B model had only 17/250 refusals
  even before any steering. No new benign check is needed, because no 4B
  refusal vector is shipped as a controller.
- **The 7B C branch** is closed by a priority decision taken before any
  fit or steering output. That is neutral with respect to GCAD and is not a
  null.
- **New compute goes only to the three fixed GCAD style controls.** No grid,
  no selection, held-out gates only.

## Findings that shaped the plan

1. **The author code cannot run Qwen3 as written (confirmed by reading it).**
   - **A loud failure:** both files compute `head_dim = hidden_size //
     num_heads` (= 80). Qwen3-4B has `head_dim` 128, so the q/k/v views
     raise.
   - **A silent failure:** the hook captures the q_proj output and the key
     extractor hooks k_proj, then each applies RoPE directly. Qwen3
     normalizes Q and K per head (`q_norm`/`k_norm`) before RoPE. A naive
     fix of the `head_dim` value alone would run and produce a **wrong
     gate**, which is the dangerous case.
   - **The plan's response:** port exactly three changes, prove the Qwen2
     path still gives bit-identical results, and prove on Qwen3 that the
     port's normed post-RoPE Q·K reproduces the model's own eager attention
     weights. This is the decisive implementation test, and it costs no
     download or GPU.
2. **"Same decoding" needs explicit values.**
   - The 7B runs inherited Qwen2.5's `generation_config`: repetition penalty
     1.05, top-p .8 and top-k 20. The penalty also applied under greedy
     decoding.
   - Qwen3-4B's defaults differ (top-p .95, temperature .6, no penalty).
   - Keeping "checkpoint defaults" would silently change decoding on both
     the greedy and the sampled tasks. The plan pins the 7B-resolved values
     and discloses this.
3. **Thinking mode.** Without `enable_thinking=False`, the Qwen3 template
   allows reasoning blocks, which would contaminate both scoring and the
   response-pooled bank. The plan uses one rendering everywhere and builds
   the teacher-forced inputs from the same generation prompt. Any think tag
   in an output counts as invalid.
4. **System message.** Qwen3 has no default system message; Qwen2.5 does.
   Native rendering per checkpoint is the fair choice, and it is disclosed.
   GCAD's gate is computed from current queries against K_pos and does not
   need system tokens at evaluation.
5. **Block mapping.** The proportional rounding formula gives blocks 12–25:
   14 blocks against 7B's 11. It preserves relative depth, at the cost of more
   hooked blocks. Because Δ norms, K_pos and D̄ are re-estimated on 4B, the
   carried-over b values are fixed settings, not matched strengths. That is
   the defined meaning of this control, and the plan says so. An
   equal-block-count window would be an equally arbitrary alternative. I
   recommend the predeclared formula and running no second window.
6. **The impolite bank provenance changes.** The 7B result used the author's
   released bank. No 4B author bank exists, so the 4B bank must be built from
   the author's own extraction text with our recipe and judge. That is
   faithful to the author's procedure, which elicits per model, but it is not
   a like-for-like copy. Reusing the author's cached 7B CSVs as
   teacher-forced text would be cheaper but off-policy, and a null from it
   would be uninterpretable. Rejected.
7. **The impolite gate needs a specificity check.** The 7B impolite record
   had no random arm. Keeping the 7B contrast primary and adding a
   pre-stated random and coherence requirement before a 4B bank is
   published is honest, and makes clear that the two outcomes are not
   comparable on that condition.

## Budget judgement

- **The figures:** 952 generations at most, 600 of them cheap elicitation
  at cap 256. Up to 900 teacher-forced forwards. 252 first-valid batches.
- **Elicitation grading (150 batches) is the dominant cost.** It is kept
  because on-policy filtering is the established recipe on 7B, and the only
  defensible basis for a 4B bank.
- **The denominators match 7B's held-out designs** (32, 32 and 60). Smaller
  sets would make the +20/+30 gates and the ≥ 8/32 counts meaningless.
- **Savings already taken:**
  - no development stage;
  - one shared greedy baseline;
  - one shared coherence plan;
  - no prompted or residual arms;
  - per-trait early exit on an extraction null.

## Required before registration (for the worker; no science changes)

1. The port file and the five CPU tests in plan §3. The Qwen2 regression must
   be bit-identical, and the Qwen3 attention-weight equality must hold.
2. Rendering tests on the cached tokenizer:
   - the generation prompt ends with an empty think block;
   - the teacher-forced token boundary holds for every item;
   - the system boundary is found.
3. Explicit `GenerationConfig` objects for all three decoding modes. Tests
   that the resolved values equal the 7B record.
4. The block formula is computed from `config.num_hidden_layers` and
   asserted equal to [12, 26).
5. Safetensors bank I/O with a hash map. No `torch.save` of new tensors.
6. The single runner imports the frozen helpers; a test asserts their
   identity, as in the recovery review. No frozen file changes.

## Not recommended

- Any 4B refusal rerun, new benign refusal pool, or political-omission work.
- Any second window, coefficient grid, or rescue elicitation.
- Publishing a 4B bank whose study did not pass.
- Any wording that links 7B/4B differences to model size.

No user question is needed. All remaining choices are routine and are fixed
above. Root registers, commits and releases each stage.

## Addendum: root's five points, resolved in plan v2

I read the frozen helpers to settle each point (no calls).

1. **Mechanics anchor.**
   - The anchor is the first **available** bank in the fixed order PF,
     impolite, SARC. It is used for the zero-coefficient and no-cache probes.
   - Each available bank gets its own nonzero probes.
   - With no bank, mechanics and evaluation are skipped and every trait
     reports `extraction_null`.
   - The ceilings are unchanged: 12 probes and 5 logit forwards.
2. **Invalid output: the exact definition.**
   - The frozen 7B style helper is `author_profanity_method_common.repetition`.
     It uses **whitespace-word** bigrams on the decoded text, with 1.0 if
     fewer than 3 words.
   - `invalid = empty or distinct_bigram < .6`, as in
     `profanity_method_transfer_common.summarize`.
   - The political study's token-ID metric is **not** used.
   - The think-tag rule is a separately counted, disclosed extra 4B guard.
3. **Impolite qualification.**
   - **The author effect is unchanged:** the frozen `gcad_trait_gain30`
     contrast.
   - **A separate predeclared 4B qualification:**
     - `trait_conservative_mean` ≥ random + 20;
     - `coherence_mean` ≥ 50 and ≥ baseline − 20;
     - effective invalid ≤ 6/60.
   - **Labels:** "Working controller" and bank publication require both.
     `author_effect_only` is reported when only the author effect passes.
   - **Not retroactive to 7B, and not tuning.** Root's proposed guards are
     adopted as written. Without them, a heavily degenerate output could be
     labelled working.
4. **Budget verified: 252 first-valid batches.**
   - Elicitation: 150 batches, all full.
   - PF trait: 12.
   - SARC trait: 12.
   - SARC descriptive impoliteness: 12.
   - **Shared coherence plan:** 160 items in 20 batches.
   - **Impolite:** 23 + 23 batches. Each plan is 22 full batches plus one
     final batch of 4; every other batch is full.
   - **Branch plans** for every pattern of extraction nulls are frozen at R1.
   - **Overlap:**
     - my read-only exact normalized check found 0/20 overlaps for each of
       PF, SARC and impolite;
     - any nonzero overlap stops preparation, with no silent drop;
     - H14 is reported only.
5. **Random controls: the exact frozen recipe.**
   - Source: `profanity_method_transfer.random_controls`, which the sarcasm
     low-strength study reused.
   - Seeds by row mod 3. Layer seed = seed + 1000·ℓ.
   - A dedicated CPU generator. FP32 `randn(hidden)`, normalized, scaled by
     `delta[ℓ].norm()`. Zeros elsewhere.
   - `controls_metadata` hashes everything.
   - **New seed triples:** 9801–3, 9811–3 and 9821–3. No other grid.
   - **Impolite sub-batches:** the author hook holds one delta per batch, so
     all three impolite arms are generated in the same three seed-group
     sub-batches of 20, with torch seed 42 before each. This keeps batch
     composition equal across arms and is disclosed.

**Verdict on v2:** ready for root to accept. After acceptance, implement
exactly one runner plus the port. No user question is needed.
