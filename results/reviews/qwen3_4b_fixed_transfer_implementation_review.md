# Qwen3-4B fixed transfer: implementation review (runner, config, amended port)

Reviewer: Claude advisor, read-only. I made no model, GPU or judge calls and
ran no tests.

**Read in full:**

- `scripts/qwen3_4b_fixed_transfer.py` (720 lines);
- `configs/qwen3_4b_fixed_transfer.json`;
- the amended `scripts/qwen3_gcad_port.py`.

**Checked against:**

- the accepted plan (`docs/qwen3_4b_fixed_transfer_plan.md`);
- my port review (D1–D3);
- the frozen helpers they import: `profanity_method_transfer(_common)`,
  `sarcasm_method_transfer_common`, `author_gcad_impolite(_common)`,
  `persona_reference_common.make_plan`, `author_profanity_method_common`
  and `tone_judge`;
- the author's `generate_batched_multiturn`.

## Verdict

**Not yet accepted.** There are three blockers beyond the constructor fix root
already sent (B0). Each is small, but any of them would end or corrupt the
study's single run. The verdict stays conditional until the fixes are in and
the worker's test receipts show them covered.

## Blockers

**B0 (known; fix already sent). Positional `blocks`.**
`qwen3_4b_fixed_transfer.py:216–217` passes `blocks` positionally, but
`layer_indices` is keyword-only, so this raises `TypeError`. Pass
`layer_indices=blocks`. Add a test that builds a steerer through
`controller()`, not directly.

**B1. The report crashes after all compute is done.**
`effective_metrics` (`:621`) calls `p.position_weighted_norm(rows)` for
every arm.

- **Baseline arms** have `shift_audit == {}`, so `n = 0` and it raises
  "missing measured development positions".
- **Steered arms** now carry `shift_audit['steerer_calls']`, which has no
  `positions`, so it raises `KeyError`.
- `main` then writes a terminal `fatal_pipeline_error` stop, after every
  generation and grade has been spent.

**Fix:**

- Compute the norm only for steered arms, as the frozen 7B report did
  (`sarcasm_gcad_low_strength.py:206`).
- Pass only the `observe_shift` site records, excluding `steerer_calls`
  and `hooks_removed`.
- Add an end-to-end report test on a synthetic fixture covering all three
  traits and every branch.

**B2. A tokenization boundary mismatch is fatal, contrary to the accepted
plan.**

- **The problem:** `boundary()` (`:277`) raises when `prefix_equal` is false.
  The plan (§4, lines 79–80) says to check every item's token boundary and
  **record the count**.
  - The Qwen3 prompt ends in `</think>\n\n`, so an eligible answer that
    begins with whitespace or a newline can legitimately merge across the
    boundary.
  - As written, one such item kills extraction **and**, through `main`,
    stops the whole study.
- **The fix:** record `prefix_equal` per item and the mismatch count in
  `encodings.json`, replacing the hard-coded `'prefix_mismatches':0` at
  `:420`, and keep the author's `prompt_len` pooling unchanged.
- **These stay fatal:**
  - `saved_prompt_ids_equal`;
  - `system_prefix_equal`;
  - the `<|im_start|>system\n` start;
  - a non-empty response.

  These are genuine pipeline errors. R2 already binds `encodings.json`, so
  the count is registered before evaluation.

**B3. Operator or precondition errors become terminal stops** (`main`,
`:712–716`). Any exception after `Context()`, including `verify`, `report`,
an unknown `--stream` or a stage run out of order, writes `stopped.json`.
Concrete cases:

- **Mechanics before every available bank exists:** `available()` calls
  `bank()`, which calls `stage()`, which raises (`:206`, `:587`).
- **Extraction or eligibility before the elicitation streams are
  accepted:** `accepted()` raises.
- **`judge` with a stream outside the active branch:**
  `active_plans` raises `KeyError`. The `native/<stream>` directory is
  created at `:386` before the payloads are validated.
- **`report` before every evaluation stream is accepted:**
  `decoded_grades` raises.

**Required:**

- Check stage preconditions up front and raise `Wait`, with no write, no
  directory and no call. This covers the stream name being in the frozen set
  for the current branch, the required accepted streams, banks and
  registrations.
- `report` returns `incomplete` instead of raising when streams are pending,
  and **never writes `stopped.json`**.
- Keep terminal stops for integrity, hash, budget, isolation and
  uncertain-start failures, and for real errors during calls.

## Required (fidelity and registration; no new science)

**R1. Descriptive SARC impoliteness grading order.** The frozen 7B SARC rule
required held-out grading in the order trait, then coherence, then
impoliteness, and published `primary_complete_descriptive_incomplete` when the
only failure was in the descriptive stream. Here there is no ordering and no
such status: a fatal error in `sarcasm_impolite` makes the whole study
incomplete.

- Gate `judge(s,'sarcasm_impolite')` on `sarcasm_trait` and
  `greedy_coherence` both being accepted.
- Add that terminal status, which keeps every primary decision.

**R2. Assert the plan shapes at `prepare`.** For the all-traits branch:

- the elicitation plans are 6 × 25 batches, all full;
- the evaluation plans are 12 + 12 + 12 + 20 + 23 + 23 batches;
- the total is **252**;
- every batch has 8 items, except that the last batch of each impolite plan
  has 4;
- no batch contains two arms of the same group.

For each of the other 7 branches, assert its count. I checked the logic by
reading `make_plan` and the stream builder (`:93–111`). The shapes and the
seeds (9900–10050, no collisions) are right, but the assertion belongs in the
code.

**R3. Random-control equivalence test.** `random_delta` (`:439`) reimplements
the frozen `profanity_method_transfer.random_controls` arithmetic rather than
importing it. Accepted plan §11 allows this only with a test showing the same
output as the frozen function on a 7B-shaped input:

- hidden size 3584;
- blocks 9–19;
- seeds 8603–8605;
- tensors exactly equal under `torch.equal`.

**R4. Bind this review at R1.** Add `results/reviews/qwen3_4b_fixed_transfer_implementation_review.md`
to the reviewed documents that `registered()` checks (`:157`) and to the
manifest's external hashes, as done for the design and port reviews.

**R5. Use `cfg['traits']['impolite']['base_coef']`** in place of the literal
`3.5` at `:527`.

## Verified (no change needed)

**Configuration and binding.**

- `validate_config` pins: model and revision, blocks `[12…25]` (also
  recomputed by formula at `prepare`), scale, `inverse`, seeds, every budget,
  b = 5 / 3.5 / 2, and the random seed triples PF 9801–9803, impolite
  9821–9823 and SARC 9811–9813.
- The architecture tuple is `qwen3` / 36 / 2560 / 128.
- Weights, sources, author files and the judge binary are hashed. Plans are
  re-derived on every `verify`.
- R1 is the exact manifest hash. R2 binds eligibility, banks, encodings,
  mechanics and the controls, with `no_selection: true`. Evaluation is
  locked behind R2.

**Task, arm and ID binding.**

- **Elicitation:**
  - 5 pairs × 20 questions × 2 sides, with exact IDs and order;
  - the overlap check raises rather than dropping;
  - `pv.system` gives the 7B rendering ("You are a/an {adjective}
    assistant. {instruction}").
- **Evaluation:**
  - the same 32 IDs, in order, for every greedy arm;
  - 60 author conversations (`g..-s.`), with source IDs and prompt bytes
    checked;
  - mechanics uses C01–C02, and evaluation IDs don't collide.

**Decoding and thinking mode.**

- `render` always passes `enable_thinking=False`, and `prepare` asserts the
  `<think>\n\n</think>\n\n` ending on every prompt.
- The impolite author call goes through `NonthinkingTokenizer`, which
  forces the same flag, and actual IDs are checked against the frozen ones.
- `decoding()` pins repetition penalty 1.05, top-p .8, top-k 20,
  temperature 1 and the caps 256, 160, 500 and 32. The resolved configs are
  saved.
- The author call passes `do_sample=True` and temperature 1. The other
  processors come from the pinned `generation_config`.
- Left padding with EOS as PAD.

**Extraction and eligibility.**

- **Eligibility:** `p.eligible_pairs` is the frozen mask (positive ≥ 50,
  negative < 50, coherence ≥ 50 on both, refusal counted as None), with at
  least 10 pairs.
- **Load and budget:** an eager load. The teacher-forced proxy charges
  every forward against the global 900 (3E per trait).
- **Δ:** pos mean(0) minus neg mean(0), cast with `.float()` after the mean,
  over 36 layers. This is identical to the 7B code
  (`profanity_method_transfer.py:318`).
- **K and D̄:** come from the positive side only.
- **Bank file:** FP32 safetensors with per-tensor hashes and provenance.

**Port (D1–D3).**

- `validate_geometry` is called in all three entry points: model type,
  projection shapes, no `o_proj` bias, no RoPE scaling or sliding window,
  `q_norm`/`k_norm` required on Qwen3 and absent on Qwen2.
- Applied and skipped counters by phase. The controller fails on any skip.
  `validate_hook_audit` requires exactly one applied call per generated
  position.
- Nesting is refused, and cleanup is checked through `hook_signature`.
- The arithmetic diff from the author code is still only the three accepted
  changes.

**Mechanics.**

- The anchor is the first available bank in the order PF, impolite, SARC.
  With no bank, there is a no-call `Wait`.
- Probes: 2 × (ordinary + zero + each available bank + no-cache), at most
  12. Logit forwards: ordinary, zero and one per bank, at most 5.
- Gates: token and logit identity at zero; nonzero logits and hidden
  changes per bank; prefill ≥ 1 and decode = n − 1 applied calls.
- Cache/no-cache agreement is diagnostic only. A failure stops the study as
  `port_or_mechanics_failure`.

**Impolite batches.**

- Three sub-batches of 20 by index mod 3, with identical composition across
  arms and `torch.manual_seed(42)` before each.
- One realized control per sub-batch, asserted equal across its rows.
- The hook-call count equals the padded generation width.
- The author's `with steerer_ctx` enters once per turn, and there is one
  turn, so the single-use context manager is valid.

**Gates and invalid outputs.**

- **PF and SARC:** the frozen `summarize` and `gates`, with `invalid`
  replaced by the union (empty, or whitespace distinct-bigram < .6, or a
  think tag) counted once. `frozen_invalid` and `think_tag_count` are
  reported separately.
- **Impolite:**
  - the author effect is the frozen `gcad_trait_gain30`;
  - the qualification is `trait_conservative_mean` ≥ random + 20,
    `coherence_mean` ≥ 50 and ≥ baseline − 20, and the union invalid count
    ≤ 6;
  - the labels are `passed`, `author_effect_only` and `heldout_failure`, as
    accepted.
  - Routing the random arm through `native.metrics`'s `residual` slot is
    acceptable: only `gcad_trait_gain30` and the conditions are used. Keep
    the residual contrasts out of the published result, as the code does
    now.

**Native grading.**

- **Envelope:** the frozen `native.ratings` schema, parsed through
  `prior.parse`.
- **Attempts:** first-valid, at most 2 per batch, structural failures only.
  Stray or uncertain attempts are fatal.
- **Budget:** the global 504-start check runs before each invocation and
  again on read. The pinned binary and version are checked.

**Refusal row.** It is carried forward with zero new compute, and its
figures match the plan's §9.

## Notes (no action required)

- Greedy calls also pass sampling arguments while `do_sample=False`.
  Transformers ignores them, apart from warnings, and the resolved config is
  saved.
- The author sets `padding_side` on the `NonthinkingTokenizer` wrapper, not
  on the inner tokenizer. That is harmless only because `load_model`
  already set the inner tokenizer to left padding. Optionally assert the
  inner value after the call.
- Impolite conversations repeat question text across samples, so the same
  text can appear twice in a grading batch. As on 7B, disclose it.
- The judge runtime record comes from `configs/qwen3_14b_baseline.json`,
  with the same pinned binary. State this in the report.

## Acceptance condition

**Accepted for R1** once all of the following are in place, and the worker's
test receipts show a test for each:

- B0–B3;
- R1–R5;
- the plan's port CPU tests.

No scientific choice changes.

## Addendum: amendments checked (runner of about 850 lines, plus tests)

I reviewed only the amendments made against the findings above, and the new
tests. I made no calls and ran no tests. Root reports 15 port tests passing
in the exact GCAD environment. The worker reports 17 runner tests passing
and 2 journal tests still being fixed.

### Findings resolved

- **B0.** `controller()` now passes `layer_indices=blocks` by keyword
  (`:298`).
  `test_actual_runner_controller_keyword_geometry_zero_hooks_and_cleanup`
  builds the steerer through the runner.
- **B1.** `measured_norm` (`:702`) gives a baseline arm 0 and requires its
  audit to be empty. Steered arms validate the hook audit and drop
  `steerer_calls` before `position_weighted_norm`. Impolite arms use the
  same function. Covered by:
  - `test_invalid_union_once_and_baseline_zero_norm_active_requires_sites`;
  - the report fixtures for every branch (`test_actual_report_*` and
    `test_report_all_other_branches_*`).
- **B2.** `boundary()` (`:351–356`) now records `prefix_equal` without
  raising. `encodings.json` stores the actual `prefix_mismatches` count
  (`:506`). The author's `prompt_len` pooling is unchanged.
  - **Still fatal:** a saved prompt-ID mismatch, a system-prefix mismatch, a
    missing system start, or an empty response.
  - **Test:** `test_boundary_merge_is_recorded_not_repaired_but_pipeline_checks_fatal`.
  - This matches plan §4 and root's ruling.
- **B3.** Missing preconditions now raise `Wait`.
  - **`prerequisites()` / `judge_prerequisites()`** check the stage files,
    the accepted streams, the banks, R2 and a frozen-stream allowlist. A
    stream outside the active branch also waits.
  - **No directory is created early:** `run_journal` validates the payloads
    before `mkdir`.
  - **Uncommitted registrations wait:** `committed_or_wait` is used in
    `registered()` and `evaluation_lock`.
  - **`report`** returns `incomplete` with `primary_pending` and never
    writes a stop (`main`, `:832`).
  - **Integrity, budget, isolation and uncertain-start failures** stay
    fatal.
  - **Tests:** `test_missing_preconditions_unknown_stream_and_dry_flags_never_write`
    and `test_report_pending_primary_no_stop_*`.
- **R1.** `sarcasm_impolite` waits until `sarcasm_trait` and
  `greedy_coherence` are accepted.
  - A fatal error inside that stream is marked `descriptive_only`, and the
    report publishes `primary_complete_descriptive_incomplete` with the
    primary results kept (`:749–800`).
  - **Test:** `test_descriptive_only_fatal_keeps_primary_results_and_public_status`.
- **R2.** `validate_plans` (`:115`) runs at `prepare` and on every `verify`.
  It asserts:
  - fully packed shapes, with only the final partial batch;
  - unique blinded IDs;
  - one item per group in each batch;
  - the stream set and size for all 8 branches;
  - 252 in total.
- **R3.** `test_random_controls_exact_frozen_7b_arithmetic_*` compares
  `random_delta` with the frozen `prior.random_controls` on a 7B-shaped
  delta (28 × 3584, blocks 9–19, seeds 8603–8605):
  - the tensors are exactly equal;
  - the global RNG is untouched;
  - the safetensors round-trip is exact;
  - an existing controls file is never overwritten.
- **R4.** `implementation_review` is in the config, the manifest's external
  hashes and `registered()`.
- **R5.** The impolite coefficient is read from the config (`:614`).

### One remaining requirement (small)

**G1. The descriptive stream must be last among all primary evaluation
streams.**

- **The problem:** `sarcasm_impolite` currently waits only for the SARC
  primaries. If impolite is also active and its `impolite_trait` or
  `impolite_coherence` stream isn't yet accepted, a fatal error in
  `sarcasm_impolite` writes the terminal stop. That blocks the remaining
  **impolite primary** grading, and the report could then never reach
  `primary_complete_descriptive_incomplete`.
- **The fix:** in `judge_prerequisites`, require **every** primary
  evaluation stream in the active branch to be accepted before
  `sarcasm_impolite`: `*_trait`, `greedy_coherence` and
  `impolite_coherence`. This restores the 7B rule that the descriptive
  journal runs only after the primary decision is complete.
- **Test:** with impolite pending, `sarcasm_impolite` must raise `Wait`.

### Verdict

**Accepted for CPU prepare and the source-hash freeze** once G1 and its
test are in, and the worker's receipt shows the **full** runner test file
passing, including the two journal tests that are failing now.

- Root's 15-test port receipt already covers the port tests the plan
  requires.
- No scientific choice changed. B2 follows the accepted plan, as root
  ruled.
