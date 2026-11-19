# Qwen3.8-27B sarcasm GCAD: review 06 (independent C6 analyzer)

Advisor review, 2026-11-18, of these files:
- `scripts/qwen38_sarcasm_analyze.py`, sha256
  `26dc69b19ea138070c1d9880aa2b2ad6464052389348ccc87f681527b861f461`
  (360 lines, modified 17:57:02);
- `tests/test_qwen38_sarcasm_analyze.py`, sha256
  `6a25e4e285832cc715d49e358fc1beb3abc8ddca8f9754785f4aba6c150145b2`
  (modified 17:57:31).

The review was read-only, plus CPU tests. No model, judge or GPU call was
made, and no study outputs, scores or generated text were inspected. Only the
run's `config.json` keys and `manifest.json` structure were read. The frozen
plan, the reviews and the R0 source are unchanged.

`OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= outputs/qwen38-27b-nf4-01/env/bin/python -m pytest -m '' -q tests/test_qwen38_sarcasm_analyze.py`
gives **16 passed**.

## Verdict

**Accepted for its own C6 registration at the hashes above.** It closes the
three C6 items from review 03. Nothing blocks registration. O1 and O2 below
are optional; if adopted, they must land before registration, and I would
re-review only the changed lines.

## Checks against the signed runner, judge and frozen helpers

| Item | Analyzer behaviour | Status |
| --- | --- | --- |
| Exact gates | Calls the unchanged `sarcasm_method_transfer_common.summarize`/`gates(baseline, selected, random)` on the full 32 rows per arm. This is the same formula as `final_decision` for one method: trait +20, ≥ 8 answers with SARC ≥ 50, coherence ≥ 50 and ≥ baseline − 20, invalid ≤ 3, and SARC ≥ random + 20. Before importing, it re-hashes every R0 `scripts/*.py` file in `manifest['files']`, including both helpers. It also records the helper hashes. | pass |
| No reselection | Requires `selection.json` with `selected.gcad` ∈ {gcad_1, gcad_2}. It never recomputes the selection. `prompted_validity_concern` uses the same expression as `choose`. | pass |
| Row provenance | Exactly 128 IDs, equal to the frozen `inputs.json['final']`. `question_id`, `arm` and `user` must equal the frozen values. Each arm must have exactly H01–H32, which matches the final fixture's IDs. Integer `generated_ids` and boolean cap and EOS flags are required. | pass |
| H14 subset | n = 31 per arm. Labelled "descriptive only; full32 gates remain authoritative". A test shows that the exclusion cannot rescue a failed full-32 gate. | pass (see O2) |
| Impoliteness | Arm means over 32. Within-arm Pearson for all four arms, which includes the selected arm. A pooled Pearson carries a caveat that it is confounded by arm. Trait refusals are excluded from the correlations. Constant inputs give `pearson: null` with a reason, not NaN. | pass |
| Primary-complete path | The signed judge writes each `{metric}.json` only after the whole stream completes. It writes `primary-scores.json` right after coherence on final, and `scores.json` only after impoliteness. The analyzer accepts trait and coherence from the per-metric or primary files, cross-checks them for agreement, and rejects partially populated streams. With no complete impoliteness stream it reports `primary_complete_descriptive_incomplete`, with null means and correlations and no partial substitution. A missing primary stream gives `incomplete` and no gate. | pass |
| Null branches | **Extraction null** requires `eligibility.extraction_null`, fewer than `minimum_pairs` pairs, and no selection or final; otherwise it raises. **Development null** requires `selection_null`, an empty selection and no final. If no final exists and no explicit null is recorded, the outcome is `incomplete`, or `port_or_mechanics_failure` when a smoke or mechanics failure record exists. It is never a scientific null, which is consistent with plan §9a. | pass |
| Ledger | **Counters:** reads `counters/{generation,teacher_forced,logit}`, the same layout `reserve()` writes, with per-stage tallies and the 468/302/8/92,544 maxima. **Generations:** completed generations count only rows whose receipt hash matches; they are kept separate from reservations. **GPU time:** sessions read `unix`/`seconds` as written by the runner; unfinished sessions are charged up to analysis time and labelled as process wall time, not kernel time. **Judge:** starts are split into study and audit. Batch groups are parsed from the `{metric}-{batch:03d}-{attempt}` naming, checked against 130/260. Failure records and starts without a completed row are listed. | pass |
| Examples | Four fixed rules: H01; H14; lowest selected coherence; greatest paired sarcasm gain, with refusals counted as 0. Ties break by question ID. All four arms are shown, with output, generated IDs, scores, cap and EOS flags, and provenance: the `final/result.json` sha256, the row ID, and the score-file sha256 per metric. Examples are labelled as not representative. | pass |
| Integrity | Every input is hash-tracked on first read and re-verified at the end. The R0 `frozen` artifacts (config, inputs, rendered, rubrics, plans) must match the manifest. Output is written atomically, refuses to overwrite, and refuses any path inside the run root. Final analysis refuses to run unless the registration hashes (analyzer, tests, R0 manifest) match and the timezone-aware `registered_at_utc` is no later than the earliest `final/rows/*.start.json` `started_unix`. | pass |
| Config keys | `final_role`, `cache_disclosure`, `max_gpu_seconds`, `maximum_judge_batches`/`starts`, `maximum_generated_tokens`, `minimum_pairs` and the count maxima all exist in the frozen run `config.json`. | pass |

## Required in the C6 registration (PM process, no code change)

- **R1. Which closeout is authoritative.** The signed runner's `report()` also
  writes `closeout.json`. The registration should name the independent
  analyzer's output as the reported closeout. It should require that
  `primary_outcome` and every gate boolean equal the runner closeout, when
  both exist. Any disagreement is an implementation stop under plan §9a, not
  a choice between the two.
- **R2. Registration timestamp provenance.** `registered_at_utc` is supplied
  by the PM, and the analyzer trusts it. Record the registration file's
  sha256 in `results/protocols/` before approving the final stage. The final
  approval should also bind that hash, so the timestamp cannot be backfilled.

## Optional, before registration only

- **O1.** No action needed: the GPU ledger's unfinished-session charge depends
  on when the analysis runs. It is labelled conservative, which is
  acceptable.
- **O2.** The H14-excluded subset reports trait, coherence, refusals, caps,
  EOS and mean tokens. It omits `individual_trait_ge50` and `invalid`. With
  those two, all five gate quantities would be visible on n = 31. The frozen
  `summarize` hard-requires 32 rows, so this would need a few local lines
  calling the same `text_checks`/`hits` helpers. Lexicon hits are optional.
  This is a completeness improvement only; plan §7 does not specify which
  figures to include.

## Scope note

This review covers the analyzer as an independent, offline reader. It does
not change the R0 runner, the judge or the frozen helpers, and it can no
longer affect selection or gates for development or final.
