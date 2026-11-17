# Qwen2.5-7B benign transfer diagnostic: implementation review

Reviewer: Claude advisor, read-only, except for this file. No GPU, target-model
or judge call was made, and no source file was edited.

**Scope:**
- `scripts/refusal_7b_benign_transfer.py` (262 lines, frozen);
- `tests/test_refusal_7b_benign_transfer.py`;
- `scripts/refusal_7b_benign_transfer_runbook.md`;
- the reused half-dose and transfer helpers.

These were checked against `docs/refusal_7b_benign_transfer_protocol.md` and
the [design review](refusal_7b_benign_transfer_design_review.md).

## Verdict

**Acceptable for registration. No blocking findings.**

**Tests:** 11 passed in 0.25 s. That is the worker's run, reported by the
manager; the reviewer did not run it. Command:

`OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_refusal_7b_benign_transfer.py`

Registration must bind these frozen sources.

## Checked and correct

- **Fixture:**
  - The 17 are derived from the original 4B primary nonempty refusals,
    cross-checked against the accepted grades.
  - The code asserts the 13 new and 4 seen membership, with only the seen 4
    in the 4B-workflow development safe set.
  - Text 67 keeps its original P label even though the review called it
    questionable. There's no relabelling.
  - The 13 come in source order from the pinned `safe250.json`.
- **Ledger:** the code asserts the 7 newly consumed reserved IDs and 98
  remaining. The inventory records final-set 57 and 64 as consumed and
  DECCP as untouched.
- **Seen 4:** existing paired labels from the half-dose development scores.
  They are not regenerated or rejudged, are reported separately and are never
  pooled.
- **Generation:**
  - One non-restartable `comparison` stage with exactly 26 generations:
    baseline and −0.25 on the 13.
  - The −0.25 arm uses the half-dose `0.5 × raw` vector through the
    unchanged hook.
  - The old `generate_condition` invariants apply: native render, greedy,
    cap 320, batch 8 split 8+5, repetition penalty 1.0, EOS/PAD as before.
  - The two arms must match exactly in inputs, rendered prompts, batch
    membership and settings.
- **Judging:** one blinded 26-item packet with the original rubric and
  audited provenance. Ratings from a wrong packet are rejected.
- **Reporting:**
  - Paired useful rescues require a valid, full and useful answer.
  - Newly-refused and useful-to-not-useful items are listed by ID, with
    invalid counts, the P/I class breakdown and caps.
  - The overlap note prevents summing harm categories.
  - There are no success fields, gates, rates or statistics.
- **Guards:** `--execute-gpu`, a committed matching registration, committed
  sources, protocol and judge registration, and the GPU0 check, all before
  model load.

## Tests, mapped to the review requests

| Request | Test |
| --- | --- |
| 13/4/7/98 membership and text 67 | `test_pinned_source13_seen4_reserved7_and_original67` |
| Tampering rejected | `test_derive_rejects_source_label_or_review_changes` |
| 26-item packet, 8+5 parity, no reissue | `test_joint26_blind_packet_8_5_parity_and_no_reissue` |
| Batch, input or settings mismatch rejected | `test_input_or_8_5_membership_change_rejected` (3 cases) |
| Packet provenance | `test_joint26_import_provenance_rejects_wrong_packet` |
| Rescue definition and non-disjoint harms | `test_rescue_requires_useful_full_valid_and_harms_not_disjoint` |
| Primary 13 and seen 4 kept separate, no rates or success | `test_summary_primary13_seen4_separate_no_rates_or_success` |
| GPU flag and the 26 ceiling | `test_no_gpu_without_flag_and_fixed26_ceiling` |
| No restart | `test_no_stage_restart_preserves_interrupted_progress` |

## Non-blocking notes

These are handled by the manager's manual check before GPU use:
1. The copied `reuse/vectors.pt` SHA should equal the half-dose parent's.
2. The 4B-workflow final safe IDs ∩ the 13 should equal {57, 64}, matching
   the hard-coded inventory disclosure.

## Limits (from the design review)

- The texts were selected on 4B refusals, so there's no population rate.
- −0.25 was chosen on development prompts including the seen 4, and it failed
  its registered unsafe gate. Both nulls stay unchanged.
- The seen 4 were graded in a different packet context.
- AI labels from a single judge family, truncation at the cap, and prompts
  exposed publicly and in earlier 4B runs.
- Results are case counts only. They don't establish a benign controller.
