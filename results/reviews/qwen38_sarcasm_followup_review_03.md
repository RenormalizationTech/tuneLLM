# Qwen3.8-27B sarcasm follow-up: review 03 (combined source signoff)

Advisor review, 2026-11-19. It covers the settled source after review 02's
fixes; no file changed after 11:31:30. The review was read-only, plus CPU
tests. No model or judge call was made, and no run root exists.

`CUDA_VISIBLE_DEVICES= outputs/qwen38-27b-nf4-01/env/bin/python -m pytest -m '' -q tests/test_qwen38_sarcasm_followup.py tests/test_qwen38_sarcasm_followup_pipeline.py`
gives **24 passed**.

## Signoff

**Signed off for `prepare` and for the single registered launch.** The
signoff covers exactly the hashes below. Any change to these files voids it.

| File | sha256 |
| --- | --- |
| `scripts/qwen38_sarcasm_followup.py` | `c3795e27cdeb4c9f6f304987af9918d9916bc14655ef27db6706a34d14b3f9a3` |
| `scripts/qwen38_sarcasm_followup_telemetry.py` | `de4e6c71c114465829d1337d509eb26a546c48d8d00554ef70a18e97aafc5710` |
| `scripts/qwen38_sarcasm_followup_analyze.py` | `8e13912b3434783d88a4fe13a8d798c157b78b3b0f101d9e4da8044396d143e8` |
| `tests/test_qwen38_sarcasm_followup.py` | `724e166fb3d25d1e1ee6acb19eb6fc8fc8fc47edee0ed6ee5662bacf8e6b0747` |
| `tests/test_qwen38_sarcasm_followup_pipeline.py` | `6adb2f7f6738437c52653270b685ef82095724f58a8ef61c81e8ec1d40b8072b` |
| `configs/qwen38_sarcasm_followup.json` | `7c19636c449f2071cf0ec9bf76ee5e1f73a5b86b849e85effc7fa62db59e16a2` |
| `docs/qwen38_sarcasm_followup_plan.md` | `329705d2d1ce5a6a18887f559fdeafe9774c15a5fae1a9bd3e45963f4d153f40` |
| `evaluation/qwen38_sarcasm_followup/development.json` (PM fixture) | `3da540a3f0e73bf67e1e7dbaf31638c7b2f240ba29a8ec3a9565e5f1072add4b` |

**Plan edit made in this pass.** One line in §6 now matches the code: the
judge plans use seeds 20261120 + metric index (trait 0, coherence 1,
impolite 2). No other design change was made. The plan freezes at these
hashes.

**Old study sources are unchanged**, and match old R0, C6 and review 03:
- `qwen38_gcad_port.py` `3d76d814…`;
- `qwen38_sarcasm.py` `ed2b942d…`;
- `qwen38_sarcasm_judge.py` `483c711d…`;
- `qwen38_sarcasm_analyze.py` `26dc69b1…`;
- `configs/qwen38_sarcasm.json` `0844af59…`.

## Review 01 and 02 items: closed

| Item | Resolution |
| --- | --- |
| B1 analyzer | present and reviewed (review 02, and this pass) |
| B2 manual norm recheck | `InstrumentedSteerer(retain_last=True)` keeps the original, requested and steered tensors for the last call at each site. In the record2 logit forward, mechanics recomputes per-token ‖requested‖ (model dtype → FP32) and ‖steered − original‖ (FP32) and requires `allclose(rtol=1e-5, atol=0)` against the logged trace. CPU test: `test_retain_last_manual_requested_and_actual_norms`. |
| B3 old prompt equality | `build_inputs` raises unless the re-rendered N == old baseline `input_ids` and S == old prompted `input_ids`, for every question. A failure-path test covers it, and the PM also confirmed it on the real tokenizer. |
| B4 tests | Now covered: recorder parity at b ∈ {0, 2, 16}; the recompute test; diagnostic neutrality, crop parity and native token means; `build_inputs` complete and failure paths; 7-arm plan packing; journal resume, Uncertain and stage budget; audit failure running GPU and skipping judging; the watchdog killing the process group; combined judge and token budget refusal; bootstrap rejecting token-level shapes; CI wording and norm-of-mean CI; and the **end-to-end analysis** over a fake run directory using the real launcher keys and the real `generate()` schema. |
| R1 non-zero | Mechanics now also requires `not torch.equal(ordinary, frozen b2)`. |
| R2 random arm | `load()` requires each site's ‖Δ_random‖ to equal the bank's within relative 1e-5, and K_pos and D̄ to be `torch.equal`. |
| A1-1 crop on N | `crop.*` keys use S and L only, and the N contrasts are skipped for them. The end-to-end test includes `crop.1` with no N crop. |
| A1-2 geometry | `inputs.model_geometry` is written from the model config. |
| A1-3 end-to-end test | Present. Asserted cases: `primary_complete_descriptive_incomplete` with `exploratory_candidate = 2` and no b16 margin; `judge_audit_failure`; and `implementation_stop` with Part A complete. The complete-closeout case follows in the same test, and the output passes `json.dumps(allow_nan=False)`. |
| A1-4 | Uses the frozen `text_checks(...)['mostly_not_letters']`. |
| A1-5 | `cosine_ci` emits `ci_wording` ∈ {positively_aligned, anti_aligned, no_clear_alignment} from the CI against 0, and is labelled not a test. |
| A1-6 | `norm_mean_ci` gives a Gram-based question-cluster CI for the norm of the pooled mean. |
| A1-7 | All requested limitations are present: random only at b16, whole S/L wording, the norm-unit correction, fixed vs free conditioning, and no conclusion that dose caused the old null. |

## Registered behaviour confirmed

**Counts and caps (config and launcher agree).**
- 232 generations: 8 mechanics + 224 Part B.
- 36,096 reserved tokens.
- 194 teacher-forced passes: 2 + 192.
- 5 logit forwards.
- Judge: 168 study starts + 1 audit, and 84 batches.
- 4 h GPU and 12 h wall limits, enforced by parent watchdogs.
- Stage-entry budget checks before Part A and Part B.
- A 2 GiB cap on activation storage; the expected size is about 1.55 GB.
- GPU0 only (`GPU-34646ab9…`), with no fallback.

**Order and conditions.**
- The stages run as audit, GPU (mechanics, Part A, Part B), judging, then
  the analyzer.
- A failed audit lets the GPU stages run and skips judging.
- Any implementation failure halts the run with preserved evidence.
- Rows are never retried; completed rows hash-replay.
- No Part A output is read by the code that runs Part B.

**Claims are fixed in code.** Part A is labelled post-hoc and observational,
with signed cosines, cluster CIs, the CI wording rule and the 3 random
reference values. Part B uses the frozen gates, with labels
`exploratory_candidate` or `exploratory_null_through_b16`, and
`b16_direction_check` as the only dose-matched control. The output always
carries `old_outcome_unchanged: development_null`.

## Residual notes (no action required)

- The mechanics recompute compares the logged norm with the same arithmetic
  applied to the retained tensors. It shows that the logging pipeline records
  the right tensors. It is not an independent second implementation, and it
  does not need to be.
- Decode telemetry per step stores each step's single-token values. Prefill
  percentiles are per call. The arm-level summary aggregates
  sums, minima and maxima weighted by tokens.

## Next

The PM runs `prepare`, which is CPU-only and creates the run root. The PM
then writes one approval binding the manifest, the launcher hash, the
algorithm string and the GPU UUID, and runs `launch`. I will review the
mechanics and closeout outputs when they exist, in new follow-up review
files.
