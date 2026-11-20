# Gemma 4 31B QASPER study: analysis amendment 01 review

Advisor review, 2026-11-20. Read-only: the advisor made no model, GPU or judge
calls, edited no source, and opened no scores, predictions, selection or
analysis output. Only file names, hashes, the analyzer traceback and the two
retention `registration.json` configs were read.

**Verdict: amendment 01 is accepted for registration and CPU-only execution**,
at the hashes below. If any listed file changes, this signoff no longer applies
to it.

## The defect

R0 `ba1983fdb9e6ca1e11ce9648e69a4c057832d6524b5014cd99cabe3fca4ba9f0` closeout
stopped in `validate_retention` (frozen analyzer line 320, "retention profile
differs"), called from `analyze` line 373.

- **Where the literals disagree:**
  - The frozen analyzer, line 319, expects `mc` = "sum full-answer conditional
    log probabilities; raw concept prompt".
  - `scripts/gemma4_retention.py` line 100 writes "…; native BOS + raw concept
    prompt".
- **Cause:** the producer's label was updated during the MC BOS fix, but the
  analyzer's copy of the literal was not. No test covered the label.
- **Missed in review:** source review 01 did not catch this.
- **Both runs used the registered method.** The two retention registrations
  differ from the expected profile in `mc` only:
  - base `d74319771730c062c8ae58ea44ed0f03f94ca16fb1e93b21afa19eef5de2ddb1`;
  - adapter `0cb1a3514d901724053a3e4f4ec1e07a79f949fa40cd74a029a131c9a9a0fd62`.
  - They match on BOS id 2, choice ids [562, 603, 565, 622], stop tokens
    [1, 106], caps 512/1024, autocast, seed 42 and batch sizes 1. This is a
    validation-label defect, not a method deviation.
- **Nothing leaked:** `analyze` writes its report only at the end, so no
  `analysis.json` and no `results/gemma4_qasper_final_01.json` exist.

## Signed amendment source

| File | SHA256 |
| --- | --- |
| `scripts/gemma4_qasper_analysis_amendment_01.py` | `78506e39bed434703d2796f5e16e5824e0f01ce1c52124c2ac02450c7b3256b3` |
| `scripts/gemma4_qasper_closeout_amendment_01.py` | `bc95dd9d26285e9e486cfa06b29370ac0b3d089562bc59eb906513cdde2d1f24` |
| `tests/test_gemma4_qasper_analysis_amendment_01.py` | `a7daf28c4f21b1818c0dc37513cd193c070079ee7100115c16dbf67396f0bb48` |

## Frozen and unchanged (re-hashed)

| File | SHA256 |
| --- | --- |
| `scripts/gemma4_qasper_analysis.py` | `f957f8a0de70b6a4624ea6c61f27c90e6ebc84a65290e7ab5be0596957337f2a` |
| `results/protocols/gemma4-qasper-01.json` (R0) | `ba1983fdb9e6ca1e11ce9648e69a4c057832d6524b5014cd99cabe3fca4ba9f0` |
| `docs/gemma4_qasper_plan.md` | `6d9d1a090d290cd339fd0ca85dc6488fdaaa50bd1e0c0f48af52740c316eac96` |
| `results/reviews/gemma4_qasper_source_review_01.md` | `73a966a1ecbbb6951f74cc8fa30ada8d774a2319aa72c54d1554223cc886c9e5` |
| `outputs/gemma4-qasper-study-01/completion-seal.json` | `97aed854a630993bbcd94967da8b56faf24249f5281260ba716b0e0caa73e861` |

## What was checked

- **Amended analyzer.** A text diff against the frozen file shows only two
  changes:
  - the module docstring, which now cites the frozen SHA and R0;
  - the single `mc` literal, which now equals the producer's string and both
    registered configs.

  The method hashes, gates, bootstrap, denominators and all other expected keys
  are unchanged.
- **Wrapper equivalence check.** I ran `equivalent_sources()` on CPU and it
  passes. Both files begin with a string docstring, so dropping `body[0]`
  removes only the docstring. The check requires exactly one replaced constant,
  and line numbers do not affect the comparison.
- **Wrapper inputs.** Running `inputs()` on CPU binds 32 files:
  - the sealed registration, result and predictions files;
  - every `scores.json` and `prompt_selection.json`;
  - `longest_row_mechanics.json` and `development-000{21,41,61}.json`;
  - `prompt_alias.json`;
  - the completion seal, `analysis.log`, the failed `pipeline-status.json` and
    R0.
- **Wrapper execute.** Before running `analyze`, `execute` checks:
  - the amendment registration SHA;
  - source equivalence;
  - every source and fixed-input hash;
  - this review's binding.

  It runs only the amended analyzer, on R0, to a new write-once file. Its
  canonical result has the same schema as the pipeline's `closeout()` output
  plus an `analysis_amendment` block. The original failure files are kept, and
  the receipt is a separate write-once file.
- **Test.**
  - **Exact source change:** the amended body must equal the frozen body with
    exactly one OLD→NEW replacement.
  - **Real configs:** both saved retention configs are read (configs only) and
    must carry NEW.
  - **Validator behavior with each real config:**
    - the frozen validator rejects it;
    - the amended validator accepts it;
    - prediction and score I/O are replaced with sentinels.
  - **Other keys still guarded:** every other profile key, the scorer
    bindings, fixture identity and the 288-item count are still rejected under
    the amended validator.
- **CPU tests: 102 passed.** These are the 19 new tests plus the 83 existing
  Gemma tests, run in `outputs/qwen38-science-01/env` with
  `CUDA_VISIBLE_DEVICES=""` and `OMP_NUM_THREADS=1`. Under `.venv` (the IFEval
  scorer environment), 10 existing worker tests fail. That is an environment
  difference, an older `transformers` without Gemma 4 support, not a source
  defect.

## Residual items (non-blocking)

- **No up-front existence check.** `execute` does not check that the canonical
  result and the status file are absent before calling `analyze`. If either
  existed, the amended analysis file would be orphaned. `prepare` checks the
  canonical result.
- **No CPU-only guard.** `execute` does not enforce CPU-only visibility. The PM
  will run it with `CUDA_VISIBLE_DEVICES=""` and `OMP_NUM_THREADS=1`.
- **Equivalence-check rejections untested.** The test does not exercise
  `equivalent_sources()` rejections directly. It fails safe: a wrong wrapper
  `NEW` gives zero replacements and the check raises. The test's exact text
  check covers the same property.
- **Disclosure:** the advisor saw from a directory listing that
  `prompt_alias.json` exists, meaning base won prompt selection. Its contents
  were not opened.

## Conditions

1. **Order:** this review, then `prepare`, then register the amendment, then
   `execute`, CPU only. No GPU job, rerun, reselection or gate change.
2. **New failures:** any validation failure after `validate_retention` (score
   recomputation, scorer packages, the 288 count, outcome) stops closeout and
   needs a separate amendment. It must not be patched into this one.
3. **Final audit:** the advisor audits the final outcome independently after
   execution.
