# Qwen3.8-27B sarcasm follow-up: review 06 erratum 01 (telemetry rounding)

Advisor erratum, 2026-11-19. It is read-only.

- No model, judge or GPU call was made.
- `qwen38_sarcasm_followup_review_06.md` (sha256 `134d0538…`) is unchanged.

## Correction

The telemetry table in review 06 ("Telemetry", actual/native ratio per token,
range over the six sites) rounded three range endpoints twice: first to three
decimals, then to two. Rounding the canonical values once, to two decimals,
gives:

| Arm and phase | Endpoint | Review 06 | Canonical value | Correct |
| --- | --- | ---: | ---: | ---: |
| b4 prefill | minimum (site 23 to 43) | 0.50 | 0.4948094030387069 | **0.49** |
| b8 decode | maximum (site 27) | 1.37 | 1.3647487508425922 | **1.36** |
| random16 prefill | minimum (site 39) | 1.86 | 1.8545092240562029 | **1.85** |

The corrected ranges are:

- b4 prefill: 0.49–0.66;
- b8 decode: 0.97–1.36;
- random16 prefill: 1.85–2.61.

The source of truth is `part_B.telemetry.arms.<arm>.<site>.<phase>.metrics.actual_ratio.mean`
in `results/qwen38_sarcasm_followup_01.json` (sha256 `61362b38…81ad`). These are
token-weighted means: each one is the `sum` divided by the `count`. The
advisor's row-level recomputation gives the same unrounded values.

## Scope

- These are display rounding errors only. Every other value in the table,
  including the gate ranges, the maximum single-token values and the
  agreement bounds, is unaffected.
- No metric, gate, outcome (`exploratory_null_through_b16`) or interpretation
  changes. Telemetry is descriptive and enters no gate.
- The archived pages `docs/qwen38_sarcasm_followup_execution.md` and
  `docs/qwen38_sarcasm_gcad_guide.md` repeat 1.37, and the execution page also
  repeats 0.50 and 1.86. They are preserved unchanged as historical records.
  This erratum and the current results page give the corrected values.
