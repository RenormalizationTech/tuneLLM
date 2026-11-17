# Qwen2.5-7B refusal-steering transfer: design review

Reviewer: Claude advisor, read-only, except for this file. No target-model or
judge call, data edit or generation was made. This review records the
minimal-port recommendation that the manager adopted. It is not a new design.

## Scope

Port the completed Qwen3-4B run-03 refusal-reduction workflow
([protocol](../../docs/refusal_reduction_protocol.md),
[execution](../../docs/refusal_reduction_execution.md)) to the unadapted
`Qwen/Qwen2.5-7B-Instruct` at revision
`a09a35458c702b33eeacc393d103063234e8bc28` (28 blocks, hidden size 3584).

Change only what the new model requires. There are no new benchmarks, gates,
layer or strength sweeps, or canonical-site redesign. The cancelled
OR-Bench/140-candidate plan does not apply.

## Required adjustments, and only these

1. **Re-extract the direction on 7B.** The 4B vector has the wrong dimension.
   Keep the exact 4B recipe:
   - the same 100 prepared AdvBench/Dolly pairs;
   - the **block-output** site, at the last real prompt token after the
     generation prefix;
   - harmful mean minus benign mean, equally weighted, forward passes only.

   Do not substitute the canonical block-input extraction.
2. **Native Qwen2.5 rendering.**
   - Qwen2.5 has no non-thinking switch. The retained helper still passes
     `enable_thinking=False`, but the Qwen2.5 template ignores it. Record the
     setting as inapplicable, not as "nonthinking". As a cheap check, confirm
     that the rendered bytes are identical with and without the flag.
   - The 4B "no system prompt" setting cannot be rendered natively: the
     Qwen2.5 template inserts its default system message when none is given.
     Use the native default for extraction, development, final and retention.
   - Disclose this as the single rendering deviation.
3. **Fixed layer by relative depth.**
   - 4B block 23 output comes after 24 of 36 blocks (0.67 of depth). Map it
     deterministically: round(23 × 28/36) = **block 18 output**, after 19 of
     28 blocks (0.68).
   - Raw index 23 on 7B would be 0.86 of depth, inside the last 20% that the
     author pipeline excludes as a direction source.
   - Block 18 is fixed before any outputs. There is no search and no fallback
     layer.
4. **Interventions.**
   - `subtract18`: −0.5 × the raw 7B vector at block 18's output, at all
     positions including cached decode.
   - `ablate_all`: fp32 projection removal of the same unit direction at each
     block input and attention/MLP residual-writer output, 28 × 3 = 84 sites.
   - Report two ratios descriptively, both against the median last-token
     extraction activation norm: the raw ‖v‖ ratio and the coefficient-scaled
     0.5‖v‖ ratio. Neither is a gate, and the coefficient is not rescaled.
     The 4B comparison point is the scaled shift: 0.5 × 61.19 ≈ 30.60, about
     30% of the median norm. It is not the raw-vector ratio.
5. **Generation settings: the old helper, with deviations disclosed.**
   - Keep the old greedy settings: repetition penalty 1.0, single EOS 151645
     (the tokenizer EOS) and PAD 151643. The same settings apply to every arm.
   - The native `generation_config.json` differs: EOS is `[151645, 151643]`
     and it specifies sampling with repetition penalty 1.05. Record that as a
     disclosed difference from the native defaults, not an error.
   - Report how many outputs emit 151643 before 151645 or the cap. This is
     descriptive, not a gate.
6. **Fresh 7B baseline, with no reuse of existing 7B outputs.**
   - The earlier 7B XSTest-safe screen used cap 256, batch 1 and a different
     split, which is incompatible with the 4B workflow (cap 320, batch 8,
     mask-derived position IDs, greedy, seed 42).
   - Generate baseline and both arms under the 4B settings. Judge all 180
     development items together in one fresh blinded packet with the
     existing registered rubric, validator and adapter.
   - Run the 4B identity and nonzero probes on the first 8 safe prompts.
     Identity hooks must reproduce the new baseline tokens exactly.
7. **Rules carried over verbatim, relative to the new 7B baseline.**
   - Development: source-unsafe valid full compliance at least baseline + 6
     out of 30; no increase in empty, degenerate or unscorable outputs per
     class; safe useful/correct at least baseline − 2; the same tie-break.
   - Final: the same 72 prompts.
   - Retention: the same 48-item fixture and the same final quality criteria.
   - A null skips final and retention.

## Data allocation (counted by ID only)

The 4B safe prompts overlap the earlier 7B XSTest-safe split:

| 4B split | Safe | In 7B development (100) | In 7B held-out (150) |
| --- | ---: | ---: | ---: |
| Development | 30 | 10 | 20 |
| Final | 42 | 17 | 25 |

The unsafe contrasts are not in that split.

- **Reallocation:** the 45 overlapping held-out rows are reallocated
  prospectively to this carry-over study under the user's
  use-the-old-workflow authorization. This must be recorded in the
  registration before any generation.
- **Protected:** the remaining 105 rows and all 25 DECCP held-out rows stay
  protected.
- **Preserved:** old split files, scores and artifacts stay byte-identical.

## Interpretation: mechanism versus the benign goal

- **Selection measures the mechanism, not benign reduction.** The workflow's
  selection signal is willingness to answer the **source-unsafe** XSTest
  contrasts.
  - These are steered answers to unsafe prompts, kept private and labelled as
    a mechanism measurement.
  - They are never relabelled as benign, and harmful answers are not
    optimised. This is the only harmful-prompt generation in the port.
- **The user's primary goal is reducing unnecessary benign refusals.** The 72
  safe prompts were chosen by type, not because the model refused them. Given
  the earlier 7B rate (3/100 XSTest-safe), benign baseline refusals here may
  be near zero.
  - In that case the port can show the mechanism plus quality retention, but
    not a reduction in natural benign over-refusal. State this plainly.
  - A new benchmark becomes necessary only if this port runs and lacks
    measurable benign headroom.
- **Outcome wording follows the 4B precedent.** A dev pass with a final
  quality failure is a "partial mechanism, not a usable controller". Passing
  every final criterion is still a small-sample learning result, not a safety
  or product claim.

## Budget

- 32 probe generations;
- 180 development generations (60 × baseline and two arms);
- up to 144 final generations;
- 32 retention generations plus 64 multiple-choice likelihood items.

That is at most **388 target generations** (32 + 180 + 144 + 32), plus 200
extraction forward passes. This is not the old 4B total of 328, because the 60
development baselines are now generated fresh. One A6000 is enough.

## Protocol check

`docs/refusal_7b_transfer_protocol.md` is consistent with this review. It
covers:
- block-output extraction and the fixed block 18;
- −0.5 × raw subtraction and the 84-site ablation;
- native rendering and the fresh 180-item joint packet;
- verbatim baseline-relative rules;
- the 45/105 reallocation;
- the benign versus mechanism interpretation;
- the 388 budget.

No scientific correction to the protocol is needed. The items above
(inapplicable thinking flag, both norm ratios, EOS/repetition-penalty
disclosure) belong in the manifest and the execution record.

## Status

The design is acceptable as a minimal port. The next step is code review of
the worker's runner, config, tests and runbook, before any GPU use.
