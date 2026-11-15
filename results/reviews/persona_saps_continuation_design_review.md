# SAPS exploratory continuation design review

Claude supported continuation of the author prompt-extraction pipeline within
the existing allocation, after the coordinator reported both original rudeness
searches closed and profanity's amended grid started. The user was informed
before launch. The original SAPS clarity-gate null remains immutable.

The review confirmed 1,142 new generations,1,500 direction forwards,two logit
forwards,64 MC,68 first-valid batches and136 new starts. Reusing the100 initial
answers and five actual initial starts keeps whole work within1,242 generations
and73 first-valid batches; maximum whole starts are141 with that actual parent.

Conditions adopted in docs/persona_saps_continuation_protocol.md:

- Verify the parent is closed as elicitation_null with every downstream output
  and report set absent, both before preparation and during verification.
- Bind the exact bank: parent source/manifest,100 answers,50 baseline captures,
  first-valid grades/journal/identity and failed eligibility/stop/result.
- Reuse frozen splits, seed-702 test/custom subsets, style draws, rendering and
  original validation/test/custom plans and seeds. No new initial samples/grades.
- State explicitly that the additional original pre-extraction quality gate
  failed. It is not part of author prompt-based extraction; proceeding does not
  assert or retroactively manufacture a pass.
- Preserve all downstream extraction/mechanics, the12 validation settings,
  quality floors/ties/gain, missing-method handling, committed selection before
  one untouched test report, random control and conditional capability.
- Preserve corrected CPU-FP32-dot to Python-c arithmetic through the source hash.
- Disclose baseline grades from a different session and the observed positive
  control's entailment2.44/clarity3.46. A later quality-bound null is plausible;
  the experiment also tests real GPU mechanics and describes tradeoffs.

This is a separate exploratory completion of the explicitly requested isolated
steering work, not a model-size or mixed-trait expansion. Code review, meaningful
CPU checks and exact child registration still precede model or grading calls.

## Code review and CPU acceptance

Claude inspected the new runner, configuration and frozen shared operations.
It confirmed exact bank/input reuse, the explicit failed-gate state and new
registration authority, read-only parent access, merged entry/exit verification,
absence of an initial-generation/rerating API, source controller arithmetic,
fresh budgets, selection/test locks and descriptive quality-risk reporting.
The review found one WIP bug: select called an undefined stop function. The
worker fixed that call inside the new module and added a meaningful test through
selection_null, permanent stop and aggregate export. Frozen parent code stayed
unchanged.

The final focused CPU run passed 27 checks in 9.23 seconds; CLI and diff checks
also passed. Checks cover the false parent gate and exact bank, branch plans,
unmodified shared source operations, registration/holdout guards, first-valid
journals, the136-start ceiling, fresh/reused accounting, null reporting and
privacy. No preparation, model/GPU or API calls occurred during implementation.
The coordinator accepts the corrected implementation within its declared
adaptations. Source/model identity remains checked again before live release.
