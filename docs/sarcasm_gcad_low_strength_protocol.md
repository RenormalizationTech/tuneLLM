# Sarcasm GCAD: lower-strength study on Qwen2.5-7B

Status: prospective protocol, reviewed by the Claude advisor (see
[design review](../results/reviews/sarcasm_gcad_low_strength_design_review.md)).
It is not a stage release. Root freezes, registers and releases mechanics,
development and selection separately.

## Why this study exists

The closed parent study (`results/sarcasm_method_transfer_01.json`) is a
development null under its frozen gates. Its GCAD grid {3.5, 5, 7} was inherited
from the profanity and impoliteness studies. The lowest strength already gave
sarcasm of 87 (32 of 32 answers at 50 or above), and coherence fell steadily
with strength. Every GCAD strength failed the coherence floor relative to
baseline; 3.5 failed only that clause.

This study brackets **below** the saturated range: base coefficient {1, 2, 3}.
It is one prospective development study. It does not change the parent's null
and it is not a rescue of it. SAPS is not included.

## Fixed inputs (unchanged from the parent)

- Original `Qwen/Qwen2.5-7B-Instruct` at the pinned revision, native BF16, the
  same isolated environment and GPU 0 guards.
- The exact accepted sarcasm artifact, fixture and GCAD bank from the parent
  (87 eligible pairs), verified by hash. No new extraction, elicitation, data
  or artifact.
- GCAD at blocks 9–19, gate scale 1.5, positive addition, `inverse=False`, the
  pinned author hook.
- The same 32 development questions, and the same 32 held-out questions, which
  are still unused for sarcasm.
- Greedy decoding, batch 1, seed 42, cap 160, default template rendering.
- The same sarcasm, coherence and impoliteness rubrics, native Claude runtime,
  blinding, separate trait/coherence journals and first-valid rules.
- The parent's sources, data, outputs and results stay byte-identical. This
  study uses a fresh run root, sources, config and registrations.

## Mechanics

Six cap-32 probes: 2 ordinary hook-free, 2 GCAD with base coefficient 0, and
2 GCAD at base coefficient 1. Add 2 explicit logit forwards. Require:

- exact token **and** logit parity between ordinary and zero-coefficient
  forwards;
- nonzero hidden-state and logit changes at coefficient 1;
- prefill and cached-decode coverage;
- complete hook cleanup.

A mechanics failure closes the study. Register successful mechanics before
development.

## Development

- Generate **96** new replies: 32 questions × GCAD {1, 2, 3}.
- **Reuse the parent's 32 baseline and 32 prompted development answers as exact
  bytes.** They are not regenerated. Verify them by hash against the closed
  parent.
- **Re-grade all 160 items** (32 baseline, 32 prompted, 96 new) in new, fully
  packed, blinded batches of 8, frozen before any output: 20 batches per metric
  for sarcasm and for coherence. Every gate uses these new grades only. The
  parent's grades are not mixed in.
- **Selection (unchanged gates):** the lowest strength meeting all of:
  - sarcasm mean ≥ baseline + 20;
  - at least 8 of 32 answers with sarcasm ≥ 50;
  - coherence ≥ 50 and ≥ baseline − 20;
  - at most 3 of 32 invalid answers, where invalid means an empty output or a
    distinct-bigram ratio below 0.6.
- The prompted arm is a descriptive reference, not a gate.
- **Grading attempts:** at most 2 identical attempts per batch, for structural
  (schema) failures only. Authentication, runtime, model, hash, isolation or
  uncertain-start errors are fatal. No answer is regenerated on the GPU to
  repair grading.
- If no strength qualifies, close as a second development null and run no
  held-out generation.

## Held-out (only if a strength is selected)

- Fresh generation on the 32 held-out questions: baseline, prompted, the
  selected GCAD strength and a random control. That is 128 answers.
- **Random control:** a per-layer random delta matched to the raw norm of each
  layer of the sarcasm delta, with the same keys, demand, gate, scale and
  selected coefficient. Seeds 9603, 9604 and 9605 by question index mod 3; layer
  seed = question seed + 1000 × layer. Draw from a dedicated CPU generator, and
  store each realized tensor with its hash.
- Grade sarcasm, coherence and descriptive impoliteness, in 16 fully packed
  batches per metric.
- **Retain** if the selected strength meets the development gates on held-out
  questions **and** has sarcasm mean ≥ random + 20. Otherwise it is a held-out
  failure.
- **H14 sensitivity:** extraction question x18 overlaps topically with held-out
  H14 (bread dough). Report the held-out metrics with H14 excluded as a
  **descriptive** figure only. The primary decision uses all 32 questions.
- Impoliteness and the profanity lexicon are descriptive coupling measures,
  never gates.

## Grading order and reporting rule

- Held-out grading runs in a fixed order: sarcasm, then coherence, then
  impoliteness. The impoliteness journal cannot start until both primary
  streams are accepted.
- If a fatal stop happens **during the held-out impoliteness journal**, after
  the primary decision is complete, publish the result with terminal status
  `primary_complete_descriptive_incomplete`. It keeps the primary decision,
  the H14 descriptive figures and the stop record. Impoliteness is descriptive
  only, so its failure never hides or changes the primary decision.
- A fatal stop at any earlier point, before the primary decision is complete,
  stays a private `incomplete` record and is not published as a result.

## Budgets (maxima)

- Target generations: 6 mechanics + 96 development + 128 held-out = **230**.
- Zero prompt, teacher-forced or MC forwards; 2 logit forwards.
- Native first-valid batches: 40 development (20 sarcasm + 20 coherence) and 48
  held-out (16 each for sarcasm, coherence and impoliteness), giving **88**.
  At most 2 identical structural attempts per batch, so **176** starts at most.
  No artifact call.
- With no held-out stage, the maxima are 102 generations and 40 batches.

## After this study

Stop. There is no further strength sweep, layer search, SAPS arm or new
artifact.

- **Pass:** retain GCAD at the selected strength as the sarcasm control for
  Qwen2.5-7B ordinary replies. Report it beside the parent's null, which stays a
  null for its own grid.
- **Development null or held-out failure:** close GCAD sarcasm on this
  checkpoint. Sarcasm rests on the original-Mistral SAPS reference, which uses a
  different model and the rewrite task, and that limitation is stated.
