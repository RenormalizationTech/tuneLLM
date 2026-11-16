# Sarcasm GCAD lower-strength study: design review

Reviewer: Claude advisor, read-only. Scope: the scientific design in
[docs/sarcasm_gcad_low_strength_protocol.md](../../docs/sarcasm_gcad_low_strength_protocol.md).
Source code and registrations are reviewed separately.

## Verdict

**Accepted as one bounded, prospective development study.**

## Why it is justified

- In the parent study, the lowest GCAD strength (3.5) already saturated
  sarcasm (87.1, 32 of 32 answers at 50 or above). That exceeds the prompted
  reference (58.1), while coherence fell steadily: 52.4, 38.8 and 25.5 at 3.5,
  5 and 7.
- GCAD 3.5 failed only the relative coherence clause (52.4 against a floor of
  65.9). The grid was inherited from profanity and impoliteness, not bracketed
  for this bank. Strength therefore needs calibrating for this bank and trait.
  Different rubrics, scores and question cohorts mean the results don't
  establish that one bank is more potent than another.
- Testing below a saturated lowest dose is the standard response. It changes
  no gate, data, bank or held-out set, and it does not alter the parent's null.

## Design points checked

1. **Re-grading the baseline and prompted answers in the new batches is
   required.** The relative gate must compare arms graded in the same batch
   context. Reusing the parent's exact baseline and prompted answers avoids
   regeneration: greedy decoding would reproduce them, and hash verification
   confirms them.
2. **Arithmetic.**
   - Development: 32 + 32 + 96 = 160 items per metric, so 20 batches each for
     sarcasm and coherence (40).
   - Held-out: 4 arms × 32 = 128 items, so 16 batches each for sarcasm,
     coherence and impoliteness (48).
   - 88 first-valid batches and at most 176 starts.
   - Generations: 6 + 96 + 128 = 230.
3. **Mechanics.** Exact token and logit parity at coefficient 0, and a nonzero
   effect at coefficient 1, cover the new lowest strength. The bank and hook
   are unchanged, so six probes are enough.
4. **Invalid answers and grading attempts.** Invalid means an empty output or
   a distinct-bigram ratio below 0.6, at most 3 of 32. Grading allows at most 2
   identical attempts, for structural failures only. Authentication, runtime,
   model, hash, isolation and uncertain-start errors are fatal, and no answer
   is regenerated on the GPU.
5. **The H14 exclusion is descriptive only.** It is predeclared, and the
   primary decision uses all 32 held-out questions.
6. **There is one study and then a stop.** No further sweep, no SAPS α 3 and no
   new artifact. A second null closes GCAD sarcasm on this checkpoint and falls
   back to the Mistral SAPS reference, with its model and task limits stated.

## Interpretation limits

- The development questions are the same as the parent's, so selection here is
  a second look at the same 32 questions. The held-out set stays the only
  confirmatory evidence.
- The prompted reference scored about 10.5 coherence points below baseline.
  That could reflect real changes in the answers, the rubric's sensitivity to
  sarcastic style, or both; these data cannot separate them. The gate is
  unchanged. A lower selected strength may trade sarcasm intensity for
  coherence, so report both.
- Everything else from the parent carries over: one Claude judge, which also
  generated and reviewed the artifact; reconstructed inputs; ordinary
  questions only; coherence is not correctness; coupling is descriptive only.
- A pass establishes a sarcasm control on this checkpoint and task. It is not
  evidence of independence from other traits.
