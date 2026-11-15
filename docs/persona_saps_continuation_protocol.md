# SAPS exploratory continuation using the unchanged parent bank

Status: Claude and the coordinator agreed on this bounded continuation under the
user's existing request to establish isolated steering for all three traits.
The user was informed before launch. New source acceptance and exact registration
precede child execution. No original null, frozen source or quality floor changes.

## Why a separate study

The [original study](../results/persona_saps_sarcasm_01.json) closed as
elicitation_null solely because prompted clarity was 3.46 rather than 3.5.
The same control produced sarcasm mean 3.04, with 36/50 scores at least 3 and
grammar 4.64. Entailment was 2.44, against literal baseline 4.60. The original
combined gate failed; it is never relabelled as passed or rerun.

That pre-extraction quality gate was our additional check, not part of the
[published SAPS extraction procedure](https://aclanthology.org/2026.findings-emnlp.1594/).
The method extracts directions from prompts, not from generated positive
answers. This separate exploratory study completes the author-style pipeline
while preserving all downstream quality/selection/report criteria. The observed
prompted quality makes a later selection_null plausible; do not promise usable
control. Actual GPU mechanics and the full fixed-versus-adaptive development
tradeoff are informative outcomes even if no setting qualifies.

## Parent bank and new paths

Parent: `outputs/persona-saps-sarcasm-01`. Assert its stop/result remain
elicitation_null and no directions, preflight, validation, test, retention,
test/custom judging or downstream grades exist. Check this before preparation
and throughout child verification. Parent access is read-only.

Child: `outputs/persona-saps-sarcasm-continuation-01`, aggregate
`results/persona_saps_sarcasm_continuation_01.json`. Use new modules/config and
safe child guards. No runtime guard monkeypatches or edits to parent sources.

Reuse exactly the parent manifest and frozen inputs: author data, seed-702
100-item test subset and first-20 custom subset, train/validation split, 1,000
style draws, rendering/token IDs, reference/custom rubric, retention fixture,
all plans/branch plans and seeds. No new choices based on the initial scores.

Reuse all 100 initial answers and their five first-valid batches unchanged,
including starts, receipts, model identity, accepted labels and raw outputs.
Bind the parent's failed eligibility, stop, result, initial/run.json, predictions,
h0.safetensors and capture metadata. The baseline h0 is the original read-only
first-prefill capture over 50 validation inputs and four candidate blocks.
Never regenerate/rerate these conditions, rewrite their journal bindings or
pretend they are fresh child work. Initial control observations are not a new
eligibility pass; an explicit child registration authorizes the new method.

Use one merged/deduplicated model/source/input verification at live entry/exit,
including actual consumed parent records. Read helpers check their own phase
bindings without recursively reopening unrelated historical studies or rehashing
the whole model. Source and package identities remain fixed.

## Unchanged downstream method

The existing official pinned Qwen3-4B native HF BF16/SDPA model, GPU0 guard,
TF32 off, nonthinking, greedy seed42, batch1, cap60 and cache remain unchanged.
The original TransformerLens/backend/dtype/BOS and decoding differences stay
disclosed; no TransformerLens tensors are transferred into the HF basis.

1. Learn directions using all500 train strings: 500 plain plus500 first-pass
   styled prompt forwards, then500 styled target-pass forwards. Reuse the
   pre-frozen 1,000 instruction draws. Capture final-prompt-token raw outputs
   of all36 decoder blocks, including block35 before final norm. FP32 mean
   contrast, per-block units and second-pass targets, exact save/reload.
2. Ten unscored cap32 generations: four ordinary with read-only capture, four
   actual-zero hooks, two nonzero probes; two extra logit forwards. Check source
   operations, exact parity, actual deltas, cached scope and cleanup. Commit
   vector/preflight bindings before scored steering.
3. Validation: blocks11/15/19/23; fixed coefficient1; adaptive1 or1.4. These
   twelve settings add600 generations on the50 validation strings. Reuse the
   original baseline and original grades. Gram/clarity means must be at least
   3.5, entailment at least3, and empty/degenerate/capped counts at most baseline+3.
   Highest style wins, ties smaller alpha, earlier block, fixed. Require style
   gain at least0.5; absent methods are skipped, never filled by a quality failure.
4. Commit selection before any test output. Up to five arms on100 untouched
   strings: baseline, best fixed, best adaptive, random for the locked winner,
   prompted style-index0 control. Fresh test-only h0 comes from the read-only
   baseline before fixed arms. No per-example layer oracle or test reselection.
5. Reference report criteria remain quality floors and style gains at least0.5
   over baseline and random, plus prompted control at least50% scoring>=3.
   Custom scores cover the same20 fixed test strings across the existing arms,
   separately on saved bytes, no extra generation and no reference-score context.
6. Only after the reference report passes: fresh matched48-item base/winner
   capability,32 generations and64 MC. Batch1 with case-specific h0 for fixed
   projection, unchanged official scoring/tolerances and restored language seed.

Fixed c follows the corrected author arithmetic: CPU FP32 dot, convert dot and
target separately to Python scalars, then Python subtraction/alpha; apply Python
c to the model-dtype unit. The adaptive target remains a zero-dimensional tensor
and the controller uses the last current activation, broadcasting over the whole
prefill and then each cached token. Alpha1.4 intentionally overshoots.

Keep the original matched-controller random supplement, source-seeded units,
per-position shift norms split by prefill/decode and report-only ratio flag
outside0.8–1.25. Full-prefix Frobenius norm scales by sqrt(prefix width), not width.
Random and selected trajectories need not have equal actual norms; no second
random construction or extra criterion is added.

## Fresh and reused budgets

| Work | New child maximum |
| --- | ---: |
| Generations | 1,142 = 10+600+500+32 |
| Direction prompt forwards | 1,500 |
| Unscored logit forwards | 2 |
| Capability MC | 64 |
| First-valid judging batches | 68 = 30+25+13 |
| Native grading starts | 136 |

The parent consumed100 generations and five native starts, all first-valid.
Whole allocation remains1,242 generations and73 first-valid batches, as in the
original maximum; no repeated initial generation or grading. With five actual
parent starts, whole maximum is141 starts. Missing-method branches and nulls
reduce actual work. Report fresh/reused costs separately; no double counting.

Keep pinned native Claude2.1.323 tools-off isolation, actual model/usage checks,
two identical-payload schema attempts maximum, first valid retained, permanent
infrastructure/hash/identity/uncertain-start stops and nonfatal resource waits.
Reference packets use up to20 distinct source strings, custom up to8; all exact
parent plans remain frozen. No initial-rerating API phase is exposed.

Disclose old/new judging sessions, one structured Claude versus three original
judges, co-rating contrast effects, strict sentence-alone versus custom contextual
sarcasm, hostile/profane input confounds, incomplete data lineage, and the failed
pre-extraction control. Source questions and raw labels remain private where
appropriate; public output is aggregate-only. No human-validity or general
assistant-persona-control claim. No joint traits, new model size or further
search expansion follows automatically. Audit and consolidate after closure.
