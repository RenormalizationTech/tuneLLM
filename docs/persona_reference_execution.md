# Literature-first isolated steering execution

User authorization: establish rudeness, sarcasm and profanity separately on 4B,
following published workflows first. Joint control is deferred for later
discussion. Physics LoRA and completed refusal work are independent.

## Rudeness / published impolite Persona Vectors

Status: closed as `selection_null`. No quality-eligible development setting met
the ten-point gain requirement. See the
[consolidated closeout](persona_reference_rudeness_closeout.md) and
[aggregate result](../results/persona_reference_rudeness_01.json).

Protocol, source/data/license, code and initial exact manifest registration were
committed as 95ca211 before target outputs. Claude accepted the disclosed method
adaptation after source/protocol/code inspection. The dedicated worker completed
34 CPU checks, including author-operation differentials, native tokenizer masks,
layer/final-norm indexing, zero-hook/cache behavior, conditional fallback budgets,
first-valid journals, null reporting and public-report privacy. The worker did
not perform live model or grading calls.

The coordinator prepared the fresh ignored run
`outputs/persona-reference-rudeness-01` and released200 elicitation generations
on authorized GPU0. No other jobs were changed. Later stages remain conditional
on the registered elicitation and selection gates. The complete normal budget
is 422 generations and53 first-valid grading batches; the single conditional
fallback raises the maxima to 502 and 64. These are maxima, not completed work.

Measured elicitation cost: 921.49 seconds in generation,926.88 seconds for the
stage including loading and serialization; mean 249.305 generated tokens,
28/200 capped,8.438GB peak PyTorch allocation. The count includes every saved
generation. The original-scale packet comprises 25 blinded eight-question batches.

All 25 grading batches were first-attempt valid. The fixed filter retained 66/100
pairs spanning all 20 questions; all retained native trajectories are used
(132 teacher-forced responses). Across all 100 positive and100 negative responses,
reference trait means were57.52 versus1.55 and coherence79.55 versus86.50;
neither side had a semantic refusal. These are prompted elicitation observations,
not evidence of activation steering.

Retained pairs by source instruction index 0..4:7,20,17,16,6. Excluded-pair
criteria overlap: 32 positives scored below 50 and five pairs failed coherence;
none failed through refusal or a negative score>=50. Nineteen of 132 retained
responses reached cap 512. Extraction took 14.35 seconds; one raw/retokenized
response mismatch and zero prompt-prefix mismatches were diagnostic only.
At coefficient 2, shift/per-token-residual norm ratios over the nine candidate
layers ranged 0.243–0.481; no setting was chosen from these ratios.

The ten cap 32 preflight generations passed exact zero-token parity, nonzero
hidden/logit changes and cached hook coverage/cleanup. Maximum logit delta was
15.0625. Vector/preflight bindings were committed as f8613d7 before the scored
layer screen. This stage adds 100 generations: common baseline plus nine sites
at coefficient 2 on the ten fixed development questions.

The layer screen completed 100 generations in 679.40 generation seconds,
684.71 whole-stage seconds, followed by thirteen first-valid grading batches.
The fixed rule chose author layer 8 / block 7. At that site, four new strengths
added 40 generations and five first-valid grading batches. The best eligible
gain was 0.5 (2.2 versus 1.7); stronger settings failed coherence. The registered
fallback did not apply because some coefficient-2 sites passed quality.

Total: 350 generations, 132 teacher-forced responses, zero MC and 43 native
starts, all first-valid. No report-set, random/prompt, custom or capability
generation was triggered. The worker independently verified the completed
outputs, journals, deterministic selection, null stop, privacy and genuine cap
flags. The original search can lock an ineffective shallow site while weaker deeper
settings remain unexplored. The separate registered grid below tests that
limitation; the original result remains unchanged.

Initial registration:
[`persona-reference-rudeness-01.json`](../results/protocols/persona-reference-rudeness-01.json).
Method and limitations: [protocol](persona_reference_protocol.md),
[design/code review](../results/reviews/persona_reference_design_review.md).
Private raw answers, grading receipts and logs remain ignored. The reusable
operations are source-tested; effective narrow rudeness control is unvalidated.

## Sarcasm

Claude reviewed the bounded SAPS design and implementation, including its
corrections. The worker froze the separate native HF port after 24 CPU checks.
Exact registration was committed in 17daf9b after rudeness closed, before GPU0
execution. The initial 50 baseline and 50 prompted-control answers were graded
in five first-valid batches. The study closed as elicitation_null: prompted
sarcasm mean 3.04, with 36/50 scoring at least 3 and grammar 4.64, but clarity
3.46 fell below the registered 3.5 minimum. Baseline style was zero; baseline
clarity 4.76. Prompted entailment was 2.44 (not an initial-gate criterion).
This is a failed combined control gate, not absence of elicited sarcasm.
No directions, intervention search, held-out, custom or capability stage ran
in that original study.
Generation took 112.21 seconds, whole stage 116.81 seconds. Mean tokens were
25.2 / 30.6 (baseline / prompted), with zero capped outputs. The paper is
privately archived for rubric provenance. This is literal-to-sarcastic
paraphrasing, not demonstrated generic
assistant persona transfer. See [protocol](sarcasm_reference_protocol.md).

A separate [registered continuation](persona_saps_continuation_protocol.md) now
follows the author's prompt-activation extraction despite our added initial gate.
The original failure is retained, and downstream quality floors are unchanged.
Its runner passed 27 CPU checks and Claude's review. Registration 4da9c59
preceded all new calls. Direction extraction is running: 1,500 prompt forwards,
followed by mechanics checks before any scored intervention validation. The
original 100 answers and five grading calls are reused without rerating.

## Profanity

New casual non-slur swearing artifacts were generated with the published PV
general template and separately registered provenance. The first native request
was structurally valid and passed all mechanical checks. Root and Claude read
all five instruction pairs, forty questions, rubric, split and matched complete
systems and accepted them without edits or reroll. Exact semantic/dataset/system
hashes were committed in ec78901 before promotion. The helper reconstructed
raw response through parse and split to exact promoted bytes, committed in
17daf9b with the public origin record. Two hundred profanity elicitation answers
were graded in 25 first-valid original-scale batches. The fixed filter retained
30 pairs across 18 questions, with zero semantic refusals. Positive/negative
trait means were 38.72 / 0.12 and coherence 74.38 / 85.67. All 60 retained
trajectories were used for extraction. Preflight passed exact zero parity,
nonzero logit changes and cached hook coverage. The original parent was
permanently closed as amended_before_steering, with its old selection unrun;
the new grid below owns selection and reporting. Generation took 1,133.83 seconds (whole stage 1,139.20 seconds).
The accepted description and checks are in the
[protocol](profanity_reference_protocol.md). Shared PV sources stay frozen;
new profile provenance and a separate read-only lexical supplement are fixed
before profanity elicitation. Private raw generation records remain retained;
the committed origin hashes them. Friendly-swearing reference scores, broader
custom profanity scores and incomplete lexical counts remain distinct.

## Separate symmetric PV grid extensions

Claude and the coordinator agreed on a full nine-layer by three-strength grid
(0.5, 1, 2), with unchanged quality criteria and one untouched report. This fixes
the sequential search limitation without changing the original rudeness null.
The prospective profanity amendment ee47667 was committed while only
elicitation was running, before any steered profanity output. The parent will
hand off after extraction/preflight; its original selection will remain unrun.

The new runner passed 31 focused CPU checks and Claude's source/protocol review.
Rudeness child registration b74d659 preceded 160 fresh generations. Baseline and
eleven old candidate conditions (120 outputs) were reused with unchanged
first-valid grades; coefficients 3/4 cannot enter the grid. All 27 cells are now
complete: 16 passed quality, but the best gain was only 1.2 points (2.9 versus
1.7), below ten. The child closed selection_null without held-out generation.
The worker independently verified reuse, new outputs, grades and selection.
See the [full grid closeout](persona_pv_grid_rudeness_closeout.md).

Profanity child registration 5b60449 preceded a fresh baseline and all 27 settings
(280 generations). Generation is complete and 35 blinded grading batches are
running. Selection and any held-out work remain conditional on the fixed rules.
The new protocol is [here](persona_pv_grid_protocol.md). Combined trait steering
remains deferred; these are separate isolated studies.
