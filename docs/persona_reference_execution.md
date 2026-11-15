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
is 422 generations and 53 first-valid grading batches; the single conditional
fallback raises the maxima to 502 and 64. These are maxima, not completed work.

Measured elicitation cost: 921.49 seconds in generation,926.88 seconds for the
stage including loading and serialization; mean 249.305 generated tokens,
28/200 capped,8.438GB peak PyTorch allocation. The count includes every saved
generation. The original-scale packet comprises 25 blinded eight-question batches.

All 25 grading batches were first-attempt valid. The fixed filter retained 66/100
pairs spanning all 20 questions; all retained native trajectories are used
(132 teacher-forced responses). Across all 100 positive and 100 negative responses,
reference trait means were57.52 versus 1.55 and coherence79.55 versus 86.50;
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

The original SAPS adaptation closed as elicitation_null after our added initial
quality gate: prompted style 3.04,36/50 reaching author sarcasm ≥3 and grammar 4.64, but
clarity 3.46<3.5. Entailment2.44 was a known quality risk. The 100 answers andfive
first-valid calls remain unchanged; the original study ran no directions or
interventions. This was a failed combined control gate, not absence of sarcasm.

The separately registered [continuation](persona_saps_continuation_protocol.md)
completed the author-style prompt-activation method without that extra gate,
while keeping all downstream criteria. Its source passed 27 CPU checks and
Claude's review. Initial lock 4da9c59 preceded calls;1,500 prompt forwards learned
directions and targets. Ten generation/two logit probes passed exact zero parity,
nonzero delta16.625, cached coverage and cleanup. Strength lock 94e98a3 preceded
600 validation generations. Thirty first-valid batches selected fixed block 23,
alpha 1 (style 1.40); the best adaptive controller at the same site also qualified.
Higher adaptive alpha 1.4 failed entailment despite a larger style score.

Selection lockca547ac preceded 500 untouched test answers. All 25 reference batches
were first-attempt valid. Fixed steering scored 1.31/5 with 27/100 reaching the author sarcasm threshold ≥3
versus 0 baseline and random, meeting every registered reference gate. Grammar
4.77, clarity 4.04 and meaning preservation 3.44 exceeded their floors; meaning
preservation nevertheless fell from 4.47 baseline. Adaptive scored 1.16/26 of 100;
prompting 3.08/67 of 100 with meaning preservation 2.66. All 500 answers ended in EOS.
The worker independently audited extraction, mechanics, validation selection,
held-out generation and reference grades: PASS.

The separate 20-question contextual review, in 13 first-valid batches on saved
answers, found clear sarcasm 4 fixed/3 adaptive/20 prompt/0 baseline/0 random.
Fixed content preservation was 13/20 versus 18/20 baseline. This does not establish
generic assistant-persona control or remove the observed quality tradeoff.
Conditional compact capability passed: general MC11/16 and physics MC7/16
unchanged; math 8/8 both; strict instruction following 6/8 to7/8. These tiny samples
are pipeline checks, not broad retention evidence.

The continuation is complete. The final worker audit passed reference/custom
receipts, official capability scores, fresh matched captures, source/parent
bindings, exact costs and aggregate privacy, with no actionable findings. Claude
reviewed the final scientific scope and documentation. Combined cost including
reused parent work is
1,242 generations,1,500 prompt forwards,two logit forwards,64 MC and 73 native
starts. Child 68 starts comprise30 validation,25 reference and 13 custom; parent
five remain separate. See the [closeout](persona_saps_sarcasm_continuation_closeout.md)
and [aggregate](../results/persona_saps_sarcasm_continuation_01.json). The result
is exploratory sarcastic-paraphrase evidence, with no new model-size or joint
study implied.

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
(280 generations). All 35 grading batches were first-attempt valid. The child closed
selection_null: all 21 quality-eligible settings scored zero against baseline
zero. The fixed lexical diagnostic found no matches across all 280 grid outputs.
No held-out, custom or capability work was triggered. See the
[profanity closeout](persona_pv_grid_profanity_closeout.md).
The new protocol is [here](persona_pv_grid_protocol.md). Combined trait steering
remains deferred; these are separate isolated studies.
