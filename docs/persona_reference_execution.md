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
flags. The search can lock an ineffective shallow site while weaker deeper
settings remain unexplored; no post-hoc rescue sweep was added.

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
execution. The initial 50 baseline and 50 prompted-control answers are saved;
five original-scale grading batches are running before direction extraction.
Generation took 112.21 seconds, whole stage 116.81 seconds. Mean tokens were
25.2 / 30.6 (baseline / prompted), with zero capped outputs. The paper is
privately archived for rubric provenance. This is literal-to-sarcastic
paraphrasing, not demonstrated generic
assistant persona transfer. See [protocol](sarcasm_reference_protocol.md).

## Profanity

New casual non-slur swearing artifacts were generated with the published PV
general template and separately registered provenance. The first native request
was structurally valid and passed all mechanical checks. Root and Claude read
all five instruction pairs, forty questions, rubric, split and matched complete
systems and accepted them without edits or reroll. Exact semantic/dataset/system
hashes were committed in ec78901 before promotion. The helper reconstructed
raw response through parse and split to exact promoted bytes, committed in
17daf9b with the public origin record. No profanity-profile Qwen outputs exist.
The accepted description and checks are in the
[protocol](profanity_reference_protocol.md). Shared PV sources stay frozen;
new profile provenance and a separate read-only lexical supplement are fixed
before profanity elicitation. Private raw generation records remain retained;
the committed origin hashes them. Friendly-swearing reference scores, broader
custom profanity scores and incomplete lexical counts remain distinct.
