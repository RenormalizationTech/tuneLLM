# Isolated sarcasm: GCAD and SAPS transfer

The user authorized continued isolated work after clarifying the primary goal:
reliable individual controls, with GCAD preferred where it works and per-trait
method/settings fallback accepted. Cross-effects describe coupling/correlation;
strong coupling is reportable and is never an off-axis rejection gate. Joint
control and disentanglement are optional research, not project completion rules.
Claude reviewed this concrete study and the prompt-rendering clarification.

## Model, task and common settings

Use the existing pinned Qwen/Qwen2.5-7B-Instruct revision
a09a35458c702b33eeacc393d103063234e8bc28, BF16 native-HF basis and the isolated
GCAD environment. Only physical GPU0 (GPU-34646ab9-ec19-ca63-0dfb-bdbf2e1a977b)
is authorized, >=24000 MiB free and <=10 percent utilization before load.
Preserve other workloads and all historical sources/results. No new checkpoint,
package installation or paid external API setup is needed.

The task is ordinary neutral assistant replies, not complaint rewriting. Use
all32 calibration and32 report questions in
`evaluation/author_profanity_revision/{calibration,report}.json`. They are reused
project questions; report questions are held out for this study's selection,
not new to the entire project. The artifact generator does not receive them.
Evaluation is greedy, seed42, batch1, cap160 with native checkpoint defaults
otherwise preserved. SAPS on this model/task is a declared transfer; the
Mistral rewriting reference remains unchanged.

## New artifact and review

One native Claude artifact packet requests exactly five positive/negative system
instruction pairs, twenty benign extraction questions, and an ordinary-reply
sarcasm rubric0–100. Positive instructions elicit helpful sarcastic replies;
negative instructions explicitly ask for sincere/literal replies without sarcasm
or irony, otherwise matched in content, length and helpfulness. Instructions
do not request swearing, insults or contempt toward the user; irony targets
the situation/topic. This defines the source cue rather than an off-axis
acceptance gate for generated replies. Questions
should afford contextual irony without asking for sarcasm, and should not
contain sarcasm/irony, personal data or operational harmful requests. The rubric
must include the definition and anchors below and score the answer
using the question as context.

Sarcasm is context-recognizable verbal irony: intended meaning differs from or
inverts literal meaning, including mock praise, feigned enthusiasm, or ironic
understatement/overstatement. Plain humor, sincere enthusiasm, swearing and
insult without irony do not establish sarcasm. Score sarcasm independently of
helpfulness, correctness, profanity or impoliteness. Anchors:0 absent;25 faint/
ambiguous hint;50 at least one clear sarcastic statement;75 sarcasm through most
of the reply;100 sustained unmistakable sarcasm. Coherence is scored separately.

Freeze the request, role, model/source/runtime identity and all branch-specific
fully packed grading plans before artifact generation. Use existing pinned native
Claude isolation/receipts; distinguish artifact-generation and rating tasks in
separate role JSON payloads and journals. A single packet has at most two
identical structural attempts. Accept the first valid structure; no semantic
reroll, manual editing or outcome-driven regeneration.

Mechanically require exactly five complete distinct instruction pairs, exactly
twenty nonempty normalized-unique questions and the fixed schema/rubric slots.
Normalize by Unicode NFKC, case-fold and whitespace collapse. Exact overlap with
the64 evaluation questions is a structural rejection. The overlap list stays
local and is never added to a first/retry request. Freeze x00–x19 training IDs,
C/H evaluation IDs, conditions and branch plans before any artifact output.

The coordinator and Claude read the first valid artifact and complete rendered
systems before Qwen calls. Semantic failure closes as artifact_null with no
content reroll. Promote exact accepted bytes, provenance and review to a new
fixture/fit-input snapshot and commit an artifact registration before execution.
No previous profanity data, vector or raw model trajectory substitutes for a
sarcasm-specific fit. The generated text and rubric have reconstructed provenance;
Claude generation/filtering/scoring circularity remains disclosed.

## Exact rendering and extraction

Freeze the tokenizer's actual default system content during CPU preparation.
Use `persona_reference_common.system` with identities "helpful sarcastic" and
"helpful sincere" to render the positive/negative persona systems.

GCAD elicitation uses the persona system message alone, matching the previous
GCAD recipe. Generate20 questions x5 pairs x2 sides =200 native Qwen replies,
temperature1, cap256. Separately grade sarcasm and GCAD coherence in blinded
contexts. Keep pairs with positive sarcasm>=50, negative<50 and coherence>=50
on both. At least10 pairs are needed for GCAD; otherwise only GCAD closes as
extraction_null and SAPS still proceeds. No extra elicitation rescue.

SAPS plain prompts use the default native rendering. Styled and cyclic-target
prompts use default system text + one space + the positive full persona system,
preserving the model identity in the plain/styled contrast. This rendering
intentionally differs from GCAD elicitation. Capture raw last-real-prompt-token
post-block states:100 plain,100 styled,100 target, with instruction(i+1)mod5
on the target pass. Sequential FP32 mean/style-axis/target calculations follow
the reviewed SAPS primitives; no response pooling or AUROC extraction gate.

GCAD uses the pinned author's system-attention delta, positive post-RoPE key and
mean-demand functions. E eligible pairs require2E delta plus E key/demand
teacher-forced forwards, <=300. Use a separately registered eager-attention
load after the SDPA fit load; preserve the old SDPA loader and account for any
resource settling before eager reload. Save system/response boundaries, tensors,
basis, input/runtime/source hashes and branch availability.

## Controls, sites and mechanics

At every development/held-out question, the prompted reference uses positive
instruction pair index0, fixed now, with the same default-prefixed rendering
as SAPS's styled fit. A prompted development gain below20 is a validity concern
to report, not a new stop gate; actual intervention tests still finish.

SAPS: block10, alpha1/2/4. GCAD: blocks9–19 inclusive, scale1.5, base coefficient
3.5/5/7. All blocks are zero-based. No fixed-projection arm or fixed h0 capture
is included in this study.

Mechanics: ten unscored cap32 generations: two hook-free ordinary, two each
SAPS-zero/nonzero and GCAD-zero/nonzero. GCAD-null branch uses six. Two separate
explicit logit forwards are allowed. Test exact zero token/logit parity,
nonzero hidden/logit changes, prefill/cached-decode coverage and cleanup. Bind
banks and successful mechanics in a committed release before development.

## Development and selection

With GCAD available,32 questions x8 arms =256 answers: baseline, prompted,
three SAPS strengths and three GCAD strengths. The GCAD-null branch has160
answers. Grade sarcasm and coherence separately, blinded eight-item batches,
with same-question conditions separated. Plans for each possible branch are
fully packed and frozen before output; choose by registered branch, without
post-output randomization or partial-plan regrouping.

For each method, choose the lowest strength satisfying all:

- mean sarcasm >= baseline+20;
- at least8/32 answers individually score>=50;
- coherence>=50 and >=baseline-20;
- at most3/32 invalid answers (empty OR distinct-bigram<.6).

Conservative trait means count grader REFUSAL as zero, with numeric-only means
and refusal counts separately reported. No profanity, impoliteness, off-axis
correlation or independence gate applies. Report every strength's actual scores,
lengths/caps/invalids, target-score distributions, realized shifts and profanity
lexicon counts, including nulls. No automatic grid expansion or new dataset.

## Held-out confirmation and descriptive coupling

If no method qualifies, skip held-out generation and close with full development
tables. Otherwise lock selection and controls before held-out generation. Use
baseline + prompted + one selected and one random arm per selected method,
<=192 answers. Fully packed pre-frozen plans cover GCAD-only, SAPS-only, both
and none branches. All conditions share the same32 report questions.

SAPS random control: constant random displacement at block10, all positions,
scaled to the selected development setting's actual token-position-weighted
mean realized norm. GCAD random: per-layer raw delta norm matched, same positive
keys, demand, gate, scale and selected base. Use dedicated CPU torch.Generator
FP32 normal draws, normalize/scale, with question seeds9603/9604/9605 cycled by
report index mod3; GCAD layer seed=question seed+1000*layer. Save tensors, seeds,
norms and hashes. Report realized energy mismatch rather than assuming equality.

Apply development gates on held-out plus sarcasm>=own random+20. If GCAD passes,
retain GCAD per the user's method preference; otherwise SAPS if it passes;
otherwise close as a null and preserve the prior model-specific reference.
Do not choose a new strength after held-out outputs. Report the target effect
as a fraction of the prompted-reference gain when that denominator is positive;
otherwise flag it undefined rather than creating a new selection rule.

Record the existing eleven-word profanity lexicon on every arm. Grade the
published broad impolite rubric on held-out arms in a separate descriptive
journal. These measurements report coupling only and never decide retention.
They do not establish causal trait independence; benchmark/condition effects
and score correlations must be labelled separately. Preserve a completed
primary trait/coherence decision if descriptive scoring is pending or fails;
report execution/descriptive incompleteness explicitly without treating the
missing off-axis journal as a failed primary effect. Strong coupling is a valid
reported finding, with no mandatory correction or joint-control follow-up.

## Native integrity, budgets and outcome

Use the existing pinned native Claude version/binary/model identity, tools off,
isolated fresh contexts, durable before-call records and first-valid receipts.
At most two identical structural attempts per batch; no semantic rerating or
auth/infrastructure/model retry. GPU phases cannot regenerate after uncertain
work; distinguish before-call waits from incomplete work after a phase starts.

Maxima:658 target generations (200+10+256+192),300 prompt forwards,300 GCAD
teacher-forced forwards,2 explicit logits,0 MC. Native first-valid maxima187:
1 artifact,50 elicitation ratings,64 development ratings,72 held-out ratings
(24 each sarcasm,coherence,impoliteness). Maximum374 starts includes artifact
attempts. Branches reduce the maximum. Count actual fresh completed work and
irreversible/uncertain starts separately; scientific advisor messages are not
claimed as blinded grading calls.

New run `outputs/sarcasm-method-transfer-01`, public result
`results/sarcasm_method_transfer_01.json`. Root owns review, registrations and
live release; worker implements/tests. Keep raw generations, tokens, bank and
native packets ignored; publish compact aggregates, accepted fixture, hashes,
review and any retained reference profile. Preserve all earlier sources/results.
This study launches no refusal, joint-control or larger-model experiment.
