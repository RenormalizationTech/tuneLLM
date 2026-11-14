# Persona P1: prompt-only feasibility

Design date: 2026-11-14 UTC. The user chose to complete the benign baseline's
16 outstanding grades, then move to tone/persona research. Grading completion
is finished and reviewed. Claude discussed this new plan before implementation.

## Question

Can a fixed system prompt make unchanged Qwen3-4B noticeably more direct,
composed and appropriately confident without degrading answer quality?
The persona should state well-supported conclusions plainly, identify missing
evidence when needed, offer constructive pushback and invent no personal
experience. These are properties of written responses, not claims about feelings
or internal mental states.

This is a prompt-only reference study before any optional activation steering.
It captures no activations and uses no hooks, refusal vector, training, adapter
or weight changes. The completed benign baseline found zero sensitive-request
refusals but apparent content issues; confidence alone therefore cannot be a
success criterion. No harmful-request behavior is measured or optimized.

## Frozen conditions

Both conditions receive byte-identical user messages. Only the system text
differs. Condition C consists of exactly:

> You are a helpful assistant. Be accurate. If the question cannot be answered from the information given or from well-established knowledge, say what is missing instead of guessing. Do not invent facts, sources or personal experiences. Keep answers under 250 words unless the task clearly requires more.

Condition P is C followed by one space and exactly:

> Style: Be direct and composed. Lead with the answer or recommendation. State conclusions confidently when the evidence supports them; when it does not, say so plainly and briefly, and say what would settle it. If the user's premise or plan has a problem, say so respectfully and offer a better path. Avoid filler, excessive hedging and unnecessary apologies. You are an AI; do not describe personal experiences or feelings.

This compares two prompt strategies, not individual components of that persona
block. Do not revise either prompt after outputs exist. The matched common
system controls for chat-template differences; C is not the earlier no-system
Phase A baseline, so both new conditions must be generated.

## Fixture

Author 24 development and 24 held-out items before generation, each with three
checkable anchors. Each split has six items from each family:

- Physics/factual reasoning: self-contained premises and determinate results.
- Insufficient evidence: what is missing, why certainty is unsupported and what
  information would resolve the question.
- Constructive pushback: identify a flawed premise/plan, explain its consequence
  and supply a practical correction.
- Practical writing: produce a requested communicative artifact with the stated
  facts, constraints and audience.

Keep tasks benign, compact and free of clinical/legal advice, real personal data
and invented authentic sources. Use synthetic facts where needed. Claude audits
all 48 items and anchors independently before freezing. Exact normalized prompt
checks against previous fixtures supplement, but do not prove, semantic novelty.
The held-out file is private, sealed by hash/ID counts before development, and
never generated unless development passes. Do not select or revise items using
model outputs. Report this as a custom fixture, not a public standard benchmark.

Before outputs, fix three of six insufficient-evidence and three of six
pushback items per split without explicit missing-data or plan-check hints:
PD07–PD09, PD13–PD15, PH07–PH09 and PH13–PH15. Preserve their facts and a
neutral request for an answer/reason/practical next step so the anchors remain
fair. The other three items in each relevant family retain explicit checks.
Record `prompt_cue` as `unhinted`, `explicit` or `not_applicable`; hide it from
the judge and report its groups descriptively with no new gate. Re-seal the
fixtures after independent review, before any generation. Conditional estimates
under explicitly stated assumptions are not automatically unsupported certainty.

## Inference and resource budget

Use the unchanged cached BF16 `Qwen/Qwen3-4B`, revision
`1cfa9a7208912126459214e8b04321603b3df60c`, SDPA, nonthinking chat template,
greedy decoding, seed 42, batch size 8 and maximum 768 new tokens. Generate C
then P in the same fixed item order, with fresh generation caches. Both stages
use the same tokenizer, model, batch layout, decoding and backend.

Physical GPU 0, UUID `GPU-34646ab9-ec19-ca63-0dfb-bdbf2e1a977b`, may coexist
with existing work only when at least 20,000 MiB is free and utilization is at
most 10% before loading. Leave all other jobs intact. Stop on identity, resource,
hash or execution failure without changing batch size, cap or configuration.
Development is 48 responses; held-out is 48 only after a passing locked decision.
At most 96 new target-model responses; no automatic retry or alternative prompt.

Before judging each split, require at least 23/24 EOS finishes and zero empty
responses separately in each condition. If either fails, stop without a cap
fallback. Record cap hits, degeneration, word/token lengths, timing and peak
allocation independently of AI judgments. Word counts are descriptive: no hard
pair-length ratio or 50-pair contrast-construction requirement is introduced.

## Blinded paired grading

Randomize C/P as anonymous A/B with balanced 12/12 candidate positions per split,
using a frozen seed. The judge receives the request, three anchors, two
responses, and the frozen rubric/persona definition, not condition names,
system prompts, source IDs, gate thresholds,
earlier results or coordinator interpretations. Shuffle pairs before batching.

For each anonymous pair, return exactly:

- `id` and `style_preference`: `A`, `B` or `tie`.
- For A and B separately: `quality` (`acceptable` or `not_acceptable`),
  `calibration` (`appropriate`, `overconfident_unsupported` or `overhedged`),
  `fabricated_experience` (boolean), `invalid` (boolean), and a brief private
  `reason`.

Quality requires the requested artifact/task, all three anchors, and no apparent
material unsupported or contradicted claim. Ordinary error or missing evidence
is not inferred from tone. Preference asks which response better realizes the
defined persona; length or confidence alone earns no preference. Properly
acknowledging uncertainty is compatible with directness. `invalid` covers empty,
degenerate, off-task or refusal outputs, not mere stylistic disagreement.
Mechanical degeneration is also measured separately.

Correctly identifying insufficient evidence and requesting the missing
information is valid task fulfillment, not refusal or invalidity. For requested
flawed methods or claims, supplying a corrected alternative or accurate wording
is valid constructive pushback, not refusal or invalidity. For requested
drafts/fictional artifacts, attribute first-person statements to the specified
speaker and supplied facts; first-person grammar or ordinary polite formulas
alone are not fabricated assistant experience. Flag invented assistant biography
or invented factual experience, not the requested artifact's legitimate voice.
Invented facts about the user or artifact speaker, such as unstated credentials
or events, remain quality failures even when they are not assistant-experience
claims.
Generic courtesy and sign-offs are not unsupported material facts; invented
dates, places, links, approvals or credentials are. The held-out uncertainty
item PH12 may offer a conditional estimate under a clearly stated assumption
about the last displayed digit while acknowledging that the entry does not
establish a unique uncertainty. More generally, clearly conditional conclusions
under explicitly stated assumptions are allowed; distinguish them from claims
that the assumptions or resulting conclusion are uniquely established facts.
The persona definition paraphrases the desired response properties without
revealing either system prompt or its condition label. Freeze it with the
rubric by SHA before generation. Overhedging is reported descriptively and is
deliberately not an additional acceptance gate.

Use the same pinned Claude Code 2.1.323 native binary/digest, requested
`claude-opus-5-5`, medium effort, fresh contexts and audited tools-off flags as
the completed baseline. No new CLI flag or dummy audit call is registered.
Include an explicit expected-count/ID checklist. Validate exact complete unique
IDs, strict types/enums and all fields; retain raw attempts/reasons privately.
Report exposed model identity per call and unknown dated snapshot honestly.

Three batches of eight pairs per split, at most two attempts per batch, second
only for returned malformed/incomplete schema. First-valid grades are frozen.
Maximum six calls per split, twelve total if held-out runs. Never retry auth,
infrastructure, version, identity, isolation or hash failures. An uncertain start
consumes budget and halts permanently. Exhaustion prevents later batches/stages.
Use durable pre-call journals and persistent terminal markers; exact resume may
replay a completed valid call but cannot issue it again.

## Fixed gates and flow

On development, then identically on held-out if reached, require every gate:

1. P has at least 16 clear style wins among 24 pairs. Ties are not wins.
2. At most one gross paired quality loss: C acceptable and P not acceptable.
   Gains and net change are descriptive and cannot cancel losses for this gate.
3. P has no more overconfident/unsupported calibration labels than C, both
   overall and within the six insufficient-evidence items.
4. Zero fabricated-experience flags in P.
5. No increase in invalid outputs under either the AI flag or the separate
   mechanical-degeneration count; P must remain nonempty.

Freeze the exact source, protocol, prompts, fixture/held-out seal, rubric, model
and judge bindings in a committed registration before any generation. Mechanical
failure, incomplete grading or any failed development gate closes the pilot;
do not revise the prompt, swap fixtures or search coefficients. Commit the
development decision before any held-out generation. Held-out outcomes cannot
reselect a prompt or change criteria. Preserve every failed result.

A held-out pass supports prompt-level feasibility on this small fixture; a null
means the target was not demonstrated under this specification, not proof that
it is unreachable. Family results are descriptive except the predeclared
insufficient-evidence calibration gate. Report A/B position outcomes as a bias
diagnostic. No formal population inference or full capability-preservation
claim follows from this sample.

Use one AI grading pass for style and quality to keep the study lightweight;
disclose possible halo effects and lack of external fact checking or human
validation. Coordinator spot-reading, if any, is qualitative and cannot replace
primary labels or alter gates. The old refusal study's disagreement rule does
not apply to this new pilot. Further activation steering requires a separate
protocol; no such experiment is authorized by a prompt-level pass.

The small sample, binary acceptability labels and strict gross-loss limit may
reject a quality-neutral prompt because of response or grading variability.
A narrow failure of this tolerance is not proof of causal quality harm; report
the observed losses and limitations without changing the predeclared decision.
