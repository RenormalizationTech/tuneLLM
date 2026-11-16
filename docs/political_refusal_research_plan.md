# Political-information steering and the next model stages

User-selected sequence, 2026-11-16; discussed with the Claude advisor.
This is a planning record, not an experiment registration. The proposal below
was superseded after the [completed 7B baseline screen](political_refusal_headroom_status.md):
there were zero explicit refusals on 32 sensitive-topic items. The user chose
to proceed to 4B and keep conventional refusal interventions. The available
coverage branch was not executed; no 7B steering result is claimed. Earlier
protocols, measured results and null decisions stay unchanged.

The next implementation plan is [the fixed 4B transfer](qwen3_4b_fixed_transfer_plan.md).
Its new style work uses GCAD; conventional refusal remains a separate track
with its historical quality limitations. The 7B proposal below is retained as
planning history, not an instruction to launch its unexecuted extension.

## Current evidence

Qwen2.5-7B-Instruct has positive GCAD references for profanity, impoliteness and
sarcasm, with separate banks, settings, tasks and quality limits. The original
Qwen3-4B refusal study tested fixed subtraction and directional projection;
its selected setting had a partial effect and failed its quality rule.
The 7B XSTest study measured the unsteered baseline only. Its three semantic
refusals out of 100 benign prompts do not settle political-information access
or establish a GCAD refusal result.

## New 7B study

Test the pinned, unadapted BF16 Qwen2.5-7B on benign political and historical
information requests. Compare baseline and a prompted reference with fresh
GCAD, fixed residual subtraction and full directional-ablation interventions.
Use the same questions, model, decoding and scoring for the three methods;
report unsuccessful settings as well as successes. Prefer GCAD if it meets its
registered criteria, otherwise retain a supported alternative. A failed single
setting is not a universal negative result for an intervention family.

The [source audit](../results/reviews/political_refusal_source_audit_01.md)
precedes all model outputs. The proposed corpus uses licensed published
[DECCP](https://github.com/AUGMXNT/deccp) contrast questions and the released
[routing benchmark](https://github.com/gregfrank/routing-is-learned) questions.
Exact rows, semantic groups, exclusions, compact cited factual references and
Chinese translations are being reviewed. English/Chinese variants of the same
question stay together in one partition. Translations authored here will be
labelled as adaptations. No unavailable or unlicensed dataset is presented as
a public project release.

Measure full refusal, partial refusal, evasive non-answer, substantive coverage
of the frozen references, and contradictions separately. Legitimate uncertainty
and corrections of false premises can be valid answers. Reference facts are
shown to the grader, never supplied to the generating model. Framing is
descriptive; no political position is an accuracy requirement. Fewer refusal
phrases alone cannot establish improved information access.

The distinction is motivated by work on
[factual elicitation](https://arxiv.org/abs/2603.05494) and
[political routing](https://arxiv.org/abs/2603.18280).
Our study will be a method transfer with its own reference and translation
limitations, not an exact reproduction of those results. Model nationality
does not determine an individual checkpoint's behavior; any later comparison
uses matched prompts and reports language effects.

Freeze data, rules, call budgets and branch plans before generation. Development
headroom is assessed first; the prompted arm is a reference, not a veto on
whether steering can work. Extraction uses matched questions with answering
and refusing/evasive instruction contexts. GCAD gets a fresh attention-specific
bank; it does not reuse the old residual direction. Select settings using only
development data, then compare against matched random controls on held-out
questions. No new harmful-request experiment or general boundary-preservation
claim is included in this political-information study.

## Fixed-workflow Qwen3-4B control

After the 7B study, freeze each retained method's procedure, data, decoding,
selection rules and transfer settings. Adapt the implementation to Qwen3's
actual head dimensions and Q/K normalization, validate hooks and cached decode,
and extract model-specific banks. Vectors cannot be copied between checkpoints.
Run a small fixed-setting control without a new 4B tuning campaign.

Report the actual result for each model, including any 4B null or failure. A
7B success and 4B failure would establish a difference under these settings;
family, architecture, training and size all differ, so size alone is not a
causal explanation. No outcome is assumed in advance.

## Publication and larger models

Publish code, benchmark tables, limitations and model cards to GitHub and
RenormalizationTech on Hugging Face. For dynamic GCAD, a usable release needs
the bank, exact base-model reference, configuration and inference code; a vector
file alone is not a statically merged steered model. Release packaging must
state which base weights and runtime it requires and preserve source licences.

Then extend the established workflow to
[Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) and
[Gemma 4](https://ai.google.dev/gemma/docs/core). Their exact checkpoints,
memory plans and architecture adapters are separate later steps. Qwen3.8 uses
a hybrid linear/full-attention architecture, so the current Qwen2 attention
hooks are not a drop-in implementation. No large-model download or experiment
is launched by this planning record. Simultaneous trait control remains optional.
