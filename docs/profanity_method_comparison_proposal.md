# Three-method profanity comparison: original proposal

Completed: [GCAD strength 5 passed held-out confirmation](profanity_method_transfer_closeout.md)
and is retained. The proposal below records the pre-output reasoning.

User request: test the interventions that produced effects on other traits
against profanity. The fourth task omitted from the previous comparison was
rudeness/impoliteness. The user subsequently authorized all three methods and the bounded fallback
below. Claude has now reviewed and accepted the concrete
[execution protocol](profanity_method_transfer_protocol.md); no claim is made
that the methods already work on profanity.

## Candidate interventions

| Candidate | Existing evidence | Profanity adaptation required |
| --- | --- | --- |
| Fixed projection targeting | Original-Mistral sarcasm reference produced sarcasm with meaning loss | Profanity-specific unit direction and styled target projection; compute adjustment once per prompt |
| Adaptive SAPS | Original-Mistral sarcasm reference produced sarcasm with a strength/meaning trade-off | Same profanity-specific direction and target; recompute adjustment from the current activation during generation |
| GCAD | Original-Qwen impoliteness reference produced impoliteness while retaining more coherence than residual addition | New profanity-specific attention delta, positive-key and average-demand tensors |

All three are inference-time interventions, without LoRA or weight training.
Our previous raw constant-addition profanity study remains a completed negative
result. Neither the Mistral sarcasm direction nor the Qwen impoliteness tensors
can be relabelled as profanity inputs. Removing the projection onto a profanity
direction would suppress that feature; it is not the positive-control adaptation
requested here. Refusal results also had quality trade-offs and should not be
called three uniformly successful methods.

## Matched comparison to review with Claude

Prefer the same pinned Qwen2.5-7B checkpoint and evaluation task for all arms,
with a common fresh baseline. Label SAPS on Qwen/profanity as a method transfer,
not an exact reproduction of the original-Mistral sarcasm experiment. The code
worker found reusable HF SAPS helpers; runtime/hook parity must be checked.

Use matched neutral/profane content for constructing each method's appropriate
representations. Existing response-averaged features alone do not implement
the SAPS prompt-state target learner or the GCAD attention/key extraction.
Audit the available author extraction scripts and missing inputs before calls.

Freeze a small development budget for selecting strengths, followed by the
same held-out prompts for qualifying methods and baseline. Report every
candidate's actual generation behavior even if none qualifies. Evaluate semantic
profanity and direct swearing frequency separately from toxicity, relevance,
coherence, repetition and truncation. Resolve the previous universal 48-token
cap prospectively rather than changing historical results. Do not tune on the
held-out outputs or presume gains from another trait will transfer.

The execution protocol now fixes fresh Qwen extraction, ordinary assistant
replies, sites, three strengths per method, cap 160, separate trait/coherence
ratings and selection/held-out rules. The user authorized implementation of all three interventions. Reusable components passed CPU tests; the worker is implementing the fresh
runner. Root will review and register exact inputs/source/runtime before model
calls. No new questions, downloads or artifact-generation requests are needed.

## Review status

The initial requests encountered an expired Claude login, so independent CPU
implementation proceeded first. The user restored login; the existing advisor
then accepted the study and clarified its budget, thresholds, random controls
and read-only state capture. See the
[design review](../results/reviews/profanity_method_transfer_design_review.md).
The runner and study are now complete; see the closeout and retained profile.

## User-selected decision and fallback

Try fixed projection targeting, adaptive SAPS and GCAD separately. If a method
passes a prospectively fixed style/quality rule and confirms its effect on
held-out questions, retain that method for profanity. If several pass, select
using a rule frozen before outputs; avoid deciding from anecdotes.

If none works, close this bounded comparison without more automatic rescue
grids or new datasets. Retain the documented literature method as the canonical
profanity reference workflow and explicitly state that its positive effect has
not been reproduced with our reconstructed data. Missing author contrastive
data limits reproduction; it is not proof of why our tests failed. Do not
label the retained reference a functioning profanity controller.

Scientific review is complete. Execution depends on code acceptance and exact
pre-output registration. No new user decision or paid external API key is needed.

## CPU preparation completed

The worker implemented explicit-parameter intervention helpers in
`scripts/profanity_transfer_common.py`. Independent coordinator review and
all 22 focused CPU tests passed. Tests compare the actual released SAPS hooks
and exercise released GCAD hooks on a locally initialized tiny Qwen model.
See [CPU review](../results/reviews/profanity_transfer_cpu_review.md).
These implementation checks preceded the completed experiment; the later
closeout reports the confirmed GCAD result separately.
