# Benign helpfulness: grading completion record

The user chose to finish the 16 unaccepted ratings before starting tone/persona
work. This new completion was registered in `a280ac3` before any additional
grading call. The [protocol](benign_helpfulness_completion_protocol.md),
[registration](../results/protocols/benign-helpfulness-completion-01.json) and
[design review](../results/reviews/benign_helpfulness_completion_review.md)
describe the authorized scope. The original incomplete study remains a
historical record; its files are preserved.

## Preparation

Claude reviewed the plan before implementation. The dedicated worker wrote a
new helper and 17 focused CPU checks, all passing after final changes. The
coordinator reviewed source, original first-valid receipt reconstruction,
the two eight-item payloads, write boundaries, failure persistence and reporting.
No old importer or full submission was modified to make the previous run pass.

The snapshot binds 402 original inputs, 18 local files and 12 active sources.
It includes the 315 historical diagnosis inputs, the original outputs, accepted
20 ratings, both rejected attempts and the coordinator's pre-grading AI draft.
The coordinator independently verified the frozen bindings before release.
Manifest SHA-256:
`8f911528223d0617816dd348e643ea14c8e70b7138f60106d0598257eed847f6`.

The new 16 are original packet entries 20–35, split into consecutive groups of
eight. Only an explicit expected-count/ID checklist was added to the payloads.
Semantic rubric, judge/model/medium setting and audited tools-off flags are
unchanged. Claude Code 2.1.323 is resolved through the same process-local
binary pin, with digest checks before and after calls.

No GPU use, target-model generation, held-out response, activation intervention,
training or model release is part of this completion. The maximum call budget
is four; an exhausted batch or fatal error permanently stops this registration.

## Outcome

**Completion succeeded.** Both eight-item batches passed on their first attempts,
using two of the four permitted calls. All 36 grades were reconstructed from
the saved first-valid journals: 20 retained original ratings and 16 new ratings.
The original incomplete result and failed submissions remain unchanged.

The original decision function returned `insufficient_headroom`: zero counted
refusals among 24 legitimate sensitive requests, with no disagreement between
primary and coordinator refusal labels. All 36 responses ended with EOS; no
cap, empty, mechanical-degeneration or AI-unscorable flag occurred. No further
refusal experiment follows from this baseline result.

| Descriptive AI assessment | Sensitive requests | Physics tasks | Total |
| --- | ---: | ---: | ---: |
| Items | 24 | 12 | 36 |
| Partial/full refusal labels | 0 | 0 | 0 |
| Anchors satisfied | 65/72 | 36/36 | 101/108 |
| All three anchors satisfied | 18 | 12 | 30 |
| Apparent incorrect/unsupported claim flag | 9 | 2 | 11 |
| Problematic repetition flag | 0 | 0 | 0 |

Anchor coverage is not overall correctness: an answer can contain the requested
points and add an apparent error. These are contextual AI judgments without
external fact checking, not verified factual accuracy or human writing ratings.
The counts overlap and should not be subtracted to invent a correctness score.
The fixture supplies explicitly legitimate contexts and does not establish low
over-refusal on every ambiguous request or behavior on harmful requests.

See the [combined report](../results/benign_helpfulness_completed_01.json).
Every item has source provenance identifying its original/completion batch,
first-valid attempt, receipt and model identity. Post-call verification matched
all frozen inputs and the pinned executable. The 17 focused CPU checks passed;
no Qwen/GPU run, new response or held-out use was needed.

The practical conclusion is to close this baseline and move to the user-selected
benign tone/persona study. Its quality and uncertainty checks must remain
separate from style preference: sounding confident cannot count as being right.
No usable new persona or activation-control configuration has been established
by this grading completion.

## Interpretation limits

The retained 20 and new 16 use different batch contexts and grading times;
the new batches have eight items and an ID checklist. A server-side dated
model snapshot is unavailable, so drift cannot be excluded. The original
isolation audit had no executable digest; the installed version-specific native
binary is now pinned and checked, but historical byte identity remains unproven.
All grades are AI assessments, not human validation or externally verified
factual correctness. The fixture is custom and benign-only.

Claude's final read-only review accepted the report, counts, unchanged decision
function, retained-label provenance and interpretation limits. The coordinator
also checked that all original 20 per-item codes in the public combined report
match the retained labels exactly and that raw texts/private reasons are excluded.
