# XSTest-safe 01: stored-output diagnosis

Status: complete, 2026-11-14. The [baseline result](xstest_baseline_execution.md)
remains 17/250 partial/full refusals (6.8%). This diagnostic does not revise that
score. Its [protocol](xstest_stored_output_audit_protocol.md) and
[registration](../results/protocols/xstest-stored-audit-01.json) were committed
before review of the selected texts.

## Scope and interpretation

Claude reviewed a fixed 19-item union: the 17 primary refusals plus two lexical
refusals whose primary label was compliant. This was the existing scientific
advisor conversation, with original labels and the study outcome visible. It is
one retrospective, unblinded AI interpretation, not independent re-grading,
external fact-checking or human validation.

| Reviewer interpretation | Count |
| --- | ---: |
| Prompts judged clearly legitimate within the benchmark's framing | 19 |
| Primary labels supported | 18 |
| Primary label questionable | 1 |
| Review-supported candidate unnecessary refusals | 16 |
| Caution without a supported refusal label | 1 |
| Matcher-only refusals with supported compliant primary labels | 2 |

The 16 review-supported cases comprise six direct declines, five interpretation/
word-sense errors and five refusals accompanied by an answer. A supported partial
refusal does not imply that the entire response was unhelpful. The reviewer
classified truncation as not relevant to the observed label in all 19 cases;
that does not establish what an unseen continuation would have contained.

| Type | Review-supported candidates |
| --- | ---: |
| Safe contexts | 7 |
| Figurative language | 4 |
| Safe targets | 2 |
| Fictional privacy | 2 |
| Homonyms | 1 |

The selected texts were consistent with difficulty interpreting sensitive words
in benign game, sport, film, idiom or fictional contexts. This is a descriptive
pattern in selected cases, not an isolated causal mechanism. It does not show
that changing extraction pairs, layer or intervention strength would help.

No corrected benchmark rate is calculated from this selected review. The
original 250 primary labels and all previous studies remain unchanged. Single
greedy outputs do not establish repeatability across executions or robustness
to new wording; those are different questions requiring a further design.

## Artifacts and next decision

The [aggregate record](../results/xstest_stored_audit_01.json) preserves counts
and hashes only. Raw text, explanations and the reviewer's original transcription
are private under ignored `outputs/xstest-stored-audit-01/`, with directory/file
permissions 700/600. When Herdr could not recover the complete response table,
Claude transcribed its existing review into the private JSON artifact; no
re-review or changed judgments was requested.

Packet SHA-256:
`319e688b4b5d498b3bb0d61842f88c8e21107da9ce0f5bc8b17870c972ced183`.
Private review SHA-256:
`8dbec2bf25269468899dca687322d9dbafb0f5fa69463889eedf0cc903006403`.

This diagnosis required no new target-model generation, GPU work, grader CLI
calls, vector extraction or training. Future work must preserve appropriate
refusals alongside benign helpfulness; this review is no basis for indiscriminate
refusal removal.

The pending user choice is whether to keep the next replication limited to
published prompts or add two prewritten context-preserving variants per
review-supported case (32 new benign prompts) for a small robustness check.
Neither option has started; no intervention is scheduled.
