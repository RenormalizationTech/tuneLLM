# Run 03: diagnosing the stored quality regressions

The read-only diagnosis is complete. **The observations do not support
treating a larger token cap as a sufficient fix.** Apparent content errors,
repetition and drift were also present. They do not identify an optimal block,
coefficient or replacement vector.

This was a post-hoc, deliberately unblinded reading of selected existing output
pairs. It involved no new Qwen generation, GPU work, external fact checking or
benchmark re-grading. Original labels, selection, model outputs and the failed
final-quality result remain unchanged.

## What was reviewed

| Membership | Selection | Pairs |
| --- | --- | ---: |
| A | Safe usefulness changed: nine losses, three gains | 12 |
| B | Refusal changed to nonrefusal: six new valid answers, one invalid | 7 |
| C | Final candidate degeneration/unscorable flags, overlapping A/B | 3 |
| Union | Unique paired prompts | 19 |

The safe unscorable flag overlaps a mechanical flag. Membership rows must not
be added as independent observations. The three mechanical flags are two safe
outputs and one source-unsafe output.

The worker verified the original run, score recomputation, prediction/lock
bindings and 315 input hashes. It prepared private paired texts and public
count-only statistics. Claude reviewed the texts under the
[fixed diagnostic protocol](refusal_output_diagnosis_protocol.md) and
[rubric](../evaluation/refusal/output_diagnosis_rubric.json). The coordinator
spot-checked four safe pairs and reviewed the aggregation.

The reviewer was the Claude scientific-advisory conversation, displayed as
Opus 5.5 / `claude-opus-5-5`, with no exposed dated snapshot. This is a
context-aware AI reading, not a human judgment or an independent blinded
replication. The reviewer knew the conditions, frozen labels and study result.

## The nine safe losses

All nine candidates hit the 320-token cap. That association was the reason
to inspect content rather than infer causation from the cap flag.

| Candidate code | Meaning | Count |
| --- | --- | ---: |
| T1 | Main requested answer already present before cutoff | 7 |
| T2 | Cutoff before the main requested answer | 1 |
| T? | Main-answer delivery cannot be established | 1 |

Of the seven T1 cases, four had an uncapped baseline and three had a capped
baseline. The T2 and T? cases also had capped baselines. Thus five of the nine
baseline answers were capped yet retained their original useful/correct label.

| Reviewer's primary qualitative explanation | Count |
| --- | ---: |
| Apparent task/content error | 3 |
| Repetition or drift | 3 |
| Mixed causes | 2 |
| Unclear | 1 |
| Truncation alone | 0 |

These are qualitative explanations of selected pairs, not new usefulness
labels. Error impressions were not externally verified.

Two candidates were coded as strong loops (R3), one as clear drift/repetition
(R2), three as padding/restatement (R1), and three as having no problematic
repetition evident (R0). This agrees with the existence of the mechanical
flags without equating the heuristic with a complete semantic assessment.

Examples of the observed patterns, described without publishing the raw text:

- A candidate substituted speculative framing for a more direct treatment of
  the prompt's premise.
- A candidate repeated a generic performance-advice template before reaching
  the cap.
- A candidate gave the requested definition early, then introduced an apparent
  task-relevant contradiction during elaboration.

The cutoff cannot account for an error or loop that is already visible earlier
in the response. Conversely, the audit cannot tell us what an uncapped
continuation would have produced. The evidence therefore weakens a
truncation-only explanation without proving a single cause of the quality loss.

## The three safe gains

The reviewer associated one gain with the baseline stopping before its key
point, and two with apparent baseline content errors. One improved candidate
was still capped.

A cap hit therefore does not automatically imply uselessness. Content changed
between conditions; these examples do **not** establish inconsistent grading
or label noise. There are no stored original grader explanations to inspect.

## The seven refusal-to-nonrefusal transitions

The original result remains six new valid full-compliance answers plus one
mechanically invalid response. The retrospective review does not change those
counts.

Among the six new valid candidates, the reviewer described four as on topic
and two as partly on topic; three had no refusal structure and three combined
a disclaimer with an answer. Five were capped and one finished within budget.
The additional invalid candidate was capped and repetitively structured.

The readings included benign reinterpretation, redirection and unverified
specific details. These observations show why a nonrefusal/full-compliance
label should not be treated as verified task fulfillment, factual correctness
or operational effectiveness. None of those stronger properties was measured
for this source class.

“Completion” in the retrospective annotations means whether the response
itself finished. A completed refusal can therefore be coded complete; that
does not mean the request was fulfilled. The T code separately describes
whether the main answer had appeared before a token-limit cutoff.

## What we should change next

1. **Improve the quality measurement.** Keep willingness, task fulfillment,
   factual/grounded content, repetition and cutoff status separate. Use a
   development screen on which the baseline can usually finish. Record brief
   structured grading reasons in a future study; none can be recovered for
   run 03.
2. **Make the next intervention test focused.** A single gentler fixed shift at
   block 23, with a common adequate token budget for baseline and candidate,
   is a reasonable hypothesis to test. A better-matched contrast direction is
   another hypothesis if topic/style changes persist. Normalizing a vector
   alone is only a reparameterization unless the actual displacement changes.
3. **Defer a block search.** The diagnosis identifies no specific block as the
   cause. Changing block, coefficient, vector and scoring simultaneously would
   obscure which change helped.
4. **Use fresh held-out prompts.** This final set has now been used both for
   evaluation and diagnosis. It cannot serve as an unseen test for the next
   configuration. No new experiment or sweep is registered by this audit.

This is a proposed order of work, not a claim that a smaller coefficient,
larger cap or cleaner contrast will solve the problem. The old quality failure
is preserved.

## Limits and reproducibility

The pairs were selected because outcomes changed or flags appeared. Their
codes cannot estimate the frequency of these patterns across all prompts.
This is one unblinded AI review, with a small selected sample, no external fact
checking, no original grader explanations and no counterfactual longer output.
Counts are descriptive; no new rates, intervals, significance tests or
benchmark scores were produced.

The initial annotation JSON placed B structure fields at pair level. A
follow-up moved them into the baseline/candidate objects required by the
template and clarified response completion and reviewer-call terminology.
The initial private version was retained. The coordinator verified that all
T/E/R/completion/confidence codes, evidence notes and A explanations stayed
unchanged. This was a schema clarification, not another independent review.

- Initial private review SHA-256:
  `79dd038022868ad3b990fc0531d649e4e2ffaabb3430cf58f8fedae126a9538c`.
- Final private review SHA-256:
  `e2912c2cffef25fed465ea78132c76dd0c6cee0246fc0cf3dab99a29d2b0d492`.
- Raw packet, private notes and templates:
  `outputs/refusal-diagnosis-03/`, excluded from Git.
- [Original-data statistics and packet hashes](../results/refusal_output_diagnosis_03.json).
- [Validated qualitative counts and per-pair codes](../results/refusal_output_review_03.json).
- [Packet preparation source](../scripts/diagnose_refusal_outputs.py) and
  [annotation validator/aggregator](../scripts/summarize_refusal_diagnosis.py).

Both scripts are CPU/file operations. Packet and schema bindings, complete
unique IDs, code enums, cap consistency, required fields, note/text exclusion
and the 315 original hashes were checked. The blank reviewer template was
rejected as incomplete. The original scores and final-quality decision were
not rewritten. Generic original-data counters for source-unsafe usefulness
contain the rubric's fixed false value; they are not a quality measurement.

For the actual intervention mathematics and code mapping, see
[the two-method guide](activation_steering_methods.md). For unchanged run-03
results, see [the execution record](refusal_reduction_execution.md).
