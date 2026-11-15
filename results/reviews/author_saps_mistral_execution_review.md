# Original-Mistral SAPS execution checks

The Codex worker independently audited all four completed generation arms,
read-only and without model or judge calls, after each arm's exit-integrity
receipt. All450 source IDs, source order, templates, author stripping and
complete token trajectories passed for each arm. Two BOS tokens and final
sampled tokens were retained. FP16/RMSPre basis fingerprints, frozen source
and model hashes, and the committed direction/mechanics bindings matched.

| Arm | Answers | EOS | Cap | Empty |
| --- | ---: | ---: | ---: | ---: |
| SAPS0 baseline | 450 | 450 | 0 | 0 |
| Fixed1 | 450 | 444 | 6 | 0 |
| SAPS1 | 450 | 448 | 2 | 0 |
| SAPS1.4 | 450 | 428 | 22 | 0 |

The fixed arm's saved FP32 h0 tensor has shape450×32×4096. All450 constants
match the source's separate item() operands and Python subtraction exactly;
176 differ from prematurely performing the subtraction in FP32. Controller
site/strength, sampled finite prefill/decode changes, cache-call counts and
hook cleanup passed. No active arm or behavior grades were inspected for
these audits, and no settings changed after target output.

Budgets reconcile:1810 target generations including ten mechanics probes,
1500 training-prompt forwards,450 fixed-arm prompt captures, four separate
mechanics captures, two logit forwards and zero multiple-choice items.
These are execution/lineage checks, not evidence of successful style control.
Behavioral interpretation follows the separately accepted blinded grading
stream and its final arithmetic review.

The final stream completed90 first-valid batches from90 starts. The coordinator
and Claude independently recomputed means, threshold counts and paired
comparisons through all1800 unique arm/sentence mappings. Both capped answers
scoring above3 were complete rather than truncated; no ratings were changed.
Interpretation is accepted with style/faithfulness, task-compliance, co-rating
and released-log-lineage limitations in the closeout.
