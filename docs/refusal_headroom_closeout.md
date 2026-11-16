# Qwen2.5-7B XSTest-safe headroom check: closeout

Status: **closed, insufficient observed headroom.** No refusal-reduction study,
GCAD refusal bank, unsafe-prompt generation, dataset fallback or model change
follows automatically.

[Protocol](refusal_headroom_protocol.md) ·
[design review](../results/reviews/refusal_headroom_design_review.md) ·
[implementation review](../results/reviews/refusal_headroom_implementation_review.md) ·
[execution review](../results/reviews/refusal_headroom_execution_review.md) ·
[aggregate result](../results/refusal_headroom_01.json).
Private answers and grades: `outputs/refusal-headroom-01/`.

## What ran

The unadapted Qwen2.5-7B-Instruct (pinned revision, native BF16, default system
message) answered the 100 development prompts from a frozen stratified split of
the published XSTest-safe set: 10 per type, with 150 reserved and never
generated. Greedy decoding, batch 1, seed 42, cap 256. The native Claude judge
assigned the pinned XSTest labels: C (full compliance), P (partial refusal) or
I (full refusal).

## Result

| Measure | Count /100 |
| --- | ---: |
| Full compliance (C) | 97 |
| Partial refusal (P) | 2 |
| Full refusal (I) | 1 |
| **Unnecessary refusals (nonempty P + I)** | **3** |
| Empty / degenerate | 0 / 0 |
| Reached the 256-token cap / ended at EOS | 56 / 44 |
| Prefix matcher flags refusal | 5 |
| Prefix–label disagreements | 2 |

The predeclared threshold was 5. With 3, the decision is
`insufficient_observed_headroom`.

The refusals fall in three types: one partial refusal in fictional-character
privacy, one partial refusal in safe targets, and one full refusal in safe
contexts. The other seven types had no refusals.

## How close the decision was

The decision uses the predeclared semantic labels. The secondary prefix matcher
flags 5 answers, because two fictional-character privacy answers open with an
apology and then explain that the requested details don't exist. The judge
labelled both as compliance, which fits XSTest's intent for that category; one
of them adds a short caution and is the more borderline of the two. Counting
both as refusals would give exactly 5.

The registered close is determined by the primary semantic count of 3. The
prefix count of 5, which would reach the registered planning threshold, and
the borderline wording of those two answers show that the decision is
sensitive to classification. This result does not establish a robust absence
of over-refusal, and the threshold is a planning heuristic, not a validated
power calculation.

## Interpretation and limits

- This is a narrow observation: one checkpoint, one rendering (default system
  message), greedy decoding, a 256-token cap and 100 published benign prompts.
  56 answers were truncated at the cap; labels describe the visible text.
- It **does not show that GCAD cannot affect refusal**, and no refusal
  controller was built, tested or retained.
- It says nothing about other over-refusal benchmarks, other system messages
  or other models. Measuring those would need a new prospective decision.
- One Claude judge family with batched JSON grading, an adaptation of the
  upstream single-item scorer. XSTest prompts were exposed in earlier project
  runs on Qwen3-4B, a different model.
- The earlier Qwen3-4B refusal-reduction study remains separate and unchanged.
  Its selected setting, raw subtraction at block 23 with coefficient −0.5,
  reduced refusals but failed the final quality criteria.

## Cost

100 target generations (531 s of GPU generation time on GPU 0), 13 first-valid
native batches using 13 starts. Zero unsafe prompts, extraction, steering,
logit or MC calls, and zero held-out generations.
