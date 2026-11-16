# DECCP Qwen2.5-7B baseline headroom closeout

The frozen baseline screen is **complete** and **did not meet prespecified
headroom criteria**. It recorded some grader-classified soft evasion, while explicit refusal was
rare. No fit, extraction, intervention or refusal-steered model was tested or
published. The [final reviewed interpretation](../results/reviews/deccp_headroom_result_interpretation.md)
sets out the evidence and recommendation; the
[aggregate result](../results/deccp_headroom_01.json), original grades and
[registered protocol](deccp_headroom_protocol.md) remain unchanged.

## Observed labels

| Development stratum | N | Explicit refusal | Evasion | Uncertainty | Responsive incomplete | Direct | Refusal + evasion |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| English targets | 36 | 1 | 7 | 1 | 5 | 22 | 8/36 (22.2%) |
| English genre controls | 11 | 0 | 0 | 0 | 3 | 8 | 0/11 (0%) |
| Chinese targets, descriptive only | 11 | 0 | 5 | 0 | 1 | 5 | 5/11 (45.5%) |

Explicit refusal declines the request; the grader assigned evasion when it
judged an answer to deflect rather than address the request. This boundary is
subjective: some evasion-labelled answers contain partial responsive material.
Labels were not rerated or replaced. These are distinct outcomes: seven of the eight English positives
were evasion, and all five Chinese positives were evasion. Incomplete answers,
truncation, genuine uncertainty and responsive premise correction were not
counted as evasion merely for missing detail. No factual accuracy, fact coverage,
omission or official-framing evaluation was performed. Missing facts are not
evidence of censorship under this screen.

## Fixed English gate

All three conditions were required, on fixed English denominators only:

| Condition | Observed | Result |
| --- | --- | --- |
| Target refusal + evasion >=25% | 8/36 =22.2% | Not met |
| At least12 target positives | 8 | Not met |
| Target rate >= control rate +15pp | 22.2pp gap | Met |

The absolute12 rule gives an effective33.3% threshold at36 targets; each of the
11 controls changes its rate by9.09pp. The positive gap does not override either
failed target condition. Chinese results do not enter this gate and have no
matched Chinese control set. A failed gate means the criteria were not met,
not zero refusal/evasion or proof of no measurable signal.

Under the registered rule this baseline closes without extension. The requested
refusal-steered artifact remains **incomplete**, not a successful controller or
publication candidate. The reviewed recommendation is to close this conventional7B refusal attempt
without a public refusal controller. An alternative, separately designed
Chinese evasion-focused study is pending the user choice already requested
asynchronously. It is a different objective, not an automatic continuation.
No intervention, new dataset search or larger-model execution is started here.

## Execution and calibration

- Unadapted `Qwen/Qwen2.5-7B-Instruct`, revision
  `a09a35458c702b33eeacc393d103063234e8bc28`; BF16/SDPA, native default system,
  physical GPU0 RTX A6000, B1 greedy seed42, cap512. Explicit temperature1,
  top_p1, top_k0, repetition_penalty1.05; no activation hooks or weight updates.
- All58 development responses completed. Stage elapsed603.304 seconds;
  summed generation time592.404 seconds. No held-out response was generated.
- No empty responses. Seven of47 English responses hit the cap; zero of11
  Chinese responses did. English whitespace repetition detected zero cases;
  Chinese whitespace repetition is N/A, not quality evidence.
- Calibration20/20 exact and20/20 binary, four correct anchors per label.
  Native Claude Opus5.5, medium, pinned CLI2.1.323, tools-off isolated contexts.
  Twelve first-valid batches and12 starts:3 calibration,8 primary,1 duplicate;
  no retries. Six blind duplicates agreed6/6 and did not replace primary labels.
  Agreement shows consistency, not validity or accuracy on subjective boundary
  cases; these small checks do not establish broad human validity or zero error.
- Manifest:
  `52a2594624bd989c860afcfa0627fdae949b8b4b6727f7146882cacbb4bd8e66`.
  [Initial registration](../results/protocols/deccp-headroom-01.json) and
  [calibration lock](../results/protocols/deccp-headroom-calibration-01.json)
  preserve the execution authority.

## Data, privacy and limits

The reviewed Apache-2.0 DECCP source was pinned at
`1a6d5571f0a711d3afb7d1c43f70a23e41c359a1`. Original question strings and all132
source decisions were preserved. [Fixture](../evaluation/deccp_headroom/fixture.json)
SHA256 is `7eb5a3bc09e6e0f1ff7c3ec89c2c7f4dce6fa929506672b82f9fd579610a72c2`.
All25 protected held-out rows remain ungenerated. EN/ZH counterparts share
partitions; declared topic dependencies remain. This is not an OOD-event claim.

The DECCP author selected targets/control candidates on a different model's
responses. English controls are genre comparisons, not causal matching; source
selection, geography, topic and length can affect the gap. The Tiananmen group
has17 development rows and strongly influences this small pool. Chinese11 is
small and descriptive only. Despite hidden stratum/group/source fields, topic
and language were visible to the grader; this is not topic-blind judging.
Public-data exposure and same-family AI fixture review/anchor authorship/judging
further limit generalization. No truth claim about disputed premises follows
from these responsiveness labels.

Raw outputs, token IDs and private grading reasons remain only under ignored
`outputs/deccp-headroom-01/`: `baseline/predictions.json`, `baseline/answers/`,
`native/{calibration,primary,duplicates}/` and frozen `inputs/`. The public
aggregate contains counts/codes/provenance, not those texts or token arrays.
No HF/card/upload was performed for this baseline. Phase1's
[public joint guide](https://huggingface.co/RenormalizationTech/qwen2.5-7b-gcad-steering-controls/blob/e1e56e6384214be873984f5fb9421ff6c61b9e85/joint_controls/GUIDE.md)
is separately complete and contains no refusal controller. Qwen3.8-27B/Gemma4
may be discussed; no larger-model execution precedes user discussion/decision.
