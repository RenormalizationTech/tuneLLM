# Bounded Persona Vectors rudeness study

The registered Qwen3-4B search found no quality-eligible setting with the
required impoliteness gain. It closed as `selection_null` before report-set
generation. This establishes a completed method adaptation and a negative
development result, not a validated rudeness control or impossibility result.

The source method is [Persona Vectors](https://arxiv.org/html/2507.21509v1),
using unchanged author `impolite` instructions, questions and rubrics. Native HF,
BF16/nonthinking Qwen3-4B, greedy cap 512, reduced rollouts, raw native-token
pooling and structured Claude grading are explicit adaptations in the
[frozen protocol](persona_reference_protocol.md).

Prompted elicitation retained 66 of 100 pairs across all 20 questions. Extraction
used all 132 native trajectories. Saved raw response-mean vectors, intermediate
block indexing, exact zero-token parity and nonzero/cache hook checks passed.
These checks establish implementation behavior, not semantic effectiveness.

The nine-layer screen at coefficient 2 selected author layer 8 / decoder block 7
among quality-eligible sites. Its subsequent strength screen gave:

| Setting at block 7 | Reference trait mean | Coherence mean | Quality eligible |
| --- | ---: | ---: | --- |
| Baseline | 1.7 | 83.8 | Reference |
| 0.5 | 1.5 | 78.2 | Yes |
| 1 | 1.7 | 81.4 | Yes |
| 2 | 2.2 | 79.1 | Yes |
| 3 | 2.1 | 71.9 | No |
| 4 | 5.0 | 51.4 | No |

The best eligible gain was only 0.5, below the registered ten-point threshold.
Other coefficient-2 sites increased development trait scores while coherence
collapsed: author layer 20 scored 59.7 / 28.3, and layer 28 scored 71.5 / 21.5
(trait / coherence), with six and five degenerate outputs out of ten. Without
random controls or untouched report results, these are descriptive tradeoffs;
they do not establish trait specificity or a usable weaker setting.

The two-stage search has a limitation: a coherent but ineffective shallow site
can prevent the conditional fallback, while weaker settings at deeper sites
remain unexplored. No post-hoc rescue sweep was added. All ten development
baseline answers reached 512 tokens, making the capped-count tolerance
uninformative and limiting interpretation of whole-answer scores.

No held-out baseline/selected/random/prompt comparison, custom three-axis report
or capability run was triggered. The untouched partition remains ungenerated.
Do not treat development scores as final benchmark results or equate the broad
author impoliteness construct with independently validated narrow rudeness.

Cost: 350 generations, 132 teacher-forced responses, zero MC items and 43 native
grading calls, all first-valid. GPU stages totaled 1,912.78 seconds including
loading/capture/serialization, with 1,877.27 seconds in generation. This excludes
grading, source verification, CPU development and advisor review time. Private
answers, tensors and receipts remain in the ignored run directory.

The worker independently checked masks, vectors, source/layer conventions,
zero/nonzero probes, grading receipts, selection recomputation, genuine cap flags,
totals, absence of downstream generations and aggregate privacy. Source/model
bindings passed its earlier completed-stage audit; live stages also verified
their full bindings at entry and exit. See the
[aggregate result](../results/persona_reference_rudeness_01.json) and
[execution record](persona_reference_execution.md).

Reuse the protocol and tested operations as a bounded adaptation; retain the
search and truncation limitations when designing a future study. A new
layer/strength search needs a separate scientific decision and registration.
Sarcasm and profanity proceed separately. Joint steering remains deferred.
