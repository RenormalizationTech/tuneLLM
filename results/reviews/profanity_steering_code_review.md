# Isolated profanity pilot: code readiness review

The coordinating Codex accepted the worker's final implementation for CPU
preparation after 29 focused CPU checks passed. Tests used the cached tokenizer,
a tiny random CPU Qwen3 and mocked grading; no pilot preparation, model/GPU/API
call or download occurred in implementation.

Claude delivered a scientific/code review of the method and found one extra
eligibility criterion: nonfactual correctness was being required for extraction
items. The worker removed it and tested that false nonfactual correctness cannot
cause attrition. Content fulfillment still checks references. Claude confirmed
the extraction, sites, strengths, selection/gate arithmetic, constrained batches,
budgets and aggregate privacy against the design. Its final revised fixture
review was separately unavailable; fixture acceptance belongs to the coordinator.

Coordinator-requested corrections were implemented and tested before freezing:
reconstruct judging payloads from frozen plans and source answers; recompute the
eligible subset from accepted ratings; validate model/revision, manifest and input
hashes in safetensors and companion metadata; make fatal stop markers override a
complete/pass claim; and reject resume without an existing phase journal.
Each steered condition records real hook counts/hidden deltas, with immediate-EOS
decode accounting handled explicitly.

The tests cover fixture/diversity/position quotas, real tokenizer boundaries,
constrained anonymous plan coverage, eligibility thresholds, FP32 pooling and
scale units, deterministic random controls, exact serialization, actual cached
zero-vector hooks and exception cleanup, selection/tie-breaks, final/random and
strict-math gates, all 31 grading batches at the 62-call ceiling, first-valid
resume/fatal/uncertain behavior, payload/metadata tampering, final lock drift and
public export privacy. Existing sources remain unchanged.

The initial registration must pin final source/tests and the prepared manifest.
Eligibility must pass before extraction; vector/strength hashes are committed
before development, and selection is locked/committed before confirmation.
GPU and network execution are coordinator-owned and conditional on those checks.
