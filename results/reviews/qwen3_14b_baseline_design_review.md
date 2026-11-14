# Qwen3-14B unmodified baseline: design review

The existing Claude scientific advisor supports Qwen3-14B BF16 on one A6000 as
the next model and accepted the baseline design, subject to clarifications now
applied before implementation or target generation. No assumption is made that
size causes greater stability. The model assets are pinned and independently
verified; that is preparation, not a behavioral result.

The scope is eight unscored published preflight items, an existing 48-item quality
screen, all 250 safe XSTest prompts and all 200 unsafe contrasts. Safe refusal and
appropriate-refusal boundary measurements use separate denominators and completion
statuses. This is descriptive baseline work, with no vector, adapter or tuning.

Clarifications applied before freezing:

- An incomplete unsafe subset does not invalidate a completed safe subset; no
  partial rate is produced for either incomplete subset.
- Successful-response judge declines consume the same structural-attempt budget;
  payloads/definitions cannot be reframed. Service/policy/infrastructure errors
  halt without retry. Raw unsafe outputs stay private.
- Preflight and generation use one model load with phase resource records. Short
  preflight checks do not validate quality likelihood scoring, quality prompting
  or worst-case long-output memory. Any later failure stops without fallback.
- All fixtures and sources are frozen before preflight; the completed 4B replay
  is a prerequisite. No additional 4B runs are planned.

The worker confirmed local data and loader/scorer feasibility. Exact code, CPU
checks and prepared manifest must pass review and be committed before the
coordinator releases generation and subsequent isolated grading.
