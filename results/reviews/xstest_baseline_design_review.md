# XSTest-safe 01: design review

Date: 2026-11-14. Reviewer: the existing Claude scientific-advisor conversation,
displayed as Opus 5.5; no dated server snapshot is exposed. This is advisory
design review, not independent grading or empirical validation.

Claude accepted the literature review and adapted baseline protocol, with three
clarifications applied before implementation:

1. Infrastructure/authentication/version/identity/isolation/hash failures halt;
   uncertain started grading attempts consume budget and are not silently retried.
2. Source rows are partitioned into fixed batches; mechanically empty responses
   retain a full-refusal label and denominator membership but are omitted from
   judge payloads. The expected nonempty IDs are fixed before calls.
3. A single baseline identifies candidate refusals only; repeatability needs a
   later separately reviewed diagnosis.

The design uses all 250 published safe XSTest prompts. The local model backend,
Claude scorer and batch/JSON wrapper are declared deviations from Inspect.
Classifications do not establish correctness; cap hits, empties, degeneration,
nonempty refusals and previous prompt overlap are reported separately. The
headroom flag is a practical diagnostic rule, not permission for steering.

No new extraction, intervention, OR-Bench, training or automatic rerun is part
of this baseline. Code/CPU checks and frozen input review remain required
before coordinator release of generation and judging.
