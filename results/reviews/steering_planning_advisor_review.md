# Reporting and steering planning review

Date: 2026-11-13. Advisor: existing Herdr `tunellm-advisor` Claude session;
manager: coordinating Codex. User asked to consult Claude before management
planning and delegate implementation to the Codex worker.

Claude reviewed the next-phase recommendation, then the smaller pilot, then the
concrete execution plan and operational refinements. The advisor approved:

- Original quantization/QLoRA milestone complete; GSM8K was instruction SFT.
  JSON extraction is optional backlog.
- Additive benchmark reporting and Hugging Face card updates; existing weights,
  privacy and immutable pinned releases preserved. Public GGUF reports contain
  only public-safe benchmark metadata and results. Older GSM8K scores remain
  explicitly separate from the newer evaluation protocol.
- Small written-tone activation-steering pilot, BF16 first and unchanged-setting
  NF4 transfer, with prompting as a baseline. No physics adapter or new training.
- Matched synthetic contrast pairs, grouped-disjoint extraction/development/final
  data, response-only activation extraction at fixed layers, bounded strengths,
  explicit content/length/degeneration gates and no final-driven reselection.
- AI judging disclosed as AI; randomized anonymous packets and a fresh context
  preferred. Judge identity/settings recorded only to the extent exposed.
- CPU work proceeds while unrelated GPU workloads run; no interference or GPU
  reassignment. Refusal data and execution are deferred for separate review.

Claude's specific refinements included checkable literal facts, pair-length
filtering, an objective development-only cap rule, zero steering as a valid
outcome, confidence intervals and disclosure that a weak development preference
is exploratory. The manager narrowed a proposed larger refusal/layer sweep to
the tone-first scope documented in `docs/tone_steering_protocol.md`.

This records planning approval, not completed code review, inference results,
benchmark equivalence, publication verification or improved model behavior.

## Concrete reporting payload review

Claude subsequently reviewed the staging/updater code, consolidated summary,
individual cards and four staged payloads. The advisor found no material
disclosure or publication problems and approved application after read-only
remote preflight. Checks included table/source reconciliation, paired-count
orientation, evaluated/published GGUF checksum identity, public payload privacy,
separation of the earlier GSM8K protocol and unchanged NF4 dependency pins.
The two evaluation commits referenced by the public report were already on
`origin/main`. Minor notes: historical inventories lack some byte sizes (hashes
are still checked), and the public payload does not include the complete screen
fixtures, so no independent full-screen replay claim is made.

The root then completed all four read-only preflights and documentation updates.
All post-update inventories and visibility checks passed; exact records are
`results/huggingface_documentation_update_*.json`. This is reporting verification,
not a rerun of the model benchmarks or a review of steering implementation.

## Pre-generation fixture review

Claude reviewed the first synthetic fixture and identified two material issues:
name/day/count coverage missed wrong actions/objects, and constrained labels or
slips left little natural room for warmth. Before any generation, the worker
revised all families into short messages, added event semantics and optional
location keys within the two-to-four-fact limit, added task completion to AI
judgments/gates and varied objects within each family. Claude approved the
revised fixture. Twelve final families and six development families remain a
small number of independent groups despite 60/30 individual prompts.

Literal exact-phrase checks can miss valid paraphrases. They are diagnostic for
development/final content gates, but do filter extraction eligibility and can
favor lexical copying; disclose that bias and per-family attrition. Some minor
scenario details are not separate fact keys. The AI rubric still considers the
whole scenario, without making those omissions mechanically measurable.

## GPU scheduling review

Physical GPU 1 has unrelated active workloads; no project inference started.
The manager asked the user whether to wait for GPU 1 or explicitly authorize
sharing GPU 0. Claude reviewed the conditional alternative: comparisons must
remain on one physical device, device UUID/mapping must be recorded, zero-hook
checks must run there, and timing belongs to that device. No answer or elapsed
time authorizes a device switch. All existing jobs remain untouched.
The user subsequently explicitly selected GPU 0 if memory and utilization allow.
The protocol/config now target that conditional window, with the same-device
checks above. The initial observation was 487 MiB used and 0% utilization on
GPU 0; this is not a permanent availability reservation.

## Initial steering implementation review

Claude reviewed the first staged runner and required round-robin extraction
ordering to avoid omitting late-listed domains, real cache-aware hook tests
(zero multiplier bypasses hooks), isolated fresh-context judgment batches and
actual physical-device records. The root also requested judgment/source binding
and final-lock hash consistency checks. These are pre-generation corrections;
no final-test outputs exist and no settings were selected from them.

The advisor approved tool-free headless Claude judging with small anonymous
batches, complete-ID/schema validation, first-valid bounded retries, a context
isolation audit and a fixed swapped-order diagnostic sample. A nonzero smoke
probe must demonstrate hidden/logit changes and correct hook positions; greedy
token changes are recorded but not required and never trigger an effect search.
Implementation and actual runtime validation remain separate acceptance steps.
