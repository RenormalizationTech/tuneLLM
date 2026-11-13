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

## Registered tone feasibility outcomes and follow-up review

Tone-01's 80 pairs had a mechanical eligibility upper bound of five; Claude
approved stopping before AI judging because it could not reach the registered
50. The advisor approved one symmetric two-to-three-sentence construction A,
with one conditional neutral-to-warm rewrite fallback B, keeping all original
eligibility gates and final inputs out of calibration. New configurations,
addenda and registrations preserved every original artifact.

A's upper bound was 31, triggering B. Claude reviewed and accepted B's exact
prompt, input/source hashes, same-device requirement, unchanged development/final
systems and independent implementation. Word count is a construction instruction
with reported compliance; the token window remains the hard length gate.
Five B CPU tests and actual offline input validation passed before generation.
B's upper bound was 46, so the registered tone construction stopped. No AI
experiment ratings, vectors, development selection or final generation occurred.
Reports: `results/tone_01_feasibility.json`, `results/tone_02a_feasibility.json`
and `results/tone_02b_feasibility.json`.

## Independent judge helper review and final acceptance

The first dummy call used the requested model and reported no project/file/
memory context, but emitted a JSON fence. Its failed parser record was retained;
an offline, deterministic reparse was separately recorded without another call.
Claude accepted the normalization rule but required stronger invocation isolation
and a new final-wrapper dummy audit: remove all inherited `GIT_*` variables,
prove the neutral directory is outside Git, bind the audit to the actual calling
code, and resume interrupted batches without re-judging saved valid attempts.
Those changes are independent of the frozen generation sources. No actual
experiment judging is authorized before their final review and registration.

The final helper at SHA256
`63361396d908d219d2886a35d3bdb98d7a783e5b209d804e5aa791c45da7d4d5`
passed 27 helper tests (75 combined relevant CPU checks). A new dummy audit
proved exit code 128 outside Git, a clean Git environment, matching model ID
and no reported project/file/memory context. Claude reviewed the actual source,
resume tests and audit 02 and accepted the reusable foundation, closing all
three blockers. The sanitized record is `results/judge_helper_validation.json`.
Raw calls remain ignored/private. A new refusal rubric/validator needs its own
binding and fresh audit before real refusal ratings. An uncertain in-flight call
with no saved response stops judging; recovery cannot silently reissue it.

## Next phase review

Claude recommended moving to refusal research rather than another tone retry.
A smaller-set post-hoc tone study remains optional backlog. The manager reduced
the proposed refusal scope from about 2,000 generated/judged responses to about
400: one fixed layer, three positive strengths, small type-disjoint XSTest
subsets and prompt-side extraction without harmful completions. Claude approved
the smaller mechanism scope with primary AI refusal labels, string matching as
a diagnostic, explicit contrast-type mapping, token-length matching and
correlational projection diagnostics. Dataset counts, immutable source pins and
licenses must be verified before a definitive experimental registration; no
refusal model run is registered by this review record.
