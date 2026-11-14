# Benign helpfulness: grading-only completion

Registered design: 2026-11-14 UTC (2026-11-13 local). The user explicitly chose
to finish the 16 unaccepted ratings before moving to tone/persona research.
Claude discussed and accepted the bounded plan before worker implementation.
This is a new completion registration, not an unrecorded third attempt under
the exhausted original grading budget.

## Fixed inputs and scope

Preserve the original Phase A protocol, source, rubric, fixture, raw 36 model
responses, packet/keys, grade journals, failure logs, original incomplete result
and execution record. Keep the 20 first-valid ratings from original batch 0
byte-identical. The two incomplete batch-1 submissions remain rejected in full;
do not salvage their 15 ratings or grade only the omitted DP08 item.

Select exactly original packet entries 20 through 35, in their existing order.
Split them into two consecutive groups of eight. The source identity DP08 was
at zero-based position 12 of the original failed group; it consequently falls
at position 4 of the second completion batch. The judge sees anonymous IDs only,
never source IDs, old labels, earlier failures or coordinator observations.

No Qwen generation, GPU allocation, training, activation intervention, candidate
prompt or held-out use occurs in this completion. Only the existing 16 responses
without accepted primary labels are graded. There is no new task selection.

## Grading configuration and budget

Use the unchanged Phase A semantic rubric, requested `claude-opus-5-5`, medium
effort, fresh contexts and the same audited tools-off CLI flags. Pin the installed
Claude Code 2.1.323 native binary, SHA-256
`604d29a6869409106b6a1110bdadcc8b72e8654a5a865d23258f938a696e348e`.
Version checking and launch must resolve that same binary. Record its digest
before and after calls; retain the original audit-time executable-digest limit.
Record actual exposed model identity for every call; a dated server snapshot
is still unavailable.

Add only a structural count/ID checklist to each payload: exactly eight named
anonymous IDs and exactly eight complete records. The semantic rubric and
rating fields stay unchanged. Do not add `--json-schema` or change the CLI flags
in this registration. Local validation still requires an exact ID set, no
duplicates/extra IDs, all required fields and strict enum/boolean types.

Maximum: two fresh-context batches, at most two remote attempts per batch,
four calls total. Retry only a returned malformed/incomplete schema. Freeze the
first valid submission. No feedback about desired scores enters a retry.
Authentication, infrastructure, version, identity, isolation or hash failure
halts permanently for this registration. If a batch exhausts its two attempts, permanently close this
completion; do not execute the next batch or create another fallback. Uncertain
started attempts count against the budget and halt this completion; they cannot
be silently reissued or continued under this registration. Existing first-valid labels stay
frozen even if a later stage fails.

## Implementation and preflight

The dedicated worker writes a new helper and private output directory; the
coordinator owns this protocol and the public registration. Neither the old
importer nor its files may be changed to manufacture an accepted original full
submission. A separate combined report records the completion explicitly.

Before calls, verify and freeze hashes of old outputs/packet/rubric, first-valid
batch provenance, failed submissions, coordinator draft, original result and
the new helper/protocol. Validate the original 20 ratings against the original
payload, model identity, start journal, receipts and input binding. New grading
uses the unchanged isolation/journal foundation and a committed registration.

CPU checks must reject missing/extra/duplicate IDs, source tampering, writes into
the old run, and any additional call after exhaustion. Journals record attempts
before invocation and exact-bound resume must not repeat a completed valid
call. Synthetic tests cannot be imported as experimental grades. No dummy audit
call is registered while the existing version/flags/wrapper audit still matches.

## Combined reporting and stopping decision

Keep the original incomplete result unchanged and cite its SHA-256. Write a new
completion result with coverage, complete per-item grade provenance and the
hashes of both sources: 20 original first-valid ratings plus 16 completion
ratings. Reproduce the accepted labels from saved attempts before reporting.
Store reasons privately; publish counts and codes without response text.

Only if all 36 primary labels are present, apply the original budget and
headroom rules through the original measurement/decision functions. Reuse the
coordinator's frozen AI-only review of all 24 sensitive responses. Disagreements
remain separate and cannot replace primary labels; apply the original
`uncertain_readiness` sensitivity rule. Quality/anchor counts are descriptive:
the original baseline survey has no additional quality acceptance gate.

If coverage remains incomplete, report coverage and failure without imputing
grades or calculating a registered headroom outcome. Do not aggregate a chosen
partial subset as the full quality result. No headroom outcome launches another
refusal experiment automatically.

The combined result has different grading contexts (20 items versus two batches
of eight plus an ID checklist) and a later grading time. Server-side drift cannot
be excluded even with the same exposed model name. This is an amended completion,
not a clean single-session review or human validation. All fixtures are custom
and benign-only; there is no claim about harmful-request behavior.

Claude reviews the combined report before interpretation. Then discuss and
register a separate benign tone/persona pilot: direct and confident wording
while preserving facts and appropriate uncertainty, with no claim of personal
experience. That study is independent of the refusal vector and is not
registered or executed by this completion protocol.
