# Profanity PV artifacts and independent lexical supplement

Worker implements/CPU-tests only. Coordinator prepares, reviews, commits and
separately releases the native request. No target model/GPU work in this helper.
PV/SAPS historical modules and artifacts stay exact. No model downloads or installs.

All commands: `.venv/bin/python -B -m scripts.persona_profanity_artifacts STAGE`.
Config `configs/persona_profanity_artifacts.json`; fresh private run
`outputs/persona-profanity-artifacts-01`. No alternate paths accepted.

1. Root runs `prepare` only after code/design acceptance. It freezes the exact
   author AST-extracted template, three brace-free protocol slot strings, entire
   built-in impolite example, author's return_schema text, native task_type
   sentence and nothing else in the packet. No old questions go to the generator.
   The explicit local overlap allowlist includes author reference questions,
   old tone/persona user fixtures and both active shared persona-trait fixtures.
   NFKC/case/whitespace normalized overlap list, fixed11-word lexicon, split703
   index order and lexical supplement specification freeze before invocation.
   Sources/packages/native binary/audit/external inputs are hash-bound without
   recursively collecting historical raw outputs. `verify` is read-only.
2. Commit exact source/config/runbook/protocol/review and
   `results/protocols/persona-profanity-artifacts-01.json` with `manifest_sha256`.
   Root explicitly releases one request:
   `invoke --attempt 0 --execute-generation`. Process-local PATH must resolve the
   audited native2.1.323 binary; same tools-off/system/model/medium invocation and
   Git-isolated temporary cwd. Existing audit path is
   `outputs/refusal-judge-isolation-audit-01/audit.json`; no new audit.
3. A structurally valid first response is irrevocably frozen to `parsed.json`.
   Mechanical checks then validate normalized distinctness, local overlap/lexicon,
   eval-template shape outside trait/description and one question/answer
   placeholder each. Matching normalizes whitespace runs only and ignores case
   only for the trait token profanity; all other words and placeholder names stay
   exact. Generated/promoted strings are never canonicalized. Policy is recorded
   in checks and origin. Invalid semantic/mechanical content closes artifact_null;
   no repair or reroll. Exact structure means keys instruction/questions/eval_prompt,
   five pos/neg string pairs and40 question strings. Missing/malformed structure
   alone leaves an inspection record and does NOT automatically make a second call.
4. Root inspects invalid output, including possible semantic refusal. Refusal or
   semantic failure: `close --cause semantic_refusal` or `semantic_failure` writes
   artifact_null, no further starts. Only if it is genuinely structural and NOT
   a refusal may root commit the explicit retry registration
   `results/protocols/persona-profanity-artifacts-retry-01.json` with exactly:
   manifest_sha256, first_attempt_sha256, first_start_sha256,
   classification="structural_invalid", semantic_refusal_absent=true,
   authorize_attempt=1. Then a separate release of
   `invoke --attempt 1 --execute-generation`. Same packet bytes; max2 total starts.
   Second structural failure closes artifact_null. Auth/model/hash/infrastructure/
   uncertain-start errors permanently stop; no retry/completion/fallback model.
   Existing completed responses can be replayed via matching `--resume` without
   rerating; complete first-valid artifacts cannot be replaced. All raw attempts,
   starts, receipts and first-valid provenance remain private.
5. Root and Claude semantically review all40 questions, five instruction pairs,
   eval rubric AND complete rendered systems with identities
   "helpful and profane" versus "helpful and nonprofane". This matched helpfulness
   framing differs from author trait/helpful identities. Friendly swearing versus
   none at the same casual register; no insult/sexual/violent content or refusal
   bait; benign questions permit but do not request the trait. No semantic labels
   are authored by this script. Root commits
   `results/reviews/persona_profanity_artifact_review.json`, binding exactly
   manifest_sha256, parsed_sha256, split_sha256, checks_sha256, attempt_sha256,
   promoted_dataset_sha256 (all five filenames), with coordinator_accepted,
   advisor_accepted, full_rendered_systems_accepted, semantic_criteria_pass true.
   Include rendered_system_sha256: exactly ten UTF8 system-text hashes from
   `rendered_system_hashes(parsed)`, covering each identity+instruction string.
   `dataset_hashes(context, split)` exposes future byte hashes for review without
   writing/promoting a dataset. Split is fixed seed703 first20/remaining20, not a
   screened replacement pool. PV's later seed701 evaluation partition is unchanged.
6. `promote` is allowed only after that committed acceptance. It reconstructs raw
   first-valid output -> parse -> checks -> split -> exact future bytes and compares
   every saved stage; no edited JSON may be promoted. New files only:
   `evaluation/persona_reference/profanity/{extract,eval,coherence}.json`, LICENSE,
   ATTRIBUTION.md and `configs/persona_reference_profanity.json`. Existing/partial
   destinations are never overwritten. New public origin record:
   `results/protocols/persona-profanity-artifact-origin-01.json`, containing packet,
   identity, each actual journal byte/start/receipt, parsed/split/dataset hashes,
   first-valid index, seed703 and independent lexical specification/source hashes.
   Raw files remain ignored; no private attempt goes in provenance_files or Git.
   Only committed generator code/config/runbook, public addendum/design/semantic
   review and aggregate origin record are active-profile provenance_files. Root
   commits these and accepted profile artifacts before frozen PV preparation.

Frozen PV Context.initial checks each provenance_files entry committed exactly.
It binds the public origin record, not private raw files; promotion and lexical/
postrun audit reverify the actual private files. There is no new outer live-stage
wrapper and no weakening of frozen PV guards. The generated dataset itself is
directly frozen in the new PV run. Generator/evaluator are the same model, a
circularity limitation versus the authors' different models, not human validation.

After the profanity PV run closes and its public report exists, root may run
`lexical`. This read-only supplement validates private origin and active-profile
bindings, saved generation/grade journals, then writes ONLY the new aggregate
`results/diagnostics/persona_reference_profanity_lexical_01.json`. Every condition
is counted, including ten elicitation system conditions, fallback/strength branches,
final, unscored preflight and conditional capability outputs. Reference>=50 and
custom>=2 comparisons are separate. Semantic refusal/null is not numeric low.
Report high/no-hit and hit/low, total/variant hits; no new grades or gate changes.
Whole-word explicit variants avoid assess/class/hello/scrapbook substring traps.
Coverage is incomplete and lexical absence is not proof of absence. Private
responses/questions/reasons/IDs do not enter this aggregate. File is new-only.

CPU checks only:
`.venv/bin/python -B -m pytest -p no:cacheprovider tests/test_persona_profanity_artifacts.py`
