# Qwen3.8-27B sarcasm GCAD: review 05 (judge audit and R1 binding)

Advisor review, 2026-11-18, of `outputs/qwen38-sarcasm-gcad-01/judge-audit/`
and `results/protocols/qwen38-sarcasm-gcad-mechanics-01.json` (R1). The review
was read-only: I hashed files and read JSON. No model, judge or GPU call was
made.

## Verdict

**Judge audit accepted. R1 binding verified.** Nothing blocks the judge
streams once the elicitation outputs exist.

## Audit checks

| Check | Evidence | Status |
| --- | --- | --- |
| Snapshot identity | `bin/claude-2.1.327` has sha256 `343c744b…6ac9`, matching config and plan §8. The file is mode 0500 and has not changed since 15:51. The binding records version `2.1.327 (Claude Code)`. | pass |
| Identity around the call | `invoke()` takes `identity()` before the call and requires it to be equal afterwards. The binding is saved after the call. | pass |
| Isolation flags | The flags after the binary path are byte-identical to the audited historical pin `outputs/qwen38-judge-runtime-01/binding.json`. They are: `--safe-mode`, empty tools, strict empty MCP config, slash commands disabled, no session persistence, empty setting sources, model `claude-opus-5-5`, effort medium, the same blinded-evaluator system prompt, `--print`, and JSON output. Only the binary path differs, as registered. | pass |
| Working-directory isolation | `git rev-parse` returns 128 (not a repository). The git environment is clean; `GIT_PAGER` was removed. | pass |
| Isolation outcome | The answer to "user's current project?" is exactly `{"project":"UNKNOWN"}`. There was 1 turn, `permission_denials` is empty, no subagents ran, no web or tool use occurred, and stderr is empty. | pass |
| Model | `modelUsage` lists only `claude-opus-5-5`; canonical model is `claude-opus-5-5`; provider is first-party. | pass |
| Accounting | 1 call, which counts as the 1 audit start in plan §9. Duration was 3.50 s and cost $0.006. | pass |

## R1 hash verification

Each hash recorded in R1 equals the current file:
- manifest `13254530…`;
- synthetic result `1c55497d…`;
- synthetic decomposition `6cb39e60…`;
- judge audit result `eeca9320…`;
- judge binding `7ec181c3…`.

R1 states that the source did not change after R0. This is consistent with
the review 04 source check. The approved scope (elicitation only: 100 pairs,
200 generations, cap 256) matches plan §6.1.

## Limitation (disclosed, not a defect)

The audit is a single negative probe. It shows that this exact invocation
receives no project context, but it is not exhaustive. This is the registered
design (design review F3) and the same standard as the historical audits.
Payloads still contain opaque IDs only (review 02).

## Next reviews

1. **Eligibility and bank.** After elicitation judging, check the following:
   - the frozen §6.1 eligibility rule is applied exactly;
   - the matched-pair count is at least 10;
   - the extraction boundary counts;
   - real-bank finiteness and norms;
   - real-bank mechanics criteria 1–4, run in a fresh process.
2. **C6 analyzer.** It must be hashed before final registration.
