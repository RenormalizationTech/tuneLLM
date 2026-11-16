# Joint GCAD lexical audit — explicitly post-hoc, zero-call

Under the user’s joint-control request, PM and Claude approved this descriptive
audit after primary inspection. It is not confirmatory and changes no original labels, scores,
selection, gate or quality verdict. No model, GPU, native/API call, new lexicon
or semantic rating is allowed. Root commits this spec/source and a new input-hash
registration BEFORE running any real audit computation.

Use the exact `LEXICON`, `hits` and mechanical checks in frozen
`author_profanity_method_common.py`:11 case-insensitive whole-word non-slur
entries. Whitespace denominator is exactly `len(text.split())`; each response
has hit count, word count, and100*hits/words, undefined (null) at zero words.
This is not percentage of words containing a hit: multiple matches within one
whitespace word can yield a rate>100. No morphological/semantic expansion.

Report all12 arms, n_any_hit, n_empty, mean hit count, **mean per-response rate
(primary)** and pooled100*sum(hits)/sum(words) (**secondary**, length-weighted).
Keep all responses; also report valid/invalid splits using the unchanged union
empty OR distinct word bigram<.6. Undefined rates are excluded from rate means,
not silently replaced by zero. Check every original lexhit/invalid count and
all four conservative score means against the frozen primary report. Expected
operator observations are PFsingle14/24, fulltriple24/24, halftriple19/24;
these are not forced values or a selection criterion.

Question-paired bootstrap:2000 draws, seed106012, same sampled question indices
across all12 arms in each draw. Unadjusted percentile95% intervals for each
arm's mean rate and mean-rate difference vs baseline. Record defined replicate
counts; undefined arm/baseline means make that difference undefined. Actual
responses were reported nonempty, but implementation/tests preserve the rule.

Spearman of rate vs the four ORIGINAL scores: friendly PF, broad impolite,
sarcasm and coherence. Trait REFUSAL retains its original conservative0;
coherence remains numeric. Report within each arm, raw pooled and prompt-
centred pooled (subtract within-question mean across arms). Exclude undefined
rate pairs; constant ranks produce null. Both pooled forms remain condition-
confounded. These are descriptions, not independence/causal claims.

Interpretation: original PF rubric measures friendly/helpful swearing and
excludes user-directed insults; falling friendly PF scores need not mean fewer
literal swearwords. The proxy misses slang/variants, counts quoted/incidental
hits and is sensitive to compounds, response length and incoherence. n24
AI-authored English advice prompts and same-family authorship/judging limit
scope. All five original candidate coherence failures remain unchanged. A
half-triple CI crossing the -20 boundary is not a new quality pass; duplicate
coherence MAD7.25 is per-response, not standard error of the mean, and cannot
establish that the miss is proven within noise.

## Root registration and execution

New record: `results/protocols/joint-gcad-lexical-audit-01.json`:

- `post_hoc: true`.
- `sources_sha256`: exact mappings for script/spec/frozen lexicon+stats modules
  and immutable JSON-writing helper (`SOURCES` in the new script).
- `inputs_sha256`: exact mappings for original generation predictions, complete
  `results/joint_gcad_01.json`, original run manifest and the four primary
  accepted.json files (`INPUTS` in the new script).

Original plans are additionally verified against their hash in the bound run
manifest. Each accepted stream's binding file must match its bound SHA; exact
36-batch alias/score sets are reconstructed from the frozen plans. No private
response text or record-level token/count arrays are exported. Per-response
counts exist only transiently in memory; no new private output is necessary.
Verify all registered source/input hashes again before exclusive atomic output.
The spec/source/registration and original public result must be committed.

```bash
.venv/bin/python -B -m scripts.joint_gcad_lexical_audit --execute
```

Without `--execute`, the command performs no audit computation and writes
nothing. Output is only the NEW `results/joint_gcad_lexical_audit_01.json`.
No original record is edited. Root executes after review/registration; worker
runs synthetic CPU tests only. Public packaging requires this audit committed,
binds its source hash, labels PF as friendly-rubric PF, and supplies a separate
`joint_controls/lexical_audit.json`. Do not stage04 before both results exist
and are committed. Existing main criteria are retained verbatim, not replaced.
