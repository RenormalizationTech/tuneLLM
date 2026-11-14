# XSTest-safe250 adapted baseline

The authoritative protocol is `docs/xstest_baseline_protocol.md`. Root owns
registrations, execution releases and documentation. No new dependencies are
needed. Reference modules are never imported or run: only two pinned pure
functions are isolated through AST, without their imports, decorators or loops.

```bash
.venv/bin/python -m scripts.xstest_baseline prepare
.venv/bin/python -m scripts.xstest_baseline verify
```

Preparation freezes all250 safe rows in CSV order, overlap72/250, exact native
rendered prompts/token IDs, source/model/licence hashes and prior421 bindings.
The remaining178 IDs are not a newly reserved holdout. Definitions and the
label-mapping sentence come verbatim from pinned Inspect source. JSON `grade`
replaces the upstream final `GRADE:` output, and a brief private per-item reason
replaces the upstream step-by-step rationale. These are explicit adaptations.
The author string matcher is secondary only. No
unsafe rows are selected and no source prompt is rewritten.

After root reviews/commits the exact registration and releases physical GPU0:

```bash
.venv/bin/python -u -m scripts.xstest_baseline generate --execute-gpu
.venv/bin/python -m scripts.xstest_baseline packet
```

HF uses cached original BF16 Qwen3-4B, SDPA, native helpful system/nonthinking
prefix, greedy seed42/batch8/cap256. Requested temperature0 is implemented with
`do_sample=False`; unused sampling parameters are disabled. Only terminal EOS
is omitted from decoded answer text; generated reasoning/special markers and
whitespace are retained. Cap hits never automatically become refusals.

After a separate judging release, prepend root's verified private bin directory
to PATH for this process so `claude` resolves the pinned native2.1.323 digest:

```bash
PATH=PINNED_BIN:"$PATH" .venv/bin/python -u -m scripts.xstest_baseline judge --execute-judgments
.venv/bin/python -m scripts.xstest_baseline report
```

Audit: `outputs/refusal-judge-isolation-audit-01/audit.json`. Live authentication
is not tested by CPU preparation. No new flags, audit or unbounded retries.
Source-order batches are partitioned before dropping empty rows; last source
batch has two. Every empty output is local gradeI and stays in denominator250.
Entirely empty batches use zero calls. At most32 batches/two schema-only attempts
each; first valid frozen. Auth/infrastructure/model/isolation/hash/uncertain-start
failure permanently stops this run. `--resume` can only reuse the exact journal;
it cannot replace a valid call or continue a stopped run.

Complete means all250 labels (including mechanical empties). Incomplete output
has no refusal metric or headroom decision. Complete output reports both refusal
fraction and percentage, all C/P/I/type/mechanical counts and nonempty refusal
diagnostics. Empty labels, earlier prompt exposure and grader/runtime adaptations
are explicit. No automatic rerun, steering, paraphrase generation or model change.

Raw texts, rendered token IDs and grader reasons stay in ignored private output.
New public result: `results/xstest_baseline_01.json`; old results never change.
