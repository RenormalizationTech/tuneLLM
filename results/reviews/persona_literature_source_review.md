# Isolated persona steering: author-source review

The coordinator and Claude read primary papers and author repositories before
the user selected the literature-first workflow. The worker then audited pinned
small code/data files without executing author code, installing packages or
running models/judges. The coordinator accepts this source audit. It is not
runtime or experimental validation.

Verified repositories:

- `safety-research/persona_vectors`, commit
  `b8e0f044fe2410a6fad579f38324f03f13b4e917`, Apache-2.0.
- `SharathHN/saps-steering`, commit
  `c56fb28ac0d14c8fd70035e035c5452e3c6c6ccc`, MIT repository notice.

Thirty-two allowlisted files, 232,906 bytes, match canonical Git blobs, sizes
and SHA-256s. Evidence is under ignored `outputs/persona-literature-audit-01/`:
`manifest.json`, `audit.md` and `environment.json`. No large weights, result logs
or learned-vector binaries were fetched.

PV supplies 20 extraction/20 evaluation questions and five instruction pairs for
`impolite`. It generates responses under the respective persona systems, filters
aligned rows by trait/coherence, pools response tokens and adds an unnormalized
raw difference at last prefill/cached decode positions. The code accepts positive
scores >=50; paper prose says >50. Its judge uses probability-weighted scores,
not our native integer estimator. Map hidden-state indices explicitly; Qwen3's
final normalized state is not a last-block residual vector.

SAPS supplies normalized-duplicate-disjoint 500/50/450 sarcasm splits. It extracts
last-prompt-token contrasts and a styled-projection target; the adaptive hook's
last-token-derived correction broadcasts across the prefix during prefill.
The author batch-one shape requires care in a batched HF port. Released oracle
table code chooses a layer per test sentence and is not a fixed validation-layer
benchmark. The public release lacks individual-judge logs and an executable
full judge pipeline; evaluator substitution must be explicit.

SAPS attributes inputs to GPT-4o transformations of earlier humblebrag material
and complaint inversions, with manual review. Released strings lack row-level
lineage and a separate dataset notice. Preserve MIT repository attribution and
this provenance limit; do not claim independently verified upstream rights or
redistribute raw third-party source corpora.

Neither complete author environment matches the local stack. A small native-HF
BF16 Qwen3-4B adaptation avoids replacing it. Check tensor/hook/boundary semantics
before execution and use one basis for extraction/intervention. No full
TransformerLens/HF numerical identity or cross-model vector portability is claimed.

Whole-response averaging has literature support; span-only extraction is not an
established fix for the custom nulls. Their cause remains unidentified. Neither
selected repository supplies the exact profanity artifact, so that profile is a
new-trait extension of the published general PV procedure. Joint work is deferred.
