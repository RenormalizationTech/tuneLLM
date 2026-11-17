# File organization and storage rules

This is the shared rule for new project files. Directory READMEs explain their
entry points; this page governs storage. Apply it prospectively: keep existing
paths that are bound by protocols, manifests, scripts or published instructions.

## Where files belong

| Location | Store here | Git tracking |
| --- | --- | --- |
| Repository root | Project overview, license/notices, environment requirements, test/site configuration and the historical `plan.md` | Tracked; put new research narratives in `docs/`. |
| `docs/` | Current guides, protocols, execution summaries and clearly identified history | Tracked. Small publication figures and their source data go in `docs/assets/`. |
| `scripts/` | Runnable modules, shared helpers and matching `*_runbook.md` instructions | Tracked. Keep generated output elsewhere. |
| `configs/` | Small experiment configurations with model/data revisions and settings | Tracked. Preserve registered configs; use a new filename for a new experiment. |
| `evaluation/` | Scoring code, small frozen prompts/splits/rubrics, licenses and attribution | Tracked. Put large downloaded/prepared corpora in `data/`. |
| `tests/` | Portable synthetic fixtures/tests and explicitly opt-in research checks | Tracked. Tests must not require accidental local caches to count as portable. |
| `results/` | Compact outcome reports, publication/maintenance receipts and pointers to raw runs | Tracked. Protocol bindings go in `protocols/`, reviews in `reviews/`, follow-up analyses in `diagnostics/`. |
| `releases/` | Current HF card mirrors, `catalog.json`, selected historical metadata and publication history | Tracked. Model binaries and complete runtime payloads belong in `artifacts/releases/`. |
| `.hf/` | Pinned upstream downloads, Hub caches and local HF state | Ignored. Retain revision identities; do not commit credentials. |
| `artifacts/` | Reusable local models, quantized weights, adapters, steering banks and local tools | Ignored. Complete published mirrors go in `artifacts/releases/<HF-repo>/`. |
| `outputs/` | Per-run inputs/source snapshots, generations, grades, journals, checkpoints, staging, backups and verification details | Ignored. Keep evidence for failed/null runs as well as successful runs. |
| `data/` | Downloaded and prepared corpora with provenance, licenses and split metadata | Mostly ignored. Existing small tracked steering records are a historical exception; new small public fixtures normally go in `evaluation/`. |

Use the environment prescribed by the relevant reproduction guide. Keep model
tracks isolated; place additional local environments/caches under an identified
`outputs/<study-or-purpose>/` directory rather than adding root-level folders.
The existing `.venv/` remains the original environment. Pin a new runtime profile
in `requirements-<purpose>.txt` and record installed versions for the run. Use a
separate environment for incompatible dependencies; preserve environments bound
to existing runs.

Keep new tracked reports/fixtures small (normally below 1 MiB); explain larger
files. Model weights, model archives, raw generations, judge journals and private
review keys stay out of Git. New tracked metadata uses repo-relative paths and
includes applicable data licenses/attribution. Root automation lives in `.github/`.
Research Markdown stays directly under `docs/`, grouped through site navigation;
`docs/index.md` is its home page, so do not add a competing `docs/README.md`.

## Naming and recording a run

Use a descriptive study/model identifier and a run number. Follow existing
Python/config/result `snake_case` and run-directory `hyphen-separated` conventions.
Registrations in `results/protocols/` also follow the hyphenated run-ID style.
For example, a future study could use `configs/qwen38_sarcasm_01.json`,
`outputs/qwen38-sarcasm-01/` and `results/qwen38_sarcasm_01.json`.
These are naming examples, not completed or scheduled runs.

For registered model experiments or publication artifacts, record model/data
revisions, config, source identity and intended output path before execution
using the existing protocol workflow. Exploratory/debug runs need their own
output directory and recorded settings; label them exploratory and do not treat
them as registered evaluations. Give a fresh experiment a new directory; do not
silently overwrite an existing experiment. Preserve the
exact source of scripts that write model artifacts, results or release metadata
in `scripts/` or the run's `source/` snapshot, with its hash, before execution.
Store resumable checkpoints under that run's `checkpoints/`, and identify any
selected reusable
asset under `artifacts/`. The compact result should link the protocol/config,
raw output location, selected checkpoint and evaluation limits. Keep detailed
inventories locally when the tracked receipt can identify their verification
method, count and digest, as in the [release catalog](../releases/catalog.json).
If referenced raw evidence is later removed, add a dated receipt explaining the
removal and available recovery source.

## Current guidance and historical evidence

[Project status](project_tracks.md) governs current scope; [reproduce](reproduce.md)
governs runnable setup; the [release index](../releases/README.md) governs current
publication status. Keep these concise and update their summaries together.

Preserve registered protocols, source snapshots, fixtures, result labels,
acceptance decisions and original manifests. Record a correction or later action
in a new dated/versioned report rather than changing the old evidence. When
replacing a long mutable overview, preserve useful history with a clear history
name and working relative links. Do not rewrite a frozen document just to update
an old private/unpublished label. The website labels archived pages as historical.

## Releases and model copies

Use one canonical local payload per published package at
`artifacts/releases/<HF-repo>/`. Keep the tracked card mirror, catalog revision,
current card hash and publication receipt consistent. A local mirror is complete
only after checking every published file against that pinned HF revision; local
cache extras do not count as published files. A card-only update also refreshes
the tracked mirror/catalog and adds a receipt.
Retired packages keep their historical folders and are marked in the catalog
and index.

Release-time manifests remain immutable even when later card metadata changes.
Preserve old metadata before a local mirror update and record the exact changed
files; verify that weights/runtime files remain unchanged. Do not infer a new
quality or reload check from a metadata-only update.

Upstream BF16 weights and a derived NF4/GGUF model serve different purposes and
may both be retained. Avoid unnecessary independent copies. Hard links may share
storage: do not edit shared model files in place, and do not assume deleting one
path reclaims its displayed size.

## Cleanup

Ignored files are not automatically disposable. Before removing anything, list
exact paths and measured sizes, check active use and protocol/manifest references,
and distinguish regenerable caches from model assets and experiment evidence.
Preserve unknown or bound files. Record consequential cleanup and recovery/source
locations in a receipt. Do not use broad `git clean -fdx` as housekeeping.
Check available space before large model downloads. Storage figures should name
their measurement date and units (GB or GiB).
The [maintenance history](workspace_maintenance.md) records earlier scoped cleanups;
it is not a list of directories to delete again.
