# Local artifacts and workspace maintenance

Current policy, reviewed with Claude on 2026-11-14. This repository contains
roughly 105 GB of local files; most are useful model assets and experiment
artifacts rather than temporary files. Git tracks source, protocols and compact
reports. `.gitignore` excludes large artifacts, raw outputs and caches.

| Location | Purpose | Retention |
| --- | --- | --- |
| `.hf/` | Pinned upstream checkpoints and tokenizer caches | Keep, including deferred 14B assets. |
| `artifacts/` | Saved quantized models, adapters and local tools | Keep reusable assets and release dependencies. |
| `data/` | Dataset downloads and prepared corpora | Keep source provenance and experiment inputs. |
| `outputs/` | Raw generations, grades, journals, snapshots and manifests | Keep experiment evidence, including failed or incomplete runs. |
| `.venv/` | Reproducible working environment | Keep during active studies. |
| `.pytest_cache/`, project `__pycache__/` | Regenerable test state and bytecode | May remove when not required by an active process or manifest. |

A directory named `draft`, `pre-audit` or `preparation-drafts` is not automatically
disposable. Historical preparation and failed-run records explain source changes,
selection decisions and call budgets. Deleting them could break reproducibility.
Do not use broad `git clean -fdx` or delete model caches as routine housekeeping.

Before deleting scratch files, enumerate exact paths and sizes and check their
references in manifests, registrations and documentation. Keep unknown `/tmp`
files, agent runtime links and files belonging to other projects. Cleanup does
not change frozen sources, inputs, registrations, original labels or results.

The completed 2026-11-14 cleanup removed only four local cache directories:
`.pytest_cache/`, `scripts/__pycache__/`, `evaluation/__pycache__/` and
`tests/__pycache__/`: 99 files, 1,175,043 logical bytes (about 1.3 MB allocated).
The local receipt is `outputs/workspace-maintenance-20261114/cache_cleanup.json`.
The latest baseline's 709 protected bindings, including the earlier 681, all
passed verification after cleanup. New Python invocations may recreate bytecode; use `-B` for
housekeeping checks to avoid doing so unnecessarily.

Scientific status and the next model experiment are maintained in
[the project map](project_tracks.md). Historical frozen documents retain the
scope and transition decisions recorded at their registration time.
