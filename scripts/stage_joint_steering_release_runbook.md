# Additive joint public release — CPU packaging only

No study/model/native/network calls, downloads, uploads or commits are performed
by this builder. Do not stage until `results/joint_gcad_01.json` is terminal with
all12 measured arms and byte-identical to its committed version. The separate
post-hoc `results/joint_gcad_lexical_audit_01.json` must also be complete and
committed, binding the same primary report. No staging before that audit. No active or
partial grading files are read. The registered study and staging03 stay frozen.

## CPU verification

```bash
CUDA_VISIBLE_DEVICES='' PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
outputs/persona-author-replication-audit-01/envs/gcad/bin/python -B -c \
"import sys; sys.path.append('.venv/lib/python3.10/site-packages'); import pytest; raise SystemExit(pytest.main(['-q','-p','no:cacheprovider','tests/test_stage_joint_steering_release.py']))"
```

Tests use synthetic aggregates and temporary bundles, small published bank
files and tiny random CPU Qwen2 only; they never load7B weights. Public hook AST
matches the registered implementation after ONLY module-docstring/import changes.
Tests exercise original critical-byte identity, H2/parent integrity, strict old
base revision guard, shadow-import rejection, local-module entry point, tiny
zero/single/cache parity, privacy/allowlist and terminal-before-write guards.

## Root staging after committed terminal report

```bash
CUDA_VISIBLE_DEVICES='' .venv/bin/python -B -m scripts.stage_joint_steering_release stage
CUDA_VISIBLE_DEVICES='' .venv/bin/python -B -m scripts.stage_joint_steering_release verify
```

Fresh destination: `outputs/steering-release-staging-04/`. Never overwrite03/04.
Outside upload manifest:
`outputs/steering-release-staging-04-publication-manifest.json`, unchanged
`publish_release.py` schema `repo_id/private/files{name:{bytes,sha256}}`.
Exact target: existing public
`RenormalizationTech/qwen2.5-7b-gcad-steering-controls` (no new repository).

Old `release_manifest.json`, all banks/profiles/runtime and other old files are
byte-identical to03; only root README/METHODS change and joint_controls is added.
H2 covers every current payload file except itself. It records the original
manifest hash and original commit763865908e5e94a020606b31ab66185360f7dd44.
The original manifest's full list applies to that original commit; the unchanged
isolated loader still checks the original eight critical files at new HEAD. Consumer
inventory checks allow optional root `.gitattributes`, root `.git` metadata,
downloader `.cache` and Python `__pycache__`; H2 records this exact exclusion
policy. These are not upload payload files. The stager still requires its exact
34-file payload. Root preserves the Hub-managed `.gitattributes` and records its
unchanged remote hash; no metadata deletion/upload is implied.
The new loader validates H2, the parent manifest, original critical bytes, actual
module paths and the unchanged original base/tokenizer/tensor guards. Consumers
run `python -B -m joint_controls.example` from downloaded root with a local
pinned base and an explicit measured preset. A new context is required per call.

The outside publication manifest is review input, not an instruction to invoke
`publish_release.py`: that tool creates a repository with `exist_ok=False`, and
its resume mode requires an already exact inventory, so it cannot perform this
additive update. Root will use `HfApi.create_commit` on the existing repository,
with an exact new/changed-file allowlist and a verified `parent_commit` guard.
No repository creation or deletion is involved.

The builder prints **required_commit_message**. Root MUST pass exactly these two
lines as `create_commit(commit_message=...)`:

```text
Release manifest SHA256: 8a57ab029d9d2adee60be7229ee30a3ba347c9c53442bed96799e2b2316849f4
Joint manifest SHA256: <actual SHA256 of staging04/joint_controls/manifest.json>
```

The card does not embed H2 itself (no hash cycle). Verify both lines and payload
hashes at the final immutable HF revision after root publishes. No upload is
authorized by invoking this builder. Keep original manifests/cards/logs intact.
No newGPU validation is claimed: CPU source equivalence and prospective mechanics
are distinguished from whatever completed study evidence is actually reported.

The practical guide includes all12 arm coefficients/four means/invalid/caps,
3x4 isolated cross-effects, within-arm trait Spearman, allfive exploratory guide
verdicts with failed/N/A keys and links to paired CIs/full contrasts/calibration/
duplicate noise in sanitized JSON. Zero knob means no injection, not trait
absence. Shared-query sums are one hook, not stacked independent hook instances.
Offense, coherence loss, condition-confounded pooled correlations, same-family
AI authorship/judging and24 practical-advice English questions limit claims.
No joint refit, independence, universal benchmark or automatic recommendation.
NF4/PRL are now public separate artifacts; private project-source visibility
remains accurately described. No raw questions/responses/reasons/tokens or
private physics adapter/base weights are added.

The additional `joint_controls/lexical_audit.json` is a sanitized aggregate-only
copy with original audit/source hashes. PF is explicitly labelled **friendly
rubric**, not literal swearing frequency. The guide now shows any-hit counts,
mean per-response hit density and within-arm lexical correlations with allfour
original scores. A score decrease with increased lexical frequency is a
construct confound, not evidence that swearing was suppressed. All original
quality verdicts remain unchanged. A CI crossing the coherence boundary or
per-response duplicate MAD does not establish a pass or a standard error.
