# Published model artifacts

Model/data binaries stay under ignored `artifacts/releases/`. This directory
contains reviewable cards, licenses and provenance. Current visibility and the
GSM8K deletion are recorded in the
[2026-11-16 transition receipt](../results/hf_visibility_transition_01.json).
Historical receipts, cards, reproduction notes and manifests keep their original
private labels; those labels do not describe current repository visibility.

| Artifact | Local card | Current public Hub revision |
| --- | --- | --- |
| [Base GGUFs](https://huggingface.co/RenormalizationTech/Qwen3-4B-tuneLLM-GGUF) | [README](Qwen3-4B-tuneLLM-GGUF/README.md) | `77de6c6ef052fca13675d6b8a47ff3afc81008c8` |
| [Standalone NF4 base](https://huggingface.co/RenormalizationTech/Qwen3-4B-tuneLLM-NF4) | [README](Qwen3-4B-tuneLLM-NF4/README.md) | `670ac49990add1639cd38ba0e858bdda2d9b6fb9` |
| [PRL Writing QLoRA](https://huggingface.co/RenormalizationTech/Qwen3-4B-tuneLLM-PRL-Writing-QLoRA) | [README](Qwen3-4B-tuneLLM-PRL-Writing-QLoRA/README.md) | `3ac6a47916a86d334a7548a3d678c2fecd1c9dd7` |
| [7B GCAD trait controls](https://huggingface.co/RenormalizationTech/qwen2.5-7b-gcad-steering-controls) | [README](qwen2.5-7b-gcad-steering-controls/README.md) | `a6c49acf9ec12cac9bb6438db26604e5ca45cc74` |
| [7B refusal experiment](https://huggingface.co/RenormalizationTech/qwen2.5-7b-refusal-steering-experiment) | [README](qwen2.5-7b-refusal-steering-experiment/README.md) | `b9649ef186570f6159fbe95011024fa04655a89e` |

The earlier GSM8K adapter is deleted on HF after verified archival; its historical
head was `6a29265f7dc99577777e412fa76659d8b734fdea`.

## Local cards and complete payloads

Every current HF release has a card mirror under `releases/<HF-repo>/README.md`.
This tracked directory is a documentation index, not a copy of every weight or
runtime file. Complete release payloads live under ignored
`artifacts/releases/<HF-repo>/`; upstream downloads are cached separately in
`.hf/hub/`. Card-edit snapshots and historical staging remain in ignored
`outputs/`.

The two 7B packages are now available at these complete local destinations:

| Published package | Complete local payload |
| --- | --- |
| GCAD trait controls | `artifacts/releases/qwen2.5-7b-gcad-steering-controls/` |
| Refusal experiment | `artifacts/releases/qwen2.5-7b-refusal-steering-experiment/` |

These contain steering banks/vectors and runtime code. They use the separately
obtained `Qwen/Qwen2.5-7B-Instruct` checkpoint; they are not full base-model
releases. The public [reproduction guide](../docs/reproduce.md) explains the
small pinned refusal-bundle download and separates it from base-weight setup.

## Audited HF card polish (2026-11-17)

Claude audited all five cards before publication. They now link the public
GitHub project, results/examples, reproduction guide and references; distinguish
the completed local 27B NF4 smoke from planned steering/training; and preserve
all original loading code blocks and measured outcomes. Exact throughput and
both physics GSM8K scoring conventions are reported. The two new 7B card
folders make the local index consistent with HF.

The [publication receipt](../results/huggingface_card_polish_02.json) records
parent commits, updated card hashes and anonymous verification. Only the five
READMEs and the trait package's derived joint-manifest README hash changed on
HF. All other remote files were verified unchanged. Original manifests,
immutable artifact revisions and earlier receipts remain historical evidence.

## Current portfolio publication update (2026-11-16)

The [7B refusal experiment](https://huggingface.co/RenormalizationTech/qwen2.5-7b-refusal-steering-experiment)
is public at `3e8a151667a8fef0ebba552a3f3d74317b1c093d`. Its 16-file package
contains the frozen direction, identical safetensors mirror, standalone runtime,
pinned setting, all development-arm results and selected-case limits. Six CPU
checks passed; all 16 anonymously downloaded files and both vector loaders were
verified. There was no new full-model/GPU replay. See the
[publication receipt](../results/refusal_7b_huggingface_publication_01.json) and
[explicit manifest](../results/refusal_7b_release_manifest_01.json).

The four earlier HF cards now give concise portfolio summaries and crosslinks;
all weights, runtimes and measured results are unchanged. The trait package's
joint manifest was refreshed for its updated README; the original manifest is
preserved. Current card revisions and unchanged-file verification are in the
[documentation receipt](../results/huggingface_portfolio_documentation_01.json).
The original trait publication receipt remains historical evidence at its pinned
revision. Older statements that no refusal artifact was published describe
those earlier experiments, before this separately authorized release.

Only the NF4/PRL cards changed during the visibility transition; all other remote
file identities, including weights, stayed unchanged. The PRL dependency remains
NF4 revision `e9af208e44e083b77702a71d8aa65b7d97f78ce6`. All 16 originally
recorded dependency files were verified anonymously: small files by download,
large files by LFS metadata against prior byte-verified hashes. There was no new
weight download or GPU check. The prepared PRL test split and frozen evaluation
fixtures are now public; future reuse is not clean held-out evaluation.

The [local storage cleanup](../results/storage_cleanup_20261116_01.json)
removed redundant verification-download caches and the two duplicate PRL
pre-review weight files, preserving the retained release payload and all
physics optimizer/checkpoint files. Historical receipts still describe their
original verification paths; metadata backups and recovery instructions are
in `outputs/storage-cleanup-20261116-01/`. Published repositories are unchanged.

### Additive benchmark documentation updates (2026-11-13)

The completed [benchmark summary](../docs/benchmark_summary.md) was included
in all four then-existing Hub repositories as `benchmark_summary.json`, alongside
updated cards. These commits changed only `README.md` and added that report. Original
weights, configuration, tokenizer, evaluation files and release manifests remain
unchanged. The physics adapter's NF4 dependency pin remains valid; the table
below records historical documentation revisions. GSM8K revisions are available
in the local archive after deletion, not on the Hub.

| Artifact | Documentation revision | Verified audit |
| --- | --- | --- |
| Public GGUFs | `c089994f008cc3d6326e4e010e5cba04974d0d02` | [GGUF](../results/huggingface_documentation_update_GGUF.json) |
| Private GSM8K adapter | `6a29265f7dc99577777e412fa76659d8b734fdea` | [GSM8K](../results/huggingface_documentation_update_GSM8K-QLoRA.json) |
| Private NF4 base | `6ba2181bb08ebab104292f19fb381efb8bc6d207` | [NF4](../results/huggingface_documentation_update_NF4.json) |
| Private physics adapter | `5a3b128e520b9147b999ecb00da02216159a8cc6` | [Physics](../results/huggingface_documentation_update_PRL-Writing-QLoRA.json) |

Each audit records the remote parent and complete before/after inventories.
Read-only preflights passed, publication used an exact two-file allowlist and
parent-commit concurrency check, and post-update hashes and visibility matched.
Claude reviewed the concrete cards and reports before application. No new
inference or model-weight upload was needed for these reporting updates.

The standalone NF4 is unadapted and preserves the exact base used for the physics
workflow. Every downloaded file hash matches, and a GPU reload of the downloaded
revision reproduced both original probe logits and token sequences exactly. See
`results/nf4_huggingface_publication.json` and its model card for verification scope.

Every uploaded file was checked against its expected SHA256, using Hub LFS
metadata for large files and revision-pinned downloads for small files. The
downloaded private adapter reproduced the first eight recorded final math
responses exactly. The downloaded public Q4 GGUF loaded and passed its arithmetic
smoke prompt. Other GGUF formats have remote hash verification and prior local
load/benchmark checks. See `results/huggingface_publication.json` for the record.

## Base GGUFs

- Target repository: `RenormalizationTech/Qwen3-4B-tuneLLM-GGUF`.
- Local payload: `artifacts/releases/Qwen3-4B-tuneLLM-GGUF/`.
- Includes BF16, Q8_0 and Q4_K_M GGUFs, upstream license, model card, compact
  evaluations, checksums and a source archive of workflow commit `dab1dae`.
- The numerical measurements are small pipeline experiments. See the card's
  limits before interpreting them.

The model files use hard links to the validated local GGUFs to avoid another
14 GiB copy. Recheck the manifest hashes before uploading; these are ordinary
files for the upload client.

## Archived GSM8K QLoRA learning experiment

The bounded 1,024-example experiment completed 64 steps and selected step 64 by
validation loss. Its final math fallback accuracy regressed from 121/128 to
101/128, while required-format compliance improved from 107/128 to 128/128.
It is an experimental learning artifact with that regression prominent in its card.

- Deleted repository: `RenormalizationTech/Qwen3-4B-tuneLLM-GSM8K-QLoRA`.
- Verified ignored archive: `outputs/hf-transition-20261116-01/gsm8k-archive`.
  All three reachable revisions and 42 file entries were hash-rechecked before
  deletion; main was the only ref, with no tags, PR or conversion refs. The
  [receipt](../results/hf_visibility_transition_01.json) records hashes and
  post-deletion checks (authenticated 404, anonymous 401 and absent org listing).
  This is a revision-snapshot archive, not a claimed full Git mirror.
- Payload: `artifacts/releases/Qwen3-4B-tuneLLM-GSM8K-QLoRA/`.
- Adapter tensors matched exactly after reload. Generated tokens matched on the
  first 16 final math questions and all 128 retention questions.
- The primary eight-token retention scores were 83/128 to 76/128; the single
  secondary 64-token diagnostic scored 83/128 for both. The output-budget and
  parser limits prevent a general-retention claim.
- Both payloads include an environment lock and reproduction guide. Publication
  metadata copies replace local paths; raw local records remain unchanged. Source
  archives omit Git author history.

## Authentication and publication

Hugging Face authentication was configured for publication. To sign in again
from this project directory without putting a token in chat or version control:

```bash
HF_HOME="$PWD/.hf" .venv/bin/hf auth login
```

Use an account/token with write access to the `RenormalizationTech` organization.
The `.hf/` directory is ignored by Git. The current visibility choice is public
GGUFs, NF4, PRL Writing QLoRA and the separately published 7B steering controls.
The GSM8K adapter repository is deleted after verified archival. Credentials
are not recorded in the publication report; do not send the token to the assistant.

For a separately reviewed new release, verify identity, organization access,
exact target, payload and visibility before upload. A public creation example
is shown below; this is not an instruction to re-upload the current artifacts:

```bash
HF_HOME="$PWD/.hf" .venv/bin/hf upload \
  RenormalizationTech/Qwen3-4B-tuneLLM-GGUF \
  artifacts/releases/Qwen3-4B-tuneLLM-GGUF . --repo-type model
```

For a private new repository, add `--private`. That flag only controls creation;
an existing repository keeps its visibility. Uploaded artifacts should be checked
against their file manifest and loaded from the recorded Hub revision before
calling publication complete.


## Physics-writing pilot (public)

- Repository: [RenormalizationTech/Qwen3-4B-tuneLLM-PRL-Writing-QLoRA](https://huggingface.co/RenormalizationTech/Qwen3-4B-tuneLLM-PRL-Writing-QLoRA).
- Current card revision: `b9f7ff0222d9e4070ee88f89e4f00ff096e20359`; original
  load-verified revision: `fcd1af6d47678431d69cbdff6b1e6af4bee57854`.
- Dependency: public `RenormalizationTech/Qwen3-4B-tuneLLM-NF4` at
  `e9af208e44e083b77702a71d8aa65b7d97f78ce6`.
- Selected adapter: all-linear rank 128, alpha 256, LR 5e-5, epoch 1 / step 69,
  522,380 prediction targets. The full registered run completed three epochs.
- Training: 70 licensed PRL-associated condensed-matter theory arXiv papers;
  10 development and 20 final-test papers, with frozen LaTeX and attribution.
- Held-out NLL: 1.974938 to 1.733950; numeric GSM8K: 117/128 to 111/128.
  Human writing-quality improvement is not established.
- All 49 remote file hashes matched; downloaded weights/config/tokenizer hashes
  and eight title input/generated-ID probes matched. No logits or full-benchmark
  replay claim is made.
- Card: [physics adapter README](Qwen3-4B-tuneLLM-PRL-Writing-QLoRA/README.md).
  Complete [results](../results/physics_prl_final.json) and
  [publication proof](../results/physics_huggingface_publication.json).

The model package includes exact prepared CC BY data, full attribution, source
code, environment lock, scorer/fixture metadata and a blinded 60-pair JSON packet.
Its A/B key remains outside the payload. A separate local 30-pair review is at
`outputs/physics-writing-review/review.html`; ratings are absent and assessment
is deferred. Human ratings do not block completion. Earlier public
GGUF publication and the deleted, archived GSM8K learning experiment remain
separate records; neither is a new training or inference result.
