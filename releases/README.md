# Published model artifacts

Model/data binaries stay under ignored `artifacts/releases/`. This directory
contains reviewable cards, licenses and provenance. Current visibility and the
GSM8K deletion are recorded in the
[2026-11-16 transition receipt](../results/hf_visibility_transition_01.json).
Historical receipts, cards, reproduction notes and manifests keep their original
private labels; those labels do not describe current repository visibility.

| Artifact | Visibility | Hub revision |
| --- | --- | --- |
| [Base GGUFs](https://huggingface.co/RenormalizationTech/Qwen3-4B-tuneLLM-GGUF) | Public | `96038e0cb81737bc58489e19cf97461058a0b4e5` |
| GSM8K QLoRA learning artifact | Deleted after verified archive | Archived head `6a29265f7dc99577777e412fa76659d8b734fdea` |
| [Standalone NF4 base](https://huggingface.co/RenormalizationTech/Qwen3-4B-tuneLLM-NF4) | Public | `2ae7eec7ac38290ea28efefd1a06e779027aafc7` |
| [PRL Writing QLoRA](https://huggingface.co/RenormalizationTech/Qwen3-4B-tuneLLM-PRL-Writing-QLoRA) | Public | `b9f7ff0222d9e4070ee88f89e4f00ff096e20359` |

Only the NF4/PRL cards changed during the visibility transition; all other remote
file identities, including weights, stayed unchanged. The PRL dependency remains
NF4 revision `e9af208e44e083b77702a71d8aa65b7d97f78ce6`. All 16 originally
recorded dependency files were verified anonymously: small files by download,
large files by LFS metadata against prior byte-verified hashes. There was no new
weight download or GPU check. The prepared PRL test split and frozen evaluation
fixtures are now public; future reuse is not clean held-out evaluation.

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
