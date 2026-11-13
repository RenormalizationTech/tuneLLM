# Published model artifacts

Model/data binaries stay under ignored `artifacts/releases/`. This directory
contains reviewable cards, licenses and provenance. The initial two releases use
the visibility chosen by the user. The standalone NF4 dependency is also private.

| Artifact | Visibility | Hub revision |
| --- | --- | --- |
| [Base GGUFs](https://huggingface.co/RenormalizationTech/Qwen3-4B-tuneLLM-GGUF) | Public | `96038e0cb81737bc58489e19cf97461058a0b4e5` |
| [Experimental adapter](https://huggingface.co/RenormalizationTech/Qwen3-4B-tuneLLM-GSM8K-QLoRA) | Private | `399903ca450b44a68b84ec8b6acfe2ff36cc6e64` |
| [Standalone NF4 base](https://huggingface.co/RenormalizationTech/Qwen3-4B-tuneLLM-NF4) | Private | `e9af208e44e083b77702a71d8aa65b7d97f78ce6` |

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

## QLoRA adapter

The bounded 1,024-example experiment completed 64 steps and selected step 64 by
validation loss. Its final math fallback accuracy regressed from 121/128 to
101/128, while required-format compliance improved from 107/128 to 128/128.
It is an experimental learning artifact with that regression prominent in its card.

- Target: `RenormalizationTech/Qwen3-4B-tuneLLM-GSM8K-QLoRA`.
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
GGUFs and a private experimental adapter. Credentials are not recorded in the
publication report; do not send the token to the assistant.

The CLI is installed and its help was checked. After identity, organization
access, target repository and visibility are verified, the GGUF payload can be
uploaded with the Hub client. For example, to create a private repository:

```bash
HF_HOME="$PWD/.hf" .venv/bin/hf upload \
  RenormalizationTech/Qwen3-4B-tuneLLM-GGUF \
  artifacts/releases/Qwen3-4B-tuneLLM-GGUF . --repo-type model --private
```

For a public new repository, omit `--private`. That flag only controls creation;
an existing repository keeps its visibility. Uploaded artifacts should be checked
against their file manifest and loaded from the recorded Hub revision before
calling publication complete.
