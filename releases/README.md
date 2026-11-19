# Published artifacts and local storage

Checked 2026-11-19 UTC. Ten HF packages are public. This directory holds their
**card mirrors and selected historical metadata**, not model weights.
[catalog.json](catalog.json) records exact revisions, card hashes, payload paths,
result records and publication receipts.

## Current public releases

| Package | Card mirror | Verified public HF revision |
| --- | --- | --- |
| [4B GGUF](https://huggingface.co/RenormalizationTech/Qwen3-4B-tuneLLM-GGUF) | [README](Qwen3-4B-tuneLLM-GGUF/README.md) | [`77de6c6ef052fca13675d6b8a47ff3afc81008c8`](https://huggingface.co/RenormalizationTech/Qwen3-4B-tuneLLM-GGUF/tree/77de6c6ef052fca13675d6b8a47ff3afc81008c8) |
| [4B NF4 base](https://huggingface.co/RenormalizationTech/Qwen3-4B-tuneLLM-NF4) | [README](Qwen3-4B-tuneLLM-NF4/README.md) | [`670ac49990add1639cd38ba0e858bdda2d9b6fb9`](https://huggingface.co/RenormalizationTech/Qwen3-4B-tuneLLM-NF4/tree/670ac49990add1639cd38ba0e858bdda2d9b6fb9) |
| [Physics QLoRA](https://huggingface.co/RenormalizationTech/Qwen3-4B-tuneLLM-PRL-Writing-QLoRA) | [README](Qwen3-4B-tuneLLM-PRL-Writing-QLoRA/README.md) | [`3ac6a47916a86d334a7548a3d678c2fecd1c9dd7`](https://huggingface.co/RenormalizationTech/Qwen3-4B-tuneLLM-PRL-Writing-QLoRA/tree/3ac6a47916a86d334a7548a3d678c2fecd1c9dd7) |
| [7B trait controls](https://huggingface.co/RenormalizationTech/qwen2.5-7b-gcad-steering-controls) | [README](qwen2.5-7b-gcad-steering-controls/README.md) | [`a6c49acf9ec12cac9bb6438db26604e5ca45cc74`](https://huggingface.co/RenormalizationTech/qwen2.5-7b-gcad-steering-controls/tree/a6c49acf9ec12cac9bb6438db26604e5ca45cc74) |
| [7B refusal experiment](https://huggingface.co/RenormalizationTech/qwen2.5-7b-refusal-steering-experiment) | [README](qwen2.5-7b-refusal-steering-experiment/README.md) | [`b9649ef186570f6159fbe95011024fa04655a89e`](https://huggingface.co/RenormalizationTech/qwen2.5-7b-refusal-steering-experiment/tree/b9649ef186570f6159fbe95011024fa04655a89e) |
| [27B NF4 base](https://huggingface.co/RenormalizationTech/Qwen3.8-27B-tuneLLM-NF4) | [README](Qwen3.8-27B-tuneLLM-NF4/README.md) | [`586bb02218d196f485c321bbde3348677c63fc6a`](https://huggingface.co/RenormalizationTech/Qwen3.8-27B-tuneLLM-NF4/tree/586bb02218d196f485c321bbde3348677c63fc6a) |
| [27B GGUF](https://huggingface.co/RenormalizationTech/Qwen3.8-27B-tuneLLM-GGUF) | [README](Qwen3.8-27B-tuneLLM-GGUF/README.md) | [`b4351e7d6f521b71cb3842116cdff3cae00d7317`](https://huggingface.co/RenormalizationTech/Qwen3.8-27B-tuneLLM-GGUF/tree/b4351e7d6f521b71cb3842116cdff3cae00d7317) |
| [27B PRL abstract QLoRA](https://huggingface.co/RenormalizationTech/Qwen3.8-27B-tuneLLM-PRL-Abstract-QLoRA) | [README](Qwen3.8-27B-tuneLLM-PRL-Abstract-QLoRA/README.md) | [`3bb77b924555bdcc9f92b9c4aa6cf32d63340fbc`](https://huggingface.co/RenormalizationTech/Qwen3.8-27B-tuneLLM-PRL-Abstract-QLoRA/tree/3bb77b924555bdcc9f92b9c4aa6cf32d63340fbc) |
| [27B QASPER QLoRA](https://huggingface.co/RenormalizationTech/Qwen3.8-27B-tuneLLM-QASPER-QLoRA) | [README](Qwen3.8-27B-tuneLLM-QASPER-QLoRA/README.md) | [`95359154d2b9d94914977462e559c409c1f549c6`](https://huggingface.co/RenormalizationTech/Qwen3.8-27B-tuneLLM-QASPER-QLoRA/tree/95359154d2b9d94914977462e559c409c1f549c6) |
| [27B sarcasm GCAD controls](https://huggingface.co/RenormalizationTech/qwen38-27b-sarcasm-gcad-controls) | [README](qwen38-27b-sarcasm-gcad-controls/README.md) | [`877e4b51e654e7410f792858e93fee6727d48646`](https://huggingface.co/RenormalizationTech/qwen38-27b-sarcasm-gcad-controls/tree/877e4b51e654e7410f792858e93fee6727d48646) |

The four 27B base/adapter entries include anonymous full-file download verification and GPU reload checks.
Their catalog entries distinguish the tested model commit from the later metadata-only
verification commit; runtime file hashes are unchanged.

The separately released [27B sarcasm controls](../docs/qwen38_sarcasm_release.md) are an experimental bank/runtime package with strength 0–16 and an owner-chosen demonstration default of 8. No strength passed all registered quality gates. The new controls were downloaded anonymously and passed an isolated/offline GPU replay against the staged package. The base was re-hashed from its pinned-revision cache, whose anonymous full download was recorded on 2026-11-17; it was not freshly transferred for this check. [Public verification](../results/qwen38_sarcasm_public_verification_01.json) binds the exact revision, inventory and matching replay receipts.

## What lives where

- `releases/<HF-repo>/README.md`: tracked current card mirror for every public package.
- `artifacts/releases/<HF-repo>/`: ignored local copy containing every published file
  at the catalog revision. Local tooling/cache extras are excluded from that claim.
- `.hf/hub/`: upstream model snapshots, separate from our releases.
- `outputs/`: experiment evidence, old staging, metadata backups and verification details.

The two 7B packages contain activation banks/directions and runtimes. Obtain
their pinned base checkpoint separately; they are not full 7B model weights.
The original 27B NF4 checkpoint remains at `artifacts/qwen38-27b-nf4-01/`;
its public release payload is separately indexed under `artifacts/releases/`.
The 27B PRL abstract adapter is public as an experimental negative result, with
complete task and retention evaluation. The QASPER adapter is public after passing
the fixed custom text-only task and retention gates; its baseline cap failure
and completion amendment remain disclosed.

Per-folder metadata differs because it was captured at different release times;
a card-only tracked folder is intentional. Original licenses, export manifests,
provenance and dependency pins remain historical snapshots. Their README hashes
may describe the original card; use the catalog/card-polish receipt for current
card identity. The new 27B verification receipts bind the current cards and inventories.
Do not treat an old manifest as a new upload instruction.

## Retired release

The [GSM8K QLoRA card](Qwen3-4B-tuneLLM-GSM8K-QLoRA/README.md) is retained as
historical evidence. Its HF repository was deleted after verified archival; it
is excluded from the active releases. The local payload remains under
`artifacts/releases/Qwen3-4B-tuneLLM-GSM8K-QLoRA/`, and revision snapshots under
`outputs/hf-transition-20261116-01/gsm8k-archive/`. See the
[transition receipt](../results/hf_visibility_transition_01.json) and
[negative learning result](../results/qlora_real_01.json).

## History and verification

[HISTORY.md](HISTORY.md) preserves the former index verbatim, including earlier
private/unpublished labels and superseded "current" revisions. It is a dated
publication history. Current status is in this index and the catalog.

The [cleanup receipt](../results/repository_cleanup_01.json) records the local
mirror reconciliation: exactly nine small metadata files refreshed across the
three 4B packages; weights, runtimes and original manifests unchanged. All 133
published files were checked against pinned HF file identities. No remote
repository, experiment result or release dependency was changed.

For use/download instructions, start with the [reproduction guide](../docs/reproduce.md).
Publication tooling lives in [scripts/](../scripts/README.md); preparing a new
release requires a new reviewed payload, not reuse of this index as a manifest.
