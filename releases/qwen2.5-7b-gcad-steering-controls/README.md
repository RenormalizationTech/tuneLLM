---
license: apache-2.0
base_model: Qwen/Qwen2.5-7B-Instruct
---
# Qwen2.5-7B GCAD steering controls

## Portfolio summary

**Trait-steering experiment:** inference-time GCAD banks induce profanity,
impoliteness and sarcasm on the pinned Qwen2.5-7B-Instruct checkpoint. In the
32-question sarcasm confirmation, 29/32 steered answers scored at least 50/100
versus 0/32 baseline; coherence fell from 89.59 to 70.72. The separate joint study
retained 0/5 combined candidates. Individual effects, quality costs and coupling
are reported together. Banks, runtime and measured results are unchanged by this
documentation update. The joint manifest is refreshed for the updated card.

This artifact is part of **[tuneLLM](https://github.com/RenormalizationTech/tuneLLM)**, a public research portfolio
covering quantization, LoRA/QLoRA, refusal steering and trait steering.
[Results and saved examples](https://github.com/RenormalizationTech/tuneLLM/blob/main/docs/showcase.md) ·
[Reproduction and environment setup](https://github.com/RenormalizationTech/tuneLLM/blob/main/docs/reproduce.md) ·
[Methods and references](https://github.com/RenormalizationTech/tuneLLM/blob/main/docs/references.md).

Related releases: [GGUF](https://huggingface.co/RenormalizationTech/Qwen3-4B-tuneLLM-GGUF),
[NF4 base](https://huggingface.co/RenormalizationTech/Qwen3-4B-tuneLLM-NF4),
[physics QLoRA](https://huggingface.co/RenormalizationTech/Qwen3-4B-tuneLLM-PRL-Writing-QLoRA),
[trait controls](https://huggingface.co/RenormalizationTech/qwen2.5-7b-gcad-steering-controls),
[refusal experiment](https://huggingface.co/RenormalizationTech/qwen2.5-7b-refusal-steering-experiment).
Each release has its own verified environment; activation steering and adapter
training use different mechanisms.

Separate 27B work: the **text-only NF4 load/save/reload smoke is complete** on
Qwen3.8-27B, using two simple prompts. It is a functional check, not a quality
benchmark. A one-step cache check exceeded its numerical tolerance; that result
is retained. The NF4 checkpoint remains local; 27B sarcasm steering and QLoRA
training are planned. [Recorded scope](https://github.com/RenormalizationTech/tuneLLM/blob/main/results/qwen38_27b_nf4_execution_01.json).



Dynamic inference-time banks for individual profanity, author impoliteness and
sarcasm controls on the pinned
[Qwen2.5-7B-Instruct base](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct).
These are activation-intervention banks, not base-model weights or LoRA adapters.
Base weights are obtained separately. This bundle contains banks, runtime and
aggregate evaluation; raw response logs and private grader records are not bundled.
Project code, protocols, compact results and selected examples are available in
[the public repository](https://github.com/RenormalizationTech/tuneLLM).
Historical source tag `steering-controls-v1` and commit
`199cf41ac9db9024b14c9afddaea8e060af5b14b` identify the original implementation.


| 7B controller | Own evaluation design | Trait baseline → GCAD (random) | Coherence baseline → GCAD (random) |
|---|---|---|---|
| PF, b = 5 | 32 ordinary held-out questions, greedy cap 160 | 0 → 31.125 (0); 14/32 lexicon hits | 85.34375 → 82.6875 (85.6875) |
| SARC, b = 2 | 32 ordinary held-out questions, greedy cap 160 | .0625 → 63.78125 (0); 29/32 ≥ 50 | 89.59375 → 70.71875 (87.71875) |
| Impolite, b = 3.5 | 60 author first replies on 13 unique questions, sampled cap 500 | 1.61667 → 77.36667 (not measured) | 82.68333 → 75.9 (not measured) |

The designs and gates differ. Impolite passed its author qualitative contrasts,
with no original random arm; PF/SARC passed their registered held-out gates.
SARC has a substantial 18.875-point coherence loss and only 1.125 points of gate margin. Its
impoliteness mean 38.75 and correlations are descriptive, not independence or
causality evidence. One Claude judge family and reconstructed PF/SARC inputs
limit validation; coherence is not correctness or helpfulness. These controls
intentionally induce potentially offensive style. Apache-2.0 is the licence;
no additional use restriction is imposed here.

## Validation

The inference core is byte-identical to validated staging 02: 3 real pinned
base zero-coefficient checks, 6 forwards total. No staging 03 GPU rerun is
claimed. Current manifest validation supersedes historical deferred fields in
archival profiles. [METHODS](METHODS.md) records the repair history and scope.

## Pinned local inference

Requirements: Python3.11, torch==2.6.0+cu124, transformers==4.52.3,
accelerate==1.7.0, safetensors and tqdm.
Use base `Qwen/Qwen2.5-7B-Instruct` revision `a09a35458c702b33eeacc393d103063234e8bc28` from a local snapshot whose basename equals
that commit. The example uses local-files-only and no remote model code.
Tokenizer must use EOS151645 as PAD and left padding. Base weights are fetched
separately from their upstream repository, not supplied here.

Choose an immutable **HF_REV** of this bank repository. Read that revision's
commit message `Release manifest SHA256: H` and use H as the expected
manifest hash. This verifies file consistency for the pinned revision you choose;
it does not independently prove who published them. Publisher authenticity
rests on the HF account. Download this repository at HF_REV, never an unpinned moving
revision. No GitHub checkout is required to run the downloaded bundle. Use real control files (the loader
rejects control symlinks); a pinned local-directory download provides them:

```python
import os
from huggingface_hub import snapshot_download
PINNED_HF_SHA = os.environ["HF_REV"]
snapshot_download("RenormalizationTech/qwen2.5-7b-gcad-steering-controls",
                  revision=PINNED_HF_SHA, local_dir="controls")
```

Then from the downloaded bundle root, with H from that commit message in
BANK_MANIFEST_SHA and the pinned upstream base snapshot in LOCAL_BASE_SNAPSHOT:

```bash
cd controls
python -m runtime.example --snapshot "$LOCAL_BASE_SNAPSHOT" --trait profanity --manifest-sha256 "$BANK_MANIFEST_SHA" --zero-selftest
python -m runtime.example --snapshot "$LOCAL_BASE_SNAPSHOT" --trait profanity --manifest-sha256 "$BANK_MANIFEST_SHA" --prompt "Describe a rainy afternoon in two sentences."
```

Use `impolite` or `sarcasm` for the other isolated banks. The saved settings are
bank/task/decoding-specific, not comparable doses or jointly validated controls.
The standalone inference interface runs from this bundle. The public GitHub
repository documents the experimental pipeline; some archived raw runs, judge
records and local prerequisites remain outside the public release. See the
[reproduction guide](https://github.com/RenormalizationTech/tuneLLM/blob/main/docs/reproduce.md)
for portable checks versus historical full-study reproduction.

## Related outcomes

[Methods, source attribution and refusal closures](METHODS.md),
[4B aggregate](related_results/qwen3_4b_fixed_transfer_01.public.json), and
[4B closeout](related_results/qwen3_4b_fixed_transfer_closeout.md) are included.
All 3 fixed-transfer 4B traits were induced, but none met the quality gates. No 4B bank or
new 4B HF steering repository is published. This package contains no refusal controller. Outcomes are model-specific; checkpoint differences do not establish
a model-size effect. No 4B retuning or joint claim; separate 7B joint outcomes are below; no independence claim. The physics
adapter and NF4 base are now public and separate from this package. Original GCAD and Persona Vectors
attribution is preserved in LICENSE/NOTICE; public author sources are listed
in the included method notes.

## Additive joint-controls release

The original release_manifest.json is byte-identical and its full list describes
the original revision `763865908e5e94a020606b31ab66185360f7dd44`, not the updated root README/METHODS.
Its eight original critical bank/profile/core/loader hashes remain valid, so
the isolated loader still works at this HEAD. The new joint manifest covers
current cards, joint files and shared payload files. Read BOTH lines from this
pinned HF commit message: `Release manifest SHA256: H1` and `Joint manifest SHA256: H2`.
H1 is the original manifest; H2 is joint_controls/manifest.json. Optional root
.gitattributes and .git metadata, downloader .cache and Python __pycache__ are
not payload files and are excluded only from consumer inventory checks. Root
preserves any Hub-managed .gitattributes and verifies it separately unchanged. These check
file consistency for the pinned revision you choose, not independent publisher identity.

[Joint guide](joint_controls/GUIDE.md), [all 12 arm metrics](joint_controls/metrics.csv),
and [sanitized aggregate](joint_controls/results.json), and [post-hoc lexical audit](joint_controls/lexical_audit.json) report all outcomes, paired
intervals, calibration/duplicate noise, cross-effects and confounding limits.
PF means friendly-rubric swearing, not literal profanity incidence: a lower friendly PF score
does not establish suppressed swearwords. The lexical addendum is explicitly post-hoc,
using an unchanged narrow proxy; all original guide/gate verdicts stay unchanged.
No new base weights, LoRA, raw questions, answers, token arrays or private grading reasons are included.
The NF4 base and PRL Writing QLoRA adapter are now public; they are separate
artifacts and are not bundled here. This package contains no refusal controller.
