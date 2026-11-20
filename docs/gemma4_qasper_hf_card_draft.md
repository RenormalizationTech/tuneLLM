---
license: apache-2.0
library_name: peft
base_model: google/gemma-4-31B-it
base_model_relation: adapter
pipeline_tag: text-generation
tags:
  - qasper
  - qlora
  - gemma4
  - text-only
  - experimental
---

# Gemma 4 31B QASPER QLoRA adapter

This adapter improves short answer and JSON-format matching in one registered
custom text-only QASPER study. **The strict base score is a format floor:** all
403 baseline outputs use JSON code fences, with real newlines, which the strict
parser rejects. The improvement combines format compliance with shorter,
reference-style answers. It does not establish broad scientific comprehension
or superiority over other models.

This repository contains the byte-exact selected step-61 LoRA weights and a local
inference helper. It contains no base weights, merged model, NF4 base checkpoint,
training corpus, optimizer state, scorer or hosted inference endpoint.

## Recorded results and limits

Scores below are on a **reused project evaluation set**, already used for Qwen,
and a custom text-only projection of QASPER. This is neither a fresh heldout set
nor an official whole-benchmark result. Public papers may have been in pretraining.

| Primary answer F1, n=309 | Base | Prompt | Adapter |
| --- | ---: | ---: | ---: |
| Strict, registered gate | 0.0000 | 0.0000 | 0.6401 |
| Lenient, descriptive | 0.3338 | 0.3338 | 0.6401 |

Prompt selection compared base/verify/schema on 35 development questions. All
three tied at strict zero; base won the predefined tie rule. Prompt is therefore
an alias of base, not an independent successful prompting comparison.

The lenient primary answer-F1 gain is **0.3063**, with a paired paper-bootstrap
95% interval **[0.2513, 0.3617]** (a diagnostic, not a gate). Median non-abstaining
answer length fell from **35 to 3 words**. The overlap gain can reflect concision
and reference matching; it does not demonstrate an underlying comprehension gain.
**No evidence-selection superiority is claimed:** lenient evidence is essentially
unchanged. The evidence-strict primary subset has n=267; all-sampled n=403.

| Fixed retention task | Base | Adapter | Allowed net loss |
| --- | ---: | ---: | ---: |
| General MC | 108/128 | 105/128 | 5 |
| GSM8K fallback numeric | 126/128 | 126/128 | 5 |
| IFEval strict | 26/32 | 26/32 | 3 |

All registered task and retention gates passed. Retention limits are small-sample
point-estimate checks, **not statistical noninferiority guarantees**. The study
used one configuration and seed. There is no broad Gemma-versus-Qwen ranking.
See `evaluation_summary.json` for compact aggregates and exact provenance hashes.
The [pinned study report](https://github.com/RenormalizationTech/tuneLLM/blob/15f0d1d4d031a174027fe70035e2cc553f2d4d19/docs/gemma4_qasper_results.md)
and [independent final audit](https://github.com/RenormalizationTech/tuneLLM/blob/9323a83cef3f83901b31320a2bc6a3fc0a6f7ae2/results/reviews/gemma4_qasper_final_audit_review_02.md)
explain the registered gates and interpretation.

## Training and studied runtime

One epoch over 482 selected training records, gradient accumulation 8, 61 AdamW
steps, seed 42. Checkpoints 21/41/61 were compared by token-weighted response NLL
on 101 development records; step 61 was selected. LoRA rank 16 / alpha 32 / dropout 0.05,
learning rate 5e-5, weight decay 0.01, all 410 decoder projections (820 tensors,
122,429,440 parameters). The adapter config changes only public base identity
and revision normalization; all other fields and weight bytes are preserved.

Training used 4.84 GPU-hours. Whole-study wall time was 8.96 hours and summed GPU
time 15.60 hours. These measurements include coexistence with an idle resident
CUDA context and are not exclusive-device timing claims.

The pinned base is `google/gemma-4-31B-it` at
`842da3794eaa0b77d5f08bae87a17459d91ff475`. The helper uses the text decoder
`Gemma4ForCausalLM` with `config.text_config` and explicit
`^model.language_model.` key mapping. It applies NF4 double quantization/BF16
compute, PEFT default FP32 floating-weight preparation, SDPA, and outer BF16
autocast. It preserves the frozen `torch.inference_mode()` guard plus eval mode,
singleton cached greedy generation, repetition penalty 1, no thinking, EOS [1,106],
cap 512 and context 8192. The same prepared LoRA wrapper is disabled for baseline.
There is no truncation or silent alternate arithmetic profile.

Generic PEFT automatic loading does not reproduce this studied text-only key
mapping and FP32 floating-weight preparation. Use the bundled helper for the
reported loading and generation method.

## Local use

Use the exact versions in `requirements.txt`, including Torch 2.12.0+cu130 (the
matching CUDA 13 wheel), Transformers 5.23.0 and PEFT 0.21.0. In a fresh Python
environment, install the official CUDA 13 wheel and then the remaining pins:

```bash
python -m pip install --index-url https://download.pytorch.org/whl/cu130 'torch==2.12.0+cu130'
python -m pip install -r requirements.txt
```

One explicitly visible GPU is required; the study used a 48 GB RTX A6000.
The helper never downloads a model and never enables `trust_remote_code`.

First explicitly download the adapter at a **specific public commit** and the
base at its fixed revision. Obtain `ADAPTER_REVISION` and `MANIFEST_SHA256` from
the existing [release catalog](https://github.com/RenormalizationTech/tuneLLM/blob/main/releases/catalog.json)
and its linked verification record (`revision` and `release_manifest_sha256`),
or the reviewed release commit message. The Gemma catalog entry is added after
public package verification; the [reproduction guide](https://github.com/RenormalizationTech/tuneLLM/blob/main/docs/reproduce.md)
also links the release records.
The adapter's own commit cannot be embedded in that same commit's card. A hash
from the same unchecked local files is not an independent trust anchor.

```python
from huggingface_hub import snapshot_download

adapter = snapshot_download(
    "RenormalizationTech/Gemma-4-31B-tuneLLM-QASPER-QLoRA",
    revision="ADAPTER_REVISION", local_dir="gemma-qasper", token=False,
)
base = snapshot_download(
    "google/gemma-4-31B-it",
    revision="842da3794eaa0b77d5f08bae87a17459d91ff475",
    local_dir="gemma-base", token=False,
    allow_patterns=[
        "config.json", "generation_config.json", "chat_template.jinja",
        "tokenizer.json", "tokenizer_config.json", "model.safetensors.index.json",
        "model-*.safetensors",
    ],
)
```

Regular `local_dir` downloads are preferred. Verified legacy HF snapshot links
and HF2.4 two-hop shared blobs are supported; arbitrary file/directory links and
third hops are rejected. Every computational base file is SHA256 checked.

Create `../question.txt` with your full paragraph-ID evidence context, question and
requested JSON fields `answer`, `unanswerable` and `evidence`. This helper does
not retrieve a paper or invent missing evidence. Run from the adapter directory;
keep all input files outside that reviewed bundle, in its parent directory:

Use the exact frozen formatter rather than paraphrasing the instruction. Supply
only the context paragraphs you intend to include, in display order; the helper
does not select paragraphs, strip whitespace or rewrite their text:

```python
import sys
sys.dont_write_bytecode = True
from pathlib import Path
from runtime import format_question

user = format_question(
    [{"id": "p000000", "section": "Methods", "text": "This invented sample contains blue dye."}],
    "What does the invented sample contain?",
)
Path('../question.txt').write_text(user, encoding='utf-8')
```

That invented example produces this verbatim user prompt:

```text
Answer using only the numbered paper paragraphs. Return only JSON with keys "answer" (string), "unanswerable" (boolean), "evidence" (array of paragraph IDs). For insufficient evidence return an empty answer and evidence array with unanswerable=true.

[p000000] Methods
This invented sample contains blue dye.

Question: What does the invented sample contain?
```

The snippet writes `../question.txt` outside the adapter bundle. Alternatively,
put `paragraphs` and `question` in `../question.json`, also outside the bundle,
and use `--question-json ../question.json` instead of `--prompt-file` below.

```bash
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python -B -m runtime.example --bundle . --base ../gemma-base \
  --manifest-sha256 MANIFEST_SHA256 --prompt-file ../question.txt
```

`-B` keeps the reviewed directory free of generated Python bytecode. Add
`--baseline` for the matched disabled-adapter wrapper. Generated control markers
remain visible to callers; only a terminal EOS1/106 is removed from decoded text.
Functional local/public package verification is recorded separately from the
scientific results. Public adapter verification reuses an exact hash-verified
base cache; it does not claim a fresh full base download.

## License and attribution

The pinned official upstream card and API declare **Apache-2.0**, and Google's
[linked Gemma4 license page](https://ai.google.dev/gemma/docs/gemma_4_license)
provides the Apache2 text. The pinned upstream repository contains no separate
LICENSE or NOTICE file. `LICENSE` contains the actual extracted license text;
`upstream_license.json` binds the fetched evidence and `NOTICE` records attribution
and the two-field config change. Review the
[pinned upstream card](https://huggingface.co/google/gemma-4-31B-it/blob/842da3794eaa0b77d5f08bae87a17459d91ff475/README.md).
