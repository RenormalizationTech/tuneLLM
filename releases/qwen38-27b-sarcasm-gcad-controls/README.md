---
license: apache-2.0
base_model: RenormalizationTech/Qwen3.8-27B-tuneLLM-NF4
pipeline_tag: text-generation
library_name: pytorch
tags: [gcad, steering, sarcasm, experimental, nf4]
---
# Experimental Qwen3.8-27B sarcasm GCAD controls

**Experimental research/demonstration only. No tested strength passed all registered acceptance gates
(`exploratory_null_through_b16`). This is not a validated controller.** The owner-chosen default
b=8 failed the relative-coherence gate. Strong settings can produce mocking or demeaning text.
This is not a production deployment recommendation or an official hosted Inference API integration.

## Standalone local use

Requires Python 3.13.3 and the recorded Torch 2.12.0+cu130 / CUDA 13.0 build, plus
the exact packages in `requirements.txt`. Install that Torch build first, then install the
remaining requirements. Optional FLA/causal-conv1d/kernels packages must remain absent, as studied.

Download this controls repository at a **specific public commit** and use the release-manifest
SHA256 printed in that commit message as an external trust anchor. Do not derive the trust
anchor from an unchecked local manifest. This verifies byte consistency, not publisher identity.
The actual controls commit is assigned at publication; it cannot be self-embedded in its own files.

Explicitly download the controls using that public commit, supplied as `HF_REV`:

```python
import os
from huggingface_hub import snapshot_download
snapshot_download("RenormalizationTech/qwen38-27b-sarcasm-gcad-controls", revision=os.environ["HF_REV"], local_dir="qwen38-controls")
```

Then enter `qwen38-controls` to run the commands below. Set `MANIFEST_SHA256` from the
same commit message. Downloads are explicit user actions, not runtime side effects.

The base snapshot is pinned to `RenormalizationTech/Qwen3.8-27B-tuneLLM-NF4` at `586bb02218d196f485c321bbde3348677c63fc6a`. One explicit download example:

```python
from huggingface_hub import snapshot_download
snapshot_download("RenormalizationTech/Qwen3.8-27B-tuneLLM-NF4", revision="586bb02218d196f485c321bbde3348677c63fc6a", local_dir="../qwen38-base")
```

From the downloaded controls directory, set `MANIFEST_SHA256` to the external release digest:

```bash
python -m runtime.example --snapshot ../qwen38-base --manifest-sha256 "$MANIFEST_SHA256" --strength 8 --prompt "How does a compass work?"
```

The runtime itself performs **no implicit downloads**. Real local_dir files are preferred.
Both legacy one-hop HF snapshot-to-repository-blob links and the pinned HF Hub2.4
two-hop shared-blob cache are supported and hash-verified. The shared store must have its
regular version-1 ownership marker and regular prefix directories/files; directory links,
third links and outside-cache targets are refused. All six base runtime
files and every controls payload file are checked. Consumer `.git`, `.cache`, `.gitattributes`
and generated runtime `__pycache__` metadata are excluded from the payload inventory only.
About 18 GB of model weights must fit on one BF16-capable CUDA device; no CPU/offload fallback.

Use `--strength 0` for ordinary generation without steering hooks, `8` for the experimental
demonstration, or `16` for extreme steering. Finite strengths in [0,16] are accepted; intermediate
values other than evaluated 2/4/8 are untested. Booleans/nonfinite/out-of-range strengths, padding,
batch sizes above one, nested or concurrent use, and strength changes mid-call are refused.
The supported example is one English user turn, no system prompt, greedy decoding, repetition
penalty 1.05, native caching, seed 42 and at most160 generated tokens.

```python
from runtime import load_runtime
controls = load_runtime("../qwen38-base", manifest_sha256="EXTERNAL_RELEASE_MANIFEST_SHA256")
answer = controls.generate("How does a compass work?", strength=8, max_new_tokens=160)
print(answer["text"])
```

Loading base weights alone does not enable GCAD. The bank is derived from that model's own
generations. Only the exact checkpoint/software/GPU stack was mechanically checked;
other hardware may generate different tokens. See `METHODS.md`, `evaluation.json`, and `metrics.csv`.
Functional package verification is recorded separately; scientific results and limitations appear below.

## What this controls package does

GCAD (Gated Cropped Attention-Delta) adds a learned style direction to selected attention outputs during generation. The direction is extracted from matched sarcastic and sincere generations. A token-dependent gate varies the addition as the model generates its reply. **The base model weights stay unchanged; strength is adjustable without retraining.**

This is an experimental inference control for the pinned Qwen3.8-27B NF4 base. It needs both the steering bank and compatible inference runtime; loading the base weights alone does not enable GCAD.

## Adjusting sarcasm strength

The CLI exposes `--strength`, which sets the runtime's base coefficient `base_coef` (b). Choose it before starting a fresh generation. Keep the bank, six selected layers and gate scale fixed to reproduce this profile.

Strength 0 is off. The interface accepts finite strengths from 0 through 16, with **8 as the owner-chosen experimental demonstration default** and 16 as the extreme preset. Strength 8 failed the registered relative-coherence gate; it is not a validated balanced setting.

| Strength / arm | Sarcasm | Coherence | Impoliteness | Coherence <50 | Registered gates |
| --- | ---: | ---: | ---: | ---: | --- |
| Baseline | 0.00 | 84.28 | 1.91 | 0/32 | Reference |
| Prompted reference | 55.56 | 76.00 | 52.09 | 0/32 | Reference |
| 2 | 0.09 | 84.16 | 2.22 | 0/32 | FAIL: trait_gain, individual_ge50; essentially no judged sarcasm |
| 4 | 4.97 | 83.44 | 6.75 | 0/32 | FAIL: trait_gain, individual_ge50 |
| 8 | 88.50 | 58.94 | 70.72 | 6/32 | FAIL relative coherence: 58.94 < 64.28; owner-chosen EXPERIMENTAL demonstration default |
| 16 | 95.16 | 34.59 | 81.53 | 30/32 | FAIL: coherence_absolute, coherence_relative; extreme setting |
| Random16 | 0.00 | 84.28 | 2.09 | 0/32 | Descriptive only; setting-matched, not realised-shift-matched |

Higher settings can amplify sarcasm and impoliteness while reducing how directly and clearly the answer addresses the question. Strength is not a sarcasm probability, and doubling it does not guarantee twice the effect. Only 0 was mechanically checked as off, and 2/4/8/16 were behaviourally evaluated; intermediate settings have not been evaluated. A balanced default is not established.

Internally, the runtime adds `2 * b * sigmoid(1.5 * (demand - centre)) * delta` at each selected layer/token. The gate changes with the model state, so actual shifts vary across tokens and settings. Users normally tune b, rather than changing the bank, layers or gate scale.

## Measured scope

Results use 32 fresh, neutral, object-heavy English questions per arm, an AI judge and 160-token answers; all replies reached that cap. The bank pools 90 eligible matched pairs across five instruction pairs, using the base model's own generations. It captures their whole wording contrast, not an isolated pure sarcasm feature. Steered arms use a single user turn with no system prompt, greedy decoding, repetition penalty 1.05, batch one and native KV caching. Multi-turn conversations, added system prompts, sampling, other languages and answers beyond 160 tokens are untested. Baseline sarcasm/coherence was 0.00/84.28; an explicit sarcasm instruction gave 55.56/76.00. Strong settings produce sarcasm, but no tested strength passes every registered trait/quality gate. This package should be presented as an exploratory style-control demonstration, not a validated general-purpose controller.

At strength 16, a setting-matched random direction scored zero sarcasm. It uses one random seed set. Its realised shifts differ along its generated trajectory; it is not an energy-matched control, and there was no random control at strength 8 or the lower strengths. Coherence loss is an observed tradeoff, not a defining requirement of sarcasm. Impoliteness is a separate measured outcome; sarcasm was not shown separable from rudeness. The historical coherence rubric separates coherence from correctness, but its hallucination wording makes that separation imperfect. The old `development_null` remains unchanged; this experiment does not establish dose as its cause.

For equations, architecture details and measured shift norms, see the [GitHub theory and tuning guide](https://github.com/RenormalizationTech/tuneLLM/blob/main/docs/qwen38_sarcasm_gcad_guide.md). The [full experiment record](https://github.com/RenormalizationTech/tuneLLM/blob/main/docs/qwen38_sarcasm_followup_execution.md) includes all gates and limitations.


## Intended use and attribution

This is for experimental style-control research and creative demonstrations. Strong settings can produce mocking or demeaning text; the study does not establish a general-purpose quality-preserving controller. Only the pinned NF4 model and recorded one-GPU software stack were tested.

This controls package preserves the Apache-2.0 GCAD port licence/NOTICE and the base model's applicable licence terms. The bank is derived from that base model's own generated responses; raw generations and judge sessions are not part of this public payload.
