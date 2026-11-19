# Use the published 27B sarcasm controls

The experimental [HF controls package](https://huggingface.co/RenormalizationTech/qwen38-27b-sarcasm-gcad-controls) exposes **tunable inference-time GCAD strength**. It contains the bank and standalone runtime; the [NF4 base](https://huggingface.co/RenormalizationTech/Qwen3.8-27B-tuneLLM-NF4) is downloaded separately. Model weights remain frozen.

Status: **published and independently verified, 2026-11-19**. This is an experimental demonstration, not a validated quality-preserving controller. The original `development_null` and follow-up `exploratory_null_through_b16` remain unchanged.

## Pinned release

| Component | Immutable identity |
| --- | --- |
| Controls | [`877e4b51e654e7410f792858e93fee6727d48646`](https://huggingface.co/RenormalizationTech/qwen38-27b-sarcasm-gcad-controls/tree/877e4b51e654e7410f792858e93fee6727d48646) |
| Base | [`586bb02218d196f485c321bbde3348677c63fc6a`](https://huggingface.co/RenormalizationTech/Qwen3.8-27B-tuneLLM-NF4/tree/586bb02218d196f485c321bbde3348677c63fc6a) |
| Controls manifest SHA256 | `cfaacd0efa5a7b22a40fab5305217ee2c642b46ba230a2eb8259d5a28639e54b` |
| Frozen research port SHA256 | `3d76d8147ffbe6f802b285d643b4363e09349f0dd14a5f6a2924545b5966abe3` |

The manifest digest also appears in the HF commit message. The [exact model-card mirror](../releases/qwen38-27b-sarcasm-gcad-controls/README.md) gives the supported environment and all measured rows. Hashes establish file consistency for the chosen revision, not independent publisher identity.

## Download and run

Use the recorded Python **3.13.3**, Torch **2.12.0+cu130 / CUDA 13.0**, Transformers **5.23.0** environment and the package's `requirements.txt` and `environment.lock.json`. The runtime checks installed versions, native modeling source, optional kernel availability and the six model-file hashes before model loading. It requires one explicit CUDA device and was tested on an RTX A6000 48 GB. A different library/kernel/hardware stack is not covered by this check.

With `huggingface_hub==2.4.0` available, run these explicit downloads from a parent working directory:

```python
from huggingface_hub import snapshot_download

snapshot_download(
    "RenormalizationTech/qwen38-27b-sarcasm-gcad-controls",
    revision="877e4b51e654e7410f792858e93fee6727d48646",
    local_dir="qwen38-controls", token=False,
)
snapshot_download(
    "RenormalizationTech/Qwen3.8-27B-tuneLLM-NF4",
    revision="586bb02218d196f485c321bbde3348677c63fc6a",
    local_dir="qwen38-base", token=False,
)
```

Match the pinned Torch/CUDA build, then install the remaining requirements from `qwen38-controls/requirements.txt`. The [NF4 environment guide](reproduce.md#separate-qwen38-27b-nf4-environment) describes the separate model environment. The base weighs about 17.65 GB; the controls are a small additional download. Real `local_dir` files and the supported legacy/shared HF cache layouts both work.

```bash
cd qwen38-controls
python -m runtime.example \
  --snapshot ../qwen38-base \
  --manifest-sha256 cfaacd0efa5a7b22a40fab5305217ee2c642b46ba230a2eb8259d5a28639e54b \
  --strength 8 \
  --prompt "How does a compass work?"
```

The runtime itself makes no implicit download. It renders one user turn with thinking disabled, generates greedily with repetition penalty 1.05 and native caching, and defaults to 160 output tokens. It refuses padding, batch sizes above one, concurrent/nested use and unsupported model/bank identities. The maximum supported output cap is 160, with a 2,048-token prompt-plus-generation guard and no silent truncation.

## Choose a strength

| Strength | Intended control | Measured sarcasm / coherence, fresh32 |
| --- | --- | ---: |
| 0 | Off; no steering hooks | 0.00 / 84.28 baseline |
| 2 | Essentially no judged sarcasm | 0.09 / 84.16 |
| 4 | Limited judged sarcasm | 4.97 / 83.44 |
| **8 (default)** | Owner-chosen experimental demonstration | **88.50 / 58.94** |
| 16 | Extreme experimental setting | 95.16 / 34.59 |

`--strength` accepts finite numbers in [0,16]. Intermediate values are configurable but behaviourally untested; the knob is not a sarcasm probability. Strength 8 failed the registered relative-coherence floor of **64.28**. Strength 16 failed both coherence gates, with 30/32 answers below coherence 50. Neither is a balanced or production-validated default. Impoliteness also rose to 70.72 and 81.53, respectively.

For an application, from the downloaded controls directory:

```python
from runtime import load_runtime

controls = load_runtime(
    "../qwen38-base",
    manifest_sha256="cfaacd0efa5a7b22a40fab5305217ee2c642b46ba230a2eb8259d5a28639e54b",
)
answer = controls.generate("How does a compass work?", strength=8)
print(answer["text"])
```

Choose strength before each fresh generation; reusing previously steered cached states does not undo earlier changes. The bank, six intervention sites and gate scale stay fixed. Added system prompts, multi-turn conversations, sampling, other languages and longer replies are untested. The [theory and tuning guide](qwen38_sarcasm_gcad_guide.md) explains extraction, both gates and why actual shifts vary by token.

## Verification and evidence

The controls were newly fetched at the pinned public revision. The base files were re-hashed from the pinned-revision cache; their anonymous full download is recorded on 2026-11-17 in the earlier NF4 verification. This check does not claim a new 17 GB base transfer.

The [publication receipt](../results/qwen38_sarcasm_huggingface_publication_01.json) verifies visibility and remote hashes. The [public verification record](../results/qwen38_sarcasm_public_verification_01.json) binds the anonymous downloads and the fresh isolated/offline GPU comparison to the [local passing receipt](../results/qwen38_sarcasm_release_local_smoke_02.json).

Release checks use two independent short probes at strengths 0/8/16 and three saved 160-token study trajectories. They compare logits, generated IDs and hook cleanup exactly against the frozen implementation. They are mechanics checks, not another quality evaluation. The known strict cached/full-prefix logit discrepancy remains unresolved.

The initial preflight rejected the pinned HF client's two-hop shared-blob layout before model loading. Its [failure receipt](../results/qwen38_sarcasm_release_local_smoke_01.json) is retained. A focused wrapper fix added real-cache and negative-path tests; the GCAD arithmetic, bank and study outcomes were unchanged. The release CPU suite now has **53 passing tests**; the portable regression suite has **397**. Both local and downloaded GPU checks passed: all six probe records and all three saved replays match exactly, with `public_local_mechanics_equal: true`. The two GPU verification processes took 238.63 and 239.76 seconds including loading and checks.

The review chain is [criteria](../results/reviews/qwen38_sarcasm_release_review_01.md), [initial source](../results/reviews/qwen38_sarcasm_release_review_02.md), [cache-fix source](../results/reviews/qwen38_sarcasm_release_review_03.md), [local GPU pass and upload clearance](../results/reviews/qwen38_sarcasm_release_review_04.md), and [public download/reload review](../results/reviews/qwen38_sarcasm_release_review_05.md). Initial source bindings were superseded explicitly, not overwritten.

The [scientific execution record](qwen38_sarcasm_followup_execution.md) and [follow-up review06](../results/reviews/qwen38_sarcasm_followup_review_06.md) retain the fixed gates, question-cluster activation analysis and all limitations. The original [HF draft](qwen38_sarcasm_hf_card_draft.md) is preserved byte-for-byte as a historical staging input; the actual published card and this guide describe the current package.
