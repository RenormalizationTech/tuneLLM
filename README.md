# tuneLLM

A hands-on project for learning open-source LLM quantization and fine-tuning.
Research into behavioral representations and activation steering is an optional
extension.

The current reusable workflow is documented in
[docs/model_workflow.md](docs/model_workflow.md). Its next complete exercise uses
a saved NF4 base, a balanced corpus of condensed-matter theory PRLs, and writing
evaluations for titles, abstracts and introductions, with one BF16 LoRA control.
The verified standalone base is recorded in
[results/nf4_checkpoint.json](results/nf4_checkpoint.json).

Initial workflow:

1. Run a reproducible small-model inference baseline and fixed evaluation.
2. Convert to GGUF, quantize to Q8_0 and Q4_K_M, and measure the tradeoffs.
3. Fine-tune with QLoRA on a small licensed dataset; reload and evaluate the adapter.
4. Publish selected validated artifacts and model cards to our existing
   Hugging Face organization.

The GitHub repository and Hugging Face organization already exist. The first
artifacts are published: [public GGUFs](https://huggingface.co/RenormalizationTech/Qwen3-4B-tuneLLM-GGUF)
and a [private experimental adapter](https://huggingface.co/RenormalizationTech/Qwen3-4B-tuneLLM-GSM8K-QLoRA).
The organization is
[RenormalizationTech](https://huggingface.co/RenormalizationTech).
See [plan.md](plan.md) for priorities and the optional research backlog, and
[docs/execution.md](docs/execution.md) for environment setup and observed results.

The first model is `Qwen/Qwen3-4B`, pinned in `configs/baseline.json`.
Two BF16 runs produced identical tokens and scored 7/8 on a basic pipeline smoke
fixture. This fixture is not a capability benchmark; compact results are in
[results/initial_baseline.json](results/initial_baseline.json).
BF16, Q8_0 and Q4_K_M GGUF files passed CPU prompt comparisons and short GPU
benchmarks; sizes, checksums and measurements are in
[results/initial_quantization.json](results/initial_quantization.json).

A two-step QLoRA smoke run also passed fresh-base adapter reload with identical
generated tokens. Results are in
[results/initial_qlora_smoke.json](results/initial_qlora_smoke.json). This validates
the workflow; it does not establish a useful fine-tuned model.

A bounded run on 1,024 GSM8K examples completed 64 steps with a frozen,
committed protocol. The selected adapter improved answer-format compliance
from 107/128 to 128/128 but reduced numeric accuracy from 121/128 to 101/128.
It is documented as an experimental learning result. See
[the full result](results/qlora_real_01.json),
[the fixed protocol](docs/real_run_protocol.md), and
[release details](releases/README.md). Uploaded file hashes and pinned-revision
load checks passed; see [publication verification](results/huggingface_publication.json).

After installing the environment described in the execution record:

```bash
.venv/bin/python -m scripts.env_check --output outputs/environment.json
.venv/bin/python -m scripts.download_model --config configs/baseline.json
.venv/bin/python -m scripts.hf_baseline --output outputs/my-baseline
```

Run GPU commands with access to the NVIDIA devices. The baseline defaults to
physical GPU 1 and refuses to overwrite an existing run directory.

Large local files belong in ignored `artifacts/`, `data/`, `outputs/`, and `.hf/`
directories. Source code, configurations, small evaluation fixtures, and compact
results belong in Git. The Hugging Face cache path will be set explicitly by the
execution workflow; merely creating `.hf/` does not configure the cache.

Roles: coordinating Codex manages planning and acceptance, Claude advises on
planning and science, and a dedicated Codex worker implements and tests.
