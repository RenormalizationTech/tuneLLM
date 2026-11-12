# tuneLLM

A hands-on project for learning open-source LLM quantization and fine-tuning.
Research into behavioral representations and activation steering is an optional
extension.

Initial workflow:

1. Run a reproducible small-model inference baseline and fixed evaluation.
2. Convert to GGUF, quantize to Q8_0 and Q4_K_M, and measure the tradeoffs.
3. Fine-tune with QLoRA on a small licensed dataset; reload and evaluate the adapter.
4. Publish selected validated artifacts and model cards to our existing
   Hugging Face organization.

The GitHub repository and Hugging Face organization already exist. No Hugging
Face artifact repository has been created and no artifacts have been uploaded.
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
