# Reproduce the public portfolio

The repository contains research code and compact historical results. Raw runs,
judge journals, downloaded author code, model weights and some licensed prepared
datasets live in ignored local directories. A checkout does not recreate them.
Use a published inference bundle to try an artifact; historical runbooks describe
the original, frozen experiments and are not clean-checkout setup scripts.

## Small portable CPU checks

From the repository root, create a separate environment. Python 3.10 or 3.11 is
suitable for the original track. No Hugging Face or judge credentials are needed.

```bash
python3.11 -m venv outputs/public-cpu-env
outputs/public-cpu-env/bin/python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
outputs/public-cpu-env/bin/python -m pip install -r requirements-dev.txt
CUDA_VISIBLE_DEVICES="" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  outputs/public-cpu-env/bin/python -B -m pytest -q -p no:cacheprovider -m portable
```

`portable` selects the small, explicit CPU set in `tests/conftest.py`: scoring,
padding/masking, source-pair preparation, synthetic/random-tiny-model hook
mechanics and mocked publication/setup tests. It performs no pretrained model
generation, network call, GPU inference or judge call. Passing this slice is
not a claim that every historical integration test passed.

Default `pytest` selects this same portable profile through `pytest.ini`.
Other test modules are excluded **before import**, not reported as passes; the
terminal summary reports the module count. This keeps optional research-only
dependencies outside the portable setup. They are marked `research` when that
suite is collected. Opt in with `pytest -m research`, or override the default
with `pytest -m ""` for the entire archived suite. These commands require the
original local artifacts/environments and are not clean-checkout guarantees.
The archived suite was not made universally reproducible by these setup changes.

`artifact(*paths)` identifies explicitly declared local prerequisites. Such
tests skip if a named path is absent. If present, original hash/assertion checks
still run; changed artifacts fail. This marker does not automatically classify
every historical dependency. Some tests have additional existing tokenizer or
environment skips. `pytest -rs` shows skip reasons. No exception-to-success
conversion, automatic downloads or regenerated private fixtures are used.

The minimal GitHub Actions job runs only the portable slice with CPU Torch.
It does not certify published quality scores, full benchmark reproduction,
CUDA compatibility or judge behavior.

## Download the published refusal experiment (small files only)

This obtains the verified public inference package, not the original private
experiment directories. The helper pins public revision
`3e8a151667a8fef0ebba552a3f3d74317b1c093d` and validates all16 file hashes/sizes
against `results/refusal_7b_huggingface_publication_01.json`. It is anonymous and
does not execute the downloaded Python, load a model, or fetch base weights.

```bash
outputs/public-cpu-env/bin/python -B -m scripts.setup_refusal_artifact --download
outputs/public-cpu-env/bin/python -B -m scripts.setup_refusal_artifact
CUDA_VISIBLE_DEVICES="" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  outputs/public-cpu-env/bin/python -B -m pytest -q -m artifact tests/test_public_refusal_bundle.py
```

Network is used only with `--download`. The destination is
`artifacts/public/refusal-7b-experiment`; existing destinations are never
overwritten. The default command verifies an existing bundle without network.
If a download is interrupted, preserve it for inspection and use a fresh
`--destination artifacts/public/another-directory` rather than altering original
experiment records. This helper does not make archive-dependent tests pass by
recreating their private fixtures.

Follow that bundle's README, environment and distinct code/vector/dataset terms.
After separately obtaining its pinned base snapshot, its standalone CLI runs
baseline by default; `--steer` opts into the fixed experimental setting:

```bash
outputs/public-cpu-env/bin/python artifacts/public/refusal-7b-experiment/refusal_runtime.py \
  --snapshot /path/to/pinned/Qwen2.5-7B/snapshot --prompt "Explain a simple board-game strategy."
```

Inference requires substantial memory; the CPU test command above does not run
this example. Public bundle files are distinct from private staging inputs.

## Separate Qwen3.8-27B NF4 environment

Do not upgrade the original4B/7B environment in place. The successful text-only
smoke used Python 3.13.3, Torch 2.12.0+cu130, Transformers 5.23.0,
Accelerate 1.12.0, bitsandbytes 0.49.2 and huggingface_hub 2.4.0. The recorded GPU
was RTX A6000 with driver 580.178.04; these observations do not promise support
on every device. Transformers 5.23 requires Python >=3.11. Direct requirements
are separate in `requirements-qwen38-nf4.txt`.

Example independent environment for a compatible CUDA 13.0 system:

```bash
python3.13 -m venv outputs/qwen38-repro-env
outputs/qwen38-repro-env/bin/python -m pip install torch==2.12.0 --index-url https://download.pytorch.org/whl/cu130
outputs/qwen38-repro-env/bin/python -m pip install -r requirements-qwen38-nf4.txt
outputs/qwen38-repro-env/bin/python -B -m scripts.qwen38_nf4_smoke --inspect
```

`--inspect` reads tracked source metadata and installed official implementation
files; it does not need weights, CUDA or network. The original source download
can use the existing downloader with `configs/qwen38_27b_nf4.json` after you
explicitly choose to download the approximately55.6GB pinned base. No downloader
is invoked by inspection or tests. GPU smoke is separately opt-in with
`--execute-gpu`; see the runner's `--help` and the tracked execution results.
Stored NF4 reload uses `--load-saved`, without a fresh quantization configuration.
The reported cached/full-prefix numerical discrepancy remains a limitation;
CPU setup checks do not override it or establish cache equivalence.

## Completed 27B quantization and QLoRA workflows

The [27B evaluation and release plan](qwen38_release_plan.md) is complete:
NF4, one Q4_K_M export and independent QASPER QA/evidence and PRL grounded-abstract
adapters are public. Use the [release catalog](../releases/catalog.json) for
current metadata revisions and the separate GPU-tested revisions. The
[closeout report](../results/qwen38_science_summary_01.json) links all final
results, publication receipts and resource accounting.

For inference, follow the pinned public card for the selected format:

| Package | Verified loading instructions |
| --- | --- |
| Saved NF4 base | [NF4 card](../releases/Qwen3.8-27B-tuneLLM-NF4/README.md) |
| Q4_K_M GGUF | [GGUF card](../releases/Qwen3.8-27B-tuneLLM-GGUF/README.md) |
| QASPER adapter | [QASPER card](../releases/Qwen3.8-27B-tuneLLM-QASPER-QLoRA/README.md) |
| Experimental PRL adapter | [PRL card](../releases/Qwen3.8-27B-tuneLLM-PRL-Abstract-QLoRA/README.md) |

Adapter work used an isolated Python 3.13.3 environment with
[requirements-qwen38-science.txt](../requirements-qwen38-science.txt);
the published `environment.lock.txt` records the installed runtime. Both
adapters depend on the saved NF4 base at revision
`e7644d6f288cb8a9ba51856ca2c9fa68946f70fe`. Follow the documented FP32-casting
PEFT preparation and `PeftModel.from_pretrained` loading route. Do not pass a
new quantization configuration to the already quantized base. Native NF4
screens and PEFT-prepared science runs are distinct workflows. GGUF uses the
pinned llama.cpp backend and tokenized, thinking-disabled completion route;
its unmeasured default chat behavior is not the evaluated route.

The committed runners preserve the experiment implementation:
`qwen38_prepare_inputs`, `qwen38_train_science`, `qwen38_eval_science`,
`qwen38_qasper_analysis`, `qwen38_prl_review`, `qwen38_prl_diagnostics`,
`qwen38_retention`, and `qwen38_quant_screen`. Run modules from the repository
root. Read each module's `--help`, registered protocol and source snapshot
before any execution. Model runners require explicit execution options
(`--execute` or `--execute-gpu`, depending on the module); PRL auditing/judging
uses separate opt-in calls. None is invoked by
the portable CPU checks. Original source/input hashes, prepared datasets,
checkpoints, generated answers and judge journals remain in retained local
run directories; a clean checkout does not contain those raw inputs or
reproduce the historical quality results by itself.

The QASPER baseline preserves its original cap failure and the prospectively
registered eight-question completion. Do not silently rerun completed questions.
The release cards retain the failed strict cache-equivalence result and the
fixed-subset/AI-judgment limitations. No new GPU runs are scheduled; the separate
27B sarcasm study is deferred to preserve capacity for other projects.
