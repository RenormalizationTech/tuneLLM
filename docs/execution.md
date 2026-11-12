# Initial execution record

Started 2026-11-12. The first cycle uses one small model for both quantization
and QLoRA. Results below describe observed status, not planned success.

## Environment

- Two NVIDIA RTX A6000 GPUs, 49,140 MiB each, detected by CUDA-enabled PyTorch.
- Driver 580.178.04; system CUDA toolkit 12.4.131.
- Initial free project storage: about 3 TB; available system RAM: about 104 GiB.
- Python 3.10 virtual environment at `.venv/`.
- PyTorch 2.6.0+cu124; Transformers 4.57.6; PEFT 0.18.1;
  bitsandbytes 0.49.2; Accelerate 1.12.0; datasets 4.4.2;
  huggingface_hub 0.36.2.
- `pip check` passed and model/tokenizer/Trainer/PEFT/bitsandbytes imports passed.
- First runs use physical GPU 1. Another project has a process on GPU 0;
  its processes were left running.

Install PyTorch from its CUDA wheel index before the other requirements:

```bash
/usr/bin/python3.10 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip install -r requirements-qlora.txt -r requirements-gguf.txt -r requirements-dev.txt
```

GPU jobs need access to the NVIDIA devices. In the Codex sandbox, PyTorch
reported zero GPUs; the same check with permitted device access detected both.

## Model and tools

- Model: `Qwen/Qwen3-4B`.
- Revision: `1cfa9a7208912126459214e8b04321603b3df60c`.
- Upstream parameter count: 4,022,468,096; source weights BF16.
- Model files use `.hf/hub/`; generated files use `outputs/` and `artifacts/`.
- llama.cpp source: `artifacts/tools/llama.cpp`, commit
  `8ca5a8008b57aa39f2601546c0d07912045080a5`.

Configure llama.cpp with an explicit CUDA host compiler. The default invocation
failed during compiler detection with mixed C++ standard-library headers;
the following invocation configured successfully without changing system files:

```bash
cmake -S artifacts/tools/llama.cpp -B artifacts/tools/llama.cpp/build-gcc11 \
  -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 \
  -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-11 \
  -DCMAKE_C_COMPILER=/usr/bin/gcc-11 -DCMAKE_CXX_COMPILER=/usr/bin/g++-11 \
  -DLLAMA_BUILD_TESTS=OFF -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build artifacts/tools/llama.cpp/build-gcc11 --config Release \
  --target llama-cli llama-quantize llama-bench llama-perplexity llama-server -j 8
```

## Evaluation decisions

Use explicit non-thinking prompts. Initial greedy decoding is for a fixed
comparison; the upstream Qwen3 card recommends sampling and warns about greedy
repetition. Record truncation and inspect outputs before interpreting scores.
Small handcrafted tasks validate the pipeline; they do not establish general
model capability.

Quantization comparisons should use the high-precision GGUF baseline in the same
runtime. QLoRA comparisons should use the same NF4 base with and without the
adapter. Record runtime, template, seed, generation settings, and revisions.

Claude recommended GSM8K (`openai/gsm8k`, `main`, MIT) for the first training
exercise: reserve validation from training, keep test untouched, and score final
numeric answers. Its familiarity to the base model limits claims of new reasoning
ability. Dataset acquisition, revision and split checks precede training.

## Publication

The Hugging Face organization is `RenormalizationTech`, confirmed by the user. No Hugging Face artifact repository has been created and no artifact
has been uploaded. PEFT generated a local adapter README during saving; a complete
release card will be written for any selected release. The first release candidates are validated GGUF files and a
QLoRA adapter. Model cards will include provenance and measured limitations.

## Baseline result

Two BF16 runs completed at `outputs/hf-baseline-01/` and
`outputs/hf-baseline-02/`, with identical generated token sequences. Each scored
7/8 on the eight-item smoke fixture. The force question was answered `18`
instead of `6`; retain this error in comparisons. No response hit the token cap.
Loaded PyTorch allocation was about 7.49 GiB. End-to-end generation throughput
was 21.68 and 23.03 emitted tokens/s on these very short responses, including
EOS. These are pipeline checks, not representative sustained decode benchmarks.

## Initial quantization result

All three files reloaded through the CPU llama.cpp CLI and answered `42` to
`What is 17 + 25? Reply with only the integer.` The chat was explicitly
non-thinking, greedy, seed 42, context 512, output cap 32, eight CPU threads.
These are one-prompt reload checks; quality and GPU benchmarks remain separate.

| Format | Size (GiB) | CPU smoke score | GPU generation tokens/s | Small-text perplexity |
| --- | ---: | ---: | ---: | ---: |
| BF16 GGUF | 7.498 | 7/8 | 79.1 | 2.6299 |
| Q8_0 | 3.986 | 7/8 | 130.5 | 2.6282 |
| Q4_K_M | 2.326 | 8/8 | 186.0 | 2.6608 |

All eight CPU comparisons passed exact input-token parity against the saved HF
prompts. Q4 answered the force question correctly; a one-item difference does not
establish that Q4 is more capable. All evaluator child servers exited cleanly.

GPU measurements ran sequentially on physical GPU 1 after the QLoRA process
exited. llama-bench used three repetitions, 128 prompt tokens, 64 generation
tokens, four CPU threads, full GPU offload, and flash attention enabled. The
table reports its separate generation test; compare these figures only within
this microbenchmark, not against the HF short-response wall-time figures.

Perplexity used the same prepared text from the first 32 GSM8K source-test
question/solution pairs, context 512, four chunks, batch/microbatch 512 and flash
attention enabled. Tool-reported uncertainties were approximately 0.175, 0.175,
and 0.177 respectively; these are not presented as 95% confidence intervals.
The small sample does not establish general capability preservation.

Exact byte counts and SHA256 checksums are in
`results/initial_quantization.json`. Quantization used no importance matrix.
Q8_0 took 18.24 seconds and Q4_K_M took 72.35 seconds with eight CPU threads
while compilation was also running; these times are operational observations.

Commands (the CPU quantizer was built while the CUDA tools compiled):

```bash
.venv/bin/python artifacts/tools/llama.cpp/convert_hf_to_gguf.py \
  .hf/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c \
  --outfile artifacts/gguf/Qwen3-4B-BF16.gguf --outtype bf16
cmake -S artifacts/tools/llama.cpp -B artifacts/tools/llama.cpp/build-cpu \
  -DGGML_CUDA=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build artifacts/tools/llama.cpp/build-cpu --target llama-quantize -j 8
artifacts/tools/llama.cpp/build-cpu/bin/llama-quantize \
  artifacts/gguf/Qwen3-4B-BF16.gguf artifacts/gguf/Qwen3-4B-Q8_0.gguf Q8_0 8
artifacts/tools/llama.cpp/build-cpu/bin/llama-quantize \
  artifacts/gguf/Qwen3-4B-BF16.gguf artifacts/gguf/Qwen3-4B-Q4_K_M.gguf Q4_K_M 8
.venv/bin/python -m scripts.gguf_eval \
  --model artifacts/gguf/Qwen3-4B-Q4_K_M.gguf \
  --server-bin artifacts/tools/llama.cpp/build-cpu/bin/llama-server \
  --gpu-layers 0 --threads 4 --output outputs/my-q4-cpu-eval
env CUDA_VISIBLE_DEVICES=1 artifacts/tools/llama.cpp/build-gcc11/bin/llama-bench \
  -m artifacts/gguf/Qwen3-4B-Q4_K_M.gguf -ngl 99 -p 128 -n 64 -r 3 -t 4 -fa on -o json
```

## QLoRA preparation

GSM8K revision `740312add88f781978c0658806c59bc2815b9866` is cached locally.
`scripts.prepare_gsm8k` created 7,217 train, 256 validation, and 1,319 test rows,
with seed 42. Exact normalized question overlap and duplicate checks removed
zero rows. Calculator annotations were removed, and the same final-answer format
instruction was added to every split. The full source test split remains held out.
Preparation writes split IDs and checksums to `data/gsm8k/provenance.json`.

The first acquisition used the following explicit download step before offline
preparation:

```bash
.venv/bin/python - <<'PY'
import json
import os
from pathlib import Path
os.environ["HF_HOME"] = str(Path(".hf").resolve())
from huggingface_hub import snapshot_download
revision = "740312add88f781978c0658806c59bc2815b9866"
snapshot = snapshot_download(
    "openai/gsm8k", repo_type="dataset", revision=revision,
    cache_dir=".hf/hub", allow_patterns=["main/*.parquet", "README.md", ".gitattributes"],
)
manifest = {
    "dataset_id": "openai/gsm8k", "revision": revision,
    "snapshot": str(Path(snapshot).resolve()), "config": "main", "license": ["mit"],
}
Path("outputs").mkdir(exist_ok=True)
Path("outputs/gsm8k_snapshot.json").write_text(json.dumps(manifest, indent=2) + "\n")
PY
```

```bash
.venv/bin/python -m scripts.prepare_gsm8k
.venv/bin/python -m scripts.train_qlora --smoke --prepare-only --output outputs/qlora-preflight
.venv/bin/python -m pytest -q tests
.venv/bin/python -u -m scripts.train_qlora --smoke --output outputs/qlora-smoke-01
```

CPU preflight passed. Twelve tests passed, including the actual cached Qwen
tokenizer boundary check, assistant-only labels, EOS/padding, data splitting,
and scoring. The two-step training smoke uses eight training rows and a fixed
16-row subset from each held-out split. It compares the same NF4 base before
training and after adding the trained adapter, then reloads onto a fresh base.
This is a pipeline check and does not establish a useful tuned model.

## QLoRA smoke result

The run completed successfully at `outputs/qlora-smoke-01/`. The fresh-base
reload reproduced every generated token in all 16 post-training test responses.
The two optimizer steps trained 5,898,240 adapter parameters. Training wall time
was 4.57 seconds; peak PyTorch allocation was 4.58 GiB, excluding non-PyTorch
CUDA allocations and other processes. Adapter weights occupy 22.54 MiB.

| Measure on fixed 16-row subsets | Before | After |
| --- | ---: | ---: |
| Validation reference loss | 2.5748 | 2.0165 |
| Test reference loss | 3.2419 | 2.5676 |
| Strict final-answer score | 12/16 | 12/16 |
| Last-number fallback score | 14/16 | 14/16 |
| Responses hitting the token cap | 1/16 | 1/16 |

Loss is averaged by sample/batch on supervised answer tokens, not a
token-weighted corpus perplexity. Lower reference loss with unchanged numeric
scores is consistent with learning the reference style/format; it is not evidence
of improved reasoning. All three phases use the same NF4 base representation,
including FP32 preparation of nonquantized modules. These scores should not be
compared directly to the BF16 baseline or GGUF perplexities.

The adapter is an educational pipeline artifact. General retention and a larger
held-out evaluation remain to be measured before a useful model release.
Compact results, selected IDs, hashes and limitations are in
`results/initial_qlora_smoke.json`. Matching run-time source files were archived
under the ignored run directories. After the run, the worker added metadata
phase checkpoints and progress prints without changing numerical behavior.
The original run source hashes remain recorded rather than being relabelled.

Before a larger evaluation, use a small fixed generation batch to reduce cost,
and check batch-size effects. Left-padding, attention masks, trailing-pad removal
and fixed batch membership must be recorded; training remains right-padded.
