# Reproduce the GGUF conversion

Unpack `reproduction-dab1dae.zip` into a new working directory. It contains the
reviewed workflow source and `docs/execution.md`. The recorded machine used
Python 3.10, CUDA 12.4 and one RTX A6000 for the GPU measurements.

```bash
python3.10 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
.venv/bin/python -m pip install -r environment.lock.txt
.venv/bin/python -m scripts.download_model
git clone https://github.com/ggml-org/llama.cpp.git artifacts/tools/llama.cpp
git -C artifacts/tools/llama.cpp checkout 8ca5a8008b57aa39f2601546c0d07912045080a5
mkdir -p artifacts/gguf
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
```

The environment lock is provided beside this guide; copy it into the extracted
source directory. No importance matrix is used. Validate sizes and hashes against
`provenance.json`. Refer to `docs/execution.md` for the CUDA build flags, raw-prompt
comparison and benchmark settings. Choose a host compiler supported by your CUDA
installation; the recorded build explicitly used GCC/G++ 11.

Model/dataset revisions, file hashes and the measured limitations are in the
model card, `evaluation.json` and `provenance.json`. CPU/GPU architecture or tool
changes can alter timings and numerical outputs.
