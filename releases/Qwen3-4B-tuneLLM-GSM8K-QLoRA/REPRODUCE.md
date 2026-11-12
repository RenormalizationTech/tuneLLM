# Reproduce the learning experiment

Unpack `reproduction.zip` into a new, empty working directory and run commands
there. It contains source and configuration snapshots without Git author history.
Published metadata replaces machine paths with placeholders. New preparations
create their own local paths and hashes; compare selected IDs and dataset hashes
against the supplied `protocol.json`.

## Environment and model

Use Python 3.10 and a CUDA/BF16-capable GPU. The recorded machine used an RTX A6000,
CUDA 12.4 and PyTorch 2.6.0+cu124. Install PyTorch before the full environment lock:

```bash
python3.10 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
.venv/bin/python -m pip install -r environment.lock.txt
.venv/bin/python -m scripts.download_model
```

The lock is included beside this guide; copy it into the source directory if
needed. Alternatively install the pinned requirements files from the archive.

## Data acquisition and preparation

Public downloads use exact revisions. Raw data stays in ignored directories:

```bash
.venv/bin/python - <<'PY'
import json
import os
from pathlib import Path
os.environ["HF_HOME"] = str(Path(".hf").resolve())
from huggingface_hub import snapshot_download
Path("outputs").mkdir(exist_ok=True)
sources = [
    ("openai/gsm8k", "740312add88f781978c0658806c59bc2815b9866", "main",
     ["main/*.parquet", "README.md"], "gsm8k_snapshot.json"),
    ("cais/mmlu", "c30699e8356da336a370243923dbaf21066bb9fe", "all",
     ["all/test-*.parquet", "README.md"], "mmlu_snapshot.json"),
]
for dataset_id, revision, config, patterns, filename in sources:
    snapshot = snapshot_download(dataset_id, repo_type="dataset", revision=revision,
                                 cache_dir=".hf/hub", allow_patterns=patterns)
    manifest = {"dataset_id": dataset_id, "revision": revision, "config": config,
                "snapshot": str(Path(snapshot).resolve()), "license": ["mit"]}
    if config == "all":
        manifest["split"] = "test"
    Path("outputs", filename).write_text(json.dumps(manifest, indent=2) + "\n")
PY
.venv/bin/python -m scripts.prepare_gsm8k
.venv/bin/python -m scripts.prepare_retention
.venv/bin/python -m pytest -q tests
```

## Freeze, commit, train and evaluate

The default config selects physical GPU 1. Use `--gpu 0` at preparation if that
is the only available GPU. This changes hardware selection, not the data protocol.
Preparation allocates no model and generates no final-test predictions.

```bash
.venv/bin/python -m scripts.train_qlora_run --output outputs/replication --gpu 0
cp outputs/replication/protocol.json results/replication_protocol.json
git init
git add .gitignore README.md LICENSE plan.md configs scripts tests evaluation docs results requirements*.txt
git commit -m "Freeze replication protocol"
.venv/bin/python -u -m scripts.train_qlora_run \
  --execute outputs/replication --protocol results/replication_protocol.json
.venv/bin/python -m scripts.summarize_qlora_run \
  --run outputs/replication --output results/replication_summary.json
```

Use your normal local Git identity. Model binaries, data and credentials remain
ignored. Execution verifies the committed protocol and uses the frozen snapshot.
Steps 32/64 are selected by validation loss alone, with an earlier-step tie rule.
Final test generations occur only after selection. Check the selected IDs against
the supplied protocol before running. Hardware/library differences can change
numerical outputs; the documented result is one seeded experiment.

The optional, single secondary diagnostic is:

```bash
.venv/bin/python -m scripts.retention_sensitivity \
  --run outputs/replication --output outputs/replication-cap64
```

It preserves the primary eight-token retention results. See the model card for
the observed math regression, output-budget sensitivity and scoped reload checks.
