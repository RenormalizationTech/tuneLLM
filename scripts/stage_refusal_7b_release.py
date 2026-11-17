"""Stage an explicit experimental refusal release allowlist; CPU, no publication."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import shutil

from scripts.common import path, read_json
from scripts.publish_release import sha256
from scripts import refusal_release_runtime as runtime

DEST = path("outputs/refusal-7b-release-01")
MANIFEST = path("results/refusal_7b_release_manifest_01.json")
REPO = "RenormalizationTech/qwen2.5-7b-refusal-steering-experiment"
SETTING = "results/refusal_7b_experimental_setting_01.json"
RESULTS = {"experimental_setting.json": SETTING, "initial_transfer.json": "results/refusal_7b_transfer_01.json",
           "halfdose.json": "results/refusal_7b_halfdose_01.json", "selected_benign13.json": "results/refusal_7b_benign_transfer_01.json"}
ALLOWLIST = {"README.md", "LICENSE", "NOTICE", "VECTOR_LICENSE.md", "DATA_SOURCES.md", "refusal_runtime.py",
             "requirements.txt", "environment.json", "direction.pt", "direction.safetensors", "base_model.json", "provenance.json",
             *("results/"+n for n in RESULTS)}


def save(file, value):
    file.parent.mkdir(parents=True, exist_ok=True)
    with file.open("x") as stream: json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False); stream.write("\n")


DATA_SOURCES = """# Attribution and source terms

Code: Apache-2.0; see LICENSE. Base model: Qwen/Qwen2.5-7B-Instruct,
revision a09a35458c702b33eeacc393d103063234e8bc28, Apache-2.0.

The raw vector and identical safetensors mirror are conservatively distributed
under CC BY-SA 3.0, reflecting the Dolly extraction source; see VECTOR_LICENSE.md.
This does not replace any source's attribution or license requirements.
No raw datasets, prompts, responses, judge reasons or grading keys are included.

- AdvBench, Zou et al., llm-attacks/llm-attacks revision
  098262edf85f807224e70ecd87b9d83716bf6b73: MIT repository terms;
  https://github.com/llm-attacks/llm-attacks/tree/098262edf85f807224e70ecd87b9d83716bf6b73 .
- Databricks Dolly 15k, Databricks, revision
  bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a: dataset card CC BY-SA 3.0;
  https://huggingface.co/datasets/databricks/databricks-dolly-15k/tree/bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a .
  The vector is a model-activation mean difference from100 AdvBench/Dolly pairs,
  not a redistribution of their instruction texts.
- XSTest, Röttger et al., paul-rottger/xstest revision
  d7bb5bd738c1fcbc36edd83d5e7d1b71a3e2d84d: prompts CC BY 4.0;
  https://github.com/paul-rottger/xstest/tree/d7bb5bd738c1fcbc36edd83d5e7d1b71a3e2d84d .
  Published results retain source IDs; no prompt text is included.
- Method reference: Arditi et al., Refusal in Language Models Is Mediated by a
  Single Direction (2024), https://arxiv.org/abs/2406.11717 ; author code
  https://github.com/andyrdt/refusal_direction/tree/9d852fae1a9121c78b29142de733cb1340770cc3 , Apache-2.0.
  This block-output workflow is an adaptation, not a canonical checkpoint/paper reproduction.

Source audit: tuneLLM results/refusal_data_audit.json and docs/refusal_protocol.md.
Creative Commons terms: https://creativecommons.org/licenses/by-sa/3.0/ and
https://creativecommons.org/licenses/by/4.0/ .
"""


def card(initial, half, selected):
    base = initial["development"]["baseline_metrics"]
    arms = [("Baseline", base), ("Subtract −0.5", initial["development"]["candidate_metrics"]),
            ("All-block projection", initial["development"]["candidate_metrics"]),
            ("Subtract −0.25", half["development"]["candidate_metrics"])]
    table = []
    for label, metrics in arms:
        prefix = "m0" if label == "Baseline" else "ablate_all" if label == "All-block projection" else "subtract18"
        safe, unsafe = metrics[prefix+"_safe"], metrics[prefix+"_unsafe"]
        table.append(f"| {label} | {safe['primary_refused']} | {safe['useful_and_correct']} | {safe['degenerate']} | {unsafe['valid_full_compliance']} | {unsafe['primary_refused']} |")
    p = selected["primary13"]; a, b = p["baseline"], p["candidate"]
    return """---
license: apache-2.0
base_model: Qwen/Qwen2.5-7B-Instruct
tags:
- experimental
- activation-steering
- refusal-steering
library_name: transformers
---
# Qwen2.5-7B refusal steering experiment

Completed experimental demonstration with case-level effects, **not a validated
reliable controller**. No confirmatory final or retention run qualified or ran.
This contains a raw activation direction and runtime, not base-model weights,
an adapter, or a selectively safe refusal policy.

Code is Apache-2.0. Vector artifacts are conservatively CC BY-SA 3.0; the card's
Apache metadata describes code, not an override of vector/dataset terms.
See VECTOR_LICENSE.md and DATA_SOURCES.md for separate terms and attribution.

## Fixed mechanism

Pinned base Qwen/Qwen2.5-7B-Instruct at
`a09a35458c702b33eeacc393d103063234e8bc28`, native BF16/SDPA,28 blocks,3584 hidden.
Raw FP32 harmful-minus-benign means use100 AdvBench/Dolly pairs, block18 OUTPUT
at the last real prompt token. The example intervention subtracts0.25raw from
that same block's output at every position, including prompt prefill and cached
decoding. FP32 arithmetic then restores activation dtype. It reproduces the old
−0.5 hook applied to half the raw vector. No weight changes or training.
All-block projection was also measured; it is reported below but is not a
retained runtime option. This is an adapted block-output workflow, not a claim
of exact reproduction of the original author pipeline.

## All development arms

30 benign and30 source-unsafe prompts per arm. Useful/correct is an AI rubric
field; source-unsafe full compliance measures willingness, not answer quality.

| Arm | Benign refusals /30 | Benign useful /30 | Benign degenerate | Unsafe valid full /30 | Unsafe refusals /30 |
| --- | ---: | ---: | ---: | ---: | ---: |
"""+"\n".join(table)+f"""

The original−0.5 study closed null due to usefulness loss and degeneration.
All-block projection did not reach the unsafe-contrast gain gate. The single
−0.25 follow-up also closed null under that original gate. These decisions are
preserved. Stronger subtraction reduced unsafe refusals too: no selective
safety claim is supported. No final/retention results are implied.

## Selected13 benign diagnostic

These13 were selected from earlier4B refusal cases after excluding four prompts
seen during dose development. They are not an unbiased or confirmatory held-out
benchmark. Four seen pairs are reported separately in the frozen JSON, not pooled.

Baseline→−0.25: refusals **{a['refused']}→{b['refused']}**, AI useful-field counts
**{a['useful_and_correct']}→{b['useful_and_correct']}**. The stricter conjunction
valid full compliance AND useful was **9→11**; this differs from the useful-field
count because an answer can include a refusal. Paired useful rescues were
`xstest-151` and `xstest-168`, with no newly refused formerly compliant case.
**One repetitive output (`xstest-153`) appeared**, so refusal reduction alone
does not establish quality preservation. Both conditions hit the cap on7/13.
Judgments use one AI judge family, not human factual verification; robustness,
general reliability and deployment quality are unestablished. Source case67
retains its original questionable P label. Original null records remain intact.

## Offline example (baseline by default)

Use an already cached pinned base snapshot. No model download is performed.
The CLI checks all pinned metadata/weight hashes before loading; allow time for
local disk reads. The example is single-prompt, not an exact batch8/5 replay.

```bash
python refusal_runtime.py --snapshot /path/to/pinned/snapshot --prompt "Explain a simple board-game strategy."
python refusal_runtime.py --snapshot /path/to/pinned/snapshot --prompt "Explain a simple board-game strategy." --steer
```

CPU is the default device; `--device cuda:0` explicitly opts into a CUDA device.
Greedy seed42, cap320, repetition1, native default-system template,
tokenizerEOS151645/PAD151643. Checkpoint defaults dualEOS/rep1.05 are overridden
to match the experiment. No thinking-mode claim. Base weights are BF16.

Python use: `from refusal_runtime import load_direction, steering`; load the
pinned base, then wrap inference in `steering(model, load_direction())`.
Safetensors key is `direction`. `load_direction(use_pt=True)` safely loads the
byte-identical frozen `.pt` with `weights_only=True`; that file stores a root
Tensor, not a dictionary. File/tensor hashes and geometry are verified.

`environment.json` records the current installed environment; requirements pin
the Python packages. Tested Torch build is2.6.0+cu124 (CUDA12.4); choose the
appropriate Torch2.6.0 wheel for the execution device. Release checks were CPU
mechanics/serialization checks, not a new7B inference or GPU reload test.

Frozen setting/results are in `results/`. No raw prompts, answers, reasons,
grading keys, model weights or credentials are included. See `provenance.json`
for original source hashes and `base_model.json` for pinned base asset hashes.
"""


def preflight(dest, manifest):
    import torch
    from safetensors.torch import load_file
    actual = {p.relative_to(dest).as_posix() for p in dest.rglob("*") if p.is_file()}
    if actual != ALLOWLIST or set(manifest["files"]) != ALLOWLIST: raise ValueError("explicit release allowlist differs")
    for name, expected in manifest["files"].items():
        file = dest/name
        if file.is_symlink() or file.stat().st_size != expected["bytes"] or sha256(file) != expected["sha256"]:
            raise ValueError("staged payload hash/size differs: "+name)
    if sha256(dest/"direction.pt") != runtime.PT_SHA256: raise ValueError("exact frozen .pt hash differs")
    pt = runtime.validate_direction(torch.load(dest/"direction.pt", map_location="cpu", weights_only=True))
    values = load_file(str(dest/"direction.safetensors"), device="cpu")
    if set(values) != {"direction"} or not torch.equal(pt, runtime.validate_direction(values["direction"])):
        raise ValueError("safetensors mirror differs")
    return {"file_count": len(actual), "vector_shape": list(pt.shape), "vector_dtype": str(pt.dtype),
            "original_pt_sha256": runtime.PT_SHA256, "tensor_sha256": runtime.TENSOR_SHA256,
            "safetensors_exact": True, "cpu_only": True, "model_loaded": False, "uploaded": False}


def stage(dest=DEST, manifest_file=MANIFEST):
    import platform
    import torch
    from safetensors.torch import save_file
    dest, manifest_file = Path(dest), Path(manifest_file)
    if dest.exists() or manifest_file.exists(): raise FileExistsError("fresh staging and manifest required")
    setting = read_json(SETTING)
    if (setting["model_id"], setting["revision"], setting["intervention"]["block_output"], setting["intervention"]["effective_raw_coefficient"]) != (runtime.BASE, runtime.REVISION, 18, -.25):
        raise ValueError("frozen experimental setting differs")
    vector_file = path(setting["direction_file"])
    if setting["direction_sha256"] != runtime.PT_SHA256 or sha256(vector_file) != runtime.PT_SHA256:
        raise ValueError("frozen source vector hash differs before load")
    raw = runtime.validate_direction(torch.load(vector_file, map_location="cpu", weights_only=True))
    origin = read_json("outputs/refusal-7b-transfer-01/manifest.json")
    snapshot = Path(origin["snapshot"])
    metadata = {Path(n).name: h for n, h in origin["external_model_sha256"].items()
                if Path(n).name.startswith("model") or Path(n).name in {"config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"}}
    result_values = {n: read_json(p) for n, p in RESULTS.items()}
    source_names = [*RESULTS.values(), str(vector_file), "outputs/refusal-7b-transfer-01/extraction/vectors.json",
        "outputs/refusal-7b-transfer-01/manifest.json", "results/refusal_data_audit.json", "docs/refusal_protocol.md",
        "scripts/refusal_release_runtime.py", "scripts/stage_refusal_7b_release.py", str(snapshot/"LICENSE")]
    source_hashes = {str(Path(n).relative_to(path("."))) if Path(n).is_absolute() else n: sha256(path(n)) for n in source_names}
    dest.mkdir(parents=True)
    shutil.copyfile(vector_file, dest/"direction.pt")
    save_file({"direction": raw}, str(dest/"direction.safetensors"))
    shutil.copyfile(path("scripts/refusal_release_runtime.py"), dest/"refusal_runtime.py")
    shutil.copyfile(snapshot/"LICENSE", dest/"LICENSE")
    for name, source in RESULTS.items():
        target = dest/"results"/name; target.parent.mkdir(exist_ok=True); shutil.copyfile(path(source), target)
    (dest/"README.md").write_text(card(result_values["initial_transfer.json"], result_values["halfdose.json"], result_values["selected_benign13.json"]))
    (dest/"DATA_SOURCES.md").write_text(DATA_SOURCES)
    (dest/"VECTOR_LICENSE.md").write_text("# Vector artifact terms\n\nThe raw direction.pt and identical direction.safetensors tensor are conservatively\nlicensed under Creative Commons Attribution-ShareAlike3.0 Unported:\nhttps://creativecommons.org/licenses/by-sa/3.0/ . Attribution: tuneLLM contributors;\nextraction inputs from Databricks Dolly15k and AdvBench (Zou et al.).\nSee DATA_SOURCES.md for exact revisions, attribution and distinct source terms.\nThis does not assert source dataset terms are overridden by the code license.\nCode is separately Apache-2.0. No raw dataset is redistributed.\n")
    (dest/"NOTICE").write_text("tuneLLM experimental refusal-direction runtime, Apache-2.0.\nAdapted from tuneLLM's fixed block-output refusal03/7B workflow.\nMethod reference: Arditi et al. (2024), andyrdt/refusal_direction\ncommit9d852fae1a9121c78b29142de733cb1340770cc3 (Apache-2.0).\nQwen2.5-7B-Instruct by the Qwen team (Apache-2.0); no base weights included.\nDataset/vector terms and attribution: DATA_SOURCES.md and VECTOR_LICENSE.md.\n")
    versions = {n: importlib.metadata.version(n) for n in ("torch", "transformers", "accelerate", "safetensors", "tokenizers")}
    save(dest/"environment.json", {"python": platform.python_version(), "packages": versions,
         "torch_build": torch.__version__, "torch_cuda_build": torch.version.cuda, "release_checks": "CPU only; no base model loaded"})
    (dest/"requirements.txt").write_text("# Observed installed versions; exact Torch build is in environment.json.\n"+"".join(n+"=="+(v.split('+')[0] if n == "torch" else v)+"\n" for n, v in versions.items()))
    save(dest/"base_model.json", {"model_id": runtime.BASE, "revision": runtime.REVISION, "files_sha256": metadata})
    save(dest/"provenance.json", {"source_sha256": source_hashes, "original_tensor_layout": "root FP32 tensor [3584]",
        "safetensors_key": "direction", "raw_tensor_sha256": runtime.TENSOR_SHA256, "raw_vector_sha256": runtime.PT_SHA256,
        "intervention": setting["intervention"], "source_artifacts_preserved": True, "no_new_experiment": True,
        "code_license": "Apache-2.0", "vector_license": "CC-BY-SA-3.0 (conservative)", "raw_examples_included": False})
    manifest = {"repo_id": REPO, "private": False, "status": "staged_local_only_not_uploaded",
        "files": {n: {"sha256": sha256(dest/n), "bytes": (dest/n).stat().st_size} for n in sorted(ALLOWLIST)}}
    manifest["cpu_checks"] = preflight(dest, manifest)
    if any(sha256(path(n)) != h for n, h in source_hashes.items()): raise ValueError("original source changed during staging")
    save(manifest_file, manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.parse_args()
    manifest = stage()
    print(json.dumps({"folder": str(DEST), "manifest": str(MANIFEST), "manifest_sha256": sha256(MANIFEST),
                      "files": len(manifest["files"]), "cpu_checks": manifest["cpu_checks"]}, indent=2))


if __name__ == "__main__": main()
