"""After private publication: verify downloaded bytes and eight title probes on GPU1.

Run only after the coordinator releases GPU1. This command downloads into its
own ignored cache; it never uploads or rewrites the reference predictions.
"""
import argparse
import os
import re
import time
from pathlib import PurePosixPath

from scripts.common import path, read_json, select_gpu, write_json
from scripts.save_nf4 import reload_stored_nf4, stream_hash

BASE = "RenormalizationTech/Qwen3-4B-tuneLLM-NF4"
BASE_REVISION = "e9af208e44e083b77702a71d8aa65b7d97f78ce6"
TOKENIZER_FILES = {"added_tokens.json", "chat_template.jinja", "merges.txt", "special_tokens_map.json",
                   "tokenizer.json", "tokenizer_config.json", "vocab.json"}
BASE_FILES = TOKENIZER_FILES | {"config.json", "generation_config.json", "model.safetensors.index.json",
                                "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"}
SOURCE_FILES = ("scripts/common.py", "scripts/save_nf4.py", "scripts/batched_generation.py",
                "scripts/benchmark_screen.py")


def check_pin(config):
    if config.get("base_model_name_or_path") != BASE or config.get("revision") != BASE_REVISION:
        raise ValueError("adapter must pin the verified published NF4 base and revision")


def verify_files(directory, files):
    verified = {}
    for name, expected in files.items():
        relative = PurePosixPath(name)
        if relative.is_absolute() or ".." in relative.parts or "\\" in name:
            raise ValueError("unsafe manifest file path")
        filename = directory / name
        if (not filename.is_file() or filename.stat().st_size != expected["bytes"]
                or stream_hash(filename) != expected["sha256"]):
            raise ValueError("downloaded bytes differ from reviewed manifest")
        verified[name] = {"bytes": expected["bytes"], "sha256": expected["sha256"]}
    return verified


def aligned_probes(tasks, predictions):
    """Require unique reference IDs and the exact original first eight-title batch."""
    for rows in (tasks, predictions):
        ids = [r["id"] for r in rows]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate fixture or prediction IDs")
    probes = [r for r in tasks if r["task"] == "title"][:8]
    if len(probes) != 8:
        raise ValueError("eight title probes required")
    ids = [r["id"] for r in probes]
    references = [r for r in predictions if r["id"] in set(ids)]
    if [r["id"] for r in references] != ids:
        raise ValueError("reference probe order or membership differs")
    for position, row in enumerate(references):
        if (row.get("batch_index") != 0 or row.get("batch_position") != position
                or row.get("batch_ids") != ids or not row.get("input_ids")
                or not row.get("generated_ids")):
            raise ValueError("reference probes must be the original complete first title batch")
    return probes, references


def compare_probes(references, predictions):
    if [r["id"] for r in references] != [r["id"] for r in predictions]:
        raise ValueError("generated probe alignment differs")
    if len(predictions) != 8 or len({r["id"] for r in predictions}) != 8:
        raise ValueError("eight unique generated probes required")
    inputs = sum(a["input_ids"] == b["input_ids"] for a, b in zip(references, predictions))
    generated = sum(a["generated_ids"] == b["generated_ids"] for a, b in zip(references, predictions))
    return {"probe_count": 8, "exact_input_ids_count": inputs, "exact_generated_ids_count": generated,
            "all_input_ids_equal": inputs == 8, "all_generated_ids_equal": generated == 8,
            "logits_compared": False, "max_abs_logit_difference": None}


def publication_files(publication, manifest, manifest_file):
    if (publication.get("status") not in {"published_remote_hashes_verified", "published_and_verified"}
            or publication.get("private") is not True or manifest.get("private") is not True
            or publication.get("repo_id") != manifest.get("repo_id")
            or publication.get("manifest_sha256") != stream_hash(manifest_file)
            or not re.fullmatch(r"[0-9a-f]{40}", publication.get("revision", ""))):
        raise ValueError("private fixed-revision publication must match the staged manifest")
    names = {"adapter_config.json", "adapter_model.safetensors"}
    names |= TOKENIZER_FILES & manifest["files"].keys()
    if not names <= manifest["files"].keys():
        raise ValueError("adapter safetensors and config required")
    files = {name: manifest["files"][name] for name in sorted(names)}
    if any(any(publication.get("files", {}).get(n, {}).get(k) != f[k] for k in ("bytes", "sha256"))
           for n, f in files.items()):
        raise ValueError("published file records differ from the staged manifest")
    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--publication", default="results/physics_huggingface_publication.json")
    parser.add_argument("--manifest", default="outputs/physics_adapter_release_manifest.json")
    parser.add_argument("--reference-screen", default="outputs/physics-final-screen-nf4_adapter")
    parser.add_argument("--writing-fixture", default="data/benchmark-screen-01/writing_test.json")
    parser.add_argument("--base-publication", default="results/nf4_huggingface_publication.json")
    parser.add_argument("--output", default="outputs/physics_published_load_check.json")
    args = parser.parse_args()
    output = path(args.output)
    if output.exists():
        raise FileExistsError("refusing to overwrite publication verification")
    publication, manifest = read_json(args.publication), read_json(args.manifest)
    files = publication_files(publication, manifest, path(args.manifest))
    base = read_json(args.base_publication)
    download = base.get("download_check", {})
    if (base.get("status") != "published_and_verified" or base.get("repo_id") != BASE
            or base.get("revision") != BASE_REVISION or download.get("repo_id") != BASE
            or download.get("revision") != BASE_REVISION or download.get("all_sha256_equal") is not True
            or any(base.get("remote_load_check", {}).get(k) is not True for k in
                   ("requantization_forbidden", "exact_last_token_logits", "exact_input_and_generated_ids"))):
        raise ValueError("verified published NF4 base required")
    snapshot = path(download["snapshot"])
    base_verified = verify_files(snapshot, {n: base["files"][n] for n in sorted(BASE_FILES)})
    reference_dir = path(args.reference_screen)
    reference = read_json(reference_dir / "run.json")
    if (reference.get("completed") is not True or reference.get("representation") != "nf4"
            or not reference.get("adapter") or reference.get("generation_batch") != 8
            or reference.get("seed") != 42 or reference.get("thinking") is not False
            or reference.get("writing_sha256") != stream_hash(path(args.writing_fixture))):
        raise ValueError("completed matching NF4 adapter reference screen required")
    verify_files(path(reference["adapter"]), {"adapter_model.safetensors": files["adapter_model.safetensors"]})
    sources = {n: stream_hash(path(n)) for n in SOURCE_FILES}
    if any(reference["source_sha256"].get(n) != h for n, h in sources.items()):
        raise ValueError("reference inference source changed")
    probes, expected = aligned_probes(read_json(args.writing_fixture)["tasks"],
                                      read_json(reference_dir / "writing_predictions.json"))
    report = {"completed": False, "scope": "Eight title probes only; not the whole benchmark or a pre-save epoch-one check.",
              "gpu": "1", "seed": 42, "batch_size": 8, "max_new_tokens": 48,
              "enable_thinking": False, "attention": "sdpa", "tf32": False,
              "requantization_forbidden": True, "nonquantized_base_dtype": "float32",
              "adapter_dtype": "float32", "source_sha256": sources,
              "verifier_sha256": stream_hash(path(__file__)),
              "input_artifact_sha256": {"publication": stream_hash(path(args.publication)),
                  "manifest": stream_hash(path(args.manifest)), "base_publication": stream_hash(path(args.base_publication)),
                  "reference_metadata": stream_hash(reference_dir / "run.json"),
                  "reference_predictions": stream_hash(reference_dir / "writing_predictions.json"),
                  "fixture": stream_hash(path(args.writing_fixture))},
              "base_files": base_verified, "logits_compared": False,
              "privacy": "No texts, token IDs, probe IDs or condition identity mapping retained."}
    started = time.perf_counter()
    try:
        # Use normal ignored project credentials, but a separate download cache.
        os.environ["HF_HOME"] = str(path(".hf"))
        os.environ["HF_HUB_OFFLINE"] = "0"
        os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
        from huggingface_hub import snapshot_download
        adapter = path(snapshot_download(publication["repo_id"], revision=publication["revision"],
                       allow_patterns=list(files), cache_dir=str(path(".hf/physics-publication-check/hub"))))
        report["adapter_files"] = verify_files(adapter, files)
        check_pin(read_json(adapter / "adapter_config.json"))
        # Full-tokenizer copies are verified against the unchanged base tokenizer.
        for name in TOKENIZER_FILES & files.keys():
            if files[name]["sha256"] != base_verified[name]["sha256"]:
                raise ValueError("published tokenizer differs from verified base tokenizer")
        select_gpu("1")
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        import torch
        from transformers import AutoTokenizer, set_seed
        from unittest.mock import patch
        from scripts.batched_generation import generate_batched
        from scripts.benchmark_screen import attach_adapter
        set_seed(42)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported() or torch.cuda.device_count() != 1:
            raise RuntimeError("one selected CUDA BF16 GPU required")
        tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False)
        with patch("bitsandbytes.functional.quantize_4bit", side_effect=RuntimeError("unexpected requantization")):
            model = attach_adapter(reload_stored_nf4(snapshot), adapter)
            if any(p.dtype != torch.float32 for n, p in model.named_parameters() if "lora_" in n):
                raise ValueError("adapter tensors must be FP32")
            generated = generate_batched(model, tokenizer, probes, batch_size=8, max_new_tokens=48,
                                         seed=42, phase="published title probe")
        checks = compare_probes(expected, generated["predictions"])
        report.update(checks)
        report["completed"] = checks["all_input_ids_equal"] and checks["all_generated_ids_equal"]
        if not report["completed"]:
            raise RuntimeError("published title probe parity failed")
    except BaseException as exc:
        report.update({"completed": False, "error_type": type(exc).__name__})
        raise
    finally:
        report["elapsed_wall_seconds"] = time.perf_counter() - started
        write_json(output, report)
    print("Eight published title probes verified; logits were not compared.", flush=True)


if __name__ == "__main__":
    main()
