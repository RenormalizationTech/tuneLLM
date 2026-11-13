"""CPU-only private physics adapter staging. No upload, deletion, or checkpoint mutation.

Requires the fixed rank decision and completed final report; never selects from
test results. All source inputs stay unchanged. Failed preflight produces no
publisher manifest. Existing payload/manifest paths are never overwritten.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import random
import shutil
import subprocess
import zipfile

from scripts.common import ROOT

TARGET = "RenormalizationTech/Qwen3-4B-tuneLLM-PRL-Writing-QLoRA"
BASE = "RenormalizationTech/Qwen3-4B-tuneLLM-NF4"
DEFAULT_OUTPUT = "artifacts/releases/Qwen3-4B-tuneLLM-PRL-Writing-QLoRA"
MANIFEST = "outputs/physics_adapter_release_manifest.json"
TOKENIZER_FILES = {"added_tokens.json", "chat_template.jinja", "merges.txt", "special_tokens_map.json",
                   "tokenizer.json", "tokenizer_config.json", "vocab.json", "tokenizer.model"}
TRAINING_MODULES = ("scripts/physics_train.py", "scripts/physics_data.py", "scripts/physics_eval.py",
                    "scripts/save_nf4.py", "scripts/batched_generation.py", "scripts/training_data.py",
                    "scripts/run_support.py", "scripts/common.py")
SCORING_MODULES = ("scripts/benchmark_screen.py", "scripts/batched_generation.py", "scripts/save_nf4.py", "evaluation/gsm8k.py")
SECRET = re.compile(rb"hf_[A-Za-z0-9]{20,}|AKIA[A-Z0-9]{16}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----")
LOCAL_HOME = re.compile(rb"/(?:home|Users)/[^/\s\"'<>]+/|[A-Za-z]:\\Users\\")
LICENSES = {"paper_text": {"license": "CC-BY-4.0", "url": "https://creativecommons.org/licenses/by/4.0/"},
            "cais/mmlu": {"license": "MIT", "url": "https://huggingface.co/datasets/cais/mmlu"},
            "openai/gsm8k": {"license": "MIT", "url": "https://huggingface.co/datasets/openai/gsm8k"},
            "google/IFEval": {"license": "Apache-2.0", "url": "https://huggingface.co/datasets/google/IFEval"}}


def resolve(root, value):
    filename = Path(value).expanduser()
    return filename.resolve() if filename.is_absolute() else (root / filename).resolve()


def read(filename):
    return json.loads(Path(filename).read_text(encoding="utf-8"))


def sha(filename):
    result = hashlib.sha256()
    with Path(filename).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def bytes_sha(value):
    return hashlib.sha256(value).hexdigest()


def private_preflight(data, root):
    if SECRET.search(data):
        raise ValueError("credential-like content in private payload")
    if str(root).encode() in data or LOCAL_HOME.search(data):
        raise ValueError("local project/home path in private payload; scientific text/code must not be rewritten")


def redact_metadata(value, root):
    if isinstance(value, dict):
        forbidden = {"token", "hf_token", "access_token", "api_key", "password", "authorization", "cookie", "credentials"}
        if any(str(key).lower() in forbidden and content not in (None, False, "") for key, content in value.items()):
            raise ValueError("credential field in export metadata")
        return {redact_metadata(str(key), root): redact_metadata(content, root) for key, content in value.items()}
    if isinstance(value, list):
        return [redact_metadata(content, root) for content in value]
    if isinstance(value, str):
        value = value.replace(str(root), "<PROJECT_ROOT>")
        return re.sub(r"/(?:home|Users)/[^/\s\"'<>]+", "<HOME>", value)
    return value


def json_bytes(value, root):
    data = (json.dumps(redact_metadata(value, root), indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode()
    private_preflight(data, root)
    return data


def verify_base(publication):
    remote = publication.get("remote_load_check", {})
    download = publication.get("download_check", {})
    if publication.get("status") != "published_and_verified" or publication.get("private") is not True or publication.get("repo_id") != BASE:
        raise ValueError("verified private NF4 publication required")
    revision = publication.get("revision", "")
    if not re.fullmatch(r"[0-9a-f]{40}", revision) or remote.get("revision") != revision or download.get("revision") != revision:
        raise ValueError("NF4 publication/download/GPU-check revisions differ")
    if (remote.get("repo_id") != BASE or download.get("repo_id") != BASE
            or any(remote.get(key) is not True for key in ("exact_last_token_logits", "exact_input_and_generated_ids", "requantization_forbidden"))
            or remote.get("max_abs_logit_difference") != 0 or download.get("all_sha256_equal") is not True):
        raise ValueError("pinned-download NF4 GPU/hash verification must pass")


def portable_nf4_report(report_file, publication, snapshot):
    """Keep the upstream verification lineage, but verify only published compute files.

    physics_train.verify_nf4 expects revision to remain the original Qwen SHA.
    The different private NF4-repository SHA belongs in published_nf4_dependency.
    """
    verify_base(publication)
    original = read(report_file)
    if original.get("completed") is not True:
        raise ValueError("completed parent NF4 verification report required")
    computational = {}
    for name, expected in original["files"].items():
        if name not in TOKENIZER_FILES | {"config.json", "generation_config.json", "model.safetensors.index.json"} and not re.fullmatch(r"model-\d+-of-\d+\.safetensors", name):
            continue
        remote = publication["files"].get(name)
        if remote is None or any(remote[key] != expected[key] for key in ("bytes", "sha256")):
            raise ValueError(f"published computational manifest differs from original NF4: {name}")
        filename = Path(snapshot) / name
        if filename.stat().st_size != expected["bytes"] or sha(filename) != expected["sha256"]:
            raise ValueError(f"published computational bytes changed: {name}")
        computational[name] = {"bytes": expected["bytes"], "sha256": expected["sha256"]}
    required = {"config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "model.safetensors.index.json"}
    if len(computational) != 12 or not required <= computational.keys():
        raise ValueError("pinned Qwen3-4B NF4 requires exactly12 published computational files")
    return {**original, "files": computational, "total_bytes": sum(value["bytes"] for value in computational.values()),
            "parent_report_sha256": sha(report_file),
            "published_nf4_dependency": {"repo_id": publication["repo_id"], "revision": publication["revision"]},
            "portability": {"omitted_original_files": sorted(set(original["files"]) - computational.keys()),
                            "reason": "Raw provenance.json is intentionally not published. It is machine-specific metadata, not a computational dependency. Original upstream revision, serialization checks and parent-report hash retain its verification lineage.",
                            "verification": "All12 files checked against original report, published manifest, and streaming SHA256 of pinned downloaded bytes; no quantization or GPU execution."}}


def selected_checkpoint(final, selection, condition, checkpoint, cfg, root, descriptive_only):
    if final.get("status") != "computational_evaluation_complete_human_writing_review_pending" or final.get("writing_quality_improvement_established") is not False:
        raise ValueError("completed final report with human writing quality still unestablished required")
    if selection.get("stage") != "rank" or selection.get("split") != "development":
        raise ValueError("fixed development rank-selection report required")
    measured = final["conditions"][condition]
    if measured["representation"] != "nf4" or not measured.get("checkpoint") or resolve(root, measured["checkpoint"]) != checkpoint:
        raise ValueError("final-report adapter condition does not identify the requested NF4 checkpoint")
    decision = selection["decision"]
    fallback = decision.get("fallback_base") is True
    if fallback and any(row["gate"].get("passed") is True for row in selection["candidates"]):
        raise ValueError("all-fail/base fallback contradicts a passing candidate")
    if fallback and not descriptive_only:
        raise ValueError("all-fail/base fallback requires explicit --descriptive-only export")
    if not fallback and (decision.get("selected_id") is None or resolve(root, decision["selected_checkpoint"]) != checkpoint):
        raise ValueError("requested checkpoint differs from fixed selected adapter")
    candidates = [candidate for candidate in selection["candidates"]
                  if resolve(root, candidate["best_epoch"]["checkpoint"]) == checkpoint]
    if len(candidates) != 1:
        raise ValueError("checkpoint must identify exactly one nominated selection candidate")
    candidate = candidates[0]
    if candidate["rank"] != cfg["rank"] or candidate["learning_rate"] != cfg["learning_rate"]:
        raise ValueError("selection candidate rank/LR differs from training run")
    if not fallback and (candidate["id"] != decision["selected_id"] or candidate["gate"].get("passed") is not True):
        raise ValueError("selected adapter did not pass registered gates")
    if not fallback and decision.get("selected_epoch") != candidate["best_epoch"]["epoch"]:
        raise ValueError("fixed selection epoch differs from nominated checkpoint")
    return candidate, fallback


def verify_sources(root, run, protocol, scoring_run):
    expected = {}
    for name in TRAINING_MODULES:
        wanted = protocol["source_sha256"][name]
        if sha(run / "source" / name) != wanted or sha(root / name) != wanted:
            raise ValueError(f"training source differs from actual frozen module: {name}")
        expected[name] = wanted
    for name in SCORING_MODULES:
        wanted = scoring_run["source_sha256"][name]
        if name in expected and expected[name] != wanted:
            raise ValueError(f"training/scoring source versions conflict: {name}")
        if sha(root / name) != wanted:
            raise ValueError(f"scoring source differs from measured module: {name}")
        expected[name] = wanted
    return expected


def source_archive(root, expected):
    tracked = subprocess.run(["git", "ls-files", "-z"], cwd=root, capture_output=True, check=True).stdout.decode().split("\0")
    allowed_roots = {"scripts", "evaluation", "tests", "configs", "docs"}
    allowed_files = {"README.md", "plan.md", "LICENSE", ".gitignore", "pytest.ini", "pyproject.toml"}
    included, records = {}, {}
    for name in sorted(filter(None, tracked)):
        relative = PurePosixPath(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("tracked source path escapes project")
        if relative.parts[0] not in allowed_roots and name not in allowed_files and not re.fullmatch(r"requirements[^/]*\.txt", name):
            continue
        if relative.suffix not in {".py", ".json", ".md", ".txt", ".toml", ".ini"} and name not in {"LICENSE", ".gitignore"}:
            continue
        filename = root / name
        if filename.is_symlink() or not filename.is_file():
            raise ValueError("source archive requires regular tracked files")
        original = filename.read_bytes()
        if relative.suffix == ".py" or name == "LICENSE":
            exported = original  # Executable source is never rewritten.
        elif relative.suffix == ".json":
            exported = json_bytes(json.loads(original), root)
        else:
            exported = redact_metadata(original.decode("utf-8"), root).encode()
        private_preflight(exported, root)
        included[name] = exported
        records[name] = {"original_sha256": bytes_sha(original), "exported_sha256": bytes_sha(exported),
                         "redacted_metadata_or_documentation": original != exported}
    for name, wanted in expected.items():
        if name not in included or bytes_sha(included[name]) != wanted:
            raise ValueError(f"archive lacks exact tracked training/scoring source: {name}")
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        for name, data in {**included, "SOURCE_MANIFEST.json": json_bytes(records, root)}.items():
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, data)
    return output.getvalue(), records


def corpus_attributions(run, protocol, root):
    provenance = read(run / "inputs/corpus_provenance.json")
    manifest = {paper["id"]: paper for paper in provenance["papers"]}
    if len(provenance["papers"]) != 100 or len(manifest) != 100:
        raise ValueError("all100 paper attributions required")
    result, split_files, seen_papers = [], {}, set()
    for split, count in (("train", 70), ("validation", 10), ("test", 20)):
        filename = run / "inputs" / f"{split}.jsonl"
        raw = filename.read_bytes()
        private_preflight(raw, root)  # Never redact the exact prepared scientific data.
        if bytes_sha(raw) != protocol["raw_corpus_sha256"][split]:
            raise ValueError("prepared corpus bytes differ from frozen training protocol")
        rows = [json.loads(line) for line in raw.decode().split("\n") if line.strip()]
        if len(rows) != count:
            raise ValueError("expected70/10/20 paper counts")
        for row in rows:
            paper = manifest[row["id"]]
            proof = paper["authoritative_license"]
            license_ok = lambda value: re.fullmatch(r"https?://creativecommons\.org/licenses/by/4\.0/?", value or "")
            if (paper["split"] != split or paper["paper_id"] != row["paper_id"] or row["paper_id"] in seen_papers
                    or not license_ok(paper["license"]) or not license_ok(proof["license"]) or proof.get("verified") is not True
                    or paper["clean_text_sha256"] != bytes_sha(row["text"].encode())):
                raise ValueError("paper split/license/clean-text attribution verification failed")
            if not all(paper.get(key) for key in ("authors", "title", "doi", "arxiv_version", "source_url", "source_sha256")):
                raise ValueError("incomplete paper attribution")
            seen_papers.add(row["paper_id"])
            result.append({key: paper[key] for key in ("id", "paper_id", "arxiv_version", "authors", "title", "source_url", "doi", "split",
                                                        "license", "authoritative_license", "source_sha256", "clean_text_sha256")})
        split_files[f"reproduction_data/{split}.jsonl"] = filename
    return result, split_files, provenance


def blind_review_packet(final, adapter_condition, fixture, root):
    references = [(name, condition) for name, condition in final["conditions"].items()
                  if name != adapter_condition and condition["representation"] == "nf4" and not condition.get("checkpoint")]
    if not references:
        raise ValueError("blinded review requires a measured NF4 base reference")
    name, reference = min(references, key=lambda pair: (not bool(re.search(r"few.?shot|two.?example", pair[0] + " " + pair[1].get("prompt_condition", ""))), pair[0]))
    outputs = []
    for condition in (reference, final["conditions"][adapter_condition]):
        filename = resolve(root, condition["evaluation_run"]) / "writing_predictions.json"
        if sha(filename) != condition["raw_result_sha256"]["writing_predictions.json"]:
            raise ValueError("blinded-review writing outputs changed since final report")
        rows = read(filename)
        indexed = {row["id"]: row["output"] for row in rows}
        if len(indexed) != len(rows):
            raise ValueError("duplicate writing output IDs")
        outputs.append(indexed)
    rows = list(fixture["tasks"])
    if {row["id"] for row in rows} != set(outputs[0]) or set(outputs[0]) != set(outputs[1]):
        raise ValueError("blinded review fixture/base/adapter writing IDs differ")
    rng = random.SystemRandom()
    rng.shuffle(rows)
    pairs, key = [], []
    for row in rows:
        swap = rng.choice((False, True))
        sample = bytes_sha(row["id"].encode())[:16]
        a, b = (outputs[1][row["id"]], outputs[0][row["id"]]) if swap else (outputs[0][row["id"]], outputs[1][row["id"]])
        pairs.append({"sample_id": sample, "task": row["task"], "evidence": row["source"], "A": a, "B": b,
                      "scientific_fidelity_rating": None, "style_preference": None})
        key.append({"sample_id": sample, "fixture_id": row["id"], "A": adapter_condition if swap else name,
                    "B": name if swap else adapter_condition})
    return {"ratings": "pending; never inferred", "instructions": "Review scientific fidelity first, then clarity, concision, structure, accessibility and supported significance. Do not open the separate answer key during blinded review.", "pairs": pairs}, {"mapping": key}


def card_text(cfg, budget, final, selection, fallback, base_revision, quality=None, lr_selection=None):
    status = ("Descriptive-only experimental adapter: it failed development selection; the base remains recommended."
              if fallback else "Experimental adapter selected by the registered development heuristic.")
    base_math = final["conditions"].get("nf4_base", {}).get("summary", {}).get("gsm8k")
    adapter_math = final["conditions"].get("nf4_adapter", {}).get("summary", {}).get("gsm8k")
    math_disclosure = ""
    if base_math and adapter_math and adapter_math["fallback_correct"] < base_math["fallback_correct"]:
        math_disclosure = (f"**Measured math regression:** numeric GSM8K accuracy fell from "
                           f"{base_math['fallback_correct']}/{base_math['total']} for the saved NF4 base to "
                           f"{adapter_math['fallback_correct']}/{adapter_math['total']} for this adapter. "
                           "The final results did not trigger reselection.")
        if "strict_correct" in base_math and "strict_correct" in adapter_math:
            math_disclosure += f" Strict-format correct counts were {base_math['strict_correct']} and {adapter_math['strict_correct']}, respectively."
    lines = ["---", f"base_model: {BASE}", "base_model_relation: adapter", "library_name: peft", "license: apache-2.0",
             "pipeline_tag: text-generation",
             "tags: [qwen3, lora, qlora, physics, experimental]", "---", "", "# Qwen3-4B tuneLLM PRL Writing QLoRA", "",
             f"**{status} Human writing-quality improvement is not established.**", "",
             math_disclosure, "",
             "This private artifact records continued causal language-model training on cleaned paper LaTeX. It does not establish improved PRL acceptance, novelty, originality, human preference, or scientific correctness.", "",
             f"Requires `{BASE}` at revision `{base_revision}`. The exported PEFT config pins that NF4 repository commit, not the upstream Qwen commit.",
             f"Lineage: `{cfg['model_id']}` at `{cfg['revision']}`. Packed NF4 weights are restored from stored quantization; FP32 nonquantized modules match PEFT preparation. Adapters are FP32.", "",
             f"All-linear LoRA rank {cfg['rank']}, alpha {2 * cfg['rank']}, dropout 0.05; LR {cfg['learning_rate']}; seed {cfg['seed']}; sequence length {cfg['max_sequence_length']}.",
             f"Exported epoch {budget['checkpoint_epoch']}, step {budget['checkpoint_step']}: {budget['checkpoint_loss_tokens']} loss tokens. Full run: {budget['full_run_epochs']} epochs, {budget['full_run_steps']} steps, {budget['full_run_loss_tokens']} loss tokens. These budgets are distinct.", "",
             "## Objective final measurements", "", "| Condition | Task | Correct or writing diagnostic |", "| --- | --- | --- |"]
    for name, condition in final["conditions"].items():
        for task, values in condition["summary"].items():
            if task == "writing":
                score = f"{values['nonempty']}/{values['total']} nonempty; {values['automatic_fidelity_flags']} automatic fidelity flags"
            else:
                key = "fallback_correct" if task == "gsm8k" else "prompt_strict_correct" if task == "ifeval" else "correct"
                score = f"{values[key]}/{values['total']}"
            lines.append(f"| {name} | {task} | {score} |")
    paper_rows = [(name, condition["paper_text"]) for name, condition in final["conditions"].items()
                  if condition.get("paper_text")]
    paper_rows += list(final.get("paper_only_conditions", {}).items())
    if paper_rows:
        lines += ["", "| Condition | Held-out paper NLL | Perplexity |", "| --- | ---: | ---: |"]
        for name, values in paper_rows:
            lines.append(f"| {name} | {values['token_weighted_nll']:.6f} | {values['perplexity']:.6f} |")
    lines += ["", "Scores include all reported conditions and any math regressions or BF16-control differences; see `evaluation.json` and `selection.json` for paired results, gate failures, paper NLL, and limits. A BF16 control uses BF16 frozen weights; NF4 uses prepared FP32 nonquantized modules. This compares workflow representations.",
              "MMLU uses custom local zero-shot likelihood subsets, GSM8K uses numeric fallback, and IFEval uses its pinned official strict scorer. These are not full standard leaderboard scores. Lower paper NLL measures fit, not writing quality or benchmark capability.", "",
              "## Data, attribution and reproduction", "",
              "The exact private prepared 70/10/20 paper splits are in `reproduction_data/`. All 100 paper attributions, versioned source URLs, PRL DOIs and verified CC BY4.0 licenses are in `paper_attributions.json`. Cleaning removed comments/hidden blocks, author/funding/back matter and preserved scientific equations, citations and used macros; consult corpus provenance and the source archive for exact processing. Source versions may precede final PRL wording. Cleaned text is not guaranteed to be compilable standalone LaTeX.",
              "Paper text is CC BY4.0; model Apache2.0 does not override that license. MMLU and GSM8K dataset cards specify MIT; IFEval specifies Apache2.0. These three benchmark datasets are evaluation-only, not physics training data. `data_licenses.json` and `benchmark_dataset_licenses.json` record the separate terms and pinned verification. Git history, credentials, raw source archives and base-model weights are not in this adapter payload.",
              "`reproduction.zip` contains tracked code/config/docs without history; executable source is unchanged and verified against measured modules. Metadata/documentation path redaction is recorded. Start from the pinned NF4 dependency, exact prepared data and frozen fixtures; do not re-query live paper selection. See `REPRODUCE.md` and `environment.lock.txt`.", "",
              "Human ratings are pending. `human_review/packet.json` is the blinded review packet. Its random answer key is retained locally outside this payload; provenance records its SHA256 commitment. Automated flags cannot establish human preference or scientific fidelity. Do not treat absence of flags or insignificant paired tests as equivalence.", "",
              "## Development gate outcomes", ""]
    for stage_name, candidates in (("Learning-rate pilot", (lr_selection or {}).get("candidates", [])),
                                    ("Adapter ranks", selection["candidates"])):
        lines += [f"### {stage_name}", ""]
        for candidate in candidates:
            failed = [name for name, check in candidate["gate"].get("checks", {}).items() if check.get("passed") is False]
            math_drop = candidate["gate"].get("checks", {}).get("gsm8k_correct_drop", {}).get("value")
            math_note = f" GSM8K net correct-count drop vs base: {math_drop}." if math_drop is not None else ""
            lines.append(f"- {candidate['id']}: {'passed' if candidate['gate']['passed'] else 'failed'}; failed checks: {', '.join(failed) or 'none recorded'}.{math_note}")
        lines.append("")
    lines += ["", "## Reported limitations", ""] + ["- " + value for value in final.get("limits", [])] + [""]
    if quality:
        lines += [f"- Frozen corpus QA records {quality['unicode_replacement_characters']} replacement characters across {len(quality['papers_with_replacement_characters'])} papers. The proper-name review and unchanged split hashes are in `corpus_quality.json`; prepared scientific data remain unchanged.", ""]
    return "\n".join(lines)


def stage(args, root=ROOT):
    root = Path(root).resolve()
    checkpoint, run = resolve(root, args.checkpoint), resolve(root, args.training_run)
    selection_file, final_file = resolve(root, args.selection_report), resolve(root, args.final_report)
    out, manifest_file = resolve(root, args.output), root / MANIFEST
    key_file = root / "outputs/physics_release_review_key.json"
    if out.exists() or manifest_file.exists() or key_file.exists():
        raise FileExistsError("refusing existing release folder or publisher manifest")
    if out.is_relative_to(run) or out.is_relative_to(checkpoint):
        raise ValueError("release folder must be outside source training/checkpoint directories")
    cfg, training, protocol = read(run / "config.json"), read(run / "run.json"), read(run / "protocol.json")
    selection, final = read(selection_file), read(final_file)
    lr_file = root / "results/physics_lr_selection.json"
    lr_selection = read(lr_file)
    if final.get("lr_selection_report_sha256") != sha(lr_file):
        raise ValueError("final report must pin the learning-rate selection report")
    if lr_selection.get("stage") != "lr_pilot" or lr_selection.get("split") != "development":
        raise ValueError("registered development learning-rate selection required")
    if final["selection_report_sha256"] != sha(selection_file) or resolve(root, final["selection_report"]) != selection_file:
        raise ValueError("final report does not pin the exact fixed selection report")
    if training.get("completed") is not True or training.get("status") != "complete" or cfg["representation"] != "nf4":
        raise ValueError("completed NF4 training run required")
    if any(training.get("replay", {}).get(key) is not True for key in ("exact_adapter_tensors", "exact_logits", "exact_generated_ids")):
        raise ValueError("training run's final-epoch serialization checks must pass")
    if sha(run / "protocol.json") != training["protocol_sha256"] or sha(run / "frozen.json") != training["frozen_manifest_sha256"]:
        raise ValueError("training protocol/frozen manifest changed")
    for name, expected in read(run / "frozen.json")["sha256"].items():
        if sha(run / name) != expected:
            raise ValueError(f"frozen training source/input changed: {name}")
    condition = final["conditions"][args.adapter_condition]
    evaluation_dir = resolve(root, condition["evaluation_run"])
    if sha(evaluation_dir / "run.json") != condition["run_sha256"]:
        raise ValueError("final-report scoring run changed")
    evaluation_run = read(evaluation_dir / "run.json")
    if evaluation_run.get("completed") is not True or resolve(root, evaluation_run["adapter"]) != checkpoint:
        raise ValueError("measured adapter is not the staged checkpoint")
    candidate, fallback = selected_checkpoint(final, selection, args.adapter_condition, checkpoint, cfg, root, args.descriptive_only)
    match = re.fullmatch(r"checkpoint-(\d+)", checkpoint.name)
    if checkpoint.parent != run / "checkpoints" or not match:
        raise ValueError("explicit epoch checkpoint must belong to the supplied training run")
    step = int(match.group(1))
    epoch_rows = [row for row in read(run / "learning_curve.json") if row.get("step") == step and "eval_loss" in row]
    if len(epoch_rows) != 1 or step % cfg["steps_per_epoch"]:
        raise ValueError("checkpoint does not map to a completed registered epoch")
    epoch = step // cfg["steps_per_epoch"]
    if epoch_rows[0]["epoch"] != epoch or candidate["best_epoch"]["epoch"] != epoch or candidate["best_epoch"]["step"] != step:
        raise ValueError("selection and checkpoint epoch/step disagree")
    weights, original_config = checkpoint / "adapter_model.safetensors", checkpoint / "adapter_config.json"
    weights_hash, config_hash = sha(weights), sha(original_config)
    if candidate["checkpoint_adapter_sha256"] != weights_hash:
        raise ValueError("nominated checkpoint weights changed since selection")
    adapter_cfg = read(original_config)
    if adapter_cfg["r"] != cfg["rank"] or adapter_cfg["lora_alpha"] != 2 * cfg["rank"] or adapter_cfg["lora_dropout"] != 0.05:
        raise ValueError("adapter configuration disagrees with registered LoRA")
    publication_file = root / "results/nf4_huggingface_publication.json"
    publication = read(publication_file)
    verify_base(publication)
    base_snapshot = resolve(root, publication["download_check"]["snapshot"])
    trained_base = read(run / "inputs/nf4_report.json")
    required_base = set(trained_base["files"]) - {"provenance.json"}
    if not required_base <= publication["files"].keys():
        raise ValueError("published NF4 manifest lacks training dependency files")
    for name, expected in trained_base["files"].items():
        if name in publication["files"] and expected["sha256"] != publication["files"][name]["sha256"]:
            raise ValueError("published NF4 dependency differs from training representation")
    portable_report = portable_nf4_report(run / "inputs/nf4_report.json", publication, base_snapshot)
    copies = {"adapter_model.safetensors": weights}
    for name in sorted(TOKENIZER_FILES & publication["files"].keys() | {"LICENSE"}):
        filename, expected = base_snapshot / name, publication["files"][name]
        if filename.stat().st_size != expected["bytes"] or sha(filename) != expected["sha256"]:
            raise ValueError(f"verified NF4 tokenizer/license bytes changed: {name}")
        private_preflight(filename.read_bytes(), root)
        copies[name] = filename
    if "tokenizer.json" not in copies or "tokenizer_config.json" not in copies:
        raise ValueError("verified tokenizer files missing")
    attributions, paper_copies, corpus = corpus_attributions(run, protocol, root)
    copies.update(paper_copies)
    expected_sources = verify_sources(root, run, protocol, evaluation_run)
    archive, archive_records = source_archive(root, expected_sources)
    files = {"reproduction.zip": archive}
    files["reproduction_data/nf4_checkpoint_report.json"] = json_bytes(portable_report, root)
    metadata_sources = {"evaluation.json": final_file, "selection.json": selection_file,
                        "training_protocol.json": run / "protocol.json", "training_run.json": run / "run.json",
                        "corpus_provenance.json": run / "inputs/corpus_provenance.json",
                        "base_publication.json": publication_file,
                        "lr_selection.json": lr_file,
                        "nltk_resource.json": root / "results/physics_nltk_resource.json",
                        "global_protocol.json": root / "results/physics_prl_protocol.json",
                        "protocol_addendum_01.json": root / "results/physics_prl_protocol_addendum_01.json",
                        "benchmark_dataset_licenses.json": root / "outputs/benchmark_dataset_licenses.json"}
    quality_file = root / "results/physics_prl_corpus_quality.json"
    stability_file = root / "results/physics_ifeval_language_stability.json"
    if stability_file.exists():
        if final.get("ifeval_language_stability") != read(stability_file):
            raise ValueError("supplementary scorer diagnostic changed since final report")
        metadata_sources["ifeval_language_stability.json"] = stability_file
    quality = read(quality_file) if quality_file.exists() else None
    if quality:
        if quality.get("split_sha256") != protocol["raw_corpus_sha256"] or quality.get("current_data_not_mutated") is not True:
            raise ValueError("corpus QA does not describe the unchanged frozen splits")
        metadata_sources["corpus_quality.json"] = quality_file
    addendum = read(metadata_sources["protocol_addendum_01.json"])
    if addendum["original_global_protocol_sha256"] != sha(metadata_sources["global_protocol.json"]):
        raise ValueError("contingency addendum does not pin the unchanged original global protocol")
    license_proof = read(metadata_sources["benchmark_dataset_licenses.json"])
    if {row["repository"] for row in license_proof} != {"cais/mmlu", "openai/gsm8k", "google/IFEval"}:
        raise ValueError("pinned benchmark dataset license proofs required")
    for row in license_proof:
        licenses = row["reported_license"]
        licenses = [licenses] if isinstance(licenses, str) else licenses
        if {value.lower() for value in licenses} != {LICENSES[row["repository"]]["license"].lower()} or not re.fullmatch(r"[0-9a-f]{40}", row["revision"]):
            raise ValueError("benchmark dataset license/revision proof disagrees with export notices")
    for name, filename in metadata_sources.items():
        files[name] = json_bytes(read(filename), root)
    for name, expected in read(metadata_sources["nltk_resource.json"])["files"].items():
        if sha(root / "artifacts/tools/nltk" / name) != expected:
            raise ValueError("pinned IFEval tokenizer data changed")
    if final.get("nltk_resource_report_sha256") != sha(metadata_sources["nltk_resource.json"]):
        raise ValueError("final report must pin the scorer tokenizer resource report")
    benchmark = root / "data/benchmark-screen-01"
    benchmark_provenance = read(benchmark / "provenance.json")
    writing_provenance = read(benchmark / "writing_provenance.json")
    registered = read(metadata_sources["global_protocol.json"])
    if (benchmark_provenance["fixture_sha256"] != registered["benchmark_provenance"]["fixture_sha256"]
            or benchmark_provenance["ifeval_source_sha256"] != registered["benchmark_provenance"]["ifeval_source_sha256"]
            or sha(benchmark / "writing_provenance.json") != registered["writing_provenance_sha256"]):
        raise ValueError("benchmark/writing/scorer provenance differs from frozen global protocol")
    for split in ("development", "test"):
        filename = benchmark / f"{split}.json"
        if sha(filename) != benchmark_provenance["fixture_sha256"][split]:
            raise ValueError("frozen benchmark fixture changed")
        private_preflight(filename.read_bytes(), root)
        copies[f"reproduction_data/benchmarks/{filename.name}"] = filename
    if condition["fixture_sha256"] != benchmark_provenance["fixture_sha256"]["test"]:
        raise ValueError("final measured benchmark fixture differs from packaged test fixture")
    for name, expected in writing_provenance["files"].items():
        filename = benchmark / name
        if Path(name).name != name or sha(filename) != expected["sha256"]:
            raise ValueError("frozen writing fixture changed or unsafe name")
        private_preflight(filename.read_bytes(), root)
        copies[f"reproduction_data/benchmarks/{name}"] = filename
    if condition.get("writing_sha256") not in {entry["sha256"] for entry in writing_provenance["files"].values()}:
        raise ValueError("final measured writing fixture not included")
    writing_name = next(name for name, entry in writing_provenance["files"].items() if entry["sha256"] == condition["writing_sha256"])
    writing_fixture = read(benchmark / writing_name)
    if writing_fixture.get("source_split") != "test":
        raise ValueError("final writing fixture must be the frozen test fixture")
    packet, answer_key = blind_review_packet(final, args.adapter_condition, writing_fixture, root)
    # Scientific evidence and model outputs stay unchanged in the blinded packet.
    files["human_review/packet.json"] = (json.dumps(packet, indent=2, ensure_ascii=False) + "\n").encode()
    answer_key_bytes = json_bytes(answer_key, root)
    for name, expected in benchmark_provenance["ifeval_source_sha256"].items():
        filename = root / "artifacts/tools/ifeval" / name
        if Path(name).name != name or sha(filename) != expected:
            raise ValueError("pinned official IFEval source changed")
        private_preflight(filename.read_bytes(), root)
        copies[f"third_party/ifeval/{name}"] = filename
    files["reproduction_data/benchmarks/provenance.json"] = json_bytes(benchmark_provenance, root)
    files["reproduction_data/benchmarks/writing_provenance.json"] = json_bytes(writing_provenance, root)
    files["paper_attributions.json"] = json_bytes({"papers": attributions, "license": "CC-BY-4.0",
                                                 "changes": "See corpus_provenance.json and frozen prepare_physics_corpus.py for cleaning and selection."}, root)
    files["data_licenses.json"] = json_bytes(LICENSES, root)
    exported_config = {**adapter_cfg, "base_model_name_or_path": BASE, "revision": publication["revision"]}
    # Only these two PEFT dependency fields may change; reject other private paths.
    files["adapter_config.json"] = (json.dumps(exported_config, indent=2, ensure_ascii=False) + "\n").encode()
    freeze = training["execution_provenance"]["pip_freeze"]
    if freeze.get("returncode") != 0 or not freeze.get("stdout", "").strip():
        raise ValueError("actual training environment lock required")
    files["environment.lock.txt"] = (redact_metadata(freeze["stdout"], root).rstrip() + "\n").encode()
    budget = {"rank": cfg["rank"], "alpha": 2 * cfg["rank"], "dropout": 0.05, "learning_rate": cfg["learning_rate"],
              "checkpoint_epoch": epoch, "checkpoint_step": step,
              "checkpoint_loss_tokens": epoch * protocol["counts"]["train"]["loss_tokens_per_epoch"],
              "checkpoint_microbatches": epoch * ((protocol["counts"]["train"]["chunks"] + cfg["per_device_train_batch_size"] - 1) // cfg["per_device_train_batch_size"]),
              "full_run_epochs": cfg["epochs"], "full_run_steps": training["global_step"],
              "full_run_loss_tokens": training["exposure"]["actual_loss_tokens"]}
    export_metadata = {"private": True, "repo_id": TARGET, "descriptive_only": args.descriptive_only or fallback,
                       "selection_failed_base_recommended": fallback, "adapter_condition": args.adapter_condition,
                       "checkpoint_weight_sha256_unchanged": weights_hash,
                       "original_adapter_config_sha256": config_hash, "exported_adapter_config_sha256": bytes_sha(files["adapter_config.json"]),
                       "nf4_dependency": {"repo_id": BASE, "revision": publication["revision"]},
                       "upstream_lineage": {"model_id": cfg["model_id"], "revision": cfg["revision"]},
                       "budget": budget, "source_commit": training["execution_provenance"]["git_head"]["stdout"],
                       "source_checks": expected_sources, "source_archive_files": archive_records,
                       "metadata_redaction": "Project/home paths replaced in exported metadata/docs only; executable source and exact prepared data untouched",
                       "metadata_original_sha256": {name: sha(filename) for name, filename in metadata_sources.items()},
                       "metadata_exported_sha256": {name: bytes_sha(files[name]) for name in metadata_sources},
                       "selected_checkpoint_serialization_scope": "Training replay report covers the final epoch of the full run; do not reinterpret it as a probe of another nominated epoch.",
                       "human_writing_quality_improvement_established": False,
                       "human_review_key_sha256": bytes_sha(answer_key_bytes),
                       "human_review_key_status": "Random key retained locally, excluded from release pending ratings"}
    files["provenance.json"] = json_bytes(export_metadata, root)
    files["README.md"] = card_text(cfg, budget, final, selection, fallback, publication["revision"], quality, lr_selection).encode()
    guide = f"""# Reproduce the measured private artifact

Install environment.lock.txt in a compatible Python 3.10/CUDA environment. The
recorded torch CUDA wheel needs the official https://download.pytorch.org/whl/cu124
extra package index. Extract
reproduction.zip into a clean workspace (no Git history is included). Copy exact
reproduction_data/train.jsonl, validation.jsonl and test.jsonl to
data/physics-prl-01/; copy corpus_provenance.json to that directory as
provenance.json. Copy reproduction_data/benchmarks/ to data/benchmark-screen-01/
and third_party/ifeval/ to artifacts/tools/ifeval/. Before offline IFEval scoring,
download NLTK punkt_tab to artifacts/tools/nltk/ and verify the English resource
files against nltk_resource.json. Its source URL and archive hash are recorded;
tokenizer data are a separate dependency. These frozen scientific bytes must not be edited.
Consult corpus_quality.json if present; proper-name decoding artifacts were kept
in this frozen pilot, and the cleaned text is not guaranteed compilable LaTeX.

Download {BASE} at {publication['revision']} using your own HF credentials outside
this payload. The reproduction_data/nf4_checkpoint_report.json portable report
contains ONLY the12 computational model/tokenizer files present unchanged in
that publication. Raw provenance.json is intentionally absent remotely; do not
use the original full local report for verify_nf4. The portable report retains
the upstream Qwen revision and original serialization checks, records the parent
report hash and published NF4 pin, and recalculates total_bytes for these12 files.

Copy that portable report to results/nf4_checkpoint.json. In configs/physics.json,
set nf4_checkpoint to the pinned downloaded snapshot, or download/copy exactly its
12 listed files to the default artifacts/nf4/Qwen3-4B/. Leave nf4_report pointing
at results/nf4_checkpoint.json. Hash verification then succeeds without needing
omitted machine-specific provenance. Do not quantize packed NF4 weights again. Use
AutoModelForCausalLM.from_pretrained with that revision, use_safetensors=True,
torch_dtype=torch.float32, device_map={{"":0}}, attn_implementation="sdpa"; do not
pass a new quantization_config. Apply prepare_model_for_kbit_training identically,
then PeftModel.from_pretrained(base, adapter_folder, is_trainable=False).

Before going offline, run the archived BF16 acquisition command as well:

    python -m scripts.download_model --config configs/baseline.json

This caches the original pinned Qwen BF16 checkpoint/tokenizer for the BF16
control and default benchmark_screen tokenizer path; the private NF4 snapshot is
a separate dependency. Do not substitute the Qwen SHA for the NF4 repository SHA.

Run `git init` in the fresh extracted workspace, review and commit source and
configuration, and prepare new local protocols with the archived physics_train.py.
Copy each generated protocol.json to results/protocols/, review and commit that
exact file BEFORE invoking physics_train --run ... --protocol ... . Retain hash
verification and the recorded schedule, including its remainder chunks. The
portable report is consumed by the actual unmodified verify_nf4 implementation.
Do not re-query live paper discovery/source endpoints. Metadata <PROJECT_ROOT>
and <HOME> placeholders deliberately require local configuration, not text edits
to the prepared scientific data. Use the registered rank/LR/schedule in
training_protocol.json; selected checkpoint exposure is distinct from full run.

For inference/evaluation use the archived benchmark_screen.py and recorded
fixtures: non-thinking greedy seed42, generation batch8, MC batch16, unchanged
task-specific output caps/scorers. Source hashes and original/exported document
hashes are in provenance.json. Final results are evaluation.json; do not tune
selection from those test outcomes. Human ratings are pending.
"""
    files["REPRODUCE.md"] = guide.encode()
    files["human_review/README.md"] = b"# Blinded review pending\n\npacket.json contains all frozen final-writing pairs with randomized A/B labels and source evidence. The cryptographically random mapping key is retained locally outside this payload; provenance.json commits its SHA256. No human ratings have been supplied; review scientific fidelity first, then style. Automated flags are not human ratings.\n"
    # Everything except model weights is cheap enough to privacy-check before mkdir.
    for name, data in files.items():
        # ZIP executable content was checked before compression.
        if name != "reproduction.zip":
            private_preflight(data, root)
    for name, filename in copies.items():
        if name != "adapter_model.safetensors":
            private_preflight(filename.read_bytes(), root)
    out.mkdir(parents=True, exist_ok=False)
    key_file.parent.mkdir(parents=True, exist_ok=True)
    with key_file.open("xb") as stream:
        stream.write(answer_key_bytes)
    key_file.chmod(0o600)
    for name, data in files.items():
        target = out / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    for name, filename in copies.items():
        target = out / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(filename, target)
        if sha(target) != sha(filename):
            raise RuntimeError("copy changed payload bytes")
    if sha(weights) != weights_hash or sha(original_config) != config_hash or sha(out / "adapter_model.safetensors") != weights_hash:
        raise RuntimeError("source/exported checkpoint changed during staging")
    inventory = {str(p.relative_to(out)): {"bytes": p.stat().st_size, "sha256": sha(p)}
                 for p in sorted(out.rglob("*")) if p.is_file()}
    manifest = {"repo_id": TARGET, "private": True, "files": inventory,
                "status": "staged_private_preflight_passed_not_uploaded", "adapter_condition": args.adapter_condition,
                "descriptive_only": args.descriptive_only or fallback, "nf4_revision": publication["revision"],
                "checkpoint_weight_sha256": weights_hash, "original_config_sha256": config_hash,
                "exported_config_sha256": inventory["adapter_config.json"]["sha256"]}
    manifest_file.parent.mkdir(parents=True, exist_ok=True)
    with manifest_file.open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--training-run", required=True)
    parser.add_argument("--selection-report", required=True)
    parser.add_argument("--final-report", required=True)
    parser.add_argument("--adapter-condition", default="nf4_adapter")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--descriptive-only", action="store_true")
    args = parser.parse_args()
    stage(args)
    print(args.output, MANIFEST)


if __name__ == "__main__":
    main()
