"""Offline tokenizer-only science inputs with exact supervised/inference prefixes.

No models, torch, datasets library, network, or GPU are needed. Evidence budgets
exclude the fixed instructions/question and target. All sampled evaluation rows
survive; impossible length cases fail explicitly or remain marked ineligible.
QASPER final selection uses only seeded paper IDs/counts, never model outcomes.
"""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
import hashlib
import importlib.metadata
import json
import os
import multiprocessing
from pathlib import Path
import random
import re

from scripts import qwen38_science_data as science
from scripts.training_data import encode_row

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ["scripts/qwen38_prepare_inputs.py", "scripts/qwen38_science_data.py",
           "scripts/qwen38_science_score.py", "scripts/training_data.py", "scripts/common.py"]
VERSION = "qwen38-tokenizer-inputs-v2"


def offline_tokenizer(path):
    for key, value in {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "CUDA_VISIBLE_DEVICES": "",
                       "USE_TORCH": "0", "USE_TF": "0", "TOKENIZERS_PARALLELISM": "false"}.items():
        os.environ[key] = value
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)


def _worker_init(path):
    global _WORKER_TOKENIZER
    _WORKER_TOKENIZER = offline_tokenizer(path)


def prepare_paper(paper, task, split, tokenizer, evidence_cap, sequence_cap, generated_cap):
    @lru_cache(maxsize=512)
    def count(text):
        return len(tokenizer(text, add_special_tokens=False)["input_ids"])
    if task == "qasper":
        result = []
        for qa in science.records(paper["qas"]):
            single = {**paper, "qas": [qa]}
            def build(budget, single=single):
                return science.prepare_qasper(single, split, count, budget)[0]
            result.append(fit_item(build, tokenizer, split, evidence_cap, sequence_cap, generated_cap))
        return result
    return [fit_item(lambda budget: science.prepare_prl(paper, split, count, budget),
                     tokenizer, split, evidence_cap, sequence_cap, generated_cap)]


def _worker_job(job):
    paper, task, split, evidence_cap, sequence_cap, generated_cap = job
    return prepare_paper(paper, task, split, _WORKER_TOKENIZER, evidence_cap, sequence_cap, generated_cap)


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class GenerationTokenizer:
    """Pin text-return semantics for encode_row and inference to the same template."""
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __getattr__(self, key):
        return getattr(self.tokenizer, key)

    def __call__(self, *args, **kwargs):
        return self.tokenizer(*args, **kwargs)

    def apply_chat_template(self, *args, **kwargs):
        kwargs["return_dict"] = False
        return self.tokenizer.apply_chat_template(*args, **kwargs)


def messages(row):
    result = []
    if row.get("system"):
        result.append({"role": "system", "content": row["system"]})
    return result + [{"role": "user", "content": row["user"]}]


def measure_row(row, tokenizer, sequence_cap=8192, generated_cap=512):
    """Byte-identical rendered prefix and token-identical direct template path.

    The full reference is measured without truncation, even when ineligible.
    EOS is included in supervised target tokens; generated_cap includes any EOS.
    """
    wrapped = GenerationTokenizer(tokenizer)
    kwargs = dict(add_generation_prompt=True, enable_thinking=False, return_dict=False)
    prompt = wrapped.apply_chat_template(messages(row), tokenize=False, **kwargs)
    if not isinstance(prompt, str):
        raise TypeError("generation template must return text")
    prompt_ids = wrapped(prompt, add_special_tokens=False)["input_ids"]
    inference_ids = wrapped.apply_chat_template(messages(row), tokenize=True, **kwargs)
    if inference_ids != prompt_ids:
        raise ValueError("inference template tokens differ from rendered prompt tokens")
    # Large bound only measures the complete reference; nothing is truncated.
    full = encode_row(row, wrapped, 2 ** 31 - 1)
    first_target = next(i for i, value in enumerate(full["labels"]) if value != -100)
    if first_target != len(prompt_ids) or full["input_ids"][:first_target] != prompt_ids:
        raise ValueError("supervised prefix differs from inference generation prefix")
    fits = len(full["input_ids"]) <= sequence_cap
    if fits:
        bounded = encode_row(row, wrapped, sequence_cap)
        if bounded != full:
            raise ValueError("bounded encode_row changed tokens")
    # The parent trainer currently calls encode_row with the raw tokenizer.
    # Verify that this installed tokenizer also has identical default semantics.
    raw = encode_row(row, tokenizer, 2 ** 31 - 1)
    if raw != full:
        raise ValueError("raw-tokenizer encode_row differs; parent trainer must use GenerationTokenizer")
    return {"prompt": prompt, "prompt_tokens": len(prompt_ids),
            "target_tokens_with_eos": len(full["input_ids"]) - len(prompt_ids),
            "target_text_tokens": len(wrapped(row["assistant"], add_special_tokens=False)["input_ids"]),
            "sequence_tokens": len(full["input_ids"]), "evaluation_reserved_tokens": len(prompt_ids) + generated_cap,
            "training_length_fits": fits, "evaluation_length_fits": len(prompt_ids) + generated_cap <= sequence_cap,
            "prompt_sha256": science.sha(prompt.encode()),
            "prompt_ids_sha256": science.sha(science.canonical(prompt_ids).encode()),
            "encoded_row_sha256": science.sha(science.canonical(full).encode()),
            "raw_encode_row_identical": True, "direct_inference_template_identical": True}


def seeded_paper_prefix(papers, target, seed=42):
    """Select a whole-paper prefix; target is a question count, never a filter."""
    if type(target) is not int or target <= 0:
        raise ValueError("positive question target required")
    by_id = {str(p["id"]): p for p in papers}
    if len(by_id) != len(papers):
        raise ValueError("duplicate source paper IDs")
    order = sorted(by_id)
    random.Random(seed).shuffle(order)
    selected, count = [], 0
    for key in order:
        selected.append(by_id[key])
        count += len(science.records(by_id[key]["qas"]))
        if count >= target:
            break
    return selected, {"target_questions": target, "actual_questions": count,
        "target_met": count >= target, "seed": seed, "paper_ids": [p["id"] for p in selected],
        "policy": "sorted IDs, Random(seed).shuffle, whole-paper prefix, original question order"}


def fit_item(builder, tokenizer, split, evidence_cap=6656, sequence_cap=8192, generated_cap=512):
    """Reduce evidence only; target contents never enter the context selector.

    Training/development may reserve the measured target *length* when longer
    than generated_cap. Test selection never uses reference length. An overlong
    test reference remains a diagnostic; evaluation still reserves generated_cap.
    No target rewriting or final-row filtering is allowed.
    """
    if min(evidence_cap, sequence_cap, generated_cap) < 0 or sequence_cap <= generated_cap:
        raise ValueError("invalid sequence/evidence/generation caps")
    budget, attempts = evidence_cap, []
    while True:
        row = builder(budget)
        measured = measure_row(row, tokenizer, sequence_cap, generated_cap)
        attempts.append({"evidence_budget": budget, "prompt_tokens": measured["prompt_tokens"],
                         "sequence_tokens": measured["sequence_tokens"]})
        evaluation_excess = max(0, measured["evaluation_reserved_tokens"] - sequence_cap)
        training_excess = max(0, measured["sequence_tokens"] - sequence_cap) if split not in {"test", "fresh_test"} else 0
        excess = max(evaluation_excess, training_excess)
        if excess == 0:
            break
        if budget == 0:
            if evaluation_excess:
                raise ValueError(f"{row['id']}: question/instructions alone exceed evaluation cap")
            # Preserve impossible training references, explicitly ineligible.
            break
        budget = max(0, budget - max(64, excess))
    if not measured["training_length_fits"]:
        row["training_eligible"] = False
        row["length_exclusion_reason"] = "complete reference+EOS exceeds sequence cap; retained without truncation"
    row.update(measured)
    row["preparation_lengths"] = {"initial_evidence_budget": evidence_cap, "final_evidence_budget": budget,
        "sequence_cap": sequence_cap, "generated_cap": generated_cap, "attempts": attempts,
        "test_reference_length_used_for_context": False}
    return row


def summarize(rows):
    flags = Counter(flag for row in rows for flag in row.get("leakage_audit", {}).get("flags", []))
    kinds = Counter(ref["answer_type"] for row in rows for ref in row.get("references", []))
    return {"records": len(rows), "papers": len({r["paper_id"] for r in rows}),
        "training_eligible": sum(r["training_eligible"] for r in rows),
        "primary": sum(r.get("primary", False) for r in rows),
        "evidence_strict_primary": sum(r.get("evidence_strict_primary", False) for r in rows),
        "missing_context_evidence": sum(any(ref["missing_context_evidence"] for ref in r.get("references", [])) for r in rows),
        "nontext_evidence": sum(r.get("nontext_evidence", False) for r in rows),
        "unmatched_text_evidence": sum(r.get("unmatched_text_evidence", False) for r in rows),
        "reduced_evidence_budgets": sum(r["preparation_lengths"]["final_evidence_budget"] < r["preparation_lengths"]["initial_evidence_budget"] for r in rows),
        "overlength_references": sum(not r["training_length_fits"] for r in rows),
        "max_prompt_tokens": max((r["prompt_tokens"] for r in rows), default=0),
        "max_sequence_tokens": max((r["sequence_tokens"] for r in rows), default=0),
        "max_target_tokens_with_eos": max((r["target_tokens_with_eos"] for r in rows), default=0),
        "targets_over_256_tokens": sum(r["target_tokens_with_eos"] > 256 for r in rows),
        "targets_over_512_tokens": sum(r["target_tokens_with_eos"] > 512 for r in rows),
        "max_evaluation_reserved_tokens": max((r["evaluation_reserved_tokens"] for r in rows), default=0),
        "prompt_tokens": sum(r["prompt_tokens"] for r in rows),
        "target_tokens_with_eos": sum(r["target_tokens_with_eos"] for r in rows),
        "annotation_types": dict(sorted(kinds.items())), "prl_flags": dict(sorted(flags.items())),
        "prl_excerpted_bodies": sum(not r.get("full_body_in_context", True) for r in rows),
        "prl_syntax_flagged": sum(not r.get("context_selection", {}).get("syntax", {}).get("balanced", True) for r in rows),
        "prl_math_environment_flagged": sum(not r.get("context_selection", {}).get("syntax", {}).get("math_environment_boundaries_balanced", True) for r in rows),
        "prl_generic_brace_flagged": sum(not r.get("context_selection", {}).get("syntax", {}).get("generic_braces_balanced", True) for r in rows),
        "prl_excerpt_without_tail": sum(not r.get("full_body_in_context", True) and not r.get("context_selection", {}).get("tail_indices") for r in rows),
        "prl_empty_copy_baselines": sum(r.get("copy_baseline", {}).get("empty", False) for r in rows),
        "prl_copy_baseline_mean_token_f1": (sum(r["copy_baseline"]["normalized_token_f1"] for r in rows) / len(rows)
                                             if rows and all("copy_baseline" in r for r in rows) else None)}


def eligible_order(rows, target=480, seed=42):
    grouped = defaultdict(list)
    for row in rows:
        if row["training_eligible"]:
            grouped[row["paper_id"]].append(row)
    order = sorted(grouped)
    random.Random(seed).shuffle(order)
    prefix = []
    for key in order:
        prefix.extend(sorted(grouped[key], key=lambda r: r["id"]))
        if len(prefix) >= target:
            break
    if len(prefix) > 1500:
        raise ValueError("whole-paper training prefix exceeds 1500-question ceiling")
    return {"seed": seed, "paper_ids": order, "eligible_questions": sum(map(len, grouped.values())),
        "target_questions": target, "prefix_preview_ids": [r["id"] for r in prefix],
        "prefix_preview_count": len(prefix), "target_met": len(prefix) >= target,
        "prefix_exact_tokens": {"prompt": sum(r.get("prompt_tokens", 0) for r in prefix),
            "target_with_eos": sum(r.get("target_tokens_with_eos", 0) for r in prefix),
            "sequence": sum(r.get("sequence_tokens", 0) for r in prefix),
            "max_sequence": max((r.get("sequence_tokens", 0) for r in prefix), default=0),
            "max_target_with_eos": max((r.get("target_tokens_with_eos", 0) for r in prefix), default=0),
            "item_encoding_bindings_sha256": science.sha(science.canonical([
                {"id": r["id"], "encoded_row_sha256": r.get("encoded_row_sha256")} for r in prefix]).encode())},
        "selection_status": "preview only; trainer freezes actual eligible prefix later"}


def prepare_task(task, tokenizer, tokenizer_path, output, qasper_root=None, prl_root=None,
                 evidence_cap=6656, sequence_cap=8192, generated_cap=512,
                 test_target=400, development_target=150, seed=42, progress=None, workers=1):
    output, tokenizer_path = Path(output), Path(tokenizer_path)
    if output.exists():
        raise FileExistsError(output)
    qasper_root = Path(qasper_root or ROOT / "data/qasper-v03-01")
    prl_root = Path(prl_root or ROOT / "data/qwen38-prl-abstract-01/historical-reextracted-02")
    if type(workers) is not int or workers < 1:
        raise ValueError("positive CPU worker count required")
    if task == "qasper":
        inputs = {s: qasper_root / "original" / f"qasper-{'dev' if s == 'validation' else s}-v0.3.json"
                  for s in ("train", "validation", "test")}
        source_manifest = qasper_root / "manifest.json"
    elif task == "prl":
        inputs = {s: prl_root / f"{s}.jsonl" for s in ("train", "validation", "test")}
        source_manifest = prl_root / "manifest.json"
        if not source_manifest.exists():
            source_manifest = prl_root / "provenance.json"  # synthetic API fixtures
    else:
        raise ValueError("unknown task")
    input_hashes = {s: file_sha(p) for s, p in inputs.items()}
    source_hash = file_sha(source_manifest)
    source_hashes = {name: file_sha(ROOT / name) for name in SOURCES}
    tokenizer_files = [p for p in tokenizer_path.iterdir() if p.is_file() and (p.name.startswith("tokenizer") or p.name == "chat_template.jinja")]
    if not tokenizer_files:
        raise ValueError("local tokenizer provenance files required")
    tokenizer_hashes = {p.name: file_sha(p) for p in tokenizer_files}
    splits = {s: science.read_rows(p) for s, p in inputs.items()}
    science.validate_paper_splits(splits, task)
    if task == "qasper":
        acquired = json.loads(source_manifest.read_text())
        for split, p in inputs.items():
            if input_hashes[split] != acquired["extracted_files"][p.name]["sha256"]:
                raise ValueError("QASPER original split differs from acquisition manifest")
    else:
        lineage = json.loads(source_manifest.read_text())
        if "files" in lineage:
            for split in inputs:
                if input_hashes[split] != lineage["files"][split]["output_sha256"]:
                    raise ValueError("PRL re-extracted split differs from its manifest")
            historical = {s: science.read_rows(ROOT / "data/physics-prl-01" / f"{s}.jsonl") for s in inputs}
            science.validate_paper_splits(splits, task,
                {s: [r["paper_id"] for r in rows] for s, rows in historical.items()})
    selections, prepared = {}, {}
    for split, original in splits.items():
        if task == "qasper" and split != "train":
            papers, selections[split] = seeded_paper_prefix(original, test_target if split == "test" else development_target, seed)
        else:
            papers = original
            selections[split] = {"paper_ids": [p.get("paper_id", p["id"]) for p in papers],
                                 "policy": "all original rows, original split and order"}
        rows = []
        pool = None
        try:
            if workers > 1:
                pool = ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
                                           initializer=_worker_init, initargs=(str(tokenizer_path.resolve()),))
                batches = pool.map(_worker_job, [(p, task, split, evidence_cap, sequence_cap, generated_cap) for p in papers])
            else:
                batches = (prepare_paper(p, task, split, tokenizer, evidence_cap, sequence_cap, generated_cap) for p in papers)
            # map yields in source order, independent of worker completion order.
            for index, batch in enumerate(batches):
                rows.extend(batch)
                if progress and ((index + 1) % 5 == 0 or index + 1 == len(papers)):
                    progress({"task": task, "split": split, "papers_done": index + 1, "papers_total": len(papers), "rows": len(rows)})
        finally:
            if pool is not None:
                pool.shutdown(wait=True, cancel_futures=True)
        prepared[split] = rows
    if input_hashes != {s: file_sha(p) for s, p in inputs.items()} or source_hash != file_sha(source_manifest):
        raise ValueError("source input/manifest changed during preparation")
    if source_hashes != {name: file_sha(ROOT / name) for name in SOURCES}:
        raise ValueError("preparation source changed during preparation")
    if tokenizer_hashes != {p.name: file_sha(p) for p in tokenizer_files}:
        raise ValueError("tokenizer files changed during preparation")
    payloads = {f"{split}.jsonl": "".join(science.canonical(r) + "\n" for r in rows).encode() for split, rows in prepared.items()}
    payloads["source_manifest.json"] = source_manifest.read_bytes()
    for name in SOURCES:
        payloads["source/" + name] = (ROOT / name).read_bytes()
    package_versions = {}
    for package in ("transformers", "tokenizers", "jinja2"):
        try:
            package_versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            package_versions[package] = "not installed (injected test tokenizer)"
    manifest = {"version": VERSION, "task": task, "data_api_version": science.VERSION,
        "sequence_cap": sequence_cap, "evidence_cap": evidence_cap, "generated_cap": generated_cap,
        "thinking": False, "cpu_workers": workers, "tokenizer_sha256": tokenizer_hashes, "packages": package_versions,
        "prl_context_policy": ({"version": science.PRL_POLICY_VERSION, "omission_marker": science.PRL_OMISSION_MARKER,
            "full_body_when_fits": True, "head_tail_budget_fraction": [0.5, 0.5],
            "budget_includes_omission_marker": True, "whole_tex_blocks": True,
            "tail_ends_at_last_prose_block": True, "trailing_caption_fill": False,
            "copy_baseline_input": "selected excerpt blocks; full-body control is diagnostic only"} if task == "prl" else None),
        "input_sha256": input_hashes, "source_manifest_sha256": source_hash, "source_sha256": source_hashes,
        "selections": selections, "counts": {s: summarize(rows) for s, rows in prepared.items()},
        "eligible_training_order": eligible_order(prepared["train"], 480, seed),
        "eligible_development_order": eligible_order(prepared["validation"], development_target, seed),
        "original_split_paper_ids": {s: [p.get("paper_id", p["id"]) for p in rows] for s, rows in splits.items()},
        "evaluation_rows_dropped": 0, "historical_prl_test_policy": "20 historical test papers remain diagnostics; no fresh test created",
        "prl_grounding": "raw-source re-extraction manifest linked when supplied; claim-coverage audit still required before eligibility",
        "test_context_policy": "question/instructions and generation length only; never reference target length",
        "output_sha256": {name: science.sha(data) for name, data in payloads.items()}}
    output.mkdir(parents=True, exist_ok=False)
    for name, data in payloads.items():
        path = output / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(data)
    with (output / "manifest.json").open("x", encoding="utf-8") as stream:
        stream.write(science.canonical(manifest) + "\n")
    return manifest


def prepare_fresh_prl(tokenizer, tokenizer_path, output, fresh_root, historical_root=None,
                      evidence_cap=6656, sequence_cap=8192, generated_cap=512, workers=1, progress=None):
    """All acquired fresh rows, same selector, fresh_test semantics, never training.

    Parent acquisition gates are provenance-bound, not independently re-run here.
    Paper/version exclusion against all historical 100 is independently checked.
    """
    output, fresh_root, tokenizer_path = Path(output), Path(fresh_root), Path(tokenizer_path)
    historical_root = Path(historical_root or ROOT / "data/physics-prl-01")
    if output.exists():
        raise FileExistsError(output)
    if type(workers) is not int or workers < 1:
        raise ValueError("positive CPU worker count required")
    paths = {"fresh": fresh_root / "fresh.jsonl", "acquisition_result": fresh_root / "result.json",
             "acquisition_registration": fresh_root / "registration.json"}
    hashes = {name: file_sha(path) for name, path in paths.items()}
    result = json.loads(paths["acquisition_result"].read_text())
    if not result["completed_acquisition"] or result["test_sha256"] != hashes["fresh"]:
        raise ValueError("fresh acquisition incomplete or source hash differs")
    raw = science.read_rows(paths["fresh"])
    if len(raw) != result["papers"]:
        raise ValueError("fresh acquisition count differs")
    historical_paths = {s: historical_root / f"{s}.jsonl" for s in ("train", "validation", "test")}
    historical_hashes = {s: file_sha(path) for s, path in historical_paths.items()}
    historical_ids = {re.sub(r"v\d+$", "", str(row["paper_id"]))
                      for path in historical_paths.values() for row in science.read_rows(path)}
    fresh_ids = [re.sub(r"v\d+$", "", str(row["paper_id"])) for row in raw]
    if len(set(fresh_ids)) != len(raw) or set(fresh_ids) & historical_ids:
        raise ValueError("fresh paper/version overlaps historical papers or another fresh row")
    if len({r["id"] for r in raw}) != len(raw):
        raise ValueError("duplicate fresh item ID")
    source_hashes = {name: file_sha(ROOT / name) for name in SOURCES}
    tokenizer_files = [p for p in tokenizer_path.iterdir() if p.is_file() and (p.name.startswith("tokenizer") or p.name == "chat_template.jinja")]
    if not tokenizer_files:
        raise ValueError("local tokenizer provenance files required")
    tokenizer_hashes = {p.name: file_sha(p) for p in tokenizer_files}
    pool, prepared = None, []
    try:
        if workers > 1:
            pool = ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
                                       initializer=_worker_init, initargs=(str(tokenizer_path.resolve()),))
            batches = pool.map(_worker_job, [(p, "prl", "fresh_test", evidence_cap, sequence_cap, generated_cap) for p in raw])
        else:
            batches = (prepare_paper(p, "prl", "fresh_test", tokenizer, evidence_cap, sequence_cap, generated_cap) for p in raw)
        for index, batch in enumerate(batches):
            prepared.extend(batch)
            if progress and ((index + 1) % 5 == 0 or index + 1 == len(raw)):
                progress({"task": "prl", "split": "fresh_test", "papers_done": index + 1, "papers_total": len(raw)})
    finally:
        if pool is not None:
            pool.shutdown(wait=True, cancel_futures=True)
    if len(prepared) != len(raw) or any(r["historical_test_diagnostic"] or r["training_eligible"] for r in prepared):
        raise ValueError("fresh rows lost or mislabeled")
    if hashes != {name: file_sha(path) for name, path in paths.items()} or source_hashes != {name: file_sha(ROOT / name) for name in SOURCES}:
        raise ValueError("fresh source/provenance or implementation changed during preparation")
    if historical_hashes != {s: file_sha(path) for s, path in historical_paths.items()} or tokenizer_hashes != {p.name: file_sha(p) for p in tokenizer_files}:
        raise ValueError("historical identity inputs or tokenizer changed during preparation")
    payloads = {"fresh_test.jsonl": "".join(science.canonical(r) + "\n" for r in prepared).encode(),
                "acquisition_result.json": paths["acquisition_result"].read_bytes(),
                "acquisition_registration.json": paths["acquisition_registration"].read_bytes()}
    for name in SOURCES:
        payloads["source/" + name] = (ROOT / name).read_bytes()
    packages = {}
    for name in ("transformers", "tokenizers", "jinja2"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "not installed (injected test tokenizer)"
    manifest = {"version": VERSION, "task": "prl", "split": "fresh_test", "input_sha256": hashes,
        "historical_input_sha256": historical_hashes, "source_sha256": source_hashes,
        "tokenizer_sha256": tokenizer_hashes, "counts": {"fresh_test": summarize(prepared)},
        "cpu_workers": workers, "thinking": False,
        "packages": packages,
        "sequence_cap": sequence_cap, "evidence_cap": evidence_cap, "generated_cap": generated_cap,
        "prl_context_policy_version": science.PRL_POLICY_VERSION, "omission_marker": science.PRL_OMISSION_MARKER,
        "prl_context_policy": {"head_tail_budget_fraction": [0.5, 0.5], "full_body_when_fits": True,
            "budget_includes_omission_marker": True, "whole_tex_blocks": True, "tail_ends_at_last_prose_block": True,
            "trailing_caption_fill": False, "copy_baseline_input": "selected excerpt blocks; full-body control is diagnostic only"},
        "paper_ids": [r["paper_id"] for r in raw], "item_ids": [r["id"] for r in raw],
        "evaluation_rows_dropped": 0, "historical_test_diagnostic": False,
        "test_context_policy": "question/instructions and generation length only; never reference target length",
        "claim_coverage_audit": "required before any training eligibility; fresh rows are evaluation-only",
        "acquisition_gates": "bound to parent result and registration; paper/version exclusion independently checked",
        "output_sha256": {name: science.sha(data) for name, data in payloads.items()}}
    output.mkdir(parents=True, exist_ok=False)
    for name, data in payloads.items():
        path = output / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(data)
    with (output / "manifest.json").open("x", encoding="utf-8") as stream:
        stream.write(science.canonical(manifest) + "\n")
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", choices=["qasper", "prl"], required=True)
    p.add_argument("--tokenizer", default="artifacts/qwen38-27b-nf4-01")
    p.add_argument("--output", required=True)
    p.add_argument("--qasper-root", default="data/qasper-v03-01")
    p.add_argument("--prl-root", default="data/qwen38-prl-abstract-01/historical-reextracted-02")
    p.add_argument("--workers", type=int, default=4, help="independent CPU tokenizer workers; never models")
    p.add_argument("--fresh-prl-root", help="acquired fresh.jsonl/result.json/registration.json; emits fresh_test only")
    a = p.parse_args()
    tokenizer = offline_tokenizer(a.tokenizer)
    if a.fresh_prl_root:
        if a.task != "prl":
            p.error("--fresh-prl-root requires --task prl")
        report = prepare_fresh_prl(tokenizer, a.tokenizer, a.output, a.fresh_prl_root,
            progress=lambda value: print(science.canonical(value), flush=True), workers=a.workers)
    else:
        report = prepare_task(a.task, tokenizer, a.tokenizer, a.output, a.qasper_root, a.prl_root,
                              progress=lambda value: print(science.canonical(value), flush=True), workers=a.workers)
    print(science.canonical({"output": a.output, "counts": report["counts"],
                            "eligible_training_prefix": report.get("eligible_training_order", {}).get("prefix_preview_count", 0)}))


if __name__ == "__main__":
    main()
