"""Build compact aggregate summaries and OFFLINE documentation-only update plans.

Historical result JSONs, original payloads and manifests are read-only inputs.
Remote heads come from recorded publications; later --check-remote must confirm
they remain current. Never uploads, downloads, runs inference or edits cards.
"""
import argparse
import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.common import ROOT
from scripts.save_nf4 import stream_hash
from scripts.update_hf_documentation import ALLOW, DEPENDENCIES, PREFIX, fingerprint, safe_text, validate_stage

TASKS = {"general_mc": (128, "correct"), "physics_mc": (96, "correct"),
         "gsm8k": (128, "fallback_correct"), "ifeval": (32, "prompt_strict_correct")}
EXPECTED = {"bf16_gguf": (92, 62, 119, 25), "q8": (89, 61, 121, 25), "q4": (89, 61, 116, 24),
            "nf4_base": (85, 63, 117, 24), "bf16_base": (92, 63, 119, 24)}
PAIR_FIELDS = ("fixed", "broken", "both", "neither", "b", "c", "total", "discordant",
               "accuracy_delta", "mcnemar_exact_two_sided_p")


def read(filename):
    return json.loads(filename.read_text(encoding="utf-8"))


def serialize(value):
    return (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode()


def publication_inventory(record):
    if isinstance(record["files"], list):
        files = {r["file"]: {"sha256": r["sha256"], "bytes": r.get("bytes")}
                 for r in record["verified_files"]}
        if set(files) != set(record["files"]):
            raise ValueError("published inventory and verified hashes differ")
    else:
        files = {n: {"sha256": r["sha256"], "bytes": r.get("bytes")} for n, r in record["files"].items()}
    for name in record.get("additional_remote_files", []):
        if name != ".gitattributes":
            raise ValueError("unverified additional published files")
        files[name] = {"sha256": None, "bytes": None}
    # The NF4 publication record predates explicit recording of this Hub file.
    files.setdefault(".gitattributes", {"sha256": None, "bytes": None})
    return files


def compact_conditions(final):
    result = {}
    for name, condition in final["conditions"].items():
        scores = {}
        for task, (total, metric) in TASKS.items():
            if task not in condition["summary"]:
                continue
            saved = condition["summary"][task]
            if saved["total"] != total or not 0 <= saved[metric] <= total:
                raise ValueError("unexpected task denominator or count")
            scores[task] = {"total": total, "correct": saved[metric]}
            if task == "gsm8k":
                scores[task].update({k: saved[k] for k in ("strict_correct", "format_compliant", "hit_token_limit")})
        if name in EXPECTED and tuple(scores[t]["correct"] for t in TASKS) != EXPECTED[name]:
            raise ValueError("counts differ from the reviewed completed measurements")
        result[name] = {"scores": scores, "fixture_sha256": condition["fixture_sha256"]}
        for task in ("gsm8k", "ifeval"):
            lengths = condition.get("output_lengths", {}).get(task)
            if lengths:
                result[name]["scores"][task]["generation_diagnostics"] = {
                    k: lengths[k] for k in ("mean_returned_generated_token_ids", "hit_token_limit")}
        if "paper_text" in condition:
            result[name]["paper_text"] = {k: condition["paper_text"][k] for k in ("token_weighted_nll", "perplexity")}
        if "writing" in condition["summary"]:
            writing = condition["summary"]["writing"]
            result[name]["writing_diagnostics"] = {k: writing[k] for k in ("total", "nonempty", "automatic_fidelity_flags")}
            result[name]["writing_diagnostics"]["assessment"] = "deferred; no human quality claim"
    return result


def compact_pairs(final, conditions):
    result = {}
    for name, pair in final["paired_comparisons"].items():
        reference, candidate = name.split("__")
        if conditions[reference]["fixture_sha256"] != conditions[candidate]["fixture_sha256"]:
            raise ValueError("paired fixture mismatch")
        result[name] = {}
        for task in TASKS:
            if task not in pair:
                continue
            r = pair[task]
            before, after = (conditions[n]["scores"][task] for n in (reference, candidate))
            if (r["total"] != before["total"] or sum(r[k] for k in ("both", "neither", "fixed", "broken")) != r["total"]
                    or r["both"] + r["broken"] != before["correct"] or r["both"] + r["fixed"] != after["correct"]
                    or r["b"] != r["broken"] or r["c"] != r["fixed"]):
                raise ValueError("paired counts do not reconcile with condition totals")
            result[name][task] = {k: r[k] for k in PAIR_FIELDS}
    return result


def build(root):
    final_file = root / "results/physics_prl_final.json"
    final = read(final_file)
    if not final["status"].startswith("computational_evaluation_complete"):
        raise ValueError("completed computational evaluation required")
    publication = read(root / "results/huggingface_publication.json")
    records = {r["repo_id"]: r for r in publication["repositories"]}
    for name, filename in (("NF4", "nf4_huggingface_publication.json"),
                           ("PRL-Writing-QLoRA", "physics_huggingface_publication.json")):
        record = read(root / "results" / filename)
        records[PREFIX + name] = record
    conditions = compact_conditions(final)
    pairs = compact_pairs(final, conditions)
    provenance = read(root / "releases/Qwen3-4B-tuneLLM-GGUF/provenance.json")
    published = publication_inventory(records[PREFIX + "GGUF"])
    weights = {r["file"]: r for r in provenance["files"]}
    run_sources = {}
    for name, condition in final["conditions"].items():
        if not conditions[name]["scores"]:
            continue
        run_file = root / condition["evaluation_run"] / "run.json"
        summary_file = run_file.parent / "summary.json"
        run = read(run_file)
        if (stream_hash(run_file) != condition["run_sha256"] or run.get("completed") is not True
                or read(summary_file) != condition["summary"]
                or stream_hash(summary_file) != condition["raw_result_sha256"]["summary.json"]):
            raise ValueError("saved evaluated run or summary changed")
        source = {"commit": run["git_head"]["stdout"], "source_sha256": {
            n: h for n, h in run["source_sha256"].items()
            if n in {"scripts/benchmark_screen.py", "scripts/batched_generation.py", "scripts/save_nf4.py", "evaluation/gsm8k.py"}}}
        run_sources[name] = source
        if name in ("bf16_gguf", "q8", "q4"):
            filename = Path(condition["gguf_file"]).name
            weight = weights[filename]
            if condition["gguf_sha256"] != weight["sha256"] or published[filename]["sha256"] != weight["sha256"]:
                raise ValueError("evaluated GGUF checksum differs from published provenance")
            conditions[name]["weight"] = {k: weight[k] for k in ("file", "bytes", "sha256")}
            conditions[name]["llama_cpp_revision"] = condition["llama_cpp_revision"]
            conditions[name]["llama_server_sha256"] = condition["llama_server_sha256"]
    protocol = read(root / "results/physics_prl_protocol.json")
    datasets = read(root / "configs/benchmark_sources.json")
    common = {"schema_version": 1, "status": "completed computational measurements; writing assessment deferred",
        "model_id": final["model_id"], "upstream_revision": final["revision"],
        "items_per_condition": 384, "task_sizes": {t: n for t, (n, _) in TASKS.items()},
        "scoring": {"general_mc": "custom zero-shot MMLU four-choice conditional letter likelihood",
                    "physics_mc": "custom zero-shot MMLU physics subset, four-choice conditional letter likelihood",
                    "gsm8k": "last-number numeric fallback primary; strict #### final line secondary",
                    "ifeval": "pinned official IFEval prompt-level strict correctness"},
        "dataset_pins": {k: datasets[k] for k in ("mmlu", "ifeval", "ifeval_scorer")},
        "gsm8k_revision": "740312add88f781978c0658806c59bc2815b9866",
        "scorer_sha256": protocol["benchmark_provenance"]["ifeval_source_sha256"],
        "settings": {"seed": 42, "thinking": False, "generation": "greedy", "gsm8k_max_new_tokens": 512,
                     "ifeval_max_new_tokens": 1024, "hf_generation_batch": 8, "hf_mc_batch": 16,
                     "gguf_generation_batch": 1, "gguf_mc_batch": 1},
        "paired_interpretation": "Fixed/broken refer to candidate versus reference. Exact two-sided McNemar p-values are exploratory and unadjusted; nonsignificance or equal totals do not establish equivalence.",
        "limitations": ["Small fixed subsets, not complete standard leaderboard benchmarks; possible base pretraining exposure.",
                        "Custom prompts, numerical representations, runtimes and batch sizes can change answers.",
                        "No universal quantization ranking, equivalence or broad capability preservation established.",
                        "Greedy non-thinking decoding is deliberate; upstream recommends sampling and warns about repetition.",
                        "IFEval language detection can vary with random state; preserve the recorded primary scores and interpret close differences cautiously."],
        "aggregate_source_sha256": stream_hash(final_file)}
    full = {**common, "conditions": conditions, "paired_comparisons": pairs, "evaluated_sources": run_sources}
    earlier = read(root / "results/qlora_real_01.json")
    gsm = earlier["evaluation"]["gsm8k"]
    older = {"protocol_sha256": earlier["protocol_sha256"], "source_sha256": stream_hash(root / "results/qlora_real_01.json"),
             "total": 128, "base_fallback_correct": gsm["base"]["fallback_correct"],
             "adapter_fallback_correct": gsm["adapter"]["fallback_correct"],
             "paired": {k: gsm["paired_primary_fallback"][k] for k in PAIR_FIELDS},
             "scope": "Earlier GSM8K training/evaluation protocol; different fixture/prompt protocol from this 384-item matrix. Not directly comparable."}
    full["earlier_gsm8k_experiment"] = older
    views = {}
    for suffix, names in (("GGUF", ("bf16_gguf", "q8", "q4")), ("NF4", ("nf4_base", "bf16_base")),
                          ("PRL-Writing-QLoRA", ("nf4_base", "nf4_adapter", "bf16_base", "bf16_control"))):
        view = copy.deepcopy(common)
        view["conditions"] = {n: conditions[n] for n in names}
        view["paired_comparisons"] = {k: v for k, v in pairs.items() if set(k.split("__")) <= set(names)}
        view["evaluated_sources"] = {n: run_sources[n] for n in names}
        if suffix == "GGUF":
            view["status"] = "completed benchmark measurements"
            view["evaluated_sources"] = copy.deepcopy(view["evaluated_sources"])
            for source in view["evaluated_sources"].values():
                source["source_sha256"].pop("scripts/save_nf4.py", None)
        views[PREFIX + suffix] = view
    views[PREFIX + "GSM8K-QLoRA"] = {"schema_version": 1, "status": "completed earlier independent experiment",
        "earlier_gsm8k_experiment": older, "limitations": common["limitations"],
        "historical_files": "Existing evaluation.json, provenance and release manifests remain unchanged."}
    return full, views, records


def markdown(summary):
    lines = ["# Benchmark summary", "", "Completed fixed-subset computational measurements. Human writing assessment is deferred and is not required to complete this reporting phase.", "",
             "| Condition | General /128 | Physics /96 | GSM8K /128 | IFEval /32 |", "| --- | ---: | ---: | ---: | ---: |"]
    for name, condition in summary["conditions"].items():
        if condition["scores"]:
            lines.append("| " + name + " | " + " | ".join(str(condition["scores"][t]["correct"]) for t in TASKS) + " |")
    lines += ["", "Each full condition has 384 items. General/physics use custom zero-shot MMLU likelihood subsets; GSM8K uses last-number fallback; IFEval uses official prompt-level strict scoring. These are not full leaderboard scores.", "",
        "Saved NF4 and BF16 Transformers are different workflow representations; GGUF uses llama.cpp with batch one, while Transformers generation uses batch eight. Greedy decoding, non-thinking mode and seed 42 are fixed. GSM8K/IFEval output caps are 512/1024 tokens. Close counts do not establish equivalence or a universal quantization verdict.", "",
        "The earlier GSM8K adapter experiment reported 121→101/128 under a different protocol and must not be compared directly with this matrix. The physics adapter’s 117→111/128 math regression remains reported; lower paper NLL is not evidence of better writing. Human quality assessment is deferred, with no required pending human action for these computational results.", "",
        "The JSON companion records exact dataset/scorer revisions, evaluated implementation commits and hashes, published GGUF weight hashes, paired fixed/broken counts and exploratory unadjusted McNemar tests. Equal totals and nonsignificant tests are not equivalence findings. IFEval language-detection random-state sensitivity further limits close comparisons.", "",
        "Historical evaluation.json, provenance, publication records and release manifests are unchanged. Per-repository documentation updates are staged separately and require a reviewed, current parent commit before application.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="artifacts/report-updates/benchmark-01")
    args = parser.parse_args()
    out = ROOT / args.output
    targets = (ROOT / "results/benchmark_summary.json", ROOT / "docs/benchmark_summary.md")
    if out.exists() or any(p.exists() for p in targets):
        raise FileExistsError("refusing to overwrite report staging or consolidated summaries")
    full, views, records = build(ROOT)
    staged = []
    for repo, view in views.items():
        name = repo.split("/")[1]
        card = (ROOT / "releases" / name / "README.md").read_bytes()
        report = serialize(view)
        safe_text(card)
        safe_text(report)
        record = records[repo]
        manifest = {"schema_version": 1, "repo_id": repo, "private": record["private"],
            "parent_commit": record["revision"], "allowed_changes": sorted(ALLOW),
            "expected_inventory": publication_inventory(record),
            "updates": {"README.md": fingerprint(card), "benchmark_summary.json": fingerprint(report)},
            "remote_head_verified_now": False,
            "scope": "Offline staged against recorded publication; remote inventory/head must pass immediately before apply."}
        if repo in DEPENDENCIES:
            manifest["pinned_dependency"] = DEPENDENCIES[repo]
        with TemporaryDirectory() as temp:
            directory = Path(temp)
            (directory / "README.md").write_bytes(card)
            (directory / "benchmark_summary.json").write_bytes(report)
            validate_stage(manifest, directory)
        staged.append((name, card, report, manifest))
    safe_text(serialize(full))
    for name, card, report, manifest in staged:
        directory = out / name
        directory.mkdir(parents=True)
        (directory / "README.md").write_bytes(card)
        (directory / "benchmark_summary.json").write_bytes(report)
        (directory / "update_manifest.json").write_bytes(serialize(manifest))
        validate_stage(manifest, directory)
    targets[0].write_bytes(serialize(full))
    targets[1].write_text(markdown(full), encoding="utf-8")
    print("Offline benchmark reports staged; no remote access or update.")


if __name__ == "__main__":
    main()
