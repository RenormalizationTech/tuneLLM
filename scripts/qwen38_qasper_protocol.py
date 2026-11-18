"""Freeze QASPER evaluation and select among three bounded development prompts."""
import argparse
import importlib.util
import json
from pathlib import Path
import shutil

from scripts.qwen38_science_runtime import sha, write
from scripts.qwen38_train_science import paper_subset

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "outputs/qwen38-science-01/prepared-qasper-tokenizer-01"
OUT = ROOT / "outputs/qwen38-qasper-evaluation-01"
REGISTRY = ROOT / "results/protocols/qwen38-qasper-evaluation-01.json"


def rows(path):
    return [json.loads(line) for line in path.read_text().split("\n") if line.strip()]


def prepare():
    if OUT.exists() or REGISTRY.exists():
        raise FileExistsError("fresh evaluation registration required")
    manifest = json.loads((DATA / "manifest.json").read_text())
    for split in ("train", "validation", "test"):
        if sha(DATA / f"{split}.jsonl") != manifest["output_sha256"][f"{split}.jsonl"]:
            raise ValueError("prepared QASPER inputs changed")
    subset = paper_subset(rows(DATA / "validation.jsonl"), 32, 20261117)
    OUT.mkdir(parents=True); (OUT / "source").mkdir()
    target = OUT / "prompt_development.jsonl"
    target.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in subset))
    for name in ("qwen38_science_score.py", "qwen38_science_data.py"):
        shutil.copyfile(DATA / "source/scripts" / name, OUT / "source" / name)
    for name in ("qwen38_qasper_protocol.py", "qwen38_eval_science.py", "qwen38_science_runtime.py"):
        shutil.copyfile(ROOT / "scripts" / name, OUT / "source" / name)
    protocol = {"task": "QASPER text-only QA/evidence, one seed",
        "prepared_manifest_sha256": sha(DATA / "manifest.json"),
        "training_protocol_sha256": sha(ROOT / "results/protocols/qwen38-qasper-01.json"),
        "final_input_sha256": sha(DATA / "test.jsonl"), "development_input_sha256": sha(DATA / "validation.jsonl"),
        "prompt_selection_input_sha256": sha(target), "prompt_selection_ids": [r["id"] for r in subset],
        "prompt_selection": "largest strict primary Answer F1 among base/verify/schema; ties retain base then verify then schema",
        "prompt_selection_timing": "fixed subset before variant outputs; the full base-development run is already in progress; no QA outputs/scores inspected",
        "prompt_candidates": ["base", "verify", "schema"],
        "sampling": "seed20261117 whole-paper prefix target32 for prompt selection; original frozen150dev/403test unchanged",
        "generation": {"max_new_tokens": 512, "sequence_cap": 8192, "seed": 42, "batch_size": 1,
                       "thinking": False, "cached": True, "do_sample": False},
        "counts": {split: manifest["counts"][split] for split in ("train", "validation", "test")},
        "primary": "frozen complete-text-evidence questions plus full-context unanswerable cases; mixed answerability excluded",
        "evidence_gate": "frozen evidence_strict_primary subset; every answerable reference has nonempty text evidence",
        "success_gates": {"answer_f1_gain_each_baseline": 0.05, "paired_paper_bootstrap_ci_excludes_zero": True,
                          "max_evidence_f1_loss_vs_better_answer_baseline": 0.02,
                          "max_invalid_json_fraction": 0.02, "max_false_unanswerable_increase": 0.05,
                          "retention": {"gsm8k_total": 128, "gsm8k_max_net_loss": 5,
                                        "general_mmlu_total": 128, "general_mmlu_max_net_loss": 5,
                                        "ifeval_total": 32, "ifeval_max_strict_net_loss": 3}},
        "bootstrap": {"draws": 10000, "seed": 20261117, "unit": "paper", "method": "paired percentile95"},
        "limits": ["Public NLP papers may have been in pretraining", "No physics-generalization claim from QASPER",
                   "Normalized paragraph-ID scores are custom; exact upstream-string evidence score reported separately",
                   "Point-estimate retention/format gates are not statistical non-inferiority tests"],
        "source_sha256": {p.name: sha(p) for p in sorted((OUT / "source").glob("*.py"))}}
    write(OUT / "registration.json", protocol); write(REGISTRY, protocol)
    return {"registered": str(REGISTRY.relative_to(ROOT)), "prompt_selection_rows": len(subset),
            "test_rows": manifest["counts"]["test"]["records"]}


def select():
    protocol = json.loads(REGISTRY.read_text())
    if REGISTRY.read_bytes() != (OUT / "registration.json").read_bytes():
        raise ValueError("evaluation registration changed")
    destination = OUT / "prompt_selection.json"
    if destination.exists():
        raise FileExistsError("prompt selection already made")
    subset = rows(OUT / "prompt_development.jsonl")
    scorer_path = OUT / "source/qwen38_science_score.py"
    if sha(scorer_path) != protocol["source_sha256"][scorer_path.name]:
        raise ValueError("frozen scorer changed")
    spec = importlib.util.spec_from_file_location("frozen_qasper_score", scorer_path)
    scorer = importlib.util.module_from_spec(spec); spec.loader.exec_module(scorer)
    scores = {}
    for template in protocol["prompt_candidates"]:
        run = ROOT / f"outputs/qwen38-qasper-development-{template}-01"
        status = json.loads((run / "result.json").read_text())
        if not status.get("completed"):
            raise ValueError(f"development run incomplete: {template}")
        ids = {r["id"] for r in subset}
        predictions = {r["id"]: r["output"] for r in rows(run / "predictions.jsonl") if r["id"] in ids}
        if set(predictions) != ids:
            raise ValueError("development selection inputs do not align")
        score = scorer.score_qasper(subset, predictions)
        write(OUT / f"development_{template}_scores.json", score)
        scores[template] = score["strict"]["groups"]["primary"]["answer_f1"]
    chosen = max(protocol["prompt_candidates"], key=lambda name: scores[name])
    result = {"chosen": chosen, "strict_primary_answer_f1": scores,
              "selection_input_sha256": protocol["prompt_selection_input_sha256"],
              "selection_implementation_sha256": sha(__file__),
              "selection_implementation_note": "scorer groups API key corrected; registered data, metric, templates and tie rule unchanged",
              "rule": protocol["prompt_selection"], "final_test_used": False}
    with (OUT / "source/selection_implementation.py").open("xb") as stream:
        stream.write(Path(__file__).read_bytes())
    write(destination, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--prepare", action="store_true")
    group.add_argument("--select", action="store_true")
    args = parser.parse_args()
    print(json.dumps(prepare() if args.prepare else select(), indent=2))


if __name__ == "__main__":
    main()
