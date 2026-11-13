"""Summarize a fixed final evaluation; never choose a model from final-test results."""
import argparse

from scripts.common import ROOT, digest, path, read_json, write_json
from scripts.physics_selection import paired_bootstrap
from scripts.summarize_qlora_run import paired_counts


def relative(filename):
    return str(path(filename).resolve().relative_to(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mapping", required=True)
    parser.add_argument("--output", default="results/physics_prl_final.json")
    args = parser.parse_args()
    mapping = read_json(args.mapping)
    if mapping.get("split") != "test" or not mapping.get("selection_fixed"):
        raise ValueError("final-test mapping must declare that selection is already fixed")
    selection = path(mapping["selection_report"])
    if digest(selection) != mapping["selection_report_sha256"]:
        raise ValueError("fixed development-selection report changed")
    if path(args.output).exists():
        raise FileExistsError(args.output)
    conditions, raw, paper_reports = {}, {}, {}
    for name, spec in mapping["conditions"].items():
        out = path(spec["directory"])
        run = read_json(out / "run.json")
        if not run.get("completed"):
            raise ValueError(f"incomplete evaluation: {name}")
        summary = read_json(out / "summary.json")
        records = {}
        for task in ["general_mc", "physics_mc"]:
            if task in summary:
                records[task] = (read_json(out / f"{task}.json"), "correct")
        if "gsm8k" in summary:
            records["gsm8k"] = (read_json(out / "gsm8k_scores.json")["details"], "fallback_correct")
        if "ifeval" in summary:
            records["ifeval"] = (read_json(out / "ifeval_scores.json")["records"], "prompt_strict")
        raw[name] = records
        conditions[name] = {"representation": spec["representation"], "summary": summary,
            "evaluation_run": relative(out), "run_sha256": digest(out / "run.json"),
            "fixture_sha256": run["fixture_sha256"], "writing_sha256": run.get("writing_sha256"),
            "generation_batch": run["generation_batch"], "wall_seconds": run.get("wall_seconds"),
            "peak_pytorch_allocated_bytes": run.get("peak_pytorch_allocated_bytes"),
            "generation_diagnostics": run.get("generation_diagnostics"),
            "raw_result_sha256": {f.name: digest(f) for f in out.glob("*.json")},
            "checkpoint": spec.get("checkpoint"), "checkpoint_sha256": spec.get("checkpoint_sha256"),
            "prompt_condition": spec.get("prompt_condition", "zero-shot")}
        output_lengths = {}
        for task in ["gsm8k", "ifeval", "writing"]:
            filename = out / f"{task}_predictions.json"
            if not filename.exists():
                continue
            predictions = read_json(filename)
            output_lengths[task] = {"items": len(predictions),
                "mean_input_token_ids": sum(len(r["input_ids"]) for r in predictions) / len(predictions),
                "mean_returned_generated_token_ids": sum(len(r["generated_ids"]) for r in predictions) / len(predictions),
                "hit_token_limit": sum(bool(r["hit_token_limit"]) for r in predictions)}
        conditions[name]["output_lengths"] = output_lengths
        overlap_file = out / "training_overlap.json"
        if overlap_file.exists():
            # Per-output word counts could reveal the writing packet's A/B key.
            conditions[name]["training_overlap_diagnostic"] = {k: v for k, v in read_json(overlap_file).items() if k != "records"}
        for field in ("gguf_file", "gguf_sha256", "llama_cpp_revision", "llama_server_sha256"):
            if field in spec:
                conditions[name][field] = spec[field]
        if spec.get("paper_nll"):
            filename = path(spec["paper_nll"])
            report = read_json(filename)
            if report.get("split") != "test":
                raise ValueError("final paper evaluation requires a test-split report")
            report = report.get("metrics", report)
            if len(report["per_paper"]) != 20:
                raise ValueError("final paper evaluation requires all20 held-out papers")
            paper_reports[name] = report
            conditions[name]["paper_text"] = {k: v for k, v in report.items() if k not in {"chunks", "chunk_records", "per_paper", "records"}}
            conditions[name]["paper_nll_sha256"] = digest(filename)
    comparisons = {}
    paper_only = {}
    for name, spec in mapping.get("paper_only_conditions", {}).items():
        if name in conditions:
            raise ValueError("duplicate paper-only condition")
        filename = path(spec["paper_nll"])
        saved = read_json(filename)
        report = saved.get("metrics", saved)
        if saved.get("split") != "test" or len(report["per_paper"]) != 20:
            raise ValueError("paper-only condition requires all20 final-test papers")
        paper_reports[name] = report
        paper_only[name] = {**spec, "paper_nll_sha256": digest(filename),
                            "token_weighted_nll": report["token_weighted_nll"],
                            "perplexity": report["perplexity"]}
    for reference, candidate in mapping["paired_comparisons"]:
        a, b = conditions[reference], conditions[candidate]
        if a["fixture_sha256"] != b["fixture_sha256"]:
            raise ValueError("paired benchmark fixture hashes differ")
        comparison = {}
        for task in sorted(set(raw[reference]) & set(raw[candidate])):
            before, metric = raw[reference][task]
            after, other_metric = raw[candidate][task]
            if metric != other_metric:
                raise ValueError("paired scoring metric differs")
            comparison[task] = paired_counts(before, after, metric)
        if reference in paper_reports and candidate in paper_reports:
            comparison["paper_text_nll"] = paired_bootstrap(paper_reports[candidate], paper_reports[reference])
        comparisons[reference + "__" + candidate] = comparison
    paper_comparisons = {reference + "__" + candidate:
                         paired_bootstrap(paper_reports[candidate], paper_reports[reference])
                         for reference, candidate in mapping.get("paper_only_comparisons", [])}
    training = []
    for directory in mapping.get("training_runs", []):
        run = read_json(path(directory) / "run.json")
        cfg = read_json(path(directory) / "config.json")
        if not run.get("completed"):
            raise ValueError("incomplete registered training run")
        training.append({"run": relative(directory), "rank": cfg["rank"], "learning_rate": cfg["learning_rate"],
            "seed": cfg["seed"], "representation": cfg["representation"], "epochs": cfg["epochs"],
            "steps": run["global_step"], "exposure": run["exposure"], "trainable_parameters": run["trainable_parameters"],
            "wall_seconds": run["training_wall_seconds"], "peak_pytorch_allocated_bytes": run["peak_training_allocated_bytes"],
            "final_epoch_serialization_probe": run["replay"],
            "learning_curve": [r for r in read_json(path(directory) / "learning_curve.json") if "eval_loss" in r],
            "run_sha256": digest(path(directory) / "run.json")})
    write_json(args.output, {"status": "computational_evaluation_complete_human_writing_review_pending",
        "model_id": "Qwen/Qwen3-4B", "revision": "1cfa9a7208912126459214e8b04321603b3df60c",
        "summarizer_source_sha256": digest(__file__), "final_mapping_sha256": digest(args.mapping),
        "selection_report": relative(selection), "selection_report_sha256": digest(selection),
        "lr_selection_report_sha256": digest("results/physics_lr_selection.json"),
        "nltk_resource_report_sha256": digest("results/physics_nltk_resource.json"),
        "corpus_manifest_sha256": digest("results/physics_prl_corpus.json"),
        "conditions": conditions, "paired_comparisons": comparisons, "training_runs": training,
        "paper_only_conditions": paper_only, "paper_only_comparisons": paper_comparisons,
        "corpus_quality": read_json("results/physics_prl_corpus_quality.json") if path("results/physics_prl_corpus_quality.json").exists() else None,
        "ifeval_language_stability": read_json("results/physics_ifeval_language_stability.json") if path("results/physics_ifeval_language_stability.json").exists() else None,
        "local_human_review": read_json("results/physics_writing_review.json") if path("results/physics_writing_review.json").exists() else None,
        "human_writing_ratings": "pending", "writing_quality_improvement_established": False,
        "limits": ["One small paper corpus; arXiv versions can precede final PRL wording; existing pretraining exposure unknown.",
                   "No matched PRB corpus control, so lower text loss is not evidence of PRL-specific style learning.",
                   "Paper splits are disjoint but some authors overlap; theory labels were AI-reviewed.",
                   "Loss uses identical1024-token within-paper chunks, not whole-paper context.",
                   "Small benchmark subsets and unadjusted exploratory paired tests do not establish broad capability equivalence.",
                   "GGUF and HF inference use different backends and batch sizes; decoding settings and input IDs are checked.",
                   "Automatic fidelity flags and exact text overlap are diagnostics, not human scientific or style judgments.",
                   "Ranks receive equal maximum training budgets; their nominated checkpoints can come from different epochs.",
                   "Rank8's minimum occurred at the three-epoch budget boundary; convergence beyond that budget was not tested.",
                   "The BF16 control uses the NF4-selected rank, learning rate and epoch; it is not independently optimized."]})
    print(path(args.output))


if __name__ == "__main__":
    main()
