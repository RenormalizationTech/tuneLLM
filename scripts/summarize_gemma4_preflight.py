"""Create a compact decision report from preserved local Gemma mechanics receipts."""
import datetime
import json
from pathlib import Path

from scripts.gemma4_preflight import ROOT, sha, write


def main():
    first=ROOT/"outputs/gemma4-preflight-01";second=ROOT/"outputs/gemma4-preflight-02"
    read=lambda p:json.loads(p.read_text())
    a=read(first/"gpu_result.json");b=read(second/"gpu_result.json")
    reload=read(second/"fresh_reload.json");steady=read(second/"steady_memory.json")
    cpu=read(second/"cpu_prepare.json");weights=read(second/"weights_verification.json")
    interruption=read(first/"interruption.json")
    if weights["status"]!="passed" or reload["status"]!="passed" or steady["status"]!="passed":
        raise ValueError("preflight or fresh reload is incomplete")
    if a["revision"]!=b["revision"] or b["revision"]!=weights["revision"]:
        raise ValueError("model revision mismatch")
    if b["source_sha256"]!=sha(ROOT/"scripts/gemma4_preflight.py"):
        raise ValueError("published preflight source differs from executed attempt02")
    if a["source_sha256"]!=sha(first/"executed_source.py"):
        raise ValueError("attempt01 source snapshot differs from its receipt")
    if steady["source_sha256"]!=sha(ROOT/"scripts/gemma4_steady_memory.py"):
        raise ValueError("steady-state source differs")
    if a["status"]!="running" or interruption["exit_code"]!=143:
        raise ValueError("unexpected attempt01 termination provenance")
    probes=a["probes"]+b["probes"]
    if any(p["status"]!="passed" or not p["finite_gradients"] for p in probes):
        raise ValueError("a completed probe did not pass")
    coverage={task:{split:{k:v for k,v in values.items() if k in
        {"source_sha256","eligible","ineligible","median","p95","maximum","over_8192",
         "total_tokens","response_tokens","maximum_response_tokens"}}
        for split,values in splits.items()} for task,splits in cpu["coverage"].items()}
    # Workload scenarios use a constant token rate measured at 8192 tokens.
    # They are not bounds or confidence intervals and omit evaluation/checkpoints.
    rates=[8192/p["seconds"] for p in probes if p["sequence_length"]==8192]
    rates.append(8192/steady["full_step_seconds"])
    low,high=min(rates),max(rates)
    scenarios={}
    for task,protocol_name in [("qasper","qwen38-qasper-02"),("prl","qwen38-prl-01")]:
        protocol_path=ROOT/"results/protocols"/(protocol_name+".json");protocol=read(protocol_path)
        epochs=int(protocol["config"]["epochs"])
        lengths={r["id"]:r["length"] for r in cpu["coverage"][task]["train"]["length_rows"]}
        selected=protocol["selected_ids"]["train"]
        if set(selected)-set(lengths):raise ValueError("matched training examples missing")
        for split in ("train","validation"):
            if protocol["config"]["prepared_input_sha256"][split]!=coverage[task][split]["source_sha256"]:
                raise ValueError("cross-family inputs differ from the previous prepared data")
        total=sum(lengths[i] for i in selected)*epochs
        scenarios[task]={"matched_training_examples":len(selected),"epochs":epochs,
            "total_training_tokens":total,"constant_8192_token_rate_hours":[total/high/3600,total/low/3600],
            "comparison_selection_protocol_sha256":sha(protocol_path)}
    all_q=coverage["qasper"]["train"]["total_tokens"]
    scenarios["qasper_all_eligible_one_epoch"]={"training_examples":coverage["qasper"]["train"]["eligible"],
        "total_training_tokens":all_q,"constant_8192_token_rate_hours":[all_q/high/3600,all_q/low/3600]}
    raw_paths=[first/"gpu_result.json",first/"interruption.json",first/"executed_source.py",
        second/"cpu_prepare.json",second/"gpu_result.json",second/"fresh_reload.json",
        second/"steady_memory.json",second/"weights_verification.json"]
    result={"recorded_at_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "status":"compatibility_and_synthetic_training_feasibility_passed",
        "decision":"Awaiting owner decision before any Gemma QASPER or PRL task training",
        "scope":"Text-only compatibility, train/development tokenizer coverage, synthetic optimizer steps and fresh-process reload; no Gemma final-test generation or judging",
        "model":{"repo_id":"google/gemma-4-31B-it","revision":b["revision"],
            "loader":b["loading"]["loader"],"weights":weights["weights"],"vision_modules_loaded":False},
        "software":b["versions"],"GPU":{"name":b["GPU"]["name"],"total_bytes":b["GPU"]["total_bytes"]},
        "training_profile":{"rank":16,"alpha":32,"dropout":0.0,"batch_size":1,
            "gradient_checkpointing":"non-reentrant","attention":"SDPA","optimizer":"torch AdamW",
            "trainable_parameters":b["trainable_parameters"],"trainable_tensors":b["trainable_tensors"],
            "projection_count":cpu["target_count"],"projection_suffix_counts":cpu["target_suffix_counts"],
            "floating_weight_recipe":b["floating_weight_recipe"],"source_sha256":b["source_sha256"]},
        "masking":{"format":cpu["training_format"],"supervised_native_turn_end_id":106,
            "generic_EOS_id_not_appended":1,"CPU_tests_passed":7,"full_vs_compact_CE":b["response_only_CE"]},
        "coverage":coverage,"memory_after_NF4_load":b["memory_after_NF4_load"],
        "memory_after_PEFT_preparation":b["memory_after_PEFT_preparation"],"completed_probes":probes,
        "steady_state_probe":steady,"fresh_process_reload":reload,
        "generation_mechanics":{"short_smokes_source":{"attempt":"02","receipt_sha256":sha(second/"gpu_result.json"),
            "executed_source_sha256":b["source_sha256"]},
            "long_synthetic_probe_source":{"attempt":"01","receipt_sha256":sha(first/"gpu_result.json"),
            "executed_source_sha256":a["source_sha256"],"completed_before_later_SIGTERM":True},
            "short_smokes":[{k:v for k,v in row.items() if k not in {"output_ids","text"}}
            for row in b["base_generation_smokes"]],"long_synthetic_probe":{k:v for k,v in
            a["long_context_generation_throughput"].items() if k not in {"output_ids","text"}}},
        "training_cost_scenarios":scenarios,"measured_8192_training_tokens_per_second_range":[low,high],
        "cost_interpretation":"Simple constant-token-rate scenarios, not predicted wall time, confidence intervals or bounds. Synthetic lengths/targets, warmup, per-step optimizer updates, gradient checks and attention scaling differ from a real accumulated-gradient run. Validation, checkpointing, final generation, retention and AI review are excluded; end-to-end evaluation cost is not estimated.",
        "interruption":{"attempt":"01","exit_code":143,"cause":"Unestablished external SIGTERM",
            "larger_response_probe_in_attempt01":"incomplete; not scored as a pass or an OOM",
            "resolution":"Independent attempt02 completed 8192/512 and fresh reload; separate steady-state probe verified resident Adam moments"},
        "limits":["Synthetic mechanics do not establish task-quality improvements or convergence",
            "The exact prepared training/development inputs were reused; original paper excerpts were not expanded to exploit Gemma's context window",
            "No Gemma task adapter was trained, selected, published or compared on final-test data",
            "The separate recorded demonstration displays existing Qwen final outputs; reused project evaluation sets would not be new heldout evidence for a future Gemma study",
            "One RTX A6000/software stack; other hardware, batch sizes and longer contexts untested",
            "Short generation ends naturally; long synthetic generation forces 256 tokens for throughput and has no quality interpretation"],
        "raw_local_receipt_sha256":{p.relative_to(ROOT).as_posix():sha(p) for p in raw_paths},
        "report_source_sha256":sha(__file__)}
    write(ROOT/"results/gemma4_preflight_01.json",result)
    print(json.dumps({"status":result["status"],"training_cost_scenarios":scenarios,
        "steady_state_probe":steady},indent=2))


if __name__=="__main__":main()
