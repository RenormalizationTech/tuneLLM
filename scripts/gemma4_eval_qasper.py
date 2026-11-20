"""Registered Gemma QASPER generations and frozen Qwen-compatible scoring.

35 fixed development questions x three prompts; 403 fixed final questions per
arm. Completed rows never repeat. Interrupted rows preserve their starts and
restart only under explicit --resume; deterministic failures need an amendment.
"""
import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import time

from scripts import gemma4_qasper_runtime as rt
from scripts.qwen38_eval_science import PROMPTS
from scripts.qwen38_science_score import score_qasper

SOURCES = ["scripts/qwen38_eval_science.py", "scripts/qwen38_science_runtime.py",
           "scripts/qwen38_science_score.py"]


def coverage(args,tokenizer=None):
    """CPU token counts only; reads final prompts, generates/scores nothing."""
    from transformers import AutoTokenizer
    frozen = rt.read(rt.ROOT/"results/protocols/qwen38-qasper-evaluation-01.json")
    identity = rt.model_identity(args.base)
    tokenizer = tokenizer or AutoTokenizer.from_pretrained(args.base,local_files_only=True)
    sources = {
        "development":(rt.ROOT/"outputs/qwen38-science-01/prepared-qasper-tokenizer-01/validation.jsonl",frozen["development_input_sha256"]),
        "prompt_selection":(rt.ROOT/"outputs/qwen38-qasper-evaluation-01/prompt_development.jsonl",frozen["prompt_selection_input_sha256"]),
        "final":(rt.ROOT/"outputs/qwen38-science-01/prepared-qasper-tokenizer-01/test.jsonl",frozen["final_input_sha256"])}
    entries = []; summary = {}
    for split,(path,expected) in sources.items():
        if rt.sha(path) != expected:
            raise ValueError("coverage source bytes differ")
        data = rt.rows(path)
        for template in PROMPTS:
            lengths = []
            for row in data:
                prompt,ids = rt.render(tokenizer,PROMPTS[template]+row["user"])
                lengths.append(len(ids))
                entries.append({"split":split,"template":template,"id":row["id"],"prompt_tokens":len(ids),
                                "sequence_allowance":len(ids)+512})
            summary[split+"/"+template] = {"count":len(lengths),"max_prompt":max(lengths),
                "maximum_with_cap":max(lengths)+512,"over_8192":sum(n+512 > 8192 for n in lengths)}
    if any(s["over_8192"] for s in summary.values()):
        raise ValueError("coverage overflow: stop before any generation, no truncation")
    return rt.register(args.output,{"kind":"coverage","identity":identity,
        "summary":summary,"gpu_launched":False,"source_data":{s:{"path":str(p),"sha256":h} for s,(p,h) in sources.items()}},
        {"coverage.json":entries},SOURCES)


def adapter_binding(training):
    training = Path(training).resolve()
    result = rt.read(training / "result.json")
    cfg = rt.read(training / "registration.json")
    if not result.get("completed") or cfg["kind"] != "training" or cfg["task"] != "qasper":
        raise ValueError("completed selected Gemma QASPER training required")
    if result["registration_sha256"] != rt.sha(training / "registration.json"):
        raise ValueError("training registration differs")
    rt.validate(training,result["registration_sha256"])
    if result["selected"] != min(result["checkpoints"],key=lambda row:(row["nll"],row["step"])):
        raise ValueError("selected adapter differs from registered NLL rule")
    selected = Path(result["selected"]["path"]).resolve()
    if selected.parent.parent != training or result["selected"]["step"] not in (21,41,61):
        raise ValueError("selected adapter escapes registered training checkpoints")
    rt.verify_files(selected,result["selected_files"])
    return {"path":str(selected),"files":result["selected_files"],
            "training_result":str(training / "result.json"),
            "training_result_sha256":rt.sha(training / "result.json"),
            "training_registration_sha256":result["registration_sha256"],
            "identity":cfg["identity"]}


def prepare(args, tokenizer=None):
    from transformers import AutoTokenizer
    frozen = rt.read(rt.ROOT / "results/protocols/qwen38-qasper-evaluation-01.json")
    source = (rt.ROOT / "outputs/qwen38-qasper-evaluation-01/prompt_development.jsonl"
              if args.purpose == "development" else
              rt.ROOT / "outputs/qwen38-science-01/prepared-qasper-tokenizer-01/test.jsonl")
    expected = frozen["prompt_selection_input_sha256" if args.purpose == "development" else "final_input_sha256"]
    if rt.sha(source) != expected:
        raise ValueError("frozen evaluation input bytes differ")
    data = rt.rows(source)
    if len(data) != (35 if args.purpose == "development" else 403) or len({r["id"] for r in data}) != len(data):
        raise ValueError("exact evaluation count/unique IDs required")
    if args.purpose == "development" and [r["id"] for r in data] != frozen["prompt_selection_ids"]:
        raise ValueError("development IDs/order differ")
    identity = rt.model_identity(Path(args.base)); geometry = rt.geometry(args.base)
    if not args.coverage:
        raise ValueError("CPU all-split/all-prompt coverage registration required")
    coverage_path = Path(args.coverage)/"registration.json"
    cov = rt.read(coverage_path)
    expected_summaries = {split+"/"+template:count for split,count in
                          (("development",150),("prompt_selection",35),("final",403)) for template in PROMPTS}
    if (cov["kind"] != "coverage" or cov["identity"] != identity or
        set(cov["summary"]) != set(expected_summaries) or
        any(s["over_8192"] or s["maximum_with_cap"] > 8192 or s["count"] != expected_summaries[k]
            for k,s in cov["summary"].items())):
        raise ValueError("coverage registration differs or has overflows")
    tokenizer = tokenizer or AutoTokenizer.from_pretrained(args.base,local_files_only=True)
    templates = list(PROMPTS) if args.purpose == "development" else [args.template]
    if args.purpose == "development" and args.training:
        raise ValueError("prompt selection is unadapted only")
    selection = None
    if args.purpose == "final" and args.template != "base":
        if not args.selection:
            raise ValueError("final prompted arm requires frozen development selection")
        selection = rt.read(args.selection)
        if selection["chosen"] != args.template or selection["final_test_used"]:
            raise ValueError("final template differs from development selection")
    if args.training and args.template != "base":
        raise ValueError("adapter uses base preparation; no extra prompted adapter arm")
    rendered = []
    for template in templates:
        for index,row in enumerate(data):
            prompt,ids = rt.render(tokenizer,PROMPTS[template]+row["user"])
            if len(ids)+512 > 8192:
                raise ValueError("full prompt exceeds8192; no truncation")
            rendered.append({"ordinal":len(rendered),"id":row["id"],"paper_id":row["paper_id"],
                             "template":template,"rendered_prompt":prompt,"input_ids":ids})
    adapter = adapter_binding(args.training) if args.training else None
    if adapter and adapter["identity"] != identity:
        raise ValueError("adapter base identity differs")
    analysis = None
    if args.purpose == "final":
        if not args.analysis_registration:
            raise ValueError("PM final analysis registration required before final generation")
        analysis = {"path":str(Path(args.analysis_registration).resolve()),"sha256":rt.sha(args.analysis_registration)}
    cfg = {"kind":"evaluation","task":"qasper","purpose":args.purpose,"gpu":args.gpu,
        "identity":identity,"targets":geometry["targets"],"adapter":adapter,
        "templates":templates,"selection":selection,"final_analysis_registration":analysis,
        "coverage_registration":{"path":str(coverage_path.resolve()),"sha256":rt.sha(coverage_path)},
        "selection_sha256":rt.sha(args.selection) if args.selection else None,
        "input_path":str(source),"input_sha256":expected,"max_seconds":args.max_seconds,
        "max_new_tokens":512,"max_context":8192,"batch_size":1,"eos_token_id":[1,106],
        "autocast":True,"cached":True,"seed":42,"do_sample":False,
        "base_preparation":"identical FP32 kbit casts and rank16 wrapper; disabled for baseline"}
    return rt.register(args.output,cfg,{"data.json":data,"prepared.json":rendered},SOURCES)


def receipt_rows(run, prepared, binding=None, allow_interrupted=False):
    output = []
    for row in prepared:
        directory = Path(run)/"rows"/f"{row['ordinal']:05d}"
        if directory.exists():
            if not (directory/"result.json").exists():
                if allow_interrupted:
                    continue
                raise RuntimeError(f"interrupted generation {row['ordinal']}; explicit resume required")
            result = rt.read(directory/"result.json")
            receipt = rt.read(directory/"receipt.json")
            if rt.sha(directory/"result.json") != receipt["sha256"]:
                raise ValueError("generation receipt hash differs")
            if any(result.get(k) != v for k,v in row.items()) or (binding and result.get("registration_sha256") != binding):
                raise ValueError("row receipt identity differs")
            output.append(result)
    return output


def run_generations(args, retention=False):
    module = "scripts.gemma4_retention" if retention else "scripts.gemma4_eval_qasper"
    run = Path(args.run).resolve(); binding = args.registration_sha256
    cfg = rt.frozen_exec(module,run,binding,args.resume)
    if cfg.get("kind") != ("retention" if retention else "evaluation"):
        raise ValueError("job registration kind differs from executor")
    with rt.exclusive(run):
        return execute_rows(run,cfg,binding,args.resume,retention)


def execute_rows(run,cfg,binding,resume,retention=False):
    import torch
    from transformers import AutoTokenizer,set_seed
    from scripts.gemma4_train_qasper import elapsed_attempts,recover_attempts
    prepared = rt.read(run/"prepared.json")
    existing = receipt_rows(run,prepared,binding,allow_interrupted=resume)
    attempts = run/"attempts"; attempts.mkdir(exist_ok=True)
    if list(attempts.glob("*.json")) and not resume:
        raise ValueError("explicit --resume required")
    if resume:
        recover_attempts(run)
    previous = elapsed_attempts(run)
    if cfg["max_seconds"] is not None and previous >= cfg["max_seconds"]:
        raise TimeoutError("cumulative generation cap exhausted")
    if (run/"result.json").exists() and rt.read(run/"result.json").get("completed"):
        raise FileExistsError("completed generation immutable")
    deadline = time.monotonic()+cfg["max_seconds"]-previous if cfg["max_seconds"] is not None else float("inf")
    analysis = cfg.get("final_analysis_registration")
    if retention and cfg.get("adapter") and not analysis:
        raise ValueError("adapter retention requires prospective analysis registration")
    if analysis and rt.sha(analysis["path"]) != analysis["sha256"]:
        raise ValueError("PM final analysis registration changed")
    coverage_binding = cfg.get("coverage_registration")
    if coverage_binding and rt.sha(coverage_binding["path"]) != coverage_binding["sha256"]:
        raise ValueError("CPU coverage registration changed")
    gpu_uuid = rt.require_idle(cfg["gpu"])
    attempt_file = attempts/f"attempt-{len(list(attempts.glob('*.json')))+1:03d}.json"
    attempt = {"start_unix":time.time(),"registration_sha256":binding,"gpu_uuid":gpu_uuid,
               "occupancy_snapshot":rt.OCCUPANCY_SNAPSHOT}
    rt.atomic_json(attempt_file,attempt,new=True)
    result = {"completed":False,"generations":len(existing),"registration_sha256":binding,
              "occupancy_snapshot":rt.OCCUPANCY_SNAPSHOT}
    try:
        result["gpu_uuid"] = gpu_uuid; set_seed(42)
        adapter = cfg["adapter"]
        if adapter:
            if rt.sha(adapter["training_result"]) != adapter["training_result_sha256"]:
                raise ValueError("selected training result changed")
            rt.verify_files(adapter["path"],adapter["files"])
        model,runtime = rt.load_runtime(cfg["identity"],cfg["targets"],adapter["path"] if adapter else None)
        tokenizer = AutoTokenizer.from_pretrained(cfg["identity"]["base"],local_files_only=True)
        model.eval(); result["runtime"] = runtime
        with (nullcontext() if adapter else model.disable_adapter()), torch.inference_mode():
            for row in prepared:
                directory = run/"rows"/f"{row['ordinal']:05d}"
                if (directory/"result.json").exists():
                    continue
                if time.monotonic() >= deadline:
                    raise TimeoutError("generation deadline including load")
                directory.mkdir(parents=True,exist_ok=resume)
                start_number = len(list(directory.glob("started-*.json")))+1
                rt.atomic_json(directory/f"started-{start_number:03d}.json",
                               {"registration_sha256":binding,"row":row},new=True)
                tick = time.monotonic(); torch.cuda.reset_peak_memory_stats()
                if retention and row["task"] == "general_mc":
                    from scripts.gemma4_retention import choice_likelihoods
                    prediction = choice_likelihoods(model,row,deadline)
                else:
                    prediction = rt.generate(model,tokenizer,row["input_ids"],row.get("max_new_tokens",512),deadline)
                torch.cuda.synchronize()
                prediction = {**row,**prediction,"registration_sha256":binding,
                    "seconds":time.monotonic()-tick,"peak_allocated_bytes":torch.cuda.max_memory_allocated(),
                    "peak_reserved_bytes":torch.cuda.max_memory_reserved()}
                rt.atomic_json(directory/"result.json",prediction,new=True)
                rt.atomic_json(directory/"receipt.json",{"sha256":rt.sha(directory/"result.json")},new=True)
                result["generations"] += 1
                rt.atomic_json(run/"heartbeat.json",{"unit":"item","id":row["id"],
                    "ordinal":row["ordinal"],"completed":result["generations"],"timestamp_unix":time.time()})
                rt.atomic_json(run/"result.json",result)
                print(json.dumps({"done":result["generations"],"total":len(prepared)}),flush=True)
                if prediction.get("resource_cap_stop"):
                    raise TimeoutError("generation interrupted by cap; output preserved")
        predictions = receipt_rows(run,prepared,binding)
        if len(predictions) != len(prepared):
            raise ValueError("incomplete row receipts")
        # Export legacy scorer-compatible JSONL only after all immutable rows exist.
        output = run/"predictions.jsonl"
        encoded = "".join(json.dumps(row,ensure_ascii=False)+"\n" for row in predictions)
        if output.exists():
            if output.read_text() != encoded:
                raise ValueError("immutable prediction export differs")
        else:
            import os
            staging = output.with_name(output.name+f".tmp-{time.time_ns()}")
            with staging.open("x") as stream:
                stream.write(encoded); stream.flush(); os.fsync(stream.fileno())
            os.link(staging,output); staging.unlink()
        result["predictions_sha256"] = rt.sha(output)
        if not retention:
            scores = {}
            for template in cfg["templates"]:
                scores[template] = score_qasper(rt.read(run/"data.json"),
                    {r["id"]:r["output"] for r in predictions if r["template"] == template})
            rt.publish_or_match(run/"scores.json",scores)
        result["completed"] = True
    except BaseException as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        attempt["error_type"] = type(exc).__name__
        raise
    finally:
        attempt["end_unix"] = time.time(); attempt["completed"] = result["completed"]
        rt.atomic_json(attempt_file,attempt)
        result["elapsed_seconds"] = elapsed_attempts(run); result["gpu_hours"] = result["elapsed_seconds"]/3600
        rt.atomic_json(run/"result.json",result)
    return result


def select(run, expected):
    run = Path(run).resolve(); cfg = rt.validate(run,expected)
    status = rt.read(run/"result.json")
    if cfg["purpose"] != "development" or not status["completed"] or rt.sha(run/"predictions.jsonl") != status["predictions_sha256"]:
        raise ValueError("complete bound development predictions required")
    predictions = rt.rows(run/"predictions.jsonl"); scores = {}
    for template in PROMPTS:
        subset = [r for r in predictions if r["template"] == template]
        if len(subset) != 35 or len({r["id"] for r in subset}) != 35:
            raise ValueError("exact development arm IDs required")
        scores[template] = score_qasper(rt.read(run/"data.json"),{r["id"]:r["output"] for r in subset})["strict"]["groups"]["primary"]["answer_f1"]
    result = {"chosen":max(PROMPTS,key=lambda t:scores[t]),"strict_primary_answer_f1":scores,
        "final_test_used":False,"registration_sha256":expected,"predictions_sha256":status["predictions_sha256"],
        "rule":"maximum strict primary Answer F1; ties base,verify,schema"}
    rt.atomic_json(run/"prompt_selection.json",result,new=True)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__); sub = p.add_subparsers(dest="command",required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--base",default=None)
    prep.add_argument("--output",required=True); prep.add_argument("--gpu",choices=["0","1"],default="1")
    prep.add_argument("--purpose",choices=["development","final"],required=True)
    prep.add_argument("--template",choices=list(PROMPTS),default="base")
    prep.add_argument("--selection"); prep.add_argument("--training"); prep.add_argument("--analysis-registration")
    prep.add_argument("--coverage",required=True)
    prep.add_argument("--max-seconds",type=int,default=None)
    cov = sub.add_parser("coverage"); cov.add_argument("--base"); cov.add_argument("--output",required=True)
    for command in ("execute","select"):
        parser = sub.add_parser(command); parser.add_argument("--run",required=True)
        parser.add_argument("--registration-sha256",required=True)
        if command == "execute":
            parser.add_argument("--resume",action="store_true")
    args = p.parse_args()
    if args.command in ("prepare","coverage"):
        args.base = args.base or (rt.PREFLIGHT/"snapshot_path.txt").read_text().strip()
    result = coverage(args) if args.command == "coverage" else (prepare(args) if args.command == "prepare" else (select(args.run,args.registration_sha256)
             if args.command == "select" else run_generations(args))
             )
    print(json.dumps(result,indent=2))


if __name__ == "__main__":
    main()
