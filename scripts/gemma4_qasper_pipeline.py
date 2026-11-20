"""Persistent two-GPU registered Gemma QASPER study orchestration.

Run in a persistent terminal, not a short-lived tool cell. Task failures stop
the pipeline and preserve evidence; no implicit retry, retuning or upload.
"""
import argparse
import fcntl
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

from scripts import gemma4_qasper_analysis as analysis

ROOT=analysis.ROOT
REG=ROOT/"results/protocols/gemma4-qasper-01.json"
STUDY=ROOT/"outputs/gemma4-qasper-study-01"
PRINT_LOCK=threading.Lock()


def atomic(path,value):
    path=Path(path);temp=path.with_suffix(path.suffix+".tmp-"+str(threading.get_ident()))
    temp.write_text(json.dumps(value,indent=2,allow_nan=False)+"\n");os.replace(temp,path)


def event(stage,**extra):
    with PRINT_LOCK:
        print(json.dumps({"stage":stage,"timestamp_unix":time.time(),**extra}),flush=True)


def run_command(args,log):
    log=Path(log);log.parent.mkdir(parents=True,exist_ok=True)
    with log.open("a") as stream:
        process=subprocess.Popen([sys.executable,"-u","-B",*args],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
        while True:
            try:
                code=process.wait(timeout=45)
                if code:raise RuntimeError(f"registered subprocess failed ({code}); preserve {log}")
                return
            except subprocess.TimeoutExpired:
                event("process_alive",pid=process.pid,log=log.relative_to(ROOT).as_posix())


def verify(study):
    if analysis.sha(REG)!=study["_sha"]:raise ValueError("whole-study registration changed")
    for path,digest in study["sources"].items():
        if analysis.sha(ROOT/path)!=digest:raise ValueError("registered study source changed: "+path)


def execute_job(study,role,module):
    verify(study);run=ROOT/study["prospective_runs"][role]
    if (run/"result.json").exists():
        if not study.get("_resume"):raise FileExistsError("explicit recovery required; prior execution exists: "+role)
        status=analysis.read(run/"result.json")
        prior_path=STUDY/(role+"-pipeline.json")
        prior=analysis.read(prior_path) if prior_path.exists() else {}
        if status.get("completed")is True:
            analysis.load_job(study,role)
            if prior.get("status")!="completed" or prior.get("result_sha256")!=analysis.sha(run/"result.json"):
                raise ValueError("completed job lacks matching immutable pipeline receipt: "+role)
            event("resume_skip_completed",role=role)
            return
    cfg=analysis.read(run/"registration.json");binding=analysis.sha(run/"registration.json")
    event("start",role=role,gpu=cfg.get("gpu"),registration_sha256=binding)
    atomic(STUDY/(role+"-pipeline.json"),{"status":"running","registration_sha256":binding,"start_unix":time.time()})
    command=["-m",module,"execute","--run",str(run),"--registration-sha256",binding]
    if study.get("_resume") and list((run/"attempts").glob("*.json")):command.append("--resume")
    run_command(command,run/"console.log")
    status=analysis.read(run/"result.json")
    if status.get("completed")is not True:raise ValueError("job exited without complete receipt: "+role)
    atomic(STUDY/(role+"-pipeline.json"),{"status":"completed","registration_sha256":binding,"result_sha256":analysis.sha(run/"result.json"),"end_unix":time.time()})
    event("completed",role=role)


def prepare_eval(study,role,gpu,template="base",training=False):
    verify(study);run=ROOT/study["prospective_runs"][role]
    if study.get("_resume") and (run/"registration.json").exists():
        _,cfg,_,_=analysis.load_job(study,role,False)
        if cfg["templates"]!=[template] or cfg["gpu"]!=str(gpu) or bool(cfg.get("adapter"))!=training:
            raise ValueError("recovery evaluation role/configuration differs")
        return
    args=["-m","scripts.gemma4_eval_qasper","prepare","--output",str(run),"--purpose","final",
        "--template",template,"--gpu",str(gpu),"--coverage",str(ROOT/study["prospective_runs"]["coverage"]),
        "--analysis-registration",str(REG)]
    if template!="base":args += ["--selection",str(ROOT/study["prospective_runs"]["prompt_development"] /"prompt_selection.json")]
    if training:args += ["--training",str(ROOT/study["prospective_runs"]["training"])]
    run_command(args,STUDY/(role+"-prepare.log"))


def score_retention(study,role):
    run=ROOT/study["prospective_runs"][role]
    if study.get("_resume") and (run/"scores.json").exists():
        analysis.validate_retention(study,role,analysis.validate_training(study) if role=="retention_adapter" else None)
        return
    run_command(["-m","scripts.gemma4_retention","score","--run",str(run),
        "--registration-sha256",analysis.sha(run/"registration.json")],run/"scoring.log")


def baseline_lane(study):
    execute_job(study,"prompt_development","scripts.gemma4_eval_qasper")
    dev=ROOT/study["prospective_runs"]["prompt_development"]
    if study.get("_resume") and (dev/"prompt_selection.json").exists():
        metric=analysis.methods()["qwen38_science_score"]
        analysis.validate_selection(analysis.read(dev/"prompt_selection.json"),analysis.read(dev/"data.json"),
            analysis.rows(dev/"predictions.jsonl"),analysis.sha(dev/"registration.json"),{},metric)
    else:
        run_command(["-m","scripts.gemma4_eval_qasper","select","--run",str(dev),
            "--registration-sha256",analysis.sha(dev/"registration.json")],dev/"selection.log")
    prepare_eval(study,"final_base",1)
    execute_job(study,"final_base","scripts.gemma4_eval_qasper")
    chosen=analysis.read(dev/"prompt_selection.json")["chosen"]
    if chosen=="base":
        atomic(STUDY/"prompt_alias.json",{"alias":"base","selection_sha256":analysis.sha(dev/"prompt_selection.json")})
    else:
        prepare_eval(study,"final_prompt",1,chosen)
        execute_job(study,"final_prompt","scripts.gemma4_eval_qasper")
    execute_job(study,"retention_base","scripts.gemma4_retention")
    score_retention(study,"retention_base")


def adapter_retention(study):
    verify(study);run=ROOT/study["prospective_runs"]["retention_adapter"]
    if study.get("_resume") and (run/"registration.json").exists():analysis.load_job(study,"retention_adapter",False)
    else:
        run_command(["-m","scripts.gemma4_retention","prepare","--output",str(run),"--gpu","0", "--training",
            str(ROOT/study["prospective_runs"]["training"]),"--analysis-registration",str(REG)],STUDY/"retention-adapter-prepare.log")
    execute_job(study,"retention_adapter","scripts.gemma4_retention")
    score_retention(study,"retention_adapter")


def adapter_final(study):
    prepare_eval(study,"final_adapter",1,training=True)
    execute_job(study,"final_adapter","scripts.gemma4_eval_qasper")


def closeout(study):
    verify(study)
    # Completion seal is distinct from the prospective role/path/source mapping.
    seal={}
    for role in ("training","prompt_development","final_base","final_prompt","final_adapter","retention_base","retention_adapter"):
        run=ROOT/study["prospective_runs"][role]
        if role=="final_prompt" and (STUDY/"prompt_alias.json").exists():seal[role]={"alias":"final_base"};continue
        seal[role]={name:analysis.sha(run/name) for name in ("registration.json","result.json")}
        if role!="training":seal[role]["predictions.jsonl"]=analysis.sha(run/"predictions.jsonl")
    atomic(STUDY/"completion-seal.json",{"registration_sha256":study["_sha"],"roles":seal,"recorded_at_unix":time.time()})
    run_command(["-m","scripts.gemma4_qasper_analysis","--registration",str(REG),
        "--output",str(STUDY/"analysis.json")],STUDY/"analysis.log")
    report=analysis.read(STUDY/"analysis.json")
    if report["complete"]is not True:raise ValueError("analysis incomplete")
    compact={"model":report["model"],"model_revision":report["model_revision"],"scope":report["scope"],
        "status":report["status"],"classification":report["classification"],"completed_science_and_retention":True,
        "recommended_within_registered_scope":report["recommended_adapter"],
        "all_fixed_task_and_retention_gates_pass":report["recommended_adapter"],
        "strict_primary_answer_f1":{k:v["strict"]["primary"]["answer_f1"] for k,v in report["arms"].items()},
        "arms":report["arms"],"analysis":report["analysis"],"retention":report["retention"],
        "prompt_selection":report["prompt_selection"],"aliases":report["aliases"],
        "training":{"steps":61,"selected_step":report["training"]["selected_step"],
            "checkpoints":[{k:v for k,v in r.items() if k!="path"} for r in report["training"]["development_checkpoints"]]},
        "registration_sha256":study["_sha"],"analysis_sha256":analysis.sha(STUDY/"analysis.json"),
        "completion_seal_sha256":analysis.sha(STUDY/"completion-seal.json"),
        "analysis_source_sha256":report["analysis_source_sha256"],"limitations":report["limitations"],
        "HF_publication":"not performed; separate decision after results"}
    output=ROOT/"results/gemma4_qasper_final_01.json"
    with output.open("x") as f:f.write(json.dumps(compact,indent=2,allow_nan=False)+"\n")
    return compact


def run_pipeline():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--resume",action="store_true");args=parser.parse_args()
    if (STUDY/"pipeline-status.json").exists() and not args.resume:raise FileExistsError("prior pipeline execution exists; explicit --resume required")
    study=analysis.validate_study(REG);study["_sha"]=analysis.sha(REG);study["_resume"]=args.resume
    if args.resume:
        previous=analysis.read(STUDY/"pipeline-status.json")
        if previous["registration_sha256"]!=study["_sha"] or previous["status"]=="completed":raise ValueError("recovery registration differs or already complete")
        history=STUDY/"pipeline-history";history.mkdir(exist_ok=True)
        atomic(history/(str(time.time_ns())+".json"),previous)
    atomic(STUDY/"pipeline-status.json",{"status":"running","pid":os.getpid(),"start_unix":time.time(),"registration_sha256":study["_sha"]})
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            train=pool.submit(execute_job,study,"training","scripts.gemma4_train_qasper")
            baseline=pool.submit(baseline_lane,study)
            train.result();baseline.result()
        with ThreadPoolExecutor(max_workers=2) as pool:
            final=pool.submit(adapter_final,study);retention=pool.submit(adapter_retention,study)
            final.result();retention.result()
        result=closeout(study)
        atomic(STUDY/"pipeline-status.json",{"status":"completed","end_unix":time.time(),
            "recommended_adapter":result["recommended_within_registered_scope"],"registration_sha256":study["_sha"]})
        event("study_complete",recommended=result["recommended_within_registered_scope"])
    except BaseException as error:
        atomic(STUDY/"pipeline-status.json",{"status":"implementation_or_infrastructure_stop",
            "error":f"{type(error).__name__}: {error}","end_unix":time.time(),"registration_sha256":study["_sha"]})
        raise


def main():
    STUDY.mkdir(parents=True,exist_ok=True)
    with (STUDY/"pipeline.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        run_pipeline()


if __name__=="__main__":main()
