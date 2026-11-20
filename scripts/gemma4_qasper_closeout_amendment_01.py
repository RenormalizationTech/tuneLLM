"""Registered reporting-only recovery for the Gemma QASPER MC-label mismatch.

Preserves R0, its failed closeout, all job outputs and all original source.
No model forward, generation, checkpoint selection or gate change is allowed.
"""
import argparse
import ast
import datetime
import json
from pathlib import Path

from scripts import gemma4_qasper_analysis as original

ROOT=original.ROOT
RUN=ROOT/"outputs/gemma4-qasper-study-01"
R0=ROOT/"results/protocols/gemma4-qasper-01.json"
REG=ROOT/"results/protocols/gemma4-qasper-analysis-amendment-01.json"
OLD="sum full-answer conditional log probabilities; raw concept prompt"
NEW="sum full-answer conditional log probabilities; native BOS + raw concept prompt"
SOURCES=["scripts/gemma4_qasper_closeout_amendment_01.py",
    "scripts/gemma4_qasper_analysis_amendment_01.py",
    "tests/test_gemma4_qasper_analysis_amendment_01.py"]
REVIEW="results/reviews/gemma4_qasper_analysis_amendment_review_01.md"


def binding(path):
    path=Path(path)
    return {"path":path.relative_to(ROOT).as_posix(),"sha256":original.sha(path)}


def equivalent_sources():
    first=ast.parse((ROOT/"scripts/gemma4_qasper_analysis.py").read_text())
    second=ast.parse((ROOT/"scripts/gemma4_qasper_analysis_amendment_01.py").read_text())
    first.body=first.body[1:];second.body=second.body[1:]
    changes=[]
    class Normalize(ast.NodeTransformer):
        def visit_Constant(self,node):
            if node.value==NEW:
                changes.append(node.lineno)
                return ast.copy_location(ast.Constant(value=OLD),node)
            return node
    second=Normalize().visit(second)
    if len(changes)!=1 or ast.dump(first)!=ast.dump(second):
        raise ValueError("amended analyzer must differ only in its docstring and one MC description literal")


def inputs():
    study=original.validate_study(R0)
    seal=original.read(RUN/"completion-seal.json")
    if seal["registration_sha256"]!=original.sha(R0):raise ValueError("original completion seal differs")
    sealed={}
    for role,files in seal["roles"].items():
        if files.get("alias"):
            if role!="final_prompt" or files!={"alias":"final_base"}:raise ValueError("unexpected alias")
            continue
        run=original.local(study["prospective_runs"][role])
        for name,digest in files.items():
            if original.sha(run/name)!=digest:raise ValueError("sealed job file changed")
            sealed[(run/name).relative_to(ROOT).as_posix()]=digest
        for file in [run/"scores.json",run/"prompt_selection.json",run/"longest_row_mechanics.json",*run.glob("development-*.json")]:
            if file.exists():sealed[file.relative_to(ROOT).as_posix()]=original.sha(file)
    for file in (RUN/"completion-seal.json",RUN/"analysis.log",RUN/"pipeline-status.json",R0):
        sealed[file.relative_to(ROOT).as_posix()]=original.sha(file)
    if (RUN/"prompt_alias.json").exists():sealed[(RUN/"prompt_alias.json").relative_to(ROOT).as_posix()]=original.sha(RUN/"prompt_alias.json")
    return sealed


def prepare():
    equivalent_sources()
    if (ROOT/"results/gemma4_qasper_final_01.json").exists():raise FileExistsError("canonical result already exists")
    if original.read(RUN/"pipeline-status.json")["status"]!="implementation_or_infrastructure_stop":
        raise ValueError("preserved original analysis failure required")
    record={"id":"gemma4-qasper-analysis-amendment-01",
        "registered_at_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "scope":"Reporting-only validation-literal correction; all statistical methods, thresholds, fixtures, model outputs and checkpoint selection unchanged",
        "reason":"R0 analyzer expected pre-BOS method description although both registered retention runs correctly used native BOS token2",
        "original_registration":binding(R0),"sources":{name:original.sha(ROOT/name) for name in SOURCES},
        "review":binding(ROOT/REVIEW),"fixed_inputs":inputs(),
        "before":OLD,"after":NEW,
        "final_scores_inspected_by_PM_before_amendment":False,
        "analysis_output":"outputs/gemma4-qasper-study-01/analysis-amendment-01.json",
        "canonical_output":"results/gemma4_qasper_final_01.json",
        "no_new_GPU_work":True}
    with REG.open("x") as stream:stream.write(json.dumps(record,indent=2,allow_nan=False)+"\n")
    return {"registration":str(REG.relative_to(ROOT)),"sha256":original.sha(REG)}


def execute(expected):
    if original.sha(REG)!=expected:raise ValueError("amendment registration changed")
    record=original.read(REG);equivalent_sources()
    for name,digest in {**record["sources"],**record["fixed_inputs"]}.items():
        if original.sha(ROOT/name)!=digest:raise ValueError("registered correction input/source changed: "+name)
    original.checked_record(record["review"])
    from scripts import gemma4_qasper_analysis_amendment_01 as corrected
    report=corrected.analyze(R0,ROOT/record["analysis_output"])
    if report.get("complete")is not True:raise ValueError("corrected closeout incomplete")
    result={"model":report["model"],"model_revision":report["model_revision"],"scope":report["scope"],
        "status":report["status"],"classification":report["classification"],"completed_science_and_retention":True,
        "recommended_within_registered_scope":report["recommended_adapter"],
        "all_fixed_task_and_retention_gates_pass":report["recommended_adapter"],
        "strict_primary_answer_f1":{k:v["strict"]["primary"]["answer_f1"] for k,v in report["arms"].items()},
        "arms":report["arms"],"analysis":report["analysis"],"retention":report["retention"],
        "prompt_selection":report["prompt_selection"],"aliases":report["aliases"],
        "training":{"steps":61,"selected_step":report["training"]["selected_step"],
            "checkpoints":[{k:v for k,v in r.items() if k!="path"} for r in report["training"]["development_checkpoints"]]},
        "registration_sha256":original.sha(R0),"analysis_sha256":original.sha(ROOT/record["analysis_output"]),
        "completion_seal_sha256":original.sha(RUN/"completion-seal.json"),
        "analysis_source_sha256":report["analysis_source_sha256"],"limitations":report["limitations"],
        "analysis_amendment":{"registration":str(REG.relative_to(ROOT)),"sha256":expected,
            "original_failure_preserved":True,"no_new_GPU_work":True},
        "HF_publication":"not performed; separate decision after results"}
    with (ROOT/record["canonical_output"]).open("x") as stream:stream.write(json.dumps(result,indent=2,allow_nan=False)+"\n")
    receipt={"status":"completed","original_pipeline_status_preserved":True,"amendment_sha256":expected,
        "canonical_sha256":original.sha(ROOT/record["canonical_output"]),
        "completed_at_utc":datetime.datetime.now(datetime.timezone.utc).isoformat()}
    with (RUN/"closeout-amendment-01-status.json").open("x") as stream:stream.write(json.dumps(receipt,indent=2)+"\n")
    return receipt


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("command",choices=["prepare","execute"])
    parser.add_argument("--registration-sha256");args=parser.parse_args()
    print(json.dumps(prepare() if args.command=="prepare" else execute(args.registration_sha256),indent=2))


if __name__=="__main__":main()
