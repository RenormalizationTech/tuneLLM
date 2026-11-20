"""Bind CPU-prepared Gemma QASPER jobs before any task GPU execution."""
import datetime
import hashlib
import json
import subprocess
from pathlib import Path

from scripts import gemma4_qasper_analysis as analysis

ROOT=analysis.ROOT
STUDY=ROOT/"outputs/gemma4-qasper-study-01"
REGISTRATION=ROOT/"results/protocols/gemma4-qasper-01.json"
RUNS={name:(STUDY/sub).relative_to(ROOT).as_posix() for name,sub in {
    "coverage":"coverage","training":"training","prompt_development":"prompt-development",
    "final_base":"final-base","final_prompt":"final-prompt","final_adapter":"final-adapter",
    "retention_base":"retention-base","retention_adapter":"retention-adapter"}.items()}


def binding(path):
    path=Path(path)
    return {"path":path.relative_to(ROOT).as_posix(),"sha256":analysis.sha(path)}


def main():
    if REGISTRATION.exists():raise FileExistsError("whole-study registration already exists")
    initial={};sources={}
    for role in ("coverage","training","prompt_development","retention_base"):
        run=ROOT/RUNS[role];cfg=analysis.read(run/"registration.json")
        if (run/"attempts").exists() or (run/"result.json").exists():raise ValueError("task GPU work exists before registration")
        initial[role]=binding(run/"registration.json")
        for name,item in cfg["source_files"].items():
            if name.endswith("__init__.py") and item["sha256"]==hashlib.sha256(b"").hexdigest():continue
            if analysis.sha(ROOT/name)!=item["sha256"]:raise ValueError("prepared source differs: "+name)
            if name in sources and sources[name]!=item["sha256"]:raise ValueError("prepared jobs use different source versions")
            sources[name]=item["sha256"]
    for name in list(analysis.METHODS)+["scripts/gemma4_qasper_analysis.py","scripts/register_gemma4_qasper.py",
        "scripts/gemma4_qasper_pipeline.py","scripts/gemma4_qasper_watch.py",
        "scripts/gemma4_qasper_mechanics.py",
        "scripts/gemma4_refresh_qasper_preparation.py",
        "tests/test_gemma4_qasper_analysis.py","tests/test_gemma4_qasper.py"]:
        sources[name]=analysis.sha(ROOT/name)
    review_name="results/reviews/gemma4_qasper_source_review_01.md"
    sources[review_name]=analysis.sha(ROOT/review_name)
    train=analysis.read(ROOT/RUNS["training"]/"registration.json")
    fixed={"dropout":.05,"autocast":True,"rank":16,"alpha":32,"gradient_accumulation":8,
           "learning_rate":5e-5,"weight_decay":.01,"epochs":1,"total_steps":61,"checkpoint_steps":[21,41,61],
           "seed":42,"max_length":8192,"warmup_steps":4,"max_grad_norm":1.0,"batch_size":1}
    if any(train.get(k)!=v for k,v in fixed.items()):raise ValueError("training configuration differs from fixed design")
    for role in ("prompt_development","retention_base"):
        cfg=analysis.read(ROOT/RUNS[role]/"registration.json")
        if cfg["identity"]!=train["identity"] or cfg["versions"]!=train["versions"] or cfg["autocast"]is not True:
            raise ValueError("baseline loading profile differs from training")
        if role=="prompt_development":
            coverage=cfg["coverage_registration"]
            if coverage["sha256"]!=initial["coverage"]["sha256"] or Path(coverage["path"])!=ROOT/RUNS["coverage"]/"registration.json":
                raise ValueError("prompt-development must use the registered coverage role")
    original=analysis.read(ROOT/"results/protocols/qwen38-qasper-evaluation-01.json")
    prepared=ROOT/"outputs/qwen38-science-01/prepared-qasper-tokenizer-01"
    data={name:binding(path) for name,path in {
        "manifest":prepared/"manifest.json","train":prepared/"train.jsonl",
        "development":prepared/"validation.jsonl","final":prepared/"test.jsonl",
        "prompt_development":ROOT/"outputs/qwen38-qasper-evaluation-01/prompt_development.jsonl",
        "qwen_training_selection":ROOT/"results/protocols/qwen38-qasper-02.json",
        "qwen_evaluation_protocol":ROOT/"results/protocols/qwen38-qasper-evaluation-01.json",
        "retention_fixture":ROOT/"data/benchmark-screen-01/test.json"}.items()}
    if data["final"]["sha256"]!=original["final_input_sha256"] or data["prompt_development"]["sha256"]!=original["prompt_selection_input_sha256"]:
        raise ValueError("frozen evaluation inputs differ")
    helper=analysis.methods()["qwen38_qasper_analysis"]
    mechanics_dir=ROOT/"outputs/gemma4-qasper-mechanics-03"
    mechanics=analysis.read(mechanics_dir/"result.json")
    manifest=analysis.read(mechanics_dir/"manifest.json")
    if mechanics.get("status")!="passed" or mechanics.get("manifest_sha256")!=analysis.sha(mechanics_dir/"manifest.json"):
        raise ValueError("no-update actual-profile mechanics did not pass")
    analysis.validate_startup(mechanics,analysis.sha(mechanics_dir/"manifest.json"))
    for name,digest in manifest["sources"].items():
        if analysis.sha(ROOT/name)!=digest:raise ValueError("mechanics source differs from study source")
    if manifest["inputs"]["train.json"]!=analysis.sha(ROOT/RUNS["training"]/"train.json"):
        raise ValueError("mechanics and formal training trajectories differ")
    retention=analysis.read(ROOT/RUNS["retention_base"]/"registration.json")
    if retention["scorer_python"]!=str(ROOT/".venv/bin/python"):
        raise ValueError("preserved project scorer virtual-environment path required")
    scorer={"python":retention["scorer_python"],"python_sha256":retention["scorer_python_sha256"],
        "packages":retention["scorer_packages"],"assets":retention["ifeval_assets"],
        "python_version":subprocess.check_output([retention["scorer_python"],"-c","import platform;print(platform.python_version())"],text=True).strip()}
    record={"schema_version":1,"id":"gemma4-qasper-01",
        "registered_at_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "scope":"Gemma QASPER only; reused project evaluation; no PRL or automatic HF upload",
        "plan":binding(ROOT/"docs/gemma4_qasper_plan.md"),"model_identity":train["identity"],
        "source_review":binding(ROOT/"results/reviews/gemma4_qasper_source_review_01.md"),
        "versions":train["versions"],"data":data,"job_registrations":initial,
        "retention_scorer":scorer,
        "prelaunch_mechanics":{"manifest":binding(mechanics_dir/"manifest.json"),"result":binding(mechanics_dir/"result.json")},
        "preserved_pre_R0_failure":binding(ROOT/"outputs/gemma4-qasper-mechanics-02/result.json"),
        "prospective_runs":RUNS,"sources":sources,"gates":helper.EXPECTED_GATES,
        "bootstrap":helper.EXPECTED_BOOTSTRAP,
        "selection":{"checkpoint":"minimum token-weighted101-development NLL at21/41/61; earliest tie",
            "prompt":"maximum strict primary F1 on35 fixed development questions; base,verify,schema tie order",
            "adapter_prompt":"base","prompt_alias":"alias base if base wins; no duplicate final generation"},
        "generation":{"thinking":False,"cached":True,"batch_size":1,"greedy":True,
            "max_new_tokens":512,"sequence_cap":8192,"eos_token_id":[1,106],"autocast":True},
        "mapping_policy":"Role/path/source/arm/alias rules fixed here; completion hashes are sealed after files exist, before final analysis; no path discovery or quality-based replacement",
        "recovery":"Exact durable optimizer-boundary recovery and validated append-only completed generation rows; deterministic bugs stop and require a prospective amendment",
        "owner_scope":"User authorizes QASPER study; no additional permission gate for registered phases"}
    with REGISTRATION.open("x") as f:f.write(json.dumps(record,indent=2,allow_nan=False)+"\n")
    analysis.validate_study(REGISTRATION)
    print(json.dumps({"registration":REGISTRATION.relative_to(ROOT).as_posix(),"sha256":analysis.sha(REGISTRATION)},indent=2))


if __name__=="__main__":main()
