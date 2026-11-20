"""Frozen-method Gemma QASPER analysis; no generation or checkpoint reselection.

The original Qwen metric/gate and retention implementations are loaded only
after their source hashes match. Gemma-specific validation handles its native
template and dual end-token set. Final run paths are fixed prospectively; a
separate completion seal binds actual file hashes once those files exist.
"""
import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
import types

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
METHODS={
    "scripts/qwen38_science_score.py":"c9c3b6b03efffda64ae42ce246409d864fbaca1304759495b50da0788d383f5a",
    "scripts/qwen38_qasper_analysis.py":"97518b4a2b20523d566e4a0f63d7f3a584a3f80c32cdde447efcf2aa43cc97d1",
    "scripts/qwen38_retention.py":"af00adeba07800716ea788c202cb1dc87f0445f1a144e7b3bf2b107d50cd1241",
    "scripts/qwen38_eval_science.py":"9051b724975a71288da4e0bd0d76b5f6294a67e5f4be0d2bababf2f6db93d1e3",
    "evaluation/gsm8k.py":"badf9084ba0aa20f933cc1334871b68508c902bc42e3a42924a134785bb51fd0",
}
REVISION="842da3794eaa0b77d5f08bae87a17459d91ff475"


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b""):h.update(chunk)
    return h.hexdigest()


def read(path):
    def unique(pairs):
        result={}
        for k,v in pairs:
            if k in result:raise ValueError("duplicate JSON key")
            result[k]=v
        return result
    return json.loads(Path(path).read_text(),object_pairs_hook=unique,
        parse_constant=lambda v:(_ for _ in ()).throw(ValueError("nonfinite JSON constant")))


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().split("\n") if line.strip()]


def methods(root=ROOT):
    result={}
    for name,expected in METHODS.items():
        file=Path(root)/name;source=file.read_bytes()
        if hashlib.sha256(source).hexdigest()!=expected:raise ValueError("frozen method source changed: "+name)
        if name in ("scripts/qwen38_eval_science.py","evaluation/gsm8k.py"):continue
        module=types.ModuleType("gemma_frozen_"+Path(name).stem)
        module.__file__=str(file)
        exec(compile(source,str(file),"exec"),module.__dict__)
        result[Path(name).stem]=module
    return result


def indexed(values,label):
    if not isinstance(values,list) or not values:raise ValueError(label+": nonempty records required")
    out={}
    for row in values:
        key=row.get("id")
        if not isinstance(key,str) or not key or key in out:raise ValueError(label+": unique string IDs required")
        out[key]=row
    return out


def validate_generation(records, predictions, tokenizer, prompt_text, registration_sha):
    """Validate raw trajectory boundaries before scoring decoded model text."""
    original=indexed(records,"inputs");generated=indexed(predictions,"predictions")
    if set(original)!=set(generated):raise ValueError("missing or extra generation IDs")
    result={}
    for key,row in original.items():
        value=generated[key]
        rendered=tokenizer.apply_chat_template([{"role":"user","content":prompt_text+row["user"]}],
            tokenize=False,add_generation_prompt=True,enable_thinking=False)
        expected=tokenizer(rendered,add_special_tokens=False)["input_ids"]
        if value.get("paper_id")!=row["paper_id"] or value.get("rendered_prompt")!=rendered or value.get("input_ids")!=expected:
            raise ValueError("generation paper, prompt or token IDs differ")
        if len(expected)+512>8192:raise ValueError("full prompt exceeds registered envelope")
        ids=value.get("generated_ids")
        if not isinstance(ids,list) or not 1<=len(ids)<=512 or any(type(i)is not int or i<0 for i in ids):
            raise ValueError("invalid generated IDs/count")
        stop=[i for i,token in enumerate(ids) if token in (1,106)]
        if stop and stop!=[len(ids)-1]:raise ValueError("tokens after native stop token")
        ended=bool(stop)
        if type(value.get("ended_with_eos"))is not bool or value["ended_with_eos"]!=ended:
            raise ValueError("native turn/EOS flag differs")
        if type(value.get("hit_token_limit"))is not bool or value["hit_token_limit"]!=(len(ids)==512 and not ended):
            raise ValueError("token-limit flag differs")
        if not ended and len(ids)!=512:raise ValueError("unexplained short generation")
        if value.get("resource_cap_stop") not in (None,False):raise ValueError("resource-stopped row is incomplete")
        if value.get("registration_sha256")!=registration_sha:raise ValueError("prediction registration binding differs")
        # Keep stray thought/channel markers visible to the unchanged strict
        # parser; remove only one terminal native stop token, never model prose.
        text=tokenizer.decode(ids[:-1] if ended else ids,skip_special_tokens=False)
        if value.get("output")!=text:raise ValueError("decoded trajectory differs from scored text")
        result[key]=text
    return result


def validate_development_profile(cfg):
    fixed={"kind":"evaluation","purpose":"development","task":"qasper","adapter":None,
        "templates":["base","verify","schema"],"max_new_tokens":512,"max_context":8192,
        "eos_token_id":[1,106],"autocast":True,"batch_size":1,"cached":True,"do_sample":False,"seed":42}
    if any(cfg.get(k)!=v for k,v in fixed.items()):raise ValueError("prompt-development profile differs")


def validate_selection(selection, data, development_predictions, binding, prompts, method):
    expected=indexed(data,"prompt-development inputs")
    if len(expected)!=35:raise ValueError("35 frozen development questions required")
    if len(development_predictions)!=105 or {r.get("template") for r in development_predictions}!={"base","verify","schema"}:
        raise ValueError("exactly 105 development predictions across three prompts required")
    if selection.get("final_test_used")is not False or selection.get("registration_sha256")!=binding:
        raise ValueError("prompt selection is not bound to development-only registration")
    scores={}
    for template in ("base","verify","schema"):
        predictions={r["id"]:r["output"] for r in development_predictions if r["template"]==template}
        if set(predictions)!=set(expected) or sum(r["template"]==template for r in development_predictions)!=35:
            raise ValueError("complete unique development predictions for all three prompts required")
        scores[template]=method.score_qasper(data,predictions)["strict"]["groups"]["primary"]["answer_f1"]
    if selection.get("strict_primary_answer_f1")!=scores:raise ValueError("prompt-selection scores differ from recomputation")
    chosen=max(("base","verify","schema"),key=lambda k:scores[k])
    if selection.get("chosen")!=chosen:raise ValueError("prompt selection violates registered tie order")
    return chosen


def task_metrics(final,predictions,prompt_alias=False):
    helpers=methods();metric=helpers["qwen38_science_score"];gate=helpers["qwen38_qasper_analysis"]
    ids=indexed(final,"final input")
    counts={"records":len(final),"primary":sum(r["primary"] for r in final),
            "evidence_strict_primary":sum(r["evidence_strict_primary"] for r in final)}
    if counts!=gate.EXPECTED_COUNTS:raise ValueError("frozen 403/309/267 denominators differ")
    if set(predictions)!={"base","prompt","adapter"} or any(set(p)!=set(ids) for p in predictions.values()):
        raise ValueError("all three complete fixed final arms required")
    if prompt_alias and predictions["prompt"]!=predictions["base"]:raise ValueError("aliased prompt differs from base")
    reports={k:metric.score_qasper(final,v) for k,v in predictions.items()}
    analysis=gate.evaluate_metrics(final,reports,metric,
        {"success_gates":gate.EXPECTED_GATES,"bootstrap":gate.EXPECTED_BOOTSTRAP},prompt_alias)
    return reports,analysis


def retention_metrics(baseline,candidate):
    return methods()["qwen38_retention"].compare_scores(baseline,candidate,
        expected_counts={"general_mc":128,"gsm8k":128,"ifeval":32})


def outcome(analysis,retention=None):
    if type(analysis.get("science_gates_pass"))is not bool:raise ValueError("explicit science gate result required")
    if retention is None:
        return {"status":"science_complete_retention_pending","recommended_adapter":None,
                "complete":False,"science_gates_pass":analysis["science_gates_pass"]}
    if type(retention.get("retention_pass"))is not bool:raise ValueError("explicit retention result required")
    passed=analysis["science_gates_pass"] and retention["retention_pass"]
    return {"status":"completed","complete":True,"science_gates_pass":analysis["science_gates_pass"],
        "retention_pass":retention["retention_pass"],"recommended_adapter":passed,
        "classification":"passed_registered_task_and_retention_gates" if passed else "registered_quality_gates_failed"}


def local(value):
    path=Path(value)
    return (ROOT/path if not path.is_absolute() else path).resolve()


def checked_record(record):
    path=local(record["path"])
    if sha(path)!=record["sha256"]:raise ValueError("bound record changed: "+record["path"])
    return path


def validate_study(path):
    study=read(path);helper=methods()["qwen38_qasper_analysis"]
    if study.get("id")!="gemma4-qasper-01" or study.get("model_identity",{}).get("revision")!=REVISION:
        raise ValueError("Gemma study/model identity differs")
    if study["gates"]!=helper.EXPECTED_GATES or study["bootstrap"]!=helper.EXPECTED_BOOTSTRAP:
        raise ValueError("frozen gates or bootstrap settings differ")
    for name,digest in study["sources"].items():
        if sha(ROOT/name)!=digest:raise ValueError("registered study source changed: "+name)
    checked_record(study["plan"])
    for spec in study["data"].values():checked_record(spec)
    for spec in study["job_registrations"].values():checked_record(spec)
    for spec in study["prelaunch_mechanics"].values():checked_record(spec)
    for name,digest in METHODS.items():
        if study["sources"].get(name)!=digest:raise ValueError("mandatory method binding missing")
    scorer=study["retention_scorer"]
    if sha(scorer["python"])!=scorer["python_sha256"]:raise ValueError("retention interpreter changed")
    for name,digest in scorer["assets"]["files"].items():
        if sha(name)!=digest:raise ValueError("registered IFEval asset changed")
    study["_registration_sha256"]=sha(path)
    return study


def load_job(study,role,require_complete=True):
    run=local(study["prospective_runs"][role]);registration=run/"registration.json"
    cfg=read(registration);binding=sha(registration)
    if role in study["job_registrations"] and binding!=study["job_registrations"][role]["sha256"]:
        raise ValueError("initial job registration changed: "+role)
    if role not in study["job_registrations"]:
        seal=cfg.get("final_analysis_registration",{})
        if seal.get("sha256")!=study["_registration_sha256"] or sha(local(seal["path"]))!=seal["sha256"]:
            raise ValueError("job lacks prospective whole-study binding: "+role)
    if cfg["identity"]!=study["model_identity"] or cfg["versions"]!=study["versions"]:
        raise ValueError("job model/runtime identity differs: "+role)
    for name,item in cfg["source_files"].items():
        expected=study["sources"].get(name)
        if name.endswith("__init__.py") and item["sha256"]==hashlib.sha256(b"").hexdigest():expected=item["sha256"]
        if expected!=item["sha256"] or sha(run/"source"/name)!=expected:
            raise ValueError("job source differs from prospective study: "+name)
    for name,digest in cfg["inputs"].items():
        if sha(run/name)!=digest:raise ValueError("prepared job inputs changed")
    result=None
    if require_complete:
        result=read(run/"result.json")
        if result.get("completed")is not True or result.get("registration_sha256")!=binding or result.get("error"):
            raise ValueError("incomplete/failed job: "+role)
        if role!="training" and sha(run/"predictions.jsonl")!=result.get("predictions_sha256"):
            raise ValueError("generation output hash differs")
    return run,cfg,result,binding


def validate_startup(mechanics,binding):
    fixed={"registration_sha256":binding,"optimizer_updates":0,"outer_bf16_autocast":True,
        "restored_adapter_optimizer_scheduler_rng_buffers":True,"gradient_count":820,
        "gradients_finite":True,"gradients_cleared":True,"state_restored":True,"buffers_restored":True}
    if any(mechanics.get(k)!=v for k,v in fixed.items()):raise ValueError("startup mechanics incomplete or failed")
    parity=mechanics.get("short_compact_full_ce",{})
    expected={"passed":True,"absolute_tolerance":1e-4,"gradient_atol":1e-4,"gradient_rtol":1e-4,
        "gradient_count":820,"native_gradient_equivalence_claim":False,
        "rng_restored":True,"buffers_restored":True,"model_mode_restored":True,"gradients_cleared":True,"optimizer_updates":0}
    if any(parity.get(k)!=v for k,v in expected.items()):raise ValueError("startup arithmetic validation failed")
    g1,g2=parity["g1"],parity["g2"]
    if g1.get("passed")is not True or g2.get("passed")is not True:
        raise ValueError("startup G1/G2 failed")
    if g2.get("gradients_allclose")is not True or g2.get("unsupervised_gradients_exact_zero")is not True:
        raise ValueError("startup shared-logit gradients failed")
    for value in (g1["max_absolute_logit_difference"],g1["absolute_ce_difference"],g2["absolute_ce_difference"]):
        if not math.isfinite(value) or not 0<=value<=1e-4:raise ValueError("startup G1/G2 exceeds tolerance")
    if parity["diagnostics"].get("gating")is not False:
        raise ValueError("native BF16 gradients are descriptive diagnostics only")


def validate_training(study):
    run,cfg,result,binding=load_job(study,"training")
    fixed={"dropout":.05,"autocast":True,"rank":16,"alpha":32,"seed":42,
        "learning_rate":5e-5,"weight_decay":.01,"gradient_accumulation":8,"epochs":1,
        "total_steps":61,"checkpoint_steps":[21,41,61],"batch_size":1,"max_length":8192,
        "warmup_steps":4,"max_grad_norm":1.0,"test_data_loaded_by_trainer":False}
    if any(cfg.get(k)!=v for k,v in fixed.items()) or result.get("steps")!=61:
        raise ValueError("training recipe/schedule differs")
    selection=read(checked_record(study["data"]["qwen_training_selection"]))
    if cfg["qwen_selection_sha256"]!=study["data"]["qwen_training_selection"]["sha256"]:
        raise ValueError("training selection protocol differs")
    for split,key,count in (("train","train",482),("validation","development",101)):
        provenance=cfg["data_provenance"][split]
        values=read(run/(split+".json"))
        if (provenance["sha256"]!=study["data"][key]["sha256"] or
            provenance["ids"]!=selection["selected_ids"][split] or len(values)!=count or
            [r["id"] for r in values]!=provenance["ids"]):
            raise ValueError("training selected inputs differ")
    mechanics=read(run/"longest_row_mechanics.json")
    validate_startup(mechanics,binding)
    checkpoints=result.get("checkpoints")
    if not isinstance(checkpoints,list) or [v["step"] for v in checkpoints]!=[21,41,61]:
        raise ValueError("all three registered checkpoints required")
    for c in checkpoints:
        dev=read(run/f"development-{c['step']:05d}.json")
        if dev["registration_sha256"]!=binding or dev["count"]!=101 or not math.isfinite(dev["nll"]):
            raise ValueError("development checkpoint metrics invalid")
        if any(c.get(k)!=dev[k] for k in ("step","nll","count","loss_tokens","registration_sha256")):
            raise ValueError("checkpoint selection record differs from saved development result")
    selected=min(checkpoints,key=lambda c:(c["nll"],c["step"]))
    if selected!=result["selected"]:raise ValueError("checkpoint selection contradicts fixed development rule")
    adapter=local(selected["path"])
    if adapter.parent.parent!=run:raise ValueError("selected checkpoint escapes training run")
    for name,item in result["selected_files"].items():
        if sha(adapter/name)!=item["sha256"]:raise ValueError("selected checkpoint changed")
    return run,cfg,result,binding


def validate_eval(study,role,records,template,tokenizer,training=None,selection_sha=None):
    run,cfg,result,binding=load_job(study,role)
    expected={"kind":"evaluation","task":"qasper","purpose":"final","templates":[template],
        "max_new_tokens":512,"max_context":8192,"batch_size":1,"eos_token_id":[1,106],
        "autocast":True,"cached":True,"seed":42,"do_sample":False}
    if any(cfg.get(k)!=v for k,v in expected.items()):raise ValueError("final-generation settings differ")
    if cfg["input_sha256"]!=study["data"]["final"]["sha256"] or read(run/"data.json")!=records:
        raise ValueError("final data differ")
    if result.get("generations")!=403:raise ValueError("all 403 final generations required")
    analysis=cfg.get("final_analysis_registration")
    if not analysis or sha(local(analysis["path"]))!=analysis["sha256"] or analysis["sha256"]!=study["_registration_sha256"]:
        raise ValueError("final job lacks the prospective analysis seal")
    if template!="base" and cfg.get("selection_sha256")!=selection_sha:
        raise ValueError("prompted arm selection binding differs")
    if training:
        train_run,_,train_result,train_sha=training;adapter=cfg.get("adapter")
        if not adapter or local(adapter["path"])!=local(train_result["selected"]["path"]):
            raise ValueError("adapter final arm is not the selected checkpoint")
        if adapter["training_result_sha256"]!=sha(train_run/"result.json") or adapter["training_registration_sha256"]!=train_sha:
            raise ValueError("adapter training provenance differs")
        if adapter["files"]!=train_result["selected_files"]:raise ValueError("adapter file manifest differs")
    elif cfg.get("adapter")is not None:raise ValueError("baseline arm must disable adapters")
    helpers=methods();prompts=helpers["qwen38_qasper_analysis"].snapshot_prompts(ROOT/"scripts/qwen38_eval_science.py")
    predictions=rows(run/"predictions.jsonl")
    output=validate_generation(records,predictions,tokenizer,prompts[template],binding)
    return output,{"run":run.relative_to(ROOT).as_posix(),"registration_sha256":binding,
        "result_sha256":sha(run/"result.json"),"predictions_sha256":sha(run/"predictions.jsonl")}


def validate_retention(study,role,training=None):
    run,cfg,result,binding=load_job(study,role)
    expected={"kind":"retention","autocast":True,"batch_size":1,"mc_batch_size":1,
        "eos_token_id":[1,106],"seed":42,"generation_caps":{"gsm8k":512,"ifeval":1024},
        "choice_ids":[562,603,565,622],"mc_bos_token_id":2,
        "mc":"sum full-answer conditional log probabilities; raw concept prompt"}
    if any(cfg.get(k)!=v for k,v in expected.items()):raise ValueError("retention profile differs")
    scorer=study["retention_scorer"]
    if (cfg["scorer_python"]!=scorer["python"] or cfg["scorer_python_sha256"]!=scorer["python_sha256"] or
        cfg["scorer_packages"]!=scorer["packages"] or cfg["ifeval_assets"]!=scorer["assets"]):
        raise ValueError("retention scorer binding differs")
    if role=="retention_base" and cfg.get("adapter")is not None:raise ValueError("retention baseline has adapter")
    if training and (not cfg.get("adapter") or cfg["adapter"]["files"]!=training[2]["selected_files"]):
        raise ValueError("retention adapter differs from selected checkpoint")
    helper=methods()["qwen38_retention"]
    fixture=read(run/"fixture.json");helper.validate_fixture(fixture)
    if cfg["fixture_sha256"]!=study["data"]["retention_fixture"]["sha256"]:
        raise ValueError("retention fixture identity differs")
    predicted=rows(run/"predictions.jsonl")
    if result.get("generations")!=288:raise ValueError("complete 288-item retention run required")
    def strict(data,values):
        scored=helper.run_ifeval(data,values,Path(cfg["scorer_python"]).absolute())
        if scored["packages"]!=scorer["packages"]:raise ValueError("retention scorer packages changed")
        return scored["scores"]
    computed=helper.score_predictions(fixture,predicted,strict)
    if computed!=read(run/"scores.json"):raise ValueError("retention scores differ from independent recomputation")
    shared={"fixture_sha256":cfg["fixture_sha256"],"identity":cfg["identity"],"versions":cfg["versions"],
            "autocast":cfg["autocast"],"generation_caps":cfg["generation_caps"]}
    comparison=hashlib.sha256(json.dumps(shared,sort_keys=True,separators=(",",":")).encode()).hexdigest()
    return {"arm":"base" if role=="retention_base" else "adapter","comparison_sha256":comparison,"tasks":computed}


def analyze(registration,output,science_only=False):
    registration=local(registration);study=validate_study(registration)
    study["_registration_sha256"]=sha(registration)
    training=validate_training(study)
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(study["model_identity"]["base"],local_files_only=True)
    dev_run,dev_cfg,dev_result,dev_binding=load_job(study,"prompt_development")
    validate_development_profile(dev_cfg)
    dev_data=read(dev_run/"data.json");dev_rows=rows(dev_run/"predictions.jsonl")
    if dev_cfg["input_sha256"]!=study["data"]["prompt_development"]["sha256"] or dev_data!=rows(checked_record(study["data"]["prompt_development"])):
        raise ValueError("prompt-selection inputs differ from the frozen 35 records")
    helper=methods();metric=helper["qwen38_science_score"]
    prompts=helper["qwen38_qasper_analysis"].snapshot_prompts(ROOT/"scripts/qwen38_eval_science.py")
    if dev_result.get("generations")!=105:raise ValueError("complete 105-row prompt selection required")
    for template in ("base","verify","schema"):
        validate_generation(dev_data,[r for r in dev_rows if r["template"]==template],tokenizer,prompts[template],dev_binding)
    selection=read(dev_run/"prompt_selection.json");selection_sha=sha(dev_run/"prompt_selection.json")
    chosen=validate_selection(selection,dev_data,dev_rows,dev_binding,prompts,metric)
    final=rows(checked_record(study["data"]["final"]))
    predictions={};provenance={}
    predictions["base"],provenance["base"]=validate_eval(study,"final_base",final,"base",tokenizer)
    if chosen=="base":predictions["prompt"]=predictions["base"];provenance["prompt"]={"alias_of":"base"}
    else:predictions["prompt"],provenance["prompt"]=validate_eval(study,"final_prompt",final,chosen,tokenizer,selection_sha=selection_sha)
    predictions["adapter"],provenance["adapter"]=validate_eval(study,"final_adapter",final,"base",tokenizer,training)
    reports,analysis=task_metrics(final,predictions,chosen=="base")
    retention=None
    if not science_only:
        base=validate_retention(study,"retention_base")
        adapter=validate_retention(study,"retention_adapter",training)
        retention=retention_metrics(base,adapter)
    decision=outcome(analysis,retention)
    analysis={**analysis,"retention":{"status":"completed","comparison":retention} if retention is not None else analysis["retention"],
              "recommended_adapter":decision["recommended_adapter"],
              "overall_status":decision.get("classification",decision["status"])}
    report={"model":"Gemma 4 31B IT","model_revision":REVISION,
        "scope":"Reused cross-family project evaluation; custom text-only QASPER, not a fresh project-heldout or official benchmark result",
        **decision,"registration_sha256":sha(registration),"analysis_source_sha256":sha(__file__),
        "frozen_method_sha256":METHODS,"prompt_selection":selection,"aliases":{"prompt":"base"} if chosen=="base" else {},
        "training":{"steps":61,"selected_step":training[2]["selected"]["step"],"development_checkpoints":training[2]["checkpoints"],
                    "training_result_sha256":sha(training[0]/"result.json")},
        "arms":{name:{mode:report[mode]["groups"] for mode in ("strict","lenient")} for name,report in reports.items()},
        "analysis":analysis,"retention":retention,"generation_provenance":provenance,
        "limitations":["Public papers may be in pretraining","Single configuration/seed; no quality-driven reruns",
            "Retention gates are fixed point-estimate loss limits, not noninferiority","Strict parser gates; lenient scores do not rescue a failed gate"]}
    with local(output).open("x") as stream:stream.write(json.dumps(report,indent=2,allow_nan=False)+"\n")
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration",default="results/protocols/gemma4-qasper-01.json")
    parser.add_argument("--output",required=True)
    parser.add_argument("--science-only",action="store_true")
    args=parser.parse_args();report=analyze(args.registration,args.output,args.science_only)
    print(json.dumps({k:report[k] for k in ("status","complete","recommended_adapter")},indent=2))


if __name__=="__main__":main()
