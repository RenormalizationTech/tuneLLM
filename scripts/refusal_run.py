"""Staged BF16 positive-refusal pilot. All execution is local and separately gated.

CPU prepare/export/import/select/lock/summary; GPU stages require an approved
GPU0 window, exact committed inputs/source, and a registered fresh judge audit.
Extraction performs prompt forward passes ONLY, never extraction-source answers.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from scripts.common import path, read_json
from scripts.save_nf4 import stream_hash
from scripts.tone_common import steer
from scripts.tone_steering import committed, load_model, gpu_record
from scripts.tone_judge import save
from scripts import refusal_common as common
from scripts import refusal_judge

SOURCES=("scripts/refusal_run.py","scripts/refusal_common.py","scripts/refusal_judge.py","configs/refusal.json",
         "evaluation/refusal/response_rubric.json","scripts/tone_judge.py","scripts/tone_common.py",
         "scripts/tone_steering.py","scripts/tone_fixture.py","scripts/batched_generation.py",
         "scripts/common.py","scripts/save_nf4.py")


def check_prepared(data, report):
    manifest=read_json(data/"manifest.json")
    if report["manifest_sha256"]!=stream_hash(data/"manifest.json") or any(report.get(k)!=v for k,v in manifest.items()):
        raise ValueError("prepared input/report registration mismatch")
    for name,expected in manifest["files"].items():
        if stream_hash(data/name)!=expected:
            raise ValueError("prepared input changed")
    return manifest


def prepare(args):
    run,data=path(args.run),path(args.data)
    if run.exists():
        raise FileExistsError("fresh refusal run required")
    cfg=read_json(args.config)
    if (cfg["gpu"]!="0" or cfg["seed"]!=42 or cfg["batch_size"]!=8 or cfg["dtype"]!="bfloat16"
            or cfg["revision"]!="1cfa9a7208912126459214e8b04321603b3df60c"
            or cfg["primary_layer"]!=17 or cfg["layers"]!=[11,17,23] or cfg["multipliers"]!=[1,2,4]):
        raise ValueError("registered BF16/GPU0/fixed-layer positive-only configuration required")
    report=read_json(args.prepared_report)
    check_prepared(data,report)
    rubric=read_json(cfg["rubric"])
    if set(rubric["primary_refusal_classes"])!=common.PRIMARY:
        raise ValueError("primary numerator must include partial plus full refusals")
    snapshot=path(".hf/hub/models--Qwen--Qwen3-4B/snapshots")/cfg["revision"]
    if not (snapshot/"model.safetensors.index.json").is_file():
        raise ValueError("pinned model must already be cached; downloads prohibited")
    model_hashes={str(p):stream_hash(p) for p in snapshot.iterdir() if p.is_file()}
    run.mkdir(parents=True)
    save(run/"config.json",cfg)
    for name in ("extraction.json","development.json","final.json","screening_labels.json","manifest.json"):
        target=run/"inputs"/name
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(data/name,target)
    for name,source in (("prepared_report.json",path(args.prepared_report)),("rubric.json",path(cfg["rubric"])),("protocol.md",path(cfg["protocol"]))):
        shutil.copyfile(source,run/"inputs"/name)
    for name in SOURCES:
        target=run/"source"/name
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(path(name),target)
    save(run/"manifest.json",{"files":{str(p.relative_to(run)):stream_hash(p) for p in run.rglob("*") if p.is_file()},
        "sources":{n:stream_hash(path(n)) for n in SOURCES},"snapshot":str(snapshot),"external_model_sha256":model_hashes,
        "prepared_report_sha256":stream_hash(path(args.prepared_report)),"prepared_only":True,"no_model_outputs":True})
    print("CPU snapshot ready; root must register/commit manifest, source, rubric and fresh judge audit before GPU stages.")


def verify(run,models=False):
    manifest=read_json(run/"manifest.json")
    for name,expected in manifest["files"].items():
        if stream_hash(run/name)!=expected:
            raise ValueError("frozen refusal input/source changed")
    for name,expected in manifest["sources"].items():
        if stream_hash(path(name))!=expected:
            raise ValueError("active refusal/dependency source changed")
    if models:
        for name,expected in manifest["external_model_sha256"].items():
            if stream_hash(path(name))!=expected:
                raise ValueError("read-only BF16 model changed")
    return read_json(run/"config.json"),manifest


def judge_registration(args,run):
    if not args.judge_registration or not args.judge_audit:
        raise ValueError("committed fresh refusal judge audit registration required")
    committed(args.judge_registration)
    return refusal_judge.verify_audit(args.judge_audit,run/"inputs/rubric.json",read_json(args.judge_registration))


def resource_record(run,cfg,started):
    import importlib.metadata
    record=gpu_record(run,cfg,"bf16",started)
    record["packages"].update({k:importlib.metadata.version(k) for k in ("tokenizers","huggingface-hub","accelerate")})
    return {**record,"model_id":cfg["model_id"],"revision":cfg["revision"],
            "source_sha256":read_json(run/"manifest.json")["sources"]}


def gpu_ready(args,run):
    if not args.execute_gpu:
        raise ValueError("GPU work requires --execute-gpu only AFTER coordinator releases GPU0")
    cfg,manifest=verify(run,models=True)
    if not args.registration:
        raise ValueError("committed exact run registration required")
    committed(args.registration)
    committed(args.prepared_report)
    if stream_hash(path(args.prepared_report))!=manifest["prepared_report_sha256"]:
        raise ValueError("committed prepared report differs from snapshot")
    if read_json(args.registration).get("manifest_sha256")!=stream_hash(run/"manifest.json"):
        raise ValueError("run registration differs")
    for name in SOURCES:
        committed(name)
    judge_registration(args,run)
    uuid=subprocess.run(["nvidia-smi","--id=0","--query-gpu=uuid","--format=csv,noheader"],
                        capture_output=True,text=True,check=True).stdout.strip()
    for phase in ("extraction","preflight","development","final"):
        filename=run/phase/"run.json"
        if filename.exists() and read_json(filename).get("gpu_uuid")!=uuid:
            raise ValueError("physical GPU UUID changed across comparisons")
    return cfg,manifest


def groups(run,split):
    rows=read_json(run/f"inputs/{split}.json")["items"]
    return {label:[r for r in rows if r["label"]==label] for label in ("safe","unsafe")}


def generation_position_ids(attention_mask):
    """Match Transformers generate: real tokens count from zero, padding is1."""
    positions=attention_mask.long().cumsum(-1)-1
    return positions.masked_fill(attention_mask==0,1)


def prompt_activations(model,tokenizer,rows,layers,batch_size=8,device="cuda:0"):
    import torch
    from scripts.batched_generation import render_prompt,left_pad
    if not rows:
        raise ValueError("undefined direction: empty prompt class")
    captured={layer:[] for layer in layers}
    for start in range(0,len(rows),batch_size):
        ids=[tokenizer(render_prompt(tokenizer,{"user":r["instruction"]}),add_special_tokens=False)["input_ids"] for r in rows[start:start+batch_size]]
        inputs={k:torch.tensor(v,dtype=torch.long,device=device) for k,v in left_pad(ids,tokenizer.pad_token_id).items()}
        inputs["position_ids"]=generation_position_ids(inputs["attention_mask"])
        batch={}
        with common.capture_last_prompt(model,layers,inputs["attention_mask"],batch),torch.inference_mode():
            # Deliberately call the base transformer: no LM head/generate path.
            model.model(**inputs,use_cache=False,return_dict=True)
        for layer in layers:
            captured[layer].append(batch[layer])
    return {layer:torch.cat(values) for layer,values in captured.items()}


def extract(args,run):
    cfg,manifest=gpu_ready(args,run)
    out=run/"extraction"
    out.mkdir(exist_ok=False)
    pairs=read_json(run/"inputs/extraction.json")["pairs"]
    if not pairs:
        save(out/"numerical_failure.json",{"reason":"empty prompt classes; no model allocated","completed":False})
        return
    import torch
    started=time.perf_counter()
    model,tokenizer=load_model(cfg,manifest,"bf16")
    activations={c:prompt_activations(model,tokenizer,[p[c] for p in pairs],cfg["layers"],cfg["batch_size"]) for c in ("harmful","benign")}
    directions,diagnostics={},{}
    for layer in cfg["layers"]:
        try:
            directions[str(layer)],diagnostics[str(layer)]=common.prompt_direction(activations["harmful"][layer],activations["benign"][layer])
        except ValueError as exc:
            diagnostics[str(layer)]={"numerical_failure":str(exc)}
    if "17" not in directions:
        save(out/"numerical_failure.json",{"completed":False,"diagnostics":diagnostics})
        return
    dev=groups(run,"development")
    scores=prompt_activations(model,tokenizer,dev["safe"]+dev["unsafe"],cfg["layers"],cfg["batch_size"])
    for layer in cfg["layers"]:
        if str(layer) in directions:
            projections=(scores[layer]@directions[str(layer)]).tolist()
            diagnostics[str(layer)]["development_projection_auc"]=common.projection_auc(projections[:len(dev["safe"])],projections[len(dev["safe"]):])
    torch.save(directions,out/"vectors.pt")
    save(out/"vectors.json",{"sha256":stream_hash(out/"vectors.pt"),"diagnostics":diagnostics,
        "primary_layer":17,"diagnostic_layers_only":[11,23],"answers_generated":False,
        "position":"last real prompt token after upstream nonthinking generation prefix; no system prompt",
        "extraction_input_sha256":stream_hash(run/"inputs/extraction.json")})
    save(out/"run.json",resource_record(run,cfg,started))


def vector(run):
    import torch
    filename=run/"extraction/vectors.pt"
    if stream_hash(filename)!=read_json(run/"extraction/vectors.json")["sha256"]:
        raise ValueError("prompt-derived vectors changed")
    return torch.load(filename,map_location="cpu",weights_only=True)["17"]


def generate(model,tokenizer,rows,cfg,cap,direction=None,multiplier=0,audit=None):
    from scripts.batched_generation import generate_batched
    with steer(model,17,direction,multiplier,audit):
        return generate_batched(model,tokenizer,[{"id":r["id"],"user":r["instruction"]} for r in rows],
                                batch_size=8,max_new_tokens=cap,seed=42,phase="refusal evaluation")


def preflight(args,run):
    cfg,manifest=gpu_ready(args,run)
    direction=vector(run)
    out=run/"preflight"
    out.mkdir(exist_ok=False)
    dev=groups(run,"development")
    started=time.perf_counter()
    model,tokenizer=load_model(cfg,manifest,"bf16")
    initial={label:generate(model,tokenizer,rows,cfg,256) for label,rows in dev.items()}
    for label,value in initial.items():
        save(out/f"initial_{label}.json",value)
    hits=sum(p["hit_token_limit"] for value in initial.values() for p in value["predictions"])
    cap=320 if hits>6 else 256
    baseline=initial if cap==256 else {label:generate(model,tokenizer,rows,cfg,cap) for label,rows in dev.items()}
    for label,value in baseline.items():
        save(out/f"m0_{label}.json",value)
    import torch
    zero_audit,nonzero_audit={},{}
    zero=generate(model,tokenizer,dev["safe"][:8],cfg,cap,torch.zeros_like(direction),1,zero_audit)
    nonzero=generate(model,tokenizer,dev["safe"][:8],cfg,cap,direction,1,nonzero_audit)
    save(out/"zero_probe.json",zero)
    save(out/"nonzero_probe.json",nonzero)
    same=(len(zero["predictions"])==8 and all(a["input_ids"]==b["input_ids"] and a["generated_ids"]==b["generated_ids"]
          for a,b in zip(zero["predictions"],baseline["safe"]["predictions"])))
    from scripts.batched_generation import render_prompt,left_pad
    ids=[tokenizer(render_prompt(tokenizer,{"user":r["instruction"]}),add_special_tokens=False)["input_ids"] for r in dev["safe"][:8]]
    inputs={k:torch.tensor(v,dtype=torch.long,device="cuda:0") for k,v in left_pad(ids,tokenizer.pad_token_id).items()}
    inputs["position_ids"]=generation_position_ids(inputs["attention_mask"])
    with torch.inference_mode():
        plain=model(**inputs,use_cache=False).logits[:,-1].float()
        with steer(model,17,direction,1):
            modified=model(**inputs,use_cache=False).logits[:,-1].float()
    nonzero_audit["max_abs_last_prompt_logit_delta"]=(modified-plain).abs().max().item()
    mechanics=(same and all(a.get("prefill_calls",0)>0 and a.get("decode_calls",0)>0 for a in (zero_audit,nonzero_audit))
               and nonzero_audit["max_abs_hidden_delta"]>0 and nonzero_audit["max_abs_last_prompt_logit_delta"]>0)
    report={"cap":cap,"initial_cap":256,"initial_cap_hits":hits,"sample_ids":{k:[r["id"] for r in v] for k,v in dev.items()},
        "baseline_sha256":{k:stream_hash(out/f"m0_{k}.json") for k in dev},"zero_exact":same,"mechanics_passed":mechanics,
        "zero_vector_multiplier1":zero_audit,"fixed_nonzero_multiplier1":nonzero_audit,
        "probe_scope":"first8 development safe prompts, layer17; no token-change requirement for nonzero",
        "vector_sha256":stream_hash(run/"extraction/vectors.pt"),"manifest_sha256":stream_hash(run/"manifest.json"),
        "projected_remaining_generation_seconds":baseline["safe"]["timing"]["pipeline_wall_seconds"]*(90+168)/30+baseline["unsafe"]["timing"]["pipeline_wall_seconds"]*60/30,
        "budget_scope":"upper projection:90 safe development plus168safe/60unsafe final; excludes loading/judging/extraction; skip228final if null",
        "baseline_reuse":"final-cap defaults reused; safe/unsafe generated separately to preserve comparison batch membership"}
    save(out/"preflight.json",report)
    save(out/"run.json",resource_record(run,cfg,started))
    if not mechanics:
        raise ValueError("actual-hook zero/nonzero mechanics failed; no steered development")


def budget(args,run):
    if not args.budget:
        raise ValueError("committed timing/cap budget required")
    committed(args.budget)
    report=read_json(run/"preflight/preflight.json")
    if read_json(args.budget)!=report or not report["mechanics_passed"]:
        raise ValueError("registered timing/cap report differs or failed")
    if stream_hash(run/"manifest.json")!=report["manifest_sha256"] or stream_hash(run/"extraction/vectors.pt")!=report["vector_sha256"]:
        raise ValueError("preflight model inputs or direction changed")
    for label,h in report["baseline_sha256"].items():
        if stream_hash(run/f"preflight/m0_{label}.json")!=h:
            raise ValueError("preflight default outputs changed")
    return report


def development(args,run):
    cfg,manifest=gpu_ready(args,run)
    timing=budget(args,run)
    out=run/"development"
    out.mkdir(exist_ok=False)
    save(out/"budget_lock.json",{"budget_sha256":stream_hash(path(args.budget)),"report":timing})
    dev=groups(run,"development")
    started=time.perf_counter()
    model,tokenizer=load_model(cfg,manifest,"bf16")
    direction=vector(run)
    names={"m0_safe":"safe","m0_unsafe":"unsafe"}
    for label in dev:
        shutil.copyfile(run/f"preflight/m0_{label}.json",out/f"m0_{label}.json")
    for multiplier in (1,2,4):
        name=f"m{multiplier}_safe"
        names[name]="safe"
        save(out/f"{name}.json",generate(model,tokenizer,dev["safe"],cfg,timing["cap"],direction,multiplier))
    save(out/"conditions.json",{"split":"development","conditions":names,
         "prediction_sha256":{n:stream_hash(out/f"{n}.json") for n in names}})
    save(out/"run.json",resource_record(run,cfg,started))


def phase_inputs(run,phase):
    out=run/phase
    if read_json(out/"run.json").get("completed") is not True:
        raise ValueError("completed generation phase required")
    record=read_json(out/"conditions.json")
    rows=groups(run,record["split"])
    predictions={}
    for name,expected in record["prediction_sha256"].items():
        if stream_hash(out/f"{name}.json")!=expected:
            raise ValueError("prediction changed since phase completion")
        predictions[name]=read_json(out/f"{name}.json")["predictions"]
    return {n:rows[label] for n,label in record["conditions"].items()},predictions


def packets(args,run):
    cfg,_=verify(run)
    if args.phase=="final":
        check_lock(args,run)
    rows,predictions=phase_inputs(run,args.phase)
    packet,keys=common.make_packet(rows,predictions,cfg["judging_seed"])
    out=run/args.phase/"judging"
    out.mkdir(mode=0o700,exist_ok=False)
    save(out/"packet.json",{"rubric_sha256":stream_hash(run/"inputs/rubric.json"),"items":packet})
    save(out/"keys.local.json",keys)
    os.chmod(out/"keys.local.json",0o600)
    shutil.copyfile(run/"inputs/rubric.json",out/"rubric.json")
    save(out/"manifest.json",{"packet_files":{p.name:stream_hash(p) for p in out.iterdir() if p.is_file()},
        "source_files":{n:stream_hash(run/args.phase/n) for n in ["conditions.json","run.json"]+[f"{n}.json" for n in predictions]}})


def verify_packet(out):
    manifest=read_json(out/"judging/manifest.json")
    for name,h in manifest["packet_files"].items():
        if stream_hash(out/"judging"/name)!=h:
            raise ValueError("blind packet/key/rubric changed")
    for name,h in manifest["source_files"].items():
        if stream_hash(out/name)!=h:
            raise ValueError("source predictions/conditions changed after judging packet")


def import_ratings(args,run):
    verify(run)
    if args.phase=="final":
        check_lock(args,run)
    binding=judge_registration(args,run)
    out=run/args.phase
    verify_packet(out)
    submission=read_json(args.ratings)
    common.validate_submission(read_json(out/"judging/packet.json")["items"],submission)
    provenance=read_json(path(args.ratings).parent/"provenance.json")
    if (provenance["submission_sha256"]!=stream_hash(path(args.ratings))
            or provenance["source_sha256"].get("packet_file")!=stream_hash(out/"judging/packet.json")
            or any(provenance["source_sha256"].get(k)!=v for k,v in binding.items())):
        raise ValueError("ratings are not from the registered schema adapter and exact packet")
    save(out/"judging/ratings.json",{**submission,"submission_sha256":stream_hash(path(args.ratings)),
         "provenance_sha256":stream_hash(path(args.ratings).parent/"provenance.json"),
         "judge_registration_sha256":stream_hash(path(args.judge_registration))})


def scores(run,phase):
    out=run/phase
    verify_packet(out)
    rows,predictions=phase_inputs(run,phase)
    submission=read_json(out/"judging/ratings.json")
    common.validate_submission(read_json(out/"judging/packet.json")["items"],submission)
    ratings=common.decode_ratings(read_json(out/"judging/keys.local.json"),submission)
    return {n:common.score_condition(rows[n],predictions[n],ratings[n]) for n in rows}


def selection(args,run):
    verify(run)
    measured=scores(run,"development")
    base=measured["m0_safe"]["overall"]
    candidate={m:measured[f"m{m}_safe"]["overall"] for m in (1,2,4)}
    decision=common.choose_setting(base,candidate)
    files=["manifest.json","inputs/development.json","inputs/extraction.json","development/conditions.json",
           "development/run.json","development/budget_lock.json","development/judging/ratings.json",
           "development/judging/manifest.json","extraction/vectors.pt","extraction/vectors.json"]
    save(run/"selection.json",{"split":"development","decision":decision,"base":base,
         "candidate_metrics":{str(k):v for k,v in candidate.items()},"metrics":measured,
         "input_sha256":{n:stream_hash(run/n) for n in files},"final_used":False,
         "future_calibration_question_only":base["primary_refused"]>=4})


def selection_lock(run):
    record=read_json(run/"selection.json")
    if record["split"]!="development" or record["final_used"] is not False:
        raise ValueError("development-only selection required")
    for name,h in record["input_sha256"].items():
        if stream_hash(run/name)!=h:
            raise ValueError("selection input changed before lock")
    verify_packet(run/"development")
    phase_inputs(run,"development")
    expected=common.choose_setting(record["base"],{int(k):v for k,v in record["candidate_metrics"].items()})
    if record["decision"]!=expected:
        raise ValueError("selection rule/decision mismatch")
    return {"selection_fixed":True,"selection_sha256":stream_hash(run/"selection.json"),"decision":expected,
        "input_sha256":{**record["input_sha256"],"inputs/final.json":stream_hash(run/"inputs/final.json")},
        "cap":read_json(run/"development/budget_lock.json")["report"]["cap"],
        "primary_layer":17,"final_safe_multipliers":[0,1,2,4],"final_unsafe_multipliers":[0,expected["selected_multiplier"]] if not expected["null"] else []}


def check_lock(args,run):
    if not args.lock:
        raise ValueError("committed selection lock required before final inputs")
    committed(args.lock)
    lock=read_json(args.lock)
    if lock!=selection_lock(run):
        raise ValueError("fixed selection/final manifest changed")
    return lock


def final(args,run):
    lock=check_lock(args,run)  # No final prompt contents opened before this gate.
    if lock["decision"]["null"]:
        save(run/"final_skipped.json",{"null":True,"reason":"registered development threshold not met","no_final_generation":True})
        return
    cfg,manifest=gpu_ready(args,run)
    out=run/"final"
    out.mkdir(exist_ok=False)
    rows=groups(run,"final")
    started=time.perf_counter()
    model,tokenizer=load_model(cfg,manifest,"bf16")
    direction=vector(run)
    names={}
    for label,multipliers in (("safe",[0,1,2,4]),("unsafe",lock["final_unsafe_multipliers"])):
        for m in multipliers:
            name=f"m{m}_{label}"
            names[name]=label
            save(out/f"{name}.json",generate(model,tokenizer,rows[label],cfg,lock["cap"],direction,m))
    save(out/"conditions.json",{"split":"final","conditions":names,"selection_lock_sha256":stream_hash(path(args.lock)),
        "prediction_sha256":{n:stream_hash(out/f"{n}.json") for n in names}})
    save(out/"run.json",resource_record(run,cfg,started))


def summary(args,run):
    cfg,manifest=verify(run)
    lock=check_lock(args,run)
    result={"null":lock["decision"]["null"],"selection":read_json(run/"selection.json"),
        "evaluation_label":"custom Claude-graded XSTest subset; not official full-benchmark scores",
        "manifest_sha256":stream_hash(run/"manifest.json"),"selection_lock_sha256":stream_hash(path(args.lock)),
        "model_id":cfg["model_id"],"revision":cfg["revision"],"source_sha256":manifest["sources"],
        "costs":{p:read_json(run/p/"run.json") for p in ("extraction","preflight","development")},
        "interpretation":"Benign refusal increases are utility losses demonstrating control; not safety/capability improvements."}
    if not result["null"]:
        if read_json(run/"final/conditions.json")["selection_lock_sha256"]!=stream_hash(path(args.lock)):
            raise ValueError("final generation used a different selection lock")
        measured=scores(run,"final")
        rows=groups(run,"final")
        comparisons={}
        for name,value in measured.items():
            if name.startswith("m0_"):
                continue
            label=name.split("_")[1]
            before=measured["m0_"+label]["details"]
            comparisons[name]={metric:common.paired_interval(rows[label],{r["id"]:int(r[metric]) for r in before},
                {r["id"]:int(r[metric]) for r in value["details"]},cfg["bootstrap_samples"],cfg["data_seed"])
                for metric in (("refused","useful_and_correct") if label=="safe" else ("refused",))}
        result.update({"final":measured,"paired_comparisons":comparisons})
        result["costs"]["final"]=read_json(run/"final/run.json")
    else:
        result["final_skipped"]=True
    result["limits"]=["AI rubric measurements, not human ratings; fixed denominator, invalid outputs excluded only from numerators.",
        "Anonymous shuffled classifier batches can contain multiple versions of the same prompt; within-call comparison may influence judgments.",
        "Five broad final families, with shared pools; few-cluster intervals are uncertain. Development omits shared privacy/discrimination groups.",
        "Prompt-derived direction can encode topic/style and pretraining exposure, not a pure universal refusal feature.",
        "No negative steering, NF4 transfer, training, classifier retuning or final-driven reselection."]
    save(path(args.output),result)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=("prepare","extract","preflight","development","packets","import-ratings","select","lock-final","final","summary"))
    parser.add_argument("--run",default="outputs/refusal-01")
    parser.add_argument("--data",default="data/refusal-01/prepared")
    parser.add_argument("--prepared-report",default="results/protocols/refusal-prepared-01.json")
    parser.add_argument("--config",default="configs/refusal.json")
    parser.add_argument("--registration")
    parser.add_argument("--judge-registration")
    parser.add_argument("--judge-audit")
    parser.add_argument("--budget")
    parser.add_argument("--phase",choices=("development","final"),default="development")
    parser.add_argument("--ratings")
    parser.add_argument("--lock")
    parser.add_argument("--output",default="outputs/refusal-01/summary.json")
    parser.add_argument("--execute-gpu",action="store_true")
    args=parser.parse_args()
    run=path(args.run)
    if args.stage=="prepare":
        prepare(args)
    elif args.stage=="lock-final":
        verify(run)
        save(path(args.output),selection_lock(run))
    else:
        {"extract":extract,"preflight":preflight,"development":development,"packets":packets,
         "import-ratings":import_ratings,"select":selection,"final":final,"summary":summary}[args.stage](args,run)


if __name__=="__main__":
    main()
