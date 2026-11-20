"""Gemma fixed128 MC/128 GSM8K/32 IFEval; singleton native scoring.

MC sums full-answer log probabilities under the original raw concept prompt,
including multi-token Gemma labels. Outer BF16 autocast matches training/eval.
Frozen Qwen/GSM8K/official IFEval scoring helpers remain unchanged.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

from scripts import gemma4_qasper_runtime as rt
from scripts.gemma4_eval_qasper import adapter_binding, run_generations, SOURCES as EVAL_SOURCES
from scripts.qwen38_retention import validate_fixture, score_predictions, verify_ifeval_assets

SOURCES = EVAL_SOURCES + ["scripts/qwen38_retention.py", "evaluation/gsm8k.py",
    "scripts/benchmark_screen.py", "scripts/batched_generation.py", "scripts/common.py", "scripts/gguf_eval.py"]


def full_choices(tokenizer, prompt, choices):
    if tokenizer.bos_token_id != 2:
        raise ValueError("native Gemma MC BOS id2 required")
    prefix = tokenizer(prompt,add_special_tokens=False)["input_ids"]
    values = []
    for choice in choices:
        full = tokenizer(prompt+choice,add_special_tokens=False)["input_ids"]
        if full[:len(prefix)] != prefix or len(full) <= len(prefix) or len(full)+1 > 8192:
            raise ValueError("MC full-answer boundary/length differs; no single-token assumption")
        values.append([2]+full)
    if len(values) != 4 or len({tuple(v) for v in values}) != 4:
        raise ValueError("four distinct full answer trajectories required")
    return [2]+prefix, values


def choice_likelihoods(model,row,deadline):
    import torch
    scores = []
    for ids in row["choice_input_ids"]:
        if time.monotonic() >= deadline:
            raise TimeoutError("MC forward cap")
        n = len(row["input_ids"])
        positions = torch.arange(n-1,len(ids)-1,device=next(model.parameters()).device)
        batch = torch.tensor([ids],dtype=torch.long,device=positions.device)
        with rt.amp(model):
            logits = model(input_ids=batch,attention_mask=torch.ones_like(batch),
                           logits_to_keep=positions,use_cache=False).logits[0].float()
        targets = batch[0,n:]
        if logits.shape[0] != targets.numel() or not bool(torch.isfinite(logits).all()):
            raise ValueError("nonfinite/mismatched full-answer logits")
        scores.append(torch.log_softmax(logits,-1).gather(-1,targets[:,None]).sum().item())
    return {"choice_loglikelihood":scores,"predicted_index":max(range(4),key=lambda i:scores[i]),
            "hit_token_limit":False,"resource_cap_stop":time.monotonic() >= deadline}


def analysis_binding(training,analysis_path):
    if training and not analysis_path:
        raise ValueError("adapter retention requires prospective analysis registration")
    return ({"path":str(Path(analysis_path).resolve()),"sha256":rt.sha(analysis_path)}
            if analysis_path else None)


def prepare(args,tokenizer=None):
    from transformers import AutoTokenizer
    from scripts.benchmark_screen import choice_token_ids
    analysis = analysis_binding(args.training,getattr(args,"analysis_registration",None))
    fixture_path = rt.ROOT/"data/benchmark-screen-01/test.json"
    provenance = rt.read(rt.ROOT/"data/benchmark-screen-01/provenance.json")
    if rt.sha(fixture_path) != provenance["fixture_sha256"]["test"]:
        raise ValueError("historical retention fixture changed")
    full = rt.read(fixture_path); tasks = validate_fixture(full,False)
    fixture = {task:full[task] for task in tasks}
    assets = verify_ifeval_assets()
    identity = rt.model_identity(Path(args.base)); geometry = rt.geometry(args.base)
    tokenizer = tokenizer or AutoTokenizer.from_pretrained(args.base,local_files_only=True)
    choice_ids = choice_token_ids(tokenizer,fixture["general_mc"])
    prepared = []
    for task in tasks:
        for row in fixture[task]:
            if task == "general_mc":
                prompt = row["prompt"]; ids,choices = full_choices(tokenizer,prompt,row["choices"])
                value = {"choice_input_ids":choices}; cap = 0; key = row["id"]
            else:
                user = row["user"] if task == "gsm8k" else row["prompt"]
                key = row["id"] if task == "gsm8k" else str(row["key"])
                prompt,ids = rt.render(tokenizer,user); value = {}; cap = 512 if task == "gsm8k" else 1024
            if not ids or len(ids)+cap > 8192:
                raise ValueError("full retention prompt/output exceeds8192; no truncation")
            prepared.append({"ordinal":len(prepared),"id":str(key),"task":task,
                "rendered_prompt":prompt,"input_ids":ids,"max_new_tokens":cap,**value})
    adapter = adapter_binding(args.training) if args.training else None
    if adapter and adapter["identity"] != identity:
        raise ValueError("retention adapter base identity differs")
    scorer = Path(args.scorer_python).absolute()
    cfg = {"kind":"retention","task":"qasper","gpu":args.gpu,"identity":identity,
        "final_analysis_registration":analysis,
        "targets":geometry["targets"],"adapter":adapter,"max_seconds":args.max_seconds,
        "batch_size":1,"mc_batch_size":1,"mc_bos_token_id":2,
        "mc":"sum full-answer conditional log probabilities; native BOS + raw concept prompt",
        "generation_caps":{"gsm8k":512,"ifeval":1024},"eos_token_id":[1,106],
        "autocast":True,"choice_ids":choice_ids,"seed":42,"fixture_sha256":rt.sha(fixture_path),"ifeval_assets":assets,
        "original_root":str(rt.ROOT),"scorer_python":str(scorer),"scorer_python_sha256":rt.sha(scorer),
        "scorer_sources":{name:rt.sha(rt.ROOT/name) for name in SOURCES},
        "base_preparation":"identical FP32 kbit casts and rank16 wrapper; disabled adapter for baseline"}
    # Exercise official scorer schema on blanks during CPU preparation, never a quality call.
    validation = isolated_ifeval(cfg,fixture["ifeval"],
        [{"id":str(r["key"]),"output":""} for r in fixture["ifeval"]])
    cfg["scorer_packages"] = validation["packages"]
    return rt.register(args.output,cfg,{"fixture.json":fixture,"prepared.json":prepared},SOURCES)


def isolated_ifeval(cfg, rows, predictions):
    root = Path(cfg["original_root"])
    if rt.sha(cfg["scorer_python"]) != cfg["scorer_python_sha256"]:
        raise ValueError("official scorer interpreter changed")
    for name,digest in cfg["scorer_sources"].items():
        if rt.sha(root/name) != digest:
            raise ValueError("frozen scoring helper changed")
    for name,digest in cfg["ifeval_assets"]["files"].items():
        if rt.sha(name) != digest:
            raise ValueError("IFEval assets changed")
    command = [cfg["scorer_python"],"-B","-m","scripts.qwen38_retention","--ifeval-worker"]
    proc = subprocess.run(command,input=json.dumps({"rows":rows,"predictions":predictions}),
        capture_output=True,text=True,check=True,timeout=300,cwd=root,
        env=dict(os.environ,CUDA_VISIBLE_DEVICES="",PYTHONPATH=str(root),HF_HUB_OFFLINE="1"))
    return json.loads(proc.stdout)


def score(run,expected):
    run = Path(run).resolve(); cfg = rt.validate(run,expected)
    result = rt.read(run/"result.json")
    if not result["completed"] or rt.sha(run/"predictions.jsonl") != result["predictions_sha256"]:
        raise ValueError("complete hash-bound retention predictions required")
    def ifeval(rows,predictions):
        scored = isolated_ifeval(cfg,rows,predictions)
        if scored["packages"] != cfg["scorer_packages"]:
            raise ValueError("scoring package versions changed")
        return scored["scores"]
    scores = score_predictions(rt.read(run/"fixture.json"),rt.rows(run/"predictions.jsonl"),ifeval)
    rt.atomic_json(run/"scores.json",scores,new=True)
    return scores


def main():
    p = argparse.ArgumentParser(description=__doc__); sub = p.add_subparsers(dest="command",required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--base",default=None); prep.add_argument("--output",required=True)
    prep.add_argument("--gpu",choices=["0","1"],default="1"); prep.add_argument("--training")
    prep.add_argument("--analysis-registration",help="Required for selected-adapter retention; optional for initial base")
    prep.add_argument("--scorer-python",default=str(rt.ROOT/".venv/bin/python"))
    prep.add_argument("--max-seconds",type=int,default=None)
    for command in ("execute","score"):
        parser = sub.add_parser(command); parser.add_argument("--run",required=True)
        parser.add_argument("--registration-sha256",required=True)
        if command == "execute":
            parser.add_argument("--resume",action="store_true")
    args = p.parse_args()
    if args.command == "prepare":
        args.base = args.base or (rt.PREFLIGHT/"snapshot_path.txt").read_text().strip()
        result = prepare(args)
    elif args.command == "score":
        result = score(args.run,args.registration_sha256)
    else:
        result = run_generations(args,retention=True)
    print(json.dumps(result,indent=2))


if __name__ == "__main__":
    main()
