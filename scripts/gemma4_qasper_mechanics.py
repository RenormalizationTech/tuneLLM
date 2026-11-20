"""Hash-bound, no-update real-model checks before the Gemma QASPER study."""
import argparse
import datetime
import os
from pathlib import Path
import time

from scripts import gemma4_qasper_analysis as a
from scripts import gemma4_qasper_runtime as rt
from scripts import gemma4_train_qasper as trainer


def prepare(training,output):
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    training=Path(training).resolve()
    sources={name:a.sha(a.ROOT/name) for name in [
        "scripts/gemma4_qasper_mechanics.py","scripts/gemma4_qasper_analysis.py",
        "scripts/gemma4_qasper_runtime.py","scripts/gemma4_train_qasper.py","scripts/gemma4_preflight.py"]}
    manifest={"scope":"Mechanics only: no optimizer update, generation, quality scoring or selection",
        "registered_at_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "training":str(training),"inputs":{name:a.sha(training/name) for name in ("registration.json","train.json")},
        "sources":sources,"gpu":"0","dropout":.05,"outer_bf16_autocast":True,
        "gradient_atol":1e-4,"gradient_rtol":1e-4}
    rt.atomic_json(output/"manifest.json",manifest,new=True)
    return {"manifest":str(output/"manifest.json"),"sha256":a.sha(output/"manifest.json")}


def execute(output,binding):
    import torch
    from transformers import set_seed
    output=Path(output).resolve();manifest=a.read(output/"manifest.json")
    if a.sha(output/"manifest.json")!=binding:raise ValueError("mechanics manifest changed")
    training=Path(manifest["training"])
    for name,digest in manifest["sources"].items():
        if a.sha(a.ROOT/name)!=digest:raise ValueError("mechanics source changed: "+name)
    for name,digest in manifest["inputs"].items():
        if a.sha(training/name)!=digest:raise ValueError("mechanics input changed")
    if os.environ.get("CUDA_VISIBLE_DEVICES")!="0":raise ValueError("GPU0 visibility required")
    if (output/"result.json").exists():raise FileExistsError("mechanics receipt exists")
    cfg=a.read(training/"registration.json");rows=a.read(training/"train.json")
    started=time.time();result={"status":"failed","manifest_sha256":binding,"optimizer_updates":0}
    try:
        result["gpu_uuid"]=rt.require_idle("0")
        set_seed(42);loaded=time.time()
        model,runtime=rt.load_runtime(cfg["identity"],cfg["targets"])
        result["load_seconds"]=time.time()-loaded;result["runtime"]=runtime
        print("model_loaded",flush=True)
        parameters=[p for p in model.parameters() if p.requires_grad]
        optimizer=torch.optim.AdamW(parameters,lr=5e-5,weight_decay=.01)
        scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda step:1.)
        trainer.save_checkpoint(output,0,model,optimizer,scheduler,binding,0)
        model.train();probe_started=time.time()
        parity=trainer.compact_parity(model,rows[0],1e-4)
        result["short_compact_full_ce"]=parity
        result["short_probe_seconds"]=time.time()-probe_started
        print("short_loss_and_gradient_parity_passed",flush=True)
        def loss(row):
            tensors={k:torch.tensor([row[k]],device=parameters[0].device,dtype=torch.long)
                for k in ("input_ids","labels","attention_mask")}
            with rt.amp(model):value=rt.response_loss(model,tensors["input_ids"],tensors["labels"],tensors["attention_mask"])
            if not torch.isfinite(value):raise ValueError("nonfinite longest-row loss")
            return value,sum(i!=-100 for i in row["labels"][1:])
        probe_started=time.time()
        mechanics=trainer.no_update_mechanics(model,parameters,loss,max(rows,key=lambda r:len(r["input_ids"])),
            lambda:trainer.restore_checkpoint(output/"checkpoints/step-00000",model,optimizer,scheduler,binding))
        result.update(mechanics,registration_sha256=binding,outer_bf16_autocast=True)
        result["longest_probe_seconds"]=time.time()-probe_started
        result["memory_scope"]="No Adam moments allocated: optimizer has not stepped; earlier steady-state preflight measured those separately"
        a.validate_startup(result,binding)
        result["status"]="passed"
        for name,digest in manifest["sources"].items():
            if a.sha(a.ROOT/name)!=digest:raise ValueError("mechanics source changed during check")
    except BaseException as exc:
        result.update(status="failed",error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        result["elapsed_seconds"]=time.time()-started
        rt.atomic_json(output/"result.json",result,new=True)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=["prepare","execute"])
    parser.add_argument("--output",required=True);parser.add_argument("--training");parser.add_argument("--binding")
    args=parser.parse_args()
    result=prepare(args.training,args.output) if args.command=="prepare" else execute(args.output,args.binding)
    print(__import__("json").dumps(result,indent=2,allow_nan=False))


if __name__=="__main__":main()
