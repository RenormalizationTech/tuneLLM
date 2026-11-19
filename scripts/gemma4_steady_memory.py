"""Separate synthetic steady-state memory check after a fresh adapter reload.

Warm up Adam's moment tensors on a short synthetic sequence before measuring a
full 8192-token / 512-target update. No task examples or quality calls are used.
"""
import argparse
import json
from pathlib import Path
import time

from scripts import gemma4_preflight as preflight


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",default="outputs/gemma4-preflight-02")
    args=parser.parse_args();preflight.RUN=Path(args.run).resolve()
    run=preflight.RUN
    if not run.is_relative_to(preflight.ROOT/"outputs") or (run/"steady_memory.json").exists():
        raise ValueError("fresh steady-state result inside project outputs required")
    reload=json.loads((run/"fresh_reload.json").read_text())
    if reload["status"]!="passed":
        raise ValueError("fresh-process adapter reload must pass first")
    import torch
    from transformers import AutoTokenizer,set_seed
    from peft import PeftModel,prepare_model_for_kbit_training
    result={"scope":"synthetic mechanics only; no task training or test/judge calls",
        "status":"running","source_sha256":preflight.sha(__file__),
        "helper_sha256":preflight.sha(preflight.__file__),
        "fresh_reload_sha256":preflight.sha(run/"fresh_reload.json"),"versions":preflight.versions()}
    preflight.write(run/"steady_memory.json",result);start=time.monotonic();set_seed(42)
    try:
        model,_=preflight.load_base()
        model=prepare_model_for_kbit_training(model,use_gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant":False})
        model=PeftModel.from_pretrained(model,run/"synthetic_adapter",is_trainable=True)
        model.config.use_cache=False;model.train()
        params=[p for n,p in model.named_parameters() if p.requires_grad]
        if sum(p.numel() for p in params)!=122429440 or any("lora_" not in n for n,p in model.named_parameters() if p.requires_grad):
            raise ValueError("unexpected trainable inventory")
        tokenizer=AutoTokenizer.from_pretrained(preflight.snapshot(),local_files_only=True)
        optimizer=torch.optim.AdamW(params,lr=5e-5)
        ids,labels,mask=preflight.synthetic(tokenizer,64,8)
        loss=preflight.response_loss(model,ids,labels,mask);loss.backward();optimizer.step()
        optimizer.zero_grad(set_to_none=True);del ids,labels,mask,loss
        moment_bytes=sum(v.numel()*v.element_size() for state in optimizer.state.values()
                         for k,v in state.items() if k in {"exp_avg","exp_avg_sq"})
        if moment_bytes!=2*122429440*4:
            raise ValueError("Adam FP32 moment states were not fully allocated")
        ids,labels,mask=preflight.synthetic(tokenizer,8192,512)
        torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();tick=time.monotonic()
        loss=preflight.response_loss(model,ids,labels,mask);loss.backward()
        gradients=[p.grad for p in params if p.grad is not None]
        if len(gradients)!=len(params) or not all(bool(torch.isfinite(g).all()) for g in gradients):
            raise ValueError("missing or nonfinite steady-state gradients")
        norm=torch.nn.utils.clip_grad_norm_(params,1.0);optimizer.step();torch.cuda.synchronize()
        result.update(status="passed",sequence_length=8192,response_tokens=512,loss=loss.item(),
            gradient_tensors=len(gradients),finite_gradients=True,gradient_norm_before_clip=norm.item(),
            preexisting_Adam_moment_bytes=moment_bytes,full_step_seconds=time.monotonic()-tick,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            device_free_bytes=torch.cuda.mem_get_info()[0],synthetic_updates=2)
    except Exception as error:
        result.update(status="failed",error_type=type(error).__name__,error=str(error));raise
    finally:
        result["seconds_including_load"]=time.monotonic()-start
        preflight.write(run/"steady_memory.json",result)
        print(json.dumps(result,indent=2),flush=True)


if __name__=="__main__":main()
