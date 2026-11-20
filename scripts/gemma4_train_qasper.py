"""One fixed Gemma QASPER fit, with durable optimizer-boundary recovery.

Every optimizer update publishes an immutable, checksummed adapter/optimizer/RNG
checkpoint. An interrupted uncommitted group is rolled back to that boundary;
no checkpoint is selected from incomplete development scoring. No quality retry.
Preparation and execution are separate; PM supplies the prospective manifest SHA.
"""
import argparse
import json
import math
import os
from pathlib import Path
import random
import time

from scripts import gemma4_qasper_runtime as rt

STARTUP_VALIDATION = {
    "G1":{"gate":True,"description":"Native compact logits versus full[:,positions], upcast FP32; max absolute logit difference and CE difference <=1e-4","absolute_tolerance":1e-4},
    "G2":{"gate":True,"description":"One detached shared FP32 full-logit leaf: masked shifted CE versus gathered response CE; loss difference<=1e-4, dL allclose, exactly zero unsupervised/last-row gradients","atol":1e-4,"rtol":1e-4},
    "G3":{"gate":True,"description":"Longest selected real row, train mode/dropout0.05/outer BF16 AMP, no optimizer update; all820 trainable gradients finite, restore initialization/RNG/buffers and clear gradients"},
    "D1":{"gate":False,"description":"Native full versus compact all820 LoRA gradient max absolute/relative error and tolerance ratio; diagnostic only"},
    "D2":{"gate":False,"description":"Native compact repeated same-input gradient max difference; diagnostic only, not proof of equivalence"},
    "native_gradient_equivalence_claim":False,
    "recipe_change":False,
}


def selected_rows(source_rows, ids):
    lookup = {r["id"]: r for r in source_rows}
    if len(lookup) != len(source_rows) or len(set(ids)) != len(ids):
        raise ValueError("duplicate IDs")
    selected = [lookup[key] for key in ids]
    if any(not r["training_eligible"] for r in selected):
        raise ValueError("frozen selection no longer eligible")
    return selected


def groups(count=482):
    order = list(range(count)); random.Random(42).shuffle(order)
    return [order[n:n+8] for n in range(0, count, 8)]


def prepare(args, tokenizer=None):
    from transformers import AutoTokenizer
    protocol_path = rt.ROOT / "results/protocols/qwen38-qasper-02.json"
    protocol = rt.read(protocol_path)
    prepared_manifest = Path(args.prepared)/"manifest.json"
    if rt.sha(prepared_manifest) != protocol["config"]["prepared_manifest_sha256"]:
        raise ValueError("original prepared manifest changed")
    base = Path(args.base).resolve()
    identity = rt.model_identity(base)
    tokenizer = tokenizer or AutoTokenizer.from_pretrained(base, local_files_only=True)
    geometry = rt.geometry(base)
    inputs, provenance = {}, {}
    for split, count in (("train",482),("validation",101)):
        source = Path(args.prepared) / f"{split}.jsonl"
        expected = protocol["config"]["prepared_input_sha256"][split]
        if rt.sha(source) != expected:
            raise ValueError("original Qwen prepared bytes differ")
        selected = selected_rows(rt.rows(source), protocol["selected_ids"][split])
        if len(selected) != count:
            raise ValueError("exact Qwen selected count required")
        encoded = [{"id":r["id"], "paper_id":r["paper_id"],
                    **rt.encode(tokenizer,r["user"],r["assistant"],r.get("system"))} for r in selected]
        if any(len(r["input_ids"]) > 8192 for r in encoded):
            raise ValueError("full example exceeds8192; no reselection/truncation")
        inputs[split + ".json"] = encoded
        provenance[split] = {"path":str(source.resolve()),"sha256":expected,
                             "ids":protocol["selected_ids"][split]}
    if {r["paper_id"] for r in inputs["train.json"]} & {r["paper_id"] for r in inputs["validation.json"]}:
        raise ValueError("paper overlap")
    cfg = {"kind":"training", "task":"qasper", "gpu":args.gpu, "identity":identity,
        "targets":geometry["targets"], "preflight_geometry_sha256":rt.sha(rt.PREFLIGHT / "cpu_prepare.json"),
        "qwen_selection_sha256":rt.sha(protocol_path), "data_provenance":provenance,
        "prepared_manifest_sha256":rt.sha(prepared_manifest),
        "prepared_input_sha256":{s:p["sha256"] for s,p in provenance.items()},
        "seed":42,"rank":16,"alpha":32,"dropout":0.05,
        "dropout_difference":"Matches Qwen0.05; measured Gemma synthetic preflight0.0; new longest-row mechanics check", "batch_size":1,
        "max_length":8192,"gradient_accumulation":8,"learning_rate":5e-5,"weight_decay":.01,
        "max_grad_norm":1.0,"epochs":1,"total_steps":61,"warmup_steps":4,
        "checkpoint_steps":[21,41,61],"max_seconds":args.max_seconds,
        "optimization_weighting":"equal-example mean; final partial group divided by actual2",
        "selection":"minimum token-weighted101 development response/turn-end NLL; earliest step ties",
        "recovery":"optimizer-boundary checkpoints every update; rollback interrupted uncommitted group",
        "autocast":True,"short_ce_absolute_tolerance":1e-4,
        "startup_validation":STARTUP_VALIDATION,
        "startup_profile_difference":"dropout.05 and outerBF16 AMP; original Gemma preflight0/noAMP",
        "test_data_loaded_by_trainer":False}
    return rt.register(args.output, cfg, inputs)


def verify_checkpoint(directory, binding):
    receipt = rt.read(Path(directory) / "receipt.json")
    if receipt["registration_sha256"] != binding:
        raise ValueError("checkpoint registration mismatch")
    rt.verify_files(directory, receipt["files"])
    if set(rt.inventory(directory)) != set(receipt["files"]) | {"receipt.json"}:
        raise ValueError("checkpoint has unbound files")
    return receipt


def save_checkpoint(run, step, model, optimizer, scheduler, binding, elapsed, metrics=None):
    import torch
    import numpy as np
    parent = Path(run) / "checkpoints"; parent.mkdir(exist_ok=True)
    directory = parent / f"step-{step:05d}"
    staging = parent / f".step-{step:05d}-{time.time_ns()}"
    staging.mkdir()
    model.save_pretrained(staging, safe_serialization=True)
    torch.save({"optimizer":optimizer.state_dict(),"scheduler":scheduler.state_dict(),
        "torch_rng":torch.get_rng_state(),"cuda_rng":torch.cuda.get_rng_state_all(),
        "python_rng":random.getstate(),"numpy_rng":np.random.get_state(),
        "step":step,"next_group":step}, staging / "state.pt")
    for p in staging.iterdir():
        if p.is_file():
            with p.open("rb") as stream:
                os.fsync(stream.fileno())
    receipt = {"step":step,"registration_sha256":binding,"elapsed_seconds":elapsed,
               "metrics":metrics,"files":rt.inventory(staging)}
    rt.atomic_json(staging / "receipt.json", receipt, new=True)
    if directory.exists():
        raise FileExistsError("immutable checkpoint already present")
    os.rename(staging, directory)
    fd = os.open(parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    return receipt


def restore_checkpoint(directory, model, optimizer, scheduler, binding):
    import torch
    import numpy as np
    receipt = verify_checkpoint(directory, binding)
    rt.reload_adapter(model, directory)
    # Trusted local file, checked by the registered checkpoint inventory above.
    state = torch.load(Path(directory) / "state.pt", map_location="cpu", weights_only=False)
    if state["step"] != receipt["step"] or state["next_group"] != state["step"]:
        raise ValueError("optimizer step mismatch")
    optimizer.load_state_dict(state["optimizer"]); scheduler.load_state_dict(state["scheduler"])
    torch.set_rng_state(state["torch_rng"]); torch.cuda.set_rng_state_all(state["cuda_rng"])
    random.setstate(state["python_rng"])
    np.random.set_state(state["numpy_rng"])
    return receipt


def elapsed_attempts(run, now=None):
    """Conservatively charge killed attempts through recovery time, including load."""
    now = time.time() if now is None else now
    return sum(max(0, (r.get("end_unix") or r.get("recovery_unix") or now)-r["start_unix"])
               for r in (rt.read(p) for p in (Path(run)/"attempts").glob("*.json")))


def recover_attempts(run):
    for path in (Path(run)/"attempts").glob("*.json"):
        prior = rt.read(path)
        if prior.get("error_type") not in (None,"KeyboardInterrupt","SystemExit"):
            raise ValueError("deterministic failure requires new amended registration, not resume")
        if "end_unix" not in prior and "recovery_unix" not in prior:
            # Retain the original started record; append one recovery field, charging
            # downtime conservatively instead of silently erasing an unknown tail.
            prior["recovery_unix"] = time.time()
            rt.atomic_json(path,prior)


def no_update_mechanics(model, parameters, loss, longest, restore):
    """One longest-row backward, then restore adapter/optimizer/RNG and buffers."""
    import torch
    buffers = {n:b.detach().clone() for n,b in model.named_buffers()}
    cuda = parameters[0].device.type == "cuda"
    for p in parameters:
        p.grad = None
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    value,n = loss(longest); value.backward()
    if cuda:
        torch.cuda.synchronize()
    if any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in parameters):
        raise ValueError("longest-row mechanics missing/nonfinite gradients")
    for p in parameters:
        p.grad = None
    restore()
    with torch.no_grad():
        for name,buffer in model.named_buffers():
            buffer.copy_(buffers[name])
    return {"id":longest["id"],"length":len(longest["input_ids"]),"dropout":.05,
        "gradient_count":len(parameters),"gradients_finite":True,"finite_gradients":True,
        "gradients_cleared":all(p.grad is None for p in parameters),
        "state_restored":True,"initialization_restored":True,"rng_restored":True,
        "optimizer_scheduler_restored":True,"buffers_restored":True,
        "optimizer_updates":0,"restored_adapter_optimizer_scheduler_rng_buffers":True,
        "peak_allocated_bytes":torch.cuda.max_memory_allocated() if cuda else 0,
        "peak_reserved_bytes":torch.cuda.max_memory_reserved() if cuda else 0}


def shared_response_ce(logits,labels,positions):
    """FP32 gathered response CE on the single shared logit leaf (G2)."""
    import torch
    selected = logits[:,positions]
    return torch.nn.functional.cross_entropy(selected.reshape(-1,selected.shape[-1]),
                                             labels[:,positions+1].reshape(-1))


def gradient_diagnostics(reference,actual,tolerance=1e-4):
    """Native parameter-gradient differences are descriptive, never a gate."""
    import torch
    maximum = 0.; relative = 0.; ratio = 0.; finite = True; missing = 0
    for expected,grad in zip(reference,actual):
        if expected is None or grad is None:
            missing += 1; finite = False; continue
        value = grad.detach().float().cpu()
        expected = expected.detach().float().cpu()
        if not bool(torch.isfinite(expected).all() and torch.isfinite(value).all()):
            finite = False; continue
        difference = (value-expected).abs()
        maximum = max(maximum,difference.max().item())
        relative = max(relative,(difference/torch.maximum(expected.abs(),value.abs()).clamp_min(1e-12)).max().item())
        ratio = max(ratio,(difference/(tolerance+tolerance*expected.abs())).max().item())
    return {"gating":False,"gradient_count":len(reference),"missing_gradient_count":missing,
        "gradients_finite":finite,"max_absolute_error":maximum,"max_relative_error":relative,
        "max_tolerance_ratio":ratio,"atol":tolerance,"rtol":tolerance}


def shared_logit_parity(full_logits,labels,tolerance=1e-4):
    """G2: compare two FP32 CE graphs with exactly one detached shared leaf."""
    import torch
    if tolerance != 1e-4:
        raise ValueError("registered tolerance must remain1e-4")
    positions = torch.nonzero(labels[0,1:] != -100,as_tuple=False).flatten()
    shared = full_logits.detach().float().requires_grad_(True)
    full = torch.nn.functional.cross_entropy(shared[:,:-1].reshape(-1,shared.shape[-1]),labels[:,1:].reshape(-1))
    compact = shared_response_ce(shared,labels,positions)
    full_gradient = torch.autograd.grad(full,shared)[0]
    compact_gradient = (torch.autograd.grad(compact,shared,allow_unused=True)[0]
                        if compact.requires_grad else None)
    unsupervised = torch.ones(shared.shape[1],dtype=torch.bool,device=shared.device)
    unsupervised[positions] = False
    finite = bool(torch.isfinite(full) and torch.isfinite(compact) and torch.isfinite(full_gradient).all())
    allclose = compact_gradient is not None and bool(torch.allclose(compact_gradient,full_gradient,atol=tolerance,rtol=tolerance))
    zeros = bool(torch.count_nonzero(full_gradient[:,unsupervised]) == 0)
    zeros = zeros and compact_gradient is not None and bool(torch.count_nonzero(compact_gradient[:,unsupervised]) == 0)
    last_zero = bool(torch.count_nonzero(full_gradient[:,-1]) == 0)
    last_zero = last_zero and compact_gradient is not None and bool(torch.count_nonzero(compact_gradient[:,-1]) == 0)
    maximum = None
    if compact_gradient is not None:
        finite = finite and bool(torch.isfinite(compact_gradient).all())
        maximum = (compact_gradient-full_gradient).abs().max().item()
        if not torch.isfinite(torch.tensor(maximum)):
            maximum = None
    difference = abs(full.item()-compact.item())
    return {"passed":finite and difference <= tolerance and allclose and zeros and last_zero,
        "absolute_ce_difference":difference,"gradients_allclose":allclose,
        "unsupervised_gradients_exact_zero":zeros,"last_unused_row_gradients_exact_zero":last_zero,
        "finite":finite,"gradient_max_absolute_error":maximum,"absolute_tolerance":tolerance,
        "gradient_atol":tolerance,"gradient_rtol":tolerance,"shared_leaf_count":1,
        "unsupervised_rows":int(unsupervised.sum().item())}


def compact_parity(model,row,tolerance=1e-4):
    """G1/G2 gate arithmetic; D1/D2 qualify native gradient interpretation only.

    Native forwards use eval mode with outer BF16 autocast and autograd enabled.
    The separate G3 longest-row probe uses actual training mode/dropout0.05.
    All modes, RNGs and buffers are restored and parameter gradients cleared.
    """
    import torch
    import numpy as np
    if tolerance != 1e-4:
        raise ValueError("registered CE/gradient tolerance must remain1e-4")
    n = row["prompt_tokens"]
    prefix = row["input_ids"][max(0,n-32):n]
    answer = row["input_ids"][n:min(n+16,len(row["input_ids"])-1)]+[106]
    ids = torch.tensor([prefix+answer],dtype=torch.long,device=next(model.parameters()).device)
    labels = ids.clone(); labels[:,:len(prefix)] = -100
    positions = torch.nonzero(labels[0,1:] != -100,as_tuple=False).flatten()
    parameters = [p for p in model.parameters() if p.requires_grad]
    if not parameters:
        raise ValueError("trainable parameters required for gradient diagnostics")
    mode = model.training; modes = [(module,module.training) for module in model.modules()]
    buffers = {name:buffer.detach().clone() for name,buffer in model.named_buffers()}
    rng = {"python":random.getstate(),"numpy":np.random.get_state(),"torch":torch.get_rng_state(),
           "cuda":torch.cuda.get_rng_state_all() if parameters[0].is_cuda else None}
    def restore():
        random.setstate(rng["python"]); np.random.set_state(rng["numpy"]); torch.set_rng_state(rng["torch"])
        if rng["cuda"] is not None:
            torch.cuda.set_rng_state_all(rng["cuda"])
        with torch.no_grad():
            for name,buffer in model.named_buffers():
                buffer.copy_(buffers[name])
        for p in parameters:
            p.grad = None
    def native_compact_gradients():
        restore()
        with torch.enable_grad(), rt.amp(model):
            loss = rt.response_loss(model,ids,labels,torch.ones_like(ids))
        grads = torch.autograd.grad(loss,parameters,allow_unused=True) if loss.requires_grad else [None]*len(parameters)
        return [g.detach().float().cpu() if g is not None else None for g in grads]
    report = {"absolute_tolerance":tolerance,"gradient_atol":tolerance,"gradient_rtol":tolerance,
        "tokens":ids.shape[1],"gradient_count":len(parameters),"model_mode":"eval; autograd enabled",
        "optimizer_updates":0,"native_gradient_equivalence_claim":False,"passed":False,
        "diagnostics":{"gating":False,"interpretation":"Native BF16 backward paths may differ; repeated compact gradients qualify interpretation and are not proof of equivalence"}}
    failure = None
    try:
        restore(); model.eval()
        with torch.enable_grad(), rt.amp(model):
            logits = model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False).logits.float()
            full = torch.nn.functional.cross_entropy(logits[:,:-1].reshape(-1,logits.shape[-1]),labels[:,1:].reshape(-1))
        full_value = full.item(); detached = logits.detach()
        grads = torch.autograd.grad(full,parameters,allow_unused=True)
        reference = [g.detach().float().cpu() if g is not None else None for g in grads]
        del grads,full,logits
        restore()
        with torch.enable_grad(), rt.amp(model):
            compact_logits = model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False,
                                   logits_to_keep=positions).logits.float().detach()
            compact = torch.nn.functional.cross_entropy(compact_logits.reshape(-1,compact_logits.shape[-1]),labels[:,positions+1].reshape(-1))
        maximum = (compact_logits-detached[:,positions]).abs().max().item()
        difference = abs(full_value-compact.item())
        finite = bool(torch.isfinite(compact_logits).all() and torch.isfinite(detached).all() and np.isfinite([full_value,compact.item()]).all())
        report["g1"] = {"passed":finite and maximum <= tolerance and difference <= tolerance,
            "max_absolute_logit_difference":maximum,"absolute_ce_difference":difference,
            "full_ce":full_value,"compact_ce":compact.item(),"finite":finite,"absolute_tolerance":tolerance}
        del compact_logits,compact
        with torch.enable_grad():
            report["g2"] = shared_logit_parity(detached,labels,tolerance)
        del detached
        actual = native_compact_gradients()
        report["diagnostics"]["D1"] = gradient_diagnostics(reference,actual,tolerance)
        del reference
        repeated = native_compact_gradients()
        report["diagnostics"]["D2"] = gradient_diagnostics(actual,repeated,tolerance)
        report["passed"] = report["g1"]["passed"] and report["g2"]["passed"]
        if not report["passed"]:
            failure = "G1" if not report["g1"]["passed"] else "G2"
    finally:
        restore()
        for module,previous_mode in modes:
            module.training = previous_mode
        report.update(rng_restored=True,buffers_restored=True,model_mode_restored=model.training == mode,
                      gradients_cleared=all(p.grad is None for p in parameters))
    if failure:
        raise ValueError("startup compact/full parity failed "+failure+": "+json.dumps(report))
    return report


def execute(args):
    run = Path(args.run).resolve(); binding = args.registration_sha256
    cfg = rt.frozen_exec("scripts.gemma4_train_qasper",run,binding,args.resume)
    if cfg.get("kind") != "training":
        raise ValueError("training registration required")
    with rt.exclusive(run):
        return train(run,cfg,binding,args.resume)


def train(run,cfg,binding,resume):
    import torch
    from transformers import set_seed
    if (run / "result.json").exists() and rt.read(run / "result.json").get("completed"):
        raise FileExistsError("completed fit is immutable")
    attempts = run / "attempts"; attempts.mkdir(exist_ok=True)
    if list(attempts.glob("*.json")) and not resume:
        raise ValueError("explicit --resume required; prior attempts preserved")
    if resume:
        recover_attempts(run)
    previous = elapsed_attempts(run)
    if cfg["max_seconds"] is not None and previous >= cfg["max_seconds"]:
        raise TimeoutError("cumulative training wall cap exhausted")
    deadline = time.monotonic() + cfg["max_seconds"] - previous if cfg["max_seconds"] is not None else float("inf")
    gpu_uuid = rt.require_idle(cfg["gpu"])
    attempt_path = attempts / f"attempt-{len(list(attempts.glob('*.json')))+1:03d}.json"
    attempt = {"start_unix":time.time(),"registration_sha256":binding,"resume":resume,"gpu_uuid":gpu_uuid,
               "occupancy_snapshot":rt.OCCUPANCY_SNAPSHOT}
    rt.atomic_json(attempt_path,attempt,new=True)
    result = {"completed":False,"steps":0,"checkpoints":[],"registration_sha256":binding,
              "occupancy_snapshot":rt.OCCUPANCY_SNAPSHOT}
    def check():
        if time.monotonic() >= deadline:
            raise TimeoutError("registered cumulative training/development wall cap")
    try:
        set_seed(42)
        model,runtime = rt.load_runtime(cfg["identity"],cfg["targets"])
        result["runtime"] = runtime; check()
        parameters = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(parameters,lr=5e-5,weight_decay=.01)
        def lr_factor(step):
            return (step+1)/4 if step < 4 else max(0,(61-step)/57)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,lr_factor)
        checkpoints = sorted((run / "checkpoints").glob("step-*"))
        if checkpoints:
            if not resume:
                raise ValueError("unexpected existing checkpoints")
            receipts = [verify_checkpoint(p,binding) for p in checkpoints]
            if [r["step"] for r in receipts] != list(range(receipts[-1]["step"]+1)):
                raise ValueError("noncontiguous durable optimizer history")
            restored = restore_checkpoint(checkpoints[-1],model,optimizer,scheduler,binding)
            result["steps"] = restored["step"]
        else:
            save_checkpoint(run,0,model,optimizer,scheduler,binding,previous)
        training = rt.read(run / "train.json"); validation = rt.read(run / "validation.json")
        def loss(row):
            tensors = {k:torch.tensor([row[k]],dtype=torch.long,device=next(model.parameters()).device)
                       for k in ("input_ids","labels","attention_mask")}
            with rt.amp(model):
                value = rt.response_loss(model,tensors["input_ids"],tensors["labels"],tensors["attention_mask"])
            if not torch.isfinite(value):
                raise ValueError("nonfinite response loss")
            return value, sum(v != -100 for v in row["labels"][1:])
        def dev():
            model.eval(); total = 0.; tokens = 0
            with torch.no_grad():
                for row in validation:
                    check(); value,n = loss(row); total += value.item()*n; tokens += n
            model.train()
            return {"nll":total/tokens,"loss_tokens":tokens,"count":len(validation)}
        model.train()
        if result["steps"] == 0 and not (run/"longest_row_mechanics.json").exists():
            longest = max(training,key=lambda row:len(row["input_ids"]))
            check()
            parity = compact_parity(model,training[0],cfg["short_ce_absolute_tolerance"])
            mechanics = no_update_mechanics(model,parameters,loss,longest,
                lambda:restore_checkpoint(run/"checkpoints/step-00000",model,optimizer,scheduler,binding))
            rt.atomic_json(run/"longest_row_mechanics.json",{"registration_sha256":binding,
                           "short_compact_full_ce":parity,"outer_bf16_autocast":True,**mechanics},new=True)
        # A selected boundary's NLL is published separately, after durable optimizer
        # state. If interrupted in NLL, resume that same fixed evaluation, no choice.
        def evaluate_boundary(step):
            path = run / f"development-{step:05d}.json"
            if not path.exists():
                metrics = dev()
                rt.atomic_json(path,{"step":step,"registration_sha256":binding,**metrics},new=True)
            record = rt.read(path)
            if record["registration_sha256"] != binding:
                raise ValueError("development binding changed")
            return {**record,"path":str(run / "checkpoints" / f"step-{step:05d}")}
        if result["steps"] in cfg["checkpoint_steps"]:
            evaluate_boundary(result["steps"])
        for step,group in enumerate(groups(len(training)),1):
            if step <= result["steps"]:
                continue
            check(); optimizer.zero_grad(set_to_none=True)
            total = 0.; tokens = 0
            for index in group:
                check(); value,n = loss(training[index]); (value/len(group)).backward()
                total += value.item()*n; tokens += n
            norm = torch.nn.utils.clip_grad_norm_(parameters,1.0,error_if_nonfinite=True)
            check(); optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True)
            save_checkpoint(run,step,model,optimizer,scheduler,binding,
                            previous+time.time()-attempt["start_unix"],
                            {"group_token_weighted_nll":total/tokens,"target_tokens":tokens,
                             "gradient_norm":norm.item(),"group_indices":group})
            result["steps"] = step
            rt.atomic_json(run/"heartbeat.json",{"unit":"optimizer_step","step":step,"timestamp_unix":time.time()})
            if step in cfg["checkpoint_steps"]:
                evaluate_boundary(step)
            rt.atomic_json(run / "result.json",{**result,"gradient_norm":norm.item()})
            print(json.dumps({"step":step,"of":61,"durable":True}),flush=True)
        check()
        result["checkpoints"] = [evaluate_boundary(step) for step in cfg["checkpoint_steps"]]
        result["selected"] = min(result["checkpoints"],key=lambda r:(r["nll"],r["step"]))
        result["completed"] = result["steps"] == 61
        result["selected_files"] = rt.inventory(Path(result["selected"]["path"]))
    except BaseException as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        attempt["error_type"] = type(exc).__name__
        raise
    finally:
        attempt["end_unix"] = time.time(); attempt["completed"] = result["completed"]
        rt.atomic_json(attempt_path,attempt)
        result["elapsed_seconds"] = elapsed_attempts(run)
        result["gpu_hours"] = result["elapsed_seconds"]/3600
        rt.atomic_json(run / "result.json",result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command",required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--base",default=(rt.PREFLIGHT/"snapshot_path.txt").read_text().strip() if (rt.PREFLIGHT/"snapshot_path.txt").exists() else None)
    prep.add_argument("--prepared",default=str(rt.ROOT/"outputs/qwen38-science-01/prepared-qasper-tokenizer-01"))
    prep.add_argument("--output",required=True); prep.add_argument("--gpu",choices=["0","1"],default="0")
    prep.add_argument("--max-seconds",type=int,default=None,help="Optional operational cap; no study-level cap by default")
    run = sub.add_parser("execute"); run.add_argument("--run",required=True)
    run.add_argument("--registration-sha256",required=True); run.add_argument("--resume",action="store_true")
    args = p.parse_args()
    if args.command == "prepare":
        if args.max_seconds is not None and args.max_seconds <= 0:
            raise ValueError("positive optional operational cap required")
        result = prepare(args)
    else:
        result = execute(args)
    print(json.dumps(result,indent=2))


if __name__ == "__main__":
    main()
