"""Explicit release mechanics: separate frozen-reference/portable GPU processes.

Run with python -I from any working directory. No project imports. Download is
an explicit separate CPU/network phase; all model phases are offline. Receipts
contain hashes/equality only; private CPU logits/token tensors stay alongside the
reference receipt and are never part of the HF payload. No regrading or training.
"""
import argparse
from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path,PurePosixPath
import sys
import types

sys.dont_write_bytecode = True
REPO = "RenormalizationTech/Gemma-4-31B-tuneLLM-QASPER-QLoRA"
CANONICAL = "a149de809997eb8ec18b670a4c1d32eca13edf014bd07fc28333abad3d50b39f"
R0 = "ba1983fdb9e6ca1e11ce9648e69a4c057832d6524b5014cd99cabe3fca4ba9f0"
DEMO = "1603.03876::47ffc9811b037613c9c4d1ec1e4f13c08396ed1c"
PROMPTS = ["In an invented observation, a counter reads 2 and then 3. Repeat its final reading in a short sentence.",
           "In an invented sample, marker A is blue and marker B is red. Name marker A's color in a short sentence."]


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b""):digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path,value):
    with Path(path).open("x") as stream:stream.write(json.dumps(value,indent=2,allow_nan=False)+"\n")


def checked(path,digest):
    if sha(path)!=digest:raise ValueError("verification binding differs: "+Path(path).name)


def import_verified(name,path):
    module=types.ModuleType(name);module.__file__=str(path);module.__package__=name.rpartition(".")[0]
    sys.modules[name]=module
    exec(compile(Path(path).read_bytes(),str(path),"exec"),module.__dict__)
    return module


def bootstrap(bundle,anchor):
    """Hash all trusted source bytes BEFORE compiling any downloaded Python."""
    bundle=Path(bundle).resolve();checked(bundle/"release_manifest.json",anchor)
    manifest=read(bundle/"release_manifest.json")
    for name,item in manifest["files"].items():
        path=PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts or "\\" in name:raise ValueError("unsafe manifest path")
        file=bundle/name
        if file.stat().st_size!=item["bytes"]:raise ValueError("payload size differs")
        checked(file,item["sha256"])
    checked(__file__,manifest["provenance"]["sources"]["scripts/verify_gemma4_qasper_release.py"])
    package=types.ModuleType("runtime");package.__path__=[str(bundle/"runtime")];sys.modules["runtime"]=package
    loader=import_verified("runtime.loader",bundle/"runtime/loader.py")
    loader.validate_bundle(bundle,anchor)
    return loader,manifest


def fresh_download_caches():
    caches={key:os.environ.get(key) for key in ("HF_HOME","HF_HUB_CACHE","HF_XET_CACHE")}
    if not all(caches.values()):raise ValueError("set fresh HF_HOME, HF_HUB_CACHE and HF_XET_CACHE before download")
    for value in caches.values():
        path=Path(value)
        if path.is_symlink() or (path.exists() and (not path.is_dir() or any(p.is_file() or p.is_symlink() for p in path.rglob("*")))):
            raise ValueError("anonymous download caches must be fresh and empty")
    return caches


def download(args):
    """Root-executed anonymous adapter download only; never downloads the base."""
    if args.public_repo!=REPO or not args.public_revision or len(args.public_revision)!=40:
        raise ValueError("exact public repository/revision required")
    destination=Path(args.bundle).absolute()
    if destination.exists():raise FileExistsError("fresh anonymous download directory required")
    caches=fresh_download_caches()
    os.environ.update(HF_HUB_OFFLINE="0",TRANSFORMERS_OFFLINE="0",HF_HUB_DISABLE_IMPLICIT_TOKEN="1",
                      HUGGINGFACE_HUB_CACHE=caches["HF_HUB_CACHE"])
    from huggingface_hub import HfApi,snapshot_download
    api=HfApi(token=False);info=api.model_info(REPO,revision=args.public_revision,token=False)
    if info.private or info.sha!=args.public_revision:raise ValueError("public pinned revision differs")
    snapshot_download(REPO,revision=args.public_revision,local_dir=destination,token=False)
    loader,manifest=bootstrap(destination,args.manifest_sha256)
    listed={s.rfilename for s in info.siblings}
    if listed-set(loader.FILES)-{".gitattributes"} or set(loader.FILES)-listed:
        raise ValueError("public file allowlist differs")
    files={name:{"sha256":sha(destination/name),"bytes":(destination/name).stat().st_size} for name in sorted(loader.FILES)}
    result={"status":"anonymous_adapter_download_verified","repo_id":REPO,"revision":args.public_revision,
        "private":False,"release_manifest_sha256":args.manifest_sha256,"files":files,
        "published_file_count":len(files),"published_inventory_sha256":hashlib.sha256(json.dumps(files,sort_keys=True,separators=(",",":")).encode()).hexdigest(),
        "anonymous_pinned_download_verified":True,"base_downloaded":False,
        "fresh_download_caches":True,"download_token":False,
        "cache_location_sha256":{key:hashlib.sha256(str(Path(value).absolute()).encode()).hexdigest() for key,value in caches.items()},
        "base_scope":"Existing pinned base cache is reused and separately content-hash verified during offline reload"}
    save(args.receipt,result);return result


def saved_inputs(args,manifest):
    root=Path(args.study_root).resolve();canonical=Path(args.canonical)
    checked(canonical,CANONICAL);record=read(canonical)
    checked(args.r0,R0);study=read(args.r0)
    checked(root/"completion-seal.json",record["completion_seal_sha256"])
    seal=read(root/"completion-seal.json");rows={}
    checked(root/"training/registration.json",seal["roles"]["training"]["registration.json"])
    for role,folder in (("final_base","final-base"),("final_adapter","final-adapter")):
        path=root/folder/"predictions.jsonl"
        checked(path,seal["roles"][role]["predictions.jsonl"])
        all_rows=[json.loads(line) for line in path.read_text().split("\n") if line.strip()]
        matches=[row for row in all_rows if row["id"]==DEMO]
        if len(matches)!=1:raise ValueError("one fixed saved demonstration row required")
        rows[role]=matches[0]
        rows[role+"_ordered"]=all_rows
    if rows["final_base"]["input_ids"]!=rows["final_adapter"]["input_ids"]:
        raise ValueError("saved base/adapter prompts differ")
    data_path=root/"final-base/data.json"
    cfg=read(root/"final-base/registration.json")
    checked(data_path,cfg["inputs"]["data.json"])
    all_data=read(data_path);data=[r for r in all_data if r["id"]==DEMO]
    if len(data)!=1:raise ValueError("fixed raw prompt record missing")
    rows["raw_data_by_id"]={r["id"]:r for r in all_data}
    return root,study,rows,data[0]


def frozen_runtime(root,manifest,study):
    checked(root/"completion-seal.json",manifest["provenance"]["completion_seal_sha256"])
    seal=read(root/"completion-seal.json")
    checked(root/"training/registration.json",seal["roles"]["training"]["registration.json"])
    source=root/"training/source";cfg=read(root/"training/registration.json")
    for name,item in cfg["source_files"].items():checked(source/name,item["sha256"])
    checked(source/"scripts/gemma4_preflight.py",study["sources"]["scripts/gemma4_preflight.py"])
    path=source/"scripts/gemma4_qasper_runtime.py"
    checked(path,study["sources"]["scripts/gemma4_qasper_runtime.py"])
    checked(path,manifest["provenance"]["frozen_runtime_sha256"])
    package=types.ModuleType("scripts");package.__path__=[str(source/"scripts")];sys.modules["scripts"]=package
    import_verified("scripts.gemma4_preflight",source/"scripts/gemma4_preflight.py")
    return import_verified("scripts.gemma4_qasper_runtime",path),cfg


def tensor_hash(value):
    return hashlib.sha256(value.contiguous().numpy().tobytes()).hexdigest()


def saved_replay_plan(saved):
    plan=[("saved_demo.base",False,saved["final_base"]),
          ("saved_demo.adapter",True,saved["final_adapter"])]
    for adapter,role,ordinals in ((True,"final_adapter",range(4)),(False,"final_base",range(1))):
        plan.extend((f"saved_ordinal{ordinal}."+("adapter" if adapter else "base"),adapter,
                     saved[role+"_ordered"][ordinal]) for ordinal in ordinals)
    return plan


def validate_saved_output(tokens,text,saved_tokens=None,saved_output=None):
    if saved_tokens is not None and tokens!=saved_tokens:
        raise ValueError("saved full trajectory replay differs")
    if saved_output is not None and text!=saved_output:
        raise ValueError("saved decoded output replay differs")


def mechanics(args):
    os.environ.update(CUDA_VISIBLE_DEVICES=args.gpu,HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1",TOKENIZERS_PARALLELISM="false")
    loader,manifest=bootstrap(args.bundle,args.manifest_sha256)
    loader.validate_environment();loader.validate_base(args.base,manifest)
    root,study,saved,data=saved_inputs(args,manifest)
    import torch
    from transformers import AutoTokenizer,set_seed
    from safetensors.torch import load_file,save_file
    set_seed(42)
    if args.phase=="reference":
        frozen,cfg=frozen_runtime(root,manifest,study)
        frozen.require_idle(args.gpu)
        identity=dict(cfg["identity"],base=str(Path(args.base).resolve()))
        selected=root/"training/checkpoints/step-00061"
        checked(selected/"adapter_model.safetensors",manifest["adapter"]["source_weights_sha256"])
        model,_=frozen.load_runtime(identity,cfg["targets"],selected)
        tokenizer=AutoTokenizer.from_pretrained(args.base,local_files_only=True,trust_remote_code=False)
        model.eval()
        def render(user):return frozen.render(tokenizer,user)
        def logits(ids,adapter):
            batch=torch.tensor([ids],dtype=torch.long,device="cuda:0")
            with (nullcontext() if adapter else model.disable_adapter()),torch.inference_mode(),frozen.amp(model):
                return model(input_ids=batch,attention_mask=torch.ones_like(batch),use_cache=False,logits_to_keep=1).logits.detach().float().cpu()
        def generate(ids,adapter,cap):
            with (nullcontext() if adapter else model.disable_adapter()),torch.inference_mode():
                return frozen.generate(model,tokenizer,ids,cap,float("inf"))
        reload_proof={"tensor_count":820,"exact_tensor_equality":True}
    else:
        if not args.reference:raise ValueError("frozen CPU reference receipt required")
        reference=read(args.reference)
        if reference["release_manifest_sha256"]!=args.manifest_sha256 or reference["status"]!="reference_mechanics_passed":
            raise ValueError("frozen reference manifest/status differs")
        artifact=Path(args.reference).parent/reference["CPU_reference"]["name"]
        checked(artifact,reference["CPU_reference"]["sha256"])
        expected=load_file(str(artifact),device="cpu")
        runtime=loader.load_runtime(args.bundle,args.base,args.manifest_sha256)
        model=runtime.model;render=runtime.render;logits=runtime.next_logits;generate=runtime.generate_ids
        reload_proof=runtime.receipt["adapter_reload"]
    tensors={};cells={}
    def cell(key,ids,adapter,cap,saved_tokens=None,saved_output=None):
        value=logits(ids,adapter)
        result=generate(ids,adapter,cap)
        tokens=torch.tensor(result["generated_ids"],dtype=torch.int64)
        if not bool(torch.isfinite(value).all()):raise ValueError("nonfinite mechanics logits")
        validate_saved_output(tokens.tolist(),result["output"],saved_tokens,saved_output)
        tensors[key+".logits"]=value;tensors[key+".tokens"]=tokens
        if args.phase!="reference":
            if not torch.equal(value,expected[key+".logits"]) or not torch.equal(tokens,expected[key+".tokens"]):
                raise ValueError("portable/frozen exact mechanics parity differs: "+key)
        cells[key]={"logits_sha256":tensor_hash(value),"tokens_sha256":tensor_hash(tokens),
            "generated_tokens":tokens.numel(),"finite":True,"saved_replay_exact":saved_tokens is not None,
            "saved_decoded_output_exact":saved_output is not None,
            "portable_reference_exact":args.phase!="reference"}
    baseline=None
    flags_before=[bool(m.disable_adapters) for m in model.modules() if hasattr(m,"disable_adapters") and isinstance(m.disable_adapters,bool)]
    hooks_before=[(len(m._forward_hooks),len(m._forward_pre_hooks)) for m in model.modules()]
    for i,user in enumerate(PROMPTS):
        _,ids=render(user)
        if i==0:baseline=logits(ids,False)
        for adapter in (False,True):cell(f"invented{i}."+("adapter" if adapter else "base"),ids,adapter,32)
    for key,adapter,row in saved_replay_plan(saved):
        raw=saved["raw_data_by_id"][row["id"]]
        prompt,ids=render(raw["user"])
        if ids!=row["input_ids"] or prompt!=row["rendered_prompt"]:
            raise ValueError("CLI/native render differs from saved exact prompt")
        cell(key,ids,adapter,512,row["generated_ids"],row["output"])
    _,ids=render(PROMPTS[0])
    if not torch.equal(baseline,logits(ids,False)):raise ValueError("post-adapter baseline cleanup logits differ")
    flags_after=[bool(m.disable_adapters) for m in model.modules() if hasattr(m,"disable_adapters") and isinstance(m.disable_adapters,bool)]
    if flags_before!=flags_after or hooks_before!=[(len(m._forward_hooks),len(m._forward_pre_hooks)) for m in model.modules()]:
        raise ValueError("adapter context/hook cleanup differs")
    result={"status":"reference_mechanics_passed" if args.phase=="reference" else "portable_mechanics_passed",
        "release_manifest_sha256":args.manifest_sha256,"verification_source_sha256":sha(__file__),
        "base_repo_id":loader.BASE,"base_revision":loader.REVISION,"base_files_hash_verified":True,
        "base_scope":"reused pinned base cache; not a fresh base download","phase":args.phase,
        "adapter_reload":reload_proof,"guard":"frozen torch.inference_mode + eval + outer BF16 autocast",
        "cells":cells,"cleanup_exact":True,"no_optimizer_updates":True,"no_regrading":True,
        "timing_scope":"includes coexistence; no exclusive timing claim","working_tree_modules_imported":False,
        "device":{"name":torch.cuda.get_device_name(0),"uuid":str(torch.cuda.get_device_properties(0).uuid)},
        "frozen_source_scope":"reference phase imports only hash-verified preserved study closure, never working-project modules"}
    if args.phase=="reference":
        file=Path(args.receipt).with_suffix(".reference.safetensors")
        if file.exists():raise FileExistsError("preserve prior reference artifacts")
        save_file(tensors,str(file));result["CPU_reference"]={"name":file.name,"sha256":sha(file)}
    else:
        result["reference_receipt_sha256"]=sha(args.reference)
        if args.download_receipt:
            download_proof=read(args.download_receipt)
            if download_proof["release_manifest_sha256"]!=args.manifest_sha256 or not download_proof["anonymous_pinned_download_verified"]:
                raise ValueError("anonymous adapter download proof differs")
            if not args.local_receipt:raise ValueError("local portable receipt required for public comparison")
            local=read(args.local_receipt)
            if local["cells"]!=cells or local["release_manifest_sha256"]!=args.manifest_sha256:
                raise ValueError("anonymous public/local replay receipts differ")
            result.update(status="public_portable_mechanics_passed",repo_id=download_proof["repo_id"],revision=download_proof["revision"],
                anonymous_pinned_download_verified=True,download_receipt_sha256=sha(args.download_receipt),local_receipt_sha256=sha(args.local_receipt))
    save(args.receipt,result)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phase",choices=["reference","portable","download"],required=True)
    p.add_argument("--bundle",required=True);p.add_argument("--manifest-sha256",required=True);p.add_argument("--receipt",required=True)
    p.add_argument("--base");p.add_argument("--study-root");p.add_argument("--canonical");p.add_argument("--r0")
    p.add_argument("--gpu",choices=["0","1"],default="0");p.add_argument("--reference")
    p.add_argument("--public-repo",default=REPO);p.add_argument("--public-revision")
    p.add_argument("--download-receipt");p.add_argument("--local-receipt")
    args=p.parse_args()
    if Path(args.receipt).exists():raise FileExistsError("fresh immutable receipt required")
    if args.phase!="download" and not all((args.base,args.study_root,args.canonical,args.r0)):
        p.error("model phases require --base --study-root --canonical --r0")
    result=download(args) if args.phase=="download" else mechanics(args)
    print(json.dumps({k:v for k,v in result.items() if k not in ("cells","files")},indent=2))


if __name__=="__main__":main()
