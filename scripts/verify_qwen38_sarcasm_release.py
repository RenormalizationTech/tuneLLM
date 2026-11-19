"""Isolated local/offline release mechanics. PM executes; never judges/scores.

Works as: python -I /absolute/verify_qwen38_sarcasm_release.py --bundle ...
No repository imports. Frozen port is loaded directly by importlib. No raw
generated text, token arrays, logits, or local paths enter the output receipt.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

PORT_SHA='3d76d8147ffbe6f802b285d643b4363e09349f0dd14a5f6a2924545b5966abe3'
CANONICAL_SHA='61362b38ecb32581b58adb6f5e5caaaec7f16598d985ef1b834568210ca081ad'
AGGREGATE_SHA='2bf33ce55cae146c2cd3d11bfc2f963fd193e449917589ffd4b075ecb0a83a12'


def sha(file):
    h=hashlib.sha256()
    with Path(file).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def signature(model):return {n:(tuple(m._forward_hooks),tuple(m._forward_pre_hooks)) for n,m in model.named_modules()}


def load_frozen(file):
    if sha(file)!=PORT_SHA:raise ValueError('frozen port hash differs')
    spec=importlib.util.spec_from_file_location('_reviewed_frozen_gcad',file)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def save(file,value):
    file.parent.mkdir(parents=True,exist_ok=True)
    with file.open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')


def verify(args):
    # Direct script execution with -I does not add its directory to sys.path.
    # Only the explicitly checked downloaded bundle is added, never the project.
    if not sys.flags.isolated:raise ValueError('run this verifier in a fresh python -I process')
    if any(name=='runtime' or name.startswith('runtime.') for name in sys.modules):raise ValueError('runtime already imported; fresh process required')
    root=Path(args.bundle).absolute()
    manifest=root/'release_manifest.json'
    if sha(manifest)!=args.manifest_sha256:raise ValueError('external manifest trust anchor mismatch')
    mf=json.loads(manifest.read_text())
    for name in ['runtime/__init__.py','runtime/loader.py','runtime/gcad.py']:
        if sha(root/name)!=mf['files'][name]['sha256']:raise ValueError('runtime bootstrap source changed')
    sys.path.insert(0,str(root))
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1';os.environ['TOKENIZERS_PARALLELISM']='false'
    os.environ['CUDA_VISIBLE_DEVICES']=args.gpu_uuid
    from runtime import load_runtime
    from transformers import GenerationConfig,set_seed
    import torch
    frozen=load_frozen(Path(args.frozen_port));tick=time.monotonic()
    controls=load_runtime(args.base,args.manifest_sha256,bundle=root,device='cuda:0')
    model=controls.model;tok=controls.tokenizer;bank=controls.bank;sites=mf['sites'];before=signature(model)
    records=[]
    def native_generate(ids,mask,b,cap):
        cfg=GenerationConfig.from_pretrained(args.base,local_files_only=True)
        cfg.update(do_sample=False,repetition_penalty=1.05,max_new_tokens=cap,use_cache=True,num_beams=1,
            temperature=None,top_p=None,top_k=None,pad_token_id=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id)
        set_seed(42)
        from contextlib import nullcontext
        context=frozen.Steerer(model,bank,sites,b,1.5) if b is not None else nullcontext()
        with torch.inference_mode(),context:
            out=model.generate(input_ids=ids,attention_mask=mask,generation_config=cfg,logits_to_keep=1)
        return out[0,ids.shape[1]:].tolist()
    def native_logits(ids,mask,b):
        from contextlib import nullcontext
        ctx=frozen.Steerer(model,bank,sites,b,1.5) if b is not None else nullcontext()
        with torch.inference_mode(),ctx:return model(ids,attention_mask=mask,use_cache=True,logits_to_keep=1).logits.detach().clone()
    for pi,text in enumerate(['What causes thunder?','How does a compass work?']):
        rendered=tok.apply_chat_template([{'role':'user','content':text}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
        ids=torch.tensor([tok(rendered,add_special_tokens=False)['input_ids']],device=model.device);mask=torch.ones_like(ids)
        ordinary=native_logits(ids,mask,None);ordinary_tokens=native_generate(ids,mask,None,32)
        for b in [0,8,16]:
            expected=ordinary if b==0 else native_logits(ids,mask,b)
            actual=controls.forward(ids,mask,b).logits
            if not torch.equal(expected,actual):raise ValueError(f'exact frozen/portable logit parity failed: probe{pi} b{b}')
            a=controls.generate(text,b,32);reference=ordinary_tokens if b==0 else native_generate(ids,mask,b,32)
            if a['generated_ids']!=reference:raise ValueError('exact frozen/portable tokens differ')
            if not torch.isfinite(actual).all() or signature(model)!=before:raise ValueError('nonfinite output or hook leak')
            if b and any(v!={'prefill':1,'decode':len(reference)-1,'errors':0} for v in a['audit'].values()):raise ValueError('cached hook coverage differs')
            records.append({'probe':pi,'strength':b,'logits_equal':True,'tokens_equal':True,'finite':True,
                'suffix_length':len(reference),'suffix_sha256':digest(reference),
                'logits_sha256':hashlib.sha256(actual.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()})
        if not torch.equal(ordinary,native_logits(ids,mask,None)) or ordinary_tokens!=native_generate(ids,mask,None,32) or signature(model)!=before:
            raise ValueError('cleanup token/logit parity failed')
    aggregate=Path(args.saved_run)/'generation/result.json'
    if sha(aggregate)!=AGGREGATE_SHA:raise ValueError('saved aggregate differs from authoritative canonical result binding')
    study=json.loads(aggregate.read_text())['rows']
    chosen=[next(r for r in study if r['id']==name) for name in ['F01_baseline','F01_b8','F01_b16']]
    followup_manifest=json.loads((Path(args.saved_run)/'manifest.json').read_text())
    if sha(Path(args.saved_run)/'manifest.json')!='87d1d0e6fb925b3607403e54502f0f462a6ec5453bf4b96411d49f101e468bf3':raise ValueError('saved study manifest differs')
    replays=[]
    for row in chosen:
        raw=Path(args.saved_run)/'generation/rows'/f'{row["id"]}.json'
        receipt=json.loads(raw.with_name(raw.stem+'.receipt.json').read_text())
        start=raw.with_name(raw.stem+'.start.json')
        if sha(raw)!=receipt['sha256'] or sha(start)!=receipt['start_sha256']:raise ValueError('saved trajectory receipt differs')
        saved=json.loads(raw.read_text())
        if saved['value']['generated_ids']!=row['generated_ids'] or saved['value']['input_ids']!=row['input_ids']:raise ValueError('saved row/aggregate mismatch')
        ids=torch.tensor([row['input_ids']],device=model.device);mask=torch.ones_like(ids)
        rendered=tok.apply_chat_template([{'role':'user','content':row['user']}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
        if tok(rendered,add_special_tokens=False)['input_ids']!=row['input_ids']:raise ValueError('CLI user-only rendering differs from saved trajectory input')
        b=0 if row['arm']=='baseline' else int(row['arm'][1:])
        actual=controls.generate_ids(ids,mask,b,160)['generated_ids']
        if actual!=row['generated_ids']:raise ValueError(f'saved trajectory replay differs: {row["id"]}')
        replays.append({'id':row['id'],'strength':b,'tokens_equal':True,'suffix_length':len(actual),
            'saved_row_sha256':sha(raw),'suffix_sha256':digest(actual)})
    if signature(model)!=before:raise ValueError('hooks leaked after replay')
    return {'status':'passed_release_mechanics','scope':'Mechanical exactness only; no judge or quality evaluation',
        'release_manifest_sha256':args.manifest_sha256,'frozen_port_sha256':PORT_SHA,
        'canonical_result_sha256':CANONICAL_SHA,'saved_generation_aggregate_sha256':AGGREGATE_SHA,
        'verifier_sha256':sha(__file__),'GPU_uuid':args.gpu_uuid,'GPU_name':torch.cuda.get_device_name(0),
        'python_isolated':bool(sys.flags.isolated),'offline':True,'probe_cap':32,'study_replay_cap':160,
        'probes':records,'saved_trajectory_replays':replays,'cleanup_equal':True,'cleanup_tokens_equal':True,
        'cache_equivalence':'Strict cached/full-prefix equivalence remains unestablished; identical cached path used.',
        'elapsed_seconds':time.monotonic()-tick,'raw_outputs_recorded':False}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle',required=True);p.add_argument('--base',required=True);p.add_argument('--manifest-sha256',required=True)
    p.add_argument('--frozen-port',required=True);p.add_argument('--saved-run',required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--gpu-uuid',default='GPU-34646ab9-ec19-ca63-0dfb-bdbf2e1a977b')
    p.add_argument('--compare-receipt',type=Path,help='Require equality to local staged mechanics hashes after public download')
    args=p.parse_args()
    if args.output.exists():raise FileExistsError('fresh verifier receipt required')
    try:
        value=verify(args)
        if args.compare_receipt:
            old=json.loads(args.compare_receipt.read_text())
            for key in ['release_manifest_sha256','frozen_port_sha256','probes','saved_trajectory_replays','cleanup_equal','cleanup_tokens_equal']:
                if old[key]!=value[key]:raise ValueError(f'public/local runtime mechanics differ: {key}')
            value['local_receipt_sha256']=sha(args.compare_receipt);value['public_local_mechanics_equal']=True
    except BaseException as exc:
        # Error details can contain private filesystem paths. Record type only.
        save(args.output,{'status':'release_mechanics_failed','error_type':type(exc).__name__,
            'release_manifest_sha256':args.manifest_sha256,'verifier_sha256':sha(__file__),'raw_outputs_recorded':False})
        raise
    save(args.output,value);print(json.dumps({'status':value['status'],'receipt_sha256':sha(args.output)},indent=2))


if __name__=='__main__':main()
