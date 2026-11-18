"""Download a pinned reviewed HF release, verify every file, then reload explicitly.

Downloading and GPU reload are separate phases so downloads can overlap training.
No repository writes, quantization, adapter selection, or final quality examples.
"""
import argparse
from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path,PurePosixPath
import time

from scripts.qwen38_science_runtime import sha,write,versions

ROOT=Path(__file__).resolve().parents[1]
REPOS={k:'RenormalizationTech/Qwen3.8-27B-tuneLLM-'+v for k,v in
       {'nf4':'NF4','gguf':'GGUF','qasper':'QASPER-QLoRA','prl':'PRL-Abstract-QLoRA'}.items()}
SOURCES=['scripts/'+n+'.py' for n in ['qwen38_gguf','qwen38_gguf_check','gguf_eval',
    'qwen38_nf4_smoke','qwen38_science_runtime','qwen38_eval_adapter','qwen38_eval_science']]
REFERENCES=['results/qwen38_27b_nf4_artifact_01.json','results/qwen38_27b_nf4_smoke_02.json',
    'outputs/qwen38-27b-gguf-01/result.json','outputs/qwen38-27b-gguf-01/runtime-smoke-q4-01/result.json']


def publication(path,kind):
    record=json.loads(path.read_text())
    if record['repo_id']!=REPOS[kind] or record['status']!='published_remote_hashes_verified' or record['private'] is not False:
        raise ValueError('expected public revision with verified remote hashes required')
    if len(record['revision'])!=40 or any(c not in '0123456789abcdef' for c in record['revision']):
        raise ValueError('immutable commit SHA required')
    if not record['files'] or any(PurePosixPath(n).is_absolute() or '..' in PurePosixPath(n).parts for n in record['files']):
        raise ValueError('nonempty in-repository file manifest required')
    return record


def download_record(record):
    os.environ['HF_HOME']=str(ROOT/'.hf')
    os.environ.pop('HF_TOKEN',None);os.environ.pop('HUGGING_FACE_HUB_TOKEN',None)
    from huggingface_hub import snapshot_download,HfApi
    listed=set(HfApi(token=False).list_repo_files(record['repo_id'],revision=record['revision'],token=False))
    validate_listing(listed,record['files'])
    snapshot=Path(snapshot_download(record['repo_id'],revision=record['revision'],
        allow_patterns=sorted(listed),token=False,cache_dir=str(ROOT/'.hf/qwen38-release-verification/hub')))
    for name,expected in record['files'].items():
        file=snapshot/name
        if file.stat().st_size!=expected['bytes'] or sha(file)!=expected['sha256']:
            raise ValueError(f'downloaded bytes differ: {name}')
    files={n:{'bytes':(snapshot/n).stat().st_size,'sha256':sha(snapshot/n)} for n in sorted(listed)}
    return snapshot,files


def validate_listing(listed,files):
    if not set(files)<=listed or listed-set(files)-{'.gitattributes'}:
        raise ValueError('pinned public repository has missing or unreviewed files')


def validate_artifact(record,expected):
    for name,value in expected.items():
        actual=record['files'].get(name)
        if actual is None or any(actual[k]!=value[k] for k in ['bytes','sha256']):
            raise ValueError(f'published artifact differs from evaluated bytes: {name}')


def artifact_files(kind):
    if kind=='nf4':
        return {n:r for n,r in json.loads((ROOT/'results/qwen38_27b_nf4_artifact_01.json').read_text())['files'].items() if n!='smoke_origin.json'}
    if kind=='gguf':
        name='Qwen3.8-27B-text-Q4_K_M.gguf'
        return {name:json.loads((ROOT/'outputs/qwen38-27b-gguf-01/result.json').read_text())['files'][name]}
    raise ValueError('base representation required')


def native_replay(model,inputs):
    # Exactly the creation-time smoke call; preserve generation-config EOS and
    # native arithmetic rather than introducing an autocast or stopping change.
    return model.generate(**inputs,do_sample=False,num_beams=1,max_new_tokens=64,use_cache=True,
                          temperature=None,top_p=None,top_k=None,repetition_penalty=1.0)


def download(args):
    out=(ROOT/args.output).resolve()
    if not out.is_relative_to(ROOT/'outputs'):raise ValueError('verification records belong inside outputs')
    if out.exists():raise FileExistsError('preserve previous download record')
    record=publication(args.publication,args.kind)
    if args.kind in ['nf4','gguf']:validate_artifact(record,artifact_files(args.kind))
    out.mkdir(parents=True)
    started=time.monotonic();snapshot,files=download_record(record)
    base_record=None;base_snapshot=None
    if args.kind in ['qasper','prl']:
        if args.base_publication is None:raise ValueError('NF4 dependency publication required')
        base_record=publication(args.base_publication,'nf4');validate_artifact(base_record,artifact_files('nf4'))
        base_snapshot,_=download_record(base_record)
    result={'completed':True,'kind':args.kind,'repo_id':record['repo_id'],'revision':record['revision'],
            'publication':str(args.publication.resolve()),'publication_sha256':sha(args.publication),
            'snapshot':str(snapshot),'all_file_hashes_verified':True,'files':files,'anonymous_access_verified':True,
            'base_publication':str(args.base_publication.resolve()) if args.base_publication else None,
            'base_publication_sha256':sha(args.base_publication) if args.base_publication else None,
            'base_snapshot':str(base_snapshot) if base_snapshot else None,
            'base_revision':base_record['revision'] if base_record else None,
            'bound_local_sha256':{n:sha(ROOT/n) for n in SOURCES+REFERENCES},
            'download_seconds':time.monotonic()-started,'source_sha256':sha(__file__)}
    write(out/'download.json',result)
    return {k:v for k,v in result.items() if k not in ['files']}


def execute(args):
    out=ROOT/args.output;downloaded=json.loads((out/'download.json').read_text())
    if not downloaded['completed'] or not downloaded['all_file_hashes_verified'] or downloaded['kind']!=args.kind:
        raise ValueError('complete pinned download required')
    if sha(__file__)!=downloaded['source_sha256'] or sha(downloaded['publication'])!=downloaded['publication_sha256']:
        raise ValueError('verification source/publication changed')
    for name,expected in downloaded['bound_local_sha256'].items():
        if sha(ROOT/name)!=expected:raise ValueError('verification reference/helper changed')
    if args.kind in ['nf4','gguf']:validate_artifact(json.loads(Path(downloaded['publication']).read_text()),artifact_files(args.kind))
    if (out/'result.json').exists():raise FileExistsError('GPU reload already attempted')
    snapshot=Path(downloaded['snapshot'])
    for name,expected in downloaded['files'].items():
        if sha(snapshot/name)!=expected['sha256']:raise ValueError('downloaded file changed')
    os.environ['CUDA_VISIBLE_DEVICES']=args.gpu
    started=time.monotonic();result={'completed':False,'download_sha256':sha(out/'download.json'),
        'gpu_count':1,'gpu':args.gpu,'versions':versions(),'kind':args.kind}
    write(out/'result.json',result)
    try:
        if args.kind=='gguf':
            from scripts.qwen38_gguf_check import run
            run(argparse.Namespace(model=str(snapshot/'Qwen3.8-27B-text-Q4_K_M.gguf'),
                output=str(out/'runtime'),execute_smoke=True,gpus=args.gpu))
            fresh=json.loads((out/'runtime/result.json').read_text())
            original=json.loads((ROOT/'outputs/qwen38-27b-gguf-01/runtime-smoke-q4-01/result.json').read_text())
            result['exact_smoke_token_replay']=fresh['completed'] and len(fresh['predictions'])==len(original['predictions']) and all(
                a['ids']==b['ids'] and a['response']['tokens']==b['response']['tokens'] for a,b in zip(fresh['predictions'],original['predictions']))
            if not result['exact_smoke_token_replay']:raise ValueError('downloaded Q4 smoke tokens differ')
        else:
            import torch
            from transformers import AutoTokenizer,Qwen3_5ForCausalLM,set_seed
            from scripts.qwen38_nf4_smoke import validate_loading
            set_seed(42)
            if args.kind=='nf4':
                model,info=Qwen3_5ForCausalLM.from_pretrained(snapshot,local_files_only=True,
                    trust_remote_code=False,dtype=torch.bfloat16,device_map={'':'cuda:0'},
                    attn_implementation='sdpa',output_loading_info=True)
                validate_loading(info)
                tokenizer=AutoTokenizer.from_pretrained(snapshot,local_files_only=True)
                reference=json.loads((ROOT/'results/qwen38_27b_nf4_smoke_02.json').read_text())['generations']
            else:
                from scripts.qwen38_eval_adapter import validate
                from safetensors.torch import load_file
                from peft import get_peft_model_state_dict,prepare_model_for_kbit_training,PeftModel
                if args.training_run is None:raise ValueError('original selected training run required')
                training=ROOT/args.training_run;selected=json.loads((training/'result.json').read_text())['selected']
                training_config=json.loads((training/'config.json').read_text())
                if training_config['nf4_artifact_manifest_sha256']!=sha(ROOT/'results/qwen38_27b_nf4_artifact_01.json'):
                    raise ValueError('training NF4 identity differs')
                original=Path(selected['path']);validate(original,args.kind,'base')
                if sha(snapshot/'adapter_model.safetensors')!=selected['adapter_files']['adapter_model.safetensors']['sha256']:
                    raise ValueError('published adapter weights differ from selected checkpoint')
                if sha(downloaded['base_publication'])!=downloaded['base_publication_sha256']:
                    raise ValueError('base publication changed')
                base_record=publication(Path(downloaded['base_publication']),'nf4')
                validate_artifact(base_record,artifact_files('nf4'))
                base=Path(downloaded['base_snapshot'])
                for name,expected in base_record['files'].items():
                    if sha(base/name)!=expected['sha256']:raise ValueError('published dependency changed')
                expected=json.loads((original/'adapter_config.json').read_text())
                expected.update(base_model_name_or_path=REPOS['nf4'],revision=downloaded['base_revision'])
                if json.loads((snapshot/'adapter_config.json').read_text())!=expected:
                    raise ValueError('portable adapter config differs beyond declared base/revision')
                base_model,info=Qwen3_5ForCausalLM.from_pretrained(base,local_files_only=True,
                    trust_remote_code=False,dtype=torch.bfloat16,device_map={'':'cuda:0'},
                    attn_implementation='sdpa',output_loading_info=True)
                validate_loading(info)
                base_model=prepare_model_for_kbit_training(base_model,use_gradient_checkpointing=True,
                    gradient_checkpointing_kwargs={'use_reentrant':False})
                base_model.config.use_cache=False
                model=PeftModel.from_pretrained(base_model,snapshot,is_trainable=False,
                    local_files_only=True,autocast_adapter_dtype=True)
                state=load_file(snapshot/'adapter_model.safetensors')
                actual=get_peft_model_state_dict(model)
                if set(actual)!=set(state) or any(not torch.equal(actual[k].detach().cpu(),state[k]) for k in state):
                    raise ValueError('published selected-adapter tensor reload differs')
                result['adapter_tensor_equality']=True
                result['public_route']='PeftModel.from_pretrained after identical FP32-casting PEFT preparation'
                tokenizer=AutoTokenizer.from_pretrained(base,local_files_only=True)
                reference=json.loads((ROOT/'results/qwen38_27b_nf4_smoke_02.json').read_text())['generations']
            model.eval().requires_grad_(False);predictions=[]
            with torch.inference_mode(),(nullcontext() if args.kind=='nf4' else torch.autocast('cuda',dtype=torch.bfloat16)):
                for probe in reference:
                    prompt=tokenizer.apply_chat_template([{'role':'user','content':probe['prompt']}],tokenize=False,
                        add_generation_prompt=True,enable_thinking=False)
                    ids=tokenizer(prompt,add_special_tokens=False)['input_ids']
                    if hashlib.sha256(prompt.encode()).hexdigest()!=probe['rendered_sha256']:
                        raise ValueError('smoke rendered prompt differs')
                    batch=torch.tensor([ids],device='cuda:0')
                    inputs={'input_ids':batch,'attention_mask':torch.ones_like(batch)}
                    sequence=native_replay(model,inputs) if args.kind=='nf4' else model.generate(**inputs,do_sample=False,
                        max_new_tokens=64,use_cache=True,logits_to_keep=1,temperature=None,top_p=None,top_k=None,
                        pad_token_id=tokenizer.pad_token_id,eos_token_id=tokenizer.eos_token_id)
                    generated=sequence[0,len(ids):].tolist()
                    if not generated:raise ValueError('empty smoke response')
                    predictions.append({'input_ids':ids,'generated_ids':generated,'output':tokenizer.decode(generated,skip_special_tokens=True)})
            result['predictions']=predictions
            if args.kind=='nf4':
                result['exact_smoke_token_replay']=len(predictions)==len(reference)==2 and all(a['input_ids']==b['input_ids'] and a['generated_ids']==b['generated_ids'] for a,b in zip(predictions,reference))
                if not result['exact_smoke_token_replay']:raise ValueError('downloaded NF4 smoke tokens differ')
            result['requantization_supplied']=False
        result['completed']=True
    except Exception as error:
        result['error']=f'{type(error).__name__}: {error}'
        raise
    finally:
        result['elapsed_seconds']=time.monotonic()-started;write(out/'result.json',result)
    return {k:v for k,v in result.items() if k!='predictions'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--download',action='store_true')
    p.add_argument('--execute-gpu',action='store_true')
    p.add_argument('--kind',choices=list(REPOS),required=True)
    p.add_argument('--publication',type=Path)
    p.add_argument('--base-publication',type=Path)
    p.add_argument('--training-run',type=Path)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--gpu',default='0')
    a=p.parse_args()
    if a.download and a.publication:result=download(a)
    elif a.execute_gpu and a.gpu.isdecimal():result=execute(a)
    else:p.error('download requires publication, or choose explicit GPU reload')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
