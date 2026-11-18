"""Fixed development-only cross-task assistant NLL; descriptive, no selection."""
import argparse
from contextlib import nullcontext
import json
import os
from pathlib import Path
import shutil
import time

from scripts.qwen38_science_runtime import BASE, sha,write,versions
from scripts.qwen38_train_science import paper_subset

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'outputs/qwen38-transfer-01'
REGISTRY=ROOT/'results/protocols/qwen38-transfer-01.json'
SOURCES=['qwen38_transfer.py','qwen38_science_runtime.py','qwen38_nf4_smoke.py',
         'qwen38_eval_adapter.py','qwen38_eval_science.py','training_data.py','qwen38_train_science.py','common.py']


def rows(path):return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def prepare():
    if RUN.exists() or REGISTRY.exists():raise FileExistsError('preserve prior registration')
    from transformers import AutoTokenizer
    from scripts.training_data import encode_row
    tokenizer=AutoTokenizer.from_pretrained(BASE,local_files_only=True)
    qasper=ROOT/'outputs/qwen38-science-01/prepared-qasper-tokenizer-01'
    prl=ROOT/'outputs/qwen38-science-01/prepared-prl-audited-01'
    selected={};inputs={}
    for task,directory in [('qasper',qasper),('prl',prl)]:
        source=directory/'validation.jsonl';manifest=json.loads((directory/'manifest.json').read_text())
        if sha(source)!=manifest['output_sha256'][source.name]:raise ValueError('development input changed')
        eligible=[r for r in rows(source) if r['training_eligible']]
        selected[task]=paper_subset(eligible,32,20261117) if task=='qasper' else eligible
        inputs[task]={'prepared_manifest_sha256':sha(directory/'manifest.json'),'input_sha256':sha(source)}
    RUN.mkdir();(RUN/'source').mkdir()
    for task,examples in selected.items():
        target=RUN/f'{task}.jsonl'
        target.write_text(''.join(json.dumps({'id':r['id'],'paper_id':r['paper_id'],**encode_row(r,tokenizer,8192)})+'\n' for r in examples))
    for name in SOURCES:shutil.copyfile(ROOT/'scripts'/name,RUN/'source'/name)
    protocol={'scope':'development only; assistant/EOS NLL transfer diagnostic, no final examples or selection',
        'arms':['base','qasper','prl'],'counts':{t:len(r) for t,r in selected.items()},
        'ids':{t:[r['id'] for r in examples] for t,examples in selected.items()},
        'input_sha256':{t:sha(RUN/f'{t}.jsonl') for t in selected},'parent_inputs':inputs,
        'source_sha256':{n:sha(RUN/'source'/n) for n in SOURCES},'versions':versions(),
        'nf4_manifest_sha256':sha(ROOT/'results/qwen38_27b_nf4_artifact_01.json'),
        'max_seconds_per_arm':450,'seed':42,'max_length':8192,'weighting':'token-weighted assistant/EOS NLL',
        'base_preparation':'same rank16 PEFT prep/casts; adapter disabled for baseline',
        'cache':False,'interpretation':'NLL transfer is descriptive and does not establish scientific summary or QA quality',
        'timing':'registered before either adapter optimization; QASPER development selection already frozen'}
    write(RUN/'registration.json',protocol);write(REGISTRY,protocol)
    return {'prepared':str(RUN.relative_to(ROOT)),'counts':protocol['counts']}


def execute(arm,gpu,adapter):
    protocol=json.loads(REGISTRY.read_text())
    if REGISTRY.read_bytes()!=(RUN/'registration.json').read_bytes():raise ValueError('registration changed')
    for name,expected in protocol['source_sha256'].items():
        if sha(ROOT/'scripts'/name)!=expected:raise ValueError('transfer source changed')
    for task,expected in protocol['input_sha256'].items():
        if sha(RUN/f'{task}.jsonl')!=expected:raise ValueError('encoded probe changed')
    if versions()!=protocol['versions']:raise ValueError('runtime changed')
    manifest=ROOT/'results/qwen38_27b_nf4_artifact_01.json'
    if sha(manifest)!=protocol['nf4_manifest_sha256']:raise ValueError('NF4 manifest changed')
    for name,expected in json.loads(manifest.read_text())['files'].items():
        if sha(BASE/name)!=expected['sha256']:raise ValueError('NF4 artifact changed')
    if (arm=='base') != (adapter is None):raise ValueError('adapter required only for task arms')
    if adapter:
        from scripts.qwen38_eval_adapter import validate
        audit=validate(adapter,arm,'base')
    else:audit=None
    out=RUN/arm
    if out.exists():raise FileExistsError('arm already attempted')
    out.mkdir();write(out/'binding.json',{'registration_sha256':sha(REGISTRY),'adapter_audit':audit,'gpu':gpu})
    os.environ['CUDA_VISIBLE_DEVICES']=gpu
    import torch
    from transformers import set_seed
    from scripts.qwen38_science_runtime import load_runtime,answer_loss
    set_seed(42);started=time.monotonic();result={'completed':False,'records':[],'totals':{}}
    try:
        model,_=load_runtime()
        if adapter:
            from peft import get_peft_model_state_dict,set_peft_model_state_dict
            from safetensors.torch import load_file
            state=load_file(Path(adapter)/'adapter_model.safetensors')
            report=set_peft_model_state_dict(model,state,adapter_name='default')
            actual=get_peft_model_state_dict(model)
            if report.unexpected_keys or set(actual)!=set(state) or any(not torch.equal(actual[k].detach().cpu(),state[k]) for k in state):raise ValueError('adapter state mismatch')
        model.eval()
        with (nullcontext() if adapter else model.disable_adapter()), torch.inference_mode():
            for task in ['qasper','prl']:
                total,tokens=0.,0
                for row in rows(RUN/f'{task}.jsonl'):
                    if time.monotonic()-started>protocol['max_seconds_per_arm']:raise TimeoutError('fixed transfer cap')
                    batch={k:torch.tensor([row[k]],device='cuda:0') for k in ['input_ids','attention_mask','labels']}
                    with torch.autocast('cuda',dtype=torch.bfloat16):loss,n=answer_loss(model,**batch)
                    if not torch.isfinite(loss):raise ValueError('nonfinite probe NLL')
                    value=float(loss);total+=value*n;tokens+=n
                    result['records'].append({'task':task,'id':row['id'],'paper_id':row['paper_id'],'nll':value,'tokens':n})
                result['totals'][task]={'nll':total/tokens,'tokens':tokens}
                print(json.dumps({'arm':arm,'task':task,'records':len(result['records'])}),flush=True)
        result['completed']=True
    except Exception as error:
        result['error']=f'{type(error).__name__}: {error}'
        raise
    finally:
        result['elapsed_seconds']=time.monotonic()-started;write(out/'result.json',result)
    return {k:v for k,v in result.items() if k!='records'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepare',action='store_true')
    p.add_argument('--arm',choices=['base','qasper','prl'])
    p.add_argument('--gpu',default='0')
    p.add_argument('--adapter',type=Path)
    a=p.parse_args()
    if a.prepare:result=prepare()
    elif a.arm and a.gpu.isdecimal():result=execute(a.arm,a.gpu,a.adapter)
    else:p.error('choose prepare or an explicit arm/GPU')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
