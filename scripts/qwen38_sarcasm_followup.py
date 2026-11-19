"""One registered launcher: preflight/audit/mechanics/replay/doses/judge/analysis.

`prepare` is CPU only. `launch --approval FILE` is the sole execution command.
The parent enforces process-group wall deadlines, including model load. No
stage asks questions. Unknown starts never retry; completed rows hash-replay.
Old study sources, artifacts, configuration and outcomes are read-only.
"""
import argparse
from contextlib import contextmanager, nullcontext
from datetime import datetime,timezone
import fcntl
import gc
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

from scripts.common import ROOT,path,read_json
from scripts.qwen38_science_runtime import sha
from scripts.tone_judge import save,save_or_match

CONFIG='configs/qwen38_sarcasm_followup.json'
ALGORITHM='preflight,audit,mechanics,PartA,PartB,judging,analysis'
SOURCES=['scripts/qwen38_sarcasm_followup.py','scripts/qwen38_sarcasm_followup_telemetry.py',
         'scripts/qwen38_sarcasm_followup_analyze.py','tests/test_qwen38_sarcasm_followup.py',
         'tests/test_qwen38_sarcasm_followup_pipeline.py',
         'tests/test_qwen38_gcad_port.py',CONFIG,'scripts/qwen38_gcad_port_NOTICE','scripts/qwen3_gcad_port_LICENSE']


def versions():
    return {n:importlib.metadata.version(n) for n in ['torch','transformers','bitsandbytes','accelerate','safetensors','numpy','tokenizers']}


def prompt(tok,user,system=None):
    messages=([{'role':'system','content':system}] if system else [])+[{'role':'user','content':user}]
    text=tok.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
    ids=tok(text,add_special_tokens=False)['input_ids']
    span=None
    if system:
        end=text.index('<|im_end|>\n')+len('<|im_end|>\n')
        system_ids=tok(text[:end],add_special_tokens=False)['input_ids'];span=len(system_ids)
        if ids[:span]!=system_ids:raise ValueError('system prefix token mismatch')
    user_text=f'<|im_start|>user\n{user}<|im_end|>\n'
    start=text.index(user_text);prefix=tok(text[:start],add_special_tokens=False)['input_ids']
    user_ids=tok(user_text,add_special_tokens=False)['input_ids']
    if ids[len(prefix):len(prefix)+len(user_ids)]!=user_ids:raise ValueError('user-turn token boundaries differ')
    return {'prompt':text,'input_ids':ids,'system_length':span,'user_turn_ids':user_ids}


def build_inputs(cfg,tok):
    old=path(cfg['old_run']);source=read_json(old/'inputs.json')
    old_rows=read_json(old/'development/result.json')['rows']
    lookup={(r['question_id'],r['arm']):r for r in old_rows}
    contexts={'S':source['elicitation'][0]['system'],'L':source['elicitation'][1]['system'],'N':None}
    eos=read_json(path(cfg['model'])/'generation_config.json')['eos_token_id'];eos=[eos] if isinstance(eos,int) else eos
    from scripts.qwen38_sarcasm_followup_telemetry import replay_ids
    replays=[];old_files={}
    for qi in range(1,33):
        q=f'C{qi:02d}';base=lookup[q,'baseline'];rendered={k:prompt(tok,base['user'],v) for k,v in contexts.items()}
        if rendered['N']['input_ids']!=lookup[q,'baseline']['input_ids'] or rendered['S']['input_ids']!=lookup[q,'prompted']['input_ids']:
            raise ValueError('re-rendered N/S prompt IDs differ from old generating contexts')
        if len(rendered['S']['input_ids'])!=len(rendered['L']['input_ids']) or rendered['S']['system_length']!=rendered['L']['system_length']:
            raise ValueError('S/L prompt and system lengths must match exactly')
        if not all(r['user_turn_ids']==rendered['S']['user_turn_ids'] for r in rendered.values()):raise ValueError('user tokens differ')
        for content,arm in [('B','baseline'),('P','prompted')]:
            row=lookup[q,arm];raw=old/'development/rows'/f'{row["id"]}.json'
            receipt=raw.with_name(raw.stem+'.receipt.json');start=raw.with_name(raw.stem+'.start.json')
            rec=read_json(receipt)
            if sha(raw)!=rec['sha256'] or sha(start)!=rec['start_sha256']:raise ValueError('old row receipt differs')
            stored=read_json(raw)
            if any(stored[k]!=row[k] for k in ('input_ids','generated_ids','output','prompt')):raise ValueError('old aggregate/raw row differ')
            for p in [raw,receipt,start]:old_files[str(p)]=sha(p)
            if len(row['generated_ids'])!=160 or not row['hit_token_limit']:raise ValueError('old capped160 trajectory required')
            masks=[];suffixes=[]
            for context,r in rendered.items():
                ids,mask=replay_ids(tok,r['input_ids'],row['generated_ids'],row['output'],eos)
                masks.append(mask[len(r['input_ids']):]);suffixes.append(ids[len(r['input_ids']):])
                replays.append({'id':f'{q}_{content}_{context}','question_id':q,'content':content,'context':context,
                    'source_id':row['id'],'user':row['user'],**r,'full_ids':ids,'response_mask':mask,
                    'saved_generated_ids':row['generated_ids'],'saved_output':row['output']})
            if not all(m==masks[0] for m in masks) or not all(s==suffixes[0] for s in suffixes):raise ValueError('continuation not held fixed')
    fresh=read_json(cfg['questions'])
    if [r['id'] for r in fresh]!=[f'F{i:02d}' for i in range(1,33)]:raise ValueError('fixed fresh F01..F32 required')
    import unicodedata
    norm=lambda text:' '.join(unicodedata.normalize('NFKC',text).casefold().split())
    historical=[*read_json('evaluation/sarcasm_method_transfer/extract.json')['questions'],
        *[r['user'] for r in read_json('evaluation/author_profanity_revision/calibration.json')],
        *[r['user'] for r in read_json('evaluation/author_profanity_revision/report.json')]]
    if len({norm(r['user']) for r in fresh})!=32 or {norm(r['user']) for r in fresh}&set(map(norm,historical)):
        raise ValueError('fresh fixture duplicate or normalized historical overlap')
    generation=[{'id':r['id']+'_'+arm,'question_id':r['id'],'user':r['user'],'arm':arm,
                 **prompt(tok,r['user'],contexts['S'] if arm=='prompted' else None)} for r in fresh for arm in cfg['arms']]
    probes=[{**r,**prompt(tok,r['user'])} for r in source['probes']]
    if any(len(r['full_ids'])>cfg['maximum_sequence_tokens'] for r in replays) or any(len(r['input_ids'])+160>cfg['maximum_sequence_tokens'] for r in generation):
        raise ValueError('sequence guard exceeded; no silent truncation')
    geometry=read_json(path(cfg['model'])/'config.json')
    return {'replays':replays,'generations':generation,'probes':probes,'contexts':contexts,
        'model_geometry':{k:geometry[k] for k in ['layer_types','num_hidden_layers','hidden_size','head_dim']},
        'eos_ids':eos,'literal_sarcastic_system_lengths':sorted({r['system_length'] for r in replays if r['context']!='N'})},old_files


def prepare():
    cfg=read_json(CONFIG);run=path(cfg['run'])
    if run.exists():raise FileExistsError('fresh follow-up preparation only')
    if cfg['arms']!=['baseline','prompted','b2','b4','b8','b16','random16'] or cfg['maximum_generations']!=232 or cfg['maximum_teacher_forced']!=194:
        raise ValueError('PM +random16 variant required')
    from transformers import AutoTokenizer
    from scripts.author_gcad_impolite import closure
    from scripts.persona_reference_common import make_plan
    tok=AutoTokenizer.from_pretrained(path(cfg['model']),local_files_only=True)
    inputs,old_files=build_inputs(cfg,tok)
    plans={m:make_plan([f'F{i:02d}' for i in range(1,33)],cfg['arms'],cfg['plan_seed']+i) for i,m in enumerate(['trait','coherence','impolite'])}
    old=path(cfg['old_run']);bank=old/'extraction/bank.safetensors'
    if sha(bank)!=cfg['bank_sha256']:raise ValueError('fixed old bank differs')
    sources=closure(SOURCES)
    names=[*sources,cfg['protocol'],cfg['questions'],cfg['question_provenance'],cfg['model_manifest'],cfg['old_result'],
           str(old/'manifest.json'),str(old/'inputs.json'),str(old/'development/result.json'),str(old/'rubrics.json'),str(bank),
           cfg['judge_source'],*[str(old/f'random-{seed}.safetensors') for seed in [9831,9832,9833]],
           'evaluation/sarcasm_method_transfer/extract.json','evaluation/author_profanity_revision/calibration.json','evaluation/author_profanity_revision/report.json']
    installed=path(cfg['runtime']).parent.parent/'lib/python3.13/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py'
    names.append(str(installed))
    names.extend(str(p.relative_to(ROOT)) for p in sorted((ROOT/'results/reviews').glob('qwen38_sarcasm_followup_review_*.md')))
    correction=ROOT/'results/reviews/qwen38_sarcasm_norm_interpretation_correction_01.md'
    if correction.is_file():names.append(str(correction.relative_to(ROOT)))
    hashes={str(n):sha(path(n)) for n in names};hashes.update(old_files)
    manifest=read_json(cfg['model_manifest'])
    for name,r in manifest['files'].items():
        p=path(cfg['model'])/name
        if p.stat().st_size!=r['bytes'] or sha(p)!=r['sha256']:raise ValueError('saved NF4 model differs')
    if sha(path(cfg['judge_source']))!=cfg['judge_sha256']:raise ValueError('judge source snapshot differs')
    run.mkdir(parents=True)
    save(run/'config.json',cfg);save(run/'inputs.json',inputs);save(run/'plans.json',plans)
    shutil.copyfile(old/'rubrics.json',run/'rubrics.json');shutil.copyfile(bank,run/'bank.safetensors')
    for seed in [9831,9832,9833]:shutil.copyfile(old/f'random-{seed}.safetensors',run/f'random-{seed}.safetensors')
    (run/'bin').mkdir();binary=run/'bin/claude-2.1.327';shutil.copyfile(path(cfg['judge_source']),binary);binary.chmod(0o500)
    for name in sources:
        out=run/'source'/name;out.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path(name),out)
    save(run/'manifest.json',{'source_and_external_sha256':hashes,'packages':versions(),'model_files':manifest['files'],
        'frozen':{str(p.relative_to(run)):sha(p) for p in run.rglob('*') if p.is_file()},'algorithm':ALGORITHM,'no_calls':True})
    print(json.dumps({'run':str(run),'manifest_sha256':sha(run/'manifest.json'),'algorithm':ALGORITHM},indent=2))


class BudgetStop(RuntimeError):pass
class Uncertain(RuntimeError):pass


class Context:
    def __init__(self,approval):
        self.cfg=read_json(CONFIG);self.run=path(self.cfg['run']);self.manifest=read_json(self.run/'manifest.json')
        self.approval=Path(approval).resolve();record=read_json(self.approval)
        expected={'manifest_sha256':sha(self.run/'manifest.json'),'launcher_sha256':sha(Path(__file__)),
                  'algorithm':ALGORITHM,'gpu_uuid':self.cfg['gpu_uuid']}
        if any(record.get(k)!=v for k,v in expected.items()):raise ValueError('exact prospective end-to-end approval required')
        if self.cfg!=read_json(self.run/'config.json') or versions()!=self.manifest['packages']:raise ValueError('configuration/runtime changed')
        for name,h in self.manifest['source_and_external_sha256'].items():
            if sha(path(name))!=h:raise ValueError(f'bound source/input changed: {name}')
        for name,h in self.manifest['frozen'].items():
            if sha(self.run/name)!=h:raise ValueError(f'frozen copy changed: {name}')
        self.inputs=read_json(self.run/'inputs.json');self.deadline=None

    @contextmanager
    def lock(self):
        with (self.run/'.launcher.lock').open('a') as f:
            fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
            yield

    def task(self,stage,key,call):
        root=self.run/stage/'rows';root.mkdir(parents=True,exist_ok=True)
        start=root/f'{key}.start.json';result=root/f'{key}.json';receipt=root/f'{key}.receipt.json'
        binding={'manifest_sha256':sha(self.run/'manifest.json'),'stage':stage,'key':key}
        if result.exists():
            value=read_json(result)
            if value['binding']!=binding or read_json(start)['binding']!=binding:raise ValueError('row binding differs')
            save_or_match(receipt,{'sha256':sha(result),'start_sha256':sha(start)})
            return value['value']
        if start.exists() or receipt.exists():raise Uncertain(f'unknown or failed {stage}/{key}: never retry')
        save(start,{'binding':binding,'started_unix':time.time()});tick=time.monotonic()
        try:
            value=call();save(result,{'binding':binding,'value':value,'seconds':time.monotonic()-tick})
            save(receipt,{'sha256':sha(result),'start_sha256':sha(start)});return value
        except BaseException as exc:
            extra={k:getattr(exc,k) for k in ('stdout','stderr') if isinstance(getattr(exc,k,None),str)}
            save(root/f'{key}.failure.json',{'error_type':type(exc).__name__,'error':str(exc),'seconds':time.monotonic()-tick,'no_retry':True,**extra})
            raise

    def reserve(self,kind,stage,key,tokens=0):
        limits={'generation':self.cfg['maximum_generations'],'teacher_forced':self.cfg['maximum_teacher_forced'],
                'logit':self.cfg['maximum_logit_forwards'],'judge':self.cfg['maximum_judge_starts']}
        caps={'generation':{'mechanics':8,'generation':224},'teacher_forced':{'mechanics':2,'diagnostic':192},
              'logit':{'mechanics':5},'judge':{'audit':1,'judge':168}}
        root=self.run/'counters'/kind;root.mkdir(parents=True,exist_ok=True)
        records=[read_json(p) for p in root.glob('*.json')]
        if len(records)>=limits[kind] or stage not in caps[kind] or sum(v['stage']==stage for v in records)>=caps[kind][stage]:raise BudgetStop('call count cap')
        if kind=='generation' and sum(v['tokens'] for v in records)+tokens>self.cfg['maximum_generated_tokens']:raise BudgetStop('token reservation cap')
        save(root/f'{len(records):04d}.json',{'stage':stage,'key':key,'tokens':tokens,'reserved_unix':time.time()})

    def stage_budget(self,seconds):
        if self.deadline is None or time.time()+seconds>self.deadline:raise BudgetStop('remaining GPU time cannot cover next complete stage')


def load(s):
    os.environ['CUDA_VISIBLE_DEVICES']=s.cfg['gpu_uuid'];os.environ['TOKENIZERS_PARALLELISM']='false'
    import torch
    from transformers import AutoTokenizer,Qwen3_5ForCausalLM
    from scripts.qwen38_nf4_smoke import validate_loading
    from scripts.qwen38_gcad_port import validate_geometry
    from safetensors.torch import load_file
    for n,r in s.manifest['model_files'].items():
        if sha(path(s.cfg['model'])/n)!=r['sha256']:raise ValueError('NF4 model bytes changed')
    model,info=Qwen3_5ForCausalLM.from_pretrained(path(s.cfg['model']),local_files_only=True,trust_remote_code=False,
        dtype=torch.bfloat16,device_map={'':'cuda:0'},attn_implementation='sdpa',output_loading_info=True)
    validate_loading(info);model.eval();model.config.use_cache=True;validate_geometry(model,s.cfg['layers'])
    bank=load_file(str(s.run/'bank.safetensors'));random=load_file(str(s.run/'random-9831.safetensors'))
    if not all(torch.isfinite(v).all() for v in bank.values()):raise ValueError('nonfinite frozen bank')
    for i in s.cfg['layers']:
        if not torch.allclose(random[f'{i}.delta'].float().norm(),bank[f'{i}.delta'].float().norm(),rtol=1e-5,atol=0):
            raise ValueError('random16 delta norm differs from bank')
        if not torch.equal(random[f'{i}.key'],bank[f'{i}.key']) or not torch.equal(random[f'{i}.demand'],bank[f'{i}.demand']):
            raise ValueError('random16 key/demand differ from bank')
    return model,AutoTokenizer.from_pretrained(path(s.cfg['model']),local_files_only=True),bank,random


def generate(s,model,tok,row,bank,coef,stage,observer='instrument'):
    import torch
    from transformers import GenerationConfig,set_seed
    from scripts.qwen38_sarcasm_followup_telemetry import InstrumentedSteerer,signature
    from scripts.qwen38_gcad_port import Steerer
    set_seed(s.cfg['seed']);cap=s.cfg['mechanics_cap'] if stage=='mechanics' else s.cfg['target_cap']
    if len(row['input_ids'])<=1 or tok(row['prompt'],add_special_tokens=False)['input_ids']!=row['input_ids']:raise ValueError('frozen prompt IDs differ')
    ctx=(InstrumentedSteerer if observer=='instrument' else Steerer)(model,bank,s.cfg['layers'],coef) if coef is not None else nullcontext()
    defaults=GenerationConfig.from_pretrained(path(s.cfg['model']),local_files_only=True)
    defaults.update(do_sample=False,repetition_penalty=1.05,max_new_tokens=cap,use_cache=True,num_beams=1,
        temperature=None,top_p=None,top_k=None,pad_token_id=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id)
    ids=torch.tensor([row['input_ids']],device=model.device);before=signature(model);tick=time.monotonic()
    with torch.inference_mode(),ctx:
        during=signature(model)
        for name,old in before.items():
            if name in during and during[name]!=old and name.startswith('model.layers.'):
                site=int(name.split('.')[2])
                if site not in s.cfg['layers'] or '.self_attn' not in name:raise ValueError('hook outside selected full attention')
        s.reserve('generation',stage,row['id'],cap)
        output=model.generate(input_ids=ids,attention_mask=torch.ones_like(ids),generation_config=defaults,logits_to_keep=1)
    if signature(model)!=before:raise ValueError('generation hook cleanup failed')
    suffix=output[0,ids.shape[1]:].tolist();eos=defaults.eos_token_id;eos=[eos] if isinstance(eos,int) else eos
    ended=bool(suffix and suffix[-1] in eos);audit=ctx.audit if coef is not None else {}
    if any(v!={'prefill':1,'decode':len(suffix)-1,'errors':0} for v in audit.values()):raise ValueError('cache/site coverage differs')
    telemetry=ctx.summary() if coef is not None and observer=='instrument' else None
    if telemetry:
        for phases in telemetry.values():
            for phase in phases.values():
                if not phase['calls']:continue
                coefficient=phase['metrics']['coefficient']
                if coefficient['min']<0 or coefficient['max']>2*coef:raise ValueError('coefficient outside registered range')
                if coef==0 and (phase['metrics']['requested_norm']['sum']!=0 or phase['metrics']['actual_norm']['sum']!=0):raise ValueError('zero shift telemetry nonzero')
    return {**row,'generated_ids':suffix,'output':tok.decode(suffix,skip_special_tokens=True),
        'ended_eos':ended,'hit_token_limit':len(suffix)==cap and not ended,'audit':audit,'telemetry':telemetry,
        'generation_seconds':time.monotonic()-tick,'cache_path':'native cached generate; batch1'}


def mechanics(s,model,tok,bank):
    import torch
    from scripts.qwen38_gcad_port import Steerer
    from scripts.qwen38_sarcasm_followup_telemetry import InstrumentedSteerer,diagnostic_forward,signature
    def execute():
        records=[]
        for probe in s.inputs['probes']:
            variants={}
            for name,b,observer in [('ordinary',None,'instrument'),('zero',0.,'instrument'),('frozen2',2.,'frozen'),('record2',2.,'instrument')]:
                variants[name]=generate(s,model,tok,{**probe,'id':probe['id']+'_'+name},bank,b,'mechanics',observer)
                save(s.run/'mechanics'/f'{probe["id"]}_{name}.json',variants[name])
            if variants['ordinary']['generated_ids']!=variants['zero']['generated_ids'] or variants['frozen2']['generated_ids']!=variants['record2']['generated_ids']:raise ValueError('instrumented cached token parity failed')
            records.append(variants)
        ids=torch.tensor([s.inputs['probes'][0]['input_ids']],device=model.device);values=[];before=signature(model)
        for name,ctx in [('ordinary',nullcontext()),('zero',InstrumentedSteerer(model,bank,s.cfg['layers'],0.)),
            ('frozen2',Steerer(model,bank,s.cfg['layers'],2.)),('record2',InstrumentedSteerer(model,bank,s.cfg['layers'],2.,retain_last=True)),('cleanup',nullcontext())]:
            s.reserve('logit','mechanics',name)
            with torch.inference_mode(),ctx:
                values.append(model(ids,attention_mask=torch.ones_like(ids),use_cache=True,logits_to_keep=1).logits.detach().clone())
                if name=='record2':
                    for i,last in ctx.last.items():
                        manual_requested=last['requested'].float().norm(dim=-1).flatten()
                        manual_actual=(last['steered'].float()-last['original'].float()).norm(dim=-1).flatten()
                        recorded=ctx.trace[i][last['phase']][-1]
                        if not torch.allclose(manual_requested,recorded['requested_norm'],rtol=1e-5,atol=0) or not torch.allclose(manual_actual,recorded['actual_norm'],rtol=1e-5,atol=0):
                            raise ValueError('manual requested/actual telemetry norm mismatch')
        if not torch.equal(values[0],values[1]) or not torch.equal(values[0],values[4]) or not torch.equal(values[2],values[3]) or torch.equal(values[0],values[2]) or signature(model)!=before:raise ValueError('exact/nonzero logit parity or cleanup failed')
        cell=next(r for r in s.inputs['replays'] if r['id']=='C01_P_S')
        ids=torch.tensor([cell['full_ids']],device=model.device);mask=torch.tensor(cell['response_mask'],device=model.device)
        positions=torch.where(mask)[0]-1
        s.reserve('teacher_forced','mechanics','capture-off')
        with torch.inference_mode():ordinary=model(ids,attention_mask=torch.ones_like(ids),use_cache=False,logits_to_keep=positions).logits.detach().clone()
        s.reserve('teacher_forced','mechanics','capture-on')
        diagnostics,observed=diagnostic_forward(model,ids,cell['response_mask'],cell['system_length'],bank,s.cfg['layers'])
        if not torch.equal(ordinary,observed):raise ValueError('activation capture changed logits')
        for d in diagnostics['decomposition'].values():
            if not(d['gated_relative_error']<=.02 and d['omitted_gate_relative_error']>=.2 and
                d['omitted_gate_relative_error']>=10*d['gated_relative_error'] and d['split_relative_error']<=.01):raise ValueError('cropped decomposition mechanics failure')
        return {'passed':True,'records':records,'decomposition':diagnostics['decomposition'],'capture_neutral':True,
                'manual_requested_actual_norm_check':True,'ordinary_differs_from_b2':True}
    result=s.task('mechanics','phase',execute);save_or_match(s.run/'mechanics/result.json',result)


def gpu_worker(s,deadline):
    import torch
    from safetensors.torch import save_file
    from scripts.qwen38_sarcasm_followup_telemetry import diagnostic_forward
    s.deadline=deadline;model,tok,bank,random=load(s)
    mechanics(s,model,tok,bank)
    pending=sum(not (s.run/'diagnostic/rows'/f'{r["id"]}.json').exists() for r in s.inputs['replays'])
    if pending:s.stage_budget(s.cfg['stage_A_worst_case_seconds']*pending/192)
    diagnostics=[]
    for row in s.inputs['replays']:
        def execute(row=row):
            s.reserve('teacher_forced','diagnostic',row['id'])
            ids=torch.tensor([row['full_ids']],device=model.device)
            value,_=diagnostic_forward(model,ids,row['response_mask'],row['system_length'],bank,s.cfg['layers'])
            f=s.run/'diagnostic'/f'{row["id"]}.safetensors';save_file(value.pop('vectors'),str(f))
            size=sum(p.stat().st_size for p in (s.run/'diagnostic').glob('*.safetensors'))
            if size>s.cfg['maximum_activation_bytes']:raise BudgetStop('activation storage cap')
            return {'id':row['id'],'question_id':row['question_id'],'context':row['context'],'content':row['content'],
                'tensor_file':str(f.relative_to(s.run)),'tensor_sha256':sha(f),**value}
        diagnostics.append(s.task('diagnostic',row['id'],execute))
    save_or_match(s.run/'diagnostic/result.json',{'completed':True,'rows':diagnostics,
        'activation_bytes':sum(p.stat().st_size for p in (s.run/'diagnostic').glob('*.safetensors'))})
    pending=sum(not (s.run/'generation/rows'/f'{r["id"]}.json').exists() for r in s.inputs['generations'])
    if pending:s.stage_budget(pending*s.cfg['target_cap']/s.cfg['conservative_tokens_per_second']+s.cfg['model_load_allowance_seconds'])
    results=[]
    for row in s.inputs['generations']:
        arm=row['arm'];coef=float(arm[1:]) if arm.startswith('b') and arm!='baseline' else (16. if arm=='random16' else None)
        active=random if arm=='random16' else bank
        results.append(s.task('generation',row['id'],lambda row=row,active=active,coef=coef:generate(s,model,tok,row,active,coef,'generation')))
    save_or_match(s.run/'generation/result.json',{'completed':True,'rows':results})


def judge_identity(s):
    from scripts import tone_judge as native
    binary=s.run/'bin/claude-2.1.327'
    if sha(binary)!=s.cfg['judge_sha256']:raise ValueError('judge snapshot changed')
    version=subprocess.check_output([str(binary),'--version'],text=True,timeout=30).strip()
    if version!=s.cfg['judge_version']:raise ValueError('judge version changed')
    return {'binary':str(binary),'sha256':sha(binary),'version':version,'argv':native.argv(str(binary))}


def judge_call(s,payload,stage,key):
    from scripts import tone_judge as native
    before=judge_identity(s);s.reserve('judge',stage,key)
    attempt=native.invoke(before['binary'],payload)
    if judge_identity(s)!=before:raise ValueError('judge identity changed during invocation')
    return {'attempt':attempt,'payload_sha256':native.canonical_hash(payload),'identity':before}


def audit(s):
    from scripts import tone_judge as native
    payload={'task':'Context isolation check only. No external information is supplied.',
             'question':"What is the name of the user's current project? If it is not in this JSON, return UNKNOWN.",
             'return_schema':{'project':'string'},'required_output':'JSON only'}
    response=s.task('audit','isolation',lambda:judge_call(s,payload,'audit','isolation'))
    outer,models=native.outer_result(response['attempt']);value=native.result_json(outer.get('result'))
    if value!={'project':'UNKNOWN'}:raise ValueError('isolated judge audit failed')
    save_or_match(s.run/'audit/result.json',{'passed':True,'models':models,'identity':response['identity']})


def judging(s):
    from scripts.qwen38_sarcasm_judge import parse
    from scripts import tone_judge as native
    rows=read_json(s.run/'generation/result.json')['rows'];lookup={(r['question_id'],r['arm']):r for r in rows}
    plans=read_json(s.run/'plans.json');rubrics=read_json(s.run/'rubrics.json');scores={r['id']:{} for r in rows}
    for metric,batches in plans.items():
        for bi,batch in enumerate(batches):
            payload={'rubric':rubrics[metric],'items':[{'id':r['id'],'question':lookup[r['question_id'],r['condition']]['user'],
                'answer':lookup[r['question_id'],r['condition']]['output']} for r in batch],
                'return_schema':{'ratings':[{'id':'exact opaque item id','score':'integer0..100'+(' or REFUSAL' if metric=='trait' else '')}]},
                'required_output':'JSON only, one complete rating per supplied item; no other fields'}
            for ai in range(2):
                key=f'{metric}-{bi:03d}-{ai}';response=s.task('judge',key,lambda key=key:judge_call(s,payload,'judge',key))
                if response['payload_sha256']!=native.canonical_hash(payload):raise ValueError('saved judge payload differs')
                try:accepted=parse(response['attempt'],[r['id'] for r in batch],metric);break
                except native.InvalidSubmission:
                    if ai==1:raise
            for entry in batch:scores[lookup[entry['question_id'],entry['condition']]['id']][metric]=accepted[entry['id']]
        save_or_match(s.run/'judge'/f'{metric}.json',{k:v[metric] for k,v in scores.items()})
    save_or_match(s.run/'judge/scores.json',scores)


def session_spent(run,kind):
    total=0.
    for start in (run/'sessions').glob(f'{kind}-*.start.json'):
        done=start.with_name(start.name.replace('.start.json','.done.json'))
        total+=read_json(done)['seconds'] if done.exists() else max(0,time.time()-read_json(start)['unix'])
    return total


def process_worker(s,kind,remaining,extra=None):
    root=s.run/'sessions';root.mkdir(exist_ok=True);name=f'{kind}-{time.time_ns()}'
    save(root/(name+'.start.json'),{'unix':time.time(),'remaining_seconds':remaining});tick=time.monotonic()
    args=[sys.executable,'-m','scripts.qwen38_sarcasm_followup',f'_{kind}', '--approval',str(s.approval)]
    if extra is not None:args+=['--deadline',str(extra)]
    with (root/(name+'.log')).open('x') as log:
        p=subprocess.Popen(args,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            code=p.wait(timeout=remaining)
            if code==75:raise BudgetStop(f'{kind} worker stage-entry budget stop; inspect {name}.log')
            if code:raise RuntimeError(f'{kind} worker exited {code}; inspect {name}.log')
        except BaseException:
            if p.poll() is None:
                os.killpg(p.pid,signal.SIGTERM)
                try:p.wait(timeout=5)
                except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
            raise
        finally:save(root/(name+'.done.json'),{'seconds':time.monotonic()-tick,'returncode':p.returncode})


def preflight_gpu(s):
    raw=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True,timeout=30)
    found=[line.split(',') for line in raw.splitlines() if line.split(',')[0].strip()==s.cfg['gpu_uuid']]
    if len(found)!=1 or int(found[0][1])<s.cfg['minimum_free_mib'] or int(found[0][2])>s.cfg['maximum_gpu_utilization_percent']:
        raise RuntimeError('allocated GPU unavailable/busy; no fallback GPU')


def launch(s):
    from scripts.qwen38_sarcasm_followup_analyze import analyze
    if (s.run/'closed.json').exists():
        print(json.dumps(read_json(s.run/'closed.json'),indent=2));return
    if not (s.run/'generation/result.json').exists():preflight_gpu(s)
    root=s.run/'sessions';root.mkdir(exist_ok=True);tick=time.monotonic()
    remaining=s.cfg['max_wall_seconds']-session_spent(s.run,'launcher')
    if remaining<=0:raise BudgetStop('end-to-end wall cap')
    name=f'launcher-{time.time_ns()}';save(root/(name+'.start.json'),{'unix':time.time()})
    status='completed';error=None;phase='audit'
    try:
        try:process_worker(s,'audit',max(1,remaining-(time.monotonic()-tick)))
        except Exception as exc:
            save_or_match(s.run/'audit/failure.json',{'error':str(exc),'type':type(exc).__name__,'skip_judging':True})
        if not (s.run/'generation/result.json').exists():
            phase='gpu'
            preflight_gpu(s)
            gpu_remaining=s.cfg['max_gpu_seconds']-session_spent(s.run,'gpu')
            remaining_wall=remaining-(time.monotonic()-tick)
            if min(gpu_remaining,remaining_wall)<=0:raise BudgetStop('GPU/wall cap')
            process_worker(s,'gpu',min(gpu_remaining,remaining_wall),time.time()+min(gpu_remaining,remaining_wall))
        if (s.run/'audit/result.json').exists():
            phase='judge'
            left=remaining-(time.monotonic()-tick)
            if left<=0:raise BudgetStop('no remaining wall time for judging')
            process_worker(s,'judge',left)
        else:status='judge_audit_failure'
    except BudgetStop as exc:status='budget_stop';error=str(exc)
    except subprocess.TimeoutExpired as exc:status='budget_stop';error=str(exc)
    except BaseException as exc:status='judge_failure' if phase=='judge' else 'implementation_stop';error=f'{type(exc).__name__}: {exc}'
    finally:save(root/(name+'.done.json'),{'seconds':time.monotonic()-tick,'status':status,'error':error})
    result=analyze(s.run,status,error)
    save_or_match(path(s.cfg['result']),result)
    closed={'status':result['outcome'],'result':s.cfg['result'],'result_sha256':sha(path(s.cfg['result']))}
    save(s.run/'closed.json',closed);print(json.dumps(closed,indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['prepare','launch','_gpu','_judge','_audit'])
    p.add_argument('--approval',type=Path);p.add_argument('--deadline',type=float);args=p.parse_args()
    if args.command=='prepare':prepare();return
    if not args.approval:p.error('--approval is required')
    s=Context(args.approval)
    if args.command=='_gpu':
        if args.deadline is None:p.error('parent-supplied --deadline required')
        try:gpu_worker(s,args.deadline)
        except BudgetStop as exc:
            print(f'budget_stop: {exc}',file=sys.stderr);raise SystemExit(75)
    elif args.command=='_judge':judging(s)
    elif args.command=='_audit':audit(s)
    else:
        def stop_requested(signum,frame):raise KeyboardInterrupt('launcher termination requested')
        signal.signal(signal.SIGTERM,stop_requested)
        with s.lock():launch(s)


if __name__=='__main__':main()
