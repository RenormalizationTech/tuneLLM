"""Bounded local NF4 study. Every external call requires exact PM registration.

Run with the isolated config runtime, via python -m scripts.qwen38_sarcasm.
Prepare is CPU-only. Started-but-unfinished rows are never retried on resume.
"""
import argparse
from contextlib import nullcontext, contextmanager
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time
import unicodedata

from scripts.common import ROOT, path, read_json
from scripts import tone_judge as journal
from scripts.qwen38_science_runtime import sha

CONFIG = 'configs/qwen38_sarcasm.json'
SOURCE = ['scripts/qwen38_sarcasm.py', 'scripts/qwen38_gcad_port.py',
          'scripts/qwen38_sarcasm_judge.py', 'scripts/qwen38_gcad_port_NOTICE',
          'scripts/qwen3_gcad_port_LICENSE', 'tests/test_qwen38_gcad_port.py',
          'tests/test_qwen38_sarcasm.py', CONFIG]


def save(p, value):
    journal.save(p, value)


def match(p, value):
    journal.save_or_match(p, value)


def normalize(s):
    return ' '.join(unicodedata.normalize('NFKC', s).casefold().split())


def render(tok, row):
    messages = ([{'role':'system','content':row['system']}] if row.get('system') else [])
    return tok.apply_chat_template(messages+[{'role':'user','content':row['user']}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)


def rows(cfg):
    data=read_json(cfg['extraction'])
    if len(data['questions'])!=20 or len(data['instruction'])!=5:
        raise ValueError('exact historical extraction fixture required')
    elicit=[]
    for qi, question in enumerate(data['questions']):
        for ii, instruction in enumerate(data['instruction']):
            for side,adjective in [('pos','helpful sarcastic'),('neg','helpful sincere')]:
                elicit.append(dict(id=f'x{qi:02d}_i{ii}_{side}',pair=f'x{qi:02d}_i{ii}',side=side,
                    question_id=f'x{qi:02d}',condition=f'i{ii}_{side}',instruction_index=ii,
                    user=question,system=f'You are a {adjective} assistant. {instruction[side]}'))
    dev,final=read_json(cfg['development']),read_json(cfg['final'])
    groups=[data['questions'],[r['user'] for r in dev],[r['user'] for r in final]]
    norm=[set(map(normalize,g)) for g in groups]
    if list(map(len,norm))!=[20,32,32] or any(norm[i]&norm[j] for i in range(3) for j in range(i)):
        raise ValueError('normalized overlap/duplicates: PM review required')
    positive=elicit[0]['system']
    evaluation=lambda source,arms:[dict(id=f'{r["id"]}_{arm}',question_id=r['id'],user=r['user'],arm=arm,
        system=positive if arm=='prompted' else '') for r in source for arm in arms]
    return {'elicitation':elicit,'development':evaluation(dev,['baseline','prompted','gcad_1','gcad_2']),
            'final':evaluation(final,['baseline','prompted','selected','random']),
            'probes':read_json(cfg['mechanics'])[:2], 'normalized_overlap_counts':[0,0,0]}


def packages():
    return {n:importlib.metadata.version(n) for n in ['torch','transformers','bitsandbytes','accelerate','safetensors','tokenizers']}


def rubrics():
    from scripts.author_gcad_impolite_common import coherence_definition
    author='outputs/persona-author-replication-audit-01/gcad'
    return {'trait':read_json('evaluation/sarcasm_method_transfer/extract.json')['eval_prompt'],
            'coherence':coherence_definition(path(author+'/eval/prompts.py')),
            'impolite':read_json(author+'/data/trait_data_eval/impolite.json')['eval_prompt']}


def prepare():
    cfg=read_json(CONFIG); run=path(cfg['run'])
    if run.exists(): raise FileExistsError('fresh R0 root required')
    if cfg['layers']!=[23,27,31,35,39,43] or cfg['strengths']!=[1.,2.] or cfg['maximum_generations']!=468:
        raise ValueError('prospective design settings differ')
    data=rows(cfg)
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(path(cfg['model']),local_files_only=True)
    rendered={r['id']:dict(prompt=render(tok,r),input_ids=tok(render(tok,r),add_special_tokens=False)['input_ids'])
              for name in ['elicitation','development','final','probes'] for r in data[name]}
    if (cfg['synthetic_instruction_index'],cfg['synthetic_question_index'])!=(0,0) or hashlib.sha256(cfg['synthetic_response'].encode()).hexdigest()!=cfg['synthetic_response_sha256']:
        raise ValueError('fixed synthetic inputs differ')
    fixed_response_ids=tok(cfg['synthetic_response'],add_special_tokens=False)['input_ids']
    data['synthetic']=[{**rendered[r['id']],'generated_ids':fixed_response_ids,'output':cfg['synthetic_response']}
                       for r in data['elicitation'][:2]]
    if any(len(r['input_ids'])+cfg['elicitation_cap']>cfg['max_sequence_tokens'] for r in rendered.values()):
        raise ValueError('sequence guard; no silent truncation')
    model_manifest=read_json(cfg['model_manifest'])
    for name,record in model_manifest['files'].items():
        f=path(cfg['model'])/name
        if f.stat().st_size!=record['bytes'] or sha(f)!=record['sha256']:
            raise ValueError(f'saved NF4 identity changed: {name}')
    binary=path(cfg['judge_source'])
    if sha(binary)!=cfg['judge_sha256'] or subprocess.check_output([str(binary),'--version'],text=True,timeout=30).strip()!=cfg['judge_version']:
        raise ValueError('judge source identity changed')
    from scripts import persona_reference_common as pv
    plans={}
    for ni,name in enumerate(['elicitation','development','final']):
        rr=data[name]
        questions=list(dict.fromkeys(r['question_id'] for r in rr))
        arms=[f'i{i}_{side}' for i in range(5) for side in ['pos','neg']] if name=='elicitation' else list(dict.fromkeys(r['arm'] for r in rr))
        plans[name]={metric:pv.make_plan(questions,arms,cfg['plan_seed']+100*ni+mi)
                     for mi,metric in enumerate(['trait','coherence']+(['impolite'] if name=='final' else []))}
    # Static transitive Python import closure is snapshotted and verified.
    from scripts.author_gcad_impolite import closure
    sources=closure(SOURCE)
    files=list(sources)+[cfg[k] for k in ['protocol','model_manifest','extraction','development','final']]
    files+=['results/reviews/qwen38_sarcasm_design_review_01.md','results/reviews/qwen38_sarcasm_implementation_review_01.md']
    author='outputs/persona-author-replication-audit-01/gcad'
    files += [author+'/eval/prompts.py',author+'/data/trait_data_eval/impolite.json']
    for runtime in [cfg['runtime'],'outputs/qwen38-science-01/env/bin/python']:
        installed=path(runtime).parent.parent/'lib/python3.13/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py'
        files.append(str(installed))
    hashes={str(f):sha(path(f)) for f in files}
    run.mkdir(parents=True)
    target=path(cfg['judge_snapshot']); target.parent.mkdir(parents=True)
    shutil.copyfile(binary,target); target.chmod(0o500)
    if sha(target)!=cfg['judge_sha256']:raise ValueError('judge copy differs')
    save(run/'config.json',cfg); save(run/'inputs.json',data); save(run/'rendered.json',rendered); save(run/'plans.json',plans);save(run/'rubrics.json',rubrics())
    for name in sources:
        out=run/'source'/name;out.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path(name),out)
    save(run/'manifest.json',{'files':hashes,'model_files':model_manifest['files'],
        'judge_snapshot_sha256':sha(target),'packages':packages(),
        'frozen':{p.name:sha(p) for p in run.glob('*.json')},
        'source_copies':{str(p.relative_to(run)):sha(p) for p in (run/'source').rglob('*') if p.is_file()},
        'no_model_or_judge_calls':True})
    print(json.dumps({'run':str(run),'manifest_sha256':sha(run/'manifest.json')},indent=2))


class Context:
    def __init__(self):
        self.cfg=read_json(CONFIG);self.run=path(self.cfg['run']);self.manifest=read_json(self.run/'manifest.json')
        if self.cfg!=read_json(self.run/'config.json'):raise ValueError('config changed')
        for name,h in self.manifest['files'].items():
            if sha(path(name))!=h:raise ValueError(f'R0 source/input differs: {name}')
        for name,h in {**self.manifest['frozen'],**self.manifest['source_copies']}.items():
            if sha(self.run/name)!=h:raise ValueError(f'frozen artifact changed: {name}')
        if packages()!=self.manifest['packages']:raise ValueError('runtime package versions changed')
        self.data=read_json(self.run/'inputs.json');self.rendered=read_json(self.run/'rendered.json')

    def approve(self, stage, dependencies=None):
        expected={'manifest_sha256':sha(self.run/'manifest.json'),'stage':stage,
                  'dependencies':dependencies or {}}
        if read_json(self.run/'approvals'/f'{stage}.json')!=expected:
            raise ValueError('exact PM phase registration required')

    def need(self, stage):
        value=read_json(self.run/stage/'result.json')
        if not value.get('passed',value.get('completed',False)):raise ValueError(f'{stage} prerequisite failed')
        return value

    @contextmanager
    def gpu_deadline(self,stage):
        # Include load, hash verification and GPU synchronization in wall accounting.
        # An interrupted session is charged wall time until now, conservatively.
        root=self.run/'gpu-sessions';root.mkdir(exist_ok=True)
        spent=0.
        for p in root.glob('*.start.json'):
            done=p.with_name(p.name.replace('.start.json','.done.json'))
            spent+=read_json(done)['seconds'] if done.exists() else max(0,time.time()-read_json(p)['unix'])
        remaining=self.cfg['max_gpu_seconds']-spent
        if remaining<=0:raise RuntimeError('8 GPU-hour hard cap reached')
        name=f'{stage}-{time.time_ns()}';start=time.monotonic()
        save(root/(name+'.start.json'),{'unix':time.time(),'remaining_seconds':remaining})
        def expired(signum,frame):raise TimeoutError('registered GPU wall-time hard cap reached')
        old=signal.signal(signal.SIGALRM,expired);signal.setitimer(signal.ITIMER_REAL,remaining)
        try:yield
        finally:
            signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,old)
            save(root/(name+'.done.json'),{'seconds':time.monotonic()-start})

    def reserve(self,kind,stage,key,tokens=0):
        limits={'generation':self.cfg['maximum_generations'],'teacher_forced':self.cfg['maximum_teacher_forced'],
                'logit':self.cfg['maximum_logit_forwards']}
        root=self.run/'counters'/kind;root.mkdir(parents=True,exist_ok=True)
        count=len(list(root.glob('*.json')))
        if count>=limits[kind]:raise RuntimeError(f'{kind} hard count cap reached')
        entries=[read_json(p) for p in root.glob('*.json')]
        caps={'generation':{'synthetic-smoke':6,'mechanics':6,'elicitation':200,'development':128,'final':128},
              'teacher_forced':{'synthetic-smoke':2,'extraction':300},'logit':{'synthetic-smoke':4,'mechanics':4}}
        if stage not in caps[kind] or sum(r['stage']==stage for r in entries)>=caps[kind][stage]:
            raise RuntimeError('per-stage count cap reached or unregistered stage')
        if kind=='generation' and sum(r['tokens'] for r in entries)+tokens>self.cfg['maximum_generated_tokens']:
            raise RuntimeError('maximum generated-token reservation cap reached')
        save(root/f'{count:04d}.json',{'stage':stage,'key':key,'tokens':tokens})

    @contextmanager
    def lock(self):
        with (self.run/'.worker.lock').open('a') as f:
            fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
            yield

    def task(self, stage, key, fn):
        root=self.run/stage/'rows';root.mkdir(parents=True,exist_ok=True)
        result=root/f'{key}.json';start=root/f'{key}.start.json';receipt=root/f'{key}.receipt.json'
        if result.exists():
            if read_json(receipt)['sha256']!=sha(result):raise ValueError('completed row hash differs')
            return read_json(result)
        if start.exists():raise RuntimeError('uncertain or failed row; no automatic retry')
        save(start,{'key':key,'stage':stage,'manifest_sha256':sha(self.run/'manifest.json'),'started_unix':time.time()})
        tick=time.monotonic()
        try:
            value=fn(); value['seconds']=time.monotonic()-tick
            save(result,value);save(receipt,{'sha256':sha(result),'start_sha256':sha(start)})
            return value
        except BaseException as exc:
            save(root/f'{key}.failure.json',{'error':f'{type(exc).__name__}: {exc}','seconds':time.monotonic()-tick,'no_retry':True})
            raise


def load(s):
    os.environ['CUDA_VISIBLE_DEVICES']=s.cfg['gpu_uuid']
    import torch
    from transformers import Qwen3_5ForCausalLM,AutoTokenizer
    from scripts.qwen38_nf4_smoke import validate_loading
    for name,record in s.manifest['model_files'].items():
        if sha(path(s.cfg['model'])/name)!=record['sha256']:raise ValueError('NF4 model changed')
    model,info=Qwen3_5ForCausalLM.from_pretrained(path(s.cfg['model']),local_files_only=True,
        trust_remote_code=False,dtype=torch.bfloat16,device_map={'':'cuda:0'},attn_implementation='sdpa',output_loading_info=True)
    validate_loading(info);model.eval();model.config.use_cache=True
    from scripts.qwen38_gcad_port import validate_geometry
    validate_geometry(model,s.cfg['layers'])
    return model,AutoTokenizer.from_pretrained(path(s.cfg['model']),local_files_only=True)


def boundary(tok,prompt,prompt_ids,generated_ids,judged_output,eos_ids):
    if not prompt.startswith('<|im_start|>system\n') or '<|im_end|>\n' not in prompt:
        raise ValueError('explicit first system span required')
    system=prompt[:prompt.index('<|im_end|>\n')+len('<|im_end|>\n')]
    pids=tok(prompt,add_special_tokens=False)['input_ids'];sys=tok(system,add_special_tokens=False)['input_ids']
    if pids!=prompt_ids or pids[:len(sys)]!=sys:
        raise ValueError('saved system/prompt prefix mismatch')
    if tok.decode(generated_ids,skip_special_tokens=True)!=judged_output:
        raise ValueError('raw trajectory decode differs from judged output')
    suffix=list(generated_ids)
    if suffix and suffix[-1] in eos_ids:suffix.pop()
    ids=pids+suffix
    mask=[False]*len(pids)+[token not in tok.all_special_ids for token in suffix]
    if not any(mask):raise ValueError('empty response mask on raw trajectory')
    return ids,len(sys),len(pids),mask


def extract_one(s,model,tok,trajectory):
    import torch
    from scripts.qwen38_gcad_port import extract
    eos=read_json(path(s.cfg['model'])/'generation_config.json')['eos_token_id']
    eos=[eos] if isinstance(eos,int) else eos
    ids,sl,pl,mask=boundary(tok,trajectory['prompt'],trajectory['input_ids'],trajectory['generated_ids'],trajectory['output'],eos)
    if len(ids)>s.cfg['max_sequence_tokens']:raise ValueError('sequence cap; no truncation')
    result=extract(model,torch.tensor([ids],device=model.device),sl,pl,mask,s.cfg['layers'])
    return result,{'system_length':sl,'prompt_length':pl,'input_ids':ids,'response_tokens':sum(mask),'prefix_checks':2,
                   'diagnostics':{str(i):r['diagnostics'] for i,r in result.items()}}


def signature(model):
    return {n:[list(m._forward_hooks),list(m._forward_pre_hooks)] for n,m in model.named_modules()}


def completion_metadata(suffix,eos,cap):
    eos=[eos] if isinstance(eos,int) else eos
    ended=bool(suffix and suffix[-1] in (eos or []));hit=len(suffix)==cap and not ended
    return {'output_ids':suffix,'generated_ids':suffix,'cap_hit':hit,'hit_token_limit':hit,'ended_eos':ended}


def frozen_rendering(s,tok,row):
    prompt=render(tok,row);ids=tok(prompt,add_special_tokens=False)['input_ids']
    if s.rendered[row['id']]!={'prompt':prompt,'input_ids':ids}:
        raise ValueError('current prompt text/IDs differ from frozen rendered.json')
    return prompt,ids


def generate(s,model,tok,row,bank=None,strength=None,cap=None,seed=42,sampling=False):
    import torch
    from transformers import set_seed,GenerationConfig
    from scripts.qwen38_gcad_port import Steerer
    set_seed(seed);prompt,ids=frozen_rendering(s,tok,row)
    if len(ids)<=1:raise ValueError('fixed prompts must have more than one token')
    batch=torch.tensor([ids],device=model.device); cap=cap or s.cfg['target_cap']
    ctx=Steerer(model,bank,s.cfg['layers'],strength,s.cfg['scale']) if bank is not None else nullcontext()
    before=signature(model)
    defaults=GenerationConfig.from_pretrained(path(s.cfg['model']),local_files_only=True)
    defaults.update(do_sample=sampling,max_new_tokens=cap,use_cache=True,num_beams=1,
        repetition_penalty=1.05,temperature=1. if sampling else None,top_p=.8 if sampling else None,top_k=20 if sampling else None,
        pad_token_id=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id)
    with torch.inference_mode(),ctx:
        if bank is not None:
            for i,b in enumerate(model.model.layers):
                if i not in s.cfg['layers']:
                    for name,m in b.named_modules():
                        if 'attn' in name and (m._forward_hooks or m._forward_pre_hooks):
                            if signature(model)[f'model.layers.{i}.{name}']!=before[f'model.layers.{i}.{name}']:
                                raise ValueError('hook outside selected full attention sites')
        s.reserve('generation',s.active_stage,row.get('id'),cap)
        out=model.generate(input_ids=batch,attention_mask=torch.ones_like(batch),generation_config=defaults,logits_to_keep=1)
    if signature(model)!=before:raise ValueError('hook signatures changed after cleanup')
    suffix=out[0,len(ids):].tolist()
    audit=ctx.audit if bank is not None else {}
    for a in audit.values():
        if a!={'prefill':1,'decode':len(suffix)-1,'errors':0}:raise ValueError('site prefill/decode coverage differs')
    return {'id':row.get('id'),'prompt':prompt,'input_ids':ids,
            **completion_metadata(suffix,defaults.eos_token_id,cap),
            'output':tok.decode(suffix,skip_special_tokens=True),'audit':audit}


def mechanics(s,synthetic=False):
    import torch
    from safetensors.torch import save_file,load_file
    from scripts import qwen38_gcad_port as port
    stage='synthetic-smoke' if synthetic else 'mechanics'
    deps={} if synthetic else {'bank_sha256':sha(s.run/'extraction/bank.safetensors')}
    s.approve(stage,deps)
    if not synthetic:s.need('extraction')
    # Entire mechanics phase is one non-retryable row, preserving failures.
    def execute():
        model,tok=load(s); torch.cuda.reset_peak_memory_stats()
        root=s.run/stage; bankfile=root/'bank.safetensors'; diagnostics=[]
        if synthetic:
            pos,neg=s.data['elicitation'][:2]
            s.reserve('teacher_forced',stage,'positive')
            p,pmeta=extract_one(s,model,tok,s.data['synthetic'][0])
            s.reserve('teacher_forced',stage,'negative')
            n,nmeta=extract_one(s,model,tok,s.data['synthetic'][1])
            diagnostics=[pmeta,nmeta]
            save(root/'synthetic-boundaries.json',diagnostics)
            for side in diagnostics:
                for d in side['diagnostics'].values():
                    if not (d['gated_relative_error']<=.02 and d['omitted_gate_relative_error']>=.2
                        and d['omitted_gate_relative_error']>=10*d['gated_relative_error'] and d['split_relative_error']<=.01):
                        raise ValueError('gated decomposition mechanics failure')
            tensors=port.bank([p],[n],s.cfg['layers']);save_file(tensors,str(bankfile))
        else:tensors=load_file(str(s.run/'extraction/bank.safetensors'))
        if any(tensors[f'{i}.delta'].norm()<=0 for i in s.cfg['layers']):raise ValueError('zero delta site')
        records=[]
        for j,row in enumerate(s.data['probes']):
            variants={}
            for label,coef in [('ordinary',None),('zero',0.),('one',1.)]:
                variants[label]=generate(s,model,tok,row,tensors if coef is not None else None,coef,s.cfg['mechanics_cap'])
                save(root/f'probe-{j}-{label}.json',variants[label])
            if variants['ordinary']['output_ids']!=variants['zero']['output_ids']:raise ValueError('zero token parity failed')
            records.append(variants)
        row=s.data['probes'][0];ids=torch.tensor([tok(render(tok,row),add_special_tokens=False)['input_ids']],device=model.device)
        sig=signature(model)
        observed={};last=[]
        # Capture final normalized hidden without installing Transformers output hooks.
        handle=model.model.norm.register_forward_hook(lambda m,a,o:last.append(o[:,-1].detach().clone()))
        try:
            values=[]
            for coef in [None,0.,1.,None]:
                s.reserve('logit',stage,str(coef))
                with torch.inference_mode(),(port.Steerer(model,tensors,s.cfg['layers'],coef) if coef is not None else nullcontext()):
                    values.append(model(ids,attention_mask=torch.ones_like(ids),use_cache=True,logits_to_keep=1).logits.detach().clone())
        finally:handle.remove()
        if signature(model)!=sig or not torch.equal(values[0],values[1]) or not torch.equal(values[0],values[3]):
            raise ValueError('zero or cleanup exact logit parity failure')
        change=float((values[2].float()-values[0].float()).abs().max())
        if not change>0 or torch.equal(last[0],last[2]) or not all(torch.isfinite(v).all() for v in values):
            raise ValueError('nonzero shift/finite mechanics failure')
        save_file({f'logits_{i}':v.cpu() for i,v in enumerate(values)},str(root/'logits.safetensors'))
        return {'passed':True,'generations':6,'teacher_forced':2 if synthetic else 0,'logit_forwards':4,
            'max_logit_change':change,'records':records,'bank_norms':{str(i):float(tensors[f'{i}.delta'].norm()) for i in s.cfg['layers']},
            'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'cache_disclosure':s.cfg['cache_disclosure']}
    result=s.task(stage,'phase',execute);match(s.run/stage/'result.json',result)


def generation_phase(s,stage):
    from safetensors.torch import load_file
    s.need('synthetic-smoke')
    deps={}
    if stage=='development':
        s.need('mechanics');deps={'mechanics_sha256':sha(s.run/'mechanics/result.json')}
    if stage=='final':
        selection=read_json(s.run/'selection.json')
        if selection['selection_null']:raise ValueError('development null; no final')
        deps={'selection_sha256':sha(s.run/'selection.json')}
    s.approve(stage,deps)
    model,tok=load(s);outputs=[]
    bank=load_file(str(s.run/'extraction/bank.safetensors')) if stage!='elicitation' else None
    for index,row in enumerate(s.data[stage]):
        arm=row.get('arm','baseline');coef=None;active=None
        if arm.startswith('gcad_'):coef=float(arm.split('_')[1]);active=bank
        if arm in ['selected','random']:
            coef=float(selection['selected']['gcad'].split('_')[1]);active=bank
            if arm=='random':active=load_file(str(s.run/f'random-{s.cfg["random_seeds"][(index//4)%3]}.safetensors'))
        result=s.task(stage,row['id'],lambda row=row,coef=coef,active=active,index=index:generate(s,model,tok,row,active,coef,
            s.cfg['elicitation_cap'] if stage=='elicitation' else s.cfg['target_cap'],
            s.cfg['plan_seed']+index if stage=='elicitation' else s.cfg['seed'],stage=='elicitation'))
        outputs.append({**row,**result})
    match(s.run/stage/'result.json',{'completed':True,'rows':outputs,'generations':len(outputs)})


def eligibility(s):
    scores=read_json(s.run/'judgments/elicitation/scores.json')
    chosen=[]
    for p,n in zip(s.data['elicitation'][::2],s.data['elicitation'][1::2]):
        a,b=scores[p['id']],scores[n['id']]
        if a['trait'] is not None and b['trait'] is not None and a['trait']>=50 and b['trait']<50 and min(a['coherence'],b['coherence'])>=50:
            chosen.append(p['pair'])
    match(s.run/'eligibility.json',{'pairs':chosen,'extraction_null':len(chosen)<s.cfg['minimum_pairs'],
        'scores_sha256':sha(s.run/'judgments/elicitation/scores.json')})


def extraction(s):
    import torch
    from safetensors.torch import save_file,load_file
    from scripts import qwen38_gcad_port as port
    e=read_json(s.run/'eligibility.json')
    if e['extraction_null']:raise ValueError('extraction null; no bank')
    s.approve('extraction',{'eligibility_sha256':sha(s.run/'eligibility.json')})
    model,tok=load(s);outputs=read_json(s.run/'elicitation/result.json')['rows'];collected={}
    for row in outputs:
        if row['pair'] not in e['pairs']:continue
        def execute(row=row):
            s.reserve('teacher_forced','extraction',row['id'])
            r,meta=extract_one(s,model,tok,row)
            out=s.run/'extraction'/f'{row["id"]}.safetensors'
            save_file({f'{i}.{k}':v for i,value in r.items() for k,v in value.items() if isinstance(v,torch.Tensor)},str(out))
            return {'boundary':meta,'tensor_file':str(out.relative_to(s.run)),'sha256':sha(out),
                    'counts':{str(i):{k:r[i][k] for k in ['key_count','query_count']} for i in s.cfg['layers']}}
        record=s.task('extraction',row['id'],execute);file=s.run/record['tensor_file']
        if sha(file)!=record['sha256']:raise ValueError('extraction row tensor differs')
        values=load_file(str(file));collected[row['id']]={i:{**{k:values[f'{i}.{k}'] for k in ['delta','key_sum','query_sum']},**record['counts'][str(i)]} for i in s.cfg['layers']}
    positives=[collected[p+'_pos'] for p in e['pairs']];negatives=[collected[p+'_neg'] for p in e['pairs']]
    tensors=port.bank(positives,negatives,s.cfg['layers']);out=s.run/'extraction/bank.safetensors'
    if out.exists():
        prior=load_file(str(out))
        if not all(torch.equal(prior[k],v) for k,v in tensors.items()):raise ValueError('bank differs')
    else:save_file(tensors,str(out))
    match(s.run/'extraction/result.json',{'passed':True,'pair_count':len(e['pairs']),'teacher_forced':len(collected),'sha256':sha(out)})


def selection(s):
    import torch
    from safetensors.torch import load_file,save_file
    from scripts import sarcasm_method_transfer_common as scientific
    predictions=read_json(s.run/'development/result.json')['rows'];scores=read_json(s.run/'judgments/development/scores.json')
    metrics={arm:scientific.summarize([r for r in predictions if r['arm']==arm],scores) for arm in ['baseline','prompted','gcad_1','gcad_2']}
    decision=scientific.choose(metrics,{a:{'method':'gcad','strength':float(a.split('_')[1])} for a in ['gcad_1','gcad_2']})
    tensors=load_file(str(s.run/'extraction/bank.safetensors'))
    random_hashes={}
    for seed in s.cfg['random_seeds']:
        random={k:v.clone() for k,v in tensors.items()}
        for i in s.cfg['layers']:
            gen=torch.Generator(device='cpu').manual_seed(seed+1000*i)
            v=torch.randn(tensors[f'{i}.delta'].shape,generator=gen)
            random[f'{i}.delta']=v/v.norm()*tensors[f'{i}.delta'].norm()
        p=s.run/f'random-{seed}.safetensors'
        if not p.exists():save_file(random,str(p))
        elif not all(torch.equal(load_file(str(p))[k],v) for k,v in random.items()):raise ValueError('random tensors changed')
        random_hashes[p.name]=sha(p)
    match(s.run/'selection.json',{**decision,'metrics':metrics,'random_sha256':random_hashes,
        'scores_sha256':sha(s.run/'judgments/development/scores.json'),'bank_sha256':sha(s.run/'extraction/bank.safetensors')})


def report(s):
    from scripts import sarcasm_method_transfer_common as scientific
    selected=read_json(s.run/'selection.json')
    if selected['selection_null']:
        match(s.run/'closeout.json',{'outcome':'development_null','selection_sha256':sha(s.run/'selection.json')});return
    predictions=read_json(s.run/'final/result.json')['rows']
    complete=(s.run/'judgments/final/scores.json').exists()
    scores_file=s.run/('judgments/final/scores.json' if complete else 'judgments/final/primary-scores.json')
    scores=read_json(scores_file)
    metrics={arm:scientific.summarize([r for r in predictions if r['arm']==arm],scores)
             for arm in ['baseline','prompted','selected','random']}
    gates=scientific.gates(metrics['baseline'],metrics['selected'],metrics['random'])
    h14_excluded={}
    for arm in metrics:
        ids=[r['id'] for r in predictions if r['arm']==arm and r['question_id']!='H14']
        h14_excluded[arm]={'n':len(ids),'trait_conservative':sum(scores[i]['trait'] or 0 for i in ids)/len(ids),
                          'coherence':sum(scores[i]['coherence'] for i in ids)/len(ids)}
    def correlation(pairs):
        import math
        if len(pairs)<2:return {'n':len(pairs),'pearson':None}
        x,y=zip(*pairs);mx=sum(x)/len(x);my=sum(y)/len(y)
        denominator=math.sqrt(sum((a-mx)**2 for a in x)*sum((b-my)**2 for b in y))
        return {'n':len(pairs),'pearson':sum((a-mx)*(b-my) for a,b in pairs)/denominator if denominator else None}
    coupling={}
    if complete:
        for arm in metrics:
            coupling[arm]=correlation([(scores[r['id']]['trait'],scores[r['id']]['impolite'])
                for r in predictions if r['arm']==arm and scores[r['id']]['trait'] is not None])
        coupling['pooled_confounded_by_arm']=correlation([(v['trait'],v['impolite']) for v in scores.values() if v['trait'] is not None])
    match(s.run/'closeout.json',{'outcome':('retain' if gates['all_pass'] else 'final_failure') if complete else 'primary_complete_descriptive_incomplete',
        'primary_outcome':'retain' if gates['all_pass'] else 'final_failure',
        'metrics':metrics,'gates':gates,'selection_sha256':sha(s.run/'selection.json'),
        'scores_sha256':sha(scores_file),'h14_excluded_descriptive_only':h14_excluded,'coupling':coupling,
        'final_role':s.cfg['final_role'],'cache_disclosure':s.cfg['cache_disclosure']})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['prepare','synthetic-smoke','elicitation','eligibility','extraction','mechanics','development','selection','final','report'])
    a=p.parse_args()
    if a.stage=='prepare':prepare();return
    s=Context()
    with s.lock():
        s.active_stage=a.stage
        if a.stage in ['synthetic-smoke','mechanics','elicitation','extraction','development','final']:
            with s.gpu_deadline(a.stage):
                if a.stage in ['synthetic-smoke','mechanics']:mechanics(s,a.stage=='synthetic-smoke')
                elif a.stage=='extraction':extraction(s)
                else:generation_phase(s,a.stage)
        else:globals()[a.stage](s)


if __name__=='__main__':main()
