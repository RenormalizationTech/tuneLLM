"""Isolated author-SAPS adaptation on native HF Qwen3; no preparation/live by default."""
import argparse
from contextlib import contextmanager, nullcontext
import importlib.metadata
import json
import os
from pathlib import Path
import re
import shutil
import time

from scripts import persona_saps_common as p
from scripts import persona_reference as pv
from scripts import tone_judge as f
from scripts.persona_reference_common import validate_generation, private_export
from scripts.batched_generation import generate_batched, render_prompt
from scripts.benchmark_screen import choice_token_ids, hf_multiple_choice, ifeval_scores
from evaluation.gsm8k import score_gsm8k
from scripts.benign_helpfulness import resource_check
from scripts.common import path, read_json
from scripts.profanity_steering import save_vectors
from scripts.save_nf4 import stream_hash
from scripts.tone_steering import load_model, committed

RUN='outputs/persona-saps-sarcasm-01'
CONFIG='configs/persona_saps_sarcasm.json'
PIN='c56fb28ac0d14c8fd70035e035c5452e3c6c6ccc'
save=f.save
check_hashes=pv.check_hashes
stop=pv.stop


def validate_config(cfg):
    fixed={'run':RUN,'result':'results/persona_saps_sarcasm_01.json','model_id':'Qwen/Qwen3-4B','revision':'1cfa9a7208912126459214e8b04321603b3df60c',
        'hf_home':'.hf','gpu':'GPU-34646ab9-ec19-ca63-0dfb-bdbf2e1a977b','gpu_uuid':'GPU-34646ab9-ec19-ca63-0dfb-bdbf2e1a977b','minimum_free_mib':12000,'maximum_utilization_percent':10,
        'dtype':'bfloat16','attn_implementation':'sdpa','enable_thinking':False,'seed':42,'batch_size':1,'max_new_tokens':60,'blocks':list(p.BLOCKS),
        'alphas':{'fixed':[1.],'saps':[1.,1.4]},'test_seed':702,'test_count':100,'custom_count':20,'draw_seed':42,
        'plan_seeds':{'initial':721,'validation':722,'test':723,'custom':724},'maximum_generations':1242,'maximum_teacher_forced':1500,'maximum_logit_forwards':2,'maximum_mc_items':64,'maximum_judge_calls':146,
        'protocol':'docs/sarcasm_reference_protocol.md','design_review':'results/reviews/sarcasm_reference_design_review.md','fixture_directory':'evaluation/persona_saps',
        'paper_provenance':'outputs/persona-saps-paper-01/provenance.json','author_audit':'outputs/persona-literature-audit-01/manifest.json','judge_runtime_config':'configs/qwen3_14b_baseline.json',
        'retention_fixture':'outputs/refusal-03/inputs/retention.json','pv_result':'results/persona_reference_rudeness_01.json',
        'initial_registration':'results/protocols/persona-saps-sarcasm-01.json','strength_registration':'results/protocols/persona-saps-sarcasm-strength-01.json','selection_registration':'results/protocols/persona-saps-sarcasm-selection-01.json'}
    if cfg!=fixed or f.canonical_hash(cfg)!=f.canonical_hash(fixed):raise ValueError('fixed settings/paths/budgets differ')


def guard(run):
    if run.resolve()!=path(RUN).resolve():raise ValueError('only fresh SAPS output path writable')


def task_rows(rows,prompts,style=False):
    instruction=prompts['style'][0] if style else prompts['plain']
    return [{'id':r['id'],'user':p.task_user(r['sentence'],instruction)} for r in rows]


def prepare(args,run):
    guard(run)
    if run.exists():raise FileExistsError('one fresh SAPS preparation required')
    cfg=read_json(CONFIG);validate_config(cfg);previous=read_json(cfg['pv_result'])
    if previous['status'] not in ('complete','elicitation_null','selection_null'):raise ValueError('PV rudeness must close before SAPS')
    directory=path(cfg['fixture_directory']);data={k:read_json(directory/(k+'.json')) for k in ('train','val','test','prompts')};split=p.partition(data)
    audit=read_json(cfg['author_audit']);author={r['local_path']:r['sha256'] for r in audit['files'] if r['repo']=='SharathHN/saps-steering'}
    if any(r['revision']!=PIN for r in audit['files'] if r['repo']=='SharathHN/saps-steering'):raise ValueError('author commit differs')
    source=path('outputs/persona-literature-audit-01/saps-steering')
    for local,remote in [(f'{k}.json',f'data/sarcasm/{k}.json') for k in ('train','val','test')]+[('prompts.json','prompts/sarcasm.json'),('LICENSE','LICENSE')]:
        if (directory/local).read_bytes()!=(source/remote).read_bytes():raise ValueError('published source bytes modified')
    paper=read_json(cfg['paper_provenance']);paperroot=path(cfg['paper_provenance']).parent
    external={str(paperroot/n):h for n,h in paper['files'].items()};external.update(author)
    runtime=read_json(cfg['judge_runtime_config']);record=read_json(runtime['judge_runtime_record']);judge=f.validate_audit(runtime['judge_audit'])
    model=read_json('outputs/xstest-baseline-01/manifest.json')
    extra=[cfg['protocol'],cfg['design_review'],cfg['pv_result'],cfg['paper_provenance'],cfg['author_audit'],cfg['judge_runtime_config'],runtime['judge_runtime_record'],runtime['judge_audit'],str(path(runtime['judge_audit']).parent/'attempt.json'),cfg['retention_fixture'],'evaluation/persona_traits/rubric.json']
    external.update({n:stream_hash(path(n)) for n in extra});external[record['binary']]=pv.BINARY_SHA
    if record['binary_sha256']!=pv.BINARY_SHA:raise ValueError('native pin differs')
    sources=pv.source_closure(['scripts/persona_saps.py','scripts/persona_saps_common.py','scripts/persona_saps_runbook.md',CONFIG,
        *[str(file.relative_to(path('.'))) for file in directory.iterdir() if file.is_file()]])
    sourcehash={n:stream_hash(path(n)) for n in sources};check_hashes([sourcehash,external,model['model_sha256']])
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(model['snapshot'],local_files_only=True,trust_remote_code=False)
    if tok.pad_token_id is None or tok.eos_token_id is None or tok.pad_token_id==tok.eos_token_id:raise ValueError('distinct pad/EOS required')
    draws=p.instruction_draws();rendered={}
    def encoded(rows):return {r['id']:{'rendered_prompt':render_prompt(tok,r),'input_ids':tok(render_prompt(tok,r),add_special_tokens=False)['input_ids']} for r in rows}
    for name,rows in [('validation',split['validation']),('test',split['test'])]:
        for style in (False,True):rendered[name+('_prompt' if style else '_plain')]=encoded(task_rows(rows,data['prompts'],style))
    for name,indices in [('train_plain',None),('train_first',draws[:500]),('train_target',draws[500:])]:
        rendered[name]=encoded([{'id':r['id'],'user':p.task_user(r['sentence'],data['prompts']['plain'] if indices is None else data['prompts']['style'][indices[i]])} for i,r in enumerate(split['train'])])
    fixture=read_json(cfg['retention_fixture'])
    for task in ('general_mc','physics_mc'):choice_token_ids(tok,fixture[task])
    for folder in ('artifacts/tools/ifeval','artifacts/tools/nltk'):
        for file in path(folder).rglob('*'):
            if file.is_file() and '__pycache__' not in file.parts:external[str(file)]=stream_hash(file)
    run.mkdir(mode=0o700);save(run/'config.json',cfg)
    for n,v in {'data':data,'split':split,'plans':p.plans(split,cfg),'draws':draws,'rendered':rendered,'rubric':read_json(directory/'rubric.json'),'retention':fixture}.items():save(run/'inputs'/(n+'.json'),v)
    for n in sources:
        target=run/'source'/n;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path(n),target)
    save(run/'manifest.json',{'files':{str(file.relative_to(run)):stream_hash(file) for file in run.rglob('*') if file.is_file()},'sources':sourcehash,'external_sha256':external,
        'snapshot':model['snapshot'],'model_sha256':model['model_sha256'],'packages':{k:importlib.metadata.version(k) for k in pv.PACKAGES},
        'binary':record['binary'],'binary_sha256':pv.BINARY_SHA,'cli_version':judge['cli_version'],'audit_sha256':stream_hash(path(runtime['judge_audit'])),
        'tokenizer_ids':{'eos':tok.eos_token_id,'pad':tok.pad_token_id},'no_live_calls':True})
    check_hashes([sourcehash,external])


class Context:
    def __init__(self,run):
        guard(run);self.run=run;self.cfg=read_json(run/'config.json');self.m=read_json(run/'manifest.json');validate_config(self.cfg)

    def verify(self,models=False):
        maps=[{str(self.run/n):h for n,h in self.m['files'].items()},self.m['sources'],self.m['external_sha256']]
        if models:maps.append(self.m['model_sha256'])
        check_hashes(maps)
        if any(importlib.metadata.version(k)!=v for k,v in self.m['packages'].items()):raise ValueError('runtime packages differ')
        split=p.partition(read_json(self.run/'inputs/data.json'))
        if split!=read_json(self.run/'inputs/split.json') or read_json(self.run/'inputs/plans.json')!=p.plans(split,self.cfg) or read_json(self.run/'inputs/draws.json')!=p.instruction_draws():raise ValueError('split/plan/draw reconstruction differs')
        return self

    def initial(self):
        for n in (*self.m['sources'],self.cfg['protocol'],self.cfg['design_review'],self.cfg['initial_registration']):committed(n)
        if read_json(self.cfg['initial_registration'])['manifest_sha256']!=stream_hash(self.run/'manifest.json'):raise ValueError('initial registration differs')


def stage_sources(s,name):
    out=s.run/name;record=read_json(out/'run.json')
    if not record['completed'] or record['gpu_uuid']!=s.cfg['gpu_uuid']:raise ValueError('incomplete/wrong-device phase')
    check_hashes([{str(out/n):h for n,h in record['files_sha256'].items()}])
    return {str(out/'run.json'):stream_hash(out/'run.json'),**{str(out/n):h for n,h in record['files_sha256'].items()}}


def initial_gate(s):
    values=metric_values(s,'initial');return {'passed':p.elicitation_gate(values),'metrics':values,'accepted_sha256':stream_hash(s.run/'grades/initial/accepted.json')}


def required_elicitation(s):
    record=initial_gate(s)
    if read_json(s.run/'eligibility.json')!=record or not record['passed']:raise ValueError('elicitation failed/record differs')


def cost_state(s):
    return {k:sum(read_json(file).get(k,0) for file in s.run.glob('*/run.json')) for k in ('saved_generations','teacher_forced_sequences','logit_forwards','mc_items')}


def charge(progress,existing,**amounts):
    limits={'saved_generations':1242,'teacher_forced_sequences':1500,'logit_forwards':2,'mc_items':64}
    for k,amount in amounts.items():
        if progress[k]+existing[k]+amount>limits[k]:raise ValueError('bounded work exceeded before call')


@contextmanager
def gpu_stage(s,name):
    s.initial()
    try:resource=resource_check(s.cfg)
    except RuntimeError as exc:raise pv.ResourceWait('resources busy; phase not started') from exc
    out=s.run/name;out.mkdir(mode=0o700,exist_ok=False)
    import torch
    began=time.perf_counter();completed=False;existing=cost_state(s);progress={k:0 for k in existing};progress.update(generation_wall_seconds=0.,prompt_forward_wall_seconds=0.)
    try:
        model,tok=load_model(s.cfg,s.m,'bf16')
        if model.config.num_hidden_layers!=36 or any(v.dtype!=torch.bfloat16 or v.device!=torch.device('cuda:0') for v in model.parameters()):raise ValueError('native 36-block BF16 one CUDA required')
        torch.cuda.reset_peak_memory_stats();yield model,tok,out,progress,existing;torch.cuda.synchronize();completed=True
    finally:
        save(out/'run.json',{'completed':completed,**resource,**progress,'elapsed_wall_seconds':time.perf_counter()-began,
            'count_scope':'Saved/completed work only; interrupted unsaved calls may incur additional work','wall_scope':'synchronized whole stage including load/tokenization/capture/serialization',
            'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'memory_scope':'PyTorch allocator, not total VRAM',
            'torch_cuda_runtime':torch.version.cuda,'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),'packages':s.m['packages'],
            'dtype':'bfloat16','attention':'sdpa','seed':42,'tf32_matmul':torch.backends.cuda.matmul.allow_tf32,'tf32_cudnn':torch.backends.cudnn.allow_tf32,
            'files_sha256':{str(file.relative_to(out)):stream_hash(file) for file in out.rglob('*') if file.is_file()}})


def generate(s,model,tok,rows,progress,existing,setting=None,tensors=None,h0=None,random=False,cap=60,capture_baseline=False):
    results=[];batches=[];timing={'generation_wall_seconds':0.,'pipeline_wall_seconds':0.};audits=[];captured={}
    for row in rows:
        charge(progress,existing,saved_generations=1);audit={};capt={};context=nullcontext()
        if setting:
            block=setting['block'];u=tensors['units'][block];target=tensors['targets'][block]
            c=p.fixed_scalar(target,h0[row['id']][block],u,setting['alpha']) if setting['method']=='fixed' else None
            context=p.intervention(model,block,u,target,setting['alpha'],setting['method'],c,tensors[f'random.{block}'] if random else None,audit)
        capture_context=p.capture(model,p.BLOCKS,capt,first_prefill=True) if capture_baseline else nullcontext()
        with capture_context,context:value=generate_batched(model,tok,[row],batch_size=1,max_new_tokens=cap,seed=42,phase='SAPS')
        for pred in value['predictions']:pred['output']=tok.decode(pred['generated_ids'],skip_special_tokens=True)
        # Consolidate B1 calls without altering their token trajectories.
        bi=len(batches)
        for pred in value['predictions']:pred['batch_index']=bi
        for b in value['batches']:b['batch_index']=bi
        results.extend(value['predictions']);batches.extend(value['batches'])
        for k in timing:timing[k]+=value['timing'][k]
        if setting:audits.append({'id':row['id'],**audit})
        if capture_baseline:captured[row['id']]=capt
        progress['saved_generations']+=1;progress['generation_wall_seconds']+=value['timing']['generation_wall_seconds']
    result={'predictions':results,'batches':batches,'timing':timing,'settings':{'batch_size':1,'max_new_tokens':cap,'seed':42,'do_sample':False,'use_cache':True,'enable_thinking':False}}
    if setting:result['hook_audits']=audits
    return result,captured


def initial(args,s):
    split=read_json(s.run/'inputs/split.json');prompts=read_json(s.run/'inputs/data.json')['prompts']
    with gpu_stage(s,'initial') as (model,tok,out,progress,existing):
        base,h0=generate(s,model,tok,task_rows(split['validation'],prompts),progress,existing,capture_baseline=True)
        control,_=generate(s,model,tok,task_rows(split['validation'],prompts,True),progress,existing)
        save(out/'predictions.json',{'baseline':base,'prompt':control});save_h0(out/'h0.safetensors',h0)
        save(out/'capture.json',{'scope':'read-only first generation prefill, all four candidate raw block outputs, validation only','blocks':list(p.BLOCKS),'source_count':50,'h0_sha256':stream_hash(out/'h0.safetensors')})


def save_h0(file,values):
    save_vectors(file,{f'{id}.{block}':tensor for id,blocks in values.items() for block,tensor in blocks.items()},{'scope':'read-only baseline first-prefill last token; raw post-block; no final norm'})


def read_h0(s,phase):
    from safetensors.torch import load_file
    stage_sources(s,phase);values=load_file(str(s.run/phase/'h0.safetensors'),device='cpu');out={}
    for key,tensor in values.items():
        id,block=key.rsplit('.',1);out.setdefault(id,{})[int(block)]=tensor
    rows=read_json(s.run/'inputs/split.json')['validation' if phase=='initial' else 'test']
    if set(out)!={r['id'] for r in rows} or any(set(v)!=set(p.BLOCKS) for v in out.values()):raise ValueError('baseline capture split/site coverage differs')
    return out


def eligibility(args,s):
    value=initial_gate(s);save(s.run/'eligibility.json',value)
    if not value['passed']:stop(s.run,'elicitation_null')


def direction_identity(s):
    return {'model_id':s.cfg['model_id'],'revision':s.cfg['revision'],'method':'SAPS','basis':'fresh native HF raw post-block outputs, all36 including pre-final-norm block35',
        'normalization':'FP32 mean(style-plain), per-block unit; target from second independent style pass','manifest_sha256':stream_hash(s.run/'manifest.json'),
        'draws_sha256':stream_hash(s.run/'inputs/draws.json'),'data_sha256':stream_hash(s.run/'inputs/data.json')}


def directions(args,s):
    required_elicitation(s);encoded=read_json(s.run/'inputs/rendered.json');train=read_json(s.run/'inputs/split.json')['train']
    with gpu_stage(s,'directions') as (model,tok,out,progress,existing):
        import torch
        accum=torch.zeros(36,model.config.hidden_size)
        def capture_one(which,id):
            charge(progress,existing,teacher_forced_sequences=1);ids=encoded[which][id]['input_ids'];inputs=torch.tensor([ids],device='cuda:0');collected={};torch.cuda.synchronize();began=time.perf_counter()
            with p.capture(model,range(36),collected),torch.inference_mode():model(input_ids=inputs,attention_mask=torch.ones_like(inputs),position_ids=torch.arange(len(ids),device=inputs.device)[None],use_cache=False,logits_to_keep=1)
            torch.cuda.synchronize();progress['prompt_forward_wall_seconds']+=time.perf_counter()-began;progress['teacher_forced_sequences']+=1
            return torch.stack([collected[i] for i in range(36)])
        for row in train:
            plain=capture_one('train_plain',row['id']);styled=capture_one('train_first',row['id']);accum+=styled-plain
        raw=accum/len(train);units=torch.nn.functional.normalize(raw,dim=1)
        if not torch.isfinite(units).all() or (raw.norm(dim=1)<=0).any():raise ValueError('nonfinite/zero direction')
        targets=torch.zeros(36)
        for row in train:targets+=(capture_one('train_target',row['id'])*units).sum(1)
        targets/=len(train);tensors={'raw':raw,'units':units,'targets':targets}
        for block in p.BLOCKS:
            r=torch.randn(model.config.hidden_size,generator=torch.Generator().manual_seed(7200+block));tensors[f'random.{block}']=r/r.norm()
        identity=direction_identity(s);save_vectors(out/'vectors.safetensors',tensors,identity)
        save(out/'vectors.json',{'identity':identity,'vectors_sha256':stream_hash(out/'vectors.safetensors'),'eligibility_sha256':stream_hash(s.run/'eligibility.json'),
            'accepted_initial_sha256':stream_hash(s.run/'grades/initial/accepted.json'),'raw_norms':raw.norm(dim=1).tolist(),'unit_norms':units.norm(dim=1).tolist(),
            'all_blocks':list(range(36)),'candidate_blocks':list(p.BLOCKS),'random_seeds':{str(b):7200+b for b in p.BLOCKS},'round_trip_exact':True,'draws_reconstruction_not_original_runtime_rng_identity':True})


def vectors(s):
    from safetensors import safe_open
    from safetensors.torch import load_file
    required_elicitation(s);stage_sources(s,'directions');record=read_json(s.run/'directions/vectors.json');file=s.run/'directions/vectors.safetensors'
    if record['identity']!=direction_identity(s) or record['vectors_sha256']!=stream_hash(file) or record['eligibility_sha256']!=stream_hash(s.run/'eligibility.json') or record['accepted_initial_sha256']!=stream_hash(s.run/'grades/initial/accepted.json'):raise ValueError('direction/model/eligibility binding differs')
    with safe_open(str(file),framework='pt',device='cpu') as archive:
        if archive.metadata()!=direction_identity(s):raise ValueError('safe direction metadata differs')
    values=load_file(str(file),device='cpu')
    if values['units'].shape!=values['raw'].shape or values['units'].shape[0]!=36 or values['targets'].shape!=(36,):raise ValueError('direction dimensions differ')
    return values


def preflight(args,s):
    tensors=vectors(s);rows=task_rows(read_json(s.run/'inputs/split.json')['validation'][:4],read_json(s.run/'inputs/data.json')['prompts']);block=19
    with gpu_stage(s,'preflight') as (model,tok,out,progress,existing):
        import torch
        a,captured=generate(s,model,tok,rows,progress,existing,cap=32,capture_baseline=True)
        # Actual fixed-zero hook uses the real unit but c=0 on every position.
        bpred=[];baudit=[]
        for row in rows:
            charge(progress,existing,saved_generations=1);audit={}
            with p.intervention(model,block,tensors['units'][block],tensors['targets'][block],1.,'fixed',0.0,audit=audit):value=generate_batched(model,tok,[row],batch_size=1,max_new_tokens=32,seed=42,phase='SAPS actual zero')
            bpred.extend(value['predictions']);baudit.append(audit);progress['saved_generations']+=1;progress['generation_wall_seconds']+=value['timing']['generation_wall_seconds']
        smoke,_=generate(s,model,tok,rows[:2],progress,existing,{'method':'saps','block':block,'alpha':1.},tensors,cap=32)
        ids=rows[0];encoded=tok(render_prompt(tok,ids),add_special_tokens=False)['input_ids'];inputs=torch.tensor([encoded],device='cuda:0')
        with torch.inference_mode():
            charge(progress,existing,logit_forwards=1);normal=model(input_ids=inputs,attention_mask=torch.ones_like(inputs),use_cache=False,logits_to_keep=1).logits.float();progress['logit_forwards']+=1
            with p.intervention(model,block,tensors['units'][block],tensors['targets'][block],1.,'saps'):
                charge(progress,existing,logit_forwards=1);steered=model(input_ids=inputs,attention_mask=torch.ones_like(inputs),use_cache=False,logits_to_keep=1).logits.float();progress['logit_forwards']+=1
        exact=all(x['input_ids']==y['input_ids'] and x['generated_ids']==y['generated_ids'] for x,y in zip(a['predictions'],bpred));delta=(steered-normal).abs().max().item()
        passed=exact and delta>0 and all(v.get('prefill_calls',0)>0 and v.get('decode_calls',0)>0 and v['hooks_removed'] for v in [*baudit,*smoke['hook_audits']])
        passed=passed and all(v['prefill_max_abs_hidden_delta']>0 for v in smoke['hook_audits'])
        save(out/'ordinary.json',a);save(out/'zero.json',{'predictions':bpred,'hook_audits':baudit});save(out/'nonzero.json',smoke)
        save(out/'checks.json',{'passed':passed,'ordinary_with_read_only_capture_vs_actual_zero_hook_exact':exact,
            'parity_reference':'ordinary generation WITH read-only all-four-block first-prefill capture, compared to actual zero intervention',
            'capture_hook_in_ordinary':True,'nonzero_logit_max_abs_delta':delta,'all_four_capture_blocks':list(p.BLOCKS)})
        if not passed:raise ValueError('source mechanics/zero/nonzero/cache probes failed')
    save(s.run/'strength.json',strength_record(s))


def strength_record(s):
    return {'manifest_sha256':stream_hash(s.run/'manifest.json'),**{n:stream_hash(s.run/n) for n in ('eligibility.json','directions/vectors.safetensors','directions/vectors.json','directions/run.json','preflight/checks.json','preflight/run.json','initial/h0.safetensors','initial/run.json')}}


def strength(s):
    committed(s.cfg['strength_registration'])
    if read_json(s.cfg['strength_registration'])['strength_sha256']!=stream_hash(s.run/'strength.json') or read_json(s.run/'strength.json')!=strength_record(s) or not read_json(s.run/'preflight/checks.json')['passed']:raise ValueError('committed vector/preflight binding differs')
    return vectors(s)


def validation(args,s):
    tensors=strength(s);h0=read_h0(s,'initial');rows=task_rows(read_json(s.run/'inputs/split.json')['validation'],read_json(s.run/'inputs/data.json')['prompts'])
    with gpu_stage(s,'validation') as (model,tok,out,progress,existing):
        save(out/'predictions.json',{name:generate(s,model,tok,rows,progress,existing,setting,tensors,h0)[0] for name,setting in p.SETTINGS.items()})
        save(out/'baseline_binding.json',{'initial_run_sha256':stream_hash(s.run/'initial/run.json'),'h0_sha256':stream_hash(s.run/'initial/h0.safetensors'),'initial_accepted_sha256':stream_hash(s.run/'grades/initial/accepted.json')})


def selection_result(s):
    baseline=metric_values(s,'initial');observed=metric_values(s,'validation');record=p.select(observed,baseline['baseline'])
    return {**record,'metrics':observed,'baseline':baseline,'bindings':{'manifest_sha256':stream_hash(s.run/'manifest.json'),'strength_sha256':stream_hash(s.run/'strength.json'),
        **{n:stream_hash(s.run/n) for n in ('initial/run.json','validation/run.json','grades/initial/accepted.json','grades/validation/accepted.json')}}}


def select(args,s):
    strength(s);value=selection_result(s);save(s.run/'selection.json',value)
    if value['selection_null']:stop(s.run,'selection_null')


def selected(s,locked=False):
    record=read_json(s.run/'selection.json')
    if record!=selection_result(s) or record['selection_null']:raise ValueError('selection null/drift')
    if locked:
        strength(s);committed(s.cfg['selection_registration'])
        if read_json(s.cfg['selection_registration'])['selection_sha256']!=stream_hash(s.run/'selection.json'):raise ValueError('committed selection differs')
    return record


def test(args,s):
    chosen=selected(s,True);tensors=vectors(s);split=read_json(s.run/'inputs/split.json');prompts=read_json(s.run/'inputs/data.json')['prompts'];rows=task_rows(split['test'],prompts)
    with gpu_stage(s,'test') as (model,tok,out,progress,existing):
        base,h0=generate(s,model,tok,rows,progress,existing,capture_baseline=True);save_h0(out/'h0.safetensors',h0);results={'baseline':base}
        for method in ('fixed','saps'):
            name=chosen['best'][method]
            if name:results[method]=generate(s,model,tok,rows,progress,existing,p.SETTINGS[name],tensors,h0)[0]
        results['random']=generate(s,model,tok,rows,progress,existing,chosen['setting'],tensors,h0,True)[0]
        results['prompt']=generate(s,model,tok,task_rows(split['test'],prompts,True),progress,existing)[0]
        save(out/'predictions.json',results);save(out/'selection_binding.json',{'selection_sha256':stream_hash(s.run/'selection.json'),'h0_scope':'fresh TEST-only read-only baseline first-prefill all four blocks','validation_h0_not_reused':True})
        save(out/'norm_diagnostics.json',p.norm_diagnostics(results[chosen['setting']['method']]['hook_audits'],results['random']['hook_audits']))


def phase_data(s,phase):
    if phase in ('test','custom'):
        choice=selected(s,True);key=phase+'_'+choice['branch'];generation='test';split='test'
    else:key=phase;generation=phase;split='validation'
    stage_sources(s,generation);values=read_json(s.run/generation/'predictions.json');rows=read_json(s.run/'inputs/split.json')[split]
    plans=read_json(s.run/'inputs/plans.json')[key];arms={e['condition'] for batch in plans for e in batch}
    if set(values)!=arms:raise ValueError('generation/eligible method arm set differs')
    rendered=read_json(s.run/'inputs/rendered.json')
    for name,value in values.items():validate_generation(value,rows,rendered[split+('_prompt' if name=='prompt' else '_plain')],s.m['tokenizer_ids']['eos'],cap=60,batch_size=1)
    return rows,values,plans,generation


def packet_batches(s,phase):
    rows,values,plan,generation=phase_data(s,phase);texts={(name,r['id']):r['output'] for name,v in values.items() for r in v['predictions']};sources={r['id']:r['sentence'] for r in rows}
    return [[{'id':e['id'],'original_complaint':sources[e['question_id']],'generated_sentence':texts[e['condition'],e['question_id']]} for e in batch] for batch in plan]


def packet_record(s,phase):
    generation='test' if phase=='custom' else phase
    return {'manifest_sha256':stream_hash(s.run/'manifest.json'),'packet_sha256':stream_hash(s.run/'judging'/phase/'packet.json'),'plans_sha256':stream_hash(s.run/'inputs/plans.json'),
        'sources':stage_sources(s,generation),'rubric_sha256':stream_hash(s.run/'inputs/rubric.json')}


def packet(args,s):
    if (s.run/'stopped.json').exists():raise RuntimeError('permanent stop')
    batch=packet_batches(s,args.phase);out=s.run/'judging'/args.phase;out.mkdir(mode=0o700,parents=True,exist_ok=False)
    save(out/'packet.json',batch);save(out/'manifest.json',packet_record(s,args.phase))


def bodies(s,phase):
    out=s.run/'judging'/phase
    if read_json(out/'manifest.json')!=packet_record(s,phase) or read_json(out/'packet.json')!=packet_batches(s,phase):raise ValueError('packet/source/plan drift')
    rubric=read_json(s.run/'inputs/rubric.json')['custom' if phase=='custom' else 'reference']
    return [{'rubric':rubric,'items':b,'expected_ids':[r['id'] for r in b],'expected_count':len(b),'return_schema':rubric['output_schema']} for b in read_json(out/'packet.json')]


def parse(outer,ids,custom=False):
    try:
        result=f.result_json(outer.get('result'))
        if not isinstance(result,dict) or set(result)!={'ratings'}:raise ValueError('ratings-only schema')
        return p.validate_ratings(ids,result['ratings'],custom)
    except (ValueError,KeyError,TypeError) as exc:raise f.InvalidSubmission('malformed/incomplete ratings') from exc


def grade_binding(s,phase,body):
    return {'manifest_sha256':stream_hash(s.run/'manifest.json'),'sources':s.m['sources'],'phase':phase,'packet_manifest_sha256':stream_hash(s.run/'judging'/phase/'manifest.json'),
        'payload_sha256':[f.canonical_hash(b) for b in body],'binary_sha256':pv.BINARY_SHA,'cli_version':s.m['cli_version'],'audit_sha256':s.m['audit_sha256'],'argv':f.argv('claude'),'maximum_total_calls':146,'maximum_attempts_per_batch':2}


def audit_journals(s):
    root=s.run/'grades'
    if len(list(root.glob('*/batch-*-attempt-*-start.json')))>146:raise ValueError('146-start budget exceeded')
    for directory in root.iterdir() if root.exists() else []:
        if not directory.is_dir():continue
        if directory.name not in p.PHASES:raise ValueError('unexpected grade phase')
        body=bodies(s,directory.name)
        for file in directory.glob('*attempt*'):
            match=re.fullmatch(r'batch-(\d{3})-attempt-([01])(?:-start|-receipt|-interruption)?\.json',file.name)
            if not match or int(match[1])>=len(body):raise ValueError('stray attempt artifacts')
        for bi,b in enumerate(body):
            first=False
            for ai in (0,1):
                stem=f'batch-{bi:03d}-attempt-{ai}';file=directory/(stem+'.json')
                if first and list(directory.glob(stem+'*')):raise ValueError('post-first-valid attempt')
                if not file.exists():continue
                attempt=read_json(file)
                if attempt.get('canonical_stdin_sha256')!=f.canonical_hash(b):raise ValueError('attempt payload drift')
                try:outer,_=pv.checked_outer(attempt)
                except (RuntimeError,f.ModelMismatch):
                    if not (s.run/'stopped.json').exists():raise
                    continue
                try:parse(outer,b['expected_ids'],directory.name=='custom')
                except f.InvalidSubmission:continue
                first=True


def finish_grades(s,phase,body,directory,bound):
    ratings=[];records=[]
    for bi,b in enumerate(body):
        record={'expected_ids':b['expected_ids'],'canonical_stdin_sha256':f.canonical_hash(b),'attempts':[]};f.save_or_match(directory/f'batch-{bi:03d}-input.json',record)
        for ai in range(2):
            if (s.run/'stopped.json').exists():raise RuntimeError('permanent stop')
            stem=f'batch-{bi:03d}-attempt-{ai}'
            if not list(directory.glob(stem+'*')) and len(list((s.run/'grades').glob('*/batch-*-attempt-*-start.json')))>=146:raise RuntimeError('146-start limit before call')
            attempt,receipt=f.replay_or_call(directory,bi,ai,b,bound,'claude');outer,models=pv.checked_outer(attempt);entry={'receipt':receipt,'actual_model_usage_ids':models,'structural_valid':False};record['attempts'].append(entry)
            try:valid=parse(outer,b['expected_ids'],phase=='custom')
            except f.InvalidSubmission:
                if ai==1:f.save_or_match(directory/f'batch-{bi:03d}-provenance.json',record);raise
                continue
            entry['structural_valid']=True;record['first_valid_attempt']=ai;ratings.extend(valid);break
        f.save_or_match(directory/f'batch-{bi:03d}-provenance.json',record);records.append(record)
    return {'ratings':ratings,'batches':records,'binding_sha256':bound,'first_valid_only':True}


def judge(args,s):
    s.initial();body=bodies(s,args.phase);binary=shutil.which('claude')
    if not binary or Path(binary).resolve()!=path(s.m['binary']).resolve() or stream_hash(Path(binary))!=pv.BINARY_SHA or f.version('claude')!=s.m['cli_version']:raise ValueError('pinned native identity/version required')
    f.validate_audit(read_json(s.cfg['judge_runtime_config'])['judge_audit']);root=s.run/'grades';out=root/args.phase
    if args.resume and not out.exists():raise ValueError('resume requires existing phase')
    if out.exists() and not args.resume:raise FileExistsError('explicit exact journal resume required')
    root.mkdir(mode=0o700,exist_ok=True);out.mkdir(mode=0o700,exist_ok=True)
    with f.exclusive_run(root):
        f.save_or_match(out/'run_binding.json',grade_binding(s,args.phase,body));audit_journals(s)
        accepted=finish_grades(s,args.phase,body,out,stream_hash(out/'run_binding.json'));s.pending_accept=(out/'accepted.json',accepted)


def read_reviews(s,phase):
    body=bodies(s,phase);out=s.run/'grades'/phase
    for file in out.glob('*attempt*'):
        match=re.fullmatch(r'batch-(\d{3})-attempt-([01])(?:-start|-receipt|-interruption)?\.json',file.name)
        if not match or int(match[1])>=len(body):raise ValueError('unregistered saved attempt')
    if read_json(out/'run_binding.json')!=grade_binding(s,phase,body):raise ValueError('grade binding differs')
    bound=stream_hash(out/'run_binding.json');ratings=[];records=[]
    for bi,b in enumerate(body):
        record=read_json(out/f'batch-{bi:03d}-provenance.json');first=record['first_valid_attempt'];base={'expected_ids':b['expected_ids'],'canonical_stdin_sha256':f.canonical_hash(b),'attempts':[]}
        if type(first) is not int or first not in (0,1) or len(record['attempts'])!=first+1 or read_json(out/f'batch-{bi:03d}-input.json')!=base or any(record[k]!=v for k,v in base.items() if k!='attempts'):raise ValueError('first-valid/membership/input differs')
        for ai,entry in enumerate(record['attempts']):
            stem=f'batch-{bi:03d}-attempt-{ai}';file=out/(stem+'.json');start=out/(stem+'-start.json');receipt={'run_binding_sha256':bound,'canonical_stdin_sha256':f.canonical_hash(b),'file':file.name,'sha256':stream_hash(file),'start_sha256':stream_hash(start)}
            if entry['receipt']!=receipt or read_json(out/(stem+'-receipt.json'))!=receipt or any(read_json(start).get(k)!=receipt[k] for k in ('canonical_stdin_sha256','run_binding_sha256')):raise ValueError('receipt/start binding differs')
            attempt=read_json(file);outer,models=pv.checked_outer(attempt)
            if entry['actual_model_usage_ids']!=models or attempt.get('canonical_stdin_sha256')!=f.canonical_hash(b):raise ValueError('attempt/model/payload differs')
            try:valid=parse(outer,b['expected_ids'],phase=='custom')
            except f.InvalidSubmission:
                if ai==first or entry['structural_valid'] is not False:raise
                continue
            if ai!=first or entry['structural_valid'] is not True:raise ValueError('earlier valid not preserved')
            ratings.extend(valid)
        if first==0 and list(out.glob(f'batch-{bi:03d}-attempt-1*')):raise ValueError('extra post-valid attempt')
        records.append(record)
    if read_json(out/'accepted.json')!={'ratings':ratings,'batches':records,'binding_sha256':bound,'first_valid_only':True}:raise ValueError('accepted ratings differ')
    _,_,plan,_=phase_data(s,phase);aliases={e['id']:e for batch in plan for e in batch};decoded={}
    for r in ratings:
        e=aliases[r['id']];decoded.setdefault(e['question_id'],{})[e['condition']]=r
    return decoded


def metric_values(s,phase):
    rows,values,plan,generation=phase_data(s,phase);grades=read_reviews(s,phase)
    return {name:p.metrics(value['predictions'],[{**grades[r['id']][name],'id':r['id']} for r in value['predictions']]) for name,value in values.items()}


def reference_result(s):
    choice=selected(s,True);observed=metric_values(s,'test');return {'metrics':observed,**p.reference_gates(observed,choice['setting']['method'])}


def custom_result(s):
    grades=read_reviews(s,'custom');arms=next(iter(grades.values())).keys();metrics={}
    for arm in arms:
        ratings=[values[arm] for values in grades.values()];metrics[arm]={'n':len(ratings),**{t:sum(r[t]>=2 for r in ratings) for t in ('profanity','rudeness','sarcasm')},
            'ambiguous':{t:sum(r[t]==1 for r in ratings) for t in ('profanity','rudeness','sarcasm')},'content_ok':sum(r['content_ok'] for r in ratings),'incoherent':sum(not r['coherent'] for r in ratings),'gross_failures':sum(r['gross_failure'] for r in ratings)}
    return {'metrics':metrics,'separate_custom_construct':True,'not_a_selection_gate':True}


def retention(args,s):
    choice=selected(s,True)
    if not reference_result(s)['all_criteria_pass']:raise ValueError('reference test criteria failed; skip off-task capability')
    tensors=vectors(s);fixture=read_json(s.run/'inputs/retention.json');setting=choice['setting'];block=setting['block'];prompts=read_json(s.run/'inputs/data.json')['prompts']
    with gpu_stage(s,'retention') as (model,tok,out,progress,existing):
        import torch
        from langdetect import DetectorFactory
        old_seed=DetectorFactory.seed;results={};h0={};hook_audits=[]
        try:
            for condition in ('baseline','selected'):
                directory=out/condition;directory.mkdir();summary={};DetectorFactory.seed=0
                for task in ('general_mc','physics_mc','gsm8k','ifeval'):
                    records=[];rows=fixture[task]
                    for row in rows:
                        key=task+':'+str(row.get('id',row.get('key')));capt={};audit={};context=nullcontext()
                        if condition=='selected':
                            u=tensors['units'][block];target=tensors['targets'][block];c=p.fixed_scalar(target,h0[key][block],u,setting['alpha']) if setting['method']=='fixed' else None
                            context=p.intervention(model,block,u,target,setting['alpha'],setting['method'],c,audit=audit)
                        capture=p.capture(model,p.BLOCKS,capt,first_prefill=True) if condition=='baseline' else nullcontext()
                        with capture,context:
                            if task.endswith('_mc'):
                                charge(progress,existing,mc_items=1);record=hf_multiple_choice(model,tok,[row],batch_size=1)[0];progress['mc_items']+=1
                            else:
                                charge(progress,existing,saved_generations=1);r={'id':str(row['key']),'user':row['prompt']} if task=='ifeval' else row;cap=1024 if task=='ifeval' else 512
                                value=generate_batched(model,tok,[r],batch_size=1,max_new_tokens=cap,seed=42,phase='SAPS off-task capability');record=value['predictions'][0];progress['saved_generations']+=1;progress['generation_wall_seconds']+=value['timing']['generation_wall_seconds']
                        if condition=='baseline':h0[key]=capt
                        else:hook_audits.append({'task':task,'source_key':key,**audit})
                        records.append(record)
                    save(directory/(task+'_predictions.json'),records)
                    if task.endswith('_mc'):summary[task]={'total':len(records),'correct':sum(r['correct'] for r in records)}
                    elif task=='gsm8k':summary[task]={k:v for k,v in score_gsm8k(rows,records).items() if k!='details'}
                    else:summary[task]={k:v for k,v in ifeval_scores(rows,records).items() if k!='records'}
                results[condition]=summary
        finally:DetectorFactory.seed=old_seed
        save(out/'metrics.json',results);save(out/'hook_audits.json',hook_audits);save_h0(out/'h0.safetensors',h0)
        save(out/'selection_binding.json',{'selection_sha256':stream_hash(s.run/'selection.json'),'language_detector_seed':0,'batch_size':1,'scope':'independent off-task transfer, not original paraphrase headline'})


def report(args,s):
    audit_journals(s);costs={file.parent.name:read_json(file) for file in s.run.glob('*/run.json')};stopped=read_json(s.run/'stopped.json') if (s.run/'stopped.json').exists() else None
    result={'status':'incomplete','scope':'Isolated SAPS sarcasm method adaptation, native Qwen3-4B HF basis; no test oracle',
        'manifest_sha256':stream_hash(s.run/'manifest.json'),'source_sha256':s.m['sources'],'model':{'model_id':s.cfg['model_id'],'revision':s.cfg['revision']},
        'costs':{phase:{k:v for k,v in value.items() if k not in ('files_sha256','cuda_visible_devices')} for phase,value in costs.items()},'totals':cost_state(s),
        'calls_started':len(list((s.run/'grades').glob('*/batch-*-attempt-*-start.json'))),'stop':stopped,
        'limits':['Fresh native HF BF16 basis, Qwen3 nonthinking and locked small search/subset differ from author TransformerLens models.',
            'Single isolated Claude co-rates original0..5 scales instead of the authors three-judge average; custom0..3 scores are separate.',
            'Author-scale batches co-rate up to20 items; contrast/context effects may differ from single-item judgments.',
            'Author sarcasm requires sentence-alone internal inversion; calibrated custom sarcasm is contextual. The constructs are not interchangeable.',
            'Source complaints include profanity/hostile language; underlying transformed-data licensing/provenance is incomplete. No independent trait control claim.',
            'Frozen1000 draws reconstruct algorithm, not original runtime RNG consumption. Decoding slices generated IDs rather than author heuristic template stripping.',
            'No per-test layer oracle, human-validity or causal-generalization claim. Missing methods are skipped, not filled with quality failures.',
            'Random control matches scalar controller, not actual trajectory norms; norm-ratio flag is descriptive only. Capability is independent off-task transfer.']}
    if (s.run/'eligibility.json').exists():
        eligibility=initial_gate(s)
        if read_json(s.run/'eligibility.json')!=eligibility:raise ValueError('eligibility drift')
        result['eligibility']={k:v for k,v in eligibility.items() if k!='accepted_sha256'}
    if (s.run/'selection.json').exists():
        choice=selection_result(s)
        if read_json(s.run/'selection.json')!=choice:raise ValueError('selection drift')
        result['selection']={k:v for k,v in choice.items() if k!='bindings'}
    if (s.run/'grades/test/accepted.json').exists():
        result['reference']=reference_result(s);result['norm_diagnostics']=read_json(s.run/'test/norm_diagnostics.json')
        if (s.run/'grades/custom/accepted.json').exists():
            result['custom']=custom_result(s)
            if not result['reference']['all_criteria_pass']:result.update(status='complete',capability={'skipped':'reference_test_failed'})
            elif (s.run/'retention/run.json').exists():
                stage_sources(s,'retention');values=read_json(s.run/'retention/metrics.json');a,b=values['baseline'],values['selected'];gates={
                    'mc':sum(b[k]['correct'] for k in ('general_mc','physics_mc'))>=sum(a[k]['correct'] for k in ('general_mc','physics_mc'))-2,
                    'gsm8k':b['gsm8k']['strict_correct']>=a['gsm8k']['strict_correct']-1,'ifeval':b['ifeval']['prompt_strict_correct']>=a['ifeval']['prompt_strict_correct']-1}
                result.update(status='complete',capability={'metrics':values,'gates':gates,'all_criteria_pass':all(gates.values())})
    if stopped:
        result['status']=stopped['stop_cause'] if stopped['stop_cause'] in ('elicitation_null','selection_null') else 'incomplete'
        if 'reference' in result:result['reference']['all_criteria_pass']=False
    if result['status']=='incomplete' and stopped is None:raise ValueError('exclusive report cannot be published early')
    private_export(result);s.verify();save(path(s.cfg['result']),result)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=('prepare','verify','initial','packet','judge','eligibility','directions','preflight','validation','select','test','retention','report'))
    parser.add_argument('--phase',choices=p.PHASES,default='initial');parser.add_argument('--execute-gpu',action='store_true');parser.add_argument('--execute-judgments',action='store_true');parser.add_argument('--resume',action='store_true');args=parser.parse_args();run=path(RUN)
    if args.stage=='prepare':prepare(args,run);return
    gpu=args.stage in ('initial','directions','preflight','validation','test','retention');live=gpu or args.stage=='judge'
    if gpu and not args.execute_gpu:raise ValueError('separate root release + --execute-gpu required')
    if args.stage=='judge' and not args.execute_judgments:raise ValueError('separate root release + --execute-judgments required')
    guard(run)
    if live and (run/'stopped.json').exists():raise RuntimeError('permanent stop')
    s=None;waiting=False;completed=False
    try:
        s=Context(run).verify(models=live or args.stage=='verify')
        if args.stage=='verify':print('SAPS bindings verified; no calls.');return
        globals()[args.stage](args,s);completed=True
    except pv.ResourceWait:waiting=True;raise
    except BaseException as exc:
        if live and run.exists():stop(run,'live_stage_fatal',exc)
        raise
    finally:
        if live and s is not None and not waiting:
            try:
                s.verify(models=True)
                if completed and hasattr(s,'pending_accept'):f.save_or_match(*s.pending_accept)
            except BaseException as exc:stop(run,'post_call_integrity_failure',exc);raise


if __name__=='__main__':main()
