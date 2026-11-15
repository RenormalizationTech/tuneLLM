"""Separate registered SAPS continuation; failed parent gate is preserved, not passed."""
import argparse
from contextlib import contextmanager, nullcontext
import importlib.metadata
import os
from pathlib import Path
import re
import shutil
import time

from scripts import persona_saps as shared
from scripts import persona_saps_common as p
from scripts import persona_reference as pv
from scripts import tone_judge as f
from scripts.persona_reference_common import validate_generation, private_export
from scripts.batched_generation import generate_batched, render_prompt
from scripts.benchmark_screen import hf_multiple_choice, ifeval_scores
from evaluation.gsm8k import score_gsm8k
from scripts.benign_helpfulness import resource_check
from scripts.common import path, read_json
from scripts.profanity_steering import save_vectors
from scripts.save_nf4 import stream_hash
from scripts.tone_steering import load_model, committed

RUN='outputs/persona-saps-sarcasm-continuation-01'
CONFIG='configs/persona_saps_continuation.json'
PHASES=('validation','test','custom')
save=f.save
stage_sources=shared.stage_sources
task_rows=shared.task_rows
save_h0=shared.save_h0
direction_identity=shared.direction_identity
parse=shared.parse


def expected_config():
    cfg=read_json('configs/persona_saps_sarcasm.json')
    cfg.update(run=RUN,result='results/persona_saps_sarcasm_continuation_01.json',protocol='docs/persona_saps_continuation_protocol.md',
        design_review='results/reviews/persona_saps_continuation_design_review.md',parent='outputs/persona-saps-sarcasm-01',parent_config='configs/persona_saps_sarcasm.json',parent_result='results/persona_saps_sarcasm_01.json',
        initial_registration='results/protocols/persona-saps-sarcasm-continuation-01.json',strength_registration='results/protocols/persona-saps-sarcasm-continuation-strength-01.json',selection_registration='results/protocols/persona-saps-sarcasm-continuation-selection-01.json',
        maximum_generations=1142,maximum_judge_calls=136,maximum_whole_generations=1242,maximum_whole_starts=141,
        extraction_authority='new registered continuation omits extra pre-extraction quality gate; parent gate remains failed')
    return cfg


def validate_config(cfg):
    expected=expected_config()
    if cfg!=expected or f.canonical_hash(cfg)!=f.canonical_hash(expected):raise ValueError('exact parent runtime/plans and child bounds/authority required')


def guard(run):
    if run.resolve()!=path(RUN).resolve():raise ValueError('new continuation path only; parent never writable')


def parent_paths(parent):
    forbidden=('directions','preflight','validation','test','custom','retention','strength.json','selection.json',
        'judging/validation','judging/test','judging/custom','grades/validation','grades/test','grades/custom')
    if any((parent.run/n).exists() for n in forbidden):raise ValueError('original closed parent must have no downstream work')


def parent_bank(parent,cfg):
    parent_paths(parent);gate=shared.initial_gate(parent);report=read_json(cfg['parent_result']);stop=read_json(parent.run/'stopped.json')
    control=gate['metrics']['prompt'];criteria={'style_count':control['style_at_least3']>=25,'style_mean':control['sarcasm']>=3,
        'grammar':control['grammaticality']>=3.5,'clarity':control['clarity']>=3.5}
    if gate!=read_json(parent.run/'eligibility.json') or gate['passed'] is not False or [k for k,v in criteria.items() if not v]!=['clarity']:
        raise ValueError('original failed extra gate must remain failed solely on clarity')
    if control['clarity']!=3.46 or control['sarcasm']!=3.04 or control['grammaticality']!=4.64 or control['style_at_least3']!=36 or control['entailment']!=2.44:raise ValueError('known parent control bank differs')
    if report['status']!='elicitation_null' or report['eligibility']!={k:v for k,v in gate.items() if k!='accepted_sha256'} or stop!={'stop_cause':'elicitation_null','error_type':None}:raise ValueError('parent null report/stop changed')
    parent.initial();shared.read_h0(parent,'initial');stage_sources(parent,'initial')
    accepted=read_json(parent.run/'grades/initial/accepted.json');cost=read_json(parent.run/'initial/run.json')
    if len(accepted['ratings'])!=100 or len(accepted['batches'])!=5 or any(x['first_valid_attempt']!=0 for x in accepted['batches']) or len(list((parent.run/'grades/initial').glob('*-start.json')))!=5:raise ValueError('exact five first-valid parent batches required')
    if cost['saved_generations']!=100 or any(cost[k] for k in ('teacher_forced_sequences','logit_forwards','mc_items')):raise ValueError('parent work must be initial100 only')
    return {'original_gate_passed':False,'failed_gate':gate,'failed_components':['clarity'],'metrics':gate['metrics'],'costs':{'initial':cost},
        'totals':{k:cost[k] for k in ('saved_generations','teacher_forced_sequences','logit_forwards','mc_items')},'native_starts':5,'firstvalid_batches':5,
        'parent_result_sha256':stream_hash(path(cfg['parent_result'])),'parent_stop_sha256':stream_hash(parent.run/'stopped.json'),
        'initial_accepted_sha256':stream_hash(parent.run/'grades/initial/accepted.json'),'h0_sha256':stream_hash(parent.run/'initial/h0.safetensors')}


def prepare(args,run):
    guard(run)
    if run.exists():raise FileExistsError('one fresh continuation required')
    cfg=read_json(CONFIG);validate_config(cfg);parent=shared.Context(path(cfg['parent']));bank=parent_bank(parent,cfg)
    sources=pv.source_closure(['scripts/persona_saps_continuation.py','scripts/persona_saps_continuation_runbook.md',CONFIG])
    sourcehash={n:stream_hash(path(n)) for n in sources}
    external={**parent.m['sources'],**parent.m['external_sha256'],**{str(parent.run/n):h for n,h in parent.m['files'].items()},**stage_sources(parent,'initial')}
    for n in (cfg['protocol'],cfg['design_review'],cfg['parent_result'],parent.cfg['initial_registration']):external[n]=stream_hash(path(n))
    for n in ('manifest.json','eligibility.json','stopped.json'):external[str(parent.run/n)]=stream_hash(parent.run/n)
    for folder in ('grades/initial','judging/initial'):
        for file in (parent.run/folder).iterdir():
            if file.is_file() and file.name!='.resume.lock':external[str(file)]=stream_hash(file)
    pv.check_hashes([sourcehash,external,parent.m['model_sha256']])
    run.mkdir(mode=0o700);save(run/'config.json',cfg)
    for name in ('data','split','plans','draws','rendered','rubric','retention'):
        target=run/'inputs'/(name+'.json');target.parent.mkdir(exist_ok=True);shutil.copyfile(parent.run/'inputs'/(name+'.json'),target)
    save(run/'inputs/parent_bank.json',bank)
    for n in sources:
        target=run/'source'/n;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path(n),target)
    save(run/'manifest.json',{'files':{str(file.relative_to(run)):stream_hash(file) for file in run.rglob('*') if file.is_file()},'sources':sourcehash,'external_sha256':external,
        'snapshot':parent.m['snapshot'],'model_sha256':parent.m['model_sha256'],'tokenizer_ids':parent.m['tokenizer_ids'],'packages':parent.m['packages'],
        'binary':parent.m['binary'],'binary_sha256':parent.m['binary_sha256'],'cli_version':parent.m['cli_version'],'audit_sha256':parent.m['audit_sha256'],
        'parent_manifest_sha256':stream_hash(parent.run/'manifest.json'),'parent_result_sha256':bank['parent_result_sha256'],'original_gate_passed':False,
        'new_authority':'explicit continuation registration omits our extra pre-extraction gate; no relabelling of parent failure','no_live_calls':True})
    pv.check_hashes([sourcehash,external])


class Context:
    def __init__(self,run):
        guard(run);self.run=run;self.cfg=read_json(run/'config.json');validate_config(self.cfg);self.m=read_json(run/'manifest.json');self.parent=shared.Context(path(self.cfg['parent']))

    def verify(self,models=False):
        mappings=[{str(self.run/n):h for n,h in self.m['files'].items()},self.m['sources'],self.m['external_sha256']]
        if models:mappings.append(self.m['model_sha256'])
        pv.check_hashes(mappings);parent_paths(self.parent)
        if any(importlib.metadata.version(k)!=v for k,v in self.m['packages'].items()):raise ValueError('runtime package changed')
        bank=read_json(self.run/'inputs/parent_bank.json')
        if bank['original_gate_passed'] is not False or self.m['original_gate_passed'] is not False or read_json(self.parent.run/'eligibility.json')!=bank['failed_gate']:
            raise ValueError('parent failed gate must not be relabelled')
        if self.m['parent_manifest_sha256']!=stream_hash(self.parent.run/'manifest.json') or self.m['parent_result_sha256']!=stream_hash(path(self.cfg['parent_result'])):raise ValueError('parent bank/report binding differs')
        for n in ('data','split','plans','draws','rendered','rubric','retention'):
            if (self.run/'inputs'/(n+'.json')).read_bytes()!=(self.parent.run/'inputs'/(n+'.json')).read_bytes():raise ValueError('exact frozen parent input bytes required')
        return self

    def initial(self):
        for n in (*self.m['sources'],self.cfg['protocol'],self.cfg['design_review'],self.cfg['initial_registration']):committed(n)
        if read_json(self.cfg['initial_registration'])['manifest_sha256']!=stream_hash(self.run/'manifest.json'):raise ValueError('new continuation registration required')


def require_authority(s):
    s.initial();bank=read_json(s.run/'inputs/parent_bank.json')
    if bank['original_gate_passed'] is not False or bank['failed_components']!=['clarity']:raise ValueError('failed parent bank, not a fake eligibility pass, required')


def cost_state(s):return {k:sum(read_json(file).get(k,0) for file in s.run.glob('*/run.json')) for k in ('saved_generations','teacher_forced_sequences','logit_forwards','mc_items')}


def charge(progress,existing,**amounts):
    limits={'saved_generations':1142,'teacher_forced_sequences':1500,'logit_forwards':2,'mc_items':64}
    for k,amount in amounts.items():
        if progress[k]+existing[k]+amount>limits[k]:raise ValueError('fresh continuation budget exceeded BEFORE work')


@contextmanager
def gpu_stage(s,name):
    s.initial()
    try:resource=resource_check(s.cfg)
    except RuntimeError as exc:raise pv.ResourceWait('resources busy; no continuation phase started') from exc
    out=s.run/name;out.mkdir(mode=0o700,exist_ok=False)
    import torch
    began=time.perf_counter();completed=False;progress={k:0 for k in ('saved_generations','teacher_forced_sequences','logit_forwards','mc_items')};progress.update(generation_wall_seconds=0.,prompt_forward_wall_seconds=0.);existing=cost_state(s)
    try:
        model,tok=load_model(s.cfg,s.m,'bf16')
        if model.config.num_hidden_layers!=36 or any(v.dtype!=torch.bfloat16 or v.device!=torch.device('cuda:0') for v in model.parameters()):raise ValueError('same36-block native BF16 single-GPU basis required')
        torch.cuda.reset_peak_memory_stats();yield model,tok,out,progress,existing;torch.cuda.synchronize();completed=True
    finally:
        save(out/'run.json',{'completed':completed,**resource,**progress,'elapsed_wall_seconds':time.perf_counter()-began,
            'count_scope':'Saved/completed fresh child work only; interrupted unsaved calls may add cost','wall_scope':'synchronized whole stage including load/capture/tokenization/serialization',
            'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'memory_scope':'PyTorch allocator, not total VRAM',
            'dtype':'bfloat16','attention':'sdpa','seed':42,'torch_cuda_runtime':torch.version.cuda,'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),'packages':s.m['packages'],
            'tf32_matmul':torch.backends.cuda.matmul.allow_tf32,'tf32_cudnn':torch.backends.cudnn.allow_tf32,'files_sha256':{str(file.relative_to(out)):stream_hash(file) for file in out.rglob('*') if file.is_file()}})


def generate(s,model,tok,rows,progress,existing,setting=None,tensors=None,h0=None,random=False,cap=60,capture_baseline=False):
    charge(progress,existing,saved_generations=len(rows))
    return shared.generate(s,model,tok,rows,progress,existing,setting,tensors,h0,random,cap,capture_baseline)


def vectors(s):
    from safetensors import safe_open
    from safetensors.torch import load_file
    require_authority(s);stage_sources(s,'directions');record=read_json(s.run/'directions/vectors.json');file=s.run/'directions/vectors.safetensors'
    if record['identity']!=direction_identity(s) or record['vectors_sha256']!=stream_hash(file) or record['parent_bank_sha256']!=stream_hash(s.run/'inputs/parent_bank.json') or record['accepted_initial_sha256']!=stream_hash(s.parent.run/'grades/initial/accepted.json'):raise ValueError('new direction/failed-bank binding differs')
    with safe_open(str(file),framework='pt',device='cpu') as archive:
        if archive.metadata()!=direction_identity(s):raise ValueError('continuation safetensor identity differs')
    tensors=load_file(str(file),device='cpu')
    if tensors['units'].shape!=tensors['raw'].shape or tensors['units'].shape[0]!=36 or tensors['targets'].shape!=(36,):raise ValueError('all36 block directions required')
    return tensors


def strength_record(s):
    return {'manifest_sha256':stream_hash(s.run/'manifest.json'),'parent_initial_sha256':stream_hash(s.parent.run/'initial/run.json'),'parent_h0_sha256':stream_hash(s.parent.run/'initial/h0.safetensors'),
        **{n:stream_hash(s.run/n) for n in ('inputs/parent_bank.json','directions/vectors.safetensors','directions/vectors.json','directions/run.json','preflight/checks.json','preflight/run.json')}}


def strength(s):
    committed(s.cfg['strength_registration'])
    if read_json(s.cfg['strength_registration'])['strength_sha256']!=stream_hash(s.run/'strength.json') or read_json(s.run/'strength.json')!=strength_record(s) or not read_json(s.run/'preflight/checks.json')['passed']:raise ValueError('new vector/preflight registration differs')
    return vectors(s)


def validation(args,s):
    tensors=strength(s);h0=shared.read_h0(s.parent,'initial');rows=task_rows(read_json(s.run/'inputs/split.json')['validation'],read_json(s.run/'inputs/data.json')['prompts'])
    with gpu_stage(s,'validation') as (model,tok,out,progress,existing):
        save(out/'predictions.json',{name:generate(s,model,tok,rows,progress,existing,setting,tensors,h0)[0] for name,setting in p.SETTINGS.items()})
        save(out/'baseline_binding.json',{'parent_initial_run_sha256':stream_hash(s.parent.run/'initial/run.json'),'parent_h0_sha256':stream_hash(s.parent.run/'initial/h0.safetensors'),'parent_accepted_sha256':stream_hash(s.parent.run/'grades/initial/accepted.json'),'rerated_or_regenerated':False})


def selection_result(s):
    initial=read_json(s.run/'inputs/parent_bank.json')['metrics'];observed=metric_values(s,'validation');choice=p.select(observed,initial['baseline'])
    return {**choice,'metrics':observed,'reused_initial_metrics':initial,'parent_gate_remains_failed':True,'bindings':{'manifest_sha256':stream_hash(s.run/'manifest.json'),
        'strength_sha256':stream_hash(s.run/'strength.json'),'parent_bank_sha256':stream_hash(s.run/'inputs/parent_bank.json'),
        'validation_run_sha256':stream_hash(s.run/'validation/run.json'),'validation_accepted_sha256':stream_hash(s.run/'grades/validation/accepted.json')}}


def phase_data(s,phase):
    if phase not in PHASES:raise ValueError('no initial generation or rerating API in continuation')
    if phase in ('test','custom'):
        choice=selected(s,True);key=phase+'_'+choice['branch'];generation='test';split='test'
    else:key=phase;generation=phase;split='validation'
    stage_sources(s,generation);values=read_json(s.run/generation/'predictions.json');rows=read_json(s.run/'inputs/split.json')[split];plans=read_json(s.run/'inputs/plans.json')[key]
    if set(values)!={e['condition'] for batch in plans for e in batch}:raise ValueError('locked eligible arms differ')
    rendered=read_json(s.run/'inputs/rendered.json')
    for name,value in values.items():validate_generation(value,rows,rendered[split+('_prompt' if name=='prompt' else '_plain')],s.m['tokenizer_ids']['eos'],cap=60,batch_size=1)
    return rows,values,plans,generation


def packet_record(s,phase):
    return shared.packet_record(s,phase)

def directions(args,s):
    require_authority(s);encoded=read_json(s.run/'inputs/rendered.json');train=read_json(s.run/'inputs/split.json')['train']
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
        save(out/'vectors.json',{'identity':identity,'vectors_sha256':stream_hash(out/'vectors.safetensors'),'parent_bank_sha256':stream_hash(s.run/'inputs/parent_bank.json'),
            'accepted_initial_sha256':stream_hash(s.parent.run/'grades/initial/accepted.json'),'raw_norms':raw.norm(dim=1).tolist(),'unit_norms':units.norm(dim=1).tolist(),
            'all_blocks':list(range(36)),'candidate_blocks':list(p.BLOCKS),'random_seeds':{str(b):7200+b for b in p.BLOCKS},'round_trip_exact':True,'draws_reconstruction_not_original_runtime_rng_identity':True})

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

def select(args,s):
    strength(s);value=selection_result(s);save(s.run/'selection.json',value)
    if value['selection_null']:pv.stop(s.run,'selection_null')

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

def packet_batches(s,phase):
    rows,values,plan,generation=phase_data(s,phase);texts={(name,r['id']):r['output'] for name,v in values.items() for r in v['predictions']};sources={r['id']:r['sentence'] for r in rows}
    return [[{'id':e['id'],'original_complaint':sources[e['question_id']],'generated_sentence':texts[e['condition'],e['question_id']]} for e in batch] for batch in plan]

def packet(args,s):
    if (s.run/'stopped.json').exists():raise RuntimeError('permanent stop')
    batch=packet_batches(s,args.phase);out=s.run/'judging'/args.phase;out.mkdir(mode=0o700,parents=True,exist_ok=False)
    save(out/'packet.json',batch);save(out/'manifest.json',packet_record(s,args.phase))

def bodies(s,phase):
    out=s.run/'judging'/phase
    if read_json(out/'manifest.json')!=packet_record(s,phase) or read_json(out/'packet.json')!=packet_batches(s,phase):raise ValueError('packet/source/plan drift')
    rubric=read_json(s.run/'inputs/rubric.json')['custom' if phase=='custom' else 'reference']
    return [{'rubric':rubric,'items':b,'expected_ids':[r['id'] for r in b],'expected_count':len(b),'return_schema':rubric['output_schema']} for b in read_json(out/'packet.json')]

def grade_binding(s,phase,body):
    return {'manifest_sha256':stream_hash(s.run/'manifest.json'),'sources':s.m['sources'],'phase':phase,'packet_manifest_sha256':stream_hash(s.run/'judging'/phase/'manifest.json'),
        'payload_sha256':[f.canonical_hash(b) for b in body],'binary_sha256':pv.BINARY_SHA,'cli_version':s.m['cli_version'],'audit_sha256':s.m['audit_sha256'],'argv':f.argv('claude'),'maximum_total_calls':136,'maximum_attempts_per_batch':2}

def audit_journals(s):
    root=s.run/'grades'
    if len(list(root.glob('*/batch-*-attempt-*-start.json')))>136:raise ValueError('136-start budget exceeded')
    for directory in root.iterdir() if root.exists() else []:
        if not directory.is_dir():continue
        if directory.name not in PHASES:raise ValueError('unexpected grade phase')
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
            if not list(directory.glob(stem+'*')) and len(list((s.run/'grades').glob('*/batch-*-attempt-*-start.json')))>=136:raise RuntimeError('136-start limit before call')
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
    audit_journals(s);bank=read_json(s.run/'inputs/parent_bank.json');costs={file.parent.name:read_json(file) for file in s.run.glob('*/run.json')};new=cost_state(s)
    clean=lambda records:{name:{k:v for k,v in value.items() if k not in ('files_sha256','cuda_visible_devices')} for name,value in records.items()}
    stopped=read_json(s.run/'stopped.json') if (s.run/'stopped.json').exists() else None;starts=len(list((s.run/'grades').glob('*/batch-*-attempt-*-start.json')))
    result={'status':'incomplete','scope':'Separate exploratory SAPS continuation; original extra pre-extraction gate failed and parent elicitation_null is unchanged',
        'manifest_sha256':stream_hash(s.run/'manifest.json'),'source_sha256':s.m['sources'],'model':{'model_id':s.cfg['model_id'],'revision':s.cfg['revision']},
        'parent':{'status':'elicitation_null','original_gate_passed':False,'failed_gate_metrics':bank['metrics'],'costs':clean(bank['costs']),'totals':bank['totals'],'native_starts':5,'firstvalid_batches':5,
            'parent_manifest_sha256':s.m['parent_manifest_sha256'],'parent_result_sha256':s.m['parent_result_sha256'],'initial_grade_reused_not_rerated':True},
        'child':{'authority':s.cfg['extraction_authority'],'costs':clean(costs),'totals':new,'native_starts':starts,'maximum_new_starts':136},
        'whole_totals':{k:new[k]+bank['totals'][k] for k in new},'whole_native_starts':starts+5,'maximum_whole_starts':141,'stop':stopped,
        'limits':['Parent control clarity3.46 failed3.5 floor; entailment2.44 versus baseline4.60 is a pre-stated quality risk, not a passed gate.',
            'The omitted pre-extraction quality check was our extra gate, not author prompt-based extraction. Downstream floors and held-out locks are unchanged.',
            'Reuse across old/new judging sessions limits ordinal comparisons; first-valid parent100 answers/grades and h0 remain unchanged.',
            'Fresh native HF BF16 Qwen3, nonthinking/token-slice decoding, reconstructed1000 draws differ from author TransformerLens/runtime RNG.',
            'One tools-off Claude co-rates up to20 original-scale items instead of three author judges; contrast effects and contextual custom versus sentence-alone sarcasm limit interpretation.',
            'Source complaints include profanity/hostility and incomplete transformed-data lineage. No human-validity or independent/general assistant-persona-control claim.',
            'Random matches scalar controller, not actual trajectory norms; per-position norm ratio is report-only. Capability is independent off-task transfer.',
            'No original null is relabelled, no additional initial responses/ratings, no test oracle, new model size or joint traits.']}
    if (s.run/'selection.json').exists():
        choice=selection_result(s)
        if choice!=read_json(s.run/'selection.json'):raise ValueError('continuation selection drift')
        result['selection']={k:v for k,v in choice.items() if k!='bindings'}
    if (s.run/'grades/test/accepted.json').exists():
        result['reference']=reference_result(s);result['norm_diagnostics']=read_json(s.run/'test/norm_diagnostics.json')
        if (s.run/'grades/custom/accepted.json').exists():
            result['custom']=custom_result(s)
            if not result['reference']['all_criteria_pass']:result.update(status='complete',capability={'skipped':'reference_report_failed'})
            elif (s.run/'retention/run.json').exists():
                stage_sources(s,'retention');values=read_json(s.run/'retention/metrics.json');a,b=values['baseline'],values['selected'];gates={
                    'mc':sum(b[k]['correct'] for k in ('general_mc','physics_mc'))>=sum(a[k]['correct'] for k in ('general_mc','physics_mc'))-2,
                    'gsm8k':b['gsm8k']['strict_correct']>=a['gsm8k']['strict_correct']-1,'ifeval':b['ifeval']['prompt_strict_correct']>=a['ifeval']['prompt_strict_correct']-1}
                result.update(status='complete',capability={'metrics':values,'gates':gates,'all_criteria_pass':all(gates.values())})
    if stopped:
        result['status']='selection_null' if stopped['stop_cause']=='selection_null' else 'incomplete'
        if 'reference' in result:result['reference']['all_criteria_pass']=False
    if result['status']=='incomplete' and stopped is None:raise ValueError('exclusive result waits for closure')
    private_export(result);s.verify();save(path(s.cfg['result']),result)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=('prepare','verify','directions','preflight','validation','packet','judge','select','test','retention','report'))
    parser.add_argument('--phase',choices=PHASES,default='validation');parser.add_argument('--execute-gpu',action='store_true');parser.add_argument('--execute-judgments',action='store_true');parser.add_argument('--resume',action='store_true');args=parser.parse_args();run=path(RUN)
    if args.stage=='prepare':prepare(args,run);return
    gpu=args.stage in ('directions','preflight','validation','test','retention');live=gpu or args.stage=='judge'
    if gpu and not args.execute_gpu:raise ValueError('separate root release + --execute-gpu required')
    if args.stage=='judge' and not args.execute_judgments:raise ValueError('separate root release + --execute-judgments required')
    guard(run)
    if live and (run/'stopped.json').exists():raise RuntimeError('continuation permanently stopped')
    s=None;waiting=False;completed=False
    try:
        s=Context(run).verify(models=live or args.stage=='verify')
        if args.stage=='verify':print('Continuation bindings verified; failed parent gate preserved; no calls.');return
        globals()[args.stage](args,s);completed=True
    except pv.ResourceWait:waiting=True;raise
    except BaseException as exc:
        if live and run.exists():pv.stop(run,'continuation_live_stage_fatal',exc)
        raise
    finally:
        if live and s is not None and not waiting:
            try:
                s.verify(models=True)
                if completed and hasattr(s,'pending_accept'):f.save_or_match(*s.pending_accept)
            except BaseException as exc:pv.stop(run,'continuation_post_call_integrity_failure',exc);raise


if __name__=='__main__':main()
