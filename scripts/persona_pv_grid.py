"""Separate bounded symmetric PV grid; frozen parent results are read-only inputs."""
import argparse
from contextlib import contextmanager, nullcontext
import importlib.metadata
import os
from pathlib import Path
import re
import shutil
import time
from types import SimpleNamespace

from scripts import persona_pv_grid_common as p
from scripts import persona_reference as pv
from scripts import persona_reference_common as ref
from scripts import persona_profanity_artifacts as actor
from scripts import tone_judge as f
from scripts.benchmark_screen import hf_multiple_choice, ifeval_scores
from scripts.batched_generation import generate_batched
from evaluation.gsm8k import score_gsm8k
from scripts.benign_helpfulness import resource_check
from scripts.common import path, read_json
from scripts.save_nf4 import stream_hash
from scripts.tone_steering import load_model, committed

save=f.save


def config_path(trait):return f'configs/persona_pv_grid_{trait}.json'


def guard(run,trait):
    if trait not in ('rudeness','profanity') or run.resolve()!=path(f'outputs/persona-pv-grid-{trait}-01').resolve():raise ValueError('new child output only; parent writes forbidden')


def validate_config(cfg):
    trait=cfg.get('trait')
    if trait not in ('rudeness','profanity'):raise ValueError('isolated child profile required')
    fixed={'trait':trait,'run':f'outputs/persona-pv-grid-{trait}-01','result':f'results/persona_pv_grid_{trait}_01.json',
        'parent':f'outputs/persona-reference-{trait}-01','parent_config':f'configs/persona_reference_{trait}.json',
        'protocol':'docs/persona_pv_grid_protocol.md','design_review':'results/reviews/persona_pv_grid_design_review.md',
        'initial_registration':f'results/protocols/persona-pv-grid-{trait}-01.json','selection_registration':f'results/protocols/persona-pv-grid-{trait}-selection-01.json',
        'prospective_amendment':'results/protocols/persona-reference-profanity-grid-amendment-01.json' if trait=='profanity' else None,
        'parent_handoff':'results/protocols/persona-reference-profanity-grid-handoff-01.json' if trait=='profanity' else None,
        'parent_result':'results/persona_reference_rudeness_01.json' if trait=='rudeness' else None,
        'plan_seeds':{'grid':731,'final':732,'custom':733} if trait=='rudeness' else {'grid':741,'final':742,'custom':743},'author_layers':list(p.LAYERS),'coefficients':[.5,1.,2.],
        'maximum_new_generations':232 if trait=='rudeness' else 352,'maximum_new_starts':60 if trait=='rudeness' else 90,'maximum_new_mc':64,
        'lexical_result':'results/diagnostics/persona_pv_grid_profanity_lexical_01.json' if trait=='profanity' else None}
    if cfg!=fixed or f.canonical_hash(cfg)!=f.canonical_hash(fixed):raise ValueError('registered symmetric grid/budgets/paths differ')


def parent_paths(parent,cfg):
    if any((parent.run/n).exists() for n in ('final','custom','report','retention','capability','judging/final','judging/custom','grades/final','grades/custom')):raise ValueError('parent report partition must remain ungenerated/unjudged')
    if cfg['trait']=='profanity' and any((parent.run/n).exists() for n in ('layer','fallback','strength','strength_fallback')):raise ValueError('original profanity selection forbidden after prospective handoff')


def parent_scope(parent,cfg):
    """Only completed used phases; never recursively capture unrelated old outputs."""
    pv.strength(parent);used={str(parent.run/'manifest.json'):stream_hash(parent.run/'manifest.json'),str(parent.run/'strength.json'):stream_hash(parent.run/'strength.json')}
    parent_paths(parent,cfg)
    phases=['elicit','extraction','preflight'];grade_phases=['elicit']
    if cfg['trait']=='rudeness':
        report=read_json(cfg['parent_result']);selection=read_json(parent.run/'selection.json');choice=pv.layer_lock(parent)
        if report['status']!='selection_null' or not selection['selection_null'] or choice['author_layer']!=8 or choice['block']!=7 or choice['fallback_used']:raise ValueError('expected closed rudeness parent/reuse layout differs')
        if selection!=pv.selection_result(parent):raise ValueError('parent null does not recompute')
        phases+=['layer','strength'];grade_phases+=['layer','strength']
        for n in ('selection.json','stopped.json','layer_choice.json'):used[str(parent.run/n)]=stream_hash(parent.run/n)
        for n in (cfg['parent_result'],parent.cfg['layer_registration']):used[n]=stream_hash(path(n))
    else:
        forbidden=('layer','fallback','strength','strength_fallback','final','retention')
        if any((parent.run/n).exists() for n in forbidden) or (parent.run/'selection.json').exists():raise ValueError('profanity handoff must precede ALL original selection outputs')
        for n in (cfg['prospective_amendment'],cfg['parent_handoff']):committed(n);used[n]=stream_hash(path(n))
        amendment=read_json(cfg['prospective_amendment'])
        if amendment.get('parent_manifest_sha256')!=stream_hash(parent.run/'manifest.json') or amendment.get('child_run')!=cfg['run'] or amendment.get('no_steered_parent_outputs_at_amendment') is not True:raise ValueError('actual prospective amendment binding differs')
        handoff=read_json(cfg['parent_handoff'])
        if read_json(parent.run/'stopped.json').get('stop_cause')!='amended_before_steering' or handoff.get('status')!='amended_before_steering':raise ValueError('permanent amended-before-steering parent stop/handoff required')
        used[str(parent.run/'stopped.json')]=stream_hash(parent.run/'stopped.json')
        if handoff.get('parent_manifest_sha256')!=stream_hash(parent.run/'manifest.json') or handoff.get('parent_strength_sha256')!=stream_hash(parent.run/'strength.json') or handoff.get('original_selection_superseded') is not True:
            raise ValueError('prospective parent handoff binding differs')
        if handoff.get('amendment_sha256')!=stream_hash(path(cfg['prospective_amendment'])):raise ValueError('handoff amendment differs')
    for phase in phases:used.update(pv.phase_sources(parent,phase))
    for phase in grade_phases:
        pv.read_reviews(parent,phase)
        for file in (parent.run/'grades'/phase).iterdir():
            if file.is_file() and file.name!='.resume.lock':used[str(file)]=stream_hash(file)
        for file in (parent.run/'judging'/phase).iterdir():
            if file.is_file():used[str(file)]=stream_hash(file)
    for n in (parent.cfg['initial_registration'],parent.cfg['strength_registration']):used[n]=stream_hash(path(n))
    costs={phase:read_json(parent.run/phase/'run.json') for phase in phases};totals={k:sum(v.get(k,0) for v in costs.values()) for k in ('saved_generations','teacher_forced_sequences','mc_items')}
    starts=sum(len(list((parent.run/'grades'/phase).glob('*-start.json'))) for phase in grade_phases)
    if totals['saved_generations']!=(350 if cfg['trait']=='rudeness' else 210):raise ValueError('parent generation scope differs')
    return used,{'phases':phases,'grade_phases':grade_phases,'costs':costs,'totals':totals,'native_starts':starts}


def artifact_origin():
    artifact=actor.Context(path(actor.RUN)).verify();actor.accepted_artifacts(artifact);origin=read_json(artifact.cfg['origin_record']);actual=actor.origin_bindings(artifact)
    if actual!=origin['private_bindings']:raise ValueError('private profanity origin differs')
    pv.check_hashes([actual]);return artifact,origin,actual


def prepare(args,run):
    cfg=read_json(config_path(args.trait));validate_config(cfg);guard(run,args.trait)
    if run.exists():raise FileExistsError('fresh child only')
    parent=pv.Context(path(cfg['parent']),args.trait);parent.initial();used,costs=parent_scope(parent,cfg)
    sources=pv.source_closure(['scripts/persona_pv_grid.py','scripts/persona_pv_grid_common.py','scripts/persona_pv_grid_runbook.md',config_path(args.trait)])
    sourcehash={n:stream_hash(path(n)) for n in sources}
    external={**parent.m['sources'],**parent.m['external_sha256'],**{str(parent.run/n):h for n,h in parent.m['files'].items()},**used}
    for n in (cfg['protocol'],cfg['design_review']):external[n]=stream_hash(path(n))
    if args.trait=='profanity':
        _,_,private=artifact_origin();external.update(private)
    pv.check_hashes([sourcehash,external,parent.m['model_sha256']])
    split=read_json(parent.run/'inputs/split.json');plans=p.plans(split,args.trait,cfg['plan_seeds']);reuse=p.reuse_map(args.trait)
    run.mkdir(mode=0o700);save(run/'config.json',cfg)
    for name in ('split','rendered','rubric','custom_rubric','eval','retention'):
        value=read_json(parent.run/'inputs'/(name+'.json'));save(run/'inputs'/(name+'.json'),value)
    save(run/'inputs/plans.json',plans);save(run/'inputs/reuse.json',reuse);save(run/'inputs/parent_costs.json',costs)
    if args.trait=='profanity':save(run/'inputs/lexical_spec.json',{'frozen_actor_spec':actor.lexical_spec(),'scope':'all parent plus child generated conditions; separate author/custom thresholds, no new gates','implementation_sha256':stream_hash(Path(__file__))})
    for n in sources:
        target=run/'source'/n;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path(n),target)
    save(run/'manifest.json',{'files':{str(file.relative_to(run)):stream_hash(file) for file in run.rglob('*') if file.is_file()},'sources':sourcehash,'external_sha256':external,
        'snapshot':parent.m['snapshot'],'model_sha256':parent.m['model_sha256'],'tokenizer_ids':parent.m['tokenizer_ids'],'packages':parent.m['packages'],
        'binary':parent.m['binary'],'binary_sha256':parent.m['binary_sha256'],'cli_version':parent.m['cli_version'],'audit_sha256':parent.m['audit_sha256'],
        'parent_manifest_sha256':stream_hash(parent.run/'manifest.json'),'parent_strength_sha256':stream_hash(parent.run/'strength.json'),'reused_grid_outputs':len(reuse)*10,'new_grid_outputs':len(p.missing(args.trait))*10,'no_live_calls':True})
    pv.check_hashes([sourcehash,external])


class Context:
    def __init__(self,run,trait):
        guard(run,trait);self.run=run;self.cfg=read_json(run/'config.json');validate_config(self.cfg);self.m=read_json(run/'manifest.json')
        if self.cfg['trait']!=trait:raise ValueError('child/profile mismatch')
        self.parent=pv.Context(path(self.cfg['parent']),trait);self.runtime=self.parent.cfg

    def verify(self,models=False):
        mappings=[{str(self.run/n):h for n,h in self.m['files'].items()},self.m['sources'],self.m['external_sha256']]
        if models:mappings.append(self.m['model_sha256'])
        pv.check_hashes(mappings)
        if any(importlib.metadata.version(k)!=v for k,v in self.m['packages'].items()):raise ValueError('package versions differ')
        if self.m['parent_manifest_sha256']!=stream_hash(self.parent.run/'manifest.json') or self.m['parent_strength_sha256']!=stream_hash(self.parent.run/'strength.json'):raise ValueError('parent vector/snapshot binding differs')
        if read_json(self.run/'inputs/reuse.json')!=p.reuse_map(self.cfg['trait']) or read_json(self.run/'inputs/plans.json')!=p.plans(read_json(self.run/'inputs/split.json'),self.cfg['trait'],self.cfg['plan_seeds']):raise ValueError('reuse/plan reconstruction differs')
        parent_paths(self.parent,self.cfg)
        return self

    def initial(self):
        for n in (*self.m['sources'],self.cfg['protocol'],self.cfg['design_review'],self.cfg['initial_registration']):committed(n)
        if read_json(self.cfg['initial_registration'])['manifest_sha256']!=stream_hash(self.run/'manifest.json'):raise ValueError('child registration differs')


def stage_sources(s,phase):
    phase='final' if phase=='custom' else phase;directory=s.run/phase;record=read_json(directory/'run.json')
    if record['completed'] is not True or record['gpu_uuid']!=s.runtime['gpu_uuid']:raise ValueError('child incomplete/wrong-device stage')
    pv.check_hashes([{str(directory/n):h for n,h in record['files_sha256'].items()}])
    return {str(directory/'run.json'):stream_hash(directory/'run.json'),**{str(directory/n):h for n,h in record['files_sha256'].items()}}


def rows(s,phase):return read_json(s.run/'inputs/split.json')['selection' if phase=='grid' else 'report']


def child_values(s,phase):
    stage_sources(s,phase);phase='final' if phase=='custom' else phase;value=read_json(s.run/phase/'predictions.json')
    arms=p.missing(s.cfg['trait']) if phase=='grid' else ('baseline','selected','random','prompt')
    if set(value)!=set(arms):raise ValueError('child arm set differs; reused conditions cannot regenerate')
    rendered=read_json(s.run/'inputs/rendered.json')
    for arm,out in value.items():ref.validate_generation(out,rows(s,phase),rendered['prompt_control' if arm=='prompt' else 'user_only'],s.m['tokenizer_ids']['eos'])
    return value


def reused(s):
    result={};sources={};grades={}
    for name,entry in p.reuse_map(s.cfg['trait']).items():
        phase=entry['phase']
        if phase not in sources:sources[phase]=pv.values(s.parent,phase);grades[phase]=pv.read_reviews(s.parent,phase)
        out=sources[phase][entry['condition']];result[name]=(out,{r['id']:grades[phase][r['id']][entry['condition']] for r in out['predictions']})
    return result


def packet_batches(s,phase):
    if phase in ('final','custom'):selected(s,True)
    outputs=child_values(s,phase);questions={r['id']:r['user'] for r in rows(s,phase)};answers={(name,r['id']):r['output'] for name,value in outputs.items() for r in value['predictions']}
    return [[{'id':e['id'],'question':questions[e['question_id']],'answer':answers[e['condition'],e['question_id']]} for e in batch] for batch in read_json(s.run/'inputs/plans.json')[phase]]


def packet_record(s,phase):
    return {'manifest_sha256':stream_hash(s.run/'manifest.json'),'packet_sha256':stream_hash(s.run/'judging'/phase/'packet.json'),'plans_sha256':stream_hash(s.run/'inputs/plans.json'),
        'sources':stage_sources(s,phase),'rubric_sha256':stream_hash(s.run/'inputs'/('custom_rubric.json' if phase=='custom' else 'rubric.json'))}


def packet(args,s):
    if (s.run/'stopped.json').exists():raise RuntimeError('child closed')
    values=packet_batches(s,args.phase);out=s.run/'judging'/args.phase;out.mkdir(mode=0o700,parents=True,exist_ok=False)
    save(out/'packet.json',values);save(out/'manifest.json',packet_record(s,args.phase))


def bodies(s,phase):
    out=s.run/'judging'/phase
    if read_json(out/'manifest.json')!=packet_record(s,phase) or read_json(out/'packet.json')!=packet_batches(s,phase):raise ValueError('child anonymous packet/source drift')
    rubric=read_json(s.run/'inputs'/('custom_rubric.json' if phase=='custom' else 'rubric.json'))
    return [{'rubric':rubric,'items':batch,'expected_ids':[r['id'] for r in batch],'expected_count':len(batch),'return_schema':rubric['output_schema']} for batch in read_json(out/'packet.json')]


def grade_binding(s,phase,body):
    return {'manifest_sha256':stream_hash(s.run/'manifest.json'),'sources':s.m['sources'],'phase':phase,'packet_manifest_sha256':stream_hash(s.run/'judging'/phase/'manifest.json'),
        'payload_sha256':[f.canonical_hash(b) for b in body],'binary_sha256':pv.BINARY_SHA,'cli_version':s.m['cli_version'],'audit_sha256':s.m['audit_sha256'],'argv':f.argv('claude'),
        'maximum_total_calls':s.cfg['maximum_new_starts'],'maximum_attempts_per_batch':2}


def audit_journals(s):
    root=s.run/'grades'
    if len(list(root.glob('*/batch-*-attempt-*-start.json')))>s.cfg['maximum_new_starts']:raise ValueError('child native call limit exceeded')
    for directory in root.iterdir() if root.exists() else []:
        if not directory.is_dir():continue
        if directory.name not in p.PHASES:raise ValueError('unexpected child grading phase')
        body=bodies(s,directory.name)
        for file in directory.glob('*attempt*'):
            match=re.fullmatch(r'batch-(\d{3})-attempt-([01])(?:-start|-receipt|-interruption)?\.json',file.name)
            if not match or int(match[1])>=len(body):raise ValueError('stray child attempt artifact')
        for bi,b in enumerate(body):
            valid=False
            for ai in (0,1):
                stem=f'batch-{bi:03d}-attempt-{ai}';file=directory/(stem+'.json')
                if valid and list(directory.glob(stem+'*')):raise ValueError('extra attempt after first valid')
                if not file.exists():continue
                attempt=read_json(file)
                if attempt.get('canonical_stdin_sha256')!=f.canonical_hash(b):raise ValueError('attempt payload differs')
                try:outer,_=pv.checked_outer(attempt)
                except (RuntimeError,f.ModelMismatch):
                    if not (s.run/'stopped.json').exists():raise
                    continue
                try:pv.parse(outer,b['expected_ids'],directory.name=='custom')
                except f.InvalidSubmission:continue
                valid=True


def judge(args,s):
    s.initial();body=bodies(s,args.phase);binary=shutil.which('claude')
    if not binary or Path(binary).resolve()!=path(s.m['binary']).resolve() or stream_hash(Path(binary))!=pv.BINARY_SHA or f.version('claude')!=s.m['cli_version']:raise ValueError('pinned native binary/version required')
    f.validate_audit(read_json(s.runtime['judge_runtime_config'])['judge_audit']);root=s.run/'grades';directory=root/args.phase
    if args.resume and not directory.exists():raise ValueError('resume requires existing child phase')
    if directory.exists() and not args.resume:raise FileExistsError('explicit exact child journal resume required')
    root.mkdir(mode=0o700,exist_ok=True);directory.mkdir(mode=0o700,exist_ok=True)
    with f.exclusive_run(root):
        f.save_or_match(directory/'run_binding.json',grade_binding(s,args.phase,body));audit_journals(s)
        accepted=p.finish_judgments(s.run,directory,body,stream_hash(directory/'run_binding.json'),s.cfg['maximum_new_starts'],args.phase=='custom');s.pending_accept=(directory/'accepted.json',accepted)


def read_reviews(s,phase):
    body=bodies(s,phase);directory=s.run/'grades'/phase
    for file in directory.glob('*attempt*'):
        match=re.fullmatch(r'batch-(\d{3})-attempt-([01])(?:-start|-receipt|-interruption)?\.json',file.name)
        if not match or int(match[1])>=len(body):raise ValueError('stray child saved attempt')
    ratings=p.read_judgments(directory,body,grade_binding(s,phase,body),phase=='custom')
    aliases={e['id']:e for batch in read_json(s.run/'inputs/plans.json')[phase] for e in batch};decoded={}
    for r in ratings:
        e=aliases[r['id']];decoded.setdefault(e['question_id'],{})[e['condition']]=r
    return decoded


def metrics(s,phase):
    outputs=child_values(s,phase);grades=read_reviews(s,phase)
    result={name:p.measures(value['predictions'],[{**grades[r['id']][name],'id':r['id']} for r in value['predictions']]) for name,value in outputs.items()}
    if phase=='grid':
        for name,(out,labels) in reused(s).items():result[name]=p.measures(out['predictions'],[{**labels[r['id']],'id':r['id']} for r in out['predictions']])
    return result


def selection_result(s):
    observed=metrics(s,'grid');return {**p.choose(observed),'metrics':observed,'bindings':{'manifest_sha256':stream_hash(s.run/'manifest.json'),
        'parent_manifest_sha256':s.m['parent_manifest_sha256'],'parent_strength_sha256':s.m['parent_strength_sha256'],'reuse_sha256':stream_hash(s.run/'inputs/reuse.json'),
        'grid_run_sha256':stream_hash(s.run/'grid/run.json'),'grid_accepted_sha256':stream_hash(s.run/'grades/grid/accepted.json')}}


def select(args,s):
    result=selection_result(s);save(s.run/'selection.json',result)
    if result['selection_null']:pv.stop(s.run,'selection_null')


def selected(s,locked=False):
    record=read_json(s.run/'selection.json')
    if record!=selection_result(s) or record['selection_null']:raise ValueError('child selection null/drift')
    if locked:
        committed(s.cfg['selection_registration'])
        if read_json(s.cfg['selection_registration'])['selection_sha256']!=stream_hash(s.run/'selection.json'):raise ValueError('committed child selection differs')
    return record['setting']


def cost_state(s):return {k:sum(read_json(file).get(k,0) for file in s.run.glob('*/run.json')) for k in ('saved_generations','teacher_forced_sequences','mc_items')}


def charge(s,progress,existing,generations=0,mc=0):
    if progress['saved_generations']+existing['saved_generations']+generations>s.cfg['maximum_new_generations'] or progress['mc_items']+existing['mc_items']+mc>64:raise ValueError('child generation/MC limit BEFORE work')


@contextmanager
def gpu_stage(s,phase):
    s.initial()
    try:resource=resource_check(s.runtime)
    except RuntimeError as exc:raise pv.ResourceWait('resources busy; child phase not started') from exc
    out=s.run/phase;out.mkdir(mode=0o700,exist_ok=False)
    import torch
    began=time.perf_counter();completed=False;progress={'saved_generations':0,'teacher_forced_sequences':0,'mc_items':0,'generation_wall_seconds':0.};existing=cost_state(s)
    try:
        model,tok=load_model(s.runtime,s.m,'bf16')
        if model.config.num_hidden_layers!=36 or any(v.dtype!=torch.bfloat16 or v.device!=torch.device('cuda:0') for v in model.parameters()):raise ValueError('same native BF16 single-GPU representation required')
        torch.cuda.reset_peak_memory_stats();yield model,tok,out,progress,existing;torch.cuda.synchronize();completed=True
    finally:
        save(out/'run.json',{'completed':completed,**resource,**progress,'elapsed_wall_seconds':time.perf_counter()-began,
            'count_scope':'Saved/completed child work only; interrupted unsaved work may be additional','peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),
            'memory_scope':'PyTorch allocator, not total VRAM','dtype':'bfloat16','attention':'sdpa','seed':42,'torch_cuda_runtime':torch.version.cuda,'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),
            'tf32_matmul':torch.backends.cuda.matmul.allow_tf32,'tf32_cudnn':torch.backends.cudnn.allow_tf32,'files_sha256':{str(file.relative_to(out)):stream_hash(file) for file in out.rglob('*') if file.is_file()}})


def generated(s,model,tok,items,progress,existing,vector=None,block=None):
    charge(s,progress,existing,generations=len(items));adapter=SimpleNamespace(cfg=s.runtime)
    return pv.generated(adapter,model,tok,items,progress,existing,vector,block)


def grid(args,s):
    tensors=pv.strength(s.parent)
    with gpu_stage(s,'grid') as (model,tok,out,progress,existing):
        results={}
        for name in p.missing(s.cfg['trait']):
            setting=p.SETTINGS.get(name);vector=tensors[f"{setting['block']}.raw"]*setting['coefficient'] if setting else None
            results[name]=generated(s,model,tok,rows(s,'grid'),progress,existing,vector,setting['block'] if setting else None)
        save(out/'predictions.json',results);save(out/'reuse_binding.json',{'manifest_sha256':s.m['parent_manifest_sha256'],'reuse_sha256':stream_hash(s.run/'inputs/reuse.json'),'reused_outputs':s.m['reused_grid_outputs'],'new_outputs':s.m['new_grid_outputs']})


def final(args,s):
    setting=selected(s,True);tensors=pv.strength(s.parent);block=setting['block'];coefficient=setting['coefficient'];items=rows(s,'final')
    with gpu_stage(s,'final') as (model,tok,out,progress,existing):
        import torch
        raw=tensors[f'{block}.raw'];shift=raw*coefficient;random=tensors[f'{block}.random_unit']*raw.norm()*coefficient
        instruction=read_json(s.run/'inputs/eval.json')['instruction'][1]['pos'];prompted=[{**r,'system':ref.system(s.runtime,instruction,'pos')} for r in items]
        results={'baseline':generated(s,model,tok,items,progress,existing),'selected':generated(s,model,tok,items,progress,existing,shift,block),
            'random':generated(s,model,tok,items,progress,existing,random,block),'prompt':generated(s,model,tok,prompted,progress,existing)}
        save(out/'predictions.json',results);norm=read_json(s.parent.run/'extraction/vectors.json')['layers'][str(block)]['mean_per_token_residual_norm']
        save(out/'diagnostics.json',{'selection_sha256':stream_hash(s.run/'selection.json'),'parent_vector_sha256':stream_hash(s.parent.run/'extraction/vectors.safetensors'),
            'fresh_saved_tensor_reload':True,'selected_fp32_norm':shift.norm().item(),'random_fp32_norm':random.norm().item(),
            'actual_selected_bf16_norm':shift.to(torch.bfloat16).float().norm().item(),'actual_random_bf16_norm':random.to(torch.bfloat16).float().norm().item(),'selected_shift_over_per_token_norm':shift.norm().item()/norm})


def reference_result(s):
    observed=metrics(s,'final');return {'metrics':observed,**ref.reference_gates(observed)}


def custom_result(s):
    grades=read_reviews(s,'custom');outputs=child_values(s,'custom');result={}
    for name,value in outputs.items():
        ratings=[grades[r['id']][name] for r in value['predictions']];result[name]={'n':len(ratings),'target_present':sum(r[s.cfg['trait']]>=2 for r in ratings),
            'clear_presence':{t:sum(r[t]>=2 for r in ratings) for t in ref.TRAITS},'ambiguous':{t:sum(r[t]==1 for r in ratings) for t in ref.TRAITS},
            'content_ok':sum(r['content_ok'] for r in ratings),'incoherent':sum(not r['coherent'] for r in ratings),'gross_failures':sum(r['gross_failure'] for r in ratings)}
    a,b=result['baseline'],result['selected'];gates={'target':b['target_present']>=5,'gain':b['target_present']>=a['target_present']+3,
        'content':b['content_ok']>=a['content_ok']-1,'gross':b['gross_failures']<=a['gross_failures']+1,'coherence':b['incoherent']<=a['incoherent']+1}
    return {'metrics':result,'gates':gates,'custom_criteria_pass':all(gates.values()),'not_reference_selection':True}


def retention(args,s):
    setting=selected(s,True)
    if not reference_result(s)['all_criteria_pass']:raise ValueError('reference report failed; skip capability')
    tensors=pv.strength(s.parent);block=setting['block'];shift=tensors[f'{block}.raw']*setting['coefficient']
    with gpu_stage(s,'retention') as (model,tok,out,progress,existing):
        from langdetect import DetectorFactory
        old=DetectorFactory.seed;results={}
        try:
            for name in ('baseline','selected'):
                charge(s,progress,existing,generations=16,mc=32);directory=out/name;directory.mkdir();audit={};DetectorFactory.seed=0;summary={};timing=0.
                with (nullcontext() if name=='baseline' else ref.response_hook(model,block,shift,audit)):
                    fixture=read_json(s.run/'inputs/retention.json')
                    for task in ('general_mc','physics_mc'):
                        records=hf_multiple_choice(model,tok,fixture[task],batch_size=4);save(directory/(task+'_predictions.json'),records)
                        summary[task]={'total':len(records),'correct':sum(r['correct'] for r in records)};progress['mc_items']+=len(records)
                    for task,cap in [('gsm8k',512),('ifeval',1024)]:
                        items=fixture[task] if task=='gsm8k' else [{'id':str(r['key']),'user':r['prompt']} for r in fixture[task]]
                        value=generate_batched(model,tok,items,batch_size=4,max_new_tokens=cap,seed=42,phase='PV grid capability/'+task)
                        save(directory/(task+'_predictions.json'),value['predictions']);save(directory/(task+'_timing.json'),value['timing'])
                        score=score_gsm8k(fixture[task],value['predictions']) if task=='gsm8k' else ifeval_scores(fixture[task],value['predictions'])
                        save(directory/(task+'_scores.json'),score);summary[task]={k:v for k,v in score.items() if k not in ('details','records')}
                        progress['saved_generations']+=len(items);timing+=value['timing']['generation_wall_seconds']
                if name=='selected':save(directory/'hook_audit.json',audit)
                results[name]=summary;progress['generation_wall_seconds']+=timing
        finally:DetectorFactory.seed=old
        save(out/'metrics.json',results);save(out/'selection_binding.json',{'selection_sha256':stream_hash(s.run/'selection.json'),'language_detector_seed':0})


def report(args,s):
    audit_journals(s);parent=read_json(s.run/'inputs/parent_costs.json');new=cost_state(s);costs={file.parent.name:read_json(file) for file in s.run.glob('*/run.json')};stopped=read_json(s.run/'stopped.json') if (s.run/'stopped.json').exists() else None
    clean=lambda values:{name:{k:v for k,v in value.items() if k not in ('files_sha256','cuda_visible_devices')} for name,value in values.items()}
    result={'status':'incomplete','scope':f"Separate symmetric PV grid extension for {s.cfg['trait']}; original parent result immutable",
        'manifest_sha256':stream_hash(s.run/'manifest.json'),'source_sha256':s.m['sources'],'parent_manifest_sha256':s.m['parent_manifest_sha256'],
        'model':{'model_id':s.runtime['model_id'],'revision':s.runtime['revision']},'reused_grid_outputs':s.m['reused_grid_outputs'],'new_grid_outputs':s.m['new_grid_outputs'],
        'parent':{'costs':clean(parent['costs']),'totals':parent['totals'],'native_starts':parent['native_starts'],'selection_role':'historical closed null' if s.cfg['trait']=='rudeness' else 'prospectively superseded before layer search'},
        'child':{'costs':clean(costs),'totals':new,'native_starts':len(list((s.run/'grades').glob('*/batch-*-attempt-*-start.json')))},
        'combined_totals':{k:new[k]+parent['totals'][k] for k in new},'stop':stopped,
        'limits':['Separate grid extension fixes sequential-search coverage; historical parent null remains unchanged.',
            'Reused parent labels are not rerated; cross-session ordinal comparisons and dependence are limitations.',
            'One greedy512 rollout, raw-native FP32 pooling, co-rated trait/coherence and integer Claude scores differ from author sampling/probability-weighted scores.',
            'Reference trait and calibrated contextual0..3 target remain separate; no human-validity or independent-trait-control claim.',
            'Raw shift ratios are diagnostic; no extra TF/preflight, broader search, joint control or model-size change.']}
    if s.cfg['trait']=='rudeness':
        result['parent']['original_result_sha256']=stream_hash(path(s.cfg['parent_result']))
        result['limits']+=['All10 parent baseline answers were capped: cap-count quality criterion is uninformative.',
            'Parent block7 coefficient3/4 outcomes remain in the original parent result, descriptive only and excluded from this fixed grid.']
    else:result['parent']['status']='amended_before_steering'
    if (s.run/'selection.json').exists():
        choice=selection_result(s)
        if choice!=read_json(s.run/'selection.json'):raise ValueError('child selection does not recompute')
        result['selection']={k:v for k,v in choice.items() if k!='bindings'}
    if (s.run/'grades/final/accepted.json').exists():
        result['reference']=reference_result(s)
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
    if result['status']=='incomplete' and not stopped:raise ValueError('exclusive report waits for closure')
    ref.private_export(result);s.verify();save(path(s.cfg['result']),result)


def lexical(args,s):
    if s.cfg['trait']!='profanity':raise ValueError('fixed profanity diagnostic only')
    report=read_json(s.cfg['result'])
    if report['status']=='incomplete' and not report.get('stop'):raise ValueError('wait for child closure')
    artifact,origin,actual=artifact_origin();phases={};sources={}
    parent_rows=read_json(s.parent.run/'inputs/elicit_rows.json');parent_values=pv.values(s.parent,'elicit')['elicited']['predictions'];lookup={r['id']:r for r in parent_values};grades=pv.read_reviews(s.parent,'elicit')
    phases['parent_elicit']={f'i{i}-{side}':actor.lexical_counts([lookup[r['id']] for r in parent_rows if r['instruction_index']==i and r['side']==side],actor.LEXICON,
        {r['id']:grades[r['question_id']][f'i{i}-{side}'] for r in parent_rows if r['instruction_index']==i and r['side']==side}) for i in range(5) for side in ('pos','neg')}
    phases['parent_preflight']={n:actor.lexical_counts(read_json(s.parent.run/'preflight'/(n+'.json'))['predictions'],actor.LEXICON) for n in ('ordinary','zero','nonzero')}
    for phase in p.PHASES:
        generation='final' if phase=='custom' else phase
        if not (s.run/generation/'run.json').exists():continue
        if phase=='custom' and not (s.run/'grades/custom/accepted.json').exists():continue
        values=child_values(s,phase);labels=read_reviews(s,phase) if (s.run/'grades'/phase/'accepted.json').exists() else None;sources.update(stage_sources(s,phase))
        phases[phase]={name:actor.lexical_counts(value['predictions'],actor.LEXICON,{r['id']:labels[r['id']][name] for r in value['predictions']} if labels is not None else None,phase=='custom') for name,value in values.items()}
    if (s.run/'retention/run.json').exists():
        sources.update(stage_sources(s,'retention'));phases['retention']={condition:{task:actor.lexical_counts(read_json(s.run/'retention'/condition/(task+'_predictions.json')),actor.LEXICON) for task in ('gsm8k','ifeval')} for condition in ('baseline','selected')}
    result={'scope':'Canonical report-only frozen profanity lexical supplement across parent plus child; do not also run actor parent lexical report','phases':phases,'frozen_actor_spec':actor.lexical_spec(),
        'manifest_sha256':stream_hash(s.run/'manifest.json'),'child_report_sha256':stream_hash(path(s.cfg['result'])),'origin_sha256':stream_hash(path(artifact.cfg['origin_record'])),
        'source_sha256':stream_hash(Path(__file__)),'phase_source_sha256':{str(path(n).relative_to(path('.'))):h for n,h in sources.items()},'no_new_judgments_or_gate_changes':True}
    ref.private_export(result);s.verify();pv.check_hashes([actual,sources]);save(path(s.cfg['lexical_result']),result)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=('prepare','verify','grid','packet','judge','select','final','retention','report','lexical'))
    parser.add_argument('--trait',choices=('rudeness','profanity'),default='rudeness');parser.add_argument('--phase',choices=p.PHASES,default='grid');parser.add_argument('--execute-gpu',action='store_true');parser.add_argument('--execute-judgments',action='store_true');parser.add_argument('--resume',action='store_true');args=parser.parse_args();run=path(f'outputs/persona-pv-grid-{args.trait}-01')
    if args.stage=='prepare':prepare(args,run);return
    gpu=args.stage in ('grid','final','retention');live=gpu or args.stage=='judge'
    if gpu and not args.execute_gpu:raise ValueError('separate root release + --execute-gpu required')
    if args.stage=='judge' and not args.execute_judgments:raise ValueError('separate root release + --execute-judgments required')
    guard(run,args.trait)
    if live and (run/'stopped.json').exists():raise RuntimeError('child permanently stopped')
    s=None;completed=False;waiting=False
    try:
        s=Context(run,args.trait).verify(models=live or args.stage=='verify')
        if args.stage=='verify':print('Child grid bindings verified; no calls.');return
        globals()[args.stage](args,s);completed=True
    except pv.ResourceWait:waiting=True;raise
    except BaseException as exc:
        if live and run.exists():pv.stop(run,'child_live_stage_fatal',exc)
        raise
    finally:
        if live and s is not None and not waiting:
            try:
                s.verify(models=True)
                if completed and hasattr(s,'pending_accept'):f.save_or_match(*s.pending_accept)
            except BaseException as exc:pv.stop(run,'child_post_call_integrity_failure',exc);raise


if __name__=='__main__':main()
