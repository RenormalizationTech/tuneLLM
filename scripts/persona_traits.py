"""Separate registered 4B rudeness/sarcasm studies; one shared bounded runner."""
import argparse
from contextlib import contextmanager, nullcontext
import importlib.metadata
import json
import os
from pathlib import Path
import re
import shutil
import time

from scripts import persona_traits_common as p
from scripts.profanity_steering import generated, count_generated, save_vectors
from scripts import profanity_steering as prior
from scripts import persona_rating_calibration as c
from scripts.batched_generation import generate_batched, render_prompt
from scripts.benchmark_screen import run_screen, choice_token_ids
from scripts.benign_helpfulness import resource_check
from scripts.common import path, read_json
from scripts.save_nf4 import stream_hash
from scripts.tone_common import response_encoding, steer
from scripts.tone_steering import load_model, committed

RUNS={t:f'outputs/{t}-steering-01' for t in ('rudeness','sarcasm')}
SOURCES=tuple(dict.fromkeys(('scripts/persona_traits.py','scripts/persona_traits_common.py','scripts/persona_traits_runbook.md',
    'configs/rudeness_steering.json','configs/sarcasm_steering.json',
    'evaluation/rudeness_steering/extraction.json','evaluation/sarcasm_steering/extraction.json',
    'evaluation/persona_traits/development.json','evaluation/persona_traits/confirmation.json','evaluation/persona_traits/rubric.json',*prior.SOURCES)))
f,save=c.f,c.save
BINARY_SHA=c.BINARY_SHA


class ResourceWait(RuntimeError):pass


def guard(run):
    if run.resolve() not in {path(v).resolve() for v in RUNS.values()}:raise ValueError('only new trait directories may be written')


def validate_config(cfg):
    trait=cfg.get('target_trait')
    if trait not in RUNS:raise ValueError('rudeness/sarcasm profile required')
    expected=read_json(f'configs/{trait}_steering.json')
    if cfg!=expected or cfg['run']!=RUNS[trait] or cfg['extraction_fixture']!=f'evaluation/{trait}_steering/extraction.json':raise ValueError('profile paths/config differ')
    fixed={'model_id':'Qwen/Qwen3-4B','revision':'1cfa9a7208912126459214e8b04321603b3df60c',
        'gpu':'GPU-34646ab9-ec19-ca63-0dfb-bdbf2e1a977b','gpu_uuid':'GPU-34646ab9-ec19-ca63-0dfb-bdbf2e1a977b',
        'minimum_free_mib':12000,'maximum_utilization_percent':10,'dtype':'bfloat16','attn_implementation':'sdpa','enable_thinking':False,
        'system':'You are a helpful assistant.','seed':42,'batch_size':4,'max_new_tokens':256,'layers':[17,23],'betas':[.05,.1,.2],
        'maximum_judge_calls':80,'maximum_target_generations':230,'maximum_mc_items':64,'maximum_teacher_forced_sequences':64,
        'judging_seeds':{'eligibility':501,'development':502,'confirmation':503,'confirmation_repeat':504},
        'hf_home':'.hf','shared_directory':'evaluation/persona_traits','protocol':'docs/isolated_persona_traits_protocol.md',
        'design_review':'results/reviews/isolated_persona_traits_design_review.md','fixture_review':'results/reviews/isolated_persona_traits_fixture_review.md',
        'previous_run':'outputs/profanity-steering-01','previous_result':'results/profanity_steering_01.json',
        'retention_fixture':'outputs/refusal-03/inputs/retention.json'}
    if any(type(cfg.get(k)) is not type(v) or cfg.get(k)!=v for k,v in fixed.items()) or cfg['lexicon']!=read_json('configs/persona_rating_calibration.json')['lexicon']:raise ValueError('fixed grid/budgets/lexicon differ')


def generation_phase(phase):return 'confirmation' if phase=='confirmation_repeat' else phase


def fixtures(run):return {s:read_json(run/'inputs'/(s+'.json')) for s in ('extraction','development','confirmation')}


def prepare(args,run):
    guard(run)
    if run.exists():raise FileExistsError('fresh pilot required')
    cfg=read_json(args.config);validate_config(cfg)
    if cfg['target_trait']!=args.trait or run.resolve()!=path(cfg['run']).resolve():raise ValueError('profile/run mismatch')
    data={'extraction':read_json(cfg['extraction_fixture']),**{n:read_json(path(cfg['shared_directory'])/(n+'.json')) for n in ('development','confirmation')}}
    if data['extraction']['target_trait']!=cfg['target_trait']:raise ValueError('extraction/profile mismatch')
    p.validate_fixtures(data,cfg['lexicon']);rubric=read_json(path(cfg['shared_directory'])/'rubric.json')
    for name in ('development','confirmation','rubric'):
        if path(cfg['shared_directory']).joinpath(name+'.json').read_bytes()!=path('evaluation/profanity_steering').joinpath(name+'.json').read_bytes():raise ValueError('shared task/rubric bytes must remain exact')
    corpora=[read_json(f'evaluation/{trait}_steering/extraction.json') for trait in RUNS]
    if {r['topic'] for r in corpora[0]['items']}&{r['topic'] for r in corpora[1]['items']}:raise ValueError('trait extraction topics must be disjoint')
    old_topics={r['topic'] for r in read_json('evaluation/profanity_steering/extraction.json')['items']}
    if any(old_topics&{r['topic'] for r in corpus['items']} for corpus in corpora):raise ValueError('old extraction topic overlap')
    original=read_json('evaluation/persona_calibration/rubric.json')
    if any(rubric[k]!=original[k] for k in ('scale',*p.TRAITS,'gross_failure')):raise ValueError('calibrated definitions must remain exact')
    base=path(cfg['previous_run']);old=read_json(base/'manifest.json');result=read_json(cfg['previous_result'])
    if result['status'] not in ('eligibility_null','development_null','complete'):raise ValueError('profanity study must be closed')
    protected=dict(old['protected_sha256']);protected.update(old['sources'])
    for file in base.rglob('*'):
        if file.is_file():protected[str(file)]=stream_hash(file)
    for n in (cfg['protocol'],cfg['design_review'],cfg['fixture_review'],cfg['previous_result'],cfg['retention_fixture'],
              'results/diagnostics/profanity_development_divergence_01.json'):protected[str(path(n))]=stream_hash(path(n))
    if stream_hash(path(cfg['retention_fixture']))!='5acf1096011bc8c4476d7b89c9ff27fb7fe33a82ec074d8f77f84d89ca361b4e':raise ValueError('fixed capability fixture differs')
    model=read_json('outputs/xstest-baseline-01/manifest.json');c.check_hashes(model['model_sha256']);c.check_hashes(protected)
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(model['snapshot'],local_files_only=True,trust_remote_code=False)
    ratios={r['id']:len(tok(r['target'],add_special_tokens=False)['input_ids'])/len(tok(r['neutral'],add_special_tokens=False)['input_ids']) for r in data['extraction']['items']}
    if any(not .85<=v<=1.15 for v in ratios.values()):raise ValueError('authored token-length window fails')
    encoded={}
    for r in data['extraction']['items']:
        encoded[r['id']]={side:response_encoding(tok,r,r[side],cfg['system']) for side in ('neutral','target')}
    quality=read_json(cfg['retention_fixture'])
    for task in ('general_mc','physics_mc'):choice_token_ids(tok,quality[task])
    frozen={s:[{'id':r['id'],'rendered_prompt':render_prompt(tok,{**r,'system':cfg['system']}),
        'input_ids':tok(render_prompt(tok,{**r,'system':cfg['system']}),add_special_tokens=False)['input_ids']} for r in data[s]['items']] for s in ('development','confirmation')}
    # Exact-text disjointness is mechanical; semantic family/topic audit is advisory.
    calibration=read_json('evaluation/persona_calibration/texts.json')['items'];blocked={' '.join(r['text'].casefold().split()) for r in calibration}
    if any(' '.join(r['user'].casefold().split()) in blocked for d in data.values() for r in d['items']):raise ValueError('calibration overlap')
    run.mkdir(mode=0o700);save(run/'config.json',cfg)
    for s,value in {**data,'rubric':rubric,'plans':p.plans(data),'ratios':ratios,'encoded':encoded,'rendered':frozen,'retention':quality}.items():save(run/'inputs'/(s+'.json'),value)
    for name in (*SOURCES,cfg['protocol'],cfg['design_review'],cfg['fixture_review']):
        target=run/'source'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path(name),target)
    save(run/'manifest.json',{'files':{str(file.relative_to(run)):stream_hash(file) for file in run.rglob('*') if file.is_file()},
        'sources':{n:stream_hash(path(n)) for n in SOURCES},'protected_sha256':protected,'prior_protected_count':len(old['protected_sha256']),
        'model_sha256':model['model_sha256'],'snapshot':model['snapshot'],'binary':old['binary'],'cli_version':old['cli_version'],'audit_sha256':old['audit_sha256'],
        'packages':{n:importlib.metadata.version(n) for n in ('torch','transformers','tokenizers','safetensors','langdetect','nltk')},'target_trait':cfg['target_trait'],'rendering_scope':'Model input is helpful system plus row.user only; row.context is judge metadata and never enters the model prompt.','no_model_or_grader_calls':True})
    c.check_hashes(protected)


def verify(run,models=False):
    guard(run);cfg,m=read_json(run/'config.json'),read_json(run/'manifest.json');validate_config(cfg)
    if run.resolve()!=path(cfg['run']).resolve() or m['target_trait']!=cfg['target_trait']:raise ValueError('profile/manifest/run differs')
    c.check_hashes({str(run/n):h for n,h in m['files'].items()});c.check_hashes(m['sources']);c.check_hashes(m['protected_sha256'])
    if models:c.check_hashes(m['model_sha256'])
    if any(importlib.metadata.version(n)!=v for n,v in m['packages'].items()):raise ValueError('packages differ')
    p.validate_fixtures(fixtures(run),cfg['lexicon'])
    if fixtures(run)['extraction']['target_trait']!=cfg['target_trait']:raise ValueError('snapshot extraction/profile mismatch')
    if read_json(run/'inputs/plans.json')!=p.plans(fixtures(run)):raise ValueError('constrained plans changed')
    return cfg,m


def stop(run,cause,exc=None):
    if not (run/'stopped.json').exists():save(run/'stopped.json',{'stop_cause':cause,'error_type':type(exc).__name__ if exc else None})


def registered(args,run,models=False):
    cfg,m=verify(run,models)
    if args.trait!=cfg['target_trait']:raise ValueError('requested/profile trait mismatch')
    if args.trait=='sarcasm':
        previous=read_json('results/rudeness_steering_01.json')
        if previous['status'] not in ('eligibility_null','development_null','complete'):raise ValueError('rudeness must close before sarcasm live stages')
    for n in (*SOURCES,cfg['protocol'],cfg['design_review'],cfg['fixture_review'],args.registration):committed(n)
    if read_json(args.registration)['manifest_sha256']!=stream_hash(run/'manifest.json'):raise ValueError('initial registration differs')
    return cfg,m


def source_files(run,phase):
    if phase=='eligibility':return {str(run/'inputs/extraction.json'):stream_hash(run/'inputs/extraction.json')}
    directory=run/generation_phase(phase);record=read_json(directory/'run.json')
    if record['completed'] is not True:raise ValueError('generation phase incomplete')
    c.check_hashes({str(directory/n):h for n,h in record['files_sha256'].items()})
    return {str(directory/'run.json'):stream_hash(directory/'run.json'),**{str(directory/n):h for n,h in record['files_sha256'].items()}}


def outputs(run,phase):
    source_files(run,phase)
    if phase=='eligibility':return {side:[{'id':r['id'],'output':r[side]} for r in read_json(run/'inputs/extraction.json')['items']] for side in ('neutral','target')}
    return {condition:value['predictions'] for condition,value in read_json(run/generation_phase(phase)/'predictions.json').items()}


def packet_batches(run,phase):
    data=fixtures(run);rows=data['extraction' if phase=='eligibility' else generation_phase(phase)]['items'];lookup={r['id']:r for r in rows};conditions=outputs(run,phase)
    values={(condition,r['id']):r['output'] for condition,records in conditions.items() for r in records}
    plan=read_json(run/'inputs/plans.json')[phase];batches=[]
    for batch in plan:
        batches.append([{'id':entry['id'],**{k:lookup[entry['group_id']][k] for k in ('context','user','requirements','reference','factual')},
            'text':values[entry['condition'],entry['group_id']]} for entry in batch])
    return batches


def packet(args,run):
    verify(run);phase=args.phase
    if (run/'stopped.json').exists():raise RuntimeError('stopped run')
    batches=packet_batches(run,phase);directory=run/'judging'/phase;directory.mkdir(mode=0o700,parents=True,exist_ok=False)
    save(directory/'packet.json',batches)
    save(directory/'manifest.json',{'packet_sha256':stream_hash(directory/'packet.json'),'plans_sha256':stream_hash(run/'inputs/plans.json'),
        'sources':source_files(run,phase),'rubric_sha256':stream_hash(run/'inputs/rubric.json')})


def bodies(run,phase):
    record=read_json(run/'judging'/phase/'manifest.json');batches=read_json(run/'judging'/phase/'packet.json')
    if record!={'packet_sha256':stream_hash(run/'judging'/phase/'packet.json'),'plans_sha256':stream_hash(run/'inputs/plans.json'),
        'sources':source_files(run,phase),'rubric_sha256':stream_hash(run/'inputs/rubric.json')}:raise ValueError('packet or source prediction hashes changed')
    if batches!=packet_batches(run,phase):raise ValueError('packet no longer matches frozen anonymous plan and source texts')
    if phase in ('confirmation','confirmation_repeat'):
        by_alias=lambda batches:{row['id']:row for batch in batches for row in batch}
        if by_alias(packet_batches(run,'confirmation'))!=by_alias(packet_batches(run,'confirmation_repeat')):raise ValueError('repeat aliases or per-item payload changed')
    rubric=read_json(run/'inputs/rubric.json')
    return [{'rubric':rubric,'items':batch,'expected_ids':[r['id'] for r in batch],'expected_count':8,'return_schema':rubric['output_schema']} for batch in batches]


def parse(outer,ids):
    try:
        value=f.result_json(outer.get('result'))
        if not isinstance(value,dict) or set(value)!={'ratings'}:raise ValueError('ratings-only JSON required')
        return p.validate_ratings(ids,value['ratings'])
    except (ValueError,TypeError,KeyError) as exc:raise f.InvalidSubmission('malformed/incomplete pilot ratings') from exc


def grade_binding(run,phase,m,body):
    return {'manifest_sha256':stream_hash(run/'manifest.json'),'sources':m['sources'],'phase':phase,
        'packet_manifest_sha256':stream_hash(run/'judging'/phase/'manifest.json'),'payload_sha256':[f.canonical_hash(b) for b in body],
        'binary_sha256':c.BINARY_SHA,'cli_version':m['cli_version'],'audit_sha256':m['audit_sha256'],'argv':f.argv('claude'),
        'maximum_total_calls':80,'maximum_attempts_per_batch':2}


def judge(args,run):
    if not args.execute_judgments:raise ValueError('separate network-enabled judging release required')
    guard(run)
    if (run/'stopped.json').exists():raise RuntimeError('permanent stop')
    try:
        cfg,m=registered(args,run);phase=args.phase
        if phase=='confirmation_repeat':read_reviews(run,'confirmation')
        body=bodies(run,phase);binary=shutil.which('claude')
        if not binary or Path(binary).resolve()!=path(m['binary']).resolve() or stream_hash(Path(binary))!=BINARY_SHA or f.version('claude')!=m['cli_version']:raise ValueError('pinned native binary/version required')
        f.validate_audit(read_json('configs/qwen3_14b_baseline.json')['judge_audit'])
        root=run/'grades';directory=root/phase
        if args.resume and not directory.exists():raise ValueError('resume requires an existing phase journal')
        if directory.exists() and not args.resume:raise FileExistsError('exact journal resume required')
        root.mkdir(mode=0o700,exist_ok=True);directory.mkdir(mode=0o700,exist_ok=True)
        with f.exclusive_run(root):
            f.save_or_match(directory/'run_binding.json',grade_binding(run,phase,m,body));bound=stream_hash(directory/'run_binding.json');ratings=[];records=[]
            audit_journals(run)
            for bi,b in enumerate(body):
                record={'expected_ids':b['expected_ids'],'canonical_stdin_sha256':f.canonical_hash(b),'attempts':[]}
                f.save_or_match(directory/f'batch-{bi:03d}-input.json',record)
                for ai in range(2):
                    stem=f'batch-{bi:03d}-attempt-{ai}'
                    if not list(directory.glob(stem+'*')) and len(list(root.glob('*/batch-*-attempt-*-start.json')))>=80:raise RuntimeError('fixed80-call budget exhausted before start')
                    attempt,receipt=f.replay_or_call(directory,bi,ai,b,bound,'claude');outer,models=c.checked_outer(attempt)
                    entry={'receipt':receipt,'actual_model_usage_ids':models,'structural_valid':False};record['attempts'].append(entry)
                    try:values=parse(outer,b['expected_ids'])
                    except f.InvalidSubmission:
                        if ai==1:f.save_or_match(directory/f'batch-{bi:03d}-provenance.json',record);stop(run,'schema_exhausted');raise
                        continue
                    entry['structural_valid']=True;record['first_valid_attempt']=ai;ratings.extend(values);break
                f.save_or_match(directory/f'batch-{bi:03d}-provenance.json',record);records.append(record)
            verify(run);f.save_or_match(directory/'accepted.json',{'ratings':ratings,'batches':records,'binding_sha256':bound,'first_valid_only':True})
    except BaseException as exc:stop(run,'judging_fatal',exc);raise
    finally:
        try:verify(run)
        except BaseException as exc:stop(run,'post_call_hash_failure',exc);raise


def audit_journals(run):
    root=run/'grades'
    if len(list(root.glob('*/batch-*-attempt-*-start.json')))>80:raise ValueError('call budget exceeded')
    for directory in root.iterdir() if root.exists() else []:
        if not directory.is_dir():continue
        if directory.name not in p.PHASES:raise ValueError('unknown grading phase')
        for file in directory.glob('*attempt*'):
            match=re.fullmatch(r'batch-(\d{3})-attempt-([01])(?:-start|-receipt|-interruption)?\.json',file.name)
            if not match or int(match[1])>=p.PHASES[directory.name][1]:raise ValueError('unregistered attempt artifact')
        body=bodies(run,directory.name)
        for bi,b in enumerate(body):
            for ai in (0,1):
                file=directory/f'batch-{bi:03d}-attempt-{ai}.json'
                if not file.exists():continue
                attempt=read_json(file)
                if attempt.get('canonical_stdin_sha256')!=f.canonical_hash(b):raise ValueError('attempt payload changed')
                try:outer,_=c.checked_outer(attempt);parse(outer,b['expected_ids'])
                except f.InvalidSubmission:continue
                except (RuntimeError,f.ModelMismatch):
                    if not (run/'stopped.json').exists():raise
                    continue
                if ai==0 and list(directory.glob(f'batch-{bi:03d}-attempt-1*')):raise ValueError('attempt after first valid')


def read_reviews(run,phase):
    cfg,m=verify(run);audit_journals(run);body=bodies(run,phase);directory=run/'grades'/phase
    if read_json(directory/'run_binding.json')!=grade_binding(run,phase,m,body):raise ValueError('grading binding changed')
    bound=stream_hash(directory/'run_binding.json');ratings=[];records=[]
    for bi,b in enumerate(body):
        record=read_json(directory/f'batch-{bi:03d}-provenance.json');first=record['first_valid_attempt']
        if type(first) is not int or first not in (0,1) or len(record['attempts'])!=first+1:raise ValueError('first-valid/budget differs')
        base={'expected_ids':b['expected_ids'],'canonical_stdin_sha256':f.canonical_hash(b),'attempts':[]}
        if read_json(directory/f'batch-{bi:03d}-input.json')!=base or any(record.get(k)!=v for k,v in base.items() if k!='attempts'):raise ValueError('batch input differs')
        for ai,entry in enumerate(record['attempts']):
            stem=f'batch-{bi:03d}-attempt-{ai}';file=directory/(stem+'.json');start=directory/(stem+'-start.json')
            receipt={'run_binding_sha256':bound,'canonical_stdin_sha256':f.canonical_hash(b),'file':file.name,'sha256':stream_hash(file),'start_sha256':stream_hash(start)}
            if entry['receipt']!=receipt or read_json(directory/(stem+'-receipt.json'))!=receipt or any(read_json(start).get(k)!=receipt[k] for k in ('run_binding_sha256','canonical_stdin_sha256')):raise ValueError('receipt/start binding differs')
            attempt=read_json(file);outer,models=c.checked_outer(attempt)
            if models!=entry['actual_model_usage_ids'] or attempt.get('canonical_stdin_sha256')!=f.canonical_hash(b):raise ValueError('attempt/model metadata differs')
            try:values=parse(outer,b['expected_ids'])
            except f.InvalidSubmission:
                if ai==first or entry['structural_valid'] is not False:raise
                continue
            if ai!=first or entry['structural_valid'] is not True:raise ValueError('earlier valid ratings not retained')
            ratings.extend(values)
        records.append(record)
    if read_json(directory/'accepted.json')!={'ratings':ratings,'batches':records,'binding_sha256':bound,'first_valid_only':True}:raise ValueError('accepted labels changed')
    mapping={e['id']:e for batch in read_json(run/'inputs/plans.json')[phase] for e in batch};decoded={}
    for rating in ratings:
        e=mapping[rating['id']];decoded.setdefault(e['group_id'],{})[e['condition']]=rating
    return decoded,records


def eligibility_result(run):
    cfg,_=verify(run);reviews,_=read_reviews(run,'eligibility');rows=read_json(run/'inputs/extraction.json')['items']
    result=p.eligibility(rows,reviews,read_json(run/'inputs/ratios.json'),cfg['lexicon'],cfg['target_trait'])
    result['accepted_sha256']=stream_hash(run/'grades/eligibility/accepted.json');return result


def eligibility(args,run):
    result=eligibility_result(run);save(run/'eligibility.json',result)
    if not result['sufficient']:stop(run,'eligibility_null')


@contextmanager
def gpu_stage(args,run,phase):
    if not args.execute_gpu:raise ValueError('separate GPU release and --execute-gpu required')
    guard(run)
    if (run/'stopped.json').exists():raise RuntimeError('permanent stop')
    cfg,m=registered(args,run,models=True)
    try:resource=resource_check(cfg)
    except RuntimeError as exc:raise ResourceWait('resource guard did not release this phase') from exc
    out=run/phase;out.mkdir(mode=0o700,exist_ok=False)
    import torch
    started=time.perf_counter();completed=False;progress={'saved_generations':0,'teacher_forced_sequences':0,'mc_items':0,'synchronized_generation_wall_seconds':0.0,'teacher_forced_forward_wall_seconds_including_capture':0.0}
    try:
        model,tok=load_model(cfg,m,'bf16')
        if any(v.dtype!=torch.bfloat16 or v.device!=torch.device('cuda:0') for v in model.parameters()):raise ValueError('BF16 on one logical cuda:0 required')
        torch.cuda.synchronize();yield cfg,m,model,tok,out,progress;torch.cuda.synchronize();completed=True
    except BaseException as exc:stop(run,phase+'_fatal',exc);raise
    finally:
        save(out/'run.json',{'completed':completed,**resource,**progress,'elapsed_wall_seconds':time.perf_counter()-started,
            'wall_scope':'Synchronized stage wall including model load, tokenization, capture and serialization; generation wall reported separately',
            'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),
            'memory_scope':'PyTorch allocator including model; not total VRAM','cuda_runtime':torch.version.cuda,
            'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),'dtype':'bfloat16','attention':'sdpa','seed':42,
            'tf32_matmul':torch.backends.cuda.matmul.allow_tf32,'tf32_cudnn':torch.backends.cudnn.allow_tf32,
            'failed_phase_counts':'saved/completed work only; interrupted unsaved work may be additional' if not completed else None,
            'files_sha256':{str(file.relative_to(out)):stream_hash(file) for file in out.rglob('*') if file.is_file()}})
        try:verify(run)
        except BaseException as exc:stop(run,'post_GPU_hash_failure',exc);raise


def vectors(run):
    from safetensors.torch import load_file
    from safetensors import safe_open
    cfg,_=verify(run)
    record=read_json(run/'extraction/vectors.json')
    if read_json(run/'eligibility.json')!=eligibility_result(run):raise ValueError('eligibility subset no longer matches frozen ratings')
    if (record['safetensors_sha256']!=stream_hash(run/'extraction/vectors.safetensors') or record['eligibility_sha256']!=stream_hash(run/'eligibility.json')
        or record['manifest_sha256']!=stream_hash(run/'manifest.json') or record['target_trait']!=cfg['target_trait'] or record['model_id']!=cfg['model_id'] or record['revision']!=cfg['revision'] or record['extraction_layers']!=list(p.LAYERS)
        or record['extraction_input_sha256']!=extraction_hashes(run)):raise ValueError('vector/model/input/manifest binding differs')
    expected=safe_metadata(run,cfg)
    with safe_open(str(run/'extraction/vectors.safetensors'),framework='pt',device='cpu') as file:
        if record['safe_metadata']!=expected or file.metadata()!=expected:raise ValueError('safetensors identity metadata differs')
    return load_file(str(run/'extraction/vectors.safetensors'),device='cpu')


def extraction_hashes(run):
    return {n:stream_hash(run/n) for n in ('inputs/extraction.json','inputs/encoded.json','inputs/ratios.json','inputs/rubric.json','eligibility.json','grades/eligibility/accepted.json')}


def safe_metadata(run,cfg):
    return {'model_id':cfg['model_id'],'revision':cfg['revision'],'target_trait':cfg['target_trait'],'layers':'17,23','scope':'response-token decoder-block output means; no weight edits',
        'normalization':'d=mean(target_mean-neutral_mean); u=d/L2(d); R=median(L2(neutral_mean)); delta=beta*R*u',
        'manifest_sha256':stream_hash(run/'manifest.json')}


def extract(args,run):
    verify(run)
    if read_json(run/'eligibility.json')['sufficient'] is not True:raise ValueError('insufficient eligible pairs')
    if read_json(run/'eligibility.json')!=eligibility_result(run):raise ValueError('eligibility/ratings drift before extraction')
    with gpu_stage(args,run,'extraction') as (cfg,m,model,tok,out,progress):
        import torch
        rows=read_json(run/'inputs/extraction.json')['items'];retained=read_json(run/'eligibility.json')['retained_ids'];saved=read_json(run/'inputs/encoded.json');pairs={l:[] for l in p.LAYERS};boundaries=[]
        for row in rows:
            if row['id'] not in retained:continue
            sides=[]
            for side in ('neutral','target'):
                encoded=response_encoding(tok,row,row[side],cfg['system'])
                if encoded!=saved[row['id']][side]:raise ValueError('actual extraction boundary differs from frozen tokenizer')
                ids=torch.tensor([encoded['input_ids']],device='cuda:0');mask=torch.tensor([encoded['response_mask']],dtype=torch.bool);collected={}
                torch.cuda.synchronize();forward_started=time.perf_counter()
                with p.capture_response(model,p.LAYERS,mask,collected),torch.inference_mode():
                    model(input_ids=ids,attention_mask=torch.ones_like(ids),position_ids=torch.arange(ids.shape[1],device=ids.device)[None],use_cache=False,logits_to_keep=1)
                torch.cuda.synchronize();progress['teacher_forced_forward_wall_seconds_including_capture']+=time.perf_counter()-forward_started
                sides.append(collected);progress['teacher_forced_sequences']+=1
                boundaries.append({'id':row['id'],'side':side,'response_tokens':sum(encoded['response_mask']),'excluded_whitespace_boundary_tokens':encoded['excluded_whitespace_boundary_tokens']})
            for l in p.LAYERS:pairs[l].append((sides[0][l],sides[1][l]))
        tensors={};stats={}
        for l in p.LAYERS:
            values,diagnostics=p.direction_statistics(pairs[l],l);tensors.update(values);stats[str(l)]=diagnostics
        identity=safe_metadata(run,cfg);save_vectors(out/'vectors.safetensors',tensors,identity)
        save(out/'vectors.json',{'layers':stats,'safetensors_sha256':stream_hash(out/'vectors.safetensors'),'eligibility_sha256':stream_hash(run/'eligibility.json'),
            'model_id':cfg['model_id'],'revision':cfg['revision'],'target_trait':cfg['target_trait'],'extraction_layers':list(p.LAYERS),'scope':identity['scope'],'normalization':identity['normalization'],
            'safe_metadata':identity,'extraction_input_sha256':extraction_hashes(run),
            'boundary_audit':boundaries,'tensor_dtype':'float32','actual_intervention_dtype':'bfloat16','round_trip_exact':True,'manifest_sha256':stream_hash(run/'manifest.json')})


def preflight(args,run):
    tensors=vectors(run)
    with gpu_stage(args,run,'preflight') as (cfg,m,model,tok,out,progress):
        import torch
        rows=read_json(run/'inputs/development.json')['items'][:4];zero_audit={};nonzero_audit={}
        ordinary=generated(model,tok,rows,cfg,'zero baseline',cap=32);count_generated(progress,ordinary);save(out/'ordinary.json',ordinary)
        hooked=generated(model,tok,rows,cfg,'actual zero-vector hook',torch.zeros_like(tensors['17.unit']),17,32,zero_audit);count_generated(progress,hooked);save(out/'zero.json',hooked)
        exact=all(a['generated_ids']==b['generated_ids'] and a['input_ids']==b['input_ids'] for a,b in zip(ordinary['predictions'],hooked['predictions']))
        change=p.delta(tensors,17,.05)
        smoke=generated(model,tok,rows[:2],cfg,'unscored nonzero smoke',change,17,32,nonzero_audit);count_generated(progress,smoke);save(out/'smoke.json',smoke)
        from scripts.batched_generation import left_pad
        sequences=[tok(render_prompt(tok,{**r,'system':cfg['system']}),add_special_tokens=False)['input_ids'] for r in rows[:2]]
        inputs={k:torch.tensor(v,device='cuda:0') for k,v in left_pad(sequences,tok.pad_token_id).items()}
        with torch.inference_mode():
            base=model(**inputs,logits_to_keep=1).logits[:,-1].float()
            with steer(model,17,change.to(device='cuda:0',dtype=torch.bfloat16),1):changed=model(**inputs,logits_to_keep=1).logits[:,-1].float()
        nonzero_audit['max_abs_last_prompt_logit_delta']=(changed-base).abs().max().item()
        okay=exact and all(a.get('prefill_calls',0)>0 and a.get('decode_calls',0)>0 for a in (zero_audit,nonzero_audit)) and nonzero_audit.get('max_abs_hidden_delta',0)>0 and nonzero_audit['max_abs_last_prompt_logit_delta']>0
        save(out/'checks.json',{'zero_exact':exact,'zero_audit':zero_audit,'nonzero_audit':nonzero_audit,'hooks_pass':okay,'hooks_cleanup':'context managers removed all installed handles'})
        if not okay:raise ValueError('zero parity or real-hook instrumentation failed')
    save(run/'strength.json',{'manifest_sha256':stream_hash(run/'manifest.json'),'vectors_sha256':stream_hash(run/'extraction/vectors.safetensors'),
        'metadata_sha256':stream_hash(run/'extraction/vectors.json'),'extraction_run_sha256':stream_hash(run/'extraction/run.json'),
        'preflight_run_sha256':stream_hash(run/'preflight/run.json'),'checks_sha256':stream_hash(run/'preflight/checks.json')})


def strength(args,run):
    registered(args,run);committed(args.strength_registration);record=read_json(args.strength_registration)
    if record['strength_sha256']!=stream_hash(run/'strength.json'):raise ValueError('committed strength artifact differs')
    s=read_json(run/'strength.json');current={'manifest_sha256':stream_hash(run/'manifest.json'),'vectors_sha256':stream_hash(run/'extraction/vectors.safetensors'),
        'metadata_sha256':stream_hash(run/'extraction/vectors.json'),'extraction_run_sha256':stream_hash(run/'extraction/run.json'),
        'preflight_run_sha256':stream_hash(run/'preflight/run.json'),'checks_sha256':stream_hash(run/'preflight/checks.json')}
    if s!=current or read_json(run/'preflight/checks.json')['hooks_pass'] is not True:raise ValueError('strength/probe source drift')


def dev(args,run):
    strength(args,run);tensors=vectors(run)
    with gpu_stage(args,run,'development') as (cfg,m,model,tok,out,progress):
        rows=read_json(run/'inputs/development.json')['items'];results={}
        for name in p.CONDITIONS:
            layer,beta=p.SETTINGS.get(name,(None,None));value=generated(model,tok,rows,cfg,name,None if name=='baseline' else p.delta(tensors,layer,beta),layer)
            results[name]=value;save(out/(name+'.json'),value);count_generated(progress,value)
        save(out/'predictions.json',results)
        save(out/'divergence.json',{name:p.divergence(results['baseline']['predictions'],value['predictions']) for name,value in results.items() if name!='baseline'})


def metrics(run,phase,pass_index=None):
    """Confirmation: None=two-pass consensus, 1=first only, 2=repeat only."""
    cfg,_=verify(run);reviews,_=read_reviews(run,phase);second=None
    if phase=='confirmation' and pass_index!=1:
        repeat,_=read_reviews(run,'confirmation_repeat')
        if pass_index==2:reviews=repeat
        else:second=repeat
    values=outputs(run,phase);rows=read_json(run/'inputs'/(generation_phase(phase)+'.json'))['items']
    return {name:p.measures(rows,v,{r['id']:reviews[r['id']][name] for r in rows},cfg['lexicon'],cfg['target_trait'],
        {r['id']:second[r['id']][name] for r in rows} if second else None) for name,v in values.items()}


def select(args,run):
    strength(args,run);result=p.choose(metrics(run,'development'));save(run/'selection.json',result)
    if result['development_null']:stop(run,'development_null')


def lock_record(args,run):
    strength(args,run)
    if read_json(run/'selection.json')!=p.choose(metrics(run,'development')):raise ValueError('selection no longer matches development evidence')
    return {'manifest_sha256':stream_hash(run/'manifest.json'),'selection_sha256':stream_hash(run/'selection.json'),
        'strength_sha256':stream_hash(run/'strength.json'),'strength_registration_sha256':stream_hash(path(args.strength_registration)),
        'vectors_sha256':stream_hash(run/'extraction/vectors.safetensors'),'metadata_sha256':stream_hash(run/'extraction/vectors.json'),
        'accepted_development_sha256':stream_hash(run/'grades/development/accepted.json'),'development_packet_sha256':stream_hash(run/'judging/development/manifest.json'),
        'development_run_sha256':stream_hash(run/'development/run.json'),'confirmation_input_sha256':stream_hash(run/'inputs/confirmation.json'),
        'retention_input_sha256':stream_hash(run/'inputs/retention.json'),'plans_sha256':stream_hash(run/'inputs/plans.json')}


def lock(args,run):
    if read_json(run/'selection.json')['selected'] is None:raise ValueError('development null; no final lock')
    save(run/'final_lock.json',lock_record(args,run))


def checked_lock(args,run):
    if read_json(run/'final_lock.json')!=lock_record(args,run):raise ValueError('selection/vector/rating/input hashes drifted after lock')
    committed(args.selection_registration)
    if read_json(args.selection_registration)['lock_sha256']!=stream_hash(run/'final_lock.json'):raise ValueError('committed final selection differs')
    return read_json(run/'selection.json')['setting']


def final(args,run):
    setting=checked_lock(args,run);tensors=vectors(run);layer,beta=setting['layer'],setting['beta']
    with gpu_stage(args,run,'confirmation') as (cfg,m,model,tok,out,progress):
        original=read_json(run/'development/predictions.json')[read_json(run/'selection.json')['selected']]['predictions'][:4]
        replay=generated(model,tok,read_json(run/'inputs/development.json')['items'][:4],cfg,'selected vector reload',p.delta(tensors,layer,beta),layer)
        count_generated(progress,replay);save(out/'reload.json',replay)
        fields=('id','input_ids','generated_ids','batch_ids','batch_position','padded_prompt_width','ended_eos','hit_token_limit')
        if any(any(a[k]!=b[k] for k in fields) for a,b in zip(original,replay['predictions'])):raise ValueError('fresh vector reload first-batch tokens/layout differ')
        rows=read_json(run/'inputs/confirmation.json')['items'];results={}
        for name,change in [('baseline',None),('selected',p.delta(tensors,layer,beta)),('random',p.delta(tensors,layer,beta,random=True))]:
            value=generated(model,tok,rows,cfg,name,change,layer);results[name]=value;save(out/(name+'.json'),value);count_generated(progress,value)
        save(out/'predictions.json',results);save(out/'lock.json',{'lock_sha256':stream_hash(run/'final_lock.json'),'reload_exact':True})
        save(out/'divergence.json',{name:p.divergence(results['baseline']['predictions'],value['predictions']) for name,value in results.items() if name!='baseline'})


def retention(args,run):
    setting=checked_lock(args,run);read_reviews(run,'confirmation');read_reviews(run,'confirmation_repeat');tensors=vectors(run)
    with gpu_stage(args,run,'retention') as (cfg,m,model,tok,out,progress):
        from langdetect import DetectorFactory
        old_seed=DetectorFactory.seed;results={}
        try:
            for name in ('baseline','selected'):
                directory=out/name;directory.mkdir();diagnostics=[];DetectorFactory.seed=0
                audit={};context=nullcontext() if name=='baseline' else p.audited_steer(model,setting['layer'],p.delta(tensors,setting['layer'],setting['beta']).to(device='cuda:0',dtype=next(model.parameters()).dtype),audit)
                with context:result=run_screen(model,tok,read_json(run/'inputs/retention.json'),directory,writing=None,gguf=None,tasks=['general_mc','physics_mc','gsm8k','ifeval'],diagnostics=diagnostics)
                if name!='baseline':save(directory/'hook_audit.json',audit)
                results[name]=result;save(directory/'diagnostics.json',diagnostics)
                progress['saved_generations']+=sum(result[t]['total'] for t in ('gsm8k','ifeval'));progress['mc_items']+=sum(result[t]['total'] for t in ('general_mc','physics_mc'))
                progress['synchronized_generation_wall_seconds']+=sum(d['hf_generation_timing']['generation_wall_seconds'] for d in diagnostics)
        finally:DetectorFactory.seed=old_seed
        save(out/'metrics.json',results);save(out/'lock.json',{'lock_sha256':stream_hash(run/'final_lock.json'),'language_detector_seed':0,'mc_batch_size':16,'generation_batch_size':8,'gsm8k_cap':512,'ifeval_cap':1024})


def with_stop_status(result):
    if result['stop'] is not None and result['stop']['stop_cause'] not in ('eligibility_null','development_null'):
        result['observed_status']=result['status'];result['status']='incomplete';result['acceptance_withheld_due_to_fatal_stop']=True
        if 'final_quality' in result:
            result['final_quality']['observed_all_criteria_pass']=result['final_quality']['all_criteria_pass']
            result['final_quality']['all_criteria_pass']=False
    return result


def report(args,run):
    cfg,m=verify(run);result={'scope':'Separate isolated CAA-inspired authored-pair trait study; no general style/safety claim','target_trait':cfg['target_trait'],'model_id':cfg['model_id'],'revision':cfg['revision'],
        'manifest_sha256':stream_hash(run/'manifest.json'),'source_sha256':m['sources'],'stop':read_json(run/'stopped.json') if (run/'stopped.json').exists() else None,
        'budgets':{'target_generations':230,'teacher_forced_sequences':64,'mc_items':64,'judge_calls':80},
        'costs':{n:read_json(run/n/'run.json') for n in ('extraction','preflight','development','confirmation','retention') if (run/n/'run.json').exists()},
        'calls_started':len(list((run/'grades').glob('*/batch-*-attempt-*-start.json'))),'grading_provenance':{},'status':'incomplete',
        'limits':['Authored intentions and AI judgments are not human ground truth; calibrated clear examples do not establish sensitivity on degraded or mild outputs.',
            'Small disjoint authored families and historically used capability probes do not establish general independent persona control.',
            'Primary target presence is judged; lexical profanity remains a separate fixed-lexicon diagnostic.',
            'Shared tasks are historically used screens, not newly unseen tests. Context is judge-only; model input contains row.user and helpful system.',
            'Unwelcome-task and closed-discussion phrase families are related; declared family labels do not establish semantic independence.',
            'One pinned judge family with fresh contexts; repeated ratings are not independent human truth. Server-side dated snapshot unavailable. No reselection after confirmation.']}
    result['divergence']={}
    for phase in ('development','confirmation'):
        if (run/phase/'divergence.json').exists():result['divergence'][phase]={name:{k:v for k,v in value.items() if k!='items'} for name,value in read_json(run/phase/'divergence.json').items()}
    result['confirmation_passes']={}
    for pi,phase in [(1,'confirmation'),(2,'confirmation_repeat')]:
        if (run/'grades'/phase/'accepted.json').exists():result['confirmation_passes'][str(pi)]=metrics(run,'confirmation',pass_index=pi)
    if len(result['confirmation_passes'])==2:
        first,_=read_reviews(run,'confirmation');second,_=read_reviews(run,'confirmation_repeat')
        result['repeat_diagnostics']=p.repeat_diagnostics(read_json(run/'inputs/confirmation.json')['items'],('baseline','selected','random'),first,second)
    result['judging_costs']={}
    result['hook_audits']={}
    for phase in ('development','confirmation'):
        if (run/phase/'predictions.json').exists():result['hook_audits'][phase]={name:value['hook_audit'] for name,value in read_json(run/generation_phase(phase)/'predictions.json').items() if 'hook_audit' in value}
    if (run/'confirmation/reload.json').exists():result['hook_audits']['reload']=read_json(run/'confirmation/reload.json').get('hook_audit')
    if (run/'retention/selected/hook_audit.json').exists():result['hook_audits']['retention']=read_json(run/'retention/selected/hook_audit.json')
    for phase in p.PHASES:
        costs=[];durations=[];responses=0
        for file in (run/'grades'/phase).glob('batch-*-attempt-*.json'):
            if not re.fullmatch(r'batch-\d{3}-attempt-[01]\.json',file.name):continue
            responses+=1
            try:outer=json.loads(read_json(file)['stdout'])
            except (ValueError,KeyError,TypeError):continue
            if type(outer.get('total_cost_usd')) in (int,float):costs.append(outer['total_cost_usd'])
            if type(outer.get('duration_api_ms')) in (int,float):durations.append(outer['duration_api_ms'])
        result['judging_costs'][phase]={'response_files':responses,'responses_with_cost':len(costs),'known_cost_usd':sum(costs),'known_duration_api_ms':sum(durations)}
    for phase in p.PHASES:
        if (run/'grades'/phase/'accepted.json').exists():
            _,records=read_reviews(run,phase)
            result['grading_provenance'][phase]={'accepted_sha256':stream_hash(run/'grades'/phase/'accepted.json'),
                'first_valid_attempts':[r['first_valid_attempt'] for r in records],'model_usage_ids':sorted({v for r in records for e in r['attempts'] for v in e['actual_model_usage_ids']}),
                'attempt_sha256':[e['receipt']['sha256'] for r in records for e in r['attempts']]}
    if (run/'eligibility.json').exists():
        e=read_json(run/'eligibility.json');result['eligibility']={k:v for k,v in e.items() if k not in ('retained_ids','audit')}
        if e!=eligibility_result(run):raise ValueError('eligibility report no longer matches labels')
        if not e['sufficient']:result.update(status='eligibility_null',later_phases_skipped=True)
    if (run/'extraction/vectors.json').exists():
        vectors(run);v=read_json(run/'extraction/vectors.json');result['vectors']={k:v[k] for k in ('layers','safetensors_sha256','tensor_dtype','actual_intervention_dtype','round_trip_exact')}
    if (run/'selection.json').exists():
        s=read_json(run/'selection.json')
        if s!=p.choose(metrics(run,'development')):raise ValueError('selection/ratings drift')
        result['development']=s
        if s['selected'] is None:result.update(status='development_null',later_phases_skipped=True)
        elif (run/'retention/metrics.json').exists() and all((run/'grades'/phase/'accepted.json').exists() for phase in ('confirmation','confirmation_repeat')):
            checked_lock(args,run);source_files(run,'retention');source_files(run,'confirmation');quality=read_json(run/'retention/metrics.json');confirmation=metrics(run,'confirmation')
            if any(read_json(run/n/'lock.json')['lock_sha256']!=stream_hash(run/'final_lock.json') for n in ('confirmation','retention')):raise ValueError('final/retention lock binding differs')
            result.update(status='complete',confirmation=confirmation,retention=quality,final_quality=p.final_gates(confirmation,quality),lock_sha256=stream_hash(run/'final_lock.json'))
    # Known aggregate schema only; private rows/aliases/answer text never exported.
    def privacy(v):
        if isinstance(v,dict):
            if set(v)&{'id','ids','user','context','text','output','input_ids','generated_ids','ratings','reference','retained_ids','boundary_audit'}:raise ValueError('private export field')
            for x in v.values():privacy(x)
        elif isinstance(v,list):
            for x in v:privacy(x)
        elif isinstance(v,str) and re.search(r'/home/|/tmp/|hf_[A-Za-z0-9]{12,}|Bearer\s+\S+',v):raise ValueError('private path/credential')
    with_stop_status(result);privacy(result);verify(run);save(path(args.output),result)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=('prepare','verify','packet','judge','eligibility','extract','preflight','dev','select','lock','final','retention','report'))
    parser.add_argument('--trait',choices=tuple(RUNS),default='rudeness');parser.add_argument('--config');parser.add_argument('--registration')
    parser.add_argument('--strength-registration');parser.add_argument('--selection-registration')
    parser.add_argument('--phase',choices=tuple(p.PHASES),default='eligibility');parser.add_argument('--output')
    parser.add_argument('--execute-gpu',action='store_true');parser.add_argument('--execute-judgments',action='store_true');parser.add_argument('--resume',action='store_true')
    args=parser.parse_args();run=path(RUNS[args.trait])
    defaults={'config':f'configs/{args.trait}_steering.json','registration':f'results/protocols/{args.trait}-steering-01.json',
        'strength_registration':f'results/protocols/{args.trait}-strength-01.json','selection_registration':f'results/protocols/{args.trait}-selection-01.json',
        'output':f'results/{args.trait}_steering_01.json'}
    for key,value in defaults.items():
        if getattr(args,key) is None:setattr(args,key,value)
        elif path(getattr(args,key)).resolve()!=path(value).resolve():raise ValueError('profile-specific config/registration/output paths required')
    if args.stage=='verify':verify(run,models=True);print('Pilot bindings verified.')
    else:
        live=(args.stage in ('extract','preflight','dev','final','retention') and args.execute_gpu) or (args.stage=='judge' and args.execute_judgments)
        try:globals()[args.stage](args,run)
        except ResourceWait:raise
        except BaseException as exc:
            if live:guard(run);stop(run,'live_stage_fatal',exc)
            raise


if __name__=='__main__':main()
