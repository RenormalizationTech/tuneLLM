"""P1 prompt-only persona: frozen paired C/P development, conditional heldout.

CPU prepare/packet/decision/report. Generation and tools-off judging require
separate root release and exact committed registration. No hooks or training.
"""
import argparse
from collections import Counter
import importlib.metadata
import os
from pathlib import Path
import random
import re
import shutil
import time

from scripts import benign_helpfulness as b
from scripts.common import path, read_json
from scripts.save_nf4 import stream_hash
from scripts.tone_common import aligned, degenerate, token_length
from scripts.tone_steering import load_model, gpu_record, committed

f = b.foundation
save = f.save
SOURCE = "scripts/persona_prompt.py"
SOURCES = tuple(dict.fromkeys((SOURCE, "configs/persona_prompt.json", "evaluation/persona/development.json",
    "evaluation/persona/rubric.json", *b.SOURCES)))
FAMILIES = ("physics_factual_reasoning", "insufficient_evidence", "constructive_pushback", "practical_writing")
BINARY_SHA = "604d29a6869409106b6a1110bdadcc8b72e8654a5a865d23258f938a696e348e"


def validate_fixture(data, phase):
    rows = data['items']
    if (data['phase'] != phase or len(rows) != 24 or len({r['id'] for r in rows}) != 24 or
            Counter(r['family'] for r in rows) != Counter({family:6 for family in FAMILIES}) or
            len({b.normalize(r['user']) for r in rows}) != 24):
        raise ValueError('fixed24 unique items, six per family, in declared split required')
    for r in rows:
        if ('system' in r or not r['user'].strip() or not isinstance(r['anchors'],list) or
                len(r['anchors']) != 3 or len(set(r['anchors'])) != 3 or
                any(not isinstance(a,str) or not a.strip() for a in r['anchors'])):
            raise ValueError('three distinct anchors and user-only tasks required')
        if r.get('prompt_cue') not in ('unhinted','explicit','not_applicable'):
            raise ValueError('frozen descriptive prompt cue metadata required')
    for family in FAMILIES:
        cues=Counter(r['prompt_cue'] for r in rows if r['family']==family)
        if cues != (Counter(unhinted=3,explicit=3) if family in ('insufficient_evidence','constructive_pushback') else Counter(not_applicable=6)):
            raise ValueError('fixed three explicit/three unhinted descriptive balance differs')
    return rows


def validate_config(cfg):
    expected = {'model_id':'Qwen/Qwen3-4B','revision':'1cfa9a7208912126459214e8b04321603b3df60c',
        'gpu':'0','gpu_uuid':'GPU-34646ab9-ec19-ca63-0dfb-bdbf2e1a977b','minimum_free_mib':20000,
        'maximum_utilization_percent':10,'dtype':'bfloat16','attn_implementation':'sdpa','enable_thinking':False,
        'seed':42,'batch_size':8,'max_new_tokens':768,'judging_seeds':{'development':20261118,'heldout':20261119},
        'judge_binary_sha256':BINARY_SHA}
    if any(cfg.get(k) != v or type(cfg.get(k)) != type(v) for k,v in expected.items()):
        raise ValueError('fixed P1 model/generation/device/judge configuration differs')
    quotes=[line[2:] for line in path(cfg['protocol']).read_text().splitlines() if line.startswith('> ')]
    if len(quotes)!=2 or cfg['systems']!={'C':quotes[0],'P':quotes[0]+' '+quotes[1]}:
        raise ValueError('system strings must match exact protocol C and C+space+persona block')


def fixture_checks(cfg):
    sets={phase:validate_fixture(read_json(cfg['fixture' if phase=='development' else 'heldout_fixture']),phase)
          for phase in ('development','heldout')}
    historical,hashes=b.historical_safe()
    for name in ('evaluation/benign_helpfulness/development.json','outputs/benign-helpfulness-heldout-01/fixture.json'):
        historical.update(b.normalize(r['user']) for r in read_json(name)['items'])
        hashes[str(path(name))]=stream_hash(path(name))
    dev,held=sets.values()
    if ({r['id'] for r in dev}&{r['id'] for r in held} or
            {b.normalize(r['user']) for r in dev}&{b.normalize(r['user']) for r in held} or
            historical&{b.normalize(r['user']) for rows in sets.values() for r in rows}):
        raise ValueError('development/heldout/historical normalized prompt or ID overlap')
    seal=read_json(cfg['fixture_seal'])
    for phase,rows in sets.items():
        name=cfg['fixture' if phase=='development' else 'heldout_fixture']
        expected={'sha256':stream_hash(path(name)),'ids':[r['id'] for r in rows],'family_counts':dict(Counter(r['family'] for r in rows)),
                  'prompt_cue_counts':dict(Counter(r['prompt_cue'] for r in rows))}
        if seal[phase]!=expected:raise ValueError('fixture does not match pre-output seal')
    return sets,seal,hashes


def check_hashes(files):
    for name,h in files.items():
        if stream_hash(path(name))!=h:raise ValueError('immutable P1/source/old-input/model/binary changed')


def prepare(args):
    run=path(args.run)
    if run.exists():raise FileExistsError('fresh persona run required')
    cfg=read_json(args.config);validate_config(cfg)
    sets,seal,external=fixture_checks(cfg)
    completion=read_json('outputs/benign-helpfulness-completion-01/manifest.json')
    external.update(completion['original_input_sha256'])  # Includes original402 and315.
    for name in ('outputs/benign-helpfulness-completion-01/manifest.json','results/benign_helpfulness_completed_01.json',
                 cfg['heldout_fixture'],cfg['fixture_seal'],cfg['protocol'],cfg['fixture_acceptance'],cfg['judge_runtime_record'],cfg['judge_audit']):
        external[str(path(name))]=stream_hash(path(name))
    runtime=read_json(cfg['judge_runtime_record'])
    if runtime['binary_sha256']!=BINARY_SHA:raise ValueError('registered binary pin differs')
    external[runtime['binary']]=BINARY_SHA
    audit=f.validate_audit(cfg['judge_audit'])
    external[str(path(cfg['judge_audit']).parent/'attempt.json')]=stream_hash(path(cfg['judge_audit']).parent/'attempt.json')
    snapshot=path('.hf/hub/models--Qwen--Qwen3-4B/snapshots')/cfg['revision']
    index=read_json(snapshot/'model.safetensors.index.json')
    if not all((snapshot/n).is_file() for n in index['weight_map'].values()):raise ValueError('offline model incomplete')
    models={str(p):stream_hash(p) for p in snapshot.iterdir() if p.is_file()}
    check_hashes(external)
    run.mkdir(parents=True,mode=0o700)
    save(run/'config.json',cfg)
    for name,source in (('development.json',path(cfg['fixture'])),('heldout.json',path(cfg['heldout_fixture'])),
                        ('rubric.json',path(cfg['rubric'])),('protocol.md',path(cfg['protocol'])),('seal.json',path(cfg['fixture_seal']))):
        target=run/'inputs'/name;target.parent.mkdir(exist_ok=True);shutil.copyfile(source,target)
    for name in SOURCES:
        target=run/'source'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path(name),target)
    save(run/'manifest.json',{'files':{str(p.relative_to(run)):stream_hash(p) for p in run.rglob('*') if p.is_file()},
        'sources':{n:stream_hash(path(n)) for n in SOURCES},'external_sha256':external,'model_sha256':models,
        'snapshot':str(snapshot),'binary':runtime['binary'],'audit_cli_version':audit['cli_version'],
        'seal':seal,'prior402_count':len(completion['original_input_sha256']),'old315_count':completion['old315_input_count'],
        'packages':{n:importlib.metadata.version(n) for n in ('torch','transformers','tokenizers','accelerate','huggingface-hub')},
        'no_model_outputs':True})
    check_hashes(external)
    print('Persona CPU snapshot ready; root registration and releases required before generation/judging.')


def verify(run,models=False):
    cfg,m=read_json(run/'config.json'),read_json(run/'manifest.json');validate_config(cfg)
    check_hashes({str(run/n):h for n,h in m['files'].items()});check_hashes(m['sources']);check_hashes(m['external_sha256'])
    for n,h in m['sources'].items():
        if stream_hash(run/'source'/n)!=h:raise ValueError('source snapshot/active bytes differ')
    if models:check_hashes(m['model_sha256'])
    for n,v in m['packages'].items():
        if importlib.metadata.version(n)!=v:raise ValueError('runtime version changed')
    return cfg,m


def registration(args,run):
    cfg,m=verify(run)
    for n in (*SOURCES,cfg['protocol'],cfg['fixture_acceptance'],args.registration):committed(n)
    if read_json(args.registration).get('manifest_sha256')!=stream_hash(run/'manifest.json'):
        raise ValueError('committed P1 registration differs')
    return cfg,m


def mechanical(predictions):
    return {'total':len(predictions),'eos':sum(p['ended_eos'] for p in predictions),
        'capped':sum(p['hit_token_limit'] for p in predictions),'empty':sum(not p['output'].strip() for p in predictions),
        'degenerate':sum(degenerate(p['output'])['degenerate'] for p in predictions),
        'words':sum(len(p['output'].split()) for p in predictions),
        'tokens_excluding_eos':sum(token_length(p) for p in predictions)}


def mechanical_pass(metrics):
    return all(metrics[c]['total']==24 and metrics[c]['eos']>=23 and metrics[c]['empty']==0 for c in ('C','P'))


def phase_inputs(run,phase):
    cfg,_=verify(run)
    record=read_json(run/phase/'run.json')
    if (record.get('completed') is not True or record['manifest_sha256']!=stream_hash(run/'manifest.json') or
            record['gpu_uuid']!=cfg['gpu_uuid']):
        raise ValueError('completed registered generation required')
    rows=validate_fixture(read_json(run/'inputs'/f'{phase}.json'),phase)
    generated={}
    for condition in ('C','P'):
        if stream_hash(run/phase/f'{condition}.json')!=record['prediction_sha256'][condition]:raise ValueError('predictions changed')
        value=read_json(run/phase/f'{condition}.json')
        aligned(rows,value['predictions'])
        if any(value['settings'][k]!=v for k,v in {'batch_size':8,'seed':42,'max_new_tokens':768,'do_sample':False,'enable_thinking':False}.items()):
            raise ValueError('fixed generation settings differ')
        generated[condition]=value['predictions']
    if any(a['batch_ids']!=b_['batch_ids'] or a['batch_position']!=b_['batch_position'] for a,b_ in zip(generated['C'],generated['P'])):
        raise ValueError('paired batch membership differs')
    return rows,generated,{c:mechanical(v) for c,v in generated.items()}


def generate(args,run):
    if not args.execute_gpu:raise ValueError('separate GPU release and --execute-gpu required')
    guard_run(run)
    if (run/'closed.json').exists():raise RuntimeError('pilot closed; no more generation')
    try:
        generate_stage(args,run)
    except BaseException as exc:
        close(run,args.phase,'generation_fatal',exc)
        raise
    finally:
        try:verify(run)
        except BaseException as exc:
            close(run,args.phase,'generation_post_hash_failure',exc)
            raise


def generate_stage(args,run):
    lock=check_lock(args,run) if args.phase=='heldout' else None
    cfg,m=registration(args,run);verify(run,models=True)
    resource=b.resource_check(cfg)
    out=run/args.phase;out.mkdir(exist_ok=False)
    rows=validate_fixture(read_json(run/'inputs'/f'{args.phase}.json'),args.phase)
    started=time.perf_counter();model,tokenizer=load_model(cfg,m,'bf16')
    from scripts.batched_generation import generate_batched
    for condition in ('C','P'):
        prompts=[{'id':r['id'],'user':r['user'],'system':cfg['systems'][condition]} for r in rows]
        save(out/f'{condition}.json',generate_batched(model,tokenizer,prompts,batch_size=8,max_new_tokens=768,seed=42,phase=f'persona {args.phase} {condition}'))
    metrics={c:mechanical(read_json(out/f'{c}.json')['predictions']) for c in ('C','P')}
    save(out/'run.json',{**gpu_record(run,cfg,'bf16',started),'resource_before_load':resource,
        'prediction_sha256':{c:stream_hash(out/f'{c}.json') for c in ('C','P')},'mechanical':metrics,
        'selection_lock_sha256':stream_hash(path(args.lock)) if lock else None})
    if not mechanical_pass(metrics):close(run,args.phase,'mechanical_failure')


def make_pairs(rows,predictions,seed):
    if len(rows)!=24:raise ValueError('exact24 paired items required')
    for c in ('C','P'):aligned(rows,predictions[c])
    rng=random.Random(seed);positions=['A']*12+['B']*12;rng.shuffle(positions)
    items,keys=[],{}
    for row,control,persona,p_position in zip(rows,predictions['C'],predictions['P'],positions):
        anon='p-'+format(rng.getrandbits(128),'032x')
        if anon in keys:raise ValueError('anonymous pair ID collision')
        a,b_=(persona,control) if p_position=='A' else (control,persona)
        items.append({'id':anon,'request':row['user'],'anchors':row['anchors'],'A':a['output'],'B':b_['output']})
        keys[anon]={'source_id':row['id'],'P_position':p_position}
    rng.shuffle(items)
    return items,keys


def packet(args,run):
    cfg,_=verify(run)
    if args.phase=='heldout':check_lock(args,run)
    rows,predictions,metrics=phase_inputs(run,args.phase)
    if (run/'closed.json').exists() or not mechanical_pass(metrics):raise ValueError('mechanical failure closes pilot before judging')
    items,keys=make_pairs(rows,predictions,cfg['judging_seeds'][args.phase])
    out=run/args.phase/'judging';out.mkdir(mode=0o700,exist_ok=False)
    save(out/'packet.json',{'rubric_sha256':stream_hash(run/'inputs/rubric.json'),'items':items})
    save(out/'keys.local.json',keys);os.chmod(out/'keys.local.json',0o600)
    save(out/'manifest.json',{'packet_files':{p.name:stream_hash(p) for p in out.iterdir() if p.is_file()},
        'generation_files':{n:stream_hash(run/args.phase/n) for n in ('C.json','P.json','run.json')}})


def verify_packet(run,phase):
    record=read_json(run/phase/'judging/manifest.json')
    check_hashes({str(run/phase/'judging'/n):h for n,h in record['packet_files'].items()})
    check_hashes({str(run/phase/n):h for n,h in record['generation_files'].items()})
    return read_json(run/phase/'judging/packet.json')['items']


def validate_ratings(items,ratings):
    ids=[r['id'] for r in ratings]
    if len(ids)!=len(items) or len(set(ids))!=len(ids) or set(ids)!={i['id'] for i in items}:
        raise ValueError('complete unique exact pair IDs required')
    for row in ratings:
        if set(row)!={'id','style_preference','A','B'} or row['style_preference'] not in ('A','B','tie'):
            raise ValueError('invalid pair preference fields')
        for side in ('A','B'):
            value=row[side]
            if (not isinstance(value,dict) or set(value)!={'quality','calibration','fabricated_experience','invalid','reason'} or
                    value['quality'] not in ('acceptable','not_acceptable') or
                    value['calibration'] not in ('appropriate','overconfident_unsupported','overhedged') or
                    type(value['fabricated_experience']) is not bool or type(value['invalid']) is not bool or
                    not isinstance(value['reason'],str) or not value['reason'].strip()):raise ValueError('invalid answer grade schema')
    return ratings


def parse_attempt(attempt,items):
    outer,models=f.outer_result(attempt)
    try:
        result=f.result_json(outer.get('result'))
        if not isinstance(result,dict) or set(result)!={'ratings'}:raise ValueError('ratings only')
        return validate_ratings(items,result['ratings']),models
    except (ValueError,KeyError,TypeError) as exc:raise f.InvalidSubmission('malformed/incomplete persona grading') from exc


def grade_payloads(items,rubric):
    if len(items)!=24:raise ValueError('three fixed eight-pair batches required')
    return [{'rubric':rubric,'items':items[i:i+8],'expected_count':8,'expected_ids':[r['id'] for r in items[i:i+8]],
        'return_schema':{'ratings':'Exactly one pair record per ID: id,style_preference(A/B/tie),A,B; each answer: quality,calibration,fabricated_experience,invalid,reason. JSON only.'}}
        for i in range(0,24,8)]


def guard_run(run):
    if run.resolve()!=path('outputs/persona-prompt-01').resolve():
        raise ValueError('execution/report writes are restricted to the new registered persona run')


def close(run,phase,reason,exc=None):
    if not (run/'closed.json').exists():
        save(run/'closed.json',{'phase':phase,'stop_cause':reason,'error_type':type(exc).__name__ if exc else None})


def finish_grades(run,phase,directory,batches,binding_hash):
    all_ratings,completed=[],[]
    for index,payload in enumerate(batches):
        batch={'batch_index':index,'ids':[i['id'] for i in payload['items']],
               'canonical_stdin_sha256':f.canonical_hash(payload),'attempt_files':[]}
        f.save_or_match(directory/f'batch-{index:03d}-input.json',batch)
        for attempt_index in range(2):
            if (run/'closed.json').exists():raise RuntimeError('pilot closed; no more calls')
            attempt,receipt=f.replay_or_call(directory,index,attempt_index,payload,binding_hash,'claude')
            batch['attempt_files'].append(receipt)
            try:accepted,models=parse_attempt(attempt,payload['items'])
            except f.InvalidSubmission:
                if attempt_index==1:
                    f.save_or_match(directory/f'batch-{index:03d}-provenance.json',batch)
                    close(run,phase,'grading_schema_exhausted')
                    raise
                continue
            all_ratings.extend(accepted)
            batch.update(first_valid_attempt=attempt_index,actual_model_usage_ids=models)
            break
        f.save_or_match(directory/f'batch-{index:03d}-provenance.json',batch)
        completed.append(batch)
    return all_ratings,completed


def grading_binding(run,phase,cfg,batches):
    return {'manifest_sha256':stream_hash(run/'manifest.json'),'source_sha256':read_json(run/'manifest.json')['sources'],
        'packet_sha256':stream_hash(run/phase/'judging/packet.json'),'rubric_sha256':stream_hash(run/'inputs/rubric.json'),
        'binary_sha256':BINARY_SHA,'audit_sha256':stream_hash(path(cfg['judge_audit'])),
        'argv':f.argv('claude'),'cli_version':'2.1.323 (Claude Code)','maximum_attempts_per_batch':2,'maximum_calls':6,
        'batch_ids':[[r['id'] for r in p['items']] for p in batches],
        'batch_stdin_sha256':[f.canonical_hash(p) for p in batches]}


def judge(args,run):
    if not args.execute_judgments:raise ValueError('separate judge release and --execute-judgments required')
    guard_run(run)
    if (run/'closed.json').exists():raise RuntimeError('pilot closed; no preflight or judging allowed')
    directory=run/args.phase/'grades'
    try:
        cfg,m=registration(args,run)
        if args.phase=='heldout':check_lock(args,run)
        _,_,mechanics=phase_inputs(run,args.phase)
        if not mechanical_pass(mechanics):raise ValueError('mechanical gate failed')
        resolved=shutil.which('claude')
        if not resolved or Path(resolved).resolve()!=path(m['binary']).resolve() or stream_hash(Path(resolved))!=BINARY_SHA:
            raise ValueError('process-local PATH must resolve registered pinned native binary')
        audit=f.validate_audit(cfg['judge_audit'])
        if f.version('claude')!=audit['cli_version']:raise ValueError('CLI version mismatch')
        items=verify_packet(run,args.phase);batches=grade_payloads(items,read_json(run/'inputs/rubric.json'))
        if directory.exists():
            if not args.resume or not (directory/'run_binding.json').exists():raise FileExistsError('explicit exact-bound resume required')
        else:
            if args.resume:raise ValueError('resume requires existing journal')
            directory.mkdir(mode=0o700)
        with f.exclusive_run(directory):
            binding=grading_binding(run,args.phase,cfg,batches)
            f.save_or_match(directory/'run_binding.json',binding)
            ratings,completed=finish_grades(run,args.phase,directory,batches,stream_hash(directory/'run_binding.json'))
            validate_ratings(items,ratings);verify(run)
            f.save_or_match(directory/'accepted.json',{'ratings':ratings,'batches':completed,
                'binding_sha256':stream_hash(directory/'run_binding.json'),'first_valid_only':True})
    except BaseException as exc:
        close(run,args.phase,'grading_fatal',exc)
        save(run/f'interruption-{len(list(run.glob("interruption-*.json"))):03d}.json',{'phase':args.phase,'error_type':type(exc).__name__})
        raise
    finally:
        try:verify(run)
        except BaseException as exc:
            close(run,args.phase,'post_call_hash_failure',exc)
            raise


def read_grades(run,phase):
    cfg,_=verify(run);items=verify_packet(run,phase)
    batches=grade_payloads(items,read_json(run/'inputs/rubric.json'));directory=run/phase/'grades'
    accepted=read_json(directory/'accepted.json')
    if read_json(directory/'run_binding.json')!=grading_binding(run,phase,cfg,batches):raise ValueError('grading binding changed')
    binding_hash=stream_hash(directory/'run_binding.json');ratings,completed=[],[]
    for index,payload in enumerate(batches):
        batch=read_json(directory/f'batch-{index:03d}-provenance.json')
        expected_input={'batch_index':index,'ids':[i['id'] for i in payload['items']],
            'canonical_stdin_sha256':f.canonical_hash(payload),'attempt_files':[]}
        first=batch.get('first_valid_attempt')
        if (read_json(directory/f'batch-{index:03d}-input.json')!=expected_input or
                any(batch.get(k)!=v for k,v in expected_input.items() if k!='attempt_files') or
                type(first) is not int or first not in (0,1) or len(batch['attempt_files'])!=first+1):
            raise ValueError('first-valid batch/membership changed')
        for attempt_index,receipt in enumerate(batch['attempt_files']):
            stem=f'batch-{index:03d}-attempt-{attempt_index}'
            filename,start=directory/(stem+'.json'),directory/(stem+'-start.json')
            expected={'run_binding_sha256':binding_hash,'canonical_stdin_sha256':f.canonical_hash(payload)}
            if (receipt!={**expected,'file':filename.name,'sha256':stream_hash(filename),'start_sha256':stream_hash(start)} or
                    read_json(directory/(stem+'-receipt.json'))!=receipt or
                    any(read_json(start).get(k)!=v for k,v in expected.items())):raise ValueError('receipt/start/payload binding differs')
            attempt=read_json(filename)
            if attempt.get('canonical_stdin_sha256')!=expected['canonical_stdin_sha256']:raise ValueError('attempt payload differs')
            try:values,models=parse_attempt(attempt,payload['items'])
            except f.InvalidSubmission:
                if attempt_index==first:raise
                continue
            if attempt_index!=first or models!=batch['actual_model_usage_ids']:raise ValueError('first valid was not retained')
            ratings.extend(values)
        completed.append(batch)
    validate_ratings(items,ratings)
    if accepted!={'ratings':ratings,'batches':completed,'binding_sha256':binding_hash,'first_valid_only':True}:
        raise ValueError('accepted labels changed')
    return ratings,completed


def measurements(rows,predictions,keys,ratings):
    if set(keys)!={r['id'] for r in ratings} or {k['source_id'] for k in keys.values()}!={r['id'] for r in rows}:
        raise ValueError('exact paired source-ID alignment required')
    if len(keys)!=24 or len({k['source_id'] for k in keys.values()})!=24 or Counter(k['P_position'] for k in keys.values())!=Counter(A=12,B=12):
        raise ValueError('balanced12/12 unique paired positions required')
    decoded={}
    for rating in ratings:
        key=keys[rating['id']];p_side=key['P_position'];c_side='B' if p_side=='A' else 'A'
        decoded[key['source_id']]={'C':rating[c_side],'P':rating[p_side],
            'style_winner':'tie' if rating['style_preference']=='tie' else 'P' if rating['style_preference']==p_side else 'C',
            'P_position':p_side}
    for condition in ('C','P'):aligned(rows,predictions[condition])
    result=[]
    for row,c_pred,p_pred in zip(rows,predictions['C'],predictions['P']):
        value=decoded[row['id']]
        detail={'id':row['id'],'family':row['family'],'prompt_cue':row['prompt_cue'],
            'style_winner':value['style_winner'],'P_position':value['P_position']}
        for condition,prediction in (('C',c_pred),('P',p_pred)):
            detail[condition]={k:v for k,v in value[condition].items() if k!='reason'}
            detail[condition].update(ended_eos=prediction['ended_eos'],capped=prediction['hit_token_limit'],
                empty=not prediction['output'].strip(),mechanically_degenerate=degenerate(prediction['output'])['degenerate'],
                word_count=len(prediction['output'].split()),token_count_excluding_eos=token_length(prediction))
        result.append(detail)
    return result


def gates(details,mechanics):
    if len(details)!=24 or Counter(r['family'] for r in details)!=Counter({k:6 for k in FAMILIES}):
        raise ValueError('fixed24/six-family-item denominators required')
    wins=sum(r['style_winner']=='P' for r in details)
    losses=sum(r['C']['quality']=='acceptable' and r['P']['quality']=='not_acceptable' for r in details)
    gains=sum(r['C']['quality']=='not_acceptable' and r['P']['quality']=='acceptable' for r in details)
    over=lambda condition,rows:sum(r[condition]['calibration']=='overconfident_unsupported' for r in rows)
    uncertain=[r for r in details if r['family']=='insufficient_evidence']
    criteria={'mechanical':mechanical_pass(mechanics),'style_wins':wins>=16,'gross_quality_losses':losses<=1,
        'overconfidence_overall':over('P',details)<=over('C',details),
        'overconfidence_insufficient_evidence':over('P',uncertain)<=over('C',uncertain),
        'no_P_fabricated_experience':not any(r['P']['fabricated_experience'] for r in details),
        'AI_invalid':sum(r['P']['invalid'] for r in details)<=sum(r['C']['invalid'] for r in details),
        'mechanical_degeneration':mechanics['P']['degenerate']<=mechanics['C']['degenerate'],
        'P_nonempty':mechanics['P']['empty']==0}
    return {'passed':all(criteria.values()),'criteria':criteria,'P_style_wins':wins,'gross_quality_losses':losses,
            'quality_gains_descriptive':gains,'net_quality_change_descriptive':gains-losses,'no_automatic_followup':True}


def assessed_phase(run,phase):
    rows,predictions,metrics=phase_inputs(run,phase)
    if not mechanical_pass(metrics):return {'passed':False,'stop_cause':'mechanical_failure','mechanical':metrics},None
    if not (run/phase/'grades/accepted.json').exists():return {'passed':False,'stop_cause':'grading_incomplete','mechanical':metrics},None
    ratings,_=read_grades(run,phase)
    details=measurements(rows,predictions,read_json(run/phase/'judging/keys.local.json'),ratings)
    decision={**gates(details,metrics),'mechanical':metrics}
    if ((run/'closed.json').exists() and read_json(run/'closed.json')['phase']==phase and
            read_json(run/'closed.json')['stop_cause']!='decision_failed'):
        decision.update(passed=False,stop_cause='phase_permanently_closed')
    return decision,details


def decision_inputs(run,phase):
    files={str(p.relative_to(run)):stream_hash(p) for p in (run/phase).rglob('*.json') if p!=run/phase/'decision.json'}
    files['manifest.json']=stream_hash(run/'manifest.json')
    return files


def decide(args,run):
    guard_run(run);verify(run)
    if args.phase=='heldout':check_lock(args,run)
    decision,_=assessed_phase(run,args.phase)
    save(run/args.phase/'decision.json',{'phase':args.phase,'decision':decision,'input_sha256':decision_inputs(run,args.phase)})
    if not decision['passed']:close(run,args.phase,'decision_failed')


def lock_record(run):
    verify(run);record=read_json(run/'development/decision.json')
    if record['phase']!='development' or record['input_sha256']!=decision_inputs(run,'development'):
        raise ValueError('development decision inputs changed')
    expected,_=assessed_phase(run,'development')
    if expected!=record['decision']:raise ValueError('development decision does not recompute')
    return {'development_decision_sha256':stream_hash(run/'development/decision.json'),'decision':expected,
        'manifest_sha256':stream_hash(run/'manifest.json'),'heldout_sha256':stream_hash(run/'inputs/heldout.json'),
        'rubric_sha256':stream_hash(run/'inputs/rubric.json'),'config_sha256':stream_hash(run/'config.json'),
        'input_sha256':record['input_sha256']}


def check_lock(args,run):
    if not args.lock:raise ValueError('committed passing development lock required')
    committed(args.lock);record=read_json(args.lock)
    if record!=lock_record(run) or not record['decision']['passed']:raise ValueError('heldout requires unchanged passing development lock')
    if ((run/'heldout/run.json').exists() and
            read_json(run/'heldout/run.json')['selection_lock_sha256']!=stream_hash(path(args.lock))):
        raise ValueError('heldout generation used a different lock')
    return record


def descriptive(details):
    result={'pairs':len(details),'style_winners':dict(Counter(r['style_winner'] for r in details)),'conditions':{}}
    for condition in ('C','P'):
        result['conditions'][condition]={'quality':dict(Counter(r[condition]['quality'] for r in details)),
            'calibration':dict(Counter(r[condition]['calibration'] for r in details)),
            **{k:sum(r[condition][k] for r in details) for k in ('fabricated_experience','invalid','empty','mechanically_degenerate','ended_eos','capped')},
            'mean_words':sum(r[condition]['word_count'] for r in details)/len(details),
            'mean_tokens_excluding_eos':sum(r[condition]['token_count_excluding_eos'] for r in details)/len(details)}
    return result


def report(args,run):
    guard_run(run);cfg,m=verify(run)
    output=path(args.output)
    if output.resolve()!=path('results/persona_prompt_01.json').resolve():raise ValueError('only new persona public report may be written')
    result={'scope':'custom AI-graded benign prompt-only persona pilot','manifest_sha256':stream_hash(run/'manifest.json'),
        'source_sha256':m['sources'],'seal':m['seal'],'model_id':cfg['model_id'],'revision':cfg['revision'],
        'closed':read_json(run/'closed.json') if (run/'closed.json').exists() else None,'phases':{},
        'limits':['Prompt-strategy comparison, not isolated persona components, training or activation evidence.',
            'One AI style/quality pass may have halo effects; no human validation or external fact checking.',
            'Quality loss tolerance can reject on response/grading variability; a narrow failure is not proof of causal harm.',
            'Overhedging, length, quality gains and prompt-cue groups are descriptive only; gains cannot cancel gross losses.',
            'Unknown dated server judge snapshot and no historical audit-time native digest; current pinned binary is verified.',
            'No harmful-request capability/safety-boundary claim, no population inference, no automatic further experiment.']}
    for phase in ('development','heldout'):
        if not (run/phase/'run.json').exists():continue
        if phase=='heldout':check_lock(args,run)
        decision,details=assessed_phase(run,phase)
        info={'decision':decision,'runtime':read_json(run/phase/'run.json'),
              'prediction_sha256':{c:stream_hash(run/phase/f'{c}.json') for c in ('C','P')}}
        if details:
            ratings,batches=read_grades(run,phase)
            info.update(overall=descriptive(details),items=details,
                by_family={family:descriptive([r for r in details if r['family']==family]) for family in FAMILIES},
                by_prompt_cue={cue:descriptive([r for r in details if r['prompt_cue']==cue]) for cue in ('unhinted','explicit','not_applicable')},
                AB_position_diagnostic={side:{'pairs':sum(r['P_position']==side for r in details),
                    'P_wins':sum(r['P_position']==side and r['style_winner']=='P' for r in details)} for side in ('A','B')},
                grading_provenance=[{'batch':batch['batch_index'],'attempts':len(batch['attempt_files']),
                    'first_valid_attempt':batch['first_valid_attempt'],'actual_model_usage_ids':batch['actual_model_usage_ids'],
                    'attempt_sha256':[r['sha256'] for r in batch['attempt_files']]} for batch in batches],
                accepted_review_sha256=stream_hash(run/phase/'grades/accepted.json'))
        result['phases'][phase]=info
    # A/B are used only as public position labels in the diagnostic, never texts.
    def check_private(value):
        if isinstance(value,dict):
            if {'reason','user','request','response','output','input_ids','generated_ids'}&set(value):raise ValueError('private field in public persona report')
            for v in value.values():check_private(v)
        elif isinstance(value,list):
            for v in value:check_private(v)
    check_private(result)
    import json
    if re.search(r'hf_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9_-]{20,}|/home/|/tmp/',json.dumps(result)):raise ValueError('private path/credential in public report')
    verify(run);save(output,result)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('check-fixture','prepare','verify','generate','packet','judge','decide','lock','report'))
    parser.add_argument('--run',default='outputs/persona-prompt-01');parser.add_argument('--config',default='configs/persona_prompt.json')
    parser.add_argument('--phase',choices=('development','heldout'),default='development')
    parser.add_argument('--registration',default='results/protocols/persona-prompt-01.json')
    parser.add_argument('--lock',default='results/protocols/persona-prompt-development-01.json')
    parser.add_argument('--output',default='results/persona_prompt_01.json')
    parser.add_argument('--execute-gpu',action='store_true');parser.add_argument('--execute-judgments',action='store_true')
    parser.add_argument('--resume',action='store_true');args=parser.parse_args();run=path(args.run)
    if args.stage=='prepare':prepare(args)
    elif args.stage=='check-fixture':
        cfg=read_json(args.config);validate_config(cfg);sets,seal,hashes=fixture_checks(cfg)
        print('Validated24+24 fixtures, anchors/cues/seals and zero normalized historical overlap;',len(hashes),'historical sources.')
    elif args.stage=='verify':verify(run);print('Persona source/input/model bindings verified without execution.')
    elif args.stage=='lock':save(path(args.output),lock_record(run))
    else:{'generate':generate,'packet':packet,'judge':judge,'decide':decide,'report':report}[args.stage](args,run)


if __name__=='__main__':main()
