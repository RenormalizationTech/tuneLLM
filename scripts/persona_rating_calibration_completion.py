"""Single infrastructure completion; unchanged ratings, 1 parent + at most23 new calls."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import re
import shutil
import subprocess

from scripts import persona_rating_calibration as c
from scripts.common import path, read_json
from scripts.save_nf4 import stream_hash

RUN='outputs/persona-rating-calibration-completion-01'
SOURCES=('scripts/persona_rating_calibration_completion.py','scripts/persona_rating_calibration_completion_runbook.md',
    'configs/persona_rating_calibration_completion.json',*c.SOURCES)
f,save=c.f,c.save
BINARY_SHA=c.BINARY_SHA


class BudgetExhausted(RuntimeError):pass


def guard(run):
    if run.resolve()!=path(RUN).resolve():raise ValueError('only new completion directory may be written')


def validate_config(cfg):
    fixed={'run':RUN,'parent':'outputs/persona-rating-calibration-01','parent_report':'results/persona_rating_calibration_01.json',
        'parent_manifest_sha256':'65f838f17aa1e13f3d71bad47b83562bd6cb43001b09dd99122dc4804dc9905f',
        'protocol':'docs/persona_rating_calibration_completion_protocol.md',
        'design_review':'results/reviews/persona_rating_calibration_completion_design_review.md',
        'parent_charged_calls':1,'maximum_new_calls':23,'maximum_cumulative_calls':24}
    if set(cfg)!=set(fixed) or any(type(cfg[k]) is not type(v) or cfg[k]!=v for k,v in fixed.items()):raise ValueError('fixed completion/budget settings differ')


def parent_preconditions(parent,report):
    """Read-only failure evidence; no journal repair or grading parser retries."""
    starts=list((parent/'grades').glob('pass-*/batch-*-attempt-*-start.json'))
    attempts=[p for p in (parent/'grades').glob('pass-*/batch-*-attempt-*.json') if re.fullmatch(r'batch-\d{3}-attempt-[01]\.json',p.name)]
    if len(starts)!=1 or len(attempts)!=1 or starts[0].relative_to(parent).as_posix()!='grades/pass-1/batch-000-attempt-0-start.json':raise ValueError('exactly one expected parent start/response required')
    if list((parent/'grades').glob('pass-*/accepted.json')):raise ValueError('parent cannot have accepted ratings')
    a=read_json(attempts[0]);outer=json.loads(a['stdout'])
    if a.get('returncode')!=1 or outer.get('is_error') is not True or outer.get('modelUsage')!={} or 'EAI_AGAIN' not in (a.get('stdout','')+a.get('stderr','')):
        raise ValueError('expected DNS-only failed parent precondition absent')
    def rating_json(value):
        if isinstance(value,dict):return 'ratings' in value or any(rating_json(v) for v in value.values())
        if isinstance(value,list):return any(rating_json(v) for v in value)
        return False
    try:body=f.result_json(outer.get('result'))
    except (ValueError,TypeError):body=None
    if rating_json(outer) or rating_json(body):raise ValueError('parent contains a rating response')
    if report.get('status')!='incomplete' or report.get('outcome')!='incomplete' or report.get('calls_started')!=1 or report.get('passes')!=[None,None]:raise ValueError('parent report must be incomplete with no pass labels')
    if read_json(parent/'stopped.json')!={'stop_cause':'judging_fatal','error_type':'RuntimeError'}:raise ValueError('parent stop evidence differs')
    stem=attempts[0].stem;receipt=read_json(attempts[0].with_name(stem+'-receipt.json'));start=read_json(starts[0])
    if receipt.get('sha256')!=stream_hash(attempts[0]) or receipt.get('start_sha256')!=stream_hash(starts[0]) or any(start.get(k)!=receipt.get(k) or a.get(k)!=receipt.get(k) for k in ('canonical_stdin_sha256',)):
        raise ValueError('parent failed receipt/payload binding changed')
    if receipt.get('run_binding_sha256')!=stream_hash(parent/'grades/run_binding.json') or start.get('run_binding_sha256')!=receipt['run_binding_sha256']:raise ValueError('parent start/run binding differs')
    return stream_hash(starts[0])


def prepare(args):
    run=path(RUN);guard(run)
    if run.exists():raise FileExistsError('one fresh completion only')
    cfg=read_json(args.config);validate_config(cfg);parent=path(cfg['parent']);science,old=c.verify(parent)
    if stream_hash(parent/'manifest.json')!=cfg['parent_manifest_sha256']:raise ValueError('parent manifest pin differs')
    start_sha=parent_preconditions(parent,read_json(cfg['parent_report']))
    protected=dict(old['protected_sha256']);protected.update(old['sources'])
    for p in parent.rglob('*'):
        if p.is_file():protected[str(p)]=stream_hash(p)
    for n in (cfg['parent_report'],cfg['protocol'],cfg['design_review']):protected[str(path(n))]=stream_hash(path(n))
    c.check_hashes(protected);run.mkdir(mode=0o700);save(run/'config.json',cfg)
    for name in ('texts.json','rubric.json','plans.json','lexical.json'):
        target=run/'inputs'/name;target.parent.mkdir(exist_ok=True);shutil.copyfile(parent/'inputs'/name,target)
    save(run/'inputs/budget.json',{'parent_charged_calls':1,'maximum_new_calls':23,'maximum_cumulative_calls':24,
        'maximum_attempts_per_batch':2,'parent_start_sha256':start_sha})
    for name in (*SOURCES,cfg['protocol'],cfg['design_review']):
        target=run/'source'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path(name),target)
    save(run/'manifest.json',{'files':{str(p.relative_to(run)):stream_hash(p) for p in run.rglob('*') if p.is_file()},
        'sources':{n:stream_hash(path(n)) for n in SOURCES},'protected_sha256':protected,'packages':old['packages'],
        'binary':old['binary'],'binary_sha256':old['binary_sha256'],'cli_version':old['cli_version'],'audit_sha256':old['audit_sha256'],
        'science_config':science,'budget_sha256':stream_hash(run/'inputs/budget.json'),'parent_start_sha256':start_sha,
        'no_API_GPU_or_model_calls':True})
    c.check_hashes(protected)


def verify(run):
    guard(run);cfg,m=read_json(run/'config.json'),read_json(run/'manifest.json');validate_config(cfg)
    c.check_hashes({str(run/n):h for n,h in m['files'].items()});c.check_hashes(m['sources']);c.check_hashes(m['protected_sha256'])
    if any(importlib.metadata.version(n)!=v for n,v in m['packages'].items()):raise ValueError('package differs')
    c.validate_config(m['science_config']);items=c.validate_fixture(read_json(run/'inputs/texts.json'));c.validate_plans(items,read_json(run/'inputs/plans.json'))
    parent_preconditions(path(cfg['parent']),read_json(cfg['parent_report']))
    return cfg,m


def registration(args,run):
    cfg,m=verify(run)
    for name in (*SOURCES,cfg['protocol'],cfg['design_review'],args.registration):
        relative=path(name).resolve().relative_to(path('.').resolve());result=subprocess.run(['git','show',f'HEAD:{relative.as_posix()}'],cwd=path('.'),capture_output=True,check=False,shell=False)
        if result.returncode or result.stdout!=path(name).read_bytes():raise ValueError('exact committed source/registration required')
    if read_json(args.registration)['manifest_sha256']!=stream_hash(run/'manifest.json'):raise ValueError('registration differs')
    return cfg,m


def budget_state(run,m,plans,items,rubric):
    file=run/'inputs/budget.json';budget=read_json(file)
    expected={'parent_charged_calls':1,'maximum_new_calls':23,'maximum_cumulative_calls':24,
        'maximum_attempts_per_batch':2,'parent_start_sha256':m['parent_start_sha256']}
    if stream_hash(file)!=m['budget_sha256'] or budget!=expected:raise ValueError('immutable cumulative budget changed')
    charges=run/'grades/charges';records={};slots=[]
    for p in charges.glob('*.json'):
        row=read_json(p);pi,bi,ai=(row[k] for k in ('pass_index','batch_index','attempt_index'))
        if any(type(v) is not int for v in (pi,bi,ai)) or pi not in (0,1) or bi not in range(6) or ai not in (0,1):raise ValueError('charge outside registered attempt space')
        body=c.payload(plans[pi]['batches'][bi],items,rubric)
        want={'pass_index':pi,'batch_index':bi,'attempt_index':ai,'slot':row['slot'],
            'budget_sha256':m['budget_sha256'],'run_binding_sha256':stream_hash(run/'grades/run_binding.json'),
            'canonical_stdin_sha256':f.canonical_hash(body)}
        name=f'pass-{pi+1}-batch-{bi:03d}-attempt-{ai}.json'
        if p.name!=name or row!=want or type(row['slot']) is not int:raise ValueError('charge binding differs')
        records[(pi,bi,ai)]=row;slots.append(row['slot'])
    if len(records)>23 or sorted(slots)!=list(range(1,len(records)+1)):raise ValueError('new/cumulative budget exceeded or charge sequence differs')
    starts=list((run/'grades').glob('pass-*/batch-*-attempt-*-start.json'))
    for p in starts:
        match=re.fullmatch(r'batch-(\d{3})-attempt-([01])-start.json',p.name)
        if p.parent.name not in ('pass-1','pass-2') or not match or (int(p.parent.name[-1])-1,int(match[1]),int(match[2])) not in records:raise ValueError('start without prior immutable charge')
    return {'charged_new_slots':len(records),'new_calls_started':len(starts),'cumulative_charged':1+len(records),'records':records}


def budgeted_replay(run,directory,pi,bi,ai,body,bound,m,plans,items,rubric):
    state=budget_state(run,m,plans,items,rubric);key=(pi,bi,ai);stem=f'batch-{bi:03d}-attempt-{ai}'
    artifacts=list(directory.glob(stem+'*'))
    if artifacts:
        if key not in state['records']:raise ValueError('existing attempt lacks prior charge')
    else:
        if key in state['records']:raise f.UncertainAttempt('charged slot interrupted before durable start; no repeated call')
        if state['charged_new_slots']>=23:raise BudgetExhausted('parent1 + new23 exhaust cumulative24 before another start')
        save(run/'grades/charges'/f'pass-{pi+1}-batch-{bi:03d}-attempt-{ai}.json',
            {'pass_index':pi,'batch_index':bi,'attempt_index':ai,'slot':state['charged_new_slots']+1,
             'budget_sha256':m['budget_sha256'],'run_binding_sha256':bound,'canonical_stdin_sha256':f.canonical_hash(body)})
    return f.replay_or_call(directory,bi,ai,body,bound,'claude')


def finish_pass(run,directory,pi,plan,items,rubric,bound,m,plans):
    ratings=[];records=[]
    for batch in plan['batches']:
        index=batch['batch_index'];body=c.payload(batch,items,rubric)
        record={'ids':batch['ids'],'cell_counts':batch['cell_counts'],'canonical_stdin_sha256':f.canonical_hash(body),'attempts':[]}
        f.save_or_match(directory/f'batch-{index:02d}-input.json',record)
        for ai in range(2):
            if (run/'stopped.json').exists():raise RuntimeError('completion permanently stopped')
            attempt,receipt=budgeted_replay(run,directory,pi,index,ai,body,bound,m,plans,items,rubric)
            outer,models=c.checked_outer(attempt);entry={'receipt':receipt,'actual_model_usage_ids':models,'structural_valid':False};record['attempts'].append(entry)
            try:values=c.parse_ratings(outer,batch['ids'])
            except f.InvalidSubmission:
                if ai==1:
                    f.save_or_match(directory/f'batch-{index:02d}-provenance.json',record);c.stop(run,'schema_exhausted');raise
                continue
            entry['structural_valid']=True;record['first_valid_attempt']=ai;ratings.extend(values);break
        f.save_or_match(directory/f'batch-{index:02d}-provenance.json',record);records.append(record)
    c.validate_ratings([r['id'] for r in items],ratings)
    return {'ratings':ratings,'batches':records,'binding_sha256':bound,'first_valid_only':True}


def judge(args,run):
    if not args.execute_judgments:raise ValueError('separate network-enabled release and --execute-judgments required')
    guard(run)
    if (run/'stopped.json').exists():raise RuntimeError('completion permanently stopped; no further completion')
    try:
        cfg,m=registration(args,run);resolved=shutil.which('claude')
        if not resolved or Path(resolved).resolve()!=path(m['binary']).resolve() or stream_hash(Path(resolved))!=BINARY_SHA or f.version('claude')!=m['cli_version']:raise ValueError('pinned native binary/version required')
        f.validate_audit(read_json(m['science_config']['judge_runtime_config'])['judge_audit'])
        items=read_json(run/'inputs/texts.json')['items'];rubric=read_json(run/'inputs/rubric.json');plans=read_json(run/'inputs/plans.json');root=run/'grades'
        if root.exists() and not args.resume:raise FileExistsError('exact journal resume required')
        if not root.exists() and args.resume:raise ValueError('resume requires existing journal')
        root.mkdir(mode=0o700,exist_ok=True)
        with f.exclusive_run(root):
            f.save_or_match(root/'run_binding.json',c.binding(run,m,plans,items,rubric));bound=stream_hash(root/'run_binding.json')
            c.audit_journal(run,plans,items,rubric);budget_state(run,m,plans,items,rubric)
            for pi,plan in enumerate(plans):
                directory=root/f'pass-{pi+1}';directory.mkdir(mode=0o700,exist_ok=True)
                accepted=finish_pass(run,directory,pi,plan,items,rubric,bound,m,plans)
                verify(run);f.save_or_match(directory/'accepted.json',accepted)
                if not c.pass_ok(c.pass_summary(items,accepted['ratings'])):c.stop(run,'calibration_axis_failure');return
    except BaseException as exc:c.stop(run,'budget_exhausted' if isinstance(exc,BudgetExhausted) else 'completion_fatal',exc);raise
    finally:
        try:verify(run)
        except BaseException as exc:c.stop(run,'post_call_hash_failure',exc);raise


def report(args,run):
    cfg,m=verify(run);items=read_json(run/'inputs/texts.json')['items'];rubric=read_json(run/'inputs/rubric.json');plans=read_json(run/'inputs/plans.json')
    c.audit_journal(run,plans,items,rubric);state=budget_state(run,m,plans,items,rubric);passes=[];provenance=[]
    for pi in range(2):
        ratings,records=c.read_pass(run,pi,m,items,rubric,plans);passes.append(ratings)
        provenance.append([{'batch_index':i,'first_valid_attempt':r['first_valid_attempt'],'attempt_sha256':[e['receipt']['sha256'] for e in r['attempts']],
            'actual_model_usage_ids':[e['actual_model_usage_ids'] for e in r['attempts']]} for i,r in enumerate(records)])
    result=c.decision(items,passes,m['science_config']['lexicon']);parent_report=read_json(cfg['parent_report'])
    result.update(scope='Single infrastructure completion of unchanged authored calibration; no target-model or GPU work',
        manifest_sha256=stream_hash(run/'manifest.json'),source_sha256=m['sources'],parent_report_sha256=stream_hash(path(cfg['parent_report'])),
        parent_manifest_sha256=cfg['parent_manifest_sha256'],budget_sha256=m['budget_sha256'],
        parent_charged_calls=1,maximum_new_calls=23,maximum_cumulative_calls=24,
        **{k:v for k,v in state.items() if k!='records'},cumulative_calls_started=1+state['new_calls_started'],
        accepted_pass_sha256=[stream_hash(run/'grades'/f'pass-{i+1}'/'accepted.json') if passes[i] is not None else None for i in range(2)],
        cell_bit_order=list(c.TRAITS),per_batch_cell_composition=[[b['cell_counts'] for b in p['batches']] for p in plans],grading_provenance=provenance,
        stop=read_json(run/'stopped.json') if (run/'stopped.json').exists() else None,limits=parent_report['limits']+[
            'The original sandbox DNS invocation is charged as one; no claim of zero external requests is made.',
            'A charged slot interrupted before a durable start remains conservatively spent; no automatic infrastructure recovery or further completion is authorized.'])
    if result['stop'] is not None and result['outcome']=='pass':result.update(outcome='incomplete',next_step='stop; user input required before further work')
    def privacy(v):
        if isinstance(v,dict):
            if set(v)&{'id','ids','text','context','intended','ratings','input_ids','generated_ids'}:raise ValueError('private per-item export')
            for x in v.values():privacy(x)
        elif isinstance(v,list):
            for x in v:privacy(x)
        elif isinstance(v,str) and re.search(r'/home/|/tmp/|hf_[A-Za-z0-9]{12,}|Bearer\s+\S+',v):raise ValueError('private path/credential')
    privacy(result);verify(run);save(path(args.output),result)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=('prepare','verify','judge','report'))
    parser.add_argument('--config',default='configs/persona_rating_calibration_completion.json')
    parser.add_argument('--registration',default='results/protocols/persona-rating-calibration-completion-01.json')
    parser.add_argument('--output',default='results/persona_rating_calibration_completion_01.json')
    parser.add_argument('--execute-judgments',action='store_true');parser.add_argument('--resume',action='store_true')
    args=parser.parse_args();run=path(RUN)
    if args.stage=='prepare':prepare(args)
    elif args.stage=='verify':verify(run);print('Completion bindings verified.')
    else:globals()[args.stage](args,run)


if __name__=='__main__':main()
