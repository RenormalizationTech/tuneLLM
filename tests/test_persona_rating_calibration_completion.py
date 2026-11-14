"""Mocked CPU budget/integrity checks; no preparation of a real run or API use."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import persona_rating_calibration_completion as w


def failed_parent(tmp_path):
    parent=tmp_path/'parent';parent.mkdir()
    w.save(parent/'grades/run_binding.json',{'synthetic':True});bound=w.stream_hash(parent/'grades/run_binding.json')
    directory=parent/'grades/pass-1';stem='batch-000-attempt-0'
    start={'run_binding_sha256':bound,'canonical_stdin_sha256':'synthetic-payload'}
    w.save(directory/(stem+'-start.json'),start)
    attempt={'returncode':1,'stdout':json.dumps({'is_error':True,'modelUsage':{},'api_error':'EAI_AGAIN','result':'API Error EAI_AGAIN'}),
        'stderr':'','canonical_stdin_sha256':'synthetic-payload'}
    w.save(directory/(stem+'.json'),attempt)
    w.save(directory/(stem+'-receipt.json'),{**start,'file':stem+'.json','sha256':w.stream_hash(directory/(stem+'.json')),'start_sha256':w.stream_hash(directory/(stem+'-start.json'))})
    w.save(parent/'stopped.json',{'stop_cause':'judging_fatal','error_type':'RuntimeError'})
    report={'status':'incomplete','outcome':'incomplete','calls_started':1,'passes':[None,None],'limits':['Synthetic test evidence only.']}
    report_file=tmp_path/'parent-report.json';w.save(report_file,report)
    return parent,report_file


def hashes(directory):return {str(p.relative_to(directory)):w.stream_hash(p) for p in directory.rglob('*') if p.is_file()}


def test_parent_expected_failure_is_read_only(tmp_path):
    parent,report=failed_parent(tmp_path);before=hashes(parent)
    assert w.parent_preconditions(parent,w.read_json(report))==w.stream_hash(parent/'grades/pass-1/batch-000-attempt-0-start.json')
    assert hashes(parent)==before


@pytest.mark.parametrize('change',['extra_start','accepted','ratings','different_error','model_usage','success','report'])
def test_reject_wrong_parent_preconditions(tmp_path,change):
    parent,report_file=failed_parent(tmp_path);report=w.read_json(report_file);directory=parent/'grades/pass-1'
    if change=='extra_start':w.save(directory/'batch-001-attempt-0-start.json',{})
    elif change=='accepted':w.save(directory/'accepted.json',{})
    elif change=='report':report['passes']=[[],None]
    else:
        file=directory/'batch-000-attempt-0.json';attempt=w.read_json(file);outer=json.loads(attempt['stdout'])
        if change=='ratings':outer['result']=json.dumps({'ratings':[]})
        elif change=='different_error':outer.update(api_error='EIO',result='API Error EIO')
        elif change=='model_usage':outer['modelUsage']={'unexpected':{}}
        elif change=='success':attempt['returncode']=0
        attempt['stdout']=json.dumps(outer);file.write_text(json.dumps(attempt))
    with pytest.raises(ValueError):w.parent_preconditions(parent,report)


def setup(tmp_path,monkeypatch):
    parent,report=failed_parent(tmp_path);run=tmp_path/'completion';run.mkdir()
    cfg=w.read_json('configs/persona_rating_calibration_completion.json');cfg.update(parent=str(parent),parent_report=str(report))
    science=w.read_json('configs/persona_rating_calibration.json');data=w.read_json(science['texts']);items=data['items'];rubric=w.read_json(science['rubric']);plans=w.c.make_plans(items,[421,422])
    for name,value in [('texts',data),('rubric',rubric),('plans',plans)]:w.save(run/'inputs'/(name+'.json'),value)
    start_sha=w.parent_preconditions(parent,w.read_json(report));budget={'parent_charged_calls':1,'maximum_new_calls':23,'maximum_cumulative_calls':24,'maximum_attempts_per_batch':2,'parent_start_sha256':start_sha}
    w.save(run/'inputs/budget.json',budget);w.save(run/'manifest.json',{'sources':{}})
    binary=tmp_path/'native';binary.write_text('fake executable; never run')
    m={'sources':{},'binary':str(binary),'cli_version':'2.1.323 (Claude Code)','audit_sha256':'synthetic',
        'science_config':science,'parent_start_sha256':start_sha,'budget_sha256':w.stream_hash(run/'inputs/budget.json')}
    monkeypatch.setattr(w,'guard',lambda r:None);monkeypatch.setattr(w,'verify',lambda r:(cfg,m));monkeypatch.setattr(w,'registration',lambda a,r:(cfg,m))
    monkeypatch.setattr(w,'BINARY_SHA',w.stream_hash(binary));monkeypatch.setattr(w.shutil,'which',lambda n:str(binary))
    monkeypatch.setattr(w.f,'version',lambda n:m['cli_version']);monkeypatch.setattr(w.f,'validate_audit',lambda p:{'cli_version':m['cli_version']})
    return run,parent,items,rubric,plans,m


def calls(monkeypatch,items,mode='valid'):
    labels={r['id']:{'id':r['id'],**{t:3*r['intended'][t] for t in w.c.TRAITS},'gross_failure':False} for r in items};invocations=[];counts={}
    def invoke(binary,body):
        invocations.append(copy.deepcopy(body));key=w.f.canonical_hash(body);counts[key]=counts.get(key,0)+1
        values=[labels[i] for i in body['expected_ids']]
        if mode=='retry' and counts[key]==1:values=values[:-1]
        if mode=='uncertain':raise OSError('synthetic interrupted transport')
        return {'returncode':0,'canonical_stdin_sha256':key,'isolation':{'git_not_repository':True,'git_environment_clean':True,'returncode':128},
            'stdout':json.dumps({'subtype':'success','modelUsage':{w.f.MODEL:{}},'result':json.dumps({'ratings':values})})}
    monkeypatch.setattr(w.f,'invoke',invoke);return invocations


def test_success_replay_spends_no_new_calls_parent_unchanged_and_private_export(tmp_path,monkeypatch):
    run,parent,items,rubric,plans,m=setup(tmp_path,monkeypatch);before=hashes(parent);invocations=calls(monkeypatch,items)
    w.judge(SimpleNamespace(execute_judgments=True,resume=False),run)
    assert len(invocations)==12
    state=w.budget_state(run,m,plans,items,rubric);assert state['charged_new_slots']==12 and state['cumulative_charged']==13
    charge_before=hashes(run/'grades/charges')
    w.judge(SimpleNamespace(execute_judgments=True,resume=True),run)
    assert len(invocations)==12 and hashes(run/'grades/charges')==charge_before and hashes(parent)==before
    out=tmp_path/'public.json';w.report(SimpleNamespace(output=str(out)),run);result=w.read_json(out)
    assert result['outcome']=='pass' and result['status']=='complete'
    assert result['parent_charged_calls']==1 and result['new_calls_started']==12 and result['cumulative_calls_started']==13
    assert result['maximum_new_calls']==23 and result['maximum_cumulative_calls']==24
    assert all(r['id'] not in out.read_text() and r['text'] not in out.read_text() for r in items)
    assert hashes(parent)==before


def test_exhaustion_precedes_24th_new_charge_or_start(tmp_path,monkeypatch):
    run,parent,items,rubric,plans,m=setup(tmp_path,monkeypatch);before=hashes(parent);invocations=calls(monkeypatch,items,'retry')
    with pytest.raises(w.BudgetExhausted):w.judge(SimpleNamespace(execute_judgments=True,resume=False),run)
    assert len(invocations)==23 and len(list((run/'grades/charges').glob('*.json')))==23
    assert len(list((run/'grades').glob('pass-*/batch-*-attempt-*-start.json')))==23
    assert not (run/'grades/pass-2/batch-005-attempt-1-start.json').exists()
    assert not (run/'grades/charges/pass-2-batch-005-attempt-1.json').exists()
    state=w.budget_state(run,m,plans,items,rubric);assert state['cumulative_charged']==24
    with pytest.raises(RuntimeError):w.judge(SimpleNamespace(execute_judgments=True,resume=True),run)
    assert len(invocations)==23 and hashes(parent)==before
    out=tmp_path/'public.json';w.report(SimpleNamespace(output=str(out)),run)
    assert w.read_json(out)['status']=='incomplete' and w.read_json(out)['outcome']!='pass'


def test_uncertain_call_is_charged_once_and_terminal(tmp_path,monkeypatch):
    run,parent,items,rubric,plans,m=setup(tmp_path,monkeypatch);invocations=calls(monkeypatch,items,'uncertain')
    with pytest.raises(OSError):w.judge(SimpleNamespace(execute_judgments=True,resume=False),run)
    assert len(invocations)==1 and w.budget_state(run,m,plans,items,rubric)['cumulative_charged']==2
    with pytest.raises(RuntimeError):w.judge(SimpleNamespace(execute_judgments=True,resume=True),run)
    assert len(invocations)==1


def test_budget_tamper_rejected_and_no_flag_harmless(tmp_path,monkeypatch):
    with pytest.raises(ValueError):w.judge(SimpleNamespace(execute_judgments=False),tmp_path)
    assert not list(tmp_path.iterdir())
    run,_,items,rubric,plans,m=setup(tmp_path,monkeypatch)
    file=run/'inputs/budget.json';value=w.read_json(file);value['maximum_new_calls']=24;file.write_text(json.dumps(value))
    with pytest.raises(ValueError):w.budget_state(run,m,plans,items,rubric)


def test_charged_before_start_crash_does_not_reissue(tmp_path,monkeypatch):
    run,_,items,rubric,plans,m=setup(tmp_path,monkeypatch);directory=run/'grades/pass-1';directory.mkdir(parents=True)
    w.save(run/'grades/run_binding.json',w.c.binding(run,m,plans,items,rubric));bound=w.stream_hash(run/'grades/run_binding.json')
    body=w.c.payload(plans[0]['batches'][0],items,rubric)
    w.save(run/'grades/charges/pass-1-batch-000-attempt-0.json',{'pass_index':0,'batch_index':0,'attempt_index':0,'slot':1,
        'budget_sha256':m['budget_sha256'],'run_binding_sha256':bound,'canonical_stdin_sha256':w.f.canonical_hash(body)})
    invocations=calls(monkeypatch,items)
    with pytest.raises(w.f.UncertainAttempt):w.budgeted_replay(run,directory,0,0,0,body,bound,m,plans,items,rubric)
    assert not invocations and not list(directory.glob('*start.json'))


def test_fixed_config_prevents_24_new_calls_and_parent_write_guard():
    cfg=w.read_json('configs/persona_rating_calibration_completion.json');w.validate_config(cfg)
    cfg['maximum_new_calls']=24
    with pytest.raises(ValueError):w.validate_config(cfg)
    with pytest.raises(ValueError):w.guard(w.path('outputs/persona-rating-calibration-01'))
