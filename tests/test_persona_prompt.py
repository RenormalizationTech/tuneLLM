"""CPU-only P1 fixtures, paired decoding/gates/locks and failure reporting."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import persona_prompt as p
from scripts.common import read_json

READ_GRADES = p.read_grades

def rows():return read_json('evaluation/persona/development.json')['items']


def predictions(data=None):
    data=data or rows()
    return {c:[{'id':r['id'],'output':'Synthetic test answer','generated_ids':[3,4],
                'ended_eos':True,'hit_token_limit':False,'batch_ids':[x['id'] for x in data[(i//8)*8:(i//8+1)*8]],
                'batch_position':i%8} for i,r in enumerate(data)] for c in ('C','P')}


def answer():return {'quality':'acceptable','calibration':'appropriate','fabricated_experience':False,'invalid':False,'reason':'Synthetic private note.'}


def ratings(items,keys):
    return [{'id':i['id'],'style_preference':keys[i['id']]['P_position'],'A':answer(),'B':answer()} for i in items]


def mechanics():return {c:{'total':24,'eos':24,'empty':0,'degenerate':0,'capped':0} for c in ('C','P')}


def details():
    return [{'id':r['id'],'family':r['family'],'prompt_cue':r['prompt_cue'],'style_winner':'P' if i<16 else 'tie',
             'C':answer(),'P':answer()} for i,r in enumerate(rows())]


def test_real_fixtures_sealed_disjoint_balanced_cues_and_exact_prompts():
    cfg=read_json('configs/persona_prompt.json');p.validate_config(cfg)
    sets,seal,history=p.fixture_checks(cfg)
    assert len(sets['development'])==len(sets['heldout'])==24 and len(history)==16
    assert seal['development']['sha256']=='b0537177afbdf8fd213813b250c1d9d0f81f05bdbd465422a798dfc982f89b4b'
    assert seal['heldout']['sha256']=='4799e84e2325b1f4d4b21a59973dfbb405ec2000c6df6c8fa3f07128eec96191'
    assert cfg['systems']['P'].startswith(cfg['systems']['C']+' Style:')
    bad=copy.deepcopy(cfg);bad['systems']['P']+=' '
    with pytest.raises(ValueError):p.validate_config(bad)


def test_packet_balanced_deterministic_anonymous_and_metadata_hidden():
    data=rows();pred=predictions(data)
    items,keys=p.make_pairs(data,pred,20261118)
    assert (items,keys)==p.make_pairs(data,pred,20261118)
    assert len(items)==24 and sum(k['P_position']=='A' for k in keys.values())==12
    assert all(set(i)=={'id','request','anchors','A','B'} for i in items)
    source={r['id']:r for r in data}
    for i in items:assert i['request']==source[keys[i['id']]['source_id']]['user']
    decoded=p.measurements(data,pred,keys,ratings(items,keys))
    assert all(d['style_winner']=='P' and 'reason' not in d['P'] for d in decoded)


@pytest.mark.parametrize('change',['missing','duplicate','extra','bad_boolean','bad_enum'])
def test_grade_schema_exact_ids_and_strict_types(change):
    items,keys=p.make_pairs(rows(),predictions(),1);values=ratings(items,keys)
    if change=='missing':values.pop()
    elif change=='duplicate':values[-1]=values[0]
    elif change=='extra':values.append({**values[0],'id':'extra'})
    elif change=='bad_boolean':values[0]['A']['invalid']=0
    else:values[0]['B']['calibration']='certain'
    with pytest.raises(ValueError):p.validate_ratings(items,values)


def test_style_threshold_ties_gross_losses_not_net_and_overhedging_descriptive():
    d=details();m=mechanics()
    assert p.gates(d,m)['passed']
    d[0]['style_winner']='tie';assert not p.gates(d,m)['passed']
    d=details();d[0]['P']['quality']=d[1]['P']['quality']='not_acceptable'
    d[2]['C']['quality']=d[3]['C']['quality']='not_acceptable'
    result=p.gates(d,m)
    assert result['gross_quality_losses']==2 and result['net_quality_change_descriptive']==0
    assert not result['criteria']['gross_quality_losses']
    d=details()
    for row in d:row['P']['calibration']='overhedged'
    assert p.gates(d,m)['passed']  # No invented overhedging gate.


def test_overconfidence_overall_and_insufficient_family_separate():
    d=details();d[0]['C']['calibration']='overconfident_unsupported';d[6]['P']['calibration']='overconfident_unsupported'
    result=p.gates(d,mechanics())
    assert result['criteria']['overconfidence_overall'] and not result['criteria']['overconfidence_insufficient_evidence']
    d=details();d[0]['P']['calibration']='overconfident_unsupported'
    result=p.gates(d,mechanics())
    assert not result['criteria']['overconfidence_overall'] and result['criteria']['overconfidence_insufficient_evidence']


@pytest.mark.parametrize('failure',['fabricated','invalid','degenerate','empty','eos'])
def test_independent_no_increase_and_mechanical_gates(failure):
    d,m=details(),mechanics()
    if failure=='fabricated':d[0]['P']['fabricated_experience']=True
    elif failure=='invalid':d[0]['P']['invalid']=True
    elif failure=='degenerate':m['P']['degenerate']=1
    elif failure=='empty':m['C']['empty']=1
    else:
        m['C']['eos']=23;assert p.gates(d,m)['passed'];m['C']['eos']=22
    assert not p.gates(d,m)['passed']


@pytest.fixture
def run(tmp_path,monkeypatch):
    cfg=read_json('configs/persona_prompt.json');m={'sources':{},'seal':{}}
    run=tmp_path/'persona';run.mkdir();p.save(run/'manifest.json',m)
    p.save(run/'config.json',cfg)
    p.save(run/'inputs/heldout.json',{'synthetic_sealed':True})
    p.save(run/'inputs/rubric.json',read_json('evaluation/persona/rubric.json'))
    original_path=p.path
    monkeypatch.setattr(p,'path',lambda value:run if str(value)=='outputs/persona-prompt-01' else original_path(value))
    monkeypatch.setattr(p,'verify',lambda *a,**k:(cfg,m))
    monkeypatch.setattr(p,'committed',lambda *a:None)
    items,keys=p.make_pairs(rows(),predictions(),1);grades=ratings(items,keys)
    p.save(run/'development/judging/keys.local.json',keys)
    p.save(run/'development/grades/accepted.json',{'synthetic':True})
    p.save(run/'development/run.json',{'completed':True})
    for condition in ('C','P'):p.save(run/'development'/f'{condition}.json',{'predictions':predictions()[condition]})
    monkeypatch.setattr(p,'phase_inputs',lambda *a:(rows(),predictions(),mechanics()))
    monkeypatch.setattr(p,'read_grades',lambda *a:(grades,[]))
    return run,cfg,m


def test_committed_passing_lock_recomputes_and_mutation_rejected(run):
    root,_,_=run
    p.decide(SimpleNamespace(phase='development'),root)
    locked=p.lock_record(root);assert locked['decision']['passed']
    filename=root/'passing-lock.json';p.save(filename,locked)
    assert p.check_lock(SimpleNamespace(lock=str(filename)),root)==locked
    (root/'development/grades/accepted.json').write_text('synthetic mutation')
    with pytest.raises(ValueError):p.lock_record(root)


def test_failed_development_no_heldout_and_no_gpu_before_lock(run,monkeypatch):
    root,_,_=run
    def forbidden(*a,**k):raise AssertionError('no model/resource access before passing lock')
    monkeypatch.setattr(p,'load_model',forbidden);monkeypatch.setattr(p.b,'resource_check',forbidden)
    args=SimpleNamespace(execute_gpu=True,phase='heldout',lock=str(root/'missing-lock.json'))
    with pytest.raises(FileNotFoundError):p.generate(args,root)
    assert (root/'closed.json').exists()
    with pytest.raises(RuntimeError):p.generate(args,root)


def test_generation_preflight_failure_closes_and_no_flag_is_harmless(run,monkeypatch):
    root,_,_=run
    with pytest.raises(ValueError):p.generate(SimpleNamespace(execute_gpu=False),root)
    assert not (root/'closed.json').exists()
    def fail(*a):raise ValueError('synthetic source/resource preflight failure')
    monkeypatch.setattr(p,'registration',fail)
    with pytest.raises(ValueError):p.generate(SimpleNamespace(execute_gpu=True,phase='development'),root)
    assert read_json(root/'closed.json')['stop_cause']=='generation_fatal'
    with pytest.raises(RuntimeError):p.generate(SimpleNamespace(execute_gpu=True,phase='development'),root)


def test_null_report_allows_stop_metadata_without_private_reason(run,monkeypatch,tmp_path):
    root,_,_=run
    bad=mechanics();bad['C']['eos']=22
    monkeypatch.setattr(p,'phase_inputs',lambda *a:(rows(),predictions(),bad))
    p.close(root,'development','mechanical_failure')
    output=tmp_path/'public.json'
    original_path=p.path
    monkeypatch.setattr(p,'path',lambda value:output if str(value)=='results/persona_prompt_01.json' else original_path(value))
    p.report(SimpleNamespace(output=str(output),lock='unused'),root)
    report=read_json(output)
    assert report['closed']['stop_cause']=='mechanical_failure'
    assert report['phases']['development']['decision']['passed'] is False
    assert report['phases']['development']['decision']['stop_cause']=='mechanical_failure'
    encoded=output.read_text()
    assert 'Synthetic private note' not in encoded and '"reason"' not in encoded


def test_generation_mechanical_marker_exports_null_report(tmp_path,monkeypatch):
    from scripts import batched_generation
    root=tmp_path/'mechanical';root.mkdir()
    cfg=read_json('configs/persona_prompt.json');m={'sources':{},'seal':{}}
    p.save(root/'manifest.json',m)
    p.save(root/'inputs/development.json',read_json('evaluation/persona/development.json'))
    output=tmp_path/'public-mechanical.json';original_path=p.path
    monkeypatch.setattr(p,'path',lambda value:root if str(value)=='outputs/persona-prompt-01' else output if str(value)=='results/persona_prompt_01.json' else original_path(value))
    monkeypatch.setattr(p,'verify',lambda *a,**k:(cfg,m));monkeypatch.setattr(p,'registration',lambda *a:(cfg,m))
    monkeypatch.setattr(p.b,'resource_check',lambda *a:{'synthetic':True})
    monkeypatch.setattr(p,'load_model',lambda *a:(object(),object()))
    monkeypatch.setattr(p,'gpu_record',lambda *a:{'completed':True,'gpu_uuid':cfg['gpu_uuid'],'manifest_sha256':p.stream_hash(root/'manifest.json')})
    calls=[]
    def fake_generate(*a,**k):
        condition='C' if not calls else 'P';calls.append(condition)
        values=predictions()[condition]
        if condition=='C':
            for value in values[:2]:value['ended_eos']=False;value['hit_token_limit']=True
        return {'predictions':values,'settings':{'batch_size':8,'seed':42,'max_new_tokens':768,'do_sample':False,'enable_thinking':False}}
    monkeypatch.setattr(batched_generation,'generate_batched',fake_generate)
    p.generate(SimpleNamespace(execute_gpu=True,phase='development'),root)
    assert read_json(root/'closed.json')['stop_cause']=='mechanical_failure'
    p.report(SimpleNamespace(output=str(output),lock='unused'),root)
    assert read_json(output)['phases']['development']['decision']['passed'] is False
    assert '"reason"' not in output.read_text() and calls==['C','P']


def test_active_and_snapshot_source_mutation_rejected(tmp_path,monkeypatch):
    root=tmp_path/'run';root.mkdir();active=tmp_path/'active.py';active.write_text('synthetic')
    relative='synthetic.py';snapshot=root/'source'/relative;snapshot.parent.mkdir();snapshot.write_text('synthetic')
    p.save(root/'config.json',{})
    p.save(root/'manifest.json',{'files':{},'sources':{relative:p.stream_hash(active)},'external_sha256':{},'packages':{}})
    monkeypatch.setattr(p,'validate_config',lambda *a:None)
    original_path=p.path;monkeypatch.setattr(p,'path',lambda value:active if value==relative else original_path(value))
    p.verify(root);active.write_text('changed')
    with pytest.raises(ValueError):p.verify(root)
    active.write_text('synthetic');snapshot.write_text('changed')
    with pytest.raises(ValueError):p.verify(root)


def test_grade_payloads_three_eights_and_authorization_no_calls():
    items,keys=p.make_pairs(rows(),predictions(),1)
    batches=p.grade_payloads(items,read_json('evaluation/persona/rubric.json'))
    assert len(batches)==3 and all(b['expected_count']==8 and len(b['expected_ids'])==8 for b in batches)
    with pytest.raises(ValueError):p.judge(SimpleNamespace(execute_judgments=False),Path('unused'))


def test_phase_identity_rejects_wrong_gpu_uuid(tmp_path,monkeypatch):
    cfg=read_json('configs/persona_prompt.json');p.save(tmp_path/'manifest.json',{})
    monkeypatch.setattr(p,'verify',lambda *a:(cfg,{}))
    p.save(tmp_path/'development/run.json',{'completed':True,'manifest_sha256':p.stream_hash(tmp_path/'manifest.json'),'gpu_uuid':'wrong'})
    with pytest.raises(ValueError):p.phase_inputs(tmp_path,'development')


@pytest.fixture
def grading_run(run,monkeypatch,tmp_path):
    root,cfg,m=run
    monkeypatch.setattr(p,'read_grades',READ_GRADES)
    (root/'development/grades/accepted.json').unlink();(root/'development/grades').rmdir()
    items,keys=p.make_pairs(rows(),predictions(),1)
    p.save(root/'development/judging/packet.json',{'items':items,'rubric_sha256':p.stream_hash(root/'inputs/rubric.json')})
    monkeypatch.setattr(p,'verify_packet',lambda *a:items)
    native=tmp_path/'native';native.write_bytes(b'synthetic executable identity; never executed')
    m['binary']=str(native);monkeypatch.setattr(p,'BINARY_SHA',p.stream_hash(native))
    monkeypatch.setattr(p.shutil,'which',lambda *a:str(native))
    monkeypatch.setattr(p.f,'validate_audit',lambda *a:{'cli_version':'2.1.323 (Claude Code)'})
    monkeypatch.setattr(p.f,'version',lambda *a:'2.1.323 (Claude Code)')
    monkeypatch.setattr(p,'registration',lambda *a:(cfg,m))
    args=SimpleNamespace(execute_judgments=True,phase='development',resume=False,lock='unused')
    def reply(payload,missing=False):
        values=ratings(payload['items'],keys)
        if missing:values.pop()
        return {'returncode':0,'canonical_stdin_sha256':p.f.canonical_hash(payload),
                'isolation':{'git_not_repository':True,'git_environment_clean':True,'returncode':128},
                'stdout':json.dumps({'modelUsage':{p.f.MODEL:{}},'result':json.dumps({'ratings':values})})}
    return root,args,reply


def test_first_valid_three_batches_exact_resume_and_payload_receipts(grading_run,monkeypatch):
    root,args,reply=grading_run;calls=[]
    monkeypatch.setattr(p.f,'invoke',lambda binary,payload:(calls.append(payload),reply(payload))[1])
    p.judge(args,root)
    assert len(calls)==3 and all(x['expected_count']==8 for x in calls)
    accepted,batches=p.read_grades(root,'development')
    assert len(accepted)==24 and all(b['first_valid_attempt']==0 for b in batches)
    args.resume=True;p.judge(args,root);assert len(calls)==3
    receipt=root/'development/grades/batch-000-attempt-0-receipt.json'
    changed=read_json(receipt);changed['canonical_stdin_sha256']='tampered';receipt.write_text(json.dumps(changed))
    with pytest.raises(ValueError):p.read_grades(root,'development')


def test_exhaustion_stops_later_batches_and_stages(grading_run,monkeypatch):
    root,args,reply=grading_run;calls=[]
    monkeypatch.setattr(p.f,'invoke',lambda binary,payload:(calls.append(payload),reply(payload,True))[1])
    with pytest.raises(p.f.InvalidSubmission):p.judge(args,root)
    assert len(calls)==2 and not (root/'development/grades/batch-001-input.json').exists()
    assert read_json(root/'closed.json')['stop_cause']=='grading_schema_exhausted'
    args.resume=True
    with pytest.raises(RuntimeError):p.judge(args,root)
    assert len(calls)==2


def test_infrastructure_failure_no_retry_or_uncertain_resume(grading_run,monkeypatch):
    root,args,_=grading_run;calls=[]
    def fail(*a):calls.append(1);raise OSError('synthetic infrastructure failure')
    monkeypatch.setattr(p.f,'invoke',fail)
    with pytest.raises(OSError):p.judge(args,root)
    assert (root/'development/grades/batch-000-attempt-0-start.json').exists() and len(calls)==1
    args.resume=True
    with pytest.raises(RuntimeError):p.judge(args,root)
    assert len(calls)==1
