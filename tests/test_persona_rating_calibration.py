"""CPU only; fake judge envelopes/journals, never an API or GPU call."""
import copy
from fractions import Fraction
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import persona_rating_calibration as c


def data():return c.read_json('evaluation/persona_calibration/texts.json')
def rubric():return c.read_json('evaluation/persona_calibration/rubric.json')
def config():return c.read_json('configs/persona_rating_calibration.json')
def ratings(items):return [{'id':r['id'],**{t:3*r['intended'][t] for t in c.TRAITS},'gross_failure':False} for r in items]


def test_real_fixture_balance_opaque_ids_blind_payloads_and_two_fixed_plans():
    items=c.validate_fixture(data());plans=c.make_plans(items,[421,422]);c.validate_plans(items,plans)
    assert len(plans)==2 and plans[0]!=plans[1]
    for plan in plans:
        assert len(plan['batches'])==6
        assert len({i for b in plan['batches'] for i in b['ids']})==48
        assert all(len(b['ids'])==8 and sum(b['cell_counts'].values())==8 for b in plan['batches'])
        for batch in plan['batches']:
            body=c.payload(batch,items,rubric())
            assert set(body)=={'rubric','untrusted_data_policy','items','expected_ids','expected_count','return_schema'}
            assert all(set(r)=={'id','context','text'} for r in body['items'])
            assert body['expected_ids']==batch['ids'] and body['expected_count']==8
    assert c.lexical_summary(items,config()['lexicon'])['agreements']==48


@pytest.mark.parametrize('text,present,count',[
    ('assess class hello scrap scrapbook',False,0),('Scunthorpe',False,0),
    ('DAMN! Hell, that was damn good.',True,3),('a classy assessment',False,0),
    ('a damn-good result',True,1),('fucking brilliant',True,1)])
def test_whole_word_case_and_supported_variants(text,present,count):
    result=c.lexical(text,config()['lexicon']);assert (result['present'],result['count'])==(present,count)


@pytest.mark.parametrize('key,value',[('pass_seeds',[421,422,423]),('pass_seeds',[422,421]),
    ('maximum_total_calls',25),('maximum_attempts_per_batch',3),('batch_size',16),
    ('minimum_intended_agreements',40),('minimum_lexical_agreements',45),('minimum_quadratic_kappa','0.69'),
    ('judge_model','other'),('lexicon',['damn','damn']),('lexicon',['']),('lexicon',['two words']),
    ('texts','other/path')])
def test_fixed_config_rejects_drift(key,value):
    cfg=config();c.validate_config(cfg);cfg[key]=value
    with pytest.raises(ValueError):c.validate_config(cfg)


def test_regenerated_plan_and_fixture_types_reject_tampering():
    items=c.validate_fixture(data());plans=c.make_plans(items,[421,422]);plans[0]['batches'][0]['cell_counts']['000']+=1
    with pytest.raises(ValueError):c.validate_plans(items,plans)
    d=data();d['items'][0]['intended']['profanity']=True
    with pytest.raises(ValueError):c.validate_fixture(d)
    d=data();d['items'][0]['id']=d['items'][1]['id']
    with pytest.raises(ValueError):c.validate_fixture(d)


@pytest.mark.parametrize('kind',['bool_score','float','string','range','negative','bool_flag','missing','duplicate','extra_id','extra_field'])
def test_strict_rating_schema(kind):
    items=data()['items'][:8];values=ratings(items);ids=[r['id'] for r in items]
    if kind=='bool_score':values[0]['profanity']=True
    elif kind=='float':values[0]['sarcasm']=2.0
    elif kind=='string':values[0]['rudeness']='2'
    elif kind=='range':values[0]['rudeness']=4
    elif kind=='negative':values[0]['sarcasm']=-1
    elif kind=='bool_flag':values[0]['gross_failure']=0
    elif kind=='missing':values.pop()
    elif kind=='duplicate':values[-1]=values[0]
    elif kind=='extra_id':values[-1]['id']='unrequested'
    else:values[0]['explanation']='unrequested'
    with pytest.raises(ValueError):c.validate_ratings(ids,values)


def test_quadratic_kappa_known_weighted_result_degeneracy_and_alignment():
    assert c.kappa_fraction([0,1,2,3],[0,1,2,2])==Fraction(7,8)
    assert c.kappa_fraction([0,0,3,3],[0,0,3,3])==1
    assert c.kappa_fraction([0]*48,[0]*48) is None
    with pytest.raises(ValueError):c.kappa_fraction([True],[1])
    with pytest.raises(ValueError):c.kappa_fraction([0,1],[0])


def test_exact_per_axis_41_boundary_and_gross_failure():
    items=data()['items'];values=ratings(items)
    for r in values[:7]:r['rudeness']=0 if r['rudeness']>=2 else 3
    result=c.decision(items,[values,copy.deepcopy(values)],config()['lexicon'])
    assert result['outcome']=='pass' and result['passes'][0]['intended_agreements']['rudeness']==41
    values[7]['rudeness']=0 if values[7]['rudeness']>=2 else 3
    assert c.decision(items,[values,values],config()['lexicon'])['outcome']=='fail'
    values=ratings(items);values[0]['gross_failure']=True
    assert c.decision(items,[values,values],config()['lexicon'])['outcome']=='fail'


def test_exact_lexical_46_and_kappa_point7_boundaries(monkeypatch):
    items=copy.deepcopy(data()['items']);positive=[r for r in items if r['intended']['profanity']]
    for r in positive[:2]:r['text']='Neutral synthetic sentence.'
    values=ratings(items)
    monkeypatch.setattr(c,'kappa_fraction',lambda a,b:Fraction(7,10))
    assert c.decision(items,[values,values],config()['lexicon'])['outcome']=='pass'
    monkeypatch.setattr(c,'kappa_fraction',lambda a,b:Fraction(6999,10000))
    assert c.decision(items,[values,values],config()['lexicon'])['outcome']=='fail'
    positive[2]['text']='Neutral synthetic sentence.'
    assert c.decision(items,[values,None],config()['lexicon'])['gates']['lexical'] is False
    monkeypatch.setattr(c,'kappa_fraction',lambda a,b:None)
    assert c.decision(data()['items'],[ratings(data()['items'])]*2,config()['lexicon'])['outcome']=='fail'


def setup(tmp_path,monkeypatch):
    run=tmp_path/'run';run.mkdir();cfg=config();items=data()['items'];plans=c.make_plans(items,[421,422]);binary=tmp_path/'claude';binary.write_text('fake, never executed')
    m={'binary':str(binary),'sources':{},'cli_version':'2.1.323 (Claude Code)','audit_sha256':'synthetic'}
    c.save(run/'manifest.json',{'sources':{}});c.save(run/'inputs/texts.json',data());c.save(run/'inputs/rubric.json',rubric());c.save(run/'inputs/plans.json',plans)
    monkeypatch.setattr(c,'guard',lambda r:None);monkeypatch.setattr(c,'verify',lambda r:(cfg,m));monkeypatch.setattr(c,'registration',lambda a,r:(cfg,m))
    monkeypatch.setattr(c,'BINARY_SHA',c.stream_hash(binary));monkeypatch.setattr(c.shutil,'which',lambda name:str(binary))
    monkeypatch.setattr(c.f,'version',lambda name:m['cli_version']);monkeypatch.setattr(c.f,'validate_audit',lambda p:{'cli_version':m['cli_version']})
    return run,items,plans,m


def envelope(body,values,model=None):
    return {'returncode':0,'canonical_stdin_sha256':c.f.canonical_hash(body),
        'isolation':{'git_not_repository':True,'git_environment_clean':True,'returncode':128},
        'stdout':json.dumps({'subtype':'success','modelUsage':{model or c.f.MODEL:{}},'result':json.dumps({'ratings':values})})}


def install_calls(monkeypatch,items,behavior='valid'):
    by_id={r['id']:r for r in ratings(items)};calls=[];seen={}
    def invoke(binary,body):
        calls.append(copy.deepcopy(body));key=c.f.canonical_hash(body);seen[key]=seen.get(key,0)+1
        values=[by_id[i] for i in body['expected_ids']]
        if behavior=='retry' and seen[key]==1:values=values[:-1]
        elif behavior=='exhaust':values=values[:-1]
        elif behavior=='axis_fail':values=[{**r,**{t:0 for t in c.TRAITS}} for r in values]
        elif behavior=='uncertain':raise OSError('synthetic uncertain transport')
        result=envelope(body,values,'unexpected-model' if behavior=='model' else None)
        if behavior=='missing_model':
            outer=json.loads(result['stdout']);outer.pop('modelUsage');result['stdout']=json.dumps(outer)
        return result
    monkeypatch.setattr(c.f,'invoke',invoke);return calls


def test_first_valid_resume_report_privacy_and_cell_counts(tmp_path,monkeypatch):
    run,items,plans,m=setup(tmp_path,monkeypatch);calls=install_calls(monkeypatch,items)
    c.judge(SimpleNamespace(execute_judgments=True,resume=False),run)
    assert len(calls)==12
    c.judge(SimpleNamespace(execute_judgments=True,resume=True),run);assert len(calls)==12
    out=tmp_path/'public.json';c.report(SimpleNamespace(output=str(out)),run);result=c.read_json(out)
    assert result['outcome']=='pass' and result['status']=='complete' and result['calls_started']==12
    assert result['passes'][0]['per_cell_presence']['100']=={'profanity':6,'rudeness':0,'sarcasm':0}
    assert result['per_batch_cell_composition']==[[b['cell_counts'] for b in p['batches']] for p in plans]
    assert all(r['text'] not in out.read_text() and r['id'] not in out.read_text() for r in items)


def test_one_structural_retry_per_batch_never_exceeds24_or_reframes(tmp_path,monkeypatch):
    run,items,_,_=setup(tmp_path,monkeypatch);calls=install_calls(monkeypatch,items,'retry')
    c.judge(SimpleNamespace(execute_judgments=True,resume=False),run)
    assert len(calls)==24 and all(calls[i]==calls[i+1] for i in range(0,24,2))
    c.judge(SimpleNamespace(execute_judgments=True,resume=True),run);assert len(calls)==24


@pytest.mark.parametrize('behavior,call_count',[('exhaust',2),('uncertain',1),('model',1),('missing_model',1)])
def test_fatal_or_exhausted_stop_never_resumes(tmp_path,monkeypatch,behavior,call_count):
    run,items,_,_=setup(tmp_path,monkeypatch);calls=install_calls(monkeypatch,items,behavior)
    with pytest.raises((c.f.InvalidSubmission,c.f.ModelMismatch,RuntimeError,OSError)):
        c.judge(SimpleNamespace(execute_judgments=True,resume=False),run)
    assert len(calls)==call_count and (run/'stopped.json').exists()
    with pytest.raises(RuntimeError):c.judge(SimpleNamespace(execute_judgments=True,resume=True),run)
    assert len(calls)==call_count


def test_first_pass_axis_failure_stops_and_incomplete_is_not_a_pass(tmp_path,monkeypatch):
    run,items,_,_=setup(tmp_path,monkeypatch);calls=install_calls(monkeypatch,items,'axis_fail')
    c.judge(SimpleNamespace(execute_judgments=True,resume=False),run)
    assert len(calls)==6 and not (run/'grades/pass-2').exists()
    out=tmp_path/'public.json';c.report(SimpleNamespace(output=str(out)),run)
    assert c.read_json(out)['status']=='incomplete' and c.read_json(out)['outcome']=='fail'


@pytest.mark.parametrize('artifact',['post_valid','out_of_range'])
def test_stray_attempts_rejected_by_read_and_resume(tmp_path,monkeypatch,artifact):
    run,items,plans,m=setup(tmp_path,monkeypatch);calls=install_calls(monkeypatch,items)
    c.judge(SimpleNamespace(execute_judgments=True,resume=False),run)
    name='batch-000-attempt-1-start.json' if artifact=='post_valid' else 'batch-006-attempt-0-start.json'
    c.save(run/'grades/pass-1'/name,{'synthetic':True})
    with pytest.raises(ValueError):c.read_pass(run,0,m,items,rubric(),plans)
    with pytest.raises(ValueError):c.report(SimpleNamespace(output=str(tmp_path/'public.json')),run)
    with pytest.raises(ValueError):c.judge(SimpleNamespace(execute_judgments=True,resume=True),run)
    assert len(calls)==12


def test_read_only_receipt_tamper_and_no_execution_side_effects(tmp_path,monkeypatch):
    with pytest.raises(ValueError):c.judge(SimpleNamespace(execute_judgments=False),tmp_path)
    assert not list(tmp_path.iterdir())
    run,items,plans,m=setup(tmp_path,monkeypatch);install_calls(monkeypatch,items)
    c.judge(SimpleNamespace(execute_judgments=True,resume=False),run)
    p=run/'grades/pass-1/batch-000-attempt-0-receipt.json';v=c.read_json(p);v['canonical_stdin_sha256']='changed';p.write_text(json.dumps(v))
    with pytest.raises(ValueError):c.read_pass(run,0,m,items,rubric(),plans)


def test_prepare_then_verify_preserves_full_fixture_metadata(tmp_path,monkeypatch):
    original_cfg=config();original_data=data();original_rubric=rubric()
    resolve=lambda value:Path(value) if Path(value).is_absolute() else tmp_path/value
    def put(name,value):
        p=resolve(name);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value) if not isinstance(value,str) else value)
    for name,value in [(original_cfg['texts'],original_data),(original_cfg['rubric'],original_rubric),('configs/persona_rating_calibration.json',original_cfg),
        (original_cfg['protocol'],'protocol fixture'),(original_cfg['design_review'],'design review fixture'),(original_cfg['fixture_review'],'fixture review fixture'),
        ('outputs/qwen3-14b-baseline-01/manifest.json',{'external_sha256':{},'sources':{}}),('results/qwen3_14b_baseline_01.json',{'status':'complete'}),('scripts/stub.py','fixture source')]:put(name,value)
    binary=tmp_path/'native';binary.write_text('fake native')
    put('runtime.json',{'binary':str(binary),'binary_sha256':c.stream_hash(binary)})
    put('audit.json',{'synthetic':True});put(original_cfg['judge_runtime_config'],{'judge_runtime_record':str(tmp_path/'runtime.json'),'judge_audit':str(tmp_path/'audit.json')})
    monkeypatch.setattr(c,'path',resolve);monkeypatch.setattr(c,'read_json',lambda value:json.loads(resolve(value).read_text()))
    monkeypatch.setattr(c,'SOURCES',('scripts/stub.py',original_cfg['texts'],original_cfg['rubric'],'configs/persona_rating_calibration.json'));monkeypatch.setattr(c,'BINARY_SHA',c.stream_hash(binary))
    monkeypatch.setattr(c.f,'validate_audit',lambda p:{'cli_version':'synthetic'})
    c.prepare(SimpleNamespace(config='configs/persona_rating_calibration.json'));run=resolve(c.RUN)
    c.verify(run)
    assert c.read_json(run/'inputs/texts.json')==original_data
    fixture=resolve(original_cfg['texts']);fixture.write_text(fixture.read_text()+' ')
    with pytest.raises(ValueError):c.verify(run)
