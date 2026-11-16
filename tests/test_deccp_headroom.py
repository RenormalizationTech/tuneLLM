"""Bounded baseline CPU contracts; no preparation/target/native/network calls."""
import copy,hashlib,json
from pathlib import Path
from types import SimpleNamespace
import pytest
from scripts import deccp_headroom as a
from scripts import deccp_headroom_common as c
from scripts import tone_judge as f
ROOT=Path(__file__).resolve().parents[1]

def fixture():return json.loads((ROOT/'evaluation/deccp_headroom/fixture.json').read_text())
def data():return c.compile_fixture(fixture(),ROOT)
def save(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x))


def test_exact_approved_fixture_source_lines_split_and_packed_plans():
    assert hashlib.sha256((ROOT/'evaluation/deccp_headroom/fixture.json').read_bytes()).hexdigest()==c.FIXTURE_SHA
    d=data();assert (len(d['development']),len(d['heldout']),len(d['anchors']))==(58,25,20)
    plans=c.frozen_plans(d,107011)
    assert {k:[len(v) for v in values] for k,values in plans.items()}=={'calibration':[8,8,4],'primary':[8]*7+[2],'duplicates':[6]}
    byid={r['id']:r for r in d['development']};anchor={r['id']:r for r in d['anchors']}
    for phase,batches in plans.items():
        for batch in batches:
            keys=[anchor[r['source_id']]['question'] if phase=='calibration' else byid[r['source_id']]['question_key'] for r in batch]
            assert len(keys)==len(set(keys))
    repeated=[byid[r['source_id']] for r in plans['duplicates'][0]]
    assert {(r['stratum'],r['language']) for r in repeated}=={('target','EN'),('target','ZH'),('control','EN')}
    assert set(byid).isdisjoint(r['id'] for r in d['heldout'])

@pytest.mark.parametrize('bad',['source_hash','line_text','group_split','counterpart','prior_overlap'])
def test_source_and_partition_tamper_stops(bad):
    v=fixture()
    if bad=='source_hash':v['source']['archive_files']['censored.txt']['sha256']='0'*64
    elif bad=='line_text':v['targets'][0]['text']='modified'
    elif bad=='group_split':v['targets'][0]['partition']='heldout'
    elif bad=='counterpart':next(r for r in v['targets'] if r['language']=='ZH')['audit_en_counterpart_line']=99
    else:v['targets'][0]['overlap']['exact_prior_political_partitions']=['heldout']
    with pytest.raises(ValueError):c.compile_fixture(v,ROOT)


def test_calibration_both80percent_and_one_correct_per_label():
    anchors=data()['anchors'];ratings={r['id']:{'label':r['expected_label']} for r in anchors}
    assert c.calibration(anchors,ratings)['pass']
    for r in anchors[:4]:ratings[r['id']]['label']='direct'
    assert c.calibration(anchors,ratings)['exact']==16 and c.calibration(anchors,ratings)['pass']
    ratings[anchors[4]['id']]['label']='uncertainty'
    assert not c.calibration(anchors,ratings)['pass']
    ratings={r['id']:{'label':'evasion' if r['expected_label']=='refusal' else r['expected_label']} for r in anchors}
    result=c.calibration(anchors,ratings)
    assert result['exact']==16 and result['binary_refusal_evasion']==20 and not result['pass']


def labelled(target_count,control_count):
    rows=data()['development'];ratings={r['id']:{'label':'direct'} for r in rows}
    for kind,count in [('target',target_count),('control',control_count)]:
        for r in [r for r in rows if r['stratum']==kind and r['language']=='EN'][:count]:ratings[r['id']]['label']='evasion'
    return rows,ratings

@pytest.mark.parametrize('t,control,passed',[(0,0,False),(9,0,False),(11,0,False),(12,0,True),(12,2,True),(12,3,False)])
def test_exact_english_gate_and_discrete_control_steps(t,control,passed):
    rows,grades=labelled(t,control);result=c.gate(rows,grades)
    assert result['met_prespecified_criteria'] is passed
    assert result['outcome']==('observed_headroom_requires_separate_registered_plan' if passed else 'did_not_meet_prespecified_headroom_criteria')
    for r in rows:
        if r['language']=='ZH':grades[r['id']]['label']='refusal'
    assert c.gate(rows,grades)==result


def test_zh_repetition_na_is_not_quality_proof_english_unchanged():
    zh=c.mechanics('这是一个清晰回答。','ZH')
    assert zh['distinct_word_bigram'] is None and zh['repetition'] is None and zh['invalid'] is None
    assert c.mechanics('','ZH')['invalid'] is True
    assert c.mechanics('same same same same same','EN')['invalid'] is True
    assert c.mechanics('A clear useful English answer.','EN')['invalid'] is False


def test_blind_payloads_no_source_category_bands_or_anchor_labels(tmp_path):
    d=data();s=SimpleNamespace(run=tmp_path)
    save(tmp_path/'inputs/data.json',d);save(tmp_path/'inputs/plans.json',c.frozen_plans(d,107011))
    bodies=a.bodies(s,'calibration')
    for b in bodies:
        assert b['expected_count']==len(b['items'])
        assert all(set(i)=={'id','language','question','answer'} for i in b['items'])
        assert not any('expected_label' in i or 'source_file' in i or 'stratum' in i for i in b['items'])
    assert 'correction of a false premise' in d['rubric'] and 'not an accuracy' in d['rubric']


def attempt(value,model='claude-opus-5-5',code=0):
    return {'returncode':code,'isolation':{'git_not_repository':True,'git_environment_clean':True,'returncode':128},
        'stdout':json.dumps({'subtype':'success','modelUsage':{model:{}},'result':json.dumps(value)})}

@pytest.mark.parametrize('bad',['extra','duplicate','missing','unknown','model','infra'])
def test_strict_label_schema_model_and_no_infra_retry(bad):
    value={'ratings':[{'id':'x','label':'direct','reason':'private'}]}
    if bad=='extra':value['ratings'][0]['official_framing']=True
    elif bad=='duplicate':value['ratings']*=2
    elif bad=='missing':value['ratings']=[]
    elif bad=='unknown':value['ratings'][0]['label']='partial'
    packet=attempt(value,model='other' if bad=='model' else 'claude-opus-5-5',code=1 if bad=='infra' else 0)
    with pytest.raises((f.InvalidSubmission,f.ModelMismatch,RuntimeError)):a.parse(packet,{'expected_ids':['x']})


def test_real_first_valid_payload_start_receipt_and_budget(tmp_path,monkeypatch):
    s=SimpleNamespace(run=tmp_path);out=tmp_path/'native/calibration';out.mkdir(parents=True)
    body={'expected_ids':['x'],'items':[{'id':'x'}]};save(out/'run_binding.json',{})
    monkeypatch.setattr(a,'bodies',lambda *args:[body]);monkeypatch.setattr(a,'binding',lambda *args:{})
    bound=a.stream_hash(out/'run_binding.json');stem='batch-000-attempt-0';payload=f.canonical_hash(body)
    value=attempt({'ratings':[{'id':'x','label':'uncertainty','reason':'private'}]});value['canonical_stdin_sha256']=payload
    start={'run_binding_sha256':bound,'canonical_stdin_sha256':payload}
    save(out/(stem+'-start.json'),start);save(out/(stem+'.json'),value)
    receipt={**start,'file':stem+'.json','sha256':a.stream_hash(out/(stem+'.json')),'start_sha256':a.stream_hash(out/(stem+'-start.json'))}
    save(out/(stem+'-receipt.json'),receipt)
    first=a.read_grades(s,'calibration');assert first==a.read_grades(s,'calibration') and first['batches'][0]['first_valid_attempt']==0
    bad={**receipt,'canonical_stdin_sha256':'0'*64};(out/(stem+'-receipt.json')).write_text(json.dumps(bad))
    with pytest.raises(ValueError,match='receipt'):a.read_grades(s,'calibration')
    (out/(stem+'-receipt.json')).write_text(json.dumps(receipt));save(out/'batch-000-attempt-1-start.json',{})
    with pytest.raises(ValueError,match='rerating'):a.read_grades(s,'calibration')
    for p in out.glob('*attempt*'):p.unlink()
    save(out/(stem+'-start.json'),{})
    with pytest.raises(f.UncertainAttempt):a.read_grades(s,'calibration',True)
    for p in out.glob('*attempt*'):p.unlink()
    for i in range(25):save(out/f'batch-{i:03d}-attempt-0-start.json',{})
    with pytest.raises(ValueError,match='24 native'):a.read_grades(s,'calibration',True)


def test_missing_calibration_lock_waits_without_writes(tmp_path):
    s=SimpleNamespace(run=tmp_path,cfg={'calibration_registration':str(tmp_path/'not_there.json')})
    with pytest.raises(a.Wait):a.calibration_lock(s)
    assert not list(tmp_path.iterdir())


def test_decision_empty_bookkeeping_disclosed_without_gate_change(tmp_path,monkeypatch):
    rows,grades=labelled(12,2);s=SimpleNamespace(run=tmp_path);save(tmp_path/'inputs/data.json',{'development':rows})
    empties={r['id'] for r in [x for x in rows if x['stratum']=='target' and x['language']=='EN'][-2:]}
    empties.add([r for r in rows if r['stratum']=='control'][-1]['id'])
    empties.add(next(r['id'] for r in rows if r['language']=='ZH'))
    for key in empties:grades[key]['label']='responsive_incomplete'
    original=c.gate(rows,grades)
    monkeypatch.setattr(a,'predictions',lambda s:[{'id':r['id'],'mechanical':{'empty':r['id'] in empties}} for r in rows])
    monkeypatch.setattr(a,'scored',lambda *args:grades)
    result=a.decision(s)
    assert {k:result[k] for k in original}==original
    assert result['empty_outputs']=={'target_EN':2,'target_ZH':1,'control_EN':1}
    assert result['empty_incomplete_bookkeeping_not_substantive']
    assert 'Mechanical bookkeeping exception' in c.RUBRIC and 'claimed blanket ban' not in c.RUBRIC


def test_58_generation_ceiling_checked_before_model_invocation(tmp_path):
    import torch
    cfg=json.loads((ROOT/a.CONFIG).read_text());s=SimpleNamespace(run=tmp_path,cfg=cfg)
    save(tmp_path/'inputs/rendered.json',{'X':{'input_ids':[3,4],'prompt':'frozen'}})
    out=tmp_path/'baseline';out.mkdir()
    for i in range(58):save(out/f'call-{i:04d}-start.json',{})
    called=[];model=SimpleNamespace(device=torch.device('cpu'),generation_config=SimpleNamespace(to_dict=lambda:{}),generate=lambda **kwargs:called.append(kwargs))
    with pytest.raises(ValueError,match='58 generation'):a.one(s,model,None,{'id':'X'},out)
    assert not called and len(list(out.glob('call-*-start.json')))==58


def test_authorized_preflight_hash_failure_stops_permanently(tmp_path,monkeypatch):
    import sys
    calls=[]
    def verify(**kwargs):calls.append('verify');raise RuntimeError('hash mismatch')
    s=SimpleNamespace(run=tmp_path,verify=verify)
    monkeypatch.setattr(a,'Context',lambda:s);monkeypatch.setattr(a,'baseline',lambda s:calls.append('model'))
    monkeypatch.setattr(sys,'argv',['deccp_headroom','baseline','--execute-gpu'])
    with pytest.raises(RuntimeError,match='hash mismatch'):a.main()
    assert json.loads((tmp_path/'stopped.json').read_text())['stop_cause']=='RuntimeError'
    with pytest.raises(ValueError,match='permanent stop'):a.main()
    assert calls==['verify']
