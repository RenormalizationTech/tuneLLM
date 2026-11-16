"""CPU initial-stage checks; native invocation is mocked, no target model loads."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import political_refusal as b
from scripts import political_refusal_common as c
from scripts.common import read_json


def fixture_data():
    return b.compile_fixture(read_json(b.CONFIG),require_accepted=False)


def save_json(file,value):
    file.parent.mkdir(parents=True,exist_ok=True);file.write_text(json.dumps(value,ensure_ascii=False)+'\n')


def test_actual_reviewed_metadata_compiles_exact_counts_and_all_branch_plans():
    cfg=read_json(b.CONFIG);b.validate_config(cfg);data=fixture_data();plans=b.all_plans(data,cfg)
    assert [len(data[k]) for k in ('fit','development','heldout','calibration')]==[32,48,48,16]
    choices=read_json(cfg['group_metadata'])
    assert choices['evaluation_replacements']=={'ROUTE-V2-C038':'ROUTE-V2-C036'}
    replacement=[r for r in data['development'] if r['canonical_id']=='ROUTE-V2-C036']
    assert len(replacement)==2 and all(r['event_group']=='turkish_press_after_2016' for r in replacement)
    assert not any(r['canonical_id']=='ROUTE-V2-C038' for r in data['development'])
    assert {r['event_group'] for r in data['calibration']}=={m['event_group'] for m in choices['calibration_groups'].values()}
    assert len(plans['headroom']['primary'])==12
    assert [len(plans['fit'][k]) for k in ('primary','coherence')]==[8,8]
    assert len(plans['development']['primary']['gcad+fixed+projection'])==42
    assert len(plans['heldout']['primary']['gcad+fixed+projection'])==48
    assert len(plans['heldout']['primary'])==len(plans['development']['primary'])==7
    assert plans==b.all_plans(data,cfg)
    for metric in ('primary','coherence'):
        for batch in plans['calibration'][metric]:assert len(batch)==len({x['canonical_id'] for x in batch})==8


def test_native_payload_reference_policy_never_reaches_coherence_or_generation(tmp_path):
    cfg=read_json(b.CONFIG);data=fixture_data();plans=b.all_plans(data,cfg)
    s=SimpleNamespace(run=tmp_path,cfg=cfg)
    save_json(tmp_path/'inputs/data.json',data);save_json(tmp_path/'inputs/plans.json',plans)
    save_json(tmp_path/'inputs/primary_rubric.json',b.PRIMARY_RUBRIC);save_json(tmp_path/'inputs/coherence_rubric.json',{'rule':'coherence only'})
    primary=b.bodies(s,'calibration','primary');coherence=b.bodies(s,'calibration','coherence')
    assert all(x['reference_rules']==data['reference_rules'] for x in primary)
    for payload,batch in zip(primary,plans['calibration']['primary']):
        byid={r['id']:r for r in data['calibration']}
        for item,entry in zip(payload['items'],batch):
            row=byid[entry['physical_id']]
            assert item['optional_reference_metadata']['notes_optional_details']==row['notes_optional_details']
            assert item['optional_reference_metadata']['annotation_flags']==row['annotation_flags']
            assert item['optional_reference_metadata']['not_extra_required_coverage']
            assert item['facts'][0]['reference_metadata']=={k:v for k,v in row['anchors'][0].items() if k not in ('local_id','paraphrase_en')}
    assert all('reference_rules' not in x and all(set(i)=={'id','question','answer'} for i in x['items']) for x in coherence)
    assert 'expected_judge' not in json.dumps(primary) and 'case_type' not in json.dumps(primary)
    class Tokenizer:
        def apply_chat_template(self,messages,**kwargs):return json.dumps(messages,ensure_ascii=False)
    row=copy.deepcopy(data['development'][0]);row['notes_optional_details']=['OPTIONAL_NOTE_SENTINEL'];row['annotation_flags']=['FLAG_SENTINEL'];row['anchors']=[{'local_id':'F1','paraphrase_en':'ANCHOR_SENTINEL'}]
    rendered=b.render(Tokenizer(),row,'PPLUS_TEST')
    assert all(x not in rendered for x in ('OPTIONAL_NOTE_SENTINEL','FLAG_SENTINEL','ANCHOR_SENTINEL'))
    assert json.loads(rendered)==[{'role':'system','content':'PPLUS_TEST'},{'role':'user','content':row['user']}]


def test_actual_authored_calibration_mechanics_and_exact_expected_schema_offline(monkeypatch):
    cfg=read_json(b.CONFIG);data=fixture_data()
    monkeypatch.setenv('HF_HUB_OFFLINE','1');monkeypatch.setenv('TRANSFORMERS_OFFLINE','1')
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(Path(cfg['snapshot']),local_files_only=True,trust_remote_code=False)
    observed=[];items=[];expected={}
    for row in data['calibration']:
        ids=tok(row['answer'],add_special_tokens=False)['input_ids']
        actual=c.mechanical_validity(row['answer'],ids,eos_token_ids=[])
        assert actual['invalid_mechanical']==data['calibration_expected_mechanical'][row['id']]['invalid']
        observed.append(actual['invalid_mechanical']);items.append({'id':row['id'],'fact_ids':row['fact_ids']})
        expected[row['id']]=data['calibration_expected'][row['id']]
    ratings=[{'id':i,**r,'reason':'UNIT_TEST_EXPECTED_AUTHOR_LABEL','framing':'nonblocking malformed metadata'} for i,r in expected.items()]
    assert c.calibration_check(items,{'ratings':ratings},expected)['all_pass']
    assert set(s for r in expected.values() for s in r['facts'].values())=={'present','absent','contradicted'}
    assert any(observed)


def test_preparation_requires_exact_review_acceptance_without_writing_run(tmp_path):
    cfg=read_json(b.CONFIG);cfg['data_acceptance']=str(tmp_path/'missing.json')
    with pytest.raises(FileNotFoundError):b.compile_fixture(cfg)
    save_json(Path(cfg['data_acceptance']),{'status':'accepted'})
    with pytest.raises(ValueError,match='reviewed data'):b.compile_fixture(cfg)


@pytest.fixture
def journal(tmp_path,monkeypatch):
    s=SimpleNamespace(run=tmp_path,cfg={'framing_labels':None})
    body={'task':'primary','items':[{'id':f'opaque{i}','question':'UNIT_TEST_QUESTION','answer':'UNIT_TEST_ANSWER','facts':[{'id':'F1'},{'id':'F2'}]} for i in range(8)]}
    monkeypatch.setattr(b,'bodies',lambda *args:[body])
    monkeypatch.setattr(b,'binding',lambda *args:{'payloads_sha256':b.f.canonical_hash([body]),'maximum':32})
    calls=[]
    def attempt(binary,payload):
        calls.append(payload)
        ratings=[{'id':x['id'],'class':'S','unscorable':False,'facts':{'F1':'present','F2':'absent'},'reason':'UNIT_TEST_ONLY'} for x in payload['items']]
        return {'returncode':0,'stdout':json.dumps({'subtype':'success','modelUsage':{b.f.MODEL:{}},'result':json.dumps({'ratings':ratings})}),
            'canonical_stdin_sha256':b.f.canonical_hash(payload),'isolation':{'git_not_repository':True,'git_environment_clean':True,'returncode':128}}
    monkeypatch.setattr(b.f,'invoke',attempt)
    return s,body,calls,attempt


def test_actual_foundation_first_valid_receipts_and_replay_without_new_call(journal):
    s,body,calls,_=journal
    first=b.run_journal(s,'headroom','primary','MOCK_NO_EXECUTABLE',False)
    second=b.run_journal(s,'headroom','primary','MOCK_NO_EXECUTABLE',True)
    assert first==second and len(calls)==b.native_count(s)==1
    assert first['batches'][0]['first_valid_attempt']==0
    assert not list((s.run/'native').glob('**/*attempt-1*'))
    receipt=s.run/'native/headroom/primary/batch-000-attempt-0-receipt.json';value=read_json(receipt);value['canonical_stdin_sha256']='0'*64;receipt.write_text(json.dumps(value))
    with pytest.raises(ValueError,match='receipt'):b.read_grades(s,'headroom','primary')


def test_one_schema_retry_and_no_third_attempt(journal,monkeypatch):
    s,body,calls,original=journal
    def malformed(binary,payload):
        result=original(binary,payload)
        if len(calls)==1:
            outer=json.loads(result['stdout']);outer['result']=json.dumps({'ratings':[]});result['stdout']=json.dumps(outer)
        return result
    monkeypatch.setattr(b.f,'invoke',malformed)
    result=b.run_journal(s,'headroom','primary','MOCK',False)
    assert len(calls)==2 and result['batches'][0]['first_valid_attempt']==1
    assert b.read_grades(s,'headroom','primary')==result
    assert not list((s.run/'native').glob('**/*attempt-2*'))


def test_uncertain_start_no_call_on_resume_and_initial_budget_before_start(journal,monkeypatch):
    s,body,calls,_=journal
    def interrupted(*args):calls.append('interrupted');raise OSError('UNIT_TEST_INFRASTRUCTURE')
    monkeypatch.setattr(b.f,'invoke',interrupted)
    with pytest.raises(OSError):b.run_journal(s,'headroom','primary','MOCK',False)
    with pytest.raises(b.f.UncertainAttempt):b.run_journal(s,'headroom','primary','MOCK',True)
    assert len(calls)==b.native_count(s)==1
    other=SimpleNamespace(run=s.run/'other',cfg=s.cfg)
    for i in range(32):save_json(other.run/'native/charged/primary'/f'batch-{i:03d}-attempt-0-start.json',{'charged':True})
    with pytest.raises(ValueError,match='ceiling'):b.run_journal(other,'headroom','primary','MOCK',False)
    assert b.native_count(other)==32 and len(calls)==1
    assert not (other.run/'native/headroom/primary/batch-000-attempt-0-start.json').exists()


def test_dry_flags_and_missing_resume_do_not_call_or_create_output(monkeypatch,tmp_path):
    monkeypatch.setattr(b,'Context',lambda:pytest.fail('dry command instantiated runtime'))
    for stage in ('headroom','calibration','judge'):
        monkeypatch.setattr('sys.argv',['political_refusal',stage]);b.main()
    s=SimpleNamespace(run=tmp_path,cfg={})
    with pytest.raises(b.Wait):b.run_journal(s,'headroom','primary','MOCK',True)
    assert not (tmp_path/'native').exists()


def test_authorized_preflight_failure_terminal_marker_without_native_calls(tmp_path,monkeypatch):
    s=SimpleNamespace(run=tmp_path)
    s.verify=lambda **kw:(_ for _ in ()).throw(ValueError('UNIT_TEST_CHANGED_SOURCE'))
    monkeypatch.setattr(b,'Context',lambda:s)
    monkeypatch.setattr(b.f,'invoke',lambda *args:pytest.fail('no remote call permitted'))
    monkeypatch.setattr('sys.argv',['political_refusal','calibration','--execute-native'])
    with pytest.raises(ValueError):b.main()
    assert read_json(tmp_path/'stopped.json')['error_type']=='ValueError'


def test_initial_report_calibration_null_is_public_safe_no_model_work(tmp_path):
    cfg=read_json(b.CONFIG);s=SimpleNamespace(run=tmp_path,cfg=cfg,m={'sources':{},'combined_budget':c.budget_counts()})
    save_json(tmp_path/'manifest.json',{'test':True});save_json(tmp_path/'stopped.json',{'stop_cause':'calibration_failed'})
    result=b.report(s)
    assert result['status']=='calibration_null' and result['native_starts']==0 and result['fresh_generations']==0
    b.native.public_safe(result)
