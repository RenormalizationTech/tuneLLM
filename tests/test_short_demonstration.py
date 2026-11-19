"""CPU-only recorded-demo provenance, deterministic selection and safe fallback."""
import copy
import json
from pathlib import Path

import pytest
from scripts import build_short_demonstration as demo


@pytest.fixture(scope='module')
def data():return json.loads((demo.ROOT/'docs/assets/short_demonstration.json').read_text())


PRIVATE_REGENERATION_ARTIFACTS=(
    'outputs/qwen38-qasper-final-analysis-01.json',
    'outputs/qwen38-qasper-final-runs.json',
    'outputs/qwen38-science-01/prepared-qasper-tokenizer-01/test.jsonl',
    'outputs/qwen38-qasper-final-base-complete-01/predictions.jsonl',
    'outputs/qwen38-qasper-final-schema-01/predictions.jsonl',
    'outputs/qwen38-qasper-final-adapter-01/predictions.jsonl',
    'data/qasper-v03-01/manifest.json',
    'data/qasper-v03-01/original/qasper-test-v0.3.json',
    'outputs/qwen38-sarcasm-followup-01/generation/result.json',
    'outputs/qwen38-sarcasm-followup-01/judge/trait.json',
    'outputs/qwen38-sarcasm-followup-01/judge/coherence.json',
    'outputs/qwen38-sarcasm-followup-01/judge/impolite.json',
)


def test_recorded_source_anchors_and_expected_selections(data):
    assert data['mode']=='RECORDED' and data['new_model_or_quality_calls']==0
    assert data['qasper']['record_id']=='1603.03876::47ffc9811b037613c9c4d1ec1e4f13c08396ed1c'
    assert data['qasper']['selection']['eligible_records']==112
    assert data['sarcasm']['question_id']=='F01'
    assert [r['strength'] for r in data['sarcasm']['strengths']]==[0,2,4,8,16]
    assert data['sarcasm']['provenance']['canonical_result_sha256']==demo.CANONICAL_SARC
    assert data['qasper']['attribution']['license']=='CC-BY-4.0'
    for record in data['provenance']['public_sources'].values():
        assert record['url'].startswith('https://github.com/RenormalizationTech/tuneLLM/blob/')
        assert len(record['sha256'])==64


def test_saved_scores_copied_not_regraded(data):
    q=data['qasper'];s=data['sarcasm']
    assert q['arms'][0]['recorded_answer_f1']==pytest.approx(.19047619047619047)
    assert q['arms'][2]['recorded_answer_f1']==1
    assert all(r['recorded_evidence_f1']==1 for r in q['arms'])
    b8=next(r for r in s['strengths'] if r['strength']==8)
    assert b8['scores']=={'sarcasm':85,'coherence':58,'impoliteness':55}
    assert b8['aggregate']['coherence']==58.9375
    assert not b8['aggregate']['gates']['all_pass']
    assert all(r['generated_tokens']==160 and r['cap_hit'] for r in s['strengths'])


def test_selection_is_id_and_readability_not_score_value():
    def row(key):return {'id':key,'primary':True,'evidence_strict_primary':True,'user':'Context\n\nQuestion: Which corpus?',
        'references':[{'answer':'PDTB','evidence':['p1'],'unanswerable':False}],
        'paragraphs':[{'id':'p1','text':'The corpus is PDTB.'}]}
    inputs=[row('b'),row('a')]
    predictions={arm:{key:{'output':json.dumps({'answer':'PDTB','unanswerable':False,'evidence':['p1']})} for key in ['a','b']} for arm,_ in demo.ARMS}
    scores={arm:{'a':{'valid':True,'answer_f1':0},'b':{'valid':True,'answer_f1':1}} for arm,_ in demo.ARMS}
    selected,count=demo.select_qasper(inputs,predictions,scores)
    assert selected[0]['id']=='a' and count==2
    for values in scores.values():values['a']['answer_f1']=1;values['b']['answer_f1']=0
    assert demo.select_qasper(list(reversed(inputs)),predictions,scores)[0][0]['id']=='a'
    inputs[1]['references'][0]['answer']='x'*201
    assert demo.select_qasper(inputs,predictions,scores)[0][0]['id']=='b'


def test_public_payload_no_private_runtime_or_judge_fields(data):
    demo.privacy_check(data)
    text=json.dumps(data,ensure_ascii=False)
    for forbidden in ['outputs/','/home/','input_ids','generated_ids','session_id','cost_usd','"argv"']:
        assert forbidden not in text
    for injected in [{'private':'/home/example/artifacts/file'},{'session_id':'private'},
                     {'generated_ids':[1,2]},{'nested':{'argv':['secret']}}]:
        with pytest.raises(ValueError):demo.privacy_check(injected)


def test_fallback_is_complete_and_generated_text_escaped(data):
    page=demo.render_page(data)
    assert page.count('<h4>Recorded strength ')==5
    assert all(demo.esc(r['output']) in page for r in data['sarcasm']['strengths'])
    assert 'data-demo-fallback' in page and '<noscript>' in page
    assert 'factual endorsements' not in page
    assert 'all three answers identify pdtb' in page.lower()
    assert 'No tested strength passed every registered acceptance gate.' in page
    assert 'Too little sarcasm (mean gain and answer count)' in page
    assert 'Coherence loss exceeds limit' in page
    assert 'individual ge50' not in page and 'Fail: trait gain' not in page
    assert '[all scores and measured limitations](qwen38_sarcasm_results.md)' in page
    assert data['releases']['sarcasm']['release_manifest_sha256'] in page
    malicious=copy.deepcopy(data);malicious['sarcasm']['strengths'][0]['output']='<img src=x onerror=alert(1)>'
    rendered=demo.render_page(malicious)
    assert '<img src=x onerror=alert(1)>' not in rendered
    assert '&lt;img src=x onerror=alert(1)&gt;' in rendered
    drift=copy.deepcopy(data);drift['qasper']['record_id']='different-record'
    with pytest.raises(ValueError,match='PDTB narrative'):demo.render_page(drift)
    drift=copy.deepcopy(data);drift['sarcasm']['strengths'][0]['cap_hit']=False
    with pytest.raises(ValueError,match='160-token narrative'):demo.render_page(drift)


def test_owned_writes_are_idempotent_and_refuse_drift(tmp_path):
    file=tmp_path/'recorded.json';demo.write_owned(file,'frozen\n',False)
    demo.write_owned(file,'frozen\n',False)
    with pytest.raises(FileExistsError):demo.write_owned(file,'different\n',False)
    assert file.read_text()=='frozen\n'


@pytest.mark.artifact(*PRIVATE_REGENERATION_ARTIFACTS)
def test_private_regeneration_equals_frozen_public_data(data):
    assert demo.canonical(data)==demo.canonical(demo.build_data())


def test_browser_viewer_is_local_static_text_only():
    js=(demo.ROOT/'docs/javascripts/short_demonstration.js').read_text()
    assert '.textContent = r.output' in js and '.innerHTML' not in js
    assert "url.origin !== window.location.origin" in js
    assert js.count('fetch(')==1
    assert 'WebSocket' not in js and 'POST' not in js
    assert "fallback.open = true" in js
