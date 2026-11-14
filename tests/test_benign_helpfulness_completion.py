"""Synthetic CPU-only completion guards. No experiment grades or calls authored."""
import copy
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import benign_helpfulness_completion as c
from scripts.common import read_json


def items(count=16):
    return [{"id": f"synthetic-anon-{i}", "request": "Synthetic test request", "response": "Synthetic test output",
             "anchors": ["synthetic a", "synthetic b", "synthetic c"]} for i in range(count)]


def ratings(rows):
    return [{"id": r["id"], "refusal": "none", "anchors": [True, False, True],
             **{k: False for k in c.b.FLAGS}, "reason": "Synthetic CPU test only."} for r in rows]


def response(payload, values=None):
    return {"returncode": 0, "canonical_stdin_sha256": c.f.canonical_hash(payload),
        "isolation": {"git_not_repository": True, "git_environment_clean": True, "returncode": 128},
        "stdout": json.dumps({"modelUsage": {c.f.MODEL: {}}, "result": json.dumps({"ratings": ratings(payload['items']) if values is None else values})})}


def test_only_count_id_checklist_added_to_exact_two_eights():
    rows, rubric = items(), {"synthetic": True}
    batches = c.payloads(rows, rubric)
    assert [len(p['items']) for p in batches] == [8, 8]
    assert batches[0]['items']+batches[1]['items'] == rows
    for index,payload in enumerate(batches):
        original = c.b.judge_payloads(rows[index*8:(index+1)*8], rubric)[0]
        assert payload == {**original, 'expected_count': 8, 'expected_ids': [r['id'] for r in original['items']]}
    with pytest.raises(ValueError): c.payloads(rows[:15], rubric)


@pytest.mark.parametrize('kind', ['missing','extra','duplicate','nonboolean'])
def test_strict_schema_rejects_missing_extra_duplicate_and_wrong_types(kind):
    payload = c.payloads(items(), {})[0]
    values = ratings(payload['items'])
    if kind == 'missing': values.pop()
    elif kind == 'extra': values.append({**values[0], 'id':'synthetic-extra'})
    elif kind == 'duplicate': values[-1] = values[0]
    else: values[0]['anchors'][0] = 1
    with pytest.raises(c.f.InvalidSubmission): c.b.parse_attempt(response(payload,values),payload['items'])


def test_old20_read_only_exact_labels_and_bound_source_tampering(tmp_path,monkeypatch):
    old = tmp_path/'old';old.mkdir()
    payload = c.b.judge_payloads(items(20), {})[0]
    c.save(old/'run_binding.json',{'synthetic':True});binding=c.stream_hash(old/'run_binding.json')
    monkeypatch.setattr(c.f,'invoke',lambda binary,p:response(p))
    _,receipt=c.f.replay_or_call(old,0,0,payload,binding,'claude')
    before={p.name:p.read_bytes() for p in old.iterdir()}
    def forbidden(*a,**k): raise AssertionError('retained20 must never be regraded')
    monkeypatch.setattr(c.f,'invoke',forbidden)
    actual,_=c.read_attempt(old,0,0,payload,binding,receipt)
    assert actual==ratings(payload['items'])
    assert before=={p.name:p.read_bytes() for p in old.iterdir()}
    bound={str(p):c.stream_hash(p) for p in old.iterdir()}
    target=old/receipt['file']; changed=read_json(target);changed['canonical_stdin_sha256']='mismatch';target.write_text(json.dumps(changed))
    with pytest.raises(ValueError): c.check_hashes(bound)
    with pytest.raises(ValueError): c.read_attempt(old,0,0,payload,binding,receipt)


@pytest.fixture
def run(tmp_path,monkeypatch):
    root=tmp_path/'outputs';root.mkdir()
    old=root/'original';old.mkdir();(old/'old20.txt').write_text('Synthetic immutable old artifact')
    new=root/'completion';new.mkdir()
    actual_path=c.path
    monkeypatch.setattr(c,'path',lambda value:root if str(value)=='outputs' else old if str(value)=='outputs/benign-helpfulness-01' else actual_path(value))
    cfg={'origin':str(old),'judge_audit':str(tmp_path/'audit.json')}
    c.save(tmp_path/'audit.json',{'synthetic':True})
    manifest={'sources':{},'original_input_sha256':{str(old/'old20.txt'):c.stream_hash(old/'old20.txt')}}
    c.save(new/'manifest.json',manifest)
    c.save(new/'inputs/packet16.json',{'items':items()})
    c.save(new/'inputs/rubric.json',{'synthetic':True})
    monkeypatch.setattr(c,'registration',lambda *a:(cfg,manifest))
    monkeypatch.setattr(c,'verify',lambda *a:(cfg,manifest))
    args=SimpleNamespace(execute_judgments=True,origin=str(old),resume=False,registration='synthetic')
    return new,old,args,cfg,manifest


def test_no_old_path_writes_or_harmless_guard_side_effects(run):
    new,old,args,cfg,_=run
    c.check_destinations(new,old)
    for destination in (old,old/'nested',old.parent):
        with pytest.raises(ValueError): c.check_destinations(destination,old)
    with pytest.raises(ValueError): c.check_destinations(new,old,old/'fabricated-submission.json')
    args.execute_judgments=False
    before={p.name:p.read_bytes() for p in old.iterdir()}
    with pytest.raises(ValueError):c.judge(args,new)
    assert not (new/'grades').exists() and not (new/'terminal.json').exists()
    assert before=={p.name:p.read_bytes() for p in old.iterdir()}


def test_first_valid_freezes_two_calls_and_exact_resume_reuses(run,monkeypatch):
    new,old,args,cfg,_=run;calls=[]
    monkeypatch.setattr(c.f,'invoke',lambda binary,p:(calls.append(p['expected_ids']),response(p))[1])
    c.judge(args,new)
    assert len(calls)==2
    added,_,batches,complete=c.new_coverage(new,cfg)
    assert len(added)==16 and complete and [r['first_valid_attempt'] for r in batches]==[0,0]
    before={p.name:p.read_bytes() for p in old.iterdir()}
    args.resume=True;c.judge(args,new)
    assert len(calls)==2
    assert before=={p.name:p.read_bytes() for p in old.iterdir()}


def test_exhaustion_permanent_no_second_batch_or_extra_calls(run,monkeypatch):
    new,_,args,_,_=run;calls=[]
    monkeypatch.setattr(c.f,'invoke',lambda binary,p:(calls.append(p['expected_ids']),response(p,ratings(p['items'])[:-1]))[1])
    with pytest.raises(c.f.InvalidSubmission):c.judge(args,new)
    assert len(calls)==2 and calls[0]==calls[1]
    assert not (new/'grades/batch-001-input.json').exists()
    args.resume=True
    with pytest.raises(RuntimeError,match='permanently'):c.judge(args,new)
    assert len(calls)==2 and (new/'terminal.json').exists()


@pytest.mark.parametrize('failure',[ValueError('synthetic version mismatch'),ValueError('synthetic source/hash mismatch')])
def test_preflight_fatal_persisted_before_any_call(run,monkeypatch,failure):
    new,_,args,_,_=run;calls=[]
    def fail(*a):raise failure
    monkeypatch.setattr(c,'registration',fail)
    monkeypatch.setattr(c.f,'invoke',lambda *a:calls.append(1))
    with pytest.raises(ValueError):c.judge(args,new)
    assert read_json(new/'terminal.json')['error_type']=='ValueError' and calls==[]
    args.resume=True
    with pytest.raises(RuntimeError,match='permanently'):c.judge(args,new)
    assert calls==[]


@pytest.mark.parametrize('failure',[subprocess.TimeoutExpired('claude',3),OSError('synthetic infra'),KeyboardInterrupt()])
def test_infrastructure_or_uncertain_start_stops_without_resume_call(run,monkeypatch,failure):
    new,_,args,_,_=run;calls=[]
    def invoke(*a):calls.append(1);raise failure
    monkeypatch.setattr(c.f,'invoke',invoke)
    with pytest.raises(type(failure)):c.judge(args,new)
    assert calls==[1] and (new/'grades/batch-000-attempt-0-start.json').exists()
    assert (new/'terminal.json').exists()
    args.resume=True
    with pytest.raises(RuntimeError,match='permanently'):c.judge(args,new)
    assert calls==[1]


def test_accepted16_cannot_be_complete_after_finally_hash_failure(run,monkeypatch):
    new,_,args,cfg,manifest=run
    monkeypatch.setattr(c.f,'invoke',lambda binary,p:response(p))
    calls=[]
    def verify(*a):
        calls.append(1)
        if len(calls)==2:raise ValueError('synthetic final hash failure')
        return cfg,manifest
    monkeypatch.setattr(c,'verify',verify)
    with pytest.raises(ValueError):c.judge(args,new)
    assert (new/'grades/accepted16.json').exists() and (new/'terminal.json').exists()
    added,_,_,complete=c.new_coverage(new,cfg)
    assert len(added)==16 and not complete


def test_coordinator_direct_original_alignment_and_complete_sensitive_ids():
    rows=[{'id':str(i),'class':'sensitive' if i<24 else 'physics','user':'Synthetic request','anchors':['a','b','c']} for i in range(36)]
    predictions=[{'id':r['id'],'output':'Synthetic output'} for r in rows]
    packet=[{'id':'anon-'+r['id'],'request':r['user'],'response':'Synthetic output','anchors':r['anchors']} for r in rows]
    keys={i['id']:str(index) for index,i in enumerate(packet)}
    draft={'ai_only':True,'reviewer':'Synthetic AI','primary_grades_seen_during_refusal_reading':False,
           'ratings':[{'id':str(i),'refusal':'none','reason':'Synthetic private reason'} for i in range(24)]}
    assert len(c.validate_coordinator(draft,rows,packet,keys,predictions))==24
    bad=copy.deepcopy(draft);bad['ratings'][-1]['id']='unknown'
    with pytest.raises(ValueError):c.validate_coordinator(bad,rows,packet,keys,predictions)
    bad_packet=copy.deepcopy(packet);bad_packet[0]['response']='tampered'
    with pytest.raises(ValueError):c.validate_coordinator(draft,rows,bad_packet,keys,predictions)


def test_public_privacy_rejects_private_notes_and_paths():
    for data in ({'reason':'private'},{'items':[{'response':'private'}]},{'path':'/home/test/private'},{'generated_ids':[1,2]}):
        with pytest.raises(ValueError):c.public_privacy(data)
