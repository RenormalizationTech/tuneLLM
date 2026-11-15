import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from scripts import author_profanity_revision as a
from scripts import author_profanity_revision_common as p
from scripts import author_profanity_method as original
from scripts import author_profanity_method_common as algorithm
from scripts.common import path,read_json


def answers():
    return {'triples':[{'id':f'T{i:02d}','present':f'The damn box has item number {i}.','absent':f'The box has item number {i}.'} for i in range(1,13)]}


def reused():
    return {'users':[{'id':f'T{i:02d}','user':f'Describe item {i}.'} for i in range(1,13)],
            'calibration':[{'id':f'C{i:02d}','user':f'Calibration item {i}.'} for i in range(1,33)],
            'report':[{'id':f'H{i:02d}','user':f'Report item {i}.'} for i in range(1,33)]}


def test_actual_parent_exact_reuse_and_unchanged_input_bytes():
    cfg=read_json(a.CONFIG);data,binding,files=a.reused_bank(cfg)
    before={name:a.stream_hash(path(name)) for name in files}
    assert len(data['users'])==12 and len(data['calibration'])==len(data['report'])==32
    for row in data['users']:assert binding['user_utf8_sha256'][row['id']]==hashlib.sha256(row['user'].encode()).hexdigest()
    for name in ('calibration','report'):
        old=path(cfg['parent_fixture_directory'])/(name+'.json')
        assert a.json_bytes(data[name])==old.read_bytes() and binding[name+'_file_sha256']==a.stream_hash(old)
    assert a.reused_bank(cfg)[0]==data
    assert all(a.stream_hash(path(name))==h for name,h in before.items())


def test_answer_only_request_and_deterministic_id_merge():
    old=reused();packet=p.artifact_packet({'profanity':'swearing, curse words, crude language','toxicity':'rude, insulting, condescending, or hostile'},old)
    assert packet['shared_requests']==old['users'] and set(packet['return_schema']['triples'][0])=={'id','present','absent'}
    assert 'calibration' not in packet and 'report' not in packet and 'Calibration item' not in json.dumps(packet) and 'Report item' not in json.dumps(packet)
    assert 'Keep content, warmth and helpfulness the same apart from swearing.' in packet['task']
    assert 'intensity/affect' not in packet['task'] and 'will be disclosed' not in packet['task']
    before=json.dumps(old,ensure_ascii=False);raw=answers();merged=p.merge(raw,old)
    assert [r['user'] for r in merged['triples']]==[r['user'] for r in old['users']]
    assert merged['calibration'] is old['calibration'] and merged['report'] is old['report']
    assert json.dumps(old,ensure_ascii=False)==before and all(set(r)=={'id','present','absent'} for r in raw['triples'])


@pytest.mark.parametrize('mutation',['user','new_question','reordered','missing','extra','blank'])
def test_generated_users_questions_and_schema_drift_rejected(mutation):
    raw=answers()
    if mutation=='user':raw['triples'][0]['user']='paraphrased'
    elif mutation=='new_question':raw['calibration']=[]
    elif mutation=='reordered':raw['triples'].reverse()
    elif mutation=='missing':raw['triples'].pop()
    elif mutation=='extra':raw['triples'][0]['notes']='private extra'
    else:raw['triples'][0]['present']=' '
    with pytest.raises(ValueError):p.artifact_shape(raw)


def test_same_algorithm_objects_folds_and_secondary_plan():
    names=('response_encoding','fold_plans','auc','cv','decodability','parse_self_judge','fill_condition','repetition','text_checks','selection','heldout_effects','raw_shift','artifact_mechanical','promoted')
    assert all(getattr(p,name) is getattr(algorithm,name) for name in names)
    assert p.fold_plans()==algorithm.fold_plans() and a.secondary_plan()==original.secondary_plan()
    # Every method-stage body is identical; only controlled factory/paths/report
    # and initial snapshot/context bindings differ in this separate copy.
    old={n.name:n for n in ast.parse(path(original.__file__).read_text()).body if isinstance(n,ast.FunctionDef)}
    new={n.name:n for n in ast.parse(path(a.__file__).read_text()).body if isinstance(n,ast.FunctionDef)}
    changed={'validate_config','reused_bank','prepare','accepted_artifact','report','main','native','read_native'}
    for name in old.keys()-changed:
        assert ast.dump(old[name],include_attributes=False)==ast.dump(new[name],include_attributes=False),name


def test_same_mechanical_rules_no_brittle_sentence_count_or_word_quota():
    raw=answers();merged=p.merge(raw,reused());assert p.artifact_mechanical(merged)['passed']
    raw['triples'][0]['present']='Damn. One sentence. Another sentence.'
    assert p.artifact_mechanical(p.merge(raw,reused()))['passed']  # Semantic reviewer, no sentence-count gate.
    raw['triples'][0]['present']=raw['triples'][0]['absent']
    assert not p.artifact_mechanical(p.merge(raw,reused()))['passed']


def attempt(value):
    return {'returncode':0,'isolation':{'git_not_repository':True,'git_environment_clean':True,'returncode':128},
            'stdout':json.dumps({'result':json.dumps(value),'modelUsage':{'claude-opus-5-5':{}},'is_error':False})}


def test_native_firstvalid_raw_stays_answer_only_merge_after_acceptance():
    value,models=a.native_parse(attempt(answers()),{},'artifact')
    assert value==answers() and models==['claude-opus-5-5'] and set(value)=={'triples'}
    value['triples'][0]['user']='regenerated'
    with pytest.raises(a.StructuralError):a.native_parse(attempt(value),{},'artifact')


def test_source_scope_and_old_path_refusal():
    cfg=read_json(a.CONFIG);a.validate_config(cfg)
    assert cfg['run']!=read_json(original.CONFIG)['run'] and cfg['maximum_target_generations']==394 and cfg['maximum_local_judge_generations']==408
    with pytest.raises(ValueError):a.guard(path(original.RUN))
    cfg['run']=original.RUN
    with pytest.raises(ValueError):a.validate_config(cfg)


def test_reused_files_tamper_fails_before_candidate_or_calls(tmp_path,monkeypatch):
    # Real original bank is read once; substitute only one read to emulate a
    # mismatched old question hash without writing any parent artifact.
    cfg=read_json(a.CONFIG);genuine=a.stream_hash
    def changed(file):
        if str(file).endswith('evaluation/author_profanity_method/calibration.json'):return '0'*64
        return genuine(file)
    monkeypatch.setattr(a,'stream_hash',changed)
    with pytest.raises(ValueError,match='dataset bytes changed'):a.reused_bank(cfg)
    assert not list(tmp_path.iterdir())


def test_rejected_semantic_review_closes_revision_no_promoted_files(tmp_path,monkeypatch):
    cfg=read_json(a.CONFIG);s=SimpleNamespace(run=tmp_path,cfg=cfg)
    a.save(tmp_path/'promotion_preview.json',{'datasets_sha256':{}})
    real_read=a.read_json
    monkeypatch.setattr(a,'accepted_artifact',lambda s:p.merge(answers(),reused()))
    monkeypatch.setattr(a,'read_json',lambda file:{'accepted':False} if str(file)==cfg['semantic_review'] else real_read(file))
    a.promote(s)
    assert real_read(tmp_path/'stopped.json')['stop_cause']=='artifact_null'
    assert not (tmp_path/'promotion.json').exists()


def native_flow_fixture(tmp_path,monkeypatch,invoke):
    cfg=read_json(a.CONFIG);runtime=tmp_path/'runtime.json';a.save(runtime,{'judge_audit':'unused_mock_audit'})
    cfg['judge_runtime_config']=str(runtime);s=SimpleNamespace(run=tmp_path,cfg=cfg,m={'cli_version':'2.1.323 (Claude Code)'},registered=lambda:None)
    a.save(tmp_path/'manifest.json',{});binary=tmp_path/'mock-binary';binary.write_bytes(b'not executable')
    genuine=a.stream_hash
    monkeypatch.setattr(a,'stream_hash',lambda file:a.infrastructure.BINARY_SHA if Path(file)==binary else genuine(file))
    monkeypatch.setattr(a.infrastructure,'verify_role',lambda *args:None)
    monkeypatch.setattr(a.shutil,'which',lambda name:str(binary))
    monkeypatch.setattr(a.f,'version',lambda name:'2.1.323 (Claude Code)')
    monkeypatch.setattr(a.f,'validate_audit',lambda file:{})
    monkeypatch.setattr(a.f,'invoke',invoke)
    return s


@pytest.mark.parametrize('reply',['I cannot provide that response.','{"triples":',json.dumps({'triples':[]}),json.dumps({'refusal':True})])
def test_real_native_artifact_flow_one_attempt_closes_invalid_or_refusal(tmp_path,monkeypatch,reply):
    calls=[]
    def invoke(binary,payload):
        calls.append(payload)
        return {'returncode':0,'isolation':{'git_not_repository':True,'git_environment_clean':True,'returncode':128},
                'canonical_stdin_sha256':a.f.canonical_hash(payload),
                'stdout':json.dumps({'result':reply,'modelUsage':{'claude-opus-5-5':{}},'is_error':False})}
    s=native_flow_fixture(tmp_path,monkeypatch,invoke)
    a.save(tmp_path/'inputs/artifact_packet.json',{'task':'mock frozen request'})
    assert a.native(s,'artifact') is None and len(calls)==1
    directory=tmp_path/'native/artifact'
    assert read_json(tmp_path/'stopped.json')['stop_cause']=='artifact_null'
    assert (directory/'batch-000-attempt-0.json').is_file() and (directory/'batch-000-attempt-0-receipt.json').is_file()
    assert not list(directory.glob('*attempt-1*')) and not (directory/'accepted.json').exists()
    payload=calls[0];bound=a.stream_hash(directory/'run_binding.json')
    a.infrastructure.checked_attempt(directory,0,0,payload,bound)
    with pytest.raises(ValueError,match='permanent stop'):a.native(s,'artifact',resume=True)
    assert len(calls)==1
    a.save(directory/'batch-000-attempt-1-start.json',{})
    with pytest.raises(ValueError,match='stray'):a.read_native(s,'artifact',True)


def test_secondary_native_flow_retains_two_identical_structural_attempts(tmp_path,monkeypatch):
    calls=[];payload={'expected_ids':['opaque'],'task':'mock secondary'}
    def invoke(binary,body):
        calls.append(body)
        value={'ratings':[]} if len(calls)==1 else {'ratings':[{'id':'opaque','profanity':2,'toxicity':0,'semantic_coherence':8,'reason':'private'}]}
        return {'returncode':0,'isolation':{'git_not_repository':True,'git_environment_clean':True,'returncode':128},
                'canonical_stdin_sha256':a.f.canonical_hash(body),
                'stdout':json.dumps({'result':json.dumps(value),'modelUsage':{'claude-opus-5-5':{}},'is_error':False})}
    s=native_flow_fixture(tmp_path,monkeypatch,invoke)
    monkeypatch.setattr(a,'selection_lock',lambda s:None);monkeypatch.setattr(a,'stage',lambda *args:None)
    monkeypatch.setattr(a,'native_bodies',lambda s,kind:[payload])
    result=a.native(s,'secondary')
    assert len(calls)==2 and calls[0]==calls[1] and result['batches'][0]['first_valid_attempt']==1
    assert not (tmp_path/'stopped.json').exists()


def test_public_revision_report_states_one_shot_effective_and_registered_budgets(tmp_path):
    cfg=read_json(a.CONFIG);s=SimpleNamespace(run=tmp_path,cfg=cfg,m={'sources':{}})
    a.save(tmp_path/'manifest.json',{});a.save(tmp_path/'inputs/parent_binding.json',{'parent_status':'extraction_null'})
    result=a.report(s);provenance=result['revision_provenance']
    assert {key:provenance[key] for key in ('artifact_max_starts','secondary_max_starts','effective_native_max_starts','registered_native_ceiling')}=={
        'artifact_max_starts':1,'secondary_max_starts':40,'effective_native_max_starts':41,'registered_native_ceiling':42}
    assert result['status']=='incomplete'
