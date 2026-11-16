"""Execution-only recovery CPU checks; no models, GPU, API or real prepare."""
import ast
import copy
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import political_refusal_recovery as r


def write(p,value):
    p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value,ensure_ascii=False)+'\n')


def context_at(root):
    return SimpleNamespace(run=root,cfg=r.read_json(r.CONFIG),m={'parent_charged_starts':1})


def test_science_and_aliases_are_frozen_parent_not_new_implementations():
    cfg=r.read_json(r.CONFIG);r.validate_config(cfg)
    assert r.science_equal(cfg,r.read_json(r.parent.CONFIG))
    for name in ('PRIMARY_RUBRIC','all_plans','render','compile_fixture','bodies','parse','predictions','generate_headroom'):
        assert getattr(r,name) is getattr(r.parent,name)
    # These copies need local read_grades/cumulative accounting, but their
    # scientific bodies are AST-identical to the frozen parent.
    def funcs(module):
        return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(Path(module.__file__).read_text()).body if isinstance(n,ast.FunctionDef)}
    a,b=funcs(r),funcs(r.parent)
    for name in ('calibration_result','headroom_scores','headroom_result','lock','judge_stream'):
        assert a[name]==b[name],name


@pytest.mark.parametrize('field,value',[('seed',43),('max_new_tokens',256),('plan_seeds',{}),('thresholds',{}),('systems',{}),('annotation','other.json')])
def test_only_operational_config_changes_allowed(field,value):
    cfg=r.read_json(r.CONFIG);cfg[field]=value
    assert not r.science_equal(cfg,r.read_json(r.parent.CONFIG))
    with pytest.raises(ValueError):r.validate_config(cfg)


def test_actual_parent_closed_content_free_and_all_bytes_unchanged():
    cfg=r.read_json(r.CONFIG);root=r.path(r.PARENT);before=r.file_inventory(root)
    r.parent.Context().verify(models=False)
    record=r.verify_parent(cfg)
    assert record['verified_counts_and_payload']['native_starts']==1
    assert record['verified_counts_and_payload']['first_valid_batches']==0
    assert record['verified_counts_and_payload']['qwen_generations']==0
    assert before==r.file_inventory(root)
    r.native.public_safe(record)
    assert not r.path(r.RUN).exists()


@pytest.mark.parametrize('change',['changed','added','removed','accepted','headroom'])
def test_parent_any_change_fails_closed(tmp_path,change):
    root=tmp_path/'parent';shutil.copytree(r.path(r.PARENT),root)
    cfg=copy.deepcopy(r.read_json(r.CONFIG));cfg['recovery']['parent_run']=str(root)
    record=r.read_json(r.PARENT_PROVENANCE)
    if change=='changed':write(root/'stopped.json',{'changed':True})
    elif change=='added':write(root/'new.json',{})
    elif change=='removed':(root/'stopped.json').unlink()
    elif change=='accepted':write(root/'native/calibration/primary/accepted.json',{})
    else:(root/'headroom').mkdir()
    with pytest.raises(ValueError):r.verify_parent(cfg,record)


@pytest.mark.parametrize('change',['labels','model','success','network_other'])
def test_parent_failure_must_have_no_submission(change):
    a=r.read_json(r.path(r.PARENT)/'native/calibration/primary/batch-000-attempt-0.json');outer=json.loads(a['stdout'])
    if change=='labels':outer['ratings']=[]
    elif change=='model':outer['modelUsage']={'claude-opus-5-5':{}}
    elif change=='success':a['returncode']=0
    else:outer['result']='some other infrastructure failure'
    a['stdout']=json.dumps(outer)
    with pytest.raises(ValueError):r.content_free_failure(a)


def copy_science(root):
    for name in r.SCIENCE_FILES:
        dest=root/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(r.path(r.PARENT)/name,dest)
    shutil.copyfile(r.path(r.PARENT)/'config.json',root/'inputs/parent_config.json')


def test_exact_snapshot_payloads_aliases_and_all_scientific_branches(tmp_path):
    before=r.file_inventory(r.path(r.PARENT));copy_science(tmp_path);s=context_at(tmp_path)
    r.verify_scientific_snapshots(s)
    data=r.read_json(tmp_path/'inputs/data.json');plans=r.read_json(tmp_path/'inputs/plans.json')
    assert plans==r.all_plans(data,s.cfg)
    assert [len(data[k]) for k in ('fit','development','heldout','calibration')]==[32,48,48,16]
    assert [len(plans['calibration'][k]) for k in ('primary','coherence')]==[2,2]
    assert len(plans['headroom']['primary'])==12
    assert len(plans['development']['primary']['gcad+fixed+projection'])==42
    assert len(plans['heldout']['primary'])==7
    assert before==r.file_inventory(r.path(r.PARENT))
    # Semantic equality is insufficient: even a whitespace byte change fails.
    p=tmp_path/'inputs/data.json';p.write_bytes(p.read_bytes()+b'\n')
    with pytest.raises(ValueError,match='snapshot differs'):r.verify_scientific_snapshots(s)


def test_actual_cached_tokenizer_reconstructs_source_data_and_all_renderings(monkeypatch):
    monkeypatch.setenv('HF_HUB_OFFLINE','1');monkeypatch.setenv('TRANSFORMERS_OFFLINE','1')
    from transformers import AutoTokenizer
    cfg=r.read_json(r.CONFIG);root=r.path(r.PARENT)
    tok=AutoTokenizer.from_pretrained(r.path(cfg['snapshot']),local_files_only=True,trust_remote_code=False)
    r.verify_recompiled_inputs(cfg,r.read_json(root/'inputs/data.json'),r.read_json(root/'inputs/rendered.json'),tok)


@pytest.fixture
def journal(tmp_path,monkeypatch):
    s=context_at(tmp_path)
    ps=SimpleNamespace(run=r.path(r.PARENT),cfg=r.read_json(r.parent.CONFIG))
    body=r.parent.bodies(ps,'calibration','primary')[0]
    parent_attempt=r.read_json(ps.run/'native/calibration/primary/batch-000-attempt-0.json')
    for k in list(r.os.environ):
        if k.startswith('GIT_'):monkeypatch.delenv(k)
    for k in parent_attempt['isolation']['removed_git_variable_names']:monkeypatch.setenv(k,'UNIT_TEST_VALUE_NOT_EXPORTED')
    monkeypatch.setattr(r,'bodies',lambda *args:[body])
    calls=[]
    def invoke(binary,payload):
        calls.append(payload)
        ratings=[{'id':x['id'],'class':'S','unscorable':False,'facts':{v['id']:'present' for v in x['facts']},'reason':'UNIT_TEST_ONLY'} for x in payload['items']]
        return {'returncode':0,'stdout':json.dumps({'modelUsage':{r.f.MODEL:{}},'result':json.dumps({'ratings':ratings})}),
            'canonical_stdin_sha256':r.f.canonical_hash(payload),'isolation':copy.deepcopy(parent_attempt['isolation'])}
    write(tmp_path/'manifest.json',{'test':True})
    monkeypatch.setattr(r.f,'invoke',invoke)
    return s,calls,invoke


def test_first_valid_replay_preserves_parent_charge_and_does_not_rerate(journal):
    s,calls,_=journal
    first=r.run_journal(s,'calibration','primary','MOCK',False)
    assert first==r.run_journal(s,'calibration','primary','MOCK',True)
    assert len(calls)==1 and r.native_count(s)==2
    assert first['batches'][0]['first_valid_attempt']==0
    assert not list(s.run.rglob('*attempt-1*'))
    comparison=r.first_attempt_comparison(s)
    assert comparison['isolation_equal'] and comparison['canonical_stdin_sha256_equal']


def test_first_launch_git_name_mismatch_waits_before_any_start_or_call(journal,monkeypatch):
    s,calls,_=journal
    monkeypatch.setenv('GIT_UNIT_TEST_EXTRA','NOT_EXPORTED')
    with pytest.raises(r.Wait,match='no call or start'):r.run_journal(s,'calibration','primary','MOCK',False)
    assert not calls and r.native_count(s)==1
    assert not (s.run/'native').exists() and not (s.run/'stopped.json').exists()


@pytest.mark.parametrize('field',['isolation','canonical_stdin_sha256'])
def test_real_journal_first_attempt_mismatch_fails_before_retry(journal,monkeypatch,field):
    s,calls,invoke=journal
    def mismatch(binary,body):
        a=invoke(binary,body)
        if field=='isolation':a['isolation']['removed_git_variable_names']=[]
        else:a[field]='0'*64
        return a
    monkeypatch.setattr(r.f,'invoke',mismatch)
    if field=='isolation':
        with pytest.raises(ValueError,match='isolation/payload differs'):r.run_journal(s,'calibration','primary','MOCK',False)
    else:
        # The unchanged foundation rejects bad stdin hashes even before the
        # new comparison, preserving an uncertain start rather than a receipt.
        with pytest.raises(RuntimeError,match='invocation stdin hash differs'):r.run_journal(s,'calibration','primary','MOCK',False)
    assert len(calls)==1 and not list(s.run.rglob('*attempt-1*'))
    if field=='isolation':
        comparison=r.first_attempt_comparison(s,read_json_attempt(s),require_equal=False)
        assert comparison['isolation_equal'] is False
    else:assert r.first_attempt_comparison(s)['status']=='started_no_completed_attempt'


@pytest.mark.parametrize('field',['isolation','canonical_stdin_sha256'])
def test_direct_parent_comparison_rejects_both_record_mismatches(tmp_path,field):
    attempt=r.read_json(r.path(r.PARENT)/'native/calibration/primary/batch-000-attempt-0.json')
    if field=='isolation':attempt[field]['removed_git_variable_names']=[]
    else:attempt[field]='0'*64
    s=context_at(tmp_path)
    with pytest.raises(ValueError,match='isolation/payload differs'):r.first_attempt_comparison(s,attempt)
    actual=r.first_attempt_comparison(s,attempt,require_equal=False)
    assert actual[field+'_equal'] is False


def read_json_attempt(s):
    return r.read_json(s.run/'native/calibration/primary/batch-000-attempt-0.json')


def test_parent_charge_prevents_32nd_new_start_before_call(journal):
    s,calls,_=journal
    for i in range(31):write(s.run/'native/previous/primary'/f'batch-{i:03d}-attempt-0-start.json',{})
    assert r.native_count(s)==32
    with pytest.raises(ValueError,match='BEFORE'):r.run_journal(s,'calibration','primary','MOCK',False)
    assert not calls and not list((s.run/'native/calibration').rglob('*-start.json'))


def test_one_schema_retry_and_never_third_attempt(journal,monkeypatch):
    s,calls,invoke=journal
    def malformed(binary,body):
        a=invoke(binary,body)
        if len(calls)==1:
            outer=json.loads(a['stdout']);outer['result']='{"ratings":[]}';a['stdout']=json.dumps(outer)
        return a
    monkeypatch.setattr(r.f,'invoke',malformed)
    result=r.run_journal(s,'calibration','primary','MOCK',False)
    assert len(calls)==2 and r.native_count(s)==3
    assert result['batches'][0]['first_valid_attempt']==1
    assert not list(s.run.rglob('*attempt-2*'))


def test_infrastructure_not_structural_retry_and_uncertain_resume_no_call(journal,monkeypatch):
    s,calls,_=journal
    def uncertain(*args):calls.append('started');raise OSError('UNIT_TEST_INFRA')
    monkeypatch.setattr(r.f,'invoke',uncertain)
    with pytest.raises(OSError):r.run_journal(s,'calibration','primary','MOCK',False)
    with pytest.raises(r.f.UncertainAttempt):r.run_journal(s,'calibration','primary','MOCK',True)
    assert len(calls)==1 and r.native_count(s)==2


def test_terminal_preflight_and_no_resume_calls(tmp_path,monkeypatch):
    s=context_at(tmp_path)
    s.verify=lambda **kw:(_ for _ in ()).throw(ValueError('UNIT_TEST_CHANGED_PARENT'))
    monkeypatch.setattr(r,'Context',lambda:s);monkeypatch.setattr(r.f,'invoke',lambda *a:pytest.fail('no invocation'))
    monkeypatch.setattr('sys.argv',['recovery','calibration','--execute-native'])
    with pytest.raises(ValueError):r.main()
    assert r.read_json(tmp_path/'stopped.json')['stop_cause']=='fatal_execution_or_integrity'
    s.verify=lambda **kw:None;s.registered=lambda:None
    r.main()  # Existing terminal marker is checked before any call.


def test_report_and_registration_disclose_shared_charge_without_private_text(tmp_path):
    s=context_at(tmp_path);cfg=s.cfg
    write(tmp_path/'manifest.json',{});write(tmp_path/'stopped.json',{'stop_cause':'fatal_execution_or_integrity'})
    cfg['initial_registration']=str(tmp_path/'reg.json');write(Path(cfg['initial_registration']),{})
    s.m.update({'sources':{'scripts/tone_judge.py':r.stream_hash(r.path('scripts/tone_judge.py'))},
        'combined_budget':r.c.budget_counts(),'parent_provenance_sha256':'a'*64,'dns_preflight_sha256':r.stream_hash(r.path(r.DNS_PREFLIGHT)),
        'unused_preflight_archive':r.unused_preflight_archive(),
        'binary_sha256':r.infra.BINARY_SHA})
    result=r.report(s)
    assert result['status']=='incomplete' and result['native_starts']==1 and result['first_valid_batches']==0
    assert result['recovery']['network_permission_change']['outer_os_sandbox_relaxed'] is True
    assert 'tools_files_credentials_scope_change' not in json.dumps(result)
    permission=result['recovery']['network_permission_change']
    assert 'native_argv_unchanged' not in permission and 'native_environment_unchanged' not in permission
    assert 'argv/env not recorded in receipts' in permission['native_argv_env_basis']
    assert result['recovery']['first_attempt_comparison']['status']=='not_attempted'
    assert r.registration_expected(s)['dns_preflight_sha256']==s.m['dns_preflight_sha256']
    assert result['recovery']['unused_preflight_archive']['native_starts']==0
    assert r.registration_expected(s)['unused_preflight_manifest_sha256']==s.m['unused_preflight_archive']['manifest_sha256']
    assert r.registration_expected(s)['parent_charged_starts']==1
    assert r.registration_expected(s)['global_cumulative_maximum_starts']==424
    r.native.public_safe(result)


def test_no_flag_no_context_no_parent_or_recovery_writes(monkeypatch):
    before=r.file_inventory(r.path(r.PARENT))
    monkeypatch.setattr(r,'Context',lambda:pytest.fail('dry flag must be harmless'))
    for phase in ('calibration','judge','headroom'):
        monkeypatch.setattr('sys.argv',['recovery',phase]);r.main()
    with pytest.raises(ValueError):r.guard(r.path(r.PARENT))
    assert before==r.file_inventory(r.path(r.PARENT))
    assert not r.path(r.RUN).exists()
