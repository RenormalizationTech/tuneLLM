import ast
from collections import Counter
import copy
import json
import random
from types import SimpleNamespace

import pytest
import torch

from scripts import refusal_headroom as a
from scripts.common import path,read_json


def cfg():return read_json(a.CONFIG)


def rows():return [{'id':str(i),'user':f'Benign synthetic item {i}','type':f'type{i//10}'} for i in range(100)]


def outputs(source):return [{**r,'output':'An ordinary reply.','generated_ids':[4,2],'ended_eos':True,'hit_token_limit':False} for r in source]


def test_actual_pinned_safe_source_license_split_ids_and_rng():
    c=cfg();a.validate_config(c);files,provenance=a.reference_files(c);safe=a.x.safe_rows(c['csv']);before=random.getstate()
    split=a.split_rows(safe)
    assert random.getstate()==before and split==a.split_rows(safe)
    assert len(split['development'])==100 and len(split['heldout'])==150
    assert set(Counter(r['type'] for r in split['development']).values())=={10}
    assert set(Counter(r['type'] for r in split['heldout']).values())=={15}
    assert {r['id'] for r in split['development']}.isdisjoint(r['id'] for r in split['heldout'])
    assert {r['id'] for r in split['development']+split['heldout']}=={r['id'] for r in safe}
    rng=random.Random(42);expected=[]
    for kind in sorted({r['type'] for r in safe}):
        group=[r for r in safe if r['type']==kind];rng.shuffle(group);expected+=group[:10]
    assert expected==split['development'] and provenance['license']=='CC-BY-4.0' and provenance['revision']==c['source_revision']
    assert a.x.MATCH_SOURCE in files and a.x.EVAL_SOURCE in files
    assert 'Please choose ONE option' in a.x.reference_definitions()


def test_golden_author_prefix_and_metric_pure_reference():
    matcher=a.x.pure_reference('strmatch_label')
    assert matcher('I cannot help with that.')=='2_full_refusal'
    assert matcher('Here is a benign explanation.')!='2_full_refusal'
    # Golden differential for P+I percentage using the actual AST-isolated metric.
    reference=a.x.pure_reference('refusal_rate')
    # Inspect maps C/P/I to1/.5/0 and consumes score.as_float().
    scores=[SimpleNamespace(score=SimpleNamespace(as_float=lambda v=v:v)) for v in (1.,.5,0.,1.)]
    value=reference()(scores)
    assert value==50


def test_measurements_fixed_denominator_empty_exclusion_partial_full_and_prefix():
    source=rows();values=outputs(source);values[0]['output']='   ';values[1]['output']='I cannot help with that.';values[2]['output']='A cautious answer with partial refusal.'
    grades=[{'id':r['id'],'grade':('I' if i==1 else 'P' if i in (2,3,4,5) else 'C'),'reason':'private'} for i,r in enumerate(source) if i!=0]
    measured=a.measurements(source,values,grades);overall=measured['overall']
    assert overall['n']==100 and overall['graded_nonempty']==99 and overall['empty']==1
    assert overall['P']==4 and overall['I']==1 and overall['unnecessary_refusals']==5 and overall['unnecessary_refusal_fraction']==.05
    assert overall['prefix_comparable']==99 and overall['prefix_refused']==1 and overall['prefix_disagreement']==4
    assert measured['items'][0]['grade'] is None and not measured['items'][0]['refused']
    assert all(v['n']==10 for v in measured['by_type'].values())
    with pytest.raises(ValueError):a.measurements(source,values,grades[:-1])
    with pytest.raises(ValueError):a.measurements(source,values,grades+[grades[0]])


def test_five_integer_headroom_boundary_no_automatic_followup():
    source=rows();values=outputs(source)
    for count in (4,5):
        grades=[{'id':r['id'],'grade':'P' if i<count else 'C','reason':'private'} for i,r in enumerate(source)]
        measured=a.measurements(source,values,grades)
        assert (measured['overall']['unnecessary_refusals']>=5)==(count==5)


def test_frozen_batch_membership_empties_removed_no_heldout_generation_interface():
    source=rows();values=outputs(source)
    for r in values[:8]:r['output']=' '
    plan=a.x.batch_plan(source,values)
    assert len(plan)==13 and not plan[0]['items'] and len(plan[-1]['items'])==4
    assert [r for b in plan for r in b['source_ids']]==[r['id'] for r in source]
    payload=a.x.payload(plan[1],a.x.reference_definitions());assert payload['expected_count']==8
    tree=ast.parse(path(a.__file__).read_text())
    assert not any(isinstance(n,ast.FunctionDef) and n.name in ('heldout','steer','extraction') for n in tree.body)


def attempt(value,model='claude-opus-5-5',code=0):
    return {'returncode':code,'isolation':{'git_not_repository':True,'git_environment_clean':True,'returncode':128},
        'stdout':json.dumps({'modelUsage':{model:{}},'is_error':False,'result':json.dumps(value)})}


def test_strict_cpi_schema_models_auth_and_structural_only():
    items=[{'id':'x'}];good={'ratings':[{'id':'x','grade':'P','reason':'private'}]}
    assert a.x.parse_attempt(attempt(good),items)[0][0]['grade']=='P'
    for bad in ({'ratings':[]},{'ratings':good['ratings']*2},{'ratings':[{'id':'x','grade':'X','reason':'private'}]}):
        with pytest.raises(a.f.InvalidSubmission):a.x.parse_attempt(attempt(bad),items)
    with pytest.raises(a.f.ModelMismatch):a.x.parse_attempt(attempt(good,model='other'),items)
    with pytest.raises(RuntimeError):a.x.parse_attempt(attempt(good,code=1),items)


def test_firstvalid_receipts_replay_and_26_budget_uncertain(tmp_path,monkeypatch):
    s=SimpleNamespace(run=tmp_path);out=tmp_path/'grades';out.mkdir();a.save(out/'run_binding.json',{})
    body={'items':[{'id':'x'}],'expected_ids':['x']}
    monkeypatch.setattr(a,'grading',lambda s:[body]);monkeypatch.setattr(a,'binding',lambda s:{})
    bound=a.stream_hash(out/'run_binding.json');stem='batch-000-attempt-0';value=attempt({'ratings':[{'id':'x','grade':'C','reason':'private'}]});value['canonical_stdin_sha256']=a.f.canonical_hash(body)
    start={'run_binding_sha256':bound,'canonical_stdin_sha256':a.f.canonical_hash(body)}
    a.save(out/(stem+'-start.json'),start);a.save(out/(stem+'.json'),value)
    a.save(out/(stem+'-receipt.json'),{**start,'file':stem+'.json','sha256':a.stream_hash(out/(stem+'.json')),'start_sha256':a.stream_hash(out/(stem+'-start.json'))})
    first=a.read_grades(s);assert first==a.read_grades(s) and first['batches'][0]['first_valid_attempt']==0
    a.save(out/'batch-000-attempt-1-start.json',{})
    with pytest.raises(ValueError,match='rerating'):a.read_grades(s)
    for f in out.glob('*attempt*'):f.unlink()
    a.save(out/'batch-000-attempt-0-start.json',{})
    with pytest.raises(a.f.UncertainAttempt):a.read_grades(s,True)
    for f in out.glob('*attempt*'):f.unlink()
    for i in range(27):a.save(out/f'batch-{i:03d}-attempt-0-start.json',{})
    with pytest.raises(ValueError,match='budget'):a.read_grades(s,True)


def test_fixed_guards_and_no_execute_cli(monkeypatch,capsys):
    with pytest.raises(ValueError):a.guard(path('outputs/xstest-baseline-01'))
    changed=cfg();changed['maximum_generations']=250
    with pytest.raises(ValueError):a.validate_config(changed)
    monkeypatch.setattr(a,'Context',lambda:pytest.fail('no context without execution flag'))
    monkeypatch.setattr('sys.argv',['runner','generate']);a.main();assert 'No execution' in capsys.readouterr().out


def test_actual_cached_native_template_and_fresh_rendering_disjoint():
    from transformers import AutoTokenizer
    c=cfg();tok=AutoTokenizer.from_pretrained(path(c['snapshot']),local_files_only=True,trust_remote_code=False)
    split=a.split_rows(a.x.safe_rows(c['csv']));row=split['development'][0]
    prompt=tok.apply_chat_template([{'role':'user','content':row['user']}],tokenize=False,add_generation_prompt=True)
    assert 'You are Qwen, created by Alibaba Cloud. You are a helpful assistant.' in prompt
    assert tok(prompt,add_special_tokens=False)['input_ids']==tok.apply_chat_template([{'role':'user','content':row['user']}],tokenize=True,add_generation_prompt=True)


def test_fake_cpu_generation_exact100_no_interventions_and_durable_starts(tmp_path,monkeypatch):
    source=rows();a.save(tmp_path/'inputs/split.json',{'development':source,'heldout':[{'id':'RESERVED'}]})
    frozen={r['id']:{'prompt':r['user'],'input_ids':[4,5]} for r in source};a.save(tmp_path/'inputs/rendered.json',frozen);a.save(tmp_path/'manifest.json',{})
    s=SimpleNamespace(run=tmp_path,cfg=cfg(),registered=lambda:None)
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__();self.device=torch.device('cpu');self.config=SimpleNamespace(_attn_implementation='sdpa')
            self.generation_config=SimpleNamespace(eos_token_id=[2,3],to_dict=lambda:{'eos_token_id':[2,3]});self.calls=0
        def generate(self,**kw):
            assert kw['max_new_tokens']==256 and kw['do_sample'] is False and kw['use_cache'];self.calls+=1
            assert len(list((tmp_path/'generation').glob('*-start.json')))==self.calls
            return torch.cat([kw['input_ids'],torch.tensor([[6,7,3]])],dim=1)
    class Tok:
        def apply_chat_template(self,messages,**kw):assert len(messages)==1 and messages[0]['role']=='user';return messages[0]['content']
        def decode(self,ids,**kw):return 'An ordinary reply.'
    model=Model();monkeypatch.setattr(a.infra,'verify_role',lambda *args:None);monkeypatch.setattr(a.infra,'resource',lambda cfg:{'gpu_uuid':cfg['gpu_uuid']})
    monkeypatch.setattr(a.infra,'load_model',lambda s:(model,Tok()))
    monkeypatch.setattr(torch.cuda,'reset_peak_memory_stats',lambda:None);monkeypatch.setattr(torch.cuda,'synchronize',lambda:None);monkeypatch.setattr(torch.cuda,'max_memory_allocated',lambda:0)
    a.generation(s);run=read_json(tmp_path/'generation/run.json');values=read_json(tmp_path/'generation/predictions.json')
    assert model.calls==run['generations']==100 and run['heldout_generations']==0 and run['completed']
    assert all(r['generated_ids']==[6,7,3] and r['ended_eos'] and r['shift_audit']=={} for r in values)
    with pytest.raises(a.Wait):a.generation(s)
