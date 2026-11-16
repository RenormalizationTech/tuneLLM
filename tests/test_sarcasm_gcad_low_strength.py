import copy
import json
from types import SimpleNamespace

import pytest
import torch

from scripts import sarcasm_gcad_low_strength as a
from scripts.common import path,read_json


@pytest.fixture(autouse=True)
def one_thread():
    previous=torch.get_num_threads();torch.set_num_threads(1)
    try:yield
    finally:torch.set_num_threads(previous)


def cfg():return read_json(a.CONFIG)


def test_real_closed_parent_reuse_and_no_historical_changes():
    c=cfg();root=path(c['parent_run']);names=[root/'selection.json',root/'stopped.json',root/'development/predictions.json',root/'extraction/banks.pt',path(c['parent_result'])]
    before={str(p):a.stream_hash(p) for p in names}
    p,reused,origin,external=a.parent_scope(c)
    assert origin['parent_status']=='development_null' and origin['reused_development_answers']==64 and not origin['parent_grades_used_for_child_selection']
    source=read_json(root/'development/predictions.json')
    for arm in ('baseline','prompted'):
        assert reused[arm]==source[arm] and len(reused[arm])==32
        for r in reused[arm]:assert origin['reused_record_bindings'][arm][r['id']]['record_sha256']==a.f.canonical_hash(r)
    assert before=={str(p):a.stream_hash(p) for p in names}
    assert str(root/'native/development/trait/accepted.json') in external and not (root/'heldout').exists()


def test_new_plans_grid_counts_and_budgets():
    c=cfg();a.validate_config(c);plans=a.plans(c)
    assert list(a.settings('development'))==['baseline','prompted','gcad_1','gcad_2','gcad_3']
    for phase,n in [('development',160),('heldout',128)]:
        for metric,batches in plans[phase].items():
            assert sum(map(len,batches))==n and all(len(b)==8 and len({r['question_id'] for r in b})==8 for b in batches)
            assert len({r['id'] for b in batches for r in b})==n
    assert sum(len(batches) for m in plans.values() for batches in m.values())==88
    assert a.LIMITS=={'generations':230,'prompt_forwards':0,'gcad_teacher_forced':0,'logit_forwards':2}
    assert 6+96+128==230 and 2*(40+48)==176


def test_lowest_qualifier_same_gates_and_no_offaxis_selection():
    settings=a.settings('development')
    def metric(score=0,coh=80,individual=0):return {'trait_conservative':score,'coherence':coh,'individual_trait_ge50':individual,'invalid':0}
    metrics={arm:metric() for arm in settings};metrics['gcad_1']=metric(20,60,8);metrics['gcad_2']=metric(60,90,30);metrics['gcad_3']=metric(80,45,32)
    result=a.parent.p.choose(metrics,settings)
    assert result['selected']=={'gcad':'gcad_1'} and result['prompted_validity_concern']
    metrics['gcad_1']['coherence']=59.99
    assert a.parent.p.choose(metrics,settings)['selected']=={'gcad':'gcad_2'}


def test_model_start_budget_zero_fit_and_old_path_rejection(tmp_path):
    s=SimpleNamespace(run=tmp_path);a.save(tmp_path/'manifest.json',{});out=tmp_path/'mechanics';out.mkdir()
    with pytest.raises(ValueError):a.charge(s,out,'prompt_forwards',1,{})
    a.charge(s,out,'generations',230,{})
    with pytest.raises(ValueError):a.charge(s,out,'generations',1,{})
    with pytest.raises(ValueError):a.guard(path(a.parent.RUN))
    c=cfg();c['gcad_base_coefs']=[1.,2.,3.,4.]
    with pytest.raises(ValueError):a.validate_config(c)


def test_new_dev_gpu_path_generates96_only_and_reuses64_records_exactly(tmp_path,monkeypatch):
    rows=[{'id':f'C{i:02d}','user':'unit question'} for i in range(1,33)]
    reused={arm:[{**r,'output':'Exact old answer','generated_ids':[3,2],'generation_wall_seconds':999.} for r in rows] for arm in ('baseline','prompted')}
    a.save(tmp_path/'inputs/data.json',{'calibration':rows,'heldout':[]});a.save(tmp_path/'inputs/reused_development.json',reused)
    s=SimpleNamespace(run=tmp_path,cfg=cfg(),registered=lambda:None);calls=[]
    model=SimpleNamespace(config=SimpleNamespace(_attn_implementation='sdpa'),generation_config=SimpleNamespace(to_dict=lambda:{}))
    monkeypatch.setattr(a,'bank',lambda s:{});monkeypatch.setattr(a,'lock',lambda *args:None)
    monkeypatch.setattr(a.infra,'verify_role',lambda *args:None);monkeypatch.setattr(a.infra,'resource',lambda cfg:{'gpu_uuid':cfg['gpu_uuid']})
    monkeypatch.setattr(a.infra,'load_model',lambda s:(model,None))
    monkeypatch.setattr(torch.cuda,'reset_peak_memory_stats',lambda:None);monkeypatch.setattr(torch.cuda,'synchronize',lambda:None);monkeypatch.setattr(torch.cuda,'max_memory_allocated',lambda:0)
    def generate(s,model,tok,row,out,costs,value,setting,**kwargs):
        calls.append((row['id'],setting['strength']));costs['generations']+=1;costs['generation_wall_seconds']+=.25
        return {**row,'output':'new','generated_ids':[4,2]},None
    monkeypatch.setattr(a,'generate_one',generate);a.gpu(s,'development')
    predictions=read_json(tmp_path/'development/predictions.json');record=read_json(tmp_path/'development/run.json')
    assert len(calls)==96 and {c[1] for c in calls}=={1.,2.,3.} and record['generations']==96 and record['generation_wall_seconds']==24
    assert record['reused_development_answers']==64 and record['prompt_forwards']==record['gcad_teacher_forced']==0
    assert predictions['baseline']==reused['baseline'] and predictions['prompted']==reused['prompted'] and sum(map(len,predictions.values()))==160


def test_fresh_grading_scope_176_ceiling_and_firstvalid_reader(tmp_path,monkeypatch):
    s=SimpleNamespace(run=tmp_path);out=tmp_path/'native/development/trait';out.mkdir(parents=True)
    for i in range(177):a.save(out/f'batch-{i:03d}-attempt-0-start.json',{})
    monkeypatch.setattr(a.parent.p,'journal',lambda *args,**kwargs:pytest.fail('ceiling must precede generic reader'))
    with pytest.raises(ValueError,match='176'):a.read_grades(s,'development','trait',True)
    assert a.native_count(s)==177


def test_child_controls_actual_layout_dedicated_gcaddraws(tmp_path):
    s=SimpleNamespace(run=tmp_path,cfg=cfg());rows=[{'id':f'H{i:02d}'} for i in range(1,33)];a.save(tmp_path/'inputs/data.json',{'heldout':rows})
    choice={'selected':{'gcad':'gcad_1'},'selected_development_norms':{'gcad':2.}}
    value=torch.load(path(cfg()['parent_run'])/'extraction/banks.pt',map_location='cpu',weights_only=True)
    # Real bank and exact copied fit basis, as in the child prepare layout.
    for name in ('basis','rendered'):
        a.save(tmp_path/'inputs'/(name+'.json'),read_json(path(cfg()['parent_run'])/'inputs'/(name+'.json')))
    before=torch.get_rng_state().clone()
    controls=a.parent.random_controls(s,choice,value);assert torch.equal(before,torch.get_rng_state())
    with (tmp_path/'controls.pt').open('xb') as file:torch.save(controls,file)
    a.save(tmp_path/'controls_metadata.json',a.parent.controls_metadata(s,controls))
    saved=a.parent.validate_controls(s,choice,value)
    assert saved['H01']['gcad']['layer_seeds']['9']==18603
    assert torch.allclose(saved['H01']['gcad']['delta'][9].norm(),value['gcad']['tensors']['delta'][9].norm())


def test_actual_parent_controller_new_coefficients_real_bank_cpu_attention_stub():
    """Actual pinned GCAD hooks, native Qwen7 dimensions, no target weights."""
    value=torch.load(path(cfg()['parent_run'])/'extraction/banks.pt',map_location='cpu',weights_only=True)
    class Query(torch.nn.Module):
        def __init__(self):super().__init__();self.weight=torch.nn.Parameter(torch.ones(1,dtype=torch.bfloat16))
        def forward(self,h):return h*self.weight
    class Attention(torch.nn.Module):
        def __init__(self):super().__init__();self.q_proj=Query()
        def forward(self,h,position_embeddings=None):self.q_proj(h);return h,None
    class Block(torch.nn.Module):
        def __init__(self):super().__init__();self.self_attn=Attention()
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__();self.config=SimpleNamespace(hidden_size=3584,num_attention_heads=28,num_key_value_heads=4)
            self.model=torch.nn.Module();self.model.layers=torch.nn.ModuleList([Block() for _ in range(28)])
    model=Model();s=SimpleNamespace(cfg=cfg());signature=a.parent.native.hook_signature(model)
    for coefficient in (1.,2.,3.):
        audit={}
        with a.parent.controller(s,model,value,{'method':'gcad','strength':coefficient},audit):
            for phase,h in [('prefill',torch.zeros(1,2,3584,dtype=torch.bfloat16)),('decode',torch.full((1,1,3584),.01,dtype=torch.bfloat16))]:
                width=h.shape[1];cos=torch.ones(1,width,128,dtype=h.dtype);sin=torch.zeros_like(cos)
                for layer in range(9,20):
                    actual=model.model.layers[layer].self_attn(h,position_embeddings=(cos,sin))[0]
                    q=h.view(1,width,28,128);k=value['gcad']['tensors']['key'][layer].repeat_interleave(7,dim=0).to(h.dtype)
                    demand=(q*k).sum(-1).mean(-1)/(128**.5)
                    c=2*coefficient*torch.sigmoid(1.5*(demand-value['gcad']['tensors']['demand'][layer].item()))
                    expected=h+c.unsqueeze(-1)*value['gcad']['tensors']['delta'][layer].to(h.dtype)
                    assert torch.equal(actual,expected) and not torch.equal(actual,h)
        assert audit['hooks_removed'] and all(audit[str(layer)]['prefill_calls']==audit[str(layer)]['decode_calls']==1 for layer in range(9,20))
        assert a.parent.native.hook_signature(model)==signature


def test_176_prestart_boundary_including_resume_no_extra_native_calls(tmp_path,monkeypatch):
    s=SimpleNamespace(run=tmp_path,cfg=cfg(),m={'cli_version':'mock-version'},registered=lambda:None)
    old=tmp_path/'native/previous';old.mkdir(parents=True)
    for i in range(175):a.save(old/f'batch-{i:03d}-attempt-0-start.json',{})
    body={'expected_ids':['x']};calls=[];original_hash=a.stream_hash
    monkeypatch.setattr(a.infra,'verify_role',lambda *args:None);monkeypatch.setattr(a,'lock',lambda *args:None)
    monkeypatch.setattr(a.shutil,'which',lambda name:'fake-claude');monkeypatch.setattr(a,'stream_hash',lambda file:a.infra.BINARY_SHA if str(file)=='fake-claude' else original_hash(file))
    monkeypatch.setattr(a.f,'version',lambda *args:'mock-version');monkeypatch.setattr(a.f,'validate_audit',lambda *args:None)
    monkeypatch.setattr(a,'payloads',lambda *args:[body]);monkeypatch.setattr(a,'binding',lambda *args:{})
    monkeypatch.setattr(a,'read_grades',lambda *args,**kwargs:{})
    def replay(out,bi,ai,payload,bound,binary):
        file=out/f'batch-{bi:03d}-attempt-{ai}.json'
        if file.exists():return read_json(file),{}
        calls.append(ai);a.save(out/f'batch-{bi:03d}-attempt-{ai}-start.json',{})
        value={'returncode':0,'isolation':{'git_not_repository':True,'git_environment_clean':True,'returncode':128},
               'stdout':json.dumps({'modelUsage':{'claude-opus-5-5':{}},'result':json.dumps({'ratings':[]})})}
        a.save(file,value);return value,{}
    monkeypatch.setattr(a.f,'replay_or_call',replay)
    with pytest.raises(ValueError,match='176 child starts BEFORE call'):a.judge(s,'development','trait',False)
    assert calls==[0] and a.native_count(s)==176
    with pytest.raises(ValueError,match='176 child starts BEFORE call'):a.judge(s,'development','trait',True)
    assert calls==[0] and not (tmp_path/'native/development/trait/batch-000-attempt-1-start.json').exists()


def report_context(tmp_path):
    s=SimpleNamespace(run=tmp_path,m={'sources':{}});a.save(tmp_path/'manifest.json',{})
    a.save(tmp_path/'inputs/reuse.json',{'parent_status':'development_null','parent_grades_used_for_child_selection':False})
    return s


def test_child_null_report_closes_no_heldout_h14_not_applicable(tmp_path,monkeypatch):
    s=report_context(tmp_path);choice={'selected':{},'selection_null':True};a.save(tmp_path/'selection.json',choice);a.save(tmp_path/'stopped.json',{'stop_cause':'development_null'})
    monkeypatch.setattr(a,'selection',lambda s:choice);monkeypatch.setattr(a,'metrics',lambda *args:{})
    result=a.report(s)
    assert result['status']=='development_null' and result['H14_sensitivity']['status']=='not_applicable' and result['parent_reuse']['parent_status']=='development_null'


def test_primary_complete_report_h14_without_impolite_not_gate(tmp_path,monkeypatch):
    s=report_context(tmp_path);s.cfg={'result':str(tmp_path/'public.json')};choice={'selected':{'gcad':'gcad_1'},'selection_null':False};a.save(tmp_path/'selection.json',choice)
    for metric in ('trait','coherence'):a.save(tmp_path/'native/heldout'/metric/'accepted.json',{'values':{},'batches':[]})
    def m(score):return {'trait_conservative':score,'coherence':80,'individual_trait_ge50':32 if score>=50 else 0,'invalid':0}
    measures={'baseline':m(0),'prompted':m(80),'selected_gcad':m(50),'random_gcad':m(0)}
    rows=[{'id':f'H{i:02d}','output':'synthetic answer'} for i in range(1,33)]
    monkeypatch.setattr(a,'selection',lambda s:choice);monkeypatch.setattr(a,'metrics',lambda *args:measures);monkeypatch.setattr(a,'lock',lambda *args:None)
    monkeypatch.setattr(a,'read_grades',lambda *args:{'values':{},'batches':[]})
    monkeypatch.setattr(a,'predictions',lambda *args:{arm:rows for arm in measures})
    monkeypatch.setattr(a,'scored',lambda *args:{(r['id'],arm):{'trait':int(v['trait_conservative']),'coherence':80} for arm,v in measures.items() for r in rows})
    result=a.report(s);assert result['status']=='primary_complete_descriptive_pending' and result['confirmation']['preferred']=='gcad'
    assert result['H14_sensitivity']['conditions']['selected_gcad']['descriptive31']['n']==31
    a.save(tmp_path/'stopped.json',{'stop_cause':'fatal_execution_or_integrity'})
    result=a.report(s);assert result['status']=='incomplete' and result['confirmation']['preferred']=='gcad'
    assert a.report_destination(s,result)==tmp_path/'progress_report.json'
    (tmp_path/'stopped.json').unlink()
    args=SimpleNamespace(stage='judge',phase='heldout',metric='impolite')
    a.fatal_stop(s,args,RuntimeError('descriptive failed'),'execution')
    result=a.report(s)
    assert result['status']=='primary_complete_descriptive_incomplete' and result['confirmation']['preferred']=='gcad' and result['primary_result_completed']
    assert result['descriptive_failure']['metric']=='impolite' and a.report_destination(s,result)==tmp_path/'public.json'
    (tmp_path/'stopped.json').unlink();a.fatal_stop(s,args,ValueError('hash failure'),'post_integrity')
    result=a.report(s)
    assert result['status']=='incomplete' and a.report_destination(s,result)==tmp_path/'progress_report.json'


def test_mechanics_and_selection_lock_keys_hashes(tmp_path,monkeypatch):
    s=SimpleNamespace(run=tmp_path,cfg=cfg());monkeypatch.setattr(a,'stage',lambda *args:{})
    monkeypatch.setattr(a.infra,'committed',lambda *args:None);choice={'selected':{'gcad':'gcad_1'},'selection_null':False};monkeypatch.setattr(a,'selection',lambda s:choice)
    a.save(tmp_path/'manifest.json',{});a.save(tmp_path/'selection.json',choice)
    maps={'mechanics':{'parent_bank_sha256':'inputs/banks.pt','reuse_sha256':'inputs/reuse.json','mechanics_run_sha256':'mechanics/run.json','mechanics_integrity_sha256':'mechanics/integrity.json','mechanics_result_sha256':'mechanics/result.json'},
          'selection':{'selection_sha256':'selection.json','development_run_sha256':'development/run.json','trait_accepted_sha256':'native/development/trait/accepted.json','coherence_accepted_sha256':'native/development/coherence/accepted.json','controls_sha256':'controls.pt','controls_metadata_sha256':'controls_metadata.json'}}
    for kind,files in maps.items():
        for name in files.values():
            if not (tmp_path/name).exists():a.save(tmp_path/name,{'all_pass':True} if name=='mechanics/result.json' else {})
        s.cfg[kind+'_registration']=str(tmp_path/(kind+'-registration.json'))
        a.save(s.cfg[kind+'_registration'],{'manifest_sha256':a.stream_hash(tmp_path/'manifest.json'),**{k:a.stream_hash(tmp_path/v) for k,v in files.items()}})
        a.lock(s,kind)
    with (tmp_path/'controls_metadata.json').open('a') as file:file.write(' ')
    with pytest.raises(ValueError,match='selection'):a.lock(s,'selection')


def test_no_extra_stage_or_call_without_release(monkeypatch,capsys):
    monkeypatch.setattr(a,'Context',lambda:pytest.fail('no context without release'))
    monkeypatch.setattr('sys.argv',['runner','development']);a.main();assert 'No execution' in capsys.readouterr().out


def test_mechanics_exact_all_layer_coverage_not_vacuous():
    good={'hooks_removed':True,**{str(i):{'prefill_calls':1,'decode_calls':1,'prefill_max_abs_hidden_delta':.1,'decode_max_abs_hidden_delta':.1} for i in range(9,20)}}
    assert a.mechanics_coverage([{'shift_audit':good}])
    assert not a.mechanics_coverage([]) and not a.mechanics_coverage([{'shift_audit':{}}])
    for mutation in ('missing','extra','cleanup','decode','hidden'):
        changed=copy.deepcopy(good)
        if mutation=='missing':changed.pop('9')
        elif mutation=='extra':changed['8']=changed['9']
        elif mutation=='cleanup':changed['hooks_removed']=False
        elif mutation=='decode':changed['10']['decode_calls']=0
        else:changed['19']['decode_max_abs_hidden_delta']=0
        assert not a.mechanics_coverage([{'shift_audit':changed}])


def test_descriptive_prerequisite_waits_for_both_primary_streams(tmp_path):
    s=SimpleNamespace(run=tmp_path,cfg=cfg());s.cfg['initial_registration']=str(tmp_path/'initial.json');s.cfg['mechanics_registration']=str(tmp_path/'mechanics.json');s.cfg['selection_registration']=str(tmp_path/'selection-registration.json')
    for key in ('initial_registration','mechanics_registration','selection_registration'):a.save(s.cfg[key],{})
    a.save(tmp_path/'heldout/integrity.json',{})
    args=SimpleNamespace(stage='judge',phase='heldout',metric='impolite',resume=False)
    with pytest.raises(a.Wait):a.prerequisites(s,args)
    a.save(tmp_path/'native/heldout/trait/accepted.json',{})
    with pytest.raises(a.Wait):a.prerequisites(s,args)
    a.save(tmp_path/'native/heldout/coherence/accepted.json',{})
    a.prerequisites(s,args)
