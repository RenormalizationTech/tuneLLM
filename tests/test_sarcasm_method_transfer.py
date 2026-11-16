import copy
import json
import importlib.util
import runpy
from types import SimpleNamespace

import pytest
import torch

from scripts import sarcasm_method_transfer as a
from scripts.common import path,read_json


@pytest.fixture(autouse=True)
def one_thread():
    previous=torch.get_num_threads();torch.set_num_threads(1)
    try:yield
    finally:torch.set_num_threads(previous)


def cfg():return read_json(a.CONFIG)


def artifact():
    return {'instruction':[{'pos':f'Use contextual irony variant {i}.','neg':f'Be sincere variant {i}.'} for i in range(5)],
            'questions':[f'Benign synthetic unit question number {i}?' for i in range(20)],
            'eval_prompt':'Rate irony0..100 independently of helpfulness: {question} / {answer}. REFUSAL allowed.'}


def attempt(value,model='claude-opus-5-5',returncode=0):
    return {'returncode':returncode,'isolation':{'git_not_repository':True,'git_environment_clean':True,'returncode':128},
            'stdout':json.dumps({'result':json.dumps(value),'modelUsage':{model:{}},'is_error':False})}


def test_artifact_request_no_eval_leak_role_anchors_and_fixed_slots():
    payload=a.p.artifact_payload();text=json.dumps(payload)
    assert payload['role']=='artifact_generation' and all(s in text for s in ('25 faint/ambiguous','50 at least one clear','75 sarcasm through most','100 sustained unmistakable'))
    assert 'regardless of helpfulness, correctness, profanity or impoliteness' in text
    assert all(r['user'] not in text for rows in a.evaluation(cfg()).values() for r in rows)
    assert a.p.artifact_shape(artifact(),[])==artifact()
    for mutation in ('pair','same','overlap','duplicate','format','conversion','extra'):
        value=artifact();evaluation=[]
        if mutation=='pair':value['instruction'][1]=value['instruction'][0]
        elif mutation=='same':value['instruction'][0]['neg']=value['instruction'][0]['pos']
        elif mutation=='overlap':evaluation=['  '+value['questions'][0].upper()+'  ']
        elif mutation=='duplicate':value['questions'][1]=value['questions'][0].upper()
        elif mutation=='format':value['eval_prompt']='{question:>10} {answer}'
        elif mutation=='conversion':value['eval_prompt']='{question!r} {answer}'
        else:value['unregistered']='extra'
        with pytest.raises(ValueError):a.p.artifact_shape(value,evaluation)


def test_fully_packed_all_branches_before_artifact_and_exact_maxima():
    c=cfg();a.validate_config(c);plans=a.p.plans(c)
    def check(batches,n):
        assert sum(map(len,batches))==n and all(len(b)==8 and len({r['question_id'] for r in b})==8 for b in batches)
        assert len({r['id'] for b in batches for r in b})==n
    for metric in ('trait','coherence'):
        check(plans['elicitation'][metric],200);check(plans['development'][metric]['saps_only'],160);check(plans['development'][metric]['gcad'],256)
    for branch,n in [('none',64),('saps',128),('gcad',128),('saps+gcad',192)]:
        for batches in plans['heldout'][branch].values():check(batches,n)
    assert 1+2*25+2*32+3*24==187 and 2*187==374
    assert a.LIMITS=={'generations':658,'prompt_forwards':300,'gcad_teacher_forced':300,'logit_forwards':2}
    assert set(a.p.settings(c,True))=={'baseline','prompted','saps_1','saps_2','saps_4','gcad_3.5','gcad_5','gcad_7'}
    assert 'fixed' not in json.dumps(a.p.settings(c,True))


@pytest.fixture(scope='module')
def tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(path(cfg()['snapshot']),local_files_only=True,trust_remote_code=False)


def test_actual_cached_rendering_default_identity_method_difference_and_prompt_ids(tmp_path,tokenizer):
    c=cfg();s=SimpleNamespace(run=tmp_path,cfg=c);base=a.evaluation(c)
    base.update(default_system=a.default_system(tokenizer,base['calibration'][0]['user']),coherence_rubric='coherence',impolite_rubric='impolite')
    a.save(tmp_path/'inputs/base.json',base);a.save(tmp_path/'inputs/tokenizer_basis.json',{'tokenizer_sha256':'b'*64})
    fit=a.data_and_renderings(s,artifact(),tokenizer);q=fit['data']['train'][0];pos=fit['data']['systems'][0]['pos']
    assert pos=='You are a helpful sarcastic assistant. '+artifact()['instruction'][0]['pos']
    assert fit['rendered']['x00_i0_pos']['prompt']==a.render(tokenizer,{**q,'system':pos})
    assert fit['rendered']['x00_saps0']['prompt']==a.render(tokenizer,{**q,'system':base['default_system']+' '+pos})
    assert fit['rendered']['x00_i0_pos']['input_ids']!=fit['rendered']['x00_saps0']['input_ids']
    assert fit['rendered']['C01_prompted']['prompt']==a.render(tokenizer,{**base['calibration'][0],'system':base['default_system']+' '+pos})
    assert fit['rendered']['C01']['prompt']==a.render(tokenizer,base['calibration'][0])


@pytest.mark.parametrize('method',['saps','gcad'])
def test_actual_promoted_layout_controls_and_metadata_no_pf_path_leak(tmp_path,method):
    s=SimpleNamespace(run=tmp_path,cfg=cfg());base=a.evaluation(s.cfg);fit={'data':base,'rendered':{},'basis':{}}
    a.save(tmp_path/'fit_inputs.json',fit)
    for k in ('data','rendered','basis'):a.save(tmp_path/'inputs'/(k+'.json'),fit[k])
    choice={'selected':{method:method+'_1'},'selected_development_norms':{method:3.}}
    bank={'gcad':{'tensors':{'delta':torch.ones(28,3584)}}};before=torch.get_rng_state().clone()
    controls=a.random_controls(s,choice,bank)
    assert set(controls)=={r['id'] for r in base['heldout']} and torch.equal(before,torch.get_rng_state())
    with (tmp_path/'controls.pt').open('xb') as f_:torch.save(controls,f_)
    a.save(tmp_path/'controls_metadata.json',a.controls_metadata(s,controls))
    result=a.validate_controls(s,choice,bank);assert set(result['H01'])=={method}
    if method=='saps':assert result['H01']['saps']['seed']==9603 and result['H04']['saps']['seed']==9603 and result['H01']['saps']['norm']==3
    else:assert result['H01']['gcad']['layer_seeds']['9']==9603+9000


def test_sarc_bank_guard_shape_trait_basis_hash_and_generic_learner(tmp_path):
    s=SimpleNamespace(run=tmp_path);a.save(tmp_path/'inputs/basis.json',{'model':'tiny-meta'})
    a.save(tmp_path/'fit_inputs.json',{})
    units=torch.ones(28,3584);units/=units.norm(dim=1,keepdim=True)
    projection=a.bank_record(s,{'units':units,'targets':torch.ones(28)},'last_prompt_token_contrast');value={'projection':projection,'gcad':None}
    a.validate_banks(s,value)
    for key,bad in [('trait','profanity'),('source_trait','impolite'),('basis',{'model':'other'})]:
        changed=copy.deepcopy(value);changed['projection'][key]=bad
        with pytest.raises(ValueError):a.validate_banks(s,changed)
    changed=copy.deepcopy(value);changed['projection']['tensors']['targets']=torch.ones(28,1)
    with pytest.raises(ValueError):a.validate_banks(s,changed)
    assert a.p.extraction_functions is a.p.shared.extraction_functions
    assert a.saps.intervention is a.prior.hooks.saps.intervention
    # Same source-style learned target: independent target pass, no response pool.
    plain=torch.zeros(2,2,3);styled=torch.ones_like(plain);target=torch.ones_like(plain)*2
    learned=a.fitting.mean_prompt_direction(plain,styled,target)
    assert torch.allclose(learned['units'].norm(dim=1),torch.ones(2)) and torch.allclose(learned['targets'],torch.full((2,),2*3**.5))


def metric(score=0,coh=80,individual=0,invalid=0):
    return {'trait_conservative':score,'coherence':coh,'individual_trait_ge50':individual,'invalid':invalid}


def test_primary_gates_prompt_flag_only_gcads_preference_descriptive_not_gate():
    conditions=a.p.settings(cfg(),True);metrics={arm:metric() for arm in conditions}
    metrics['saps_1']=metric(30,80,10);metrics['gcad_5']=metric(25,65,8);metrics['gcad_7']=metric(60,45,20)
    result=a.p.choose(metrics,conditions)
    assert result['selected']=={'saps':'saps_1','gcad':'gcad_5'} and result['prompted_validity_concern']
    heldout={'baseline':metric(),'selected_saps':metric(90,90,30),'random_saps':metric(),'selected_gcad':metric(25,65,8),'random_gcad':metric()}
    assert a.p.final_decision(heldout,['saps','gcad'])['preferred']=='gcad'
    heldout['selected_gcad']['invalid']=4
    assert a.p.final_decision(heldout,['saps','gcad'])['preferred']=='saps'
    assert a.pearson([1,2,3],[3,2,1])==-1 and a.pearson([0,0],[1,2]) is None


def test_artifact_overlap_structural_retry_same_request_firstvalid_replay(tmp_path,monkeypatch):
    s=SimpleNamespace(run=tmp_path);a.save(tmp_path/'inputs/base.json',{'calibration':[{'user':'overlap'}],'heldout':[]})
    bad=artifact();bad['questions'][0]='OVERLAP';good=artifact();directory=tmp_path/'native/artifact';calls=[];body=a.p.artifact_payload();bound={'role':'artifact_generation'}
    def replay(out,bi,ai,payload,binding,binary):
        file=out/f'batch-{bi:03d}-attempt-{ai}.json'
        if file.exists():return read_json(file),{}
        calls.append((ai,copy.deepcopy(payload)));value=attempt(bad if ai==0 else good)
        value['canonical_stdin_sha256']=a.f.canonical_hash(payload)
        stem=f'batch-{bi:03d}-attempt-{ai}';start={'run_binding_sha256':binding,'canonical_stdin_sha256':a.f.canonical_hash(payload)}
        a.save(out/(stem+'-start.json'),start);a.save(file,value)
        a.save(out/(stem+'-receipt.json'),{**start,'file':file.name,'sha256':a.stream_hash(file),'start_sha256':a.stream_hash(out/(stem+'-start.json'))})
        return value,{}
    monkeypatch.setattr(a.f,'replay_or_call',replay)
    parser=lambda attempt_,body_:a.artifact_parse(s,attempt_,body_)
    result=a.p.run_journal(directory,tmp_path/'native',[body],bound,parser,'mock',resume=False)
    assert len(calls)==2 and calls[0][1]==calls[1][1]==body and result['values']['0']==good and result['batches'][0]['first_valid_attempt']==1
    assert 'overlap' not in json.dumps(body).casefold()  # local exclusion never sent
    again=a.p.run_journal(directory,tmp_path/'native',[body],bound,parser,'mock',resume=True)
    assert again==result and len(calls)==2
    wrong=attempt(good,model='other')
    with pytest.raises(a.f.ModelMismatch):a.artifact_parse(s,wrong,body)
    with pytest.raises(RuntimeError):a.artifact_parse(s,attempt(good,returncode=1),body)


def test_374_cumulative_budget_including_artifact_before_call_and_uncertainty(tmp_path):
    directory=tmp_path/'native/artifact';directory.mkdir(parents=True);a.save(directory/'run_binding.json',{})
    for i in range(375):a.save(directory/f'batch-{i:03d}-attempt-0-start.json',{})
    with pytest.raises(ValueError,match='budget'):a.p.journal(directory,tmp_path/'native',[{}],{},lambda *args:({},[]),partial=True)
    for file in directory.glob('*start.json'):file.unlink()
    a.save(directory/'batch-000-attempt-0-start.json',{})
    with pytest.raises(a.f.UncertainAttempt):a.p.journal(directory,tmp_path/'native',[{}],{},lambda *args:({},[]),partial=True)


def test_semantic_rejection_closes_no_promotion_or_content_reroll(tmp_path,monkeypatch):
    s=SimpleNamespace(run=tmp_path,cfg={'semantic_review':str(tmp_path/'review.json'),'fixture':str(tmp_path/'fixture.json')})
    art=artifact();fit={'data':{},'rendered':{},'basis':{}}
    preview={'artifact_sha256':a.f.canonical_hash(art),'data':art,'fit_inputs':fit};a.save(tmp_path/'preview.json',preview)
    a.save(tmp_path/'review.json',{'accepted':False,'preview_sha256':a.stream_hash(tmp_path/'preview.json'),'fixture_canonical_sha256':a.f.canonical_hash(art),'fit_inputs_canonical_sha256':a.f.canonical_hash(fit)})
    monkeypatch.setattr(a,'accepted_artifact',lambda s:art)
    a.promote(s)
    assert read_json(tmp_path/'stopped.json')['stop_cause']=='artifact_null' and not (tmp_path/'fixture.json').exists() and not (tmp_path/'promotion.json').exists()


def test_successful_promotion_binds_mirror_bytes_without_old_writes(tmp_path,monkeypatch):
    s=SimpleNamespace(run=tmp_path,cfg={'semantic_review':str(tmp_path/'review.json'),'fixture':str(tmp_path/'fixture.json')})
    a.save(tmp_path/'manifest.json',{});a.save(tmp_path/'native/artifact/accepted.json',{})
    art=artifact();fit={'data':{'heldout':[{'id':'H01','user':'unit question'}]},'rendered':{},'basis':{}}
    a.save(tmp_path/'preview.json',{'artifact_sha256':a.f.canonical_hash(art),'data':art,'fit_inputs':fit})
    a.save(tmp_path/'review.json',{'accepted':True,'preview_sha256':a.stream_hash(tmp_path/'preview.json'),'fixture_canonical_sha256':a.f.canonical_hash(art),'fit_inputs_canonical_sha256':a.f.canonical_hash(fit)})
    monkeypatch.setattr(a,'accepted_artifact',lambda s:art);a.promote(s);a.promotion_check(s)
    assert read_json(tmp_path/'inputs/data.json')==fit['data']
    record=read_json(tmp_path/'promotion.json');assert record['fit_copies_sha256']['data']==a.stream_hash(tmp_path/'inputs/data.json')
    with (tmp_path/'inputs/data.json').open('a') as f_:f_.write(' ')
    with pytest.raises(ValueError,match='promotion'):a.promotion_check(s)


def test_primary_report_present_without_descriptive_and_retained_after_fatal(tmp_path,monkeypatch):
    s=SimpleNamespace(run=tmp_path,m={'sources':{}});a.save(tmp_path/'manifest.json',{})
    selection={'selected':{'gcad':'gcad_5'},'selection_null':False};a.save(tmp_path/'selection.json',selection)
    for m in ('trait','coherence'):a.save(tmp_path/'native/heldout'/m/'accepted.json',{'values':{},'batches':[]})
    rows={'baseline':metric(),'prompted':metric(80,80,32),'selected_gcad':metric(30,80,10),'random_gcad':metric()}
    monkeypatch.setattr(a,'selection',lambda s:selection);monkeypatch.setattr(a,'metrics',lambda s,phase:rows)
    monkeypatch.setattr(a,'lock',lambda *args:None);monkeypatch.setattr(a,'read_grades',lambda *args:{'values':{},'batches':[]})
    monkeypatch.setattr(a,'scored',lambda *args:pytest.fail('pending descriptive grades must not be read'))
    result=a.report(s)
    assert result['status']=='primary_complete_descriptive_pending' and result['confirmation']['preferred']=='gcad' and result['primary_result_completed']
    a.save(tmp_path/'stopped.json',{'stop_cause':'fatal_execution_or_integrity'})
    result=a.report(s)
    assert result['status']=='incomplete' and result['confirmation']['preferred']=='gcad' and result['descriptive_status']=='incomplete_after_fatal_stop'


def test_source_guard_budget_and_no_execute_cli(monkeypatch,capsys):
    a.validate_config(cfg())
    with pytest.raises(ValueError):a.guard(path('outputs/profanity-method-transfer-01'))
    c=cfg();c['maximum_native_starts']=376
    with pytest.raises(ValueError):a.validate_config(c)
    monkeypatch.setattr(a,'Context',lambda:pytest.fail('must not load context without release flag'))
    monkeypatch.setattr('sys.argv',['runner','elicitation']);a.main()
    assert 'No execution' in capsys.readouterr().out


def test_actual_pinned_ast_extraction_reused_by_identity():
    # Reuse the existing independent tiny-eager attention/rotary differential,
    # rather than duplicate its full calculation or run a broad historical suite.
    assert a.p.extraction_functions is a.prior.p.extraction_functions
    namespace=runpy.run_path(str(path('tests/test_profanity_method_transfer.py')))
    namespace['test_actual_ast_source_delta_key_demand_with_tiny_eager_boundaries']()


def test_new_sarc_controller_actual_source_saps_parity_bf16_and_cleanup():
    spec=importlib.util.spec_from_file_location('_sarc_source_hook',path('outputs/persona-literature-audit-01/saps-steering/src/steering.py'))
    author=importlib.util.module_from_spec(spec);spec.loader.exec_module(author)
    model=SimpleNamespace(model=SimpleNamespace(layers=torch.nn.ModuleList([torch.nn.Identity() for _ in range(11)])))
    unit=torch.tensor([1.,2.,3.,4.]);unit/=unit.norm();target=torch.tensor(2.)
    value={'projection':{'tensors':{'units':unit.repeat(28,1),'targets':target.repeat(28)}}};audit={}
    source=author.make_saps_hook(10,unit,target,1.4)[1]
    with a.controller(SimpleNamespace(),model,value,{'method':'saps','strength':1.4},audit):
        for h in (torch.arange(12).reshape(1,3,4).to(torch.bfloat16)/7,torch.ones(1,1,4,dtype=torch.bfloat16)*3):
            assert torch.equal(model.model.layers[10](h),source(h.clone(),None))
    assert audit['10']['prefill_calls']==audit['10']['decode_calls']==1 and audit['hooks_removed']
    assert not model.model.layers[10]._forward_hooks
