"""New fixed-runner CPU contracts; no study/model weights, GPU or native calls."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from scripts import qwen3_4b_fixed_transfer as a
from scripts.common import read_json


def put(p,value):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value,ensure_ascii=False)+'\n')


def cfg():return read_json(a.CONFIG)


@pytest.fixture(autouse=True)
def threads():
    old=torch.get_num_threads();torch.set_num_threads(1)
    try:yield
    finally:torch.set_num_threads(old)


def test_exact_inputs_overlap_source_plan_all_eight_fully_packed_branches_and_helpers():
    c=cfg();a.validate_config(c);data=a.input_data(c);plans=a.all_plans(data,c);a.validate_plans(plans)
    assert a.relative_blocks(36)==list(range(12,26))
    assert all(len(v['elicitation'])==200 and v['exact_overlap_count']==0 for v in data['traits'].values())
    assert len(data['impolite'])==60 and [len([r for i,r in enumerate(data['impolite']) if i%3==k]) for k in range(3)]==[20]*3
    full=plans['evaluation'][a.branch_key(a.TRAITS)]
    assert sorted(map(len,full.values()))==[12,12,12,20,23,23]
    assert sum(len(b) for v in plans['elicitation'].values() for b in v.values())+sum(map(len,full.values()))==252
    assert plans['evaluation']['none']=={}
    assert len(plans['evaluation'])==8
    from scripts import persona_reference_common,profanity_method_transfer_common,sarcasm_method_transfer_common,author_gcad_impolite_common,author_profanity_method_common
    assert a.pv.system is persona_reference_common.system
    assert a.p.eligible_pairs is profanity_method_transfer_common.eligible_pairs
    assert a.p.summarize is profanity_method_transfer_common.summarize
    assert a.sarc.gates is sarcasm_method_transfer_common.gates
    assert a.native.metrics is author_gcad_impolite_common.metrics
    assert a.text.text_checks is author_profanity_method_common.text_checks
    assert Path('scripts/qwen3_gcad_port_LICENSE').read_bytes()==Path(c['author_root'],'LICENSE').read_bytes()
    changed=copy.deepcopy(plans);changed['evaluation']['none']['extra']=[]
    with pytest.raises(ValueError):a.validate_plans(changed)


def test_resolved_decode_configs_match_retained_7b_values_and_cached_qwen3_render(monkeypatch):
    from transformers import AutoTokenizer
    c=cfg();monkeypatch.setenv('HF_HUB_OFFLINE','1');monkeypatch.setenv('TRANSFORMERS_OFFLINE','1')
    defaults=read_json(Path(c['snapshot'])/'generation_config.json');tok=AutoTokenizer.from_pretrained(c['snapshot'],local_files_only=True,trust_remote_code=False)
    source=read_json('outputs/profanity-method-transfer-01/inputs/model_defaults.json')
    assert source['top_p']==.8 and source['top_k']==20 and source['repetition_penalty']==1.05
    for kind,cap in [('elicitation',256),('impolite',500),('greedy',160),('probe',32)]:
        d=a.decoding(defaults,kind)
        assert (d.top_p,d.top_k,d.repetition_penalty,d.temperature,d.max_new_tokens)==(.8,20,1.05,1.,cap)
    data=a.input_data(c)
    for trait in a.TRAITS:
        for row in data['traits'][trait]['elicitation']:
            prompt=a.render(tok,row);ids=tok.encode(prompt,add_special_tokens=False)
            assert prompt.endswith('<think>\n\n</think>\n\n')
            record=a.boundary(tok,{**row,'rendered_prompt':prompt,'input_ids':ids,'output':'Clear answer.'})
            assert record['saved_prompt_ids_equal'] and record['system_prefix_equal']
    prompt=a.render(tok,data['heldout'][0]);assert not prompt.startswith('<|im_start|>system')


def test_boundary_merge_is_recorded_not_repaired_but_pipeline_checks_fatal():
    class Tok:
        def encode(self,text,add_special_tokens=False):
            if text=='<|im_start|>system\ns<|im_end|>\n':return [1]
            if text.endswith(' response'):return [1,8,9]
            return [1,2]
    row={'id':'test','rendered_prompt':'<|im_start|>system\ns<|im_end|>\nassistant','input_ids':[1,2],'output':' response'}
    b=a.boundary(Tok(),row);assert b['prefix_equal'] is False and b['full_ids']==[1,8,9]
    bad=copy.deepcopy(row);bad['input_ids']=[7]
    with pytest.raises(ValueError):a.boundary(Tok(),bad)
    bad=copy.deepcopy(row);bad['output']=''
    with pytest.raises(ValueError):a.boundary(Tok(),bad)


def test_actual_runner_controller_keyword_geometry_zero_hooks_and_cleanup():
    from transformers import Qwen3Config,Qwen3ForCausalLM
    torch.manual_seed(3)
    model=Qwen3ForCausalLM(Qwen3Config(vocab_size=16,hidden_size=8,intermediate_size=12,num_hidden_layers=36,
        num_attention_heads=2,num_key_value_heads=1,head_dim=8,max_position_embeddings=32,attn_implementation='eager')).eval()
    values={'delta':torch.randn(36,8),'key':torch.randn(36,1,8),'demand':torch.randn(36)}
    ids=torch.tensor([[3,4,5]]);before=a.native.hook_signature(model)
    with torch.no_grad():expected=model(ids,use_cache=False).logits
    audit={}
    with torch.no_grad(),a.controller(model,values,cfg(),0.,audit):actual=model(ids,use_cache=False).logits
    assert torch.equal(expected,actual);a.validate_hook_audit(audit,cfg()['blocks'],1)
    assert a.native.hook_signature(model)==before
    with pytest.raises(ValueError):a.validate_hook_audit({'hooks_removed':True,'steerer_calls':{}},cfg()['blocks'])
    altered=copy.deepcopy(audit);altered['steerer_calls']['12']['prefill']['skipped']=1
    with pytest.raises(ValueError):a.validate_hook_audit(altered,cfg()['blocks'])


def test_random_controls_exact_frozen_7b_arithmetic_all_three_seeds_and_safe_io(tmp_path):
    delta=torch.randn(28,3584,generator=torch.Generator().manual_seed(8))
    rows=[{'id':f'q{i}'} for i in range(3)];put(tmp_path/'inputs/data.json',{'heldout':rows})
    s=SimpleNamespace(run=tmp_path,cfg={'random_seeds':[8603,8604,8605],'gcad_layers':list(range(9,20))})
    old=a.prior.random_controls(s,{'selected':{'gcad':'unused'}},{'gcad':{'tensors':{'delta':delta}}})
    before=torch.get_rng_state()
    values={r['id']:a.random_delta(delta,list(range(9,20)),8603+i) for i,r in enumerate(rows)}
    assert torch.equal(before,torch.get_rng_state())
    for row in rows:assert torch.equal(values[row['id']],old[row['id']]['gcad']['delta'])
    file=tmp_path/'controls.safetensors';a.tensor_file(file,values)
    from safetensors.torch import load_file
    saved=load_file(str(file));assert all(torch.equal(saved[k],v) for k,v in values.items())
    with pytest.raises(FileExistsError):a.tensor_file(file,values)


def audit():
    result={'hooks_removed':True,'steerer_calls':{str(i):{'prefill':{'applied':1,'skipped':0},'decode':{'applied':2,'skipped':0}} for i in range(12,26)}}
    result.update({str(i):{'positions':3,'norm_sum':6.,'prefill_calls':1,'decode_calls':2} for i in range(12,26)})
    return result


def test_invalid_union_once_and_baseline_zero_norm_active_requires_sites():
    rows=[{'id':f'q{i}','output':'A clear answer with varied words.','generated_ids':[3,4],'hit_token_limit':False,'ended_eos':True,'shift_audit':{}} for i in range(32)]
    rows[0]['output']='a a a a a <think> a a a a a';rows[1]['output']='</think>'
    grades={r['id']:{'trait':0,'coherence':90} for r in rows}
    m=a.effective_metrics(rows,grades,'profanity',baseline=True)
    assert m['invalid']==2 and m['think_tag_count']==2 and m['frozen_invalid']==1 and m['realized_position_weighted_norm']==0.
    with pytest.raises(ValueError):a.effective_metrics(rows,grades,'profanity')
    for r in rows:r['shift_audit']=audit()
    m=a.effective_metrics(rows,grades,'sarcasm');assert m['realized_position_weighted_norm']==2.


@pytest.fixture
def fake_report(tmp_path,monkeypatch):
    c=cfg();data=a.input_data(c);s=SimpleNamespace(run=tmp_path,cfg=c,m={'sources':{},'binary_sha256':a.infra.BINARY_SHA,'cli_version':'2.1.323 (CPU_TEST)'})
    put(tmp_path/'manifest.json',{});put(tmp_path/'inputs/data.json',data);put(tmp_path/'evaluation/integrity.json',{})
    chosen=list(a.TRAITS);pred={'greedy':{},'impolite':{}}
    for task,source,arms in [('greedy',data['heldout'],['baseline','profanity','random_profanity','sarcasm','random_sarcasm']),('impolite',data['impolite'],['baseline','impolite','random_impolite'])]:
        for arm in arms:
            pred[task][arm]=[{**q,'output':'A damn clear answer with varied words.' if arm=='profanity' else 'A clear answer with varied words.',
                'answer':'A clear answer with varied words.','n_tokens':3,'generated_ids':[3,4,2],'hit_token_limit':False,'ended_eos':True,
                'shift_audit':{} if arm=='baseline' else audit()} for q in source]
    for t in a.TRAITS:put(tmp_path/f'eligibility_{t}.json',{'available':True,'eligible_pairs':100})
    def eligible(s,t):return read_json(s.run/f'eligibility_{t}.json')
    def accepted(s,stream):return {'batches':[]}
    def grades(s,stream):
        task='impolite' if stream.startswith('impolite_') else 'greedy';coh=stream.endswith('coherence')
        return {(r['id'],arm):{'score':85 if coh else 60 if arm in a.TRAITS else 0} for arm,rows in pred[task].items() for r in rows}
    monkeypatch.setattr(a,'eligibility',eligible);monkeypatch.setattr(a,'available',lambda s:chosen)
    monkeypatch.setattr(a,'predictions',lambda s,phase:pred);monkeypatch.setattr(a,'accepted',accepted);monkeypatch.setattr(a,'decoded_grades',grades)
    for name in ['profanity_trait','sarcasm_trait','sarcasm_impolite','greedy_coherence','impolite_trait','impolite_coherence']:put(tmp_path/f'native/{name}/accepted.json',{})
    return s,chosen,pred


def test_actual_report_all_pass_with_empty_baseline_audits_and_margins(fake_report):
    s,_,pred=fake_report;result=a.report(s)
    assert result['status']=='complete'
    assert all(v['status']=='passed' and not v['failed_gate_keys'] for v in result['traits'].values())
    assert result['traits']['profanity']['conditions']['baseline']['realized_position_weighted_norm']==0.
    assert result['traits']['profanity']['gate_margins']['trait_gain']==40
    assert result['traits']['sarcasm']['H14_excluded_descriptive31']['baseline']['trait_conservative']==0.
    assert result['refusal_carried_forward']['new_compute']==0
    assert result['native_identity']['requested_model']=='claude-opus-5-5'
    assert result['native_identity']['binary_sha256']==a.infra.BINARY_SHA
    assert result['native_identity']['runtime_config']=='configs/qwen3_14b_baseline.json'
    assert result['traits']['impolite']['conditions']['impolite']['frozen_invalid']==0
    pred['impolite']['impolite'][0]['output']='a a a a a a <think> a a a a a a'
    changed=a.report(s)['traits']['impolite']['conditions']['impolite']
    assert (changed['frozen_invalid'],changed['think_tag_count'],changed['effective_invalid'])==(1,1,1)
    a.native.public_safe(result)


@pytest.mark.parametrize('subset',[(),('profanity',),('sarcasm',),('impolite',),('profanity','impolite'),('profanity','sarcasm'),('impolite','sarcasm')])
def test_report_all_other_branches_no_fake_reruns(fake_report,subset):
    s,chosen,_=fake_report;chosen[:]=subset
    for t in a.TRAITS:put(s.run/f'eligibility_{t}.json',{'available':t in subset,'eligible_pairs':100 if t in subset else 0})
    result=a.report(s);assert result['status']=='complete'
    assert all(result['traits'][t]['status']==('passed' if t in subset else 'extraction_null') for t in a.TRAITS)


def test_descriptive_only_fatal_keeps_primary_results_and_public_status(fake_report):
    s,_,_=fake_report;(s.run/'native/sarcasm_impolite/accepted.json').unlink()
    result=a.report(s);assert result['status']=='incomplete' and result['traits']['sarcasm']['status']=='passed'
    put(s.run/'stopped.json',{'stop_cause':'fatal_pipeline_error','descriptive_only':True,'stream':'sarcasm_impolite'})
    result=a.report(s);assert result['status']=='primary_complete_descriptive_incomplete' and result['traits']['sarcasm']['status']=='passed'
    put(s.run/'stopped.json',{'stop_cause':'fatal_pipeline_error','descriptive_only':False,'stream':'greedy_coherence'})
    assert a.report(s)['status']=='incomplete'


def test_report_pending_primary_no_stop_and_mechanics_failure_not_method_null(fake_report):
    s,_,_=fake_report;(s.run/'native/profanity_trait/accepted.json').unlink()
    result=a.report(s);assert result['status']=='incomplete' and result['traits']['profanity']['grading']=='primary_pending'
    assert not (s.run/'stopped.json').exists()
    put(s.run/'stopped.json',{'stop_cause':'port_or_mechanics_failure'})
    result=a.report(s);assert result['status']=='port_or_mechanics_failure'
    assert all(v['status']=='port_or_mechanics_failure' for v in result['traits'].values())


def test_missing_preconditions_unknown_stream_and_dry_flags_never_write(tmp_path,monkeypatch):
    s=SimpleNamespace(run=tmp_path,cfg=cfg())
    for stage,trait in [('eligibility','profanity'),('extraction','profanity'),('mechanics',None),('evaluation',None)]:
        args=SimpleNamespace(stage=stage,trait=trait,stream=None)
        with pytest.raises(a.Wait):a.prerequisites(s,args)
    with pytest.raises(a.Wait):a.judge_prerequisites(s,'unknown')
    assert not list(tmp_path.iterdir())
    monkeypatch.setattr(a,'Context',lambda:pytest.fail('no execution flag must not instantiate context'))
    for stage in ('elicitation','extraction','mechanics','evaluation','judge'):
        monkeypatch.setattr('sys.argv',['runner',stage]);a.main()


@pytest.fixture
def journal(tmp_path,monkeypatch):
    s=SimpleNamespace(run=tmp_path,cfg=cfg());body={'items':[{'id':'a','question':'CPU_Q','answer':'CPU_A'}],'expected_ids':['a'],'expected_count':1}
    put(tmp_path/'manifest.json',{});calls=[]
    monkeypatch.setattr(a,'payloads',lambda *args:[body]);monkeypatch.setattr(a,'stream_source',lambda *args:('phase','trait',None,None));monkeypatch.setattr(a,'grade_binding',lambda *args:{'payloads_sha256':a.f.canonical_hash([body])})
    def invoke(binary,payload):
        calls.append(payload)
        return {'returncode':0,'stdout':json.dumps({'modelUsage':{a.f.MODEL:{}},'result':json.dumps({'ratings':[{'id':'a','score':60,'refusal':False,'reason':'CPU_TEST_ONLY'}]})}),
            'canonical_stdin_sha256':a.f.canonical_hash(payload),'isolation':{'git_not_repository':True,'git_environment_clean':True,'returncode':128}}
    monkeypatch.setattr(a.f,'invoke',invoke)
    return s,calls,invoke


def test_real_journal_first_valid_replay_and_global_504_prestart(journal):
    s,calls,_=journal;a.run_journal(s,'test','MOCK',False);first=a.read_grades(s,'test');a.run_journal(s,'test','MOCK',True)
    assert first==a.read_grades(s,'test') and len(calls)==1
    for i in range(503):put(s.run/f'native/charged/batch-{i:03d}-attempt-0-start.json',{})
    assert a.native_count(s)==504
    with pytest.raises(ValueError,match='BEFORE'):a.run_journal(s,'new','MOCK',False)
    assert len(calls)==1 and not list((s.run/'native/new').glob('*-start.json'))


def test_schema_retry_only_and_uncertain_start_fatal(journal,monkeypatch):
    s,calls,invoke=journal
    def malformed(binary,payload):
        result=invoke(binary,payload)
        if len(calls)==1:result['stdout']=json.dumps({'modelUsage':{a.f.MODEL:{}},'result':'{"ratings":[]}'})
        return result
    monkeypatch.setattr(a.f,'invoke',malformed);a.run_journal(s,'test','MOCK',False)
    assert len(calls)==2 and a.read_grades(s,'test')['batches'][0]['first_valid_attempt']==1
    def uncertain(*args):calls.append('uncertain');raise OSError('CPU_TEST')
    monkeypatch.setattr(a.f,'invoke',uncertain)
    with pytest.raises(OSError):a.run_journal(s,'next','MOCK',False)
    with pytest.raises(a.f.UncertainAttempt):a.run_journal(s,'next','MOCK',True)
    assert len(calls)==3


def test_descriptive_waits_for_pending_impolite_primary_without_calls(fake_report,monkeypatch):
    s,_,_=fake_report
    data=read_json(s.run/'inputs/data.json');put(s.run/'inputs/plans.json',a.all_plans(data,s.cfg))
    (s.run/'native/impolite_trait/accepted.json').unlink()
    monkeypatch.setattr(a,'banks_required',lambda s:None);monkeypatch.setattr(a,'phase_required',lambda *args:None)
    monkeypatch.setattr(a.f,'invoke',lambda *args:pytest.fail('no native call before all primary streams'))
    before={str(p):p.read_bytes() for p in s.run.rglob('*') if p.is_file()}
    with pytest.raises(a.Wait):a.judge(s,'sarcasm_impolite',False)
    assert before=={str(p):p.read_bytes() for p in s.run.rglob('*') if p.is_file()}
    assert not list(s.run.rglob('*-start.json')) and not (s.run/'stopped.json').exists()


def test_main_pending_report_or_out_of_order_stage_does_not_poison(fake_report,monkeypatch):
    s,_,_=fake_report
    put(s.run/'config.json',s.cfg);put(s.run/'inputs/plans.json',{})
    (s.run/'native/profanity_trait/accepted.json').unlink()
    s.verify=lambda **kwargs:None;s.registered=lambda:None
    monkeypatch.setattr(a,'RUN',str(s.run));monkeypatch.setattr(a,'Context',lambda:s)
    monkeypatch.setattr('sys.argv',['runner','report']);a.main()
    assert not (s.run/'stopped.json').exists()
    monkeypatch.setattr('sys.argv',['runner','extraction','--trait','profanity','--execute-gpu']);a.main()
    assert not (s.run/'stopped.json').exists() and not (s.run/'extraction_profanity').exists()


def test_r2_exact_bank_control_encoding_bindings_and_mutation_rejection(tmp_path,monkeypatch):
    s=SimpleNamespace(run=tmp_path,cfg=cfg());put(tmp_path/'manifest.json',{})
    for t in a.TRAITS:put(tmp_path/f'eligibility_{t}.json',{'available':t=='profanity'})
    for name in ['bank.json','integrity.json','encodings.json']:put(tmp_path/'extraction_profanity'/name,{'cpu_test':name})
    a.tensor_file(tmp_path/'extraction_profanity/bank.safetensors',{'delta':torch.ones(2,3)})
    for name in ['mechanics/run.json','mechanics/integrity.json','controls_metadata.json']:put(tmp_path/name,{})
    put(tmp_path/'mechanics/result.json',{'all_pass':True});a.tensor_file(tmp_path/'controls.safetensors',{'random':torch.ones(2,3)})
    monkeypatch.setattr(a,'stage',lambda *args:{});monkeypatch.setattr(a,'available',lambda s:['profanity']);monkeypatch.setattr(a,'committed_or_wait',lambda *args:None)
    expected=a.evaluation_registration_expected(s)
    assert 'profanity/encodings.json' in expected['files_sha256'] and 'controls_metadata.json' in expected['files_sha256']
    s.cfg['evaluation_registration']=str(tmp_path/'reg.json');put(Path(s.cfg['evaluation_registration']),expected);a.evaluation_lock(s)
    put(tmp_path/'controls_metadata.json',{'tampered':True})
    with pytest.raises(ValueError,match='R2'):a.evaluation_lock(s)
