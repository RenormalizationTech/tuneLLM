import ast
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from scripts import profanity_method_transfer as a
from scripts.common import path,read_json


@pytest.fixture(autouse=True)
def one_thread():
    old=torch.get_num_threads();torch.set_num_threads(1)
    try:yield
    finally:torch.set_num_threads(old)


def cfg():return read_json(a.CONFIG)


def tiny():
    from transformers import Qwen2Config,Qwen2ForCausalLM
    torch.manual_seed(14)
    return Qwen2ForCausalLM(Qwen2Config(vocab_size=32,hidden_size=16,intermediate_size=24,
        num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2,max_position_embeddings=256,
        pad_token_id=0,bos_token_id=1,eos_token_id=2,attn_implementation='eager')).eval()


class Batch(dict):
    def to(self,device):return Batch({k:v.to(device) for k,v in self.items()})


class Tok:
    def encode(self,text,add_special_tokens=False):return [3+ord(c)%29 for c in text]
    def __call__(self,text,return_tensors=None,add_special_tokens=False):
        ids=self.encode(text)
        return Batch(input_ids=torch.tensor([ids]),attention_mask=torch.ones(1,len(ids),dtype=torch.long)) if return_tensors else {'input_ids':ids}


def test_actual_ast_source_delta_key_demand_with_tiny_eager_boundaries():
    """Execute the real isolated author definitions and independently reconstruct."""
    from transformers.models.qwen2.modeling_qwen2 import repeat_kv,apply_rotary_pos_emb
    root=path(cfg()['author_root']);functions=a.p.extraction_functions(root/'core/generate_vec.py',root/'scripts/generate_vec_prompt_attn_k.py')
    model=tiny();tok=Tok();prompts=['<|im_start|>system\ns<|im_end|>\nu','<|im_start|>system\nlong<|im_end|>\nu'];answers=['abcd','ef']
    expected=[[] for _ in range(2)];ks=[[] for _ in range(2)];qs=[[] for _ in range(2)]
    for prompt,answer in zip(prompts,answers):
        inputs=tok(prompt+answer,return_tensors='pt');syslen=len(tok.encode(prompt[:prompt.index('<|im_end|>\n')+len('<|im_end|>\n')]));plen=len(tok.encode(prompt));capt={};handles=[]
        for layer in range(2):
            for name in ('q','k','v'):
                def collect(module,args,output,layer=layer,name=name):capt[layer,name]=output.detach()
                handles.append(getattr(model.model.layers[layer].self_attn,name+'_proj').register_forward_hook(collect))
        try:
            with torch.no_grad():value=model(**inputs,output_attentions=True,use_cache=False)
        finally:
            for h in handles:h.remove()
        for l in range(2):
            length=inputs['input_ids'].shape[1];v=capt[l,'v'].view(1,length,2,4).transpose(1,2)
            sysvalue=repeat_kv(v,2)[:,:,:syslen,:];weights=value.attentions[l][:,:,:,:syslen]
            projected=model.model.layers[l].self_attn.o_proj((weights@sysvalue).transpose(1,2).reshape(1,length,16)).detach()
            expected[l].append(projected[:,plen:,:].mean(1))
            q=capt[l,'q'].view(1,length,4,4).transpose(1,2);k=capt[l,'k'].view(1,length,2,4).transpose(1,2)
            cos,sin=model.model.rotary_emb(k,torch.arange(length)[None]);qr,kr=apply_rotary_pos_emb(q,k,cos,sin)
            ks[l].append(kr[0,:,:syslen,:].float());qs[l].append(qr[0,:,plen:,:].float())
    before=a.native.hook_signature(model)
    _,_,actual=functions['get_prompt_attn_delta_p_and_r'](model,tok,prompts,answers)
    key,demand=functions['compute_k_pos_and_avg_demand'](model,tok,prompts,answers)
    for layer in range(2):
        assert torch.equal(actual[layer],torch.cat(expected[layer]))
        expected_k=torch.cat(ks[layer],dim=1).sum(1)/sum(v.shape[1] for v in ks[layer])
        expected_q=torch.cat(qs[layer],dim=1).sum(1)/sum(v.shape[1] for v in qs[layer])
        assert torch.allclose(key[layer],expected_k,atol=1e-7,rtol=1e-6)
        expected_demand=(expected_q*expected_k.repeat_interleave(2,dim=0)).sum(-1).mean()/2
        assert torch.allclose(demand[layer],expected_demand,atol=1e-7,rtol=1e-6)
    assert a.native.hook_signature(model)==before
    # Loader compiles exactly the two original function ASTs, no pipeline setup.
    for file,name in ((root/'core/generate_vec.py','get_prompt_attn_delta_p_and_r'),(root/'scripts/generate_vec_prompt_attn_k.py','compute_k_pos_and_avg_demand')):
        assert functions[name].__code__.co_filename==str(file)


def test_accepted_inputs_system_identity_overlap_and_full_frozen_plans():
    c=cfg();a.validate_config(c);data=a.input_data(c)
    assert len(data['train'])==20 and not data['dropped'] and len(data['calibration'])==len(data['heldout'])==32
    assert data['systems'][0]['pos'].startswith('You are a helpful and profane assistant.')
    assert data['systems'][0]['neg'].startswith('You are a helpful and nonprofane assistant.')
    plans=a.p.frozen_plans(data,c)
    for phase,total in [('elicitation',200),('development',320),('heldout',224)]:
        for metric in ('trait','coherence'):
            batches=plans[phase][metric]
            assert sum(map(len,batches))==total and all(len(b)==8 and len({r['question_id'] for r in b})==8 for b in batches)
            assert len({r['id'] for b in batches for r in b})==total
    assert sum(len(b) for v in plans.values() for b in v.values())==186
    assert a.LIMITS=={'generations':760,'prompt_forwards':300,'gcad_teacher_forced':300,'logit_forwards':2}


def test_only_gcad_eligibility_null_and_projection_development_not_blocked():
    records=[{'id':side,'question_id':'q','instruction_index':0,'side':side} for side in ('pos','neg')]
    assert a.p.eligible_pairs(records,{'pos':{'trait':49,'coherence':100},'neg':{'trait':0,'coherence':100}})==[]
    settings=a.p.development_settings(cfg(),False)
    assert set(settings)=={'baseline','fixed_1','fixed_2','fixed_4','saps_1','saps_2','saps_4'}
    plan=a.p.frozen_plans(a.input_data(cfg()),cfg())['development']['trait'];active=a.p.active_plan(plan,settings)
    assert sum(map(len,active))==224 and all(len(b)<=8 and len({r['question_id'] for r in b})==len(b) for b in active)


def row(trait=0,coherence=80,hits=0,invalid=0):return {'trait_conservative':trait,'coherence':coherence,'lexicon_hits':hits,'invalid':invalid}


def test_exact_gates_lowest_strength_refusal_zero_and_final_tie():
    settings=a.p.development_settings(cfg(),True);metrics={k:row() for k in settings}
    metrics['fixed_1']=row(20,60,8,3);metrics['fixed_2']=row(40,80,10);metrics['gcad_3.5']=row(30,49,10)
    chosen=a.p.choose(metrics,settings);assert chosen['selected']=={'fixed':'fixed_1'}
    metrics['baseline']=row(0,80,1);assert not a.p.choose(metrics,settings)['selected'].get('fixed')=='fixed_1'
    final={'baseline':row(),**{prefix+m:row(30 if prefix=='selected_' else 10,80,8 if prefix=='selected_' else 0) for m in ('fixed','saps','gcad') for prefix in ('selected_','random_')}}
    decision=a.p.final_decision(final,['saps','gcad','fixed']);assert decision['passing']==['fixed','saps','gcad']
    records=[{'id':str(i),'output':'Damn useful response.','generated_ids':[3,2],'ended_eos':True,'hit_token_limit':False} for i in range(32)]
    grades={str(i):{'trait':None if i==0 else 32,'coherence':80} for i in range(32)}
    value=a.p.summarize(records,grades);assert value['trait_conservative']==31 and value['trait_numeric_mean']==32 and value['refusals']==1


def test_position_weighted_realized_norm_not_frobenius_and_cleanup():
    model=SimpleNamespace(model=SimpleNamespace(layers=[torch.nn.Identity()]),device=torch.device('cpu'),parameters=lambda:iter([torch.ones(1)]));audit={}
    unit=torch.tensor([1.,0.])
    context=a.p.constant_shift(model,0,unit,2.)
    with a.p.observe_shift(model,{'0':model.model.layers[0]},context,audit):
        model.model.layers[0](torch.zeros(1,4,2));model.model.layers[0](torch.zeros(1,1,2))
    assert audit['0']['positions']==5 and audit['0']['norm_sum']==10 and audit['0']['realized_position_mean_norm']==2
    assert a.p.position_weighted_norm([{'shift_audit':audit}])==2
    assert not model.model.layers[0]._forward_hooks
    with pytest.raises(RuntimeError):
        with a.p.observe_shift(model,{'0':model.model.layers[0]},a.p.constant_shift(model,0,unit,2.),{}):raise RuntimeError('injected')
    assert not model.model.layers[0]._forward_hooks


def test_controls_dedicated_rng_seeds_per_tensor_metadata_and_tamper(tmp_path):
    s=SimpleNamespace(run=tmp_path,cfg=cfg());a.save(tmp_path/'inputs/data.json',{'heldout':[{'id':f'H{i:02d}'} for i in range(32)]})
    choice={'selected':{'fixed':'fixed_1','saps':'saps_2','gcad':'gcad_3.5'},'selected_development_norms':{'fixed':2.,'saps':3.,'gcad':4.}}
    value={'gcad':{'tensors':{'delta':torch.arange(28*3584).reshape(28,3584).float()/99999}}}
    before=torch.get_rng_state().clone();controls=a.random_controls(s,choice,value)
    assert torch.equal(before,torch.get_rng_state()) and controls['H00']['fixed']['seed']==8603 and controls['H03']['fixed']['seed']==8603
    assert torch.equal(controls['H00']['fixed']['unit'],controls['H00']['saps']['unit'])
    for layer in s.cfg['gcad_layers']:
        assert controls['H00']['gcad']['layer_seeds'][str(layer)]==8603+1000*layer
        assert torch.allclose(controls['H00']['gcad']['delta'][layer].norm(),value['gcad']['tensors']['delta'][layer].norm())
    torch.save(controls,tmp_path/'controls.pt');a.save(tmp_path/'controls_metadata.json',a.controls_metadata(s,controls))
    assert a.validate_controls(s,choice,value)['H00']['fixed']['norm']==2
    metadata=read_json(tmp_path/'controls_metadata.json');metadata['items']['H00']['fixed']['unit_sha256']='0'*64
    (tmp_path/'controls_metadata.json').unlink();a.save(tmp_path/'controls_metadata.json',metadata)
    with pytest.raises(ValueError,match='metadata'):a.validate_controls(s,choice,value)


def test_reload_wait_bounded_and_midphase_failure_fatal(tmp_path,monkeypatch):
    attempts=[];sleeps=[]
    def resource(c):
        attempts.append(1)
        if len(attempts)<3:raise a.infra.ResourceWait('settling')
        return {'ok':True}
    monkeypatch.setattr(a.infra,'resource',resource);monkeypatch.setattr(a.time,'sleep',sleeps.append)
    assert a.wait_for_reload({},attempts=3)=={'ok':True} and len(attempts)==3 and sleeps==[1,1]
    monkeypatch.setattr(a.infra,'resource',lambda c:(_ for _ in ()).throw(a.infra.ResourceWait('busy')))
    with pytest.raises(RuntimeError,match='completed prompt work'):a.wait_for_reload({},attempts=2)
    # RuntimeError after started work reaches permanent-stop branch, not no-call wait.
    run=tmp_path;s=SimpleNamespace(run=run,verify=lambda **kw:None)
    monkeypatch.setattr(a,'Context',lambda:s);monkeypatch.setattr(a,'prerequisites',lambda *args:None)
    monkeypatch.setattr(a,'gpu',lambda *args:(_ for _ in ()).throw(RuntimeError('partial reload failure')))
    monkeypatch.setattr('sys.argv',['runner','extraction','--execute-gpu'])
    with pytest.raises(RuntimeError):a.main()
    assert read_json(run/'stopped.json')['stop_cause']=='fatal_execution_or_integrity'


def attempt(rows,model='claude-opus-5-5',returncode=0):
    return {'returncode':returncode,'isolation':{'git_not_repository':True,'git_environment_clean':True,'returncode':128},
        'stdout':json.dumps({'result':json.dumps({'ratings':rows}),'modelUsage':{model:{}},'is_error':False})}


def test_strict_separate_trait_coherence_model_auth_no_semantic_retry():
    payload={'expected_ids':['x']};rating={'id':'x','score':None,'refusal':True,'reason':'private'}
    assert a.parse(attempt([rating]),payload,'trait')[0]['x']['score'] is None
    with pytest.raises(a.StructuralError):a.parse(attempt([rating]),payload,'coherence')
    for bad in [dict(rating,score=True,refusal=False),dict(rating,score=101,refusal=False)]:
        with pytest.raises(a.StructuralError):a.parse(attempt([bad]),payload,'trait')
    with pytest.raises(a.f.ModelMismatch):a.parse(attempt([rating],model='other'),payload,'trait')
    with pytest.raises(RuntimeError):a.parse(attempt([rating],returncode=1),payload,'trait')


def test_forever_native_budget_stray_attempt_and_uncertain_resume(tmp_path,monkeypatch):
    s=SimpleNamespace(run=tmp_path);directory=tmp_path/'grades/development/trait';directory.mkdir(parents=True)
    a.save(directory/'run_binding.json',{});monkeypatch.setattr(a,'grade_binding',lambda *args:{})
    monkeypatch.setattr(a,'payloads',lambda *args:[{'expected_ids':['x']}])
    a.save(directory/'batch-000-attempt-0-start.json',{})
    with pytest.raises(a.f.UncertainAttempt):a.read_grades(s,'development','trait',True)
    (directory/'batch-000-attempt-0-start.json').unlink();a.save(directory/'batch-000-attempt-2.json',{})
    with pytest.raises(ValueError,match='unexpected'):a.read_grades(s,'development','trait',True)
    (directory/'batch-000-attempt-2.json').unlink()
    for i in range(373):a.save(directory/f'extra-{i}-start.json',{})
    with pytest.raises(ValueError,match='ceiling'):a.read_grades(s,'development','trait',True)


def test_initial_mechanics_selection_locks_exact_keys(tmp_path,monkeypatch):
    c=cfg();s=SimpleNamespace(run=tmp_path,cfg=c);committed=[]
    monkeypatch.setattr(a.infra,'committed',committed.append);monkeypatch.setattr(a,'stage',lambda *args:{})
    monkeypatch.setattr(a,'selection',lambda s:{'selection_null':False})
    paths=['manifest.json','extraction/run.json','extraction/integrity.json','extraction/banks.pt','mechanics/run.json','mechanics/integrity.json','mechanics/result.json',
           'selection.json','development/run.json','grades/development/trait/accepted.json','grades/development/coherence/accepted.json','controls.pt','controls_metadata.json']
    for name in paths:a.save(tmp_path/name,{'all_pass':True} if name=='mechanics/result.json' else ({'selection_null':False} if name=='selection.json' else {}))
    # Use private temporary registration destinations, never historical files.
    for kind in ('mechanics','selection'):
        c[kind+'_registration']=str(tmp_path/(kind+'-registration.json'))
    files={'extraction_run_sha256':'extraction/run.json','extraction_integrity_sha256':'extraction/integrity.json','banks_sha256':'extraction/banks.pt',
           'mechanics_run_sha256':'mechanics/run.json','mechanics_integrity_sha256':'mechanics/integrity.json','mechanics_result_sha256':'mechanics/result.json'}
    a.save(c['mechanics_registration'],{'manifest_sha256':a.stream_hash(tmp_path/'manifest.json'),**{k:a.stream_hash(tmp_path/n) for k,n in files.items()}})
    a.lock(s,'mechanics')
    files={'selection_sha256':'selection.json','development_run_sha256':'development/run.json','trait_accepted_sha256':'grades/development/trait/accepted.json',
           'coherence_accepted_sha256':'grades/development/coherence/accepted.json','controls_sha256':'controls.pt','controls_metadata_sha256':'controls_metadata.json'}
    a.save(c['selection_registration'],{'manifest_sha256':a.stream_hash(tmp_path/'manifest.json'),**{k:a.stream_hash(tmp_path/n) for k,n in files.items()}})
    a.lock(s,'selection')
    with (tmp_path/'controls_metadata.json').open('a') as f_:f_.write(' ')
    with pytest.raises(ValueError,match='lock'):a.lock(s,'selection')
    s.m={'sources':{}};a.Context.registered(s) if False else None


def test_new_only_guards_and_harmless_no_execution(monkeypatch,capsys):
    with pytest.raises(ValueError):a.guard(path('outputs/author-profanity-direction-test-01'))
    c=cfg();c['target_cap']=48
    with pytest.raises(ValueError):a.validate_config(c)
    monkeypatch.setattr(a,'Context',lambda:pytest.fail('guard-free CLI must not load a context'))
    monkeypatch.setattr('sys.argv',['runner','development']);a.main();assert 'No execution' in capsys.readouterr().out
