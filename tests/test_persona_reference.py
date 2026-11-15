"""Offline PV differential, native encoding, branch/gate and journal invariants."""
import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from scripts import persona_reference as s
from scripts import persona_reference_common as p


def cfg():return s.read_json('configs/persona_reference_rudeness.json')
def data():return {k:s.read_json(f'evaluation/persona_reference/rudeness/{k}.json') for k in ('extract','eval')}
def rating(id,trait=70,coherence=90,refusal=False):return {'id':id,'trait':None if refusal else trait,'coherence':coherence,'refusal':refusal,'reason':'synthetic test only'}
def prediction(id='a',text='A useful answer.',ids=None):
    ids=[3,2] if ids is None else ids
    return {'id':id,'output':text,'generated_ids':ids,'input_ids':[4,5],'rendered_prompt':'synthetic',
        'ended_eos':ids[-1]==2,'hit_token_limit':False,'generated_tokens_including_eos':len(ids)}


def archived(name,relative):
    file=s.path('outputs/persona-literature-audit-01/persona_vectors')/relative
    spec=importlib.util.spec_from_file_location(name,file);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def test_author_bytes_profiles_split_and_future_provenance():
    c=cfg();s.validate_config(c);d=data();split=p.split_questions(d)
    assert len(split['selection'])==len(split['report'])==10 and not {r['id'] for r in split['selection']}&{r['id'] for r in split['report']}
    source=s.path('outputs/persona-literature-audit-01/persona_vectors')
    for local,remote in [('extract.json','data_generation/trait_data_extract/impolite.json'),('eval.json','data_generation/trait_data_eval/impolite.json'),('LICENSE','LICENSE')]:
        assert s.path('evaluation/persona_reference/rudeness').joinpath(local).read_bytes()==(source/remote).read_bytes()
    wrong=copy.deepcopy(d);wrong['eval']['instruction'][0]['pos']='Changed'
    with pytest.raises(ValueError):p.split_questions(wrong)
    for key,value in [('batch_size',8),('maximum_judge_calls',129),('max_new_tokens',513),('author_layers',[36])]:
        wrong=copy.deepcopy(c);wrong[key]=value
        with pytest.raises(ValueError):s.validate_config(wrong)
    c['provenance_files']=['results/reviews/future_profile.json'];s.validate_config(c)
    c['provenance_files']=['../other']
    with pytest.raises(ValueError):s.validate_config(c)


def test_plans_exact_partial_distinct_questions_and_branch_budgets():
    plans=p.plans(data(),cfg());sizes={'elicit':[8]*25,'layer':[8]*12+[4],'fallback':[8]*11+[2],
        'strength':[8]*5,'strength_fallback':[8]*3+[6],'final':[8]*5,'custom':[8]*5}
    for phase,plan in plans.items():
        assert list(map(len,plan))==sizes[phase]
        assert all(len({e['question_id'] for e in b})==len(b) for b in plan)
        keys=[(e['question_id'],e['condition']) for b in plan for e in b];aliases=[e['id'] for b in plan for e in b]
        assert len(set(keys))==len(keys)==len(set(aliases))
    assert sum(len(plans[k]) for k in ('elicit','layer','strength','final','custom'))==53
    assert sum(len(plans[k]) for k in ('elicit','layer','fallback','strength_fallback','final','custom'))==64
    assert 'coef1' not in p.conditions('strength_fallback') and 'baseline' not in p.conditions('fallback')
    assert set(p.conditions('strength'))&{'coef2','baseline'}==set()


@pytest.fixture(scope='module')
def tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained('.hf/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c',local_files_only=True,trust_remote_code=False)


def test_actual_cached_tokenizer_all_original_systems_raw_masks(tokenizer):
    rows=p.elicitation(data(),cfg());assert len(rows)==200
    for row in rows:
        prompt=s.render_prompt(tokenizer,row);ids=tokenizer(prompt,add_special_tokens=False)['input_ids']
        answer_ids=tokenizer('A concise response.',add_special_tokens=False)['input_ids']
        item={**prediction(row['id']), 'rendered_prompt':prompt,'input_ids':ids,'generated_ids':answer_ids+[tokenizer.eos_token_id],
            'ended_eos':True,'output':tokenizer.decode(answer_ids,skip_special_tokens=True)}
        encoding=p.response_encoding(tokenizer,item)
        assert encoding['input_ids']==ids+answer_ids+[tokenizer.eos_token_id]
        assert not any(encoding['response_mask'][:len(ids)]) and sum(encoding['response_mask'])==len(answer_ids) and not encoding['response_mask'][-1]
        assert row['system'] in prompt and row['user'] in prompt
    user_only=s.render_prompt(tokenizer,{'id':'u','user':'A small question.'})
    assert '<|im_start|>system' not in user_only
    assert rows[0]['system']!=rows[1]['system']


def test_raw_specials_and_retokenization_crossing_is_only_diagnostic():
    class Tok:
        eos_token_id=9;pad_token_id=0;all_special_ids=[0,9,8]
        def __call__(self,text,**kwargs):
            if text=='xy':return {'input_ids':[4,5]}
            # Artificial merged boundary x|yz: valid native path remains authoritative.
            return {'input_ids':[4,6],'offset_mapping':[(0,1),(1,3)]}
    item={**prediction(),'rendered_prompt':'xy','input_ids':[4,5],'output':'z','generated_ids':[7,8,0,9],'ended_eos':True}
    value=p.response_encoding(Tok(),item)
    assert value['input_ids']==[4,5,7,8,0,9] and value['response_mask']==[False,False,True,False,False,False]
    assert value['nonterminal_special_tokens']==[{'position':1,'token_id':8},{'position':2,'token_id':0}]
    assert value['retokenization_boundary_crossings'] and not value['raw_vs_retokenized_response_equal']
    item['input_ids']=[3]
    with pytest.raises(ValueError):p.response_encoding(Tok(),item)


@pytest.fixture(scope='module')
def tiny():
    from transformers import Qwen3Config,Qwen3ForCausalLM
    torch.set_num_threads(1);torch.manual_seed(42)
    return Qwen3ForCausalLM(Qwen3Config(vocab_size=131,hidden_size=16,intermediate_size=32,num_hidden_layers=3,num_attention_heads=2,num_key_value_heads=1,head_dim=8,pad_token_id=0,eos_token_id=2,max_position_embeddings=128)).eval()


def test_intermediate_hidden_state_mapping_final_norm_and_fp32_raw_author_mean(tiny):
    author=archived('pv_generate_audit','generate_vec.py')
    class Tokens(dict):
        def to(self,device):return Tokens({k:v.to(device) for k,v in self.items()})
    class Tok:
        def encode(self,text,**kw):return [ord(c) for c in text]
        def __call__(self,text,**kw):return Tokens(input_ids=torch.tensor([self.encode(text)]))
    tok=Tok();pos_prompt='ABC';neg_prompt='XYZ';pos='def';neg='ghi';captured={}
    def hook(i):return lambda module,args,out:captured.update({i:out.detach().clone()})
    handles=[layer.register_forward_hook(hook(i)) for i,layer in enumerate(tiny.model.layers)]
    try:
        with torch.inference_mode():result=tiny(input_ids=torch.tensor([[ord(c) for c in pos_prompt+pos]]),output_hidden_states=True)
    finally:
        for h in handles:h.remove()
    assert torch.equal(result.hidden_states[1],captured[0]) and torch.equal(result.hidden_states[2],captured[1])
    assert not torch.equal(result.hidden_states[3],captured[2])
    assert torch.equal(result.hidden_states[3],tiny.model.norm(captured[2]))
    with torch.inference_mode():
        _,_,positive=author.get_hidden_p_and_r(tiny,tok,[pos_prompt],[pos]);_,_,negative=author.get_hidden_p_and_r(tiny,tok,[neg_prompt],[neg])
    tensors,stats=p.raw_vectors([({0:positive[1][0]},{0:negative[1][0]})],[0])
    expected=positive[1].mean(0).float()-negative[1].mean(0).float()
    assert torch.equal(tensors['0.raw'],expected) and stats['0']['author_layer']==1
    assert tensors['0.raw'].dtype==torch.float32 and torch.allclose(tensors['0.random_unit'].norm(),torch.tensor(1.))
    assert not torch.allclose(tensors['0.raw'].norm(),torch.tensor(1.))
    assert p.LAYERS[-1]==35 and max(k-1 for k in p.LAYERS)==34


def test_actual_zero_author_hook_differential_cached_positions_and_cleanup(tiny):
    author=archived('pv_hook_audit','activation_steer.py');inputs={'input_ids':torch.tensor([[3,4,5],[0,6,7]]),'attention_mask':torch.tensor([[1,1,1],[0,1,1]])}
    opts={'do_sample':False,'min_new_tokens':3,'max_new_tokens':3,'use_cache':True};vector=torch.arange(16).float()/20
    with torch.inference_mode():
        original=tiny.generate(**inputs,**opts);audit={}
        with p.response_hook(tiny,1,vector*0,audit):zero=tiny.generate(**inputs,**opts)
        with author.ActivationSteerer(tiny,vector,coeff=1,layer_idx=1,positions='response'):old=tiny.generate(**inputs,**opts)
        nonzero={}
        with p.response_hook(tiny,1,vector,nonzero):new=tiny.generate(**inputs,**opts)
    assert torch.equal(original,zero) and torch.equal(old,new)
    assert audit['prefill_calls']==1 and audit['decode_calls']==2 and audit['hooks_removed']
    assert nonzero['prefill_max_abs_hidden_delta']>0 and nonzero['decode_max_abs_hidden_delta']>0
    with pytest.raises(RuntimeError):
        with p.response_hook(tiny,1,vector):raise RuntimeError('synthetic failure')
    assert not tiny.model.layers[1]._forward_hooks and not tiny._forward_pre_hooks


def test_pair_keys_source_filters_refusal_strict_types_and_quality():
    rows=p.elicitation(data(),cfg());ratings=[rating(r['id'],70 if r['side']=='pos' else 20) for r in rows]
    eligible=p.eligible(rows,list(reversed(ratings)));assert eligible['pair_count']==100 and eligible['question_count']==20 and eligible['sufficient']
    ratings[0]['trait']=50;assert p.eligible(rows,ratings)['pair_count']==100
    ratings[1]['trait']=50;assert p.eligible(rows,ratings)['pair_count']==99
    ratings[1]=rating(rows[1]['id'],refusal=True);assert p.eligible(rows,ratings)['pair_count']==99
    ratings[2]['coherence']=49;assert p.eligible(rows,ratings)['pair_count']==98
    with pytest.raises(ValueError):p.eligible(rows[:-1],ratings[:-1])
    valid=rating('a',refusal=True);p.validate_ratings(['a'],[valid])
    for key,value in [('trait',0),('coherence',None),('refusal',1)]:
        bad={**valid,key:value}
        with pytest.raises(ValueError):p.validate_ratings(['a'],[bad])
    with pytest.raises(s.f.InvalidSubmission):s.parse({'result':'{}'},['a'])
    with pytest.raises(s.f.InvalidSubmission):s.parse({'result':json.dumps({'ratings':[rating('a',trait=True)]})},['a'])


def metric(score=30,coherence=90,**kw):return {'n':10,'conservative_trait_mean':score,'coherence_mean':coherence,'empty':0,'degenerate':0,'capped':0,**kw}


def test_metrics_degenerate_dict_fixed_denominator_and_selection_ties():
    values=[prediction('a'),prediction('b',text='word '*40)];ratings=[rating('a',80),rating('b',refusal=True)]
    result=p.metrics(values,ratings)
    assert result['degenerate']==1 and result['conservative_trait_mean']==40 and result['numeric_trait_mean']==80 and result['refusals']==1
    layers={'baseline':metric(),**{f'layer{k}':metric(60) for k in p.LAYERS}}
    assert p.choose_layer(layers)['author_layer']==4
    layers['layer4']['coherence_mean']=74;assert p.choose_layer(layers)['author_layer']==8
    strengths={'baseline':metric(50),**{f'coef{x:g}':metric(60) for x in (.5,1.,2.,3.,4.)}}
    assert p.choose_strength(strengths)=={'coefficient':.5,'gain':10,'selection_null':False}
    for m in strengths.values():m['conservative_trait_mean']=50
    assert p.choose_strength(strengths)['selection_null']
    assert p.reference_gates({'baseline':metric(10),'selected':metric(40),'random':metric(29),'prompt':metric(50)})['all_criteria_pass']


def test_hash_alias_dedup_conflict_and_old_write_guards(tmp_path,monkeypatch):
    a=tmp_path/'a';a.write_text('source');link=tmp_path/'link';link.symlink_to(a);calls=[];real=s.stream_hash
    monkeypatch.setattr(s,'stream_hash',lambda p:(calls.append(p),real(p))[1])
    s.check_hashes([{str(a):real(a)},{str(link):real(a)}]);assert len(calls)==1
    with pytest.raises(ValueError):s.check_hashes([{str(a):real(a)},{str(link):'bad'}])
    with pytest.raises(ValueError):s.guard(s.path('outputs/rudeness-steering-01'),'rudeness')
    with pytest.raises(ValueError):p.private_export({'reason':'private'})
    with pytest.raises(ValueError):p.private_export({'some':'/home/user/private'})


def setup(tmp_path,monkeypatch):
    run=tmp_path/'new';run.mkdir();c=cfg();d=data();split=p.split_questions(d);plans=p.plans(d,c);rows=p.elicitation(d,c)
    s.save(run/'manifest.json',{'sources':{}})
    for n,v in {**d,'split':split,'plans':plans,'elicit_rows':rows,'rubric':{'output_schema':'reference'},'custom_rubric':{'profanity':'test definition','output_schema':'custom'}}.items():s.save(run/'inputs'/(n+'.json'),v)
    binary=tmp_path/'binary';binary.write_text('synthetic binary never executed');m={'sources':{},'binary':str(binary),'cli_version':'2.1.323 (Claude Code)','audit_sha256':'synthetic'}
    context=SimpleNamespace(run=run,cfg=c,m=m,initial=lambda:None,verify=lambda **kw:None)
    monkeypatch.setattr(s,'BINARY_SHA',s.stream_hash(binary));monkeypatch.setattr(s.shutil,'which',lambda name:str(binary));monkeypatch.setattr(s.f,'version',lambda _:m['cli_version']);monkeypatch.setattr(s.f,'validate_audit',lambda _: {})
    monkeypatch.setattr(s,'permitted',lambda *a:None)
    monkeypatch.setattr(s,'phase_sources',lambda s,phase:{'synthetic_source':'bound'})
    stored={}
    for phase in p.PHASES:
        if phase=='elicit':stored[phase]={'elicited':{'predictions':[prediction(r['id']) for r in rows]}}
        else:stored[phase]={a:{'predictions':[prediction(r['id']) for r in (split['report'] if phase in ('final','custom') else split['selection'])]} for a in p.conditions(phase)}
    monkeypatch.setattr(s,'values',lambda s,phase:stored[phase])
    return context,stored


def mock_calls(monkeypatch,behavior='valid'):
    calls=[];seen={}
    def invoke(binary,b):
        calls.append(copy.deepcopy(b));h=s.f.canonical_hash(b);seen[h]=seen.get(h,0)+1
        if behavior=='uncertain':raise OSError('synthetic transport failure')
        ratings=[rating(r['id']) for r in b['items']]
        if 'profanity' in str(b['rubric']):ratings=[{'id':r['id'],'profanity':0,'rudeness':0,'sarcasm':0,'gross_failure':False,'content_ok':True,'coherent':True} for r in b['items']]
        if behavior=='exhaust' or behavior=='retry' and seen[h]==1:ratings=ratings[:-1]
        result={'subtype':'success','modelUsage':{s.f.MODEL:{}},'result':json.dumps({'ratings':ratings})}
        if behavior=='model':result['modelUsage']={}
        return {'returncode':0,'canonical_stdin_sha256':h,'isolation':{'git_not_repository':True,'git_environment_clean':True,'returncode':128},'stdout':json.dumps(result)}
    monkeypatch.setattr(s.f,'invoke',invoke);return calls


def run_judge(context,phase,resume=False):
    s.judge(SimpleNamespace(phase=phase,resume=resume),context)
    if hasattr(context,'pending_accept'):
        s.f.save_or_match(*context.pending_accept);del context.pending_accept


def test_partial_blind_packets_no_conditions_systems_and_payload_tamper(tmp_path,monkeypatch):
    c,_=setup(tmp_path,monkeypatch);s.packet(SimpleNamespace(phase='layer'),c);body=s.bodies(c,'layer')
    assert len(body)==13 and body[-1]['expected_count']==4
    assert all(set(r)=={'id','question','answer'} for b in body for r in b['items'])
    file=c.run/'judging/layer/packet.json';v=s.read_json(file);v[0][0]['answer']='Tampered';file.write_text(json.dumps(v))
    file=c.run/'judging/layer/manifest.json';v=s.read_json(file);v['packet_sha256']=s.stream_hash(c.run/'judging/layer/packet.json');file.write_text(json.dumps(v))
    with pytest.raises(ValueError):s.bodies(c,'layer')


def test_first_valid_replay_schema_retry_and_receipt_payload_binding(tmp_path,monkeypatch):
    c,_=setup(tmp_path,monkeypatch);calls=mock_calls(monkeypatch,'retry');s.packet(SimpleNamespace(phase='layer'),c);run_judge(c,'layer')
    assert len(calls)==26 and all(calls[i]==calls[i+1] for i in range(0,26,2))
    assert len(s.read_reviews(c,'layer'))==10
    run_judge(c,'layer',True);assert len(calls)==26
    file=c.run/'grades/layer/batch-000-attempt-1-receipt.json';v=s.read_json(file);v['canonical_stdin_sha256']='changed';file.write_text(json.dumps(v))
    with pytest.raises(ValueError):s.read_reviews(c,'layer')


@pytest.mark.parametrize('behavior,count',[('exhaust',2),('uncertain',1),('model',1)])
def test_fatal_schema_identity_uncertain_and_no_extra_calls(tmp_path,monkeypatch,behavior,count):
    c,_=setup(tmp_path,monkeypatch);calls=mock_calls(monkeypatch,behavior);s.packet(SimpleNamespace(phase='elicit'),c)
    with pytest.raises((s.f.InvalidSubmission,s.f.ModelMismatch,OSError)):
        try:run_judge(c,'elicit')
        except BaseException as exc:s.stop(c.run,'live_stage_fatal',exc);raise
    with pytest.raises(RuntimeError):s.finish_grades(c,'elicit',s.bodies(c,'elicit'),c.run/'grades/elicit','bound')
    assert len(calls)==count


def test_resume_absent_and_stray_after_first_valid_zero_extra_calls(tmp_path,monkeypatch):
    c,_=setup(tmp_path,monkeypatch);calls=mock_calls(monkeypatch);s.packet(SimpleNamespace(phase='elicit'),c)
    with pytest.raises(ValueError):run_judge(c,'elicit',True)
    assert not calls and not (c.run/'grades').exists()
    run_judge(c,'elicit');assert len(calls)==25
    s.save(c.run/'grades/elicit/batch-000-attempt-1-start.json',{})
    with pytest.raises(ValueError):s.audit_journals(c)


def test_native_budget_guard_before_new_start_and_existing_valid_resume(tmp_path,monkeypatch):
    c,_=setup(tmp_path,monkeypatch);calls=mock_calls(monkeypatch);s.packet(SimpleNamespace(phase='elicit'),c)
    directory=c.run/'grades/elicit';directory.mkdir(parents=True)
    # Test exactly 106 prior starts elsewhere; no new start marker is allowed.
    for i in range(106):s.save(c.run/'grades/other'/f'batch-{i:03d}-attempt-0-start.json',{})
    with pytest.raises(RuntimeError):s.finish_grades(c,'elicit',s.bodies(c,'elicit'),directory,'binding')
    assert not calls and not list(directory.glob('*start*'))


def test_fallback_trigger_and_reuse_no_duplicate_strength_generation(tmp_path,monkeypatch):
    c,_=setup(tmp_path,monkeypatch)
    primary={'baseline':metric(),**{f'layer{k}':metric(80,coherence=60) for k in p.LAYERS}}
    fallback={f'layer{k}':metric(50) for k in p.LAYERS}
    s.save(c.run/'layer/run.json',{});s.save(c.run/'grades/layer/accepted.json',{})
    s.save(c.run/'fallback/run.json',{});s.save(c.run/'grades/fallback/accepted.json',{})
    monkeypatch.setattr(s,'metric_values',lambda s,phase:primary if phase=='layer' else fallback)
    assert s.layer_decision(c,False)['selection_null']
    chosen=s.layer_decision(c,True);assert chosen['fallback_used'] and chosen['author_layer']==4
    s.save(c.run/'layer_choice.json',chosen);assert s.active_strength(c)=='strength_fallback'
    assert p.conditions(s.active_strength(c))==('coef0.5','coef3','coef4')
    primary['layer4']=metric(60);assert not s.layer_decision(c,True)['fallback_used']


def test_generation_cost_ceiling_before_work_and_public_privacy():
    counts={'saved_generations':500,'teacher_forced_sequences':200,'mc_items':64};zero={k:0 for k in counts}
    s.charge(zero,counts,generations=2)
    for kw in ({'generations':3},{'teacher_forced':1},{'mc':1}):
        with pytest.raises(ValueError):s.charge(zero,counts,**kw)
    p.private_export({'status':'selection_null','costs':{'saved_generations':310},'limits':['descriptive only']})
    for key in ('output','input_ids','response_mask','reason','ratings'):
        with pytest.raises(ValueError):p.private_export({key:[]})


@pytest.mark.parametrize('fallback,total',[(False,106),(True,128)])
def test_complete_branch_call_ceiling_all_first_valid_and_replay(tmp_path,monkeypatch,fallback,total):
    context,_=setup(tmp_path,monkeypatch);calls=mock_calls(monkeypatch,'retry')
    phases=['elicit','layer']+(['fallback','strength_fallback'] if fallback else ['strength'])+['final','custom']
    for phase in phases:
        s.packet(SimpleNamespace(phase=phase),context);run_judge(context,phase);s.read_reviews(context,phase)
    assert len(calls)==total
    run_judge(context,'custom',True);assert len(calls)==total
    s.audit_journals(context)


def test_masked_block_means_match_author_when_token_ids_align(tiny):
    ids=torch.tensor([[65,66,67,68,69,2]]);mask=torch.tensor([[False,False,False,True,True,False]]);captured={}
    with s.capture_response(tiny,[0,1],mask,captured),torch.inference_mode():value=tiny(input_ids=ids,output_hidden_states=True)
    for block in (0,1):assert torch.equal(captured[block]['mean'],value.hidden_states[block+1][0,3:5].float().mean(0))
    assert captured[0]['tokens']==2


def test_fatal_report_cannot_publish_complete_or_accepted(tmp_path,monkeypatch):
    context,_=setup(tmp_path,monkeypatch);s.stop(context.run,'live_stage_fatal',OSError())
    output=tmp_path/'new_report.json';context.cfg['result']=str(output)
    s.report(SimpleNamespace(),context)
    result=s.read_json(output)
    assert result['status']=='incomplete' and result['stop']['stop_cause']=='live_stage_fatal' and result['totals']['saved_generations']==0


@pytest.mark.parametrize('cause',['elicitation_null','selection_null'])
def test_early_null_reports_resource_costs_and_no_private_rows(tmp_path,monkeypatch,cause):
    context,_=setup(tmp_path,monkeypatch);s.stop(context.run,cause)
    s.save(context.run/'elicit/run.json',{'completed':True,'saved_generations':200,'teacher_forced_sequences':0,'mc_items':0,'elapsed_wall_seconds':12,'peak_allocated_bytes':345,'files_sha256':{}})
    output=tmp_path/'new_report.json';context.cfg['result']=str(output)
    if cause=='elicitation_null':
        eligible={'retained_pairs':[],'pair_count':0,'question_count':0,'sufficient':False,'accepted_sha256':'frozen'}
        monkeypatch.setattr(s,'eligibility_result',lambda _:eligible);s.save(context.run/'eligibility.json',eligible)
    else:
        selection={'selection_null':True,'coefficient':None,'metrics':{},'bindings':{'saved':'hash'}}
        monkeypatch.setattr(s,'selection_result',lambda _:selection);s.save(context.run/'selection.json',selection)
    s.report(SimpleNamespace(),context);value=s.read_json(output)
    assert value['status']==cause and value['totals']['saved_generations']==200 and value['costs']['elicit']['peak_allocated_bytes']==345
    p.private_export(value)


def saved_generation():
    row=prediction();row.update(batch_index=0,batch_position=0,batch_ids=['a'],padded_prompt_width=2)
    return {'predictions':[row],'batches':[{'batch_index':0,'ids':['a'],'padded_prompt_width':2}],
        'settings':{'batch_size':4,'max_new_tokens':512,'seed':42,'do_sample':False}}


@pytest.mark.parametrize('key,value',[('ended_eos',False),('ended_eos',1),('hit_token_limit',True),('generated_tokens_including_eos',3),('batch_position',1),('batch_ids',['other']),('padded_prompt_width',3)])
def test_actual_saved_generation_flags_and_batch_membership(key,value):
    stored=saved_generation();rendered={'a':{'input_ids':[4,5],'rendered_prompt':'synthetic'}}
    p.validate_generation(stored,[{'id':'a'}],rendered,2)
    stored['predictions'][0][key]=value
    with pytest.raises(ValueError):p.validate_generation(stored,[{'id':'a'}],rendered,2)


def test_cap_without_eos_preserves_natural_pad_and_model_order():
    stored=saved_generation();r=stored['predictions'][0];r.update(generated_ids=[0]*512,generated_tokens_including_eos=512,ended_eos=False,hit_token_limit=True)
    rendered={'a':{'input_ids':[4,5],'rendered_prompt':'synthetic'}};p.validate_generation(stored,[{'id':'a'}],rendered,2)
    r['generated_ids']=[3,2,4];r.update(generated_tokens_including_eos=3,ended_eos=True,hit_token_limit=False)
    with pytest.raises(ValueError):p.validate_generation(stored,[{'id':'a'}],rendered,2)


def test_profile_report_descriptions_are_not_hardcoded_rudeness(tmp_path,monkeypatch):
    context,_=setup(tmp_path,monkeypatch);context.cfg.update(trait_name='test_profanity_author',target_trait='profanity',result=str(tmp_path/'new.json'))
    s.stop(context.run,'elicitation_null');s.report(SimpleNamespace(),context)
    result=s.read_json(tmp_path/'new.json');assert 'test_profanity_author' in result['scope'] and 'custom profanity' in result['scope']
    assert 'impolite' not in json.dumps(result) and 'custom rudeness' not in json.dumps(result)


def test_retention_return_schema_fresh_both_and_language_seed_restored(tmp_path,monkeypatch):
    from contextlib import contextmanager,nullcontext
    from langdetect import DetectorFactory
    context,_=setup(tmp_path,monkeypatch);s.save(context.run/'inputs/retention.json',{'general_mc':[],'physics_mc':[],'gsm8k':[],'ifeval':[]})
    monkeypatch.setattr(s,'selected',lambda *a,**kw:{'block':3,'coefficient':.5});monkeypatch.setattr(s,'vectors',lambda _:{'3.raw':torch.ones(16)})
    monkeypatch.setattr(s,'metric_values',lambda *a:{'baseline':metric(10),'selected':metric(40),'random':metric(20),'prompt':metric(50)})
    monkeypatch.setattr(s.p,'response_hook',lambda *a,**kw:nullcontext())
    progress={'saved_generations':0,'teacher_forced_sequences':0,'mc_items':0,'generation_wall_seconds':0};existing={k:0 for k in ('saved_generations','teacher_forced_sequences','mc_items')}
    @contextmanager
    def stage(*a):
        out=context.run/'retention';out.mkdir();yield None,None,out,progress,existing
    monkeypatch.setattr(s,'gpu_stage',stage);calls=[]
    def screen(model,tok,fixture,directory,**kw):
        assert DetectorFactory.seed==0 and kw['tasks']==['general_mc','physics_mc','gsm8k','ifeval'];calls.append(directory.name)
        kw['diagnostics'].append({'hf_generation_timing':{'generation_wall_seconds':1}})
        return {'general_mc':{'total':16,'correct':11},'physics_mc':{'total':16,'correct':8},'gsm8k':{'total':8,'strict_correct':7},'ifeval':{'total':8,'prompt_strict_correct':6}}
    monkeypatch.setattr(s,'run_screen',screen);s.save(context.run/'selection.json',{'test':'binding'})
    old=DetectorFactory.seed;DetectorFactory.seed=321
    try:
        s.retention(SimpleNamespace(),context);assert DetectorFactory.seed==321
    finally:DetectorFactory.seed=old
    assert calls==['baseline','selected'] and progress['saved_generations']==32 and progress['mc_items']==64
    result=s.read_json(context.run/'retention/metrics.json');assert result['selected']['gsm8k']['strict_correct']==7 and result['baseline']['ifeval']['prompt_strict_correct']==6
