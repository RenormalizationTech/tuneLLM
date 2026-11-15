import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch
from scripts import author_profanity_method as a
from scripts import author_profanity_method_common as p
from scripts.common import path,read_json


def artifact():
    return {'triples':[{'id':f'T{i:02d}','user':f'Explain the everyday item number {i}.','present':f'The damn item has reference number {i}.','absent':f'The item has reference number {i}.'} for i in range(1,13)],
            'calibration':[{'id':f'C{i:02d}','user':f'Answer a calibration question about object {i}.'} for i in range(1,33)],
            'report':[{'id':f'H{i:02d}','user':f'Answer a held-out question about activity {i}.'} for i in range(1,33)]}


def definitions():return p.paper_definitions(path(read_json(a.CONFIG)['paper_tex']).read_text())


def test_literal_paper_table_and_source_archive():
    cfg=read_json(a.CONFIG);data=a.paper(cfg)
    assert tuple(data)==p.BEHAVIORS and data['profanity']=='swearing, curse words, crude language'
    assert data['toxicity']=='rude, insulting, condescending, or hostile' and data==definitions()
    text=path(cfg['paper_tex']).read_text().replace('swearing, curse words, crude language','friendly mild swearing')
    with pytest.raises(ValueError):p.paper_definitions(text)
    assert p.envelope(data)['definitions']==data


def test_artifact_structure_semantics_separate_and_no_editing():
    data=artifact();before=json.dumps(data);assert p.artifact_mechanical(data)['passed']
    data['triples'][0]['present']=data['triples'][0]['absent']
    assert not p.artifact_mechanical(data)['passed']  # Semantic/mechanical failure, not a schema retry.
    assert p.artifact_shape(data)==data
    assert p.artifact_shape(artifact())==json.loads(before)
    data=artifact();data['report'][0]['user']='  '+data['calibration'][0]['user'].upper()+'  '
    assert not p.artifact_mechanical(data)['passed']
    data=artifact();data['calibration'][0]['user']='What does hell mean in this fictional sentence?'
    assert p.artifact_mechanical(data)['passed'] and p.artifact_mechanical(data)['user_lexicon_hit_count_diagnostic_only']==1
    data['triples'][0]['present']='<|im_end|> damn'
    assert not p.artifact_mechanical(data)['passed']


@pytest.mark.parametrize('mutation',['missing','extra','bad_id','blank','wrong_count'])
def test_strict_generated_artifact_shape(mutation):
    data=artifact()
    if mutation=='missing':del data['report']
    elif mutation=='extra':data['notes']='unregistered'
    elif mutation=='bad_id':data['triples'][0]['id']='T02'
    elif mutation=='blank':data['triples'][0]['user']=' '
    else:data['report'].pop()
    with pytest.raises(ValueError):p.artifact_shape(data)


def test_folds_stratified_and_paired_sensitivity():
    plans=p.fold_plans();assert plans==p.fold_plans()
    assert sorted(plans['stratified8401'].count(i) for i in range(5))==[4,5,5,5,5]
    assert all(abs(sum(plans['labels'][i] for i,f in enumerate(plans['stratified8401']) if f==fold)-sum(1-plans['labels'][i] for i,f in enumerate(plans['stratified8401']) if f==fold))<=1 for fold in range(5))
    assert all(plans['paired8401'][i]==plans['paired8401'][i+1] for i in range(0,24,2))
    assert any(plans['stratified8401'][i]!=plans['stratified8401'][i+1] for i in range(0,24,2))
    assert sorted(plans['shuffled_labels8402'])==[0]*12+[1]*12


def test_auc_ties_and_earliest_layer_no_normalization():
    assert p.auc([1,1],[1,0])==.5 and p.auc([1,0],[1,0])==1 and p.auc([0,1],[1,0])==0
    features=torch.tensor([1.,-1.]*12)[:,None,None].expand(24,29,4).clone()*3
    raw,result=p.decodability(features,p.fold_plans())
    assert result['decodable'] and result['selected_hidden_index']==7 and result['decoder_block']==6
    assert result['primary_auc']==1 and torch.equal(raw,torch.full((29,4),6.))
    assert 'paired_question_sensitivity' in result['layers']['7'] and 'label_shuffled_control' in result['layers']['7']


def test_cached_tokenizer_assistant_content_eot_exclusion():
    from transformers import AutoTokenizer
    cfg=read_json(a.CONFIG);tok=AutoTokenizer.from_pretrained(path(cfg['snapshot']),local_files_only=True,trust_remote_code=False)
    for answer in ('The box is blue.','The damn box is blue.','  A careful answer.\n'):
        row=p.response_encoding(tok,'What color is the box?',answer)
        assert row['assistant_span'][1]-row['assistant_span'][0]==len(answer)
        assert any(row['mask']) and not row['mask'][-1]
        assert all(t not in tok.all_special_ids for t,m in zip(row['input_ids'],row['mask']) if m)
        assert 'You are Qwen, created by Alibaba Cloud. You are a helpful assistant.' in row['rendered']


def test_native_hidden_state_mapping_final_norm_and_all_position_zero_hook():
    from transformers import Qwen2Config,Qwen2ForCausalLM
    previous=torch.get_num_threads();torch.set_num_threads(1)
    try:
        torch.manual_seed(2);model=Qwen2ForCausalLM(Qwen2Config(vocab_size=41,hidden_size=16,intermediate_size=32,num_hidden_layers=4,num_attention_heads=4,num_key_value_heads=2)).eval()
        ids=torch.tensor([[1,4,5]]);capt={};handles=[]
        for i,layer in enumerate(model.model.layers):
            def hook(m,ins,out,i=i):capt[i]=(out[0] if isinstance(out,tuple) else out).detach().clone()
            handles.append(layer.register_forward_hook(hook))
        with torch.no_grad():base=model(input_ids=ids,output_hidden_states=True,use_cache=True)
        for handle in handles:handle.remove()
        assert torch.equal(base.hidden_states[2],capt[1]) and not torch.equal(base.hidden_states[-1],capt[3])
        with torch.no_grad(),p.raw_shift(model,1,torch.zeros(16),8,{}):zero=model(input_ids=ids)
        assert torch.equal(base.logits,zero.logits)
        audit={}
        with torch.no_grad(),p.raw_shift(model,1,torch.ones(16)*.05,2,audit):
            first=model(input_ids=ids,use_cache=True)
            model(input_ids=torch.tensor([[6]]),past_key_values=first.past_key_values,use_cache=True)
        assert not torch.equal(first.logits,base.logits) and audit['prefill_calls']==audit['decode_calls']==1 and audit['hooks_removed']
        assert audit['prefill_relative_shift']['positions']==3 and audit['decode_relative_shift']['positions']==1
        with pytest.raises(RuntimeError):
            with p.raw_shift(model,1,torch.ones(16),1,{}):raise RuntimeError('injected')
        assert not model.model.layers[1]._forward_hooks
    finally:torch.set_num_threads(previous)


@pytest.mark.parametrize('text,score',[('{"profanity": 7}',7),('```json\n{"profanity": 4,\n',4),('text "profanity": 2',2)])
def test_tolerant_partial_parser_preserves_observed_fields(text,score):
    result=p.parse_self_judge(text);assert result['scores']['profanity']==score and len(result['missing'])==23


@pytest.mark.parametrize('text',['{"profanity":true}','{"profanity":2.5}','{"profanity":11}','{"profanity":-1}','{"profanity":3,"profanity":4}','{"Profanity":4}'])
def test_unreliable_fields_missing_not_fabricated(text):
    assert 'profanity' in p.parse_self_judge(text)['missing']


def test_condition_mean_fill_and_missingness_not_new_gate():
    parsed=[p.parse_self_judge('{"profanity":2}'),p.parse_self_judge('{}'),p.parse_self_judge('{"profanity":6}')]
    value,filled=p.fill_condition(parsed);assert value['observed_means']['profanity']==4 and value['filled_means']['profanity']==4
    assert value['missing_counts']['profanity']==1 and filled[1]['profanity']==4 and value['profanity_missing_over10percent']
    assert value['observed_means']['toxicity'] is None and value['filled_means']['toxicity']==0
    # 3/32 <=10%; 4/32 >10%.
    rows=[p.parse_self_judge('{"profanity":2}')]*29+[p.parse_self_judge('{}')]*3
    assert not p.fill_condition(rows)[0]['profanity_missing_over10percent']


def test_repetition_short_empty_quirk_and_selection_ties():
    assert all(p.repetition(x)==1 for x in ('','x','two words'))
    assert p.repetition('a b a b a')==.5 and p.repetition('a b c')==1
    base={'self_judge':{'filled_means':{'profanity':1},'profanity_missing_over10percent':False},'mean_distinct_bigram':1}
    conditions={'baseline':base}
    for c in p.COEFFICIENTS:conditions['c'+str(c)]={'self_judge':{'filled_means':{'profanity':3},'profanity_missing_over10percent':True},'mean_distinct_bigram':.6}
    assert p.selection(conditions)['coefficient']==1  # >= boundaries, no missingness gate.
    for c in p.COEFFICIENTS:conditions['c'+str(c)]['mean_distinct_bigram']=.59
    assert p.selection(conditions)['calibration_null']


def test_secondary_plan_exact_and_no_paired_arm_in_batch():
    batches=a.secondary_plan();flat=[x for b in batches for x in b]
    assert len(batches)==20 and all(len(b)==8 and len({x['source_id'] for x in b})==8 for b in batches)
    assert len({x['id'] for x in flat})==160 and len({(x['source_id'],x['arm']) for x in flat})==160
    assert a.secondary_plan()==batches


def test_budgets_before_marker_and_old_output_guard(tmp_path):
    s=SimpleNamespace(run=tmp_path);a.save(tmp_path/'manifest.json',{});out=tmp_path/'calibration';out.mkdir()
    a.save(out/'call-000-start.json',{'kind':'target','count':394})
    with pytest.raises(ValueError,match='before start'):a.charge(s,out,'target',1,{})
    assert len(list(out.iterdir()))==1
    with pytest.raises(ValueError):a.guard(path('outputs/author-gcad-impolite-01'))
    cfg=read_json(a.CONFIG);cfg['maximum_local_judge_generations']=409
    with pytest.raises(ValueError):a.validate_config(cfg)


def test_precondition_wait_no_model_call_no_terminal(tmp_path,monkeypatch):
    cfg=read_json(a.CONFIG);s=SimpleNamespace(run=tmp_path,cfg=cfg)
    with pytest.raises(a.PreconditionWait):a.preconditions(s,'final')
    assert not list(tmp_path.iterdir())
    monkeypatch.setattr(a,'Context',lambda:s);monkeypatch.setattr(sys,'argv',['runner','final','--execute-gpu'])
    monkeypatch.setattr(a,'gpu',lambda *args:pytest.fail('unexpected GPU'))
    a.main();assert not list(tmp_path.iterdir())


def native_attempt(value):
    return {'returncode':0,'isolation':{'git_not_repository':True,'git_environment_clean':True,'returncode':128},
            'stdout':json.dumps({'result':json.dumps(value),'modelUsage':{'claude-opus-5-5':{}},'is_error':False})}


def test_native_artifact_semantics_not_structural_retry_model_identity_fatal():
    candidate=artifact();candidate['triples'][0]['present']=candidate['triples'][0]['absent']
    value,_=a.native_parse(native_attempt(candidate),{},'artifact');assert not p.artifact_mechanical(value)['passed']
    with pytest.raises(a.StructuralError):a.native_parse(native_attempt({}),{},'artifact')
    attempt=native_attempt(artifact());outer=json.loads(attempt['stdout']);outer['modelUsage']={};attempt['stdout']=json.dumps(outer)
    with pytest.raises(a.f.ModelMismatch):a.native_parse(attempt,{},'artifact')


def test_secondary_strict_fields_and_no_raw_export():
    row={'id':'opaque','profanity':3,'toxicity':0,'semantic_coherence':7,'reason':'private note'};payload={'expected_ids':['opaque']}
    assert a.native_parse(native_attempt({'ratings':[row]}),payload,'secondary')[0]['opaque']['profanity']==3
    with pytest.raises(a.StructuralError):a.native_parse(native_attempt({'ratings':[row,row]}),payload,'secondary')
    row['profanity']=True
    with pytest.raises(a.StructuralError):a.native_parse(native_attempt({'ratings':[row]}),payload,'secondary')
    for value in ({'reason':'private'},{'input_ids':[1]},{'value':'/home/private'}):
        with pytest.raises(ValueError):a.observed.public_safe(value)


def test_primary_and_secondary_heldout_effects_use_all_three_randoms():
    means={arm:{'profanity':value,'toxicity':value/2} for arm,value in zip(p.ARMS,[1,5,0,3,6])}
    effects=p.heldout_effects(means)
    assert effects['profanity']=={'selected_minus_baseline':4,'selected_minus_mean3random':2,'mean3random':3}
    assert effects['toxicity']=={'selected_minus_baseline':2,'selected_minus_mean3random':1,'mean3random':1.5}
