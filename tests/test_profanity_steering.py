"""Focused CPU checks, cached tokenizer/tiny random model, mocked remote calls."""
import copy
from collections import Counter
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from scripts import profanity_common as p
from scripts import profanity_steering as s


def cfg():return s.read_json('configs/profanity_steering.json')
def data():return {n:s.read_json(f'evaluation/profanity_steering/{n}.json') for n in ('extraction','development','confirmation')}
def review(id,hit=False,factual=False):return {'id':id,'profanity':2 if hit else 0,'rudeness':0,'sarcasm':0,'content_ok':True,'coherent':True,'factual_correct':factual,'gross_failure':False}


def test_fixture_family_diversity_phrases_positions_and_unchanged_rubric():
    fixtures=data();p.validate_fixtures(fixtures,cfg()['lexicon']);stats=p.position_phrase_stats(fixtures['extraction']['items'],cfg()['lexicon'])
    assert stats['position_bins']=={'opening':10,'middle':11,'closing':11}
    assert stats['total_lexical_hits']==32 and stats['pairs_with_multiple_hits']==0
    assert max(stats['phrase_counts'].values())<=3
    old=s.read_json('evaluation/persona_calibration/rubric.json');new=s.read_json('evaluation/profanity_steering/rubric.json')
    assert all(old[k]==new[k] for k in ('scale',*p.TRAITS,'gross_failure'))
    changed=copy.deepcopy(fixtures);changed['development']['items'][0]['topic']=changed['extraction']['items'][0]['topic']
    with pytest.raises(ValueError):p.validate_fixtures(changed,cfg()['lexicon'])
    changed=copy.deepcopy(fixtures);changed['extraction']['items'][0]['profanity_phrase']='not in text'
    with pytest.raises(ValueError):p.validate_fixtures(changed,cfg()['lexicon'])


def test_exact_thirds_hyphen_word_and_multiple_hits():
    rows=[{'profane':'one damn three','profanity_phrase':'damn'},{'profane':'one two neat-ass','profanity_phrase':'neat-ass'},{'profane':'damn damn','profanity_phrase':'damn'}]
    result=p.position_phrase_stats(rows,cfg()['lexicon'])
    assert result['position_bins']=={'opening':1,'middle':1,'closing':1}
    assert result['total_lexical_hits']==4 and result['pairs_with_multiple_hits']==1


@pytest.mark.parametrize('key,value',[('layers',[16,23]),('betas',[.05,.1,.3]),('batch_size',8),('max_new_tokens',320),('maximum_judge_calls',63),('seed',43)])
def test_fixed_config(key,value):
    value_cfg=cfg();s.validate_config(value_cfg);value_cfg[key]=value
    with pytest.raises(ValueError):s.validate_config(value_cfg)


def test_constrained_plans_all_conditions_distinct_prompts_and_deterministic():
    fixtures=data();plans=p.plans(fixtures);assert plans==p.plans(fixtures)
    for phase,(conditions,batches,seed) in p.PHASES.items():
        assert len(plans[phase])==batches
        assert all(len(b)==8 and len({r['group_id'] for r in b})==8 for b in plans[phase])
        counts=Counter(r['group_id'] for batch in plans[phase] for r in batch)
        assert set(counts.values())=={conditions}
        assert all(len(r['id'])==24 for b in plans[phase] for r in b)
    assert sum(map(len,plans.values()))==31


@pytest.fixture(scope='module')
def tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained('.hf/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c',local_files_only=True,trust_remote_code=False)


def test_all64_real_response_boundaries_and_length_windows(tokenizer):
    for row in data()['extraction']['items']:
        lengths=[]
        for side in ('neutral','profane'):
            value=s.response_encoding(tokenizer,row,row[side],cfg()['system'])
            assert len(value['input_ids'])==len(value['response_mask']) and any(value['response_mask']) and not value['response_mask'][-1]
            assert value['input_ids'][-1]==tokenizer.eos_token_id
            lengths.append(len(tokenizer(row[side],add_special_tokens=False)['input_ids']))
        assert .85<=lengths[1]/lengths[0]<=1.15


def test_eligibility_retains_all_and_enforces24_and_content_and_diversity():
    rows=data()['extraction']['items'];reviews={r['id']:{'neutral':review(r['id']),'profane':review(r['id'],True)} for r in rows};ratios={r['id']:1 for r in rows}
    result=p.eligibility(rows,reviews,ratios,cfg()['lexicon']);assert result['retained_count']==32 and result['sufficient']
    assert all(not v['factual_correct'] for side in reviews.values() for v in side.values())  # Nonfactual is not applicable, not attrition.
    assert result['retained_position_phrase_stats']['position_bins']=={'opening':10,'middle':11,'closing':11}
    for r in rows[:8]:reviews[r['id']]['profane']['content_ok']=False
    assert p.eligibility(rows,reviews,ratios,cfg()['lexicon'])['sufficient']
    reviews[rows[8]['id']]['profane']['coherent']=False
    assert not p.eligibility(rows,reviews,ratios,cfg()['lexicon'])['sufficient']


def test_vector_R_pooling_unit_random_seed_and_exact_safetensors(tmp_path):
    pairs=[({'mean':torch.tensor([3.,4.]),'token_norm_sum':12.,'tokens':2},{'mean':torch.tensor([4.,4.]),'token_norm_sum':14.,'tokens':2}),
        ({'mean':torch.tensor([0.,10.]),'token_norm_sum':30.,'tokens':3},{'mean':torch.tensor([1.,10.]),'token_norm_sum':33.,'tokens':3})]
    tensors,stats=p.direction_statistics(pairs,17)
    assert torch.equal(tensors['17.raw'],torch.tensor([1.,0.])) and stats['R_median_neutral_response_mean_norm']==7.5
    assert stats['mean_per_token_residual_norm_both_sides']==8.9
    assert torch.equal(p.delta(tensors,17,.1),torch.tensor([.75,0.]))
    torch.manual_seed(1);again,_=p.direction_statistics(pairs,17);assert torch.equal(again['17.random'],tensors['17.random'])
    file=tmp_path/'vectors.safetensors';s.save_vectors(file,tensors,{'scope':'synthetic unit test'})
    from safetensors.torch import load_file
    assert all(torch.equal(v,load_file(str(file))[k]) for k,v in tensors.items())
    with pytest.raises(FileExistsError):s.save_vectors(file,tensors)
    with pytest.raises(ValueError):p.direction_statistics([(pairs[0][0],pairs[0][0])],17)


def test_response_capture_mask_token_norm_and_exception_cleanup():
    class Block(torch.nn.Module):
        def forward(self,x):return x
    model=SimpleNamespace(model=SimpleNamespace(layers=torch.nn.ModuleList([Block()])))
    mask=torch.tensor([[False,True,True,False]]);values=torch.tensor([[[99.,0.],[3.,4.],[0.,10.],[99.,0.]]]);collected={}
    with p.capture_response(model,[0],mask,collected):model.model.layers[0](values)
    assert torch.equal(collected[0]['mean'],torch.tensor([1.5,7.])) and collected[0]['token_norm_sum']==15
    with pytest.raises(RuntimeError),p.capture_response(model,[0],mask,{}):raise RuntimeError('synthetic exception')
    assert not model.model.layers[0]._forward_hooks


def test_actual_tiny_qwen_zero_hook_cached_decode_and_cleanup():
    from transformers import Qwen3Config,Qwen3ForCausalLM
    torch.manual_seed(42)
    model=Qwen3ForCausalLM(Qwen3Config(vocab_size=31,hidden_size=16,intermediate_size=32,num_hidden_layers=2,num_attention_heads=2,num_key_value_heads=1,head_dim=8,pad_token_id=0,eos_token_id=2,max_position_embeddings=64)).eval()
    inputs={'input_ids':torch.tensor([[3,4,5],[0,6,7]]),'attention_mask':torch.tensor([[1,1,1],[0,1,1]])}
    options={'max_new_tokens':3,'min_new_tokens':3,'do_sample':False,'use_cache':True,'pad_token_id':0,'eos_token_id':2}
    with torch.inference_mode():
        original=model.generate(**inputs,**options);audit={}
        with p.audited_steer(model,1,torch.zeros(16),audit):zero=model.generate(**inputs,**options)
        assert torch.equal(original,zero) and audit['prefill_calls']==1 and audit['decode_calls']==2
        nonzero={}
        with p.audited_steer(model,1,torch.ones(16)*.1,nonzero):model.generate(**inputs,**options)
        assert nonzero['max_abs_hidden_delta']>0
        assert nonzero['hooks_removed'] and set(nonzero['sampled_phases'])=={'prefill','decode'}
    assert not model._forward_pre_hooks and not model.model.layers[1]._forward_hooks


def test_audited_hook_last_real_positions_and_zero_decode_is_allowed():
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__();self.model=SimpleNamespace(layers=torch.nn.ModuleList([torch.nn.Identity()]))
        def forward(self,input_ids,attention_mask,past_key_values=None):return self.model.layers[0](torch.zeros(*input_ids.shape,3))
    model=Tiny();audit={}
    with p.audited_steer(model,0,torch.ones(3),audit):
        value=model(input_ids=torch.tensor([[2,3,0],[4,0,0]]),attention_mask=torch.tensor([[1,1,0],[1,0,0]]))
    assert torch.equal(value[0,1],torch.ones(3)) and torch.equal(value[1,0],torch.ones(3))
    assert value[0,2].sum()==0 and value[1,1:].sum()==0
    assert audit['prefill_calls']==1 and audit.get('decode_calls',0)==0 and audit['hooks_removed']
    failed={}
    with pytest.raises(RuntimeError),p.audited_steer(model,0,torch.zeros(3),failed):raise RuntimeError('synthetic')
    assert failed['hooks_removed'] and not model._forward_pre_hooks and not model.model.layers[0]._forward_hooks


def metric(n=16,usable=0):
    return {'n':n,'usable_profane':usable,'raw_lexical':usable,'trait_sums':{'rudeness':0,'sarcasm':0},'clear_presence':{'rudeness':0,'sarcasm':0},'content_ok':n,'incoherent':0,'gross_failures':0,'factual_total':8,'factual_correct':8}


def test_selection_integer_gates_thresholds_and_tiebreaks():
    values={name:metric() for name in p.CONDITIONS};values['baseline']['usable_profane']=1
    values['l23_b050']['usable_profane']=8;values['l17_b100']['usable_profane']=8
    assert p.choose(values)['selected']=='l23_b050'
    values['l17_b050']['usable_profane']=8;assert p.choose(values)['selected']=='l17_b050'
    values['baseline']['usable_profane']=2;assert p.choose(values)['development_null']
    a,b=metric(),metric();b['trait_sums']['rudeness']=8;assert p.quality_gates(a,b,1)['rudeness_mean']
    b['trait_sums']['rudeness']=9;assert not p.quality_gates(a,b,1)['rudeness_mean']
    b=metric();b['content_ok']=14;assert not p.quality_gates(a,b,1)['content']


def test_confirmation_random_raw_and_strict_math_cannot_be_rescued_by_fallback():
    values={'baseline':metric(24,2),'selected':metric(24,12),'random':metric(24,0)}
    quality={'general_mc':{'correct':10},'physics_mc':{'correct':10},'gsm8k':{'strict_correct':8,'fallback_correct':8},'ifeval':{'prompt_strict_correct':7}}
    retention={'baseline':copy.deepcopy(quality),'selected':copy.deepcopy(quality)}
    assert p.final_gates(values,retention)['all_criteria_pass']
    values['random']['raw_lexical']=3;assert not p.final_gates(values,retention)['gates']['random_raw']
    values['random']['raw_lexical']=0;retention['selected']['gsm8k']['strict_correct']=6
    assert not p.final_gates(values,retention)['gates']['retention_gsm8k']


def journal_setup(tmp_path,monkeypatch):
    run=tmp_path/'run';run.mkdir();configuration=cfg();fixtures=data();plans=p.plans(fixtures);binary=tmp_path/'native';binary.write_text('fake executable; never run')
    m={'sources':{},'binary':str(binary),'cli_version':'2.1.323 (Claude Code)','audit_sha256':'synthetic'}
    s.save(run/'manifest.json',{'sources':{}})
    for name,value in {**fixtures,'plans':plans,'rubric':s.read_json('evaluation/profanity_steering/rubric.json'),
        'ratios':{r['id']:1 for r in fixtures['extraction']['items']}}.items():s.save(run/'inputs'/(name+'.json'),value)
    for phase,conditions in [('development',p.CONDITIONS),('confirmation',('baseline','selected','random'))]:
        results={name:{'predictions':[{'id':r['id'],'output':'A synthetic answer.','generated_ids':[3,2],'ended_eos':True,'hit_token_limit':False} for r in fixtures[phase]['items']]} for name in conditions}
        s.save(run/phase/'predictions.json',results);s.save(run/phase/'run.json',{'completed':True,'files_sha256':{'predictions.json':s.stream_hash(run/phase/'predictions.json')}})
    monkeypatch.setattr(s,'verify',lambda run,models=False:(configuration,m));monkeypatch.setattr(s,'registered',lambda a,r,models=False:(configuration,m));monkeypatch.setattr(s,'guard',lambda r:None)
    monkeypatch.setattr(s,'BINARY_SHA',s.stream_hash(binary));monkeypatch.setattr(s.shutil,'which',lambda name:str(binary));monkeypatch.setattr(s.f,'version',lambda name:m['cli_version'])
    monkeypatch.setattr(s.f,'validate_audit',lambda path:{'cli_version':m['cli_version']})
    return run,m


def install_calls(monkeypatch,mode='valid'):
    calls=[];seen=Counter()
    def invoke(binary,body):
        calls.append(copy.deepcopy(body));key=s.f.canonical_hash(body);seen[key]+=1
        if mode=='uncertain':raise OSError('synthetic transport failure')
        ratings=[review(r['id'],s.c.lexical(r['text'],cfg()['lexicon'])['present'],r['factual']) for r in body['items']]
        if mode=='null':
            for r in ratings:r['content_ok']=False
        if mode=='exhaust' or (mode=='retry' and seen[key]==1):ratings=ratings[:-1]
        outer={'subtype':'success','modelUsage':{s.f.MODEL:{}},'result':json.dumps({'ratings':ratings})}
        if mode=='model':outer['modelUsage']={}
        return {'returncode':0,'canonical_stdin_sha256':key,'isolation':{'git_not_repository':True,'git_environment_clean':True,'returncode':128},'stdout':json.dumps(outer)}
    monkeypatch.setattr(s.f,'invoke',invoke);return calls


def test_all31_constrained_batches_retry_budget_and_first_valid_resume(tmp_path,monkeypatch):
    run,m=journal_setup(tmp_path,monkeypatch);calls=install_calls(monkeypatch,'retry')
    for phase in p.PHASES:
        s.packet(SimpleNamespace(phase=phase),run);s.judge(SimpleNamespace(phase=phase,execute_judgments=True,resume=False),run)
        _,records=s.read_reviews(run,phase);assert all(r['first_valid_attempt']==1 for r in records)
    assert len(calls)==62 and all(calls[i]==calls[i+1] for i in range(0,62,2))
    s.judge(SimpleNamespace(phase='confirmation',execute_judgments=True,resume=True),run);assert len(calls)==62
    assert all(set(item)=={'id','context','user','requirements','reference','factual','text'} for body in calls for item in body['items'])


@pytest.mark.parametrize('mode,count',[('exhaust',2),('uncertain',1),('model',1)])
def test_journal_fatal_exhaustion_stops_without_extra_calls(tmp_path,monkeypatch,mode,count):
    run,m=journal_setup(tmp_path,monkeypatch);calls=install_calls(monkeypatch,mode);s.packet(SimpleNamespace(phase='eligibility'),run)
    with pytest.raises((s.f.InvalidSubmission,s.f.ModelMismatch,OSError)):s.judge(SimpleNamespace(phase='eligibility',execute_judgments=True,resume=False),run)
    with pytest.raises(RuntimeError):s.judge(SimpleNamespace(phase='eligibility',execute_judgments=True,resume=True),run)
    assert len(calls)==count


def test_packet_reconstruction_rejects_self_rehashed_wrong_text(tmp_path,monkeypatch):
    run,m=journal_setup(tmp_path,monkeypatch);s.packet(SimpleNamespace(phase='eligibility'),run)
    file=run/'judging/eligibility/packet.json';batch=s.read_json(file);batch[0][0]['text']='Wrong replacement text.';file.write_text(json.dumps(batch))
    manifest=run/'judging/eligibility/manifest.json';record=s.read_json(manifest);record['packet_sha256']=s.stream_hash(file);manifest.write_text(json.dumps(record))
    with pytest.raises(ValueError,match='frozen anonymous plan'):s.bodies(run,'eligibility')


def test_resume_absent_phase_rejected_before_mkdir_or_call(tmp_path,monkeypatch):
    run,m=journal_setup(tmp_path,monkeypatch);calls=install_calls(monkeypatch);s.packet(SimpleNamespace(phase='eligibility'),run)
    with pytest.raises(ValueError,match='existing phase journal'):s.judge(SimpleNamespace(phase='eligibility',execute_judgments=True,resume=True),run)
    assert not calls and not (run/'grades').exists()


def test_eligibility_null_export_and_no_raw_fields(tmp_path,monkeypatch):
    run,m=journal_setup(tmp_path,monkeypatch);install_calls(monkeypatch,'null');s.packet(SimpleNamespace(phase='eligibility'),run)
    s.judge(SimpleNamespace(phase='eligibility',execute_judgments=True,resume=False),run);s.eligibility(None,run)
    out=tmp_path/'public.json';s.report(SimpleNamespace(output=str(out)),run)
    result=s.read_json(out);assert result['status']=='eligibility_null' and result['eligibility']['retained_count']==0
    raw=out.read_text();assert all(r['user'] not in raw and r['neutral'] not in raw for r in data()['extraction']['items'])
    result={'status':'complete','stop':{'stop_cause':'confirmation_fatal'},'final_quality':{'all_criteria_pass':True,'gates':{'observed':True}}}
    s.with_stop_status(result);assert result['status']=='incomplete' and result['final_quality']['all_criteria_pass'] is False
    assert result['final_quality']['observed_all_criteria_pass'] is True


def test_vector_loader_rejects_unregistered_eligibility_or_manifest(tmp_path,monkeypatch):
    run,m=journal_setup(tmp_path,monkeypatch);s.save(run/'eligibility.json',{'sufficient':True,'retained_ids':['unregistered']})
    s.save(run/'extraction/vectors.json',{})
    monkeypatch.setattr(s,'eligibility_result',lambda r:{'sufficient':True,'retained_ids':[]})
    with pytest.raises(ValueError,match='eligibility subset'):s.vectors(run)


@pytest.mark.parametrize('tamper',['manifest','safe_metadata'])
def test_vector_model_manifest_and_companion_metadata_rejected(tmp_path,monkeypatch,tamper):
    run,m=journal_setup(tmp_path,monkeypatch);configuration=cfg();eligible={'sufficient':True,'retained_ids':[]}
    s.save(run/'eligibility.json',eligible);s.save(run/'inputs/encoded.json',{});s.save(run/'grades/eligibility/accepted.json',{})
    monkeypatch.setattr(s,'eligibility_result',lambda r:eligible)
    tensors={'17.raw':torch.tensor([1.,0.]),'17.unit':torch.tensor([1.,0.]),'17.random':torch.tensor([0.,1.]),'17.R':torch.tensor(2.)}
    identity=s.safe_metadata(run,configuration);directory=run/'extraction';directory.mkdir()
    s.save_vectors(directory/'vectors.safetensors',tensors,identity)
    record={'safetensors_sha256':s.stream_hash(directory/'vectors.safetensors'),'eligibility_sha256':s.stream_hash(run/'eligibility.json'),
        'manifest_sha256':s.stream_hash(run/'manifest.json'),'model_id':configuration['model_id'],'revision':configuration['revision'],
        'extraction_layers':list(p.LAYERS),'extraction_input_sha256':s.extraction_hashes(run),'safe_metadata':identity}
    file=directory/'vectors.json';s.save(file,record);assert torch.equal(s.vectors(run)['17.unit'],tensors['17.unit'])
    if tamper=='manifest':record['manifest_sha256']='wrong'
    else:record['safe_metadata']['revision']='wrong'
    file.write_text(json.dumps(record))
    with pytest.raises(ValueError):s.vectors(run)


def test_lock_binds_saved_development_labels_and_prevents_drift(tmp_path,monkeypatch):
    run=tmp_path/'run';run.mkdir();values={name:metric(16,8 if name=='l17_b050' else 0) for name in p.CONDITIONS};selection=p.choose(values)
    for name in ['manifest.json','strength.json','extraction/vectors.json','grades/development/accepted.json','judging/development/manifest.json','development/run.json','inputs/confirmation.json','inputs/retention.json','inputs/plans.json']:
        s.save(run/name,{'synthetic':True})
    (run/'extraction/vectors.safetensors').write_bytes(b'synthetic file; not loaded')
    s.save(run/'selection.json',selection);strength_reg=tmp_path/'strength.json';s.save(strength_reg,{'synthetic':True})
    monkeypatch.setattr(s,'strength',lambda a,r:None);monkeypatch.setattr(s,'metrics',lambda r,phase:values);monkeypatch.setattr(s,'committed',lambda f:None)
    args=SimpleNamespace(strength_registration=str(strength_reg),selection_registration=str(tmp_path/'selected.json'))
    s.lock(args,run);s.save(Path(args.selection_registration),{'lock_sha256':s.stream_hash(run/'final_lock.json')});assert s.checked_lock(args,run)=={'layer':17,'beta':.05}
    file=run/'grades/development/accepted.json';file.write_text('{"changed":true}')
    with pytest.raises(ValueError,match='drifted after lock'):s.checked_lock(args,run)


def test_no_live_flags_or_parent_write_paths(tmp_path):
    with pytest.raises(ValueError):s.guard(s.path('outputs/persona-rating-calibration-completion-01'))
    with pytest.raises(ValueError):s.judge(SimpleNamespace(execute_judgments=False),tmp_path)
    with pytest.raises(ValueError),s.gpu_stage(SimpleNamespace(execute_gpu=False),tmp_path,'unused'):pass
    assert not list(tmp_path.iterdir())
