"""CPU-only shared-profile checks; cached tokenizer/tiny model and mocked judge."""
import copy
from collections import Counter
import json
from types import SimpleNamespace

import pytest
import torch

from scripts import persona_traits as s
from scripts import persona_traits_common as p
from scripts import profanity_common as mechanics


def config(trait='rudeness'):return s.read_json(f'configs/{trait}_steering.json')
def data(trait='rudeness'):return {'extraction':s.read_json(f'evaluation/{trait}_steering/extraction.json'),**{n:s.read_json(f'evaluation/persona_traits/{n}.json') for n in ('development','confirmation')}}
def rating(id,target='rudeness',score=3):return {'id':id,**{t:score if t==target else 0 for t in p.TRAITS},'content_ok':True,'coherent':True,'factual_correct':False,'gross_failure':False}


def test_accepted_both_fixtures_spans_positions_and_exact_shared_bytes():
    hashes={'rudeness':'68708e2043bf00debc64353c3fa95eb04311b44dbc8c9c0939e1a91b35bffebf','sarcasm':'6e60715ae3df08a3a6fd6a1614ab95a9600d51e48217cf21bf507e966e5f0465'}
    topics=[]
    for trait in hashes:
        fixture=data(trait);p.validate_fixtures(fixture,config(trait)['lexicon'])
        assert s.stream_hash(s.path(f'evaluation/{trait}_steering/extraction.json'))==hashes[trait]
        assert p.span_stats(fixture['extraction']['items'])['position_bins']=={'opening':10,'middle':11,'closing':11}
        topics.append({r['topic'] for r in fixture['extraction']['items']})
        changed=copy.deepcopy(fixture);changed['extraction']['items'][0]['target']+=' Extra.'
        with pytest.raises(ValueError):p.validate_fixtures(changed,config(trait)['lexicon'])
    assert not topics[0]&topics[1]
    for n in ('development','confirmation','rubric'):assert s.path(f'evaluation/persona_traits/{n}.json').read_bytes()==s.path(f'evaluation/profanity_steering/{n}.json').read_bytes()


@pytest.mark.parametrize('trait',['rudeness','sarcasm'])
def test_profile_paths_fixed_grid_and_budget(trait):
    cfg=config(trait);s.validate_config(cfg)
    for k,v in [('maximum_judge_calls',81),('batch_size',8),('betas',[.05,.1,.3]),('target_trait','other'),('extraction_fixture','other/path')]:
        wrong=copy.deepcopy(cfg);wrong[k]=v
        with pytest.raises(ValueError):s.validate_config(wrong)
    assert cfg['run']==s.RUNS[trait]


def test_repeat_stable_aliases_balanced_constraints_and_no_extra_generation_phase():
    plans=p.plans(data());assert sum(map(len,plans.values()))==40
    for phase,(_,count,_) in p.PHASES.items():
        assert len(plans[phase])==count and all(len(b)==8 and len({e['group_id'] for e in b})==8 for b in plans[phase])
    project=lambda batches:{(e['group_id'],e['condition']):e['id'] for b in batches for e in b}
    assert project(plans['confirmation'])==project(plans['confirmation_repeat'])
    assert plans['confirmation']!=plans['confirmation_repeat']
    assert s.generation_phase('confirmation_repeat')=='confirmation'


@pytest.fixture(scope='module')
def tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained('.hf/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c',local_files_only=True,trust_remote_code=False)


def test_actual_cached_all128_boundaries_ratios_and_judge_context_excluded(tokenizer):
    for trait in ('rudeness','sarcasm'):
        for row in data(trait)['extraction']['items']:
            lengths=[]
            for side in ('neutral','target'):
                encoded=s.response_encoding(tokenizer,row,row[side],config(trait)['system'])
                assert any(encoded['response_mask']) and not encoded['response_mask'][-1]
                lengths.append(len(tokenizer(row[side],add_special_tokens=False)['input_ids']))
            assert .85<=lengths[1]/lengths[0]<=1.15
            prompt=s.render_prompt(tokenizer,{**row,'system':config(trait)['system']})
            assert row['user'] in prompt and row['context'] not in prompt


def test_eligibility_target_required_nonfactual_ignored_and_attrition_not_rebalanced():
    rows=data()['extraction']['items'];reviews={r['id']:{'neutral':rating(r['id'],score=0),'target':rating(r['id'])} for r in rows};ratios={r['id']:1 for r in rows}
    result=p.eligibility(rows,reviews,ratios,config()['lexicon'],'rudeness');assert result['retained_count']==32 and result['sufficient']
    for r in rows[:8]:reviews[r['id']]['target']['rudeness']=1
    result=p.eligibility(rows,reviews,ratios,config()['lexicon'],'rudeness');assert result['retained_count']==24 and result['sufficient']
    reviews[rows[8]['id']]['target']['sarcasm']=2
    assert not p.eligibility(rows,reviews,ratios,config()['lexicon'],'rudeness')['sufficient']


def test_consensus_and_quality_boolean_rules_and_entanglement():
    rows=[{'id':str(i),'factual':True} for i in range(4)];values=[{'id':str(i),'output':'A clear reply.','generated_ids':[3,2],'ended_eos':True,'hit_token_limit':False} for i in range(4)]
    a={r['id']:{**rating(r['id']),'factual_correct':True} for r in rows};b=copy.deepcopy(a)
    b['1']['rudeness']=1;b['2']['coherent']=False;b['2']['factual_correct']=False;b['3']['gross_failure']=True;b['0']['sarcasm']=3
    m=p.measures(rows,values,a,config()['lexicon'],'rudeness',b)
    assert m['usable_target']==1 and m['raw_target']==3 and m['raw_target_either']==4
    assert m['incoherent']==1 and m['gross_failures']==1 and m['factual_correct']==3
    assert m['trait_sums']['sarcasm']==3 and m['rudeness_and_sarcasm']==1
    b['0']['content_ok']=False;assert p.measures(rows,values,a,config()['lexicon'],'rudeness',b)['usable_target']==0


def metric(n=16,usable=0,target='rudeness'):
    return {'n':n,'target_trait':target,'usable_target':usable,'raw_target_either':usable,'trait_sums':{t:0 for t in p.TRAITS},'clear_presence':{t:0 for t in p.TRAITS},'content_ok':n,'incoherent':0,'gross_failures':0,'factual_total':8,'factual_correct':8}


def test_gates_random_either_and_no_kappa_gate():
    values={n:metric(16,8 if n=='l17_b050' else 0) for n in p.CONDITIONS};assert p.choose(values)['selected']=='l17_b050'
    values['l17_b050']['trait_sums']['profanity']=9;assert p.choose(values)['development_null']
    m={'baseline':metric(24,2),'selected':metric(24,12),'random':metric(24,0)}
    quality={'general_mc':{'correct':10},'physics_mc':{'correct':10},'gsm8k':{'strict_correct':8},'ifeval':{'prompt_strict_correct':7}}
    retention={'baseline':quality,'selected':copy.deepcopy(quality)};assert p.final_gates(m,retention)['all_criteria_pass']
    m['random']['raw_target_either']=3;assert not p.final_gates(m,retention)['gates']['random_raw']
    rows=[{'id':str(i)} for i in range(4)];reviews={r['id']:{'baseline':rating(r['id'],score=0)} for r in rows}
    d=p.repeat_diagnostics(rows,['baseline'],reviews,reviews)
    assert d['all_conditions']['profanity']['quadratic_weighted_kappa'] is None and not d['all_conditions']['profanity']['kappa_is_gate']


def test_divergence_length_and_pure_hook_reuse_with_tiny_qwen():
    assert p.audited_steer is mechanics.audited_steer and s.generated is s.prior.generated
    a=[{'id':'a','generated_ids':[1,2]},{'id':'b','generated_ids':[3,4]}];b=copy.deepcopy(a);b[0]['generated_ids']=[1,5,2]
    d=p.divergence(a,b);assert d['changed']==1 and d['identical']==1 and d['items'][0]['first_divergence_position']==1 and d['length_changes']['longer']==1
    from transformers import Qwen3Config,Qwen3ForCausalLM
    torch.manual_seed(42);model=Qwen3ForCausalLM(Qwen3Config(vocab_size=31,hidden_size=16,intermediate_size=32,num_hidden_layers=2,num_attention_heads=2,num_key_value_heads=1,head_dim=8,pad_token_id=0,eos_token_id=2,max_position_embeddings=64)).eval()
    x={'input_ids':torch.tensor([[3,4,5],[0,6,7]]),'attention_mask':torch.tensor([[1,1,1],[0,1,1]])};options={'do_sample':False,'max_new_tokens':3,'min_new_tokens':3,'use_cache':True}
    with torch.inference_mode():
        original=model.generate(**x,**options);audit={}
        with p.audited_steer(model,1,torch.zeros(16),audit):actual=model.generate(**x,**options)
    assert torch.equal(original,actual) and audit['prefill_calls']==1 and audit['decode_calls']==2 and audit['hooks_removed']


def setup(tmp_path,monkeypatch):
    run=tmp_path/'run';run.mkdir();cfg=config();fixtures=data();plans=p.plans(fixtures);binary=tmp_path/'native';binary.write_text('fake executable; never run')
    m={'sources':{},'binary':str(binary),'cli_version':'2.1.323 (Claude Code)','audit_sha256':'synthetic','target_trait':'rudeness'}
    s.save(run/'manifest.json',{'sources':{}})
    for name,value in {**fixtures,'plans':plans,'rubric':s.read_json('evaluation/persona_traits/rubric.json')}.items():s.save(run/'inputs'/(name+'.json'),value)
    for phase,conditions in [('development',p.CONDITIONS),('confirmation',('baseline','selected','random'))]:
        values={n:{'predictions':[{'id':r['id'],'output':'A synthetic reply.','generated_ids':[3,2],'ended_eos':True,'hit_token_limit':False} for r in fixtures[phase]['items']]} for n in conditions}
        s.save(run/phase/'predictions.json',values);s.save(run/phase/'run.json',{'completed':True,'files_sha256':{'predictions.json':s.stream_hash(run/phase/'predictions.json')}})
    monkeypatch.setattr(s,'verify',lambda r,models=False:(cfg,m));monkeypatch.setattr(s,'registered',lambda a,r,models=False:(cfg,m));monkeypatch.setattr(s,'guard',lambda r:None)
    monkeypatch.setattr(s,'BINARY_SHA',s.stream_hash(binary));monkeypatch.setattr(s.shutil,'which',lambda name:str(binary));monkeypatch.setattr(s.f,'version',lambda n:m['cli_version']);monkeypatch.setattr(s.f,'validate_audit',lambda p:{'cli_version':m['cli_version']})
    return run,cfg,m


def install_calls(monkeypatch,behavior='valid'):
    calls=[];seen=Counter();state={'phase':'eligibility'}
    def invoke(binary,body):
        calls.append(copy.deepcopy(body));key=(state['phase'],s.f.canonical_hash(body));seen[key]+=1
        if behavior=='uncertain':raise OSError('synthetic transport error')
        values=[rating(r['id'],score=0) for r in body['items']]
        if behavior=='exhaust' or (behavior=='retry' and seen[key]==1):values=values[:-1]
        outer={'subtype':'success','modelUsage':{s.f.MODEL:{}},'result':json.dumps({'ratings':values})}
        if behavior=='model':outer['modelUsage']={}
        return {'returncode':0,'canonical_stdin_sha256':s.f.canonical_hash(body),'isolation':{'git_not_repository':True,'git_environment_clean':True,'returncode':128},'stdout':json.dumps(outer)}
    monkeypatch.setattr(s.f,'invoke',invoke);return calls,state


def test_all40_batches_80_ceiling_resume_and_identical_repeat_payloads(tmp_path,monkeypatch):
    run,cfg,m=setup(tmp_path,monkeypatch);calls,state=install_calls(monkeypatch,'retry');generation_hash=s.stream_hash(run/'confirmation/predictions.json')
    for phase in p.PHASES:
        state['phase']=phase;s.packet(SimpleNamespace(phase=phase),run);s.judge(SimpleNamespace(phase=phase,execute_judgments=True,resume=False),run)
        s.read_reviews(run,phase)
    assert len(calls)==80 and all(calls[i]==calls[i+1] for i in range(0,80,2))
    by_alias=lambda phase:{r['id']:r for body in s.bodies(run,phase) for r in body['items']}
    assert by_alias('confirmation')==by_alias('confirmation_repeat')
    assert s.read_json(run/'judging/confirmation/manifest.json')['sources']==s.read_json(run/'judging/confirmation_repeat/manifest.json')['sources']
    assert s.stream_hash(run/'confirmation/predictions.json')==generation_hash and not (run/'confirmation_repeat').exists()
    s.judge(SimpleNamespace(phase='confirmation_repeat',execute_judgments=True,resume=True),run);assert len(calls)==80
    s.save(run/'grades/confirmation_repeat/batch-000-attempt-2-start.json',{})
    with pytest.raises(ValueError):s.audit_journals(run)


@pytest.mark.parametrize('behavior,count',[('exhaust',2),('uncertain',1),('model',1)])
def test_fatal_or_exhausted_journal_no_extra_calls(tmp_path,monkeypatch,behavior,count):
    run,_,_=setup(tmp_path,monkeypatch);calls,state=install_calls(monkeypatch,behavior);s.packet(SimpleNamespace(phase='eligibility'),run)
    with pytest.raises((s.f.InvalidSubmission,s.f.ModelMismatch,OSError)):s.judge(SimpleNamespace(phase='eligibility',execute_judgments=True,resume=False),run)
    with pytest.raises(RuntimeError):s.judge(SimpleNamespace(phase='eligibility',execute_judgments=True,resume=True),run)
    assert len(calls)==count


def test_repeat_wrong_text_rehashed_packet_is_rejected(tmp_path,monkeypatch):
    run,_,_=setup(tmp_path,monkeypatch);s.packet(SimpleNamespace(phase='confirmation_repeat'),run)
    file=run/'judging/confirmation_repeat/packet.json';packet=s.read_json(file);packet[0][0]['text']='Changed text';file.write_text(json.dumps(packet))
    file=run/'judging/confirmation_repeat/manifest.json';record=s.read_json(file);record['packet_sha256']=s.stream_hash(run/'judging/confirmation_repeat/packet.json');file.write_text(json.dumps(record))
    with pytest.raises(ValueError):s.bodies(run,'confirmation_repeat')


def test_trait_vector_metadata_and_fatal_report_acceptance(tmp_path,monkeypatch):
    run,cfg,m=setup(tmp_path,monkeypatch);eligible={'sufficient':True,'retained_ids':[]}
    s.save(run/'eligibility.json',eligible);monkeypatch.setattr(s,'eligibility_result',lambda r:eligible)
    file=run/'extraction/vectors.safetensors';file.parent.mkdir();file.write_bytes(b'not loaded')
    s.save(run/'extraction/vectors.json',{'safetensors_sha256':s.stream_hash(file),'eligibility_sha256':s.stream_hash(run/'eligibility.json'),
        'manifest_sha256':s.stream_hash(run/'manifest.json'),'target_trait':'sarcasm'})
    with pytest.raises(ValueError):s.vectors(run)
    meta=s.safe_metadata(run,cfg);assert meta['target_trait']=='rudeness'
    other=copy.deepcopy(cfg);other['target_trait']='sarcasm';assert s.safe_metadata(run,other)!=meta
    result={'status':'complete','stop':{'stop_cause':'fatal'},'final_quality':{'all_criteria_pass':True}}
    s.with_stop_status(result);assert result['status']=='incomplete' and result['final_quality']['all_criteria_pass'] is False


def test_old_write_guards_no_live_flags_and_incomplete_predecessor(tmp_path,monkeypatch):
    with pytest.raises(ValueError):s.guard(s.path('outputs/profanity-steering-01'))
    with pytest.raises(ValueError):s.judge(SimpleNamespace(execute_judgments=False),tmp_path)
    assert not list(tmp_path.iterdir())
    configuration=config('sarcasm');monkeypatch.setattr(s,'verify',lambda r,models=False:(configuration,{}))
    original=s.read_json
    monkeypatch.setattr(s,'read_json',lambda p:{'status':'incomplete'} if str(p)=='results/rudeness_steering_01.json' else original(p))
    with pytest.raises(ValueError,match='rudeness must close'):s.registered(SimpleNamespace(trait='sarcasm'),tmp_path)
