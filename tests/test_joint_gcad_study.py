"""Offline study contracts; no target forward, native process or snapshot prep."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from scripts import joint_gcad_study as study
from scripts import joint_gcad_common as common
from scripts import tone_judge as foundation

PROPOSAL=Path('outputs/joint-gcad-design-01/fixture-proposal.json')

def fixture():return study.compile_fixture(json.loads(PROPOSAL.read_text()),json.loads(Path(study.CONFIG).read_text()))

def save(file,value):file.parent.mkdir(parents=True,exist_ok=True);file.write_text(json.dumps(value))


def test_real_fixture_conversion_bands_exact_rubrics_and_packed_counts():
    data=fixture();original=json.loads(PROPOSAL.read_text())
    assert data['questions']==original['test']
    for old,new in [('PF','profanity'),('impolite','impolite'),('SARC','sarcasm'),('coherence','coherence')]:
        assert data['rubrics'][new]==original['rubrics'][old]['text']
        for a,b in zip(data['anchors'][new],original['calibration'][old]):
            assert a['level']==b['side'] and a['band']==[b['min_score'],b['max_score']] and a['answer']==b['answer']
    plans=common.plans(data['questions'],data['anchors'],37)
    assert sum(len(b) for b in plans['primary'].values())==144
    assert sum(len(b) for b in plans['duplicates'].values())==12
    assert sum(len(b) for b in plans['calibration'].values())==4
    for stream,metrics in plans.items():
        for m,batches in metrics.items():
            assert all(len(b)==8 and len({i['question_id'] for i in b})==8 for b in batches)
            if stream=='primary':assert len({(i['question_id'],i['condition']) for b in batches for i in b})==288
            if stream=='duplicates':
                flat=[r for b in batches for r in b];assert len({r['question_id'] for r in flat})==24
                assert all(sum(r['condition']==a for r in flat)==2 for a in common.arms())


def test_real_release_readonly_loader_and_core_identity():
    cfg=json.loads(Path(study.CONFIG).read_text());loader,profiles,banks=study.release(cfg)
    assert set(banks)==set(common.TRAITS)
    assert all(profiles[t]['controller']['base_coef']==common.FULL[t] for t in common.TRAITS)


def test_payloads_hide_bands_arms_and_use_same_unchanged_rubrics(tmp_path):
    data=fixture();s=SimpleNamespace(run=tmp_path);plans=common.plans(data['questions'],data['anchors'],37)
    save(tmp_path/'inputs/data.json',data);save(tmp_path/'inputs/plans.json',plans)
    body=study.bodies(s,'calibration','profanity')[0]
    assert body['rubric']==data['rubrics']['profanity'] and body['expected_count']==8
    assert all(set(item)=={'id','question','answer'} for item in body['items'])
    def keys(v):
        if isinstance(v,dict):return set(v)|set().union(*(keys(x) for x in v.values()))
        if isinstance(v,list):return set().union(*(keys(x) for x in v))
        return set()
    assert not keys(body)&{'min_score','max_score','band','design_note','condition'}


def test_calibration_seven_of_eight_and_ordering_no_semantic_retry():
    data=fixture();ratings={m:{a['id']:{'score':0 if a['level']=='low' else 100} for a in data['anchors'][m]} for m in common.METRICS}
    assert common.calibration_checks(data['anchors'],ratings)['pass']
    a=data['anchors']['profanity'];ratings['profanity'][a[0]['id']]['score']=100
    assert common.calibration_checks(data['anchors'],ratings)['pass']
    ratings['profanity'][a[1]['id']]['score']=100
    assert not common.calibration_checks(data['anchors'],ratings)['pass']
    for r in ratings['sarcasm'].values():r['score']=50
    assert not common.calibration_checks(data['anchors'],ratings)['metrics']['sarcasm']['pass']


def attempt(rows):
    return {'returncode':0,'isolation':{'git_not_repository':True,'git_environment_clean':True,'returncode':128},
        'stdout':json.dumps({'subtype':'success','modelUsage':{'claude-opus-5-5':{}},'result':json.dumps({'ratings':rows})})}

@pytest.mark.parametrize('bad',['coherence_null','boolscore','missing','duplicate','extra','fallback'])
def test_strict_native_schema_model_identity(bad):
    row={'id':'a','score':50,'reason':'private'};rows=[row];metric='coherence'
    if bad=='coherence_null':row['score']=None
    elif bad=='boolscore':row['score']=True
    elif bad=='missing':rows=[]
    elif bad=='duplicate':rows=[row,row]
    elif bad=='extra':row['unexpected']=0
    packet=attempt(rows)
    if bad=='fallback':packet['stdout']=packet['stdout'].replace('claude-opus-5-5','other-model')
    with pytest.raises((foundation.InvalidSubmission,foundation.ModelMismatch)):
        study.parser(metric,packet,{'expected_ids':['a']})


def test_trait_refusal_valid_but_only_conservative_zero():
    rows=[{'id':'a','score':None,'refusal':True,'reason':'private'}]
    value,models=study.parser('sarcasm',attempt(rows),{'expected_ids':['a']})
    assert value['a']['score'] is None and models==['claude-opus-5-5']


def test_resolved_greedy_overrides_sampling_checkpoint():
    model=SimpleNamespace(generation_config=SimpleNamespace(to_dict=lambda:{'do_sample':True,'temperature':.7,'top_p':.8,'top_k':20,'repetition_penalty':1.05,'eos_token_id':[151645,151643]}))
    cfg=json.loads(Path(study.CONFIG).read_text());v=study.resolved_generation(model,160,cfg['generation_overrides'])
    assert v['do_sample'] is False and (v['temperature'],v['top_p'],v['top_k'])==(1.,1.,0)
    assert v['eos_token_id']==[151645,151643]


def metrics():return {a:{'profanity':0.,'impolite':0.,'sarcasm':0.,'coherence':80.,'invalid':0} for a in common.arms()}

def test_pair_guide_checks_only_requested_targets_half_matches_half_na():
    m=metrics()
    for t,a in zip(common.TRAITS,['full_100','full_010','full_001']):m[a][t]=80
    m['full_110'].update(profanity=40,impolite=40,coherence=60)
    m['half_triple'].update(profanity=10,impolite=10,sarcasm=10)
    for t in common.TRAITS:m['half_'+t][t]=20
    g=common.guide(m)
    assert g['full_110']['all_pass'] and set(g['full_110']['targets'])=={'profanity','impolite'}
    assert g['half_triple']['all_pass']
    m['half_profanity']['profanity']=0
    assert common.guide(m)['half_triple']['targets']['profanity']['pass'] is None
    m['full_110']['coherence']=59.99
    assert not common.guide(m)['full_110']['all_pass']


def test_summary_linear_additivity_matched_bootstrap_and_null_coherence():
    preds={};grades={}
    for arm,setting in common.arms().items():
        preds[arm]=[{'id':str(i),'output':'This is a clear useful response.','generated_ids':[3,4],'ended_eos':True,'hit_token_limit':False} for i in range(24)]
        for i in range(24):grades[arm,str(i)]={m:{'score':80 if m=='coherence' else setting[m]*10+i%2} for m in common.METRICS}
    value=common.summarize(preds,grades,bootstrap_seed=15,bootstrap_repetitions=30)
    for name in ['full_triple_additivity_residual','half_triple_additivity_residual','full_110_additivity_residual']:
        assert all(abs(v)<1e-10 for v in value['factorial_contrasts'][name].values())
    grades['baseline','0']['coherence']['score']=None
    with pytest.raises(ValueError,match='coherence'):common.summarize(preds,grades,bootstrap_seed=15,bootstrap_repetitions=30)
    preds['baseline']=preds['baseline'][:23]
    with pytest.raises(ValueError,match='matched 24'):common.summarize(preds,grades,bootstrap_seed=15,bootstrap_repetitions=30)


def test_prestart_native_budget_and_missing_stage_wait_no_writes(tmp_path,monkeypatch):
    s=SimpleNamespace(run=tmp_path,cfg={'calibration_registration':str(tmp_path/'missing.json')})
    before=list(tmp_path.rglob('*'))
    with pytest.raises(study.Wait):study.lock(s,'calibration')
    assert list(tmp_path.rglob('*'))==before
    monkeypatch.setattr(study,'native_count',lambda s:321)
    with pytest.raises(ValueError,match='320'):study.read_grades(s,'primary','profanity')


def test_duplicates_wait_for_all_primary_streams(tmp_path,monkeypatch):
    s=SimpleNamespace(run=tmp_path);monkeypatch.setattr(study,'lock',lambda *a:None);monkeypatch.setattr(study,'stage',lambda *a:None)
    with pytest.raises(study.Wait,match='all primary'):study.ordered_prerequisites(s,'duplicates','profanity')
    assert not list(tmp_path.iterdir())


def test_actual_mechanics_dispatch_exact_twenty_generations_ten_logit_forwards(tmp_path,monkeypatch):
    import torch
    from contextlib import contextmanager
    data=fixture();layers=list(range(9,20));s=SimpleNamespace(run=tmp_path,cfg={'layers':layers})
    save(tmp_path/'manifest.json',{});save(tmp_path/'inputs/data.json',data)
    save(tmp_path/'inputs/rendered.json',{q['id']:{'input_ids':[3,4]} for q in data['questions']})
    out=tmp_path/'mechanics';out.mkdir();active=[0.]
    @contextmanager
    def context(*args,**kwargs):
        setting=args[3];old=active[0];active[0]=sum(setting.values())
        try:yield
        finally:active[0]=old
    monkeypatch.setattr(study,'intervention',context)
    class Fake:
        device='cpu'
        def __init__(self):self.model=SimpleNamespace(layers=[]);self.calls=0
        def __call__(self,*args,**kwargs):self.calls+=1;return SimpleNamespace(logits=torch.tensor([[[0.,active[0]]]]))
    model=Fake()
    monkeypatch.setattr(study.native,'hook_signature',lambda m:())
    def one(s,model,tok,row,out,banks,setting,**kwargs):
        study.charge(s,out,'generation',{'id':row['id'],'setting':setting})
        nonzero=any(setting.values())
        return {'id':row['id'],'generated_ids':[3,4] if not nonzero else [7,4],
            'audit':None if kwargs['mode']=='joint' and not nonzero else {'coverage':{l:{'prefill':{'applied':1,'skipped':0},'decode':{'applied':1,'skipped':0}} for l in layers},
            'measurements':{l:{'prefill':{'sum':{'norm_sum':float(nonzero)}}} for l in layers}}}
    monkeypatch.setattr(study,'one',one)
    study.mechanics(s,model,None,out,{})
    result=json.loads((out/'result.json').read_text())
    assert result['all_pass'] and result['explicit_logit_forwards']==10
    records=[json.loads(p.read_text()) for p in out.glob('call-*-start.json')]
    assert sum(r['kind']=='generation' for r in records)==20 and sum(r['kind']=='logit' for r in records)==10 and model.calls==10

@pytest.mark.parametrize('fatal_duplicate',[False,True])
def test_report_preserves_primary_analysis_and_privacy(tmp_path,monkeypatch,fatal_duplicate):
    cfg={'result':str(tmp_path/'public.json'),'judge_runtime_config':'configs/qwen3_14b_baseline.json','bootstrap_seed':11,'bootstrap_repetitions':4}
    s=SimpleNamespace(run=tmp_path,cfg=cfg,m={'sources':{'scripts/joint_gcad_study.py':'a'*64},'external_sha256':{'configs/qwen3_14b_baseline.json':'b'*64},'cli_version':'2.1.323'})
    save(tmp_path/'manifest.json',{});save(tmp_path/'inputs/profiles.json',{t:{'bank':{'sha256':'c'*64}} for t in common.TRAITS})
    save(tmp_path/'calibration.json',{'pass':True})
    for stream in ('calibration','primary',*(() if fatal_duplicate else ('duplicates',))):
        for m in common.METRICS:save(tmp_path/'native'/stream/m/'accepted.json',{})
    if fatal_duplicate:save(tmp_path/'stopped.json',{'phase':'judge','metric':'profanity','stop_cause':'RuntimeError'})
    def accepted(s,stream,m):return {'batches':[{'models':['claude-opus-5-5']}]*(1 if stream=='calibration' else 36 if stream=='primary' else 3)}
    monkeypatch.setattr(study,'accepted',accepted)
    predictions={a:[{'id':str(i),'output':'Clear useful response with no repetition.','generated_ids':[3,4],
        'hit_token_limit':False,'ended_eos':True} for i in range(24)] for a in common.arms()}
    grades={(a,str(i)):{m:{'score':80 if m=='coherence' else 0} for m in common.METRICS} for a in common.arms() for i in range(24)}
    duplicates={key:grades[key] for j,a in enumerate(common.arms()) for key in ((a,str(2*j)),(a,str(2*j+1)))}
    monkeypatch.setattr(study,'predictions',lambda s:predictions)
    monkeypatch.setattr(study,'scored',lambda s,stream:grades if stream=='primary' else duplicates)
    r=study.report(s)
    assert r['status']==('primary_complete_duplicates_incomplete' if fatal_duplicate else 'complete')
    assert r['first_valid_batches']==(148 if fatal_duplicate else 160)
    assert not any(v['all_pass'] for v in r['analysis']['guide'].values())
    assert (tmp_path/'public.json').exists() and 'reason' not in r


def test_early_fatal_report_uses_registered_result_destination(tmp_path):
    s=SimpleNamespace(run=tmp_path,cfg={'result':str(tmp_path/'public.json'),'judge_runtime_config':'configs/qwen3_14b_baseline.json'},
        m={'sources':{},'external_sha256':{'configs/qwen3_14b_baseline.json':'a'*64},'cli_version':'2.1.323'})
    save(tmp_path/'manifest.json',{});save(tmp_path/'inputs/profiles.json',{t:{'bank':{'sha256':'b'*64}} for t in common.TRAITS})
    save(tmp_path/'stopped.json',{'phase':'mechanics','metric':None,'stop_cause':'RuntimeError'})
    result=study.report(s)
    assert result['status']=='execution_stopped' and (tmp_path/'public.json').exists() and not (tmp_path/'progress.json').exists()


def test_completed_cpu_response_atomically_retained_before_later_work(tmp_path):
    import torch
    from test_joint_gcad import tiny,components
    model=tiny();model.generation_config.eos_token_id=[2]
    cfg=json.loads(Path(study.CONFIG).read_text());cfg['layers']=[0,1]
    s=SimpleNamespace(run=tmp_path,cfg=cfg);row={'id':'T01','user':'Neutral question'}
    save(tmp_path/'manifest.json',{});save(tmp_path/'inputs/rendered.json',{'T01':{'input_ids':[3,4,5],'prompt':'frozen'}})
    out=tmp_path/'generation';out.mkdir()
    tok=SimpleNamespace(decode=lambda ids,**kwargs:'Completed answer')
    banks={t:{k:v for k,v in values.items() if k in ('delta','key','demand')} for t,values in components().items()}
    record=study.one(s,model,tok,row,out,banks,common.arms()['baseline'],cap=2)
    saved=list((out/'answers').glob('*.json'))
    assert len(saved)==1 and json.loads(saved[0].read_text())['record']==record
    assert record['audit'] is None and len(list(out.glob('call-*-start.json')))==1
