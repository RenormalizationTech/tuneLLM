"""CPU-only child reuse, exact-grid, budget, holdout and parent immutability."""
import copy
from contextlib import contextmanager, nullcontext
import json
from types import SimpleNamespace

import pytest
import torch

from scripts import persona_pv_grid as s
from scripts import persona_pv_grid_common as p


def cfg(trait='rudeness'):return s.read_json(s.config_path(trait))
def split():return s.read_json('outputs/persona-reference-rudeness-01/inputs/split.json')
def rating(id,score=60):return {'id':id,'trait':score,'coherence':90,'refusal':False,'reason':'synthetic test only'}


def test_exact_grid_reuse_coverage_profiles_no_parent3or4_candidates():
    for trait in ('rudeness','profanity'):
        c=cfg(trait);s.validate_config(c);reuse=p.reuse_map(trait);missing=p.missing(trait)
        assert set(reuse)|set(missing)=={'baseline',*p.SETTINGS} and not set(reuse)&set(missing)
        assert len(reuse)==(12 if trait=='rudeness' else 0) and len(missing)==(16 if trait=='rudeness' else 28)
        assert set(x['coefficient'] for x in p.SETTINGS.values())=={.5,1.,2.} and len(p.SETTINGS)==27
        assert p.SETTINGS['l8_c0.5']['block']==7 and p.SETTINGS['l35_c2']['block']==34
        for key,value in [('maximum_new_starts',100),('maximum_new_generations',400),('coefficients',[.5,1.,2.,3.])]:
            bad=copy.deepcopy(c);bad[key]=value
            with pytest.raises(ValueError):s.validate_config(bad)
    assert cfg('profanity')['prospective_amendment']=='results/protocols/persona-reference-profanity-grid-amendment-01.json'
    assert cfg('rudeness')['plan_seeds']!=cfg('profanity')['plan_seeds']


def test_frozen_child_plans_no_reused_answers_in_packets_and_budgets():
    for trait,count,total in [('rudeness',20,60),('profanity',35,90)]:
        plans=p.plans(split(),trait,cfg(trait)['plan_seeds']);assert len(plans['grid'])==count and len(plans['final'])==len(plans['custom'])==5
        assert 2*sum(map(len,plans.values()))==total
        for phase,plan in plans.items():
            assert all(len(b)==8 and len({e['question_id'] for e in b})==8 for b in plan)
            entries=[(e['question_id'],e['condition']) for b in plan for e in b];assert len(set(entries))==len(entries)
        assert {e['condition'] for b in plans['grid'] for e in b}==set(p.missing(trait))
        assert not {e['question_id'] for b in plans['grid'] for e in b}&{e['question_id'] for b in plans['final'] for e in b}


def metric(total=200,coherence=900,**kw):return {'n':10,'score_sum':total,'coherence_sum':coherence,'conservative_trait_mean':total/10,'coherence_mean':coherence/10,'empty':0,'degenerate':0,'capped':0,**kw}


def test_integer_ten_point_gain_731_minus631_and_ties_quality():
    values={'baseline':metric(631),**{n:metric(100) for n in p.SETTINGS}}
    values['l8_c1']=metric(731);assert p.choose(values)['selected']=='l8_c1' and p.choose(values)['gain_sum']==100 and not p.choose(values)['selection_null']
    values['l8_c1']=metric(730);assert p.choose(values)['selection_null']
    values['l35_c0.5']=metric(731);values['l4_c1']=metric(731);assert p.choose(values)['selected']=='l35_c0.5'
    values['l4_c0.5']=metric(731);assert p.choose(values)['selected']=='l4_c0.5'
    values['l4_c0.5']['coherence_sum']=749;assert p.choose(values)['selected']=='l35_c0.5'
    values['l35_c0.5']['degenerate']=2;assert p.choose(values)['selected']=='l4_c1'


@pytest.mark.parametrize('trait',['rudeness','profanity'])
@pytest.mark.parametrize('forbidden',['final','custom','report','retention','capability','grades/final','judging/custom'])
def test_parent_heldout_outputs_rejected_at_prepare_and_verify(tmp_path,trait,forbidden):
    parent=SimpleNamespace(run=tmp_path/'parent');parent.run.mkdir();s.parent_paths(parent,cfg(trait));directory=parent.run/forbidden;directory.mkdir(parents=True)
    with pytest.raises(ValueError):s.parent_paths(parent,cfg(trait))


def test_new_only_guards_and_prospective_parent_selection_forbidden(tmp_path):
    with pytest.raises(ValueError):s.guard(s.path('outputs/persona-reference-rudeness-01'),'rudeness')
    with pytest.raises(ValueError):s.guard(s.path('outputs/persona-pv-grid-rudeness-01'),'profanity')
    parent=SimpleNamespace(run=tmp_path);(tmp_path/'layer').mkdir()
    with pytest.raises(ValueError):s.parent_paths(parent,cfg('profanity'))


def generation(items):
    predictions=[];batches=[]
    for start in range(0,len(items),4):
        members=items[start:start+4];ids=[r['id'] for r in members];bi=start//4
        for position,row in enumerate(members):predictions.append({'id':row['id'],'output':'Synthetic answer only.','input_ids':[4,5],'rendered_prompt':'synthetic '+row['id'],
            'generated_ids':[3,2],'generated_tokens_including_eos':2,'ended_eos':True,'hit_token_limit':False,'batch_index':bi,'batch_position':position,'batch_ids':ids,'padded_prompt_width':2})
        batches.append({'batch_index':bi,'ids':ids,'padded_prompt_width':2})
    return {'predictions':predictions,'batches':batches,'settings':{'batch_size':4,'max_new_tokens':512,'seed':42,'do_sample':False}}


def setup(tmp_path,monkeypatch,trait='rudeness'):
    run=tmp_path/'child';run.mkdir();c=cfg(trait);partition=split();plans=p.plans(partition,trait,c['plan_seeds']);runtime=s.read_json('configs/persona_reference_'+trait+'.json')
    binary=tmp_path/'native';binary.write_text('synthetic only, never executed');m={'sources':{},'binary':str(binary),'binary_sha256':'mock','cli_version':'2.1.323 (Claude Code)','audit_sha256':'synthetic','tokenizer_ids':{'eos':2,'pad':0},'parent_manifest_sha256':'parent','parent_strength_sha256':'vector','reused_grid_outputs':len(p.reuse_map(trait))*10,'new_grid_outputs':len(p.missing(trait))*10}
    s.save(run/'manifest.json',{'synthetic':True})
    rendered={k:{r['id']:{'input_ids':[4,5],'rendered_prompt':'synthetic '+r['id']} for r in [*partition['selection'],*partition['report']]} for k in ('user_only','prompt_control')}
    for name,value in [('split',partition),('plans',plans),('rendered',rendered),('reuse',p.reuse_map(trait)),('rubric',s.read_json('outputs/persona-reference-rudeness-01/inputs/rubric.json')),('custom_rubric',s.read_json('outputs/persona-reference-rudeness-01/inputs/custom_rubric.json'))]:s.save(run/'inputs'/(name+'.json'),value)
    for phase,arms in [('grid',p.missing(trait)),('final',('baseline','selected','random','prompt'))]:s.save(run/phase/'predictions.json',{arm:generation(partition['selection' if phase=='grid' else 'report']) for arm in arms})
    context=SimpleNamespace(run=run,cfg=c,m=m,runtime=runtime,parent=SimpleNamespace(run=tmp_path/'parent'),initial=lambda:None,verify=lambda **kw:None)
    monkeypatch.setattr(s.pv,'BINARY_SHA',s.stream_hash(binary));monkeypatch.setattr(s.shutil,'which',lambda _:str(binary));monkeypatch.setattr(s.f,'version',lambda _:m['cli_version']);monkeypatch.setattr(s.f,'validate_audit',lambda _: {})
    monkeypatch.setattr(s,'stage_sources',lambda context,phase:{'synthetic_phase':phase});monkeypatch.setattr(s,'selected',lambda *args,**kw:{'author_layer':8,'block':7,'coefficient':.5})
    return context


def calls(monkeypatch,behavior='valid'):
    records=[];seen={}
    def invoke(binary,body):
        records.append(copy.deepcopy(body));h=s.f.canonical_hash(body);seen[h]=seen.get(h,0)+1
        if behavior=='transport':raise OSError('synthetic uncertain start')
        custom='scale' in body['rubric'];labels=[{'id':r['id'],'profanity':0,'rudeness':0,'sarcasm':0,'gross_failure':False,'content_ok':True,'coherent':True} if custom else rating(r['id']) for r in body['items']]
        if behavior=='exhaust' or behavior=='retry' and seen[h]==1:labels=labels[:-1]
        outer={'subtype':'success','modelUsage':{s.f.MODEL:{}},'result':json.dumps({'ratings':labels})}
        if behavior=='model':outer['modelUsage']={}
        return {'returncode':0,'canonical_stdin_sha256':h,'stdout':json.dumps(outer),'isolation':{'git_not_repository':True,'git_environment_clean':True,'returncode':128}}
    monkeypatch.setattr(s.f,'invoke',invoke);return records


def judge(context,phase,resume=False):
    s.judge(SimpleNamespace(phase=phase,resume=resume),context)
    if hasattr(context,'pending_accept'):s.f.save_or_match(*context.pending_accept);del context.pending_accept


@pytest.mark.parametrize('trait,maximum',[('rudeness',60),('profanity',90)])
def test_full_child_call_budget_first_valid_schema_retry_and_resume(tmp_path,monkeypatch,trait,maximum):
    context=setup(tmp_path,monkeypatch,trait);records=calls(monkeypatch,'retry')
    for phase in p.PHASES:
        s.packet(SimpleNamespace(phase=phase),context);judge(context,phase);s.read_reviews(context,phase)
    assert len(records)==maximum and all(records[i]==records[i+1] for i in range(0,maximum,2))
    judge(context,'custom',True);assert len(records)==maximum;s.audit_journals(context)
    assert all(set(r)=={'id','question','answer'} for b in records for r in b['items'])


@pytest.mark.parametrize('behavior,n',[('exhaust',2),('model',1),('transport',1)])
def test_fatal_schema_identity_uncertain_permanent_no_extra_calls(tmp_path,monkeypatch,behavior,n):
    context=setup(tmp_path,monkeypatch);records=calls(monkeypatch,behavior);s.packet(SimpleNamespace(phase='grid'),context)
    with pytest.raises((s.f.InvalidSubmission,s.f.ModelMismatch,OSError)):
        try:judge(context,'grid')
        except BaseException as exc:s.pv.stop(context.run,'child_live_stage_fatal',exc);raise
    with pytest.raises(RuntimeError):p.finish_judgments(context.run,context.run/'grades/grid',s.bodies(context,'grid'),'bound',60)
    assert len(records)==n


def test_budget_before_start_packet_and_receipt_tamper(tmp_path,monkeypatch):
    context=setup(tmp_path,monkeypatch);records=calls(monkeypatch);s.packet(SimpleNamespace(phase='grid'),context)
    with pytest.raises(ValueError):judge(context,'grid',True)
    assert not records and not (context.run/'grades').exists()
    judge(context,'grid');file=context.run/'grades/grid/batch-000-attempt-0-receipt.json';record=s.read_json(file);record['canonical_stdin_sha256']='wrong';file.write_text(json.dumps(record))
    with pytest.raises(ValueError):s.read_reviews(context,'grid')
    file=context.run/'judging/grid/packet.json';record=s.read_json(file);record[0][0]['answer']='Tampered';file.write_text(json.dumps(record))
    file=context.run/'judging/grid/manifest.json';record=s.read_json(file);record['packet_sha256']=s.stream_hash(context.run/'judging/grid/packet.json');file.write_text(json.dumps(record))
    with pytest.raises(ValueError):s.bodies(context,'grid')


def test_reuse_exact_parent_objects_and_grade_aliases_no_parent_write(tmp_path,monkeypatch):
    context=setup(tmp_path,monkeypatch);selection=split()['selection'];parent_outputs={'layer':{'baseline':generation(selection),**{f'layer{k}':generation(selection) for k in p.LAYERS}},'strength':{n:generation(selection) for n in ('coef0.5','coef1','coef3','coef4')}}
    labels={phase:{r['id']:{name:rating('original-'+r['id']+'-'+name) for name in value} for r in selection} for phase,value in parent_outputs.items()}
    monkeypatch.setattr(s.pv,'values',lambda parent,phase:parent_outputs[phase]);monkeypatch.setattr(s.pv,'read_reviews',lambda parent,phase:labels[phase])
    before=copy.deepcopy((parent_outputs,labels));reused=s.reused(context)
    assert len(reused)==12 and sum(len(out['predictions']) for out,grade in reused.values())==120
    assert reused['l8_c0.5'][0] is parent_outputs['strength']['coef0.5'] and reused['l8_c0.5'][1][selection[0]['id']] is labels['strength'][selection[0]['id']]['coef0.5']
    assert (parent_outputs,labels)==before


@pytest.mark.parametrize('trait,count',[('rudeness',160),('profanity',280)])
def test_generate_only_missing_no_teacher_forcing_no_mechanical_regeneration(tmp_path,monkeypatch,trait,count):
    context=setup(tmp_path,monkeypatch,trait);raw={f'{layer-1}.raw':torch.arange(16).float()+1 for layer in p.LAYERS};calls_seen=[];progress={'saved_generations':0,'teacher_forced_sequences':0,'mc_items':0,'generation_wall_seconds':0.};existing={k:0 for k in ('saved_generations','teacher_forced_sequences','mc_items')}
    monkeypatch.setattr(s.pv,'strength',lambda _:raw)
    @contextmanager
    def stage(s,phase):yield None,None,context.run/'grid',progress,existing
    monkeypatch.setattr(s,'gpu_stage',stage)
    def generated(s,model,tok,items,progress,existing,vector=None,block=None):
        calls_seen.append((block,None if vector is None else vector.clone()));progress['saved_generations']+=len(items);return generation(items)
    monkeypatch.setattr(s,'generated',generated)
    # setup's synthetic file is removed only inside tmp_path to exercise new-only save.
    (context.run/'grid/predictions.json').unlink();s.grid(SimpleNamespace(),context)
    assert progress['saved_generations']==count and progress['teacher_forced_sequences']==0 and len(calls_seen)==count//10
    for name,(block,delta) in zip(p.missing(trait),calls_seen):
        setting=p.SETTINGS.get(name)
        if setting:assert torch.equal(delta,raw[f'{block}.raw']*setting['coefficient'])
        else:assert delta is None


def test_selected_holdout_requires_exact_committed_lock_before_any_generation(tmp_path,monkeypatch):
    actual_selected=s.selected
    context=setup(tmp_path,monkeypatch);record={'selection_null':False,'setting':{'author_layer':8,'block':7,'coefficient':.5},'bindings':{'some':'bound'}}
    context.cfg['selection_registration']=str(tmp_path/'lock.json');s.save(context.run/'selection.json',record);monkeypatch.setattr(s,'selection_result',lambda _:record);monkeypatch.setattr(s,'committed',lambda _:None)
    monkeypatch.setattr(s,'selected',actual_selected)
    with pytest.raises(FileNotFoundError):s.selected(context,True)
    s.save(tmp_path/'lock.json',{'selection_sha256':'wrong'})
    with pytest.raises(ValueError):s.selected(context,True)
    (tmp_path/'lock.json').write_text(json.dumps({'selection_sha256':s.stream_hash(context.run/'selection.json')}))
    assert s.selected(context,True)==record['setting']
    record['selection_null']=True
    with pytest.raises(ValueError):s.selected(context,True)


def test_private_origin_mismatch_rejected_without_any_model_or_API(tmp_path,monkeypatch):
    file=tmp_path/'origin.json';s.save(file,{'private_bindings':{'private':'expected'}});artifact=SimpleNamespace(cfg={'origin_record':str(file)})
    actor=SimpleNamespace(RUN='unused',Context=lambda _:SimpleNamespace(verify=lambda:artifact),accepted_artifacts=lambda _:None,origin_bindings=lambda _:{'private':'different'})
    monkeypatch.setattr(s,'actor',actor)
    with pytest.raises(ValueError):s.artifact_origin()


def test_child_budget_and_early_null_parent_cost_separation_privacy(tmp_path,monkeypatch):
    context=setup(tmp_path,monkeypatch);context.cfg['result']=str(tmp_path/'new.json');parent={'costs':{},'totals':{'saved_generations':350,'teacher_forced_sequences':132,'mc_items':0},'native_starts':43}
    s.save(context.run/'inputs/parent_costs.json',parent);s.save(context.run/'grid/run.json',{'saved_generations':160,'teacher_forced_sequences':0,'mc_items':0,'elapsed_wall_seconds':2});s.pv.stop(context.run,'selection_null')
    s.report(SimpleNamespace(),context);result=s.read_json(tmp_path/'new.json');assert result['status']=='selection_null' and result['child']['totals']['saved_generations']==160 and result['combined_totals']['saved_generations']==510
    assert result['parent']['native_starts']==43 and result['reused_grid_outputs']==120;s.ref.private_export(result)
    for trait in ('rudeness','profanity'):
        context.cfg=cfg(trait);progress={'saved_generations':context.cfg['maximum_new_generations'],'mc_items':64};existing={'saved_generations':0,'mc_items':0}
        with pytest.raises(ValueError):s.charge(context,progress,existing,generations=1)
        with pytest.raises(ValueError):s.charge(context,progress,existing,mc=1)


def test_capability_explicit_batch4_caps_official_scoring_and_language_restore(tmp_path,monkeypatch):
    from langdetect import DetectorFactory
    context=setup(tmp_path,monkeypatch);fixture={
        'general_mc':[{'id':f'g{i}'} for i in range(16)],'physics_mc':[{'id':f'p{i}'} for i in range(16)],
        'gsm8k':[{'id':f'm{i}','user':'Synthetic math.','assistant':'#### 4'} for i in range(8)],
        'ifeval':[{'key':i,'prompt':'Synthetic constraint.'} for i in range(8)]}
    s.save(context.run/'inputs/retention.json',fixture);s.save(context.run/'selection.json',{'synthetic':'lock'})
    monkeypatch.setattr(s,'reference_result',lambda _:{'all_criteria_pass':True});monkeypatch.setattr(s.pv,'strength',lambda _:{'7.raw':torch.ones(16)})
    monkeypatch.setattr(s,'ref',SimpleNamespace(response_hook=lambda *a,**kw:nullcontext()))
    progress={'saved_generations':0,'teacher_forced_sequences':0,'mc_items':0,'generation_wall_seconds':0.};existing={k:0 for k in ('saved_generations','teacher_forced_sequences','mc_items')}
    @contextmanager
    def stage(s,phase):
        directory=context.run/phase;directory.mkdir();yield None,None,directory,progress,existing
    monkeypatch.setattr(s,'gpu_stage',stage);calls=[]
    def mc(model,tok,rows,batch_size):
        assert batch_size==4;calls.append(('mc',len(rows),batch_size));return [{'id':r['id'],'correct':True} for r in rows]
    def generate(model,tok,rows,**kwargs):
        assert kwargs['batch_size']==4 and kwargs['seed']==42 and kwargs['max_new_tokens'] in (512,1024)
        calls.append(('generate',len(rows),kwargs['max_new_tokens']));return {'predictions':[{'id':r['id'],'output':'#### 4','hit_token_limit':False} for r in rows],'timing':{'generation_wall_seconds':1.}}
    def ifeval(rows,predictions):
        assert DetectorFactory.seed==0 and len(rows)==len(predictions)==8
        return {'total':8,'prompt_strict_correct':7,'records':[{'private':'not exported'}]}
    monkeypatch.setattr(s,'hf_multiple_choice',mc);monkeypatch.setattr(s,'generate_batched',generate);monkeypatch.setattr(s,'ifeval_scores',ifeval)
    # GSM8K uses the actual unchanged official local scoring helper on synthetic rows.
    old=DetectorFactory.seed;DetectorFactory.seed=123
    try:s.retention(SimpleNamespace(),context);assert DetectorFactory.seed==123
    finally:DetectorFactory.seed=old
    assert progress['saved_generations']==32 and progress['mc_items']==64 and progress['teacher_forced_sequences']==0
    assert calls.count(('generate',8,512))==calls.count(('generate',8,1024))==2 and calls.count(('mc',16,4))==4
    result=s.read_json(context.run/'retention/metrics.json');assert result['selected']['gsm8k']['strict_correct']==8 and result['baseline']['ifeval']['prompt_strict_correct']==7
    assert 'records' not in result['baseline']['ifeval']
