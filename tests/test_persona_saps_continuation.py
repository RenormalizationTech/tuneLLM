"""CPU-only continuation authority, immutable bank, shared math and journal bounds."""
import copy
from contextlib import nullcontext
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from scripts import persona_saps_continuation as s


def cfg():return s.read_json(s.CONFIG)
def partition():return s.read_json('outputs/persona-saps-sarcasm-01/inputs/split.json')
def rating(id,score=4):return {'id':id,**{k:score for k in s.p.FIELDS}}


def test_fixed_config_exact_parent_seeds_and_shared_operations_no_initial_api():
    c=cfg();s.validate_config(c);parent=s.read_json(c['parent_config'])
    for key in ('plan_seeds','blocks','alphas','test_seed','test_count','custom_count','draw_seed','batch_size','max_new_tokens','revision','gpu_uuid'):
        assert c[key]==parent[key]
    assert c['maximum_generations']==1142 and c['maximum_judge_calls']==136 and c['maximum_whole_starts']==141
    assert s.PHASES==('validation','test','custom')
    assert s.stage_sources is s.shared.stage_sources and s.parse is s.shared.parse and s.p is s.shared.p
    for key,value in [('maximum_generations',1242),('maximum_judge_calls',146),('extraction_authority','parent passed'),('max_new_tokens',61)]:
        wrong=copy.deepcopy(c);wrong[key]=value
        with pytest.raises(ValueError):s.validate_config(wrong)
    with pytest.raises(ValueError):s.phase_data(None,'initial')


@pytest.mark.parametrize('name',['directions','preflight','validation','test','custom','retention','strength.json','selection.json','judging/test','grades/custom'])
def test_any_parent_downstream_output_forbidden(tmp_path,name):
    parent=SimpleNamespace(run=tmp_path);s.parent_paths(parent);(tmp_path/name).mkdir(parents=True)
    with pytest.raises(ValueError):s.parent_paths(parent)


def test_actual_frozen_parent_bank_firstvalid100_failed_gate_h0_and_result_unchanged():
    c=cfg();parent=s.shared.Context(s.path(c['parent']));before=s.stream_hash(s.path(c['parent_result']));bank=s.parent_bank(parent,c)
    assert bank['original_gate_passed'] is False and bank['failed_components']==['clarity']
    assert bank['metrics']['prompt']['clarity']==3.46 and bank['metrics']['prompt']['entailment']==2.44
    assert bank['totals']=={'saved_generations':100,'teacher_forced_sequences':0,'logit_forwards':0,'mc_items':0} and bank['native_starts']==5
    assert s.stream_hash(s.path(c['parent_result']))==before
    h0=s.shared.read_h0(parent,'initial');assert len(h0)==50 and all(set(v)=={11,15,19,23} for v in h0.values())
    assert all(t.shape==(2560,) and t.dtype==torch.float32 for v in h0.values() for t in v.values())


def test_snapshot_requires_byte_exact_parent_inputs_and_preserves_false_gate(tmp_path):
    run=tmp_path/'child';parent=tmp_path/'parent';run.mkdir();parent.mkdir();c=cfg();bank={'original_gate_passed':False,'failed_gate':{'passed':False,'marker':'frozen'},'failed_components':['clarity']}
    s.save(run/'inputs/parent_bank.json',bank);s.save(parent/'eligibility.json',bank['failed_gate']);s.save(parent/'manifest.json',{});s.save(tmp_path/'parent_result.json',{'status':'elicitation_null'});c['parent_result']=str(tmp_path/'parent_result.json')
    for name in ('data','split','plans','draws','rendered','rubric','retention'):
        s.save(parent/'inputs'/(name+'.json'),{'synthetic':name});file=run/'inputs'/(name+'.json');file.parent.mkdir(exist_ok=True);file.write_bytes((parent/'inputs'/(name+'.json')).read_bytes())
    context=s.Context.__new__(s.Context);context.run=run;context.cfg=c;context.parent=SimpleNamespace(run=parent)
    context.m={'files':{},'sources':{},'external_sha256':{},'packages':{},'original_gate_passed':False,'parent_manifest_sha256':s.stream_hash(parent/'manifest.json'),'parent_result_sha256':s.stream_hash(tmp_path/'parent_result.json')}
    context.verify();file=run/'inputs/draws.json';file.write_text('{"synthetic":"new choices"}')
    with pytest.raises(ValueError):context.verify()
    file.write_bytes((parent/'inputs/draws.json').read_bytes());bank['original_gate_passed']=True;(run/'inputs/parent_bank.json').write_text(json.dumps(bank))
    with pytest.raises(ValueError):context.verify()


def test_registered_authority_required_without_fabricated_gate_pass(tmp_path):
    s.save(tmp_path/'inputs/parent_bank.json',{'original_gate_passed':False,'failed_components':['clarity']});calls=[]
    context=SimpleNamespace(run=tmp_path,initial=lambda:calls.append('new committed authority'))
    s.require_authority(context);assert calls==['new committed authority']
    file=tmp_path/'inputs/parent_bank.json';file.write_text(json.dumps({'original_gate_passed':True,'failed_components':[]}))
    with pytest.raises(ValueError):s.require_authority(context)


def test_shared_source_scalar_dtype_broadcast_and_tiny_cache_capture_cleanup():
    p=s.p;u=torch.tensor([1.,0.,0.,0.,0.,0.,0.,0.]);h0=torch.tensor([1.,0.,0.,0.,0.,0.,0.,0.]);target=torch.tensor(100_000_000.)
    c=p.fixed_scalar(target,h0,u);assert type(c) is float and c==99_999_999. and c!=(target-h0.dot(u)).item()
    from transformers import Qwen3Config,Qwen3ForCausalLM
    torch.set_num_threads(1);torch.manual_seed(42)
    model=Qwen3ForCausalLM(Qwen3Config(vocab_size=31,hidden_size=8,intermediate_size=16,num_hidden_layers=2,num_attention_heads=2,num_key_value_heads=1,head_dim=4,pad_token_id=0,eos_token_id=2,max_position_embeddings=64)).eval()
    inputs={'input_ids':torch.tensor([[3,4,5]]),'attention_mask':torch.ones(1,3,dtype=torch.long)};options={'do_sample':False,'use_cache':True,'min_new_tokens':3,'max_new_tokens':3};audit={};capture={}
    with torch.inference_mode():
        original=model.generate(**inputs,**options)
        with p.capture(model,[0,1],capture,first_prefill=True),p.intervention(model,1,u,torch.tensor(1.),1.,'fixed',0.,audit=audit):zero=model.generate(**inputs,**options)
    assert torch.equal(original,zero) and set(capture)=={0,1} and audit['prefill_calls']==1 and audit['decode_calls']==2
    with pytest.raises(RuntimeError):
        with p.intervention(model,1,u,torch.tensor(1.),1.4,'saps'):raise RuntimeError('synthetic cleanup')
    with pytest.raises(ValueError):
        with p.intervention(model,1,u,torch.tensor([1.]),1.4,'saps'):pass
    assert all(not x._forward_hooks for x in model.model.layers) and not model._forward_hooks


def generation(rows):
    return {'predictions':[{'id':r['id'],'output':'Synthetic answer.','input_ids':[4,5],'rendered_prompt':'synthetic '+r['id'],'generated_ids':[3,2],
        'generated_tokens_including_eos':2,'ended_eos':True,'hit_token_limit':False,'batch_index':i,'batch_position':0,'batch_ids':[r['id']],'padded_prompt_width':2} for i,r in enumerate(rows)],
        'batches':[{'batch_index':i,'ids':[r['id']],'padded_prompt_width':2} for i,r in enumerate(rows)],'settings':{'batch_size':1,'max_new_tokens':60,'seed':42,'do_sample':False}}


def setup(tmp_path,monkeypatch,branch='both'):
    run=tmp_path/'child';run.mkdir();c=cfg();split=partition();plans=s.read_json('outputs/persona-saps-sarcasm-01/inputs/plans.json');rubric=s.read_json('evaluation/persona_saps/rubric.json');binary=tmp_path/'native';binary.write_text('synthetic binary never executed')
    m={'sources':{},'binary':str(binary),'cli_version':'2.1.323 (Claude Code)','audit_sha256':'synthetic','tokenizer_ids':{'eos':2,'pad':0},'parent_manifest_sha256':'parent','parent_result_sha256':'parent-result'}
    rendered={name+suffix:{r['id']:{'input_ids':[4,5],'rendered_prompt':'synthetic '+r['id']} for r in split[name]} for name in ('validation','test') for suffix in ('_plain','_prompt')}
    s.save(run/'manifest.json',{'synthetic':True})
    for name,value in [('split',split),('plans',plans),('rubric',rubric),('rendered',rendered)]:s.save(run/'inputs'/(name+'.json'),value)
    arms=['baseline',*(['fixed','saps'] if branch=='both' else [branch]),'random','prompt']
    for phase,names in [('validation',list(s.p.SETTINGS)),('test',arms)]:
        s.save(run/phase/'predictions.json',{n:generation(split['validation' if phase=='validation' else 'test']) for n in names})
        s.save(run/phase/'run.json',{'completed':True,'gpu_uuid':c['gpu_uuid'],'files_sha256':{'predictions.json':s.stream_hash(run/phase/'predictions.json')}})
    context=SimpleNamespace(run=run,cfg=c,m=m,parent=SimpleNamespace(run=tmp_path/'parent'),initial=lambda:None,verify=lambda **kw:None)
    monkeypatch.setattr(s.pv,'BINARY_SHA',s.stream_hash(binary));monkeypatch.setattr(s.shutil,'which',lambda _:str(binary));monkeypatch.setattr(s.f,'version',lambda _:m['cli_version']);monkeypatch.setattr(s.f,'validate_audit',lambda _: {})
    monkeypatch.setattr(s,'stage_sources',lambda context,phase:{'synthetic':phase});monkeypatch.setattr(s,'selected',lambda *a,**kw:{'branch':branch,'setting':{'method':'saps','block':19,'alpha':1.}})
    return context


def calls(monkeypatch,behavior='valid'):
    seen={};records=[]
    def invoke(binary,body):
        records.append(copy.deepcopy(body));h=s.f.canonical_hash(body);seen[h]=seen.get(h,0)+1
        if behavior=='transport':raise OSError('synthetic uncertain start')
        labels=[{'id':r['id'],'profanity':0,'rudeness':0,'sarcasm':0,'gross_failure':False,'content_ok':True,'coherent':True} if 'scale' in body['rubric'] else rating(r['id']) for r in body['items']]
        if behavior=='low_quality':
            for r in labels:r['grammaticality']=2
        if behavior=='exhaust' or behavior=='retry' and seen[h]==1:labels=labels[:-1]
        outer={'subtype':'success','modelUsage':{s.f.MODEL:{}},'result':json.dumps({'ratings':labels})}
        if behavior=='model':outer['modelUsage']={}
        return {'returncode':0,'canonical_stdin_sha256':h,'stdout':json.dumps(outer),'isolation':{'git_not_repository':True,'git_environment_clean':True,'returncode':128}}
    monkeypatch.setattr(s.f,'invoke',invoke);return records


def judge(context,phase,resume=False):
    s.judge(SimpleNamespace(phase=phase,resume=resume),context)
    if hasattr(context,'pending_accept'):s.f.save_or_match(*context.pending_accept);del context.pending_accept


def test68_batches136_maximum_firstvalid_replay_no_initial_calls(tmp_path,monkeypatch):
    context=setup(tmp_path,monkeypatch);records=calls(monkeypatch,'retry')
    for phase in s.PHASES:
        s.packet(SimpleNamespace(phase=phase),context);judge(context,phase);s.read_reviews(context,phase)
    assert len(records)==136 and all(records[i]==records[i+1] for i in range(0,136,2))
    judge(context,'custom',True);assert len(records)==136;s.audit_journals(context)
    assert not (context.run/'initial').exists() and not (context.run/'grades/initial').exists()


@pytest.mark.parametrize('branch',['fixed','saps'])
def test_absent_method_plans_and_custom100_saved_no_added_generation(tmp_path,monkeypatch,branch):
    context=setup(tmp_path,monkeypatch,branch);records=calls(monkeypatch);s.packet(SimpleNamespace(phase='custom'),context)
    assert len(s.bodies(context,'custom'))==10 and sum(b['expected_count'] for b in s.bodies(context,'custom'))==80
    before=s.stream_hash(context.run/'test/predictions.json');judge(context,'custom');result=s.custom_result(context)
    assert len(result['metrics'])==4 and all(v['n']==20 for v in result['metrics'].values())
    assert s.stream_hash(context.run/'test/predictions.json')==before and not (context.run/'custom').exists()


@pytest.mark.parametrize('behavior,n',[('exhaust',2),('model',1),('transport',1)])
def test_fatal_schema_model_transport_no_additional_calls(tmp_path,monkeypatch,behavior,n):
    context=setup(tmp_path,monkeypatch);records=calls(monkeypatch,behavior);s.packet(SimpleNamespace(phase='validation'),context)
    with pytest.raises((s.f.InvalidSubmission,s.f.ModelMismatch,OSError)):
        try:judge(context,'validation')
        except BaseException as exc:s.pv.stop(context.run,'continuation_live_stage_fatal',exc);raise
    with pytest.raises(RuntimeError):s.finish_grades(context,'validation',s.bodies(context,'validation'),context.run/'grades/validation','binding')
    assert len(records)==n


def test_resume_absent_packet_and_receipt_tampering(tmp_path,monkeypatch):
    context=setup(tmp_path,monkeypatch);records=calls(monkeypatch);s.packet(SimpleNamespace(phase='validation'),context)
    with pytest.raises(ValueError):judge(context,'validation',True)
    assert not records and not (context.run/'grades').exists()
    judge(context,'validation');file=context.run/'grades/validation/batch-000-attempt-0-receipt.json';v=s.read_json(file);v['start_sha256']='wrong';file.write_text(json.dumps(v))
    with pytest.raises(ValueError):s.read_reviews(context,'validation')
    file=context.run/'judging/validation/packet.json';v=s.read_json(file);v[0][0]['generated_sentence']='Changed';file.write_text(json.dumps(v))
    file=context.run/'judging/validation/manifest.json';v=s.read_json(file);v['packet_sha256']=s.stream_hash(context.run/'judging/validation/packet.json');file.write_text(json.dumps(v))
    with pytest.raises(ValueError):s.bodies(context,'validation')


def test_fresh_budget_before_start_or_generation_and_parent_path_guard(tmp_path,monkeypatch):
    context=setup(tmp_path,monkeypatch);records=calls(monkeypatch);s.packet(SimpleNamespace(phase='validation'),context);directory=context.run/'grades/validation';directory.mkdir(parents=True)
    for i in range(136):s.save(context.run/'grades/other'/f'batch-{i:03d}-attempt-0-start.json',{})
    with pytest.raises(RuntimeError):s.finish_grades(context,'validation',s.bodies(context,'validation'),directory,'bound')
    assert not records and not list(directory.glob('*-start.json'))
    existing={'saved_generations':1142,'teacher_forced_sequences':1500,'logit_forwards':2,'mc_items':64};zero={k:0 for k in existing}
    for k in existing:
        with pytest.raises(ValueError):s.charge(zero,existing,**{k:1})
    with pytest.raises(ValueError):s.guard(s.path('outputs/persona-saps-sarcasm-01'))


def test_new_selection_lock_before_test_no_old_registration_authority(tmp_path,monkeypatch):
    actual=s.selected;context=setup(tmp_path,monkeypatch);monkeypatch.setattr(s,'selected',actual)
    record={'selection_null':False,'setting':{'method':'saps','block':19,'alpha':1.},'branch':'both'};context.cfg['selection_registration']=str(tmp_path/'newlock.json')
    s.save(context.run/'selection.json',record);monkeypatch.setattr(s,'selection_result',lambda _:record);monkeypatch.setattr(s,'committed',lambda _:None);monkeypatch.setattr(s,'strength',lambda _: {})
    with pytest.raises(FileNotFoundError):s.selected(context,True)
    s.save(tmp_path/'newlock.json',{'selection_sha256':'wrong'})
    with pytest.raises(ValueError):s.selected(context,True)
    (tmp_path/'newlock.json').write_text(json.dumps({'selection_sha256':s.stream_hash(context.run/'selection.json')}));assert s.selected(context,True)==record


@pytest.mark.parametrize('cause',['selection_null','continuation_live_stage_fatal'])
def test_parent_failed_bank_and_fresh_costs_separate_in_private_safe_report(tmp_path,monkeypatch,cause):
    context=setup(tmp_path,monkeypatch);context.cfg['result']=str(tmp_path/'new.json');bank={'metrics':{'baseline':{'sarcasm':0},'prompt':{'sarcasm':3.04,'clarity':3.46,'entailment':2.44}},
        'costs':{'initial':{'saved_generations':100,'teacher_forced_sequences':0,'logit_forwards':0,'mc_items':0}},'totals':{'saved_generations':100,'teacher_forced_sequences':0,'logit_forwards':0,'mc_items':0}}
    s.save(context.run/'inputs/parent_bank.json',bank);s.pv.stop(context.run,cause)
    file=context.run/'validation/run.json';file.write_text(json.dumps({'completed':True,'saved_generations':600,'teacher_forced_sequences':0,'logit_forwards':0,'mc_items':0}))
    s.report(SimpleNamespace(),context);result=s.read_json(tmp_path/'new.json')
    assert result['parent']['original_gate_passed'] is False and result['parent']['status']=='elicitation_null' and result['parent']['native_starts']==5
    assert result['child']['totals']['saved_generations']==600 and result['whole_totals']['saved_generations']==700 and result['whole_native_starts']==5
    assert result['status']==('selection_null' if cause=='selection_null' else 'incomplete');s.private_export(result)


def test_actual_select_null_writes_stop_then_complete_null_aggregate(tmp_path,monkeypatch):
    context=setup(tmp_path,monkeypatch);context.cfg['result']=str(tmp_path/'null_result.json')
    gate=s.read_json('outputs/persona-saps-sarcasm-01/eligibility.json')
    bank={'original_gate_passed':False,'failed_components':['clarity'],'failed_gate':gate,'metrics':gate['metrics'],
        'costs':{'initial':{'saved_generations':100,'teacher_forced_sequences':0,'logit_forwards':0,'mc_items':0}},
        'totals':{'saved_generations':100,'teacher_forced_sequences':0,'logit_forwards':0,'mc_items':0}}
    s.save(context.run/'inputs/parent_bank.json',bank);s.save(context.run/'strength.json',{'synthetic':'registered vector lock'})
    file=context.run/'validation/run.json';runtime=s.read_json(file);runtime.update(saved_generations=600,teacher_forced_sequences=0,logit_forwards=0,mc_items=0);file.write_text(json.dumps(runtime))
    native=calls(monkeypatch,'low_quality');s.packet(SimpleNamespace(phase='validation'),context);judge(context,'validation')
    # Actual selection_result aggregates every parsed first-valid score and invokes
    # the unchanged SAPS quality/selection rule; only the already-tested vector
    # registration prerequisite is mocked. select/report themselves are not stubs.
    monkeypatch.setattr(s,'strength',lambda _: {})
    s.select(SimpleNamespace(),context)
    selection=s.read_json(context.run/'selection.json')
    assert selection['selection_null'] is True and selection['best']=={'fixed':None,'saps':None} and selection['parent_gate_remains_failed'] is True
    assert s.read_json(context.run/'stopped.json')=={'stop_cause':'selection_null','error_type':None}
    s.report(SimpleNamespace(),context);result=s.read_json(tmp_path/'null_result.json')
    assert result['status']=='selection_null' and result['child']['native_starts']==30 and result['whole_native_starts']==35
    assert result['parent']['original_gate_passed'] is False and result['parent']['status']=='elicitation_null'
    assert result['child']['totals']['saved_generations']==600 and result['whole_totals']['saved_generations']==700
    assert len(native)==30 and not (context.run/'grades/test').exists() and not (context.run/'grades/custom').exists()
    s.private_export(result)
