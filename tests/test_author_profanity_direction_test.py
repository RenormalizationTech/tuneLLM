import ast
from contextlib import nullcontext
import copy
import json
from types import SimpleNamespace

import pytest

from scripts import author_profanity_direction_test as a
from scripts.common import path,read_json


def test_frozen_helpers_identity_and_no_tf_or_artifact_interface():
    for name in ('generate','charge','judge_local','target_values','secondary_plan'):
        assert getattr(a,name) is getattr(a.method,name)
    assert a.LIMITS=={'target':394,'local_judge':408,'logit':2,'teacher_forced':0}
    tree=ast.parse(path(a.__file__).read_text())
    names={node.name for node in tree.body if isinstance(node,ast.FunctionDef)}
    assert not names.intersection({'extract','artifact','preview','promote'})
    gpu=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='gpu')
    assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id in {'extract','artifact'} for n in ast.walk(gpu))
    assert 'teacher_forced' not in ast.unparse(gpu).split('try:')[1].split('finally:')[0]


def test_mechanics_body_same_except_reuse_authority():
    def function(module):
        tree=ast.parse(path(module.__file__).read_text())
        return next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='mechanics')
    old=function(a.method);new=function(a)
    for node in ast.walk(old):
        if isinstance(node,ast.Name):node.id={'extraction_lock':'active_direction','vector_bank':'vectors'}.get(node.id,node.id)
    assert ast.dump(old,include_attributes=False)==ast.dump(new,include_attributes=False)


def test_actual_saved_features_grouped_reuse_and_parent_bytes_unchanged():
    import torch
    root=path('outputs/author-profanity-revision-01')
    names=[root/'extract/vectors.pt',root/'extract/decodability.json',root/'inputs/folds.json',root/'stopped.json',path('results/author_profanity_revision_01.json')]
    before={str(f):a.stream_hash(f) for f in names}
    bank=torch.load(names[0],map_location='cpu',weights_only=True)
    result=a.grouped_calculation(bank,read_json(names[2]),read_json(names[1]))
    assert result['selected_hidden_index']==7 and result['decoder_block']==6 and result['grouped_auc']==1
    assert result['parent_row_gate_remains_failed'] and result['decodable_known_before_child']
    assert result['exact_parent_selected_tensor_sha256']==a.tensor_hash(bank['selected'])
    assert read_json(names[-1])['status']=='extraction_null'
    assert before=={str(f):a.stream_hash(f) for f in names}
    changed=dict(bank);changed['selected']=-bank['selected']
    with pytest.raises(ValueError,match='raw vector'):a.grouped_calculation(changed,read_json(names[2]),read_json(names[1]))


@pytest.mark.parametrize('key,value',[('maximum_teacher_forced',24),('maximum_native_starts',42),('target_cap',64),('coefficients',[1,2,3,4,6,10]),('run',a.parent_source.RUN)])
def test_fixed_settings_and_old_path_rejection(key,value):
    cfg=read_json(a.CONFIG);a.validate_config(cfg);cfg[key]=value
    with pytest.raises(ValueError):a.validate_config(cfg)
    with pytest.raises(ValueError):a.guard(path(a.parent_source.RUN))


def test_mechanics_ten_probes_two_logits_actual_raw_hook_and_cleanup(tmp_path,monkeypatch):
    import torch
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__();self.model=SimpleNamespace(layers=torch.nn.ModuleList([torch.nn.Identity() for _ in range(7)]))
            self.weight=torch.nn.Parameter(torch.ones(1));self.device=torch.device('cpu')
        def forward(self,**kwargs):
            hidden=torch.ones(2,2,4)*self.weight
            for layer in self.model.layers:hidden=layer(hidden)
            return SimpleNamespace(logits=hidden)
    class Inputs(dict):
        def to(self,device):return self
    model=Model();direction=torch.tensor([1.,-2.,3.,-4.]);calls=[];charges=[]
    monkeypatch.setattr(a,'active_direction',lambda s:{'decoder_block':6})
    monkeypatch.setattr(a,'vectors',lambda s:{'selected':direction})
    monkeypatch.setattr(a,'charge',lambda s,out,kind,count,inputs:charges.append((kind,count)))
    def generate(s,model,tok,rows,out,progress,**kw):
        calls.append(kw);audit={};vector=kw['vector'];ctx=nullcontext() if vector is None else a.p.raw_shift(model,6,vector,kw['coefficient'],audit)
        with ctx:
            logits=model().logits[:,-1,:].detach();model()
        progress['target']+=len(rows)
        return {'records':[{'id':r['id'],'generated_ids':[int(logits[0,0].item()),9],'rendered_prompt':'fixed'} for r in rows],
                'batches':[{'hook_audit':audit}]},logits
    monkeypatch.setattr(a,'generate',generate)
    tok=lambda texts,**kwargs:Inputs(input_ids=torch.ones(2,2,dtype=torch.long),attention_mask=torch.ones(2,2,dtype=torch.long))
    progress=dict.fromkeys(a.LIMITS,0)
    a.mechanics(None,model,tok,tmp_path,progress,{'triples':[{'id':str(i),'user':'fixed'} for i in range(12)]})
    result=read_json(tmp_path/'result.json')
    assert result['all_pass'] and result['block']==6 and result['direction_sha256']==a.tensor_hash(direction)
    assert len(calls)==5 and all(c['cap']==16 and c['kind']=='target' and c['logits'] for c in calls)
    assert progress=={'target':10,'local_judge':0,'logit':2,'teacher_forced':0} and charges==[('logit',1),('logit',1)]
    assert calls[0]['vector'] is None and torch.equal(calls[2]['vector'],direction) and torch.equal(calls[4]['vector'],direction)
    assert all(not layer._forward_hooks for layer in model.model.layers)


def test_mechanics_parity_failure_closes_before_success(tmp_path,monkeypatch):
    # A failed actual probe is never silently accepted/retried.
    monkeypatch.setattr(a,'active_direction',lambda s:(_ for _ in ()).throw(ValueError('reused direction mismatch')))
    with pytest.raises(ValueError,match='direction mismatch'):a.mechanics(None,None,None,tmp_path,{}, {})
    assert not list(tmp_path.iterdir())


def test_missing_release_and_generation_reentry_no_work(tmp_path):
    cfg={'initial_registration':str(tmp_path/'missing')};s=SimpleNamespace(run=tmp_path,cfg=cfg)
    with pytest.raises(a.PreconditionWait):a.preconditions(s,'mechanics')
    (tmp_path/'mechanics').mkdir()
    with pytest.raises(a.PreconditionWait,match='cannot rerun'):a.preconditions(s,'mechanics')


def test_lexical_readout_report_only_selection_unchanged(monkeypatch):
    metrics={arm:{'self_judge':{'filled_means':{'profanity':0,'toxicity':0},'profanity_missing_over10percent':False},'mean_distinct_bigram':1} for arm in ['baseline',*['c'+str(c) for c in a.p.COEFFICIENTS]]}
    monkeypatch.setattr(a.method,'condition_metrics',lambda s,name:copy.deepcopy(metrics))
    monkeypatch.setattr(a,'target_values',lambda s,name:{arm:{'records':[{'output':'Damn, a class assessment.' if i<3 else 'A scrap book.'} for i in range(32)]} for arm in metrics})
    measured=a.condition_metrics(None,'calibration')
    assert all(v['lexicon_response_hit_count']==3 and v['lexicon_response_hit_rate']==3/32 for v in measured.values())
    assert a.p.selection(metrics)==a.p.selection(measured) and a.p.selection(measured)['calibration_null']


def report_context(tmp_path):
    a.save(tmp_path/'manifest.json',{});a.save(tmp_path/'reuse/grouped.json',{'layers':{'7':{'grouped':{'pooled_oof_auc':1.,'oof_scores':[1.,0.]}}}})
    a.save(tmp_path/'reuse/origin.json',{'parent_status':'extraction_null','reused_teacher_forced_responses':24})
    return SimpleNamespace(run=tmp_path,m={'sources':{}},cfg={})


def test_calibration_null_report_all_strengths_costs_privacy_and_fatal_override(tmp_path,monkeypatch):
    s=report_context(tmp_path);choice={'calibration_null':True,'coefficient':None}
    a.save(tmp_path/'selection.json',choice);a.save(tmp_path/'stopped.json',{'stop_cause':'calibration_null'})
    a.save(tmp_path/'calibration/run.json',{'target':224,'teacher_forced':0});a.save(tmp_path/'calibration_judge/run.json',{'local_judge':224})
    rows={arm:{'lexicon_response_hit_count':0,'lexicon_response_hit_rate':0} for arm in ['baseline',*['c'+str(c) for c in a.p.COEFFICIENTS]]}
    monkeypatch.setattr(a,'choose',lambda s:choice);monkeypatch.setattr(a,'condition_metrics',lambda s,name:rows)
    result=a.report(s)
    assert result['status']=='calibration_null' and result['behavioral_calibration_completed']
    assert len(result['calibration'])==7 and result['fresh_totals']['target']==224 and result['fresh_totals']['teacher_forced']==0
    assert 'oof_scores' not in json.dumps(result) and 'generated_ids' not in json.dumps(result)
    (tmp_path/'stopped.json').unlink();a.save(tmp_path/'stopped.json',{'stop_cause':'fatal_execution_or_integrity'})
    assert a.report(s)['status']=='incomplete'


def test_primary_complete_secondary_pending_no_native_required(tmp_path,monkeypatch):
    s=report_context(tmp_path);a.save(tmp_path/'final_judge/integrity.json',{})
    means={'baseline':0,'selected':2,'random8403':1,'random8404':1,'random8405':1}
    rows={arm:{'self_judge':{'filled_means':{'profanity':x,'toxicity':x}}} for arm,x in means.items()}
    monkeypatch.setattr(a,'condition_metrics',lambda s,name:rows)
    monkeypatch.setattr(a,'read_secondary',lambda *args:pytest.fail('no secondary available/authorized'))
    result=a.report(s)
    assert result['status']=='primary_complete_secondary_pending'
    assert result['heldout_effects_primary_local']['profanity']['selected_minus_baseline']==2
    assert result['heldout_effects_primary_local']['profanity']['selected_minus_mean3random']==1


def test_secondary_stray_budget_and_uncertain_start_rejected(tmp_path,monkeypatch):
    s=SimpleNamespace(run=tmp_path);out=tmp_path/'secondary';out.mkdir();a.save(out/'run_binding.json',{})
    monkeypatch.setattr(a,'binding',lambda s:{});monkeypatch.setattr(a,'secondary_bodies',lambda s:[{}]*20)
    a.save(out/'batch-000-attempt-0-start.json',{})
    with pytest.raises(a.f.UncertainAttempt):a.read_secondary(s,True)
    for file in out.glob('*start.json'):file.unlink()
    a.save(out/'batch-000-attempt-2.json',{})
    with pytest.raises(ValueError,match='stray'):a.read_secondary(s,True)
    (out/'batch-000-attempt-2.json').unlink()
    for i in range(41):a.save(out/f'extra-{i}-start.json',{})
    with pytest.raises(ValueError,match='budget'):a.read_secondary(s,True)
