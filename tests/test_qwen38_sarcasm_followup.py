"""CPU-only follow-up telemetry parity and exact continuation contracts."""
import pytest
import torch
from scripts import qwen38_gcad_port as frozen
from scripts import qwen38_sarcasm_followup_telemetry as t
from tests.test_qwen38_gcad_port import tiny,tensors


@pytest.fixture(autouse=True)
def threads():
    old=torch.get_num_threads();torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


@pytest.mark.parametrize('strength',[0.,2.,16.])
def test_observers_bit_identical_to_frozen_logits_and_cached_tokens(strength):
    model=tiny();bank=tensors(model);ids=torch.tensor([[3,4,5]])
    original=t.signature(model)
    kwargs=dict(use_cache=True,do_sample=False,max_new_tokens=4,eos_token_id=None,pad_token_id=0)
    with torch.no_grad():
        with frozen.Steerer(model,bank,[1],strength):
            logits=model(ids,use_cache=False).logits.clone()
        with t.InstrumentedSteerer(model,bank,[1],strength) as instrument:
            assert torch.equal(logits,model(ids,use_cache=False).logits)
        with frozen.Steerer(model,bank,[1],strength):tokens=model.generate(ids,**kwargs)
        with t.InstrumentedSteerer(model,bank,[1],strength) as instrument:
            observed=model.generate(ids,**kwargs)
            assert torch.equal(tokens,observed)
            stats=instrument.summary()['1']
            assert stats['prefill']['tokens']==3 and stats['prefill']['calls']==1
            assert stats['decode']['tokens']==3 and stats['decode']['calls']==3
            assert instrument.audit=={1:{'prefill':1,'decode':3,'errors':0}}
            if strength==0:
                assert stats['prefill']['metrics']['actual_norm']['sum']==0
            else:assert stats['prefill']['metrics']['actual_norm']['sum']>0
    assert t.signature(model)==original


def test_requested_actual_rounding_and_token_weighted_means():
    original=torch.tensor([[[1000.,1000.]]],dtype=torch.bfloat16)
    requested=torch.full_like(original,.001)
    steered=original+requested
    scalar=torch.ones(1,1)
    metrics,zero=t.token_metrics(original,steered,requested,scalar,scalar,scalar,scalar)
    assert metrics['requested_norm'].item()>0 and metrics['actual_norm'].item()==0
    stats=t.Moments()
    stats.add({'native_norm':torch.tensor([2.,4.])},torch.tensor([False,False]))
    stats.add({'native_norm':torch.tensor([12.])},torch.tensor([False]))
    assert stats.summary()['metrics']['native_norm']['mean']==6
    assert stats.summary()['tokens']==3


def test_observers_never_hook_linear_blocks_and_cleanup_on_failure():
    model=tiny();original=t.signature(model)
    with pytest.raises(RuntimeError,match='synthetic stop'):
        with t.InstrumentedSteerer(model,tensors(model),[1],2):
            assert all(not m._forward_hooks and not m._forward_pre_hooks for m in model.model.layers[0].modules())
            raise RuntimeError('synthetic stop')
    assert t.signature(model)==original


def test_no_gcad_activation_capture_uses_response_mask_at_every_block():
    model=tiny();ids=torch.tensor([[3,4,5,6]])
    before=t.signature(model)
    with torch.no_grad(),t.ActivationCapture(model,[False,False,True,True],[1]) as capture:
        model(ids,use_cache=False)
    assert set(capture.values)=={'block.0','block.1','block.2','attention.1'}
    assert all(v['tokens']==2 and v['finite'] for v in capture.values.values())
    assert t.signature(model)==before


def test_replay_preserves_identical_raw_continuations_across_contexts():
    class Tok:
        all_special_ids=[2,99]
        def decode(self,ids,skip_special_tokens=True):return ''.join({3:'A',4:'B'}.get(i,'') for i in ids)
        def __call__(self,*a,**k):pytest.fail('continuation must never be retokenized')
    a,ma=t.replay_ids(Tok(),[10,11],[3,99,4,2],'AB',[2])
    b,mb=t.replay_ids(Tok(),[12,13,14,15],[3,99,4,2],'AB',[2])
    assert a[2:]==b[4:]==[3,99,4]
    assert ma==[False,False,True,False,True]
    assert mb==[False]*4+[True,False,True]
    with pytest.raises(ValueError,match='decode differs'):t.replay_ids(Tok(),[10],[3,2],'bad',[2])
    with pytest.raises(ValueError,match='empty raw'):t.replay_ids(Tok(),[10],[99,2],'',[2])


def test_finite_validation_and_zero_native_ratio_denominator():
    stats=t.Moments();stats.add({'actual_ratio':torch.tensor([9.,2.])},torch.tensor([True,False]))
    result=stats.summary()
    assert result['zero_native_norm_tokens']==1
    assert result['metrics']['actual_ratio']['mean']==2
    stats=t.Moments();stats.add({'gate':torch.tensor([float('nan')])},torch.tensor([False]))
    with pytest.raises(ValueError,match='nonfinite'):stats.summary()


def test_diagnostic_neutrality_crop_parity_and_native_token_means():
    model=tiny();bank=tensors(model);ids=torch.tensor([[3,4,5,6,7]])
    mask=[False,False,False,True,True];positions=torch.tensor([2,3])
    manual={};hooks=[]
    for i,b in enumerate(model.model.layers):
        hooks.append(b.register_forward_hook(lambda m,a,o,i=i:manual.__setitem__(i,o.detach().clone())))
    with torch.no_grad():ordinary=model(ids,use_cache=False,logits_to_keep=positions).logits.clone()
    for h in hooks:h.remove()
    before=t.signature(model)
    result,logits=t.diagnostic_forward(model,ids,mask,2,bank,[1])
    assert torch.equal(ordinary,logits)
    assert t.signature(model)==before
    for i in range(3):torch.testing.assert_close(result['vectors'][f'block.{i}'],manual[i][0,3:].float().mean(0))
    reference=frozen.extract(model,ids,2,3,mask,[1])
    torch.testing.assert_close(result['vectors']['crop.1'],reference[1]['delta'])
    assert result['vectors']['attention_tokens.1'].shape==(2,16)
    assert result['nll']>0 and result['gate']['1']['gate']['tokens']==2
    no_system,_=t.diagnostic_forward(model,ids,mask,None,bank,[1])
    assert not any(key.startswith(('crop.','attention_tokens.')) for key in no_system['vectors'])


def test_retain_last_manual_requested_and_actual_norms():
    model=tiny();bank=tensors(model);ids=torch.tensor([[3,4,5]])
    with torch.no_grad(),t.InstrumentedSteerer(model,bank,[1],2.,retain_last=True) as ctx:
        model(ids,use_cache=False)
        last=ctx.last[1];logged=ctx.trace[1]['prefill'][-1]
        torch.testing.assert_close(last['requested'].float().norm(dim=-1).flatten(),logged['requested_norm'],rtol=1e-5,atol=0)
        torch.testing.assert_close((last['steered'].float()-last['original'].float()).norm(dim=-1).flatten(),logged['actual_norm'],rtol=1e-5,atol=0)


def test_bootstrap_gram_matches_direct_vectors_and_shared_cluster_means():
    import numpy as np
    from scripts import qwen38_sarcasm_followup_analyze as a
    x=np.array([[1.,0.],[0.,2.],[1.,1.],[2.,1.]])
    direction=np.array([1.,2.]);w=a.weights(4,100,19)
    assert np.array_equal(w,a.weights(4,100,19))
    direct=w@x;cos=(direct@direction)/(np.linalg.norm(direct,axis=1)*np.linalg.norm(direction))
    assert a.cosine_ci(x,direction,w)['bootstrap95']==pytest.approx(np.quantile(cos,[.025,.975]))
    scalar=np.array([1.,2.,3.,4.])
    assert a.mean_ci(scalar,w)['bootstrap95']==pytest.approx(np.quantile(w@scalar,[.025,.975]))
    paired=a.paired_token_effect(np.array([[1.,1.],[-1.,1.]]),np.array([[0.,1.],[0.,1.]]))
    assert paired['mean_token_difference_norm']==1 and paired['norm_of_mean_difference']==0
    assert paired['mean_token_difference_to_native_L_ratio']==1


def test_plan_packing_7arms_and_frozen_gate_semantics():
    from scripts.persona_reference_common import make_plan
    from scripts.common import read_json
    from scripts import qwen38_sarcasm_followup_analyze as a
    cfg=read_json('configs/qwen38_sarcasm_followup.json')
    plans=make_plan([f'F{i:02d}' for i in range(1,33)],cfg['arms'],20261120)
    assert len(plans)==28 and all(len(batch)==8 and len({e['question_id'] for e in batch})==8 for batch in plans)
    assert len({e['id'] for batch in plans for e in batch})==224
    rows=[];scores={}
    for qi in range(32):
        for arm in cfg['arms']:
            key=f'F{qi:02d}_{arm}';rows.append({'id':key,'arm':arm,'output':'Check the connection and restart the device carefully.',
                'generated_ids':[3,4,2],'hit_token_limit':False,'ended_eos':True,'telemetry':None})
            scores[key]={'trait':10 if arm=='baseline' else 70,'coherence':80,'impolite':5}
    result=a.part_b(rows,scores,cfg)
    assert result['outcome']=='exploratory_candidate' and result['exploratory_candidate']==2
    assert not result['b16_random_margin20']
    for s in scores.values():s['trait']=10
    assert a.part_b(rows,scores,cfg)['outcome']=='exploratory_null_through_b16'


def test_new_journal_resume_uncertain_guard_and_stage_budget(tmp_path):
    from scripts import qwen38_sarcasm_followup as run
    from scripts.tone_judge import save
    from scripts.common import read_json
    s=object.__new__(run.Context);s.run=tmp_path;s.cfg=read_json(run.CONFIG);s.deadline=None
    save(tmp_path/'manifest.json',{})
    calls=[]
    assert s.task('synthetic','good',lambda:calls.append(1) or {'ok':True})=={'ok':True}
    assert s.task('synthetic','good',lambda:pytest.fail('must not call twice'))=={'ok':True}
    assert len(calls)==1
    def fail():raise RuntimeError('known failure')
    with pytest.raises(RuntimeError):s.task('synthetic','bad',fail)
    with pytest.raises(run.Uncertain):s.task('synthetic','bad',lambda:{})
    for i in range(2):s.reserve('teacher_forced','mechanics',str(i))
    with pytest.raises(run.BudgetStop):s.reserve('teacher_forced','mechanics','too-many')
    with pytest.raises(run.BudgetStop):s.stage_budget(1)
