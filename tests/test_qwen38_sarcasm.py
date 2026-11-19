"""Offline study guards, frozen sizes, boundary failures and immutable receipts."""
import pytest
from scripts import qwen38_sarcasm as r


def test_fixed_counts_and_no_normalized_overlap():
    cfg=r.read_json(r.CONFIG);data=r.rows(cfg)
    assert [len(data[n]) for n in ['elicitation','development','final']]==[200,128,128]
    assert [x['side'] for x in data['elicitation'][:2]]==['pos','neg']
    assert cfg['maximum_generated_tokens']==200*256+256*160+12*32
    assert data['final'][1]['arm']=='prompted'


def test_started_row_is_not_retried_and_completed_hash_is_checked(tmp_path):
    s=object.__new__(r.Context);s.run=tmp_path;s.cfg={'max_gpu_seconds':28800}
    r.save(tmp_path/'manifest.json',{})
    def failed():raise RuntimeError('interrupted')
    with pytest.raises(RuntimeError,match='interrupted'):s.task('test','row',failed)
    with pytest.raises(RuntimeError,match='no automatic retry'):s.task('test','row',lambda:{'ok':True})
    assert s.task('test','good',lambda:{'ok':True})['ok']
    assert s.task('test','good',lambda:pytest.fail('must reuse'))['ok']
    (tmp_path/'test/rows/good.json').write_text('{}')
    with pytest.raises(ValueError,match='hash differs'):s.task('test','good',lambda:{})


def test_registration_requires_exact_manifest_stage_and_dependencies(tmp_path):
    s=object.__new__(r.Context);s.run=tmp_path
    r.save(tmp_path/'manifest.json',{})
    r.save(tmp_path/'approvals/synthetic-smoke.json',{'stage':'synthetic-smoke','manifest_sha256':r.sha(tmp_path/'manifest.json'),'dependencies':{}})
    s.approve('synthetic-smoke')
    with pytest.raises(ValueError,match='registration'):s.approve('synthetic-smoke',{'changed':'hash'})


def test_historical_scoring_contract_and_eos_at_cap():
    from scripts import sarcasm_method_transfer_common as scientific
    assert not r.completion_metadata([3,2],2,2)['hit_token_limit']
    assert r.completion_metadata([3,4],2,2)['hit_token_limit']
    rows=[dict(id=str(i),output='Try checking the connection before restarting the device.',
               **r.completion_metadata([3,4,2],2,3)) for i in range(32)]
    scores={str(i):{'trait':70,'coherence':80} for i in range(32)}
    value=scientific.summarize(rows,scores)
    assert value['trait_conservative']==70 and value['coherence']==80 and value['caps']==0 and value['eos']==32
    base={**value,'trait_conservative':10}
    assert scientific.gates(base,value)['all_pass']


def test_stage_budgets_and_judge_plan_packing(tmp_path):
    from scripts.persona_reference_common import make_plan
    cfg=r.read_json(r.CONFIG);data=r.rows(cfg)
    batches=make_plan([f'x{i:02d}' for i in range(20)],
        [f'i{i}_{side}' for i in range(5) for side in ['pos','neg']],9840)
    assert len(batches)==25 and all(len(b)==8 and len({e['question_id'] for e in b})==8 for b in batches)
    s=object.__new__(r.Context);s.run=tmp_path;s.cfg=cfg
    for i in range(2):s.reserve('teacher_forced','synthetic-smoke',str(i))
    with pytest.raises(RuntimeError,match='per-stage'):s.reserve('teacher_forced','synthetic-smoke','2')


def test_raw_trajectory_boundary_special_mask_and_judged_text_equality():
    class Tok:
        all_special_ids=[0,2,99]
        def __call__(self,text,**kwargs):
            assert text in ['<|im_start|>system\ns<|im_end|>\n','<|im_start|>system\ns<|im_end|>\nu']
            return {'input_ids':[10,11] if text.endswith('\n') else [10,11,12]}
        def decode(self,ids,skip_special_tokens):return ''.join({3:'A',4:'B'}.get(i,'') for i in ids)
    tok=Tok();prompt='<|im_start|>system\ns<|im_end|>\nu'
    ids,sl,pl,mask=r.boundary(tok,prompt,[10,11,12],[3,99,4,2],'AB',[2])
    assert ids==[10,11,12,3,99,4] and sl==2 and pl==3
    assert mask==[False,False,False,True,False,True]
    with pytest.raises(ValueError,match='decode differs'):r.boundary(tok,prompt,[10,11,12],[3,2],'different',[2])
    with pytest.raises(ValueError,match='prefix mismatch'):r.boundary(tok,prompt,[10,11,13],[3,2],'A',[2])
    with pytest.raises(ValueError,match='empty response'):r.boundary(tok,prompt,[10,11,12],[99,2],'',[2])


def test_frozen_rendered_prompt_and_probe_ids_contract():
    class Tok:
        def apply_chat_template(self,*args,**kwargs):return 'frozen prompt'
        def __call__(self,*args,**kwargs):return {'input_ids':[3,4]}
    s=type('S',(),{'rendered':{'C01':{'prompt':'frozen prompt','input_ids':[3,4]}}})()
    assert r.frozen_rendering(s,Tok(),{'id':'C01','user':'probe'})==('frozen prompt',[3,4])
    s.rendered['C01']['input_ids']=[3,5]
    with pytest.raises(ValueError,match='frozen rendered'):r.frozen_rendering(s,Tok(),{'id':'C01','user':'probe'})


def test_frozen_rubrics_are_historical_bytes_and_judge_schema(monkeypatch):
    from scripts import qwen38_sarcasm_judge as judge
    rubric=r.rubrics()
    assert rubric['trait']==r.read_json('evaluation/sarcasm_method_transfer/extract.json')['eval_prompt']
    assert rubric['coherence'] and rubric['impolite']
    monkeypatch.setattr(judge.native,'outer_result',lambda attempt:({'result':attempt},['claude-opus-5-5']))
    assert judge.parse({'ratings':[{'id':'opaque','score':70}]},['opaque'],'coherence')=={'opaque':70}
    assert judge.parse({'ratings':[{'id':'opaque','score':'REFUSAL'}]},['opaque'],'trait')=={'opaque':None}
    with pytest.raises(judge.native.InvalidSubmission):judge.parse({'ratings':[{'id':'wrong','score':70}]},['opaque'],'trait')
