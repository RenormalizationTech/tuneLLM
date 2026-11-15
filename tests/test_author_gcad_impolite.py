import ast
from contextlib import nullcontext
import json
from pathlib import Path
import random
import sys
from types import SimpleNamespace

import pytest
import torch

from scripts import author_gcad_impolite as a
from scripts import author_gcad_impolite_common as p
from scripts.common import path,read_json

AUTHOR=path('outputs/persona-author-replication-audit-01/gcad')


@pytest.fixture(autouse=True)
def one_thread():
    previous=torch.get_num_threads();torch.set_num_threads(1)
    try: yield
    finally: torch.set_num_threads(previous)


def test_whitelisted_ast_and_full_question_plan(tmp_path):
    source=AUTHOR/'eval/eval_multiturn_pipeline.py';functions,hashes=p.author_functions(source)
    # Dangerous top-level setup is not executed; exact function AST is preserved.
    copy=tmp_path/'pipeline.py';copy.write_text('raise AssertionError("top-level must not execute")\n'+source.read_text())
    copied,other=p.author_functions(copy);assert hashes==other
    questions=read_json(AUTHOR/'data/trait_data_eval/impolite.json')['questions'];plan=p.question_plan(questions,functions['sample_question_groups'])
    rng=random.Random(42);expected=[]
    for _ in range(20):
        group=rng.sample(questions,10);rng.shuffle(group);expected.append(group)
    assert plan['full_groups']==expected and plan['first_turn_groups']==[[g[0]] for g in expected]
    assert plan['first_turn_groups']!=functions['sample_question_groups'](questions,20,1,42)
    assert len(plan['conversations'])==60 and plan==p.question_plan(questions,copied['sample_question_groups'])
    tree=ast.parse(source.read_text())
    for node in tree.body:
        if isinstance(node,ast.FunctionDef) and node.name in hashes:
            assert hashes[node.name]==p.canonical(ast.dump(node,include_attributes=False))


def test_blind_plans_separate_metrics_and_conversation_arms():
    functions,_=p.author_functions(AUTHOR/'eval/eval_multiturn_pipeline.py')
    conv=p.question_plan(read_json(AUTHOR/'data/trait_data_eval/impolite.json')['questions'],functions['sample_question_groups'])['conversations']
    plans=[p.grading_plan(conv,seed) for seed in (821,822)]
    for plan in plans:
        assert len(plan)==23 and sorted(map(len,plan))==[4]+[8]*22
        assert all(len({x['source_id'] for x in batch})==len(batch) for batch in plan)
        flat=[x for batch in plan for x in batch];assert len({x['id'] for x in flat})==180
        assert {(x['source_id'],x['arm']) for x in flat}=={(r['id'],arm) for r in conv for arm in p.ARMS}
    assert plans[0]!=plans[1]


@pytest.mark.parametrize('key,value',[('main_batch_size',8),('sampled_turns',1),('maximum_native_starts',93),('residual_block',20),('result','results/persona_reference_rudeness_01.json')])
def test_fixed_profile_and_old_path_rejection(key,value):
    cfg=read_json(a.CONFIG);cfg[key]=value
    with pytest.raises(ValueError):a.validate_config(cfg)
    with pytest.raises(ValueError):a.guard(path('outputs/persona-reference-rudeness-01'))


def test_actual_vector_bytes_safe_full_shapes_and_bad_payloads(tmp_path):
    config={'hidden_size':16,'num_hidden_layers':20,'num_attention_heads':4,'num_key_value_heads':2}
    shapes={'residual':(21,16),'delta':(20,16),'key':(20,2,4),'demand':(20,)}
    files={}
    for name,shape in shapes.items():
        file=tmp_path/(name+'.pt');torch.save(torch.ones(shape,dtype=torch.float32),file);files[name]=file
    bank=p.tensor_bank(files,config);assert all(v.dtype==torch.float32 for v in bank.values())
    torch.save({'not':'tensor'},files['delta'])
    with pytest.raises(ValueError):p.tensor_bank(files,config)
    torch.save(torch.ones((20,16),dtype=torch.bfloat16),files['delta'])
    with pytest.raises(ValueError):p.tensor_bank(files,config)


def tiny_model():
    from transformers import Qwen2Config,Qwen2ForCausalLM
    torch.manual_seed(17)
    cfg=Qwen2Config(vocab_size=41,hidden_size=16,intermediate_size=32,num_hidden_layers=20,num_attention_heads=4,num_key_value_heads=2,
                   bos_token_id=1,eos_token_id=2,pad_token_id=0,attn_implementation='sdpa')
    model=Qwen2ForCausalLM(cfg).eval();model.generation_config.eos_token_id=[2,3];model.generation_config.pad_token_id=0
    return model


def tiny_bank():
    torch.manual_seed(8)
    return {'residual':torch.randn(21,16)*.04,'delta':torch.randn(20,16)*.04,'key':torch.randn(20,2,4), 'demand':torch.randn(20)}


def actual_classes():
    return {'gcad':p.load_author_class(AUTHOR/'core/gated_steerer.py','MultiLayerGatedSteerer'),
            'residual':p.load_author_class(AUTHOR/'core/activation_steer.py','ActivationSteerer')}


@pytest.mark.parametrize('dtype',[torch.float32,torch.bfloat16])
def test_author_gcad_math_differential_all_positions(dtype):
    model=tiny_model().to(dtype);values=tiny_bank();ctx=p.steerer(model,values,actual_classes(),'gcad')
    torch.manual_seed(12);q=torch.randn(2,4,4,4,dtype=dtype);out=torch.randn(2,4,16,dtype=dtype)
    ctx._q_captured[9]=q;ctx._pos_emb[9]=(torch.ones(2,4,4,dtype=dtype),torch.zeros(2,4,4,dtype=dtype))
    actual=ctx._make_attn_hook(9)(None,(),(out,None))[0]
    key=values['key'][9].repeat_interleave(2,dim=0).to(dtype);v=values['delta'][9].to(dtype)
    demand=(q*key).sum(-1).mean(-1)/(4**.5)
    expected=out+(2*3.5*torch.sigmoid(1.5*(demand-values['demand'][9].item()))).unsqueeze(-1)*v
    assert torch.equal(actual,expected) and not torch.equal(actual[:,0],out[:,0]) and not ctx._q_captured and not ctx._pos_emb


def test_actual_hooks_zero_cached_nonzero_all_sites_and_cleanup():
    model=tiny_model();bank=tiny_bank();classes=actual_classes();ids=torch.tensor([[0,5,6],[7,8,9]]);mask=ids.ne(0).long();signature=p.hook_signature(model)
    with torch.no_grad():base=model(input_ids=ids,attention_mask=mask,use_cache=True)
    for arm in ('gcad','residual'):
        for zero in (True,False):
            ctx=p.steerer(model,bank,classes,arm,zero);audit={}
            with ctx,p.observe(model,arm,audit),torch.no_grad():
                first=model(input_ids=ids,attention_mask=mask,use_cache=True)
                model(input_ids=torch.tensor([[10],[11]]),attention_mask=torch.cat([mask,torch.ones(2,1,dtype=torch.long)],dim=1),past_key_values=first.past_key_values,use_cache=True)
            assert torch.equal(first.logits,base.logits) if zero else not torch.equal(first.logits,base.logits)
            assert len(audit['sites'])==(11 if arm=='gcad' else 1)
            assert all(x['prefill_calls']==1 and x['decode_calls']==1 for x in audit['sites'].values())
            assert p.hook_signature(model)==signature and audit['observer_handles_removed']
        with pytest.raises(RuntimeError):
            with p.steerer(model,bank,classes,arm),p.observe(model,arm,{}):raise RuntimeError('injected')
        assert p.hook_signature(model)==signature


def test_author_single_vs_multiple_one_instruction_same_hooked_output():
    model=tiny_model();single=actual_classes()['residual']
    multiple=single.__init__.__globals__['ActivationSteererMultiple']
    vector=tiny_bank()['residual'][20];ids=torch.tensor([[0,5,6],[7,8,9]]);mask=ids.ne(0).long()
    signature=p.hook_signature(model);observed=[]
    contexts=[single(model,vector,coeff=2.,layer_idx=19,positions='all'),
              multiple(model,[{'steering_vector':vector,'coeff':2.,'layer_idx':19,'positions':'all','target_module':'layer'}])]
    for context in contexts:
        outputs=[]
        def capture(module,inputs,value):outputs.append((value[0] if isinstance(value,(tuple,list)) else value).detach().clone())
        with context,torch.no_grad():
            handle=model.model.layers[19].register_forward_hook(capture)
            try:
                first=model(input_ids=ids,attention_mask=mask,use_cache=True)
                model(input_ids=torch.tensor([[10],[11]]),attention_mask=torch.cat([mask,torch.ones(2,1,dtype=torch.long)],dim=1),past_key_values=first.past_key_values,use_cache=True)
            finally:handle.remove()
        assert p.hook_signature(model)==signature;observed.append(outputs)
    assert len(observed[0])==len(observed[1])==2
    assert all(torch.equal(x,y) for x,y in zip(*observed))


def test_tiny_qwen_source_generate_actual_zero_tokens_and_cleanup():
    class Tok(FakeTokenizer):
        def decode(self,ids,skip_special_tokens=True):return ''.join(str(int(x)) for x in ids if int(x) not in (0,2,3,9))
    tok=Tok();model=tiny_model();values=tiny_bank();classes=actual_classes();signature=p.hook_signature(model)
    generate=p.author_functions(AUTHOR/'eval/eval_multiturn_pipeline.py',torch)[0]['generate_batched_multiturn'];outputs=[]
    for arm in p.ARMS:
        proxy=p.GenerationProxy(model,tok,arm,maximum=2,capture_logits=True);torch.manual_seed(42)
        with torch.no_grad():generate(proxy,tok,[['x'],['xxx']],p.steerer(model,values,classes,arm,zero=True),1,max_tokens_per_turn=4,temperature=1.0,batch_size=64)
        outputs.append([r['generated_padded_ids'] for r in proxy.records]);assert p.hook_signature(model)==signature
    assert outputs[0]==outputs[1]==outputs[2]


class FakeTokenizer:
    pad_token='eos';pad_token_id=9;eos_token='eos';eos_token_id=9;padding_side='left'
    def apply_chat_template(self,messages,tokenize=False,add_generation_prompt=True):
        assert messages[0]['role']=='user';return messages[0]['content']
    def __call__(self,texts,return_tensors=None,padding=False):
        if isinstance(texts,str):return {'input_ids':[4]*len(texts)}
        width=max(map(len,texts));ids=[[9]*(width-len(t))+[4]*len(t) for t in texts];mask=[[0]*(width-len(t))+[1]*len(t) for t in texts]
        return {'input_ids':torch.tensor(ids),'attention_mask':torch.tensor(mask)}
    def decode(self,ids,skip_special_tokens=True):return ''.join(str(int(x)) for x in ids if int(x) not in (9,11))
    def encode(self,text,add_special_tokens=False):return list(text)


class FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__();self.weight=torch.nn.Parameter(torch.ones(1));self.lm_head=torch.nn.Identity();self.generation_config=SimpleNamespace(eos_token_id=[9,11]);self.calls=[]
    @property
    def device(self):return self.weight.device
    def generate(self,**kwargs):
        self.calls.append(kwargs);ids=kwargs['input_ids'];self.lm_head(torch.zeros(len(ids),ids.shape[1],13))
        suffix=torch.tensor([[5,9,11],[6,7,11]])[:len(ids)]
        return torch.cat([ids,suffix],dim=1)


def test_unchanged_author_generation_proxy_dual_eos_padding_and_last_logits():
    functions,_=p.author_functions(AUTHOR/'eval/eval_multiturn_pipeline.py',torch);generate=functions['generate_batched_multiturn']
    model=FakeModel();tok=FakeTokenizer();proxy=p.GenerationProxy(model,tok,'baseline',maximum=2,capture_logits=True)
    rows=generate(proxy,tok,[['x'],['abc']],nullcontext(),1,max_tokens_per_turn=3,temperature=1,system_prompt=None,batch_size=64)
    assert [r['answer'] for r in rows]==['5','67']
    assert proxy.records[0]['actual_input_ids']==[9,9,4] and proxy.records[0]['attention_mask']==[0,0,1]
    assert proxy.records[0]['generated_ids']==[5,9] and proxy.records[0]['generated_padded_ids']==[5,9,11]
    assert proxy.records[1]['generated_ids']==[6,7,11] and proxy.records[1]['ended_eos'] and not proxy.records[1]['hit_token_limit']
    assert proxy.first_logits.shape==(2,13)  # Never retain BxTxVocabulary.
    assert model.calls[0]['do_sample'] is True and model.calls[0]['temperature']==1 and model.calls[0]['use_cache'] is True
    no_capture=p.GenerationProxy(model,tok,'baseline',maximum=2);no_capture.generate(**model.calls[0]);assert no_capture.first_logits is None
    with pytest.raises(ValueError,match='budget'):proxy.generate(**model.calls[0])


def test_generation_before_start_budget(tmp_path):
    s=SimpleNamespace(run=tmp_path);a.save(tmp_path/'manifest.json',{});directory=tmp_path/'main';directory.mkdir()
    a.save(directory/'call-000-start.json',{'conversations_started':190})
    before=list(directory.iterdir())
    with pytest.raises(ValueError,match='BEFORE marker'):a.generation_start(s,directory,'gcad',False,1,{'input_ids':torch.tensor([[1]])})
    assert list(directory.iterdir())==before


@pytest.mark.parametrize('bad',[True,1.5,-1,101,'50'])
def test_strict_numeric_scores(bad):
    with pytest.raises(ValueError):p.ratings({'ratings':[{'id':'x','score':bad,'reason':'brief'}]},['x'],'coherence')


def test_refusal_is_semantic_not_schema_retry_and_coherence_always_numeric():
    value={'ratings':[{'id':'x','score':None,'refusal':True,'reason':'refusal'}]}
    assert p.ratings(value,['x'],'trait')['x']['score'] is None
    with pytest.raises(ValueError):p.ratings({'ratings':[{'id':'x','score':None,'reason':'bad'}]},['x'],'coherence')
    with pytest.raises(ValueError):p.ratings(value,['x','y'],'trait')
    with pytest.raises(ValueError):p.ratings({'ratings':value['ratings']*2},['x'],'trait')


def metric_fixture():
    preds={arm:{'records':[{'id':str(i),'answer':'short answer','generated_ids':[4,2],'n_tokens':2,'hit_token_limit':False,'ended_eos':True} for i in range(60)]} for arm in p.ARMS}
    grades={m:{(arm,str(i)):{'score':(10 if arm=='baseline' else 40) if m=='trait' else (90 if arm=='gcad' else 60)} for arm in p.ARMS for i in range(60)} for m in ('trait','coherence')}
    return preds,grades


def test_numeric_and_conservative_metrics_exact_threshold_and_all_refusal():
    preds,grades=metric_fixture();result=p.metrics(preds,grades)
    assert result['all_contrasts_pass'] and result['conditions']['baseline']['degenerate']==0
    grades['trait'][('gcad','0')]['score']=None
    result=p.metrics(preds,grades);assert result['conditions']['gcad']['trait_numeric_mean']==40 and result['conditions']['gcad']['trait_conservative_mean']<40
    for i in range(60):grades['trait'][('gcad',str(i))]['score']=None
    result=p.metrics(preds,grades);assert result['conditions']['gcad']['trait_numeric_mean'] is None and result['conditions']['gcad']['trait_conservative_mean']==0 and not result['all_contrasts_pass']


def test_hash_alias_conflict_and_tamper(tmp_path):
    f=tmp_path/'input';f.write_text('frozen');alias=tmp_path/'alias';alias.symlink_to(f)
    h=a.stream_hash(f);a.check_hashes([{str(f):h},{str(alias):h}])
    with pytest.raises(ValueError,match='conflicting'):a.check_hashes([{str(f):h},{str(alias):'f'*64}])
    f.write_text('changed')
    with pytest.raises(ValueError,match='changed'):a.check_hashes([{str(f):h}])


def test_public_privacy_and_incomplete_report(tmp_path):
    cfg=read_json(a.CONFIG);s=SimpleNamespace(run=tmp_path,cfg=cfg,m={'sources':{},'model_sha256':{},'cli_version':'2.1.323 (Claude Code)'})
    a.save(tmp_path/'manifest.json',{});a.stop(s,'test_failure',OSError())
    result=a.report(s);assert result['status']=='incomplete' and result['stop']['stop_cause']=='test_failure'
    for value in ({'reason':'private'},{'generated_ids':[1]},{'value':'/home/private'},{'answer':'private'}):
        with pytest.raises(ValueError):p.public_safe(value)


def test_full_report_numeric_contrasts_and_fatal_override(tmp_path,monkeypatch):
    cfg=read_json(a.CONFIG);s=SimpleNamespace(run=tmp_path,cfg=cfg,m={'sources':{},'model_sha256':{},'cli_version':'2.1.323 (Claude Code)'})
    a.save(tmp_path/'manifest.json',{});preds,grades=metric_fixture()
    plans={};all_ratings={};provenance={m:[{'actual_model_usage_ids':['claude-opus-5-5']}] for m in ('trait','coherence')}
    for metric in ('trait','coherence'):
        plans[metric]=[[{'arm':arm,'source_id':str(i),'id':f'{arm}-{i}'} for arm in p.ARMS for i in range(60)]]
        all_ratings[metric]={f'{arm}-{i}':grades[metric][(arm,str(i))] for arm in p.ARMS for i in range(60)}
        directory=tmp_path/'grades'/metric;a.save(directory/'run_binding.json',{})
        a.save(directory/'accepted.json',{'ratings':all_ratings[metric],'batches':provenance[metric],
            'binding_sha256':a.stream_hash(directory/'run_binding.json'),'first_valid_only':True})
    a.save(tmp_path/'inputs/plans.json',plans);a.save(tmp_path/'inputs/questions.json',{'conversations':[{'question':str(i)} for i in range(60)]})
    monkeypatch.setattr(a,'predictions',lambda s:preds);monkeypatch.setattr(a,'journal_audit',lambda s,m:(all_ratings[m],provenance[m]))
    result=a.report(s);assert result['status']=='complete' and result['metrics']['all_contrasts_pass']
    a.stop(s,'fatal_after_saved_work',OSError());result=a.report(s)
    assert result['status']=='incomplete' and result['metrics']['all_contrasts_pass']


def test_live_preflight_fatal_persists_but_no_execution_is_harmless(tmp_path,monkeypatch):
    a.save(tmp_path/'manifest.json',{})
    monkeypatch.setattr(a,'path',lambda n:tmp_path if n==a.RUN else Path(n))
    class Broken:
        run=tmp_path
        def verify(self,models=False):raise OSError('preflight hash/version failure')
    monkeypatch.setattr(a,'Context',Broken);monkeypatch.setattr(sys,'argv',['runner','judge','--metric','trait','--execute-judgments'])
    with pytest.raises(OSError):a.main()
    assert read_json(tmp_path/'stopped.json')['stop_cause']=='fatal_execution_or_integrity'
    (tmp_path/'stopped.json').unlink();monkeypatch.setattr(sys,'argv',['runner','judge']);a.main()
    assert not (tmp_path/'stopped.json').exists()


def test_cached_qwen_tokenizer_implicit_system_and_author_eos_padding():
    from transformers import AutoTokenizer
    cfg=read_json(a.CONFIG);tok=AutoTokenizer.from_pretrained(path(cfg['snapshot']),local_files_only=True,trust_remote_code=False)
    assert tok.pad_token_id==151643 and tok.eos_token_id==151645
    tok.pad_token=tok.eos_token;tok.pad_token_id=tok.eos_token_id;tok.padding_side='left'
    qs=read_json(AUTHOR/'data/trait_data_eval/impolite.json')['questions'][:2]
    rendered=[tok.apply_chat_template([{'role':'user','content':q}],tokenize=False,add_generation_prompt=True) for q in qs]
    assert all('You are Qwen, created by Alibaba Cloud. You are a helpful assistant.' in text for text in rendered)
    inputs=tok(rendered,padding=True,return_tensors='pt')
    for text,ids,mask in zip(rendered,inputs['input_ids'].tolist(),inputs['attention_mask'].tolist()):
        assert [x for x,m in zip(ids,mask) if m]==tok(text)['input_ids']
        assert all(x==151645 for x,m in zip(ids,mask) if not m)
    defaults=read_json(path(cfg['snapshot'])/'generation_config.json')
    assert defaults['eos_token_id']==[151645,151643] and defaults['pad_token_id']==151643


def saved_attempt(payload,ratings):
    return {'returncode':0,'isolation':{'git_not_repository':True,'git_environment_clean':True,'returncode':128},
            'canonical_stdin_sha256':a.f.canonical_hash(payload),'stdout':json.dumps({'result':json.dumps({'ratings':ratings}),
             'modelUsage':{'claude-opus-5-5':{}},'is_error':False})}


def test_schema_only_parse_model_and_infrastructure_fatal():
    payload={'expected_ids':['x']};valid=saved_attempt(payload,[{'id':'x','score':None,'refusal':True,'reason':'refusal'}])
    scores,models=a.parse(valid,payload,'trait');assert scores['x']['refusal'] and models==['claude-opus-5-5']
    bad=saved_attempt(payload,[])
    with pytest.raises(a.StructuralError):a.parse(bad,payload,'trait')
    outer=json.loads(valid['stdout']);outer['modelUsage']={};bad={**valid,'stdout':json.dumps(outer)}
    with pytest.raises(a.f.ModelMismatch):a.parse(bad,payload,'trait')
    with pytest.raises(RuntimeError):a.parse({**valid,'returncode':1},payload,'trait')


def test_native_budget_replay_does_not_charge_and_no_extra_marker(tmp_path):
    s=SimpleNamespace(run=tmp_path);directory=tmp_path/'grades/trait';directory.mkdir(parents=True)
    for n in range(92):(directory/f'batch-{n:03d}-attempt-0-start.json').write_text('{}')
    before=set(directory.iterdir());missing=directory/'batch-000-attempt-0.json'
    with pytest.raises(ValueError,match='BEFORE marker'):a.native_budget(s,missing)
    assert set(directory.iterdir())==before
    missing.write_text('{}');a.native_budget(s,missing)  # Read-only replay remains possible.


def test_bound_receipt_pending_recovery_and_tamper(tmp_path):
    payload={'expected_ids':['x']};attempt=saved_attempt(payload,[{'id':'x','score':50,'reason':'brief'}]);bound='b'*64
    start={'run_binding_sha256':bound,'canonical_stdin_sha256':a.f.canonical_hash(payload),'started_utc':'now'}
    a.save(tmp_path/'batch-000-attempt-0-start.json',start);a.save(tmp_path/'batch-000-attempt-0.json',attempt)
    assert a.checked_attempt(tmp_path,0,0,payload,bound,True)==attempt
    with pytest.raises(ValueError,match='receipt required'):a.checked_attempt(tmp_path,0,0,payload,bound)
    # Foundation repairs only the receipt; invoking must be impossible on replay.
    value,receipt=a.f.replay_or_call(tmp_path,0,0,payload,bound,'MUST_NOT_EXECUTE')
    assert value==attempt and a.checked_attempt(tmp_path,0,0,payload,bound)==attempt
    with pytest.raises(ValueError):a.checked_attempt(tmp_path,0,0,{'expected_ids':['other']},bound)
    with pytest.raises(ValueError):a.checked_attempt(tmp_path,0,0,payload,'c'*64)


def test_journal_firstvalid_reuse_stray_attempt_and_uncertain_start(tmp_path,monkeypatch):
    s=SimpleNamespace(run=tmp_path);directory=tmp_path/'grades/trait';directory.mkdir(parents=True)
    payload={'expected_ids':['x']};bodies=[payload];authority={'bound':True}
    monkeypatch.setattr(a,'bound_payloads',lambda s,m:bodies);monkeypatch.setattr(a,'binding',lambda s,m:authority)
    a.save(directory/'run_binding.json',authority);bound=a.stream_hash(directory/'run_binding.json')
    a.save(directory/'batch-000-attempt-0-start.json',{'run_binding_sha256':bound,'canonical_stdin_sha256':a.f.canonical_hash(payload)})
    with pytest.raises(a.f.UncertainAttempt):a.journal_audit(s,'trait',True)
    a.save(directory/'batch-000-attempt-0.json',saved_attempt(payload,[{'id':'x','score':55,'refusal':False,'reason':'private'}]))
    a.f.replay_or_call(directory,0,0,payload,bound,'MUST_NOT_EXECUTE')
    ratings,provenance=a.journal_audit(s,'trait');assert ratings['x']['score']==55 and provenance[0]['first_valid_attempt']==0
    a.save(directory/'batch-000-attempt-1-start.json',{})
    with pytest.raises(ValueError,match='after first valid'):a.journal_audit(s,'trait')
