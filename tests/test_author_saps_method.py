import json
from pathlib import Path
import random
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch
from scripts import author_saps_method as a
from scripts import author_saps_method_common as p
from scripts.common import path,read_json

AUTHOR=path('outputs/persona-literature-audit-01/saps-steering/src/steering.py')


def test_pinned_helpers_fixed_python_scalar_and_adaptive_source_dtype():
    author=p.author_module(AUTHOR)
    h0=torch.tensor([1.]);u=torch.tensor([1.]);target=torch.tensor(33554432.)
    c=p.fixed_scalar(h0,u,target,1.)
    assert c==(target.item()-(h0*u).sum().item())*1. and c!=(target-(h0*u).sum()).item()
    with pytest.raises(ValueError):p.fixed_scalar(h0,u,target[None],1.)
    torch.manual_seed(11);unit=torch.randn(8);target=torch.tensor(.4);resid=torch.randn(1,3,8,dtype=torch.float16)
    name,hook=author.make_saps_hook(1,unit,target,1.4);actual=hook(resid,None)
    normalized=unit/(unit.norm()+1e-8);half=normalized.to(torch.float16);proj=(resid[:,-1,:]*half).sum(-1,keepdim=True)
    expected=resid+1.4*(target-proj)*half
    assert name=='blocks.1.hook_resid_post' and actual.dtype==torch.float16 and torch.equal(actual,expected)
    assert not torch.equal(actual[:,0],resid[:,0])  # Full-prefix source broadcast.
    fixed=author.make_fixed_projection_hook(1,unit,.3)[1](resid,None)
    assert torch.equal(fixed,resid+.3*unit.to(torch.float16))


class FakeTokenizer:
    bos_token_id=1;eos_token_id=2
    def apply_chat_template(self,messages,add_generation_prompt=True,tokenize=False):return messages[0]['content']
    def decode(self,ids,skip_special_tokens=True):return ' '.join(str(int(x)) for x in ids if int(x) not in (1,2))


class FakeTL(torch.nn.Module):
    def __init__(self,*,eos=False,fail=False,omit_last=False):
        super().__init__();self.embed=torch.nn.Embedding(12,4);self.cfg=SimpleNamespace(device='cpu');self.tokenizer=FakeTokenizer();self.hook_dict={};self.eos=eos;self.fail=fail;self.omit_last=omit_last;self.calls=[]
    def to_tokens(self,prompt):return torch.tensor([[1,1,4,5]])
    def generate(self,prompt,**kwargs):
        self.calls.append(kwargs);prefix=self.to_tokens(prompt);self.embed(prefix)
        if self.fail:raise RuntimeError('injected')
        suffix=[6,2] if self.eos else [6]*kwargs['max_new_tokens']
        for token in (suffix[:-1] if self.omit_last else suffix):self.embed(torch.tensor([[token]]))
        return self.tokenizer.decode(torch.cat([prefix[0],torch.tensor(suffix)]),skip_special_tokens=True)


@pytest.mark.parametrize('eos',[True,False])
def test_final_eos_and_cap_token_source_decode_and_complete_embedding(eos):
    model=FakeTL(eos=eos);original=model.tokenizer.decode.__func__;author=p.author_module(AUTHOR)
    value=p.generate(model,'prompt',author,None,3)
    assert value['input_ids']==[1,1,4,5] and value['leading_bos_count']==2
    assert value['generated_ids']==([6,2] if eos else [6,6,6])
    assert value['ended_eos']==eos and value['hit_token_limit']==(not eos)
    assert value['full_token_ids']==value['input_ids']+value['generated_ids'] and value['final_sampled_token_observed_in_embedding']
    assert 'decode' not in model.tokenizer.__dict__ and model.tokenizer.decode.__func__ is original and not model.embed._forward_pre_hooks
    assert model.calls==[{'max_new_tokens':3,'do_sample':False,'verbose':False}]  # No return_type override.


def test_no_fallback_to_unsupported_embedding_loop_and_cleanup_exception():
    author=p.author_module(AUTHOR);model=FakeTL(omit_last=True)
    with pytest.raises(ValueError,match='complete embedding'):p.generate(model,'prompt',author,None,3)
    model=FakeTL(fail=True)
    with pytest.raises(RuntimeError):p.generate(model,'prompt',author,None,3)
    assert 'decode' not in model.tokenizer.__dict__ and not model.embed._forward_pre_hooks


def test_source_capture_proxy_actual_draw_stream_1500_and_reconstructed_math():
    author=p.author_module(AUTHOR);prompts={'plain':'PLAIN','style':['STYLE0','STYLE1','STYLE2','STYLE3','STYLE4']};train=[f'train-{i}' for i in range(500)]
    class Model:
        cfg=SimpleNamespace(n_layers=2,d_model=2);tokenizer=FakeTokenizer()
        def __init__(self):self.texts=[]
        def to_tokens(self,text):self.texts.append(text);return torch.tensor([[len(self.texts)-1]])
        def run_with_cache(self,tokens,**kwargs):
            text=self.texts[tokens.item()];style='STYLE' in text;value=torch.tensor([[[2.,1.]]]) if style else torch.tensor([[[1.,2.]]])
            return None,{f'blocks.{layer}.hook_resid_post':value.clone() for layer in range(2)}
    model=Model();started=[];random.seed(42)
    proxy=p.CaptureProxy(model,author,train,prompts,lambda i,payload:started.append((i,payload)))
    units,target=author.learn_style_direction(proxy,model.tokenizer,train,prompts['plain'],prompts['style'],'random')
    assert len(proxy.ledger)==len(started)==1500 and [r['kind'] for r in proxy.ledger[:4]]==['plain','style','plain','style']
    expected=[random.Random(42)]
    expected=[expected[0].choice(range(5)) for _ in range(1000)]
    actual=[r['instruction_index'] for r in proxy.ledger if r['kind']=='style'];assert actual==expected
    raw=torch.tensor([[1.,-1.],[1.,-1.]]);want=torch.nn.functional.normalize(raw,dim=1)
    assert torch.equal(units,want) and torch.allclose(target,(torch.tensor([[2.,1.],[2.,1.]])*want).sum(1),atol=1e-5)


def test_pinned_loader_no_processing_overrides_and_seed_before_loading():
    cfg=read_json(a.CONFIG);calls=[]
    class Tokenizer:
        @staticmethod
        def from_pretrained(*args,**kwargs):calls.append(('tokenizer',args,kwargs));return object()
    class TL:
        @staticmethod
        def from_pretrained(*args,**kwargs):
            calls.append(('TL',args,kwargs));assert random.getstate()==random.Random(42).getstate();return SimpleNamespace(eval=lambda:None)
    a.pinned_load(cfg,TL,Tokenizer)
    call=calls[1];assert call[1]==(cfg['model_id'],) and call[2]['revision']==cfg['revision'] and call[2]['local_files_only'] is True
    assert call[2]['torch_dtype']==torch.float16 and call[2]['device']=='cuda' and call[2]['trust_remote_code'] is False
    assert not {'fold_ln','center_writing_weights','center_unembed','fold_value_biases','dtype','default_prepend_bos'}&set(call[2])


def test_actual_isolated_TL_tiny_cache_zero_trace_and_tokenizer_rng():
    # Existing provisioned environment only; no target weights, installs or API.
    executable=path('outputs/persona-author-replication-audit-01/envs/saps/bin/python')
    program=r'''
import json,random,torch
from transformer_lens import HookedTransformer,HookedTransformerConfig
from transformers import PreTrainedTokenizerFast,AutoTokenizer
from tokenizers import Tokenizer,models,pre_tokenizers,processors
from tempfile import TemporaryDirectory
from scripts import author_saps_method as a,author_saps_method_common as p
cfg=a.read_json(a.CONFIG);random.seed(42);torch.manual_seed(42);before=a.rng_record()
AutoTokenizer.from_pretrained(a.path(cfg['snapshot']),local_files_only=True,trust_remote_code=False)
assert a.rng_record()==before
base=Tokenizer(models.WordLevel({'<unk>':0,'<s>':1,'</s>':2,'word':3,'other':4},unk_token='<unk>'));base.pre_tokenizer=pre_tokenizers.WhitespaceSplit()
base.post_processor=processors.TemplateProcessing(single='<s> $A',special_tokens=[('<s>',1)])
tok=PreTrainedTokenizerFast(tokenizer_object=base,unk_token='<unk>',bos_token='<s>',eos_token='</s>',pad_token='</s>')
# Realistic local save/reload supplies the init_kwargs name/path TL requires.
temporary=TemporaryDirectory();tok.save_pretrained(temporary.name)
tok=AutoTokenizer.from_pretrained(temporary.name,local_files_only=True)
torch.set_num_threads(1)
model=HookedTransformer(HookedTransformerConfig(n_layers=2,d_model=8,d_head=4,n_heads=2,d_mlp=16,d_vocab=5,n_ctx=64,act_fn='relu',device='cpu'),tokenizer=tok)
author=p.author_module(a.path(cfg['author_root'])/'src/steering.py')
prompt='<s> word other'
for eos in (True,False):
 with torch.no_grad():model.unembed.W_U.zero_();model.unembed.b_U.zero_();model.unembed.b_U[2 if eos else 3]=100
 ordinary=p.generate(model,prompt,author,None,3)
 hook=author.make_saps_hook(1,torch.ones(8),torch.tensor(.4),0.)
 zero=p.generate(model,prompt,author,hook,3)
 assert ordinary['full_token_ids']==zero['full_token_ids']
 assert ordinary['generated_ids']==([2] if eos else [3,3,3])
 assert ordinary['leading_bos_count']==2 and ordinary['hit_token_limit']==(not eos)
 assert zero['hook_audit']['prefill_calls']==1 and zero['hook_audit']['decode_calls']==(0 if eos else 2)
 assert not any(x.fwd_hooks for x in model.hook_dict.values()) and 'decode' not in tok.__dict__
nonzero=p.generate(model,prompt,author,author.make_saps_hook(1,torch.ones(8),torch.tensor(.4),1.),3)
assert nonzero['hook_audit']['prefill_calls']==1 and nonzero['hook_audit']['decode_calls']==2
assert nonzero['hook_audit']['prefill']['max_abs_hidden_delta']>0 and nonzero['hook_audit']['finite']
def boom(resid,hook):raise RuntimeError('injected source hook failure')
try:p.generate(model,prompt,author,('blocks.1.hook_resid_post',boom),3)
except RuntimeError:pass
else:raise AssertionError('expected hook failure')
assert not any(x.fwd_hooks for x in model.hook_dict.values()) and 'decode' not in tok.__dict__
print('TINY_TL_PASS')
temporary.cleanup()
'''
    result=subprocess.run([str(executable),'-B','-c',program],cwd=path('.'),capture_output=True,text=True,timeout=90)
    assert result.returncode==0,result.stderr[-2500:];assert 'TINY_TL_PASS' in result.stdout


def test_actual_injected_vs_TL_internal_tokenizer_all_source_renderings():
    executable=path('outputs/persona-author-replication-audit-01/envs/saps/bin/python')
    program=r'''
import torch
from transformers import AutoTokenizer
from transformer_lens import HookedTransformer,HookedTransformerConfig
from scripts import author_saps_method as a,author_saps_method_common as p
torch.set_num_threads(1);cfg=a.read_json(a.CONFIG);snapshot=a.path(cfg['snapshot'])
injected=AutoTokenizer.from_pretrained(snapshot,local_files_only=True,trust_remote_code=False)
# Exact inspected TL internal tokenization kwargs, with only immutable local
# path/offline safety replacing the unpinned official tokenizer name.
internal=AutoTokenizer.from_pretrained(snapshot,add_bos_token=True,trust_remote_code=False,use_fast=True,token=None,local_files_only=True)
def tiny(tok):
 return HookedTransformer(HookedTransformerConfig(n_layers=1,d_model=8,d_head=4,n_heads=2,d_mlp=16,d_vocab=32000,n_ctx=4096,act_fn='relu',device='cpu'),tokenizer=tok)
left=tiny(injected);right=tiny(internal)
for name in ('bos_token_id','eos_token_id','pad_token_id','padding_side','add_bos_token'):
 assert getattr(left.tokenizer,name)==getattr(right.tokenizer,name),(name,getattr(left.tokenizer,name),getattr(right.tokenizer,name))
assert left.cfg.default_prepend_bos==right.cfg.default_prepend_bos and left.cfg.tokenizer_prepends_bos==right.cfg.tokenizer_prepends_bos
author=p.author_module(a.path(cfg['author_root'])/'src/steering.py');prompts=a.read_json(a.path(cfg['author_root'])/'prompts/sarcasm.json')
train=a.read_json(a.path(cfg['author_root'])/'data/sarcasm/train.json');test=a.read_json(a.path(cfg['author_root'])/'data/sarcasm/test.json');count=0
for sentence,instruction in ([(s,prompts['plain']) for s in train]+[(s,i) for s in train for i in prompts['style']]+[(s,prompts['plain']) for s in test]):
 l=author.build_chat_prompt(left.tokenizer,sentence,instruction);r=author.build_chat_prompt(right.tokenizer,sentence,instruction)
 assert l==r and torch.equal(left.to_tokens(l),right.to_tokens(r));count+=1
assert count==3450
print('TOKENIZER_EQUIVALENCE_PASS',count)
'''
    result=subprocess.run([str(executable),'-B','-c',program],cwd=path('.'),capture_output=True,text=True,timeout=120)
    assert result.returncode==0,result.stderr[-2500:];assert 'TOKENIZER_EQUIVALENCE_PASS 3450' in result.stdout


def test_frozen_plans_counts_and_same_sentence_separation():
    batches=p.plans();flat=[r for b in batches for r in b]
    assert len(batches)==90 and all(len(b)==20 and len({r['sentence_id'] for r in b})==20 for b in batches)
    assert len({r['id'] for r in flat})==len({(r['sentence_id'],r['arm']) for r in flat})==1800
    assert p.plans()==batches


@pytest.mark.parametrize('key,value',[('block',13),('dtype','bfloat16'),('maximum_native_starts',181),('instruction_mode','fixed')])
def test_fixed_profile_settings(key,value):
    cfg=read_json(a.CONFIG);cfg[key]=value
    with pytest.raises(ValueError):a.validate_config(cfg)


def test_budget_and_missing_release_preconditions_before_work(tmp_path):
    s=SimpleNamespace(run=tmp_path,cfg=read_json(a.CONFIG));out=tmp_path/'mechanics';out.mkdir();a.save(tmp_path/'manifest.json',{})
    for i in range(4):a.charge(s,out,'mechanics_projection_forwards',{'i':i})
    before=list(out.iterdir())
    with pytest.raises(ValueError):a.charge(s,out,'mechanics_projection_forwards',{})
    assert list(out.iterdir())==before
    with pytest.raises(a.PreconditionWait):a.preconditions(s,'generate','saps1p4')
    with pytest.raises(ValueError):a.guard(path('outputs/persona-saps-sarcasm-01'))


def test_metric_scales_schema_and_paired_aggregate():
    predictions={arm:[{'sentence_id':i,'output':'One complete sentence.','hit_token_limit':False,'ended_eos':True,'generated_ids':[3,2]} for i in range(450)] for arm in p.ARMS}
    ratings={(arm,i):{'sarcasm':value,'entailment':4,'grammaticality':4,'clarity':4} for arm,value in zip(p.ARMS,[0,2,3,4]) for i in range(450)}
    result=p.aggregate(predictions,ratings)
    assert all(result['descriptive_directional_checks'].values()) and result['conditions']['saps1']['sarcasm_ge3']==450 and result['conditions']['saps1']['sarcasm_ge4']==0
    assert result['paired_sentence_differences']['saps1p4_minus_saps1']['sarcasm']=={'mean':1,'higher':450,'lower':0,'equal':0}
    valid={'ratings':[{'id':'x',**ratings[('saps0',0)]}]};assert p.scores(valid,['x'])['x']['sarcasm']==0
    valid['ratings'][0]['sarcasm']=True
    with pytest.raises(ValueError):p.scores(valid,['x'])


def test_public_report_incomplete_and_privacy(tmp_path):
    s=SimpleNamespace(run=tmp_path,cfg=read_json(a.CONFIG),m={'sources':{}});a.save(tmp_path/'manifest.json',{})
    result=a.report(s);assert result['status']=='incomplete' and result['totals']['generations']==0
    assert 'no HF-coordinate transfer' in result['limits'][0]


def native_attempt(payload):
    return {'returncode':0,'isolation':{'git_not_repository':True,'git_environment_clean':True,'returncode':128},
            'canonical_stdin_sha256':a.f.canonical_hash(payload),
            'stdout':json.dumps({'result':json.dumps({'ratings':[{'id':'opaque','sarcasm':3,'entailment':4,'grammaticality':4,'clarity':4}]}),
                                 'modelUsage':{'claude-opus-5-5':{}},'is_error':False})}


def test_native_firstvalid_receipt_recovery_uncertain_and_extra_attempt(tmp_path,monkeypatch):
    s=SimpleNamespace(run=tmp_path);directory=tmp_path/'grades';directory.mkdir();payload={'expected_ids':['opaque']};authority={'bound':True}
    monkeypatch.setattr(a,'bound_bodies',lambda s:[payload]);monkeypatch.setattr(a,'binding',lambda s:authority)
    a.save(directory/'run_binding.json',authority);bound=a.stream_hash(directory/'run_binding.json')
    a.save(directory/'batch-000-attempt-0-start.json',{'run_binding_sha256':bound,'canonical_stdin_sha256':a.f.canonical_hash(payload)})
    with pytest.raises(a.f.UncertainAttempt):a.read_grades(s,True)
    a.save(directory/'batch-000-attempt-0.json',native_attempt(payload))
    assert a.read_grades(s,True)['ratings']['opaque']['sarcasm']==3
    # Completed first-valid replay publishes its missing receipt, no invocation.
    a.f.replay_or_call(directory,0,0,payload,bound,'MUST_NOT_EXECUTE')
    assert a.read_grades(s)['batches'][0]['first_valid_attempt']==0
    a.save(directory/'batch-000-attempt-1-start.json',{})
    with pytest.raises(ValueError,match='after first valid'):a.read_grades(s)


def test_native_strict_identity_payload_tamper_and_budget(tmp_path):
    payload={'expected_ids':['opaque']};attempt=native_attempt(payload)
    assert a.parse(attempt,payload)[1]==['claude-opus-5-5']
    outer=json.loads(attempt['stdout']);outer['modelUsage']={};bad={**attempt,'stdout':json.dumps(outer)}
    with pytest.raises(a.f.ModelMismatch):a.parse(bad,payload)
    with pytest.raises(a.StructuralError):a.parse(attempt,{'expected_ids':['other']})
    for bi in range(90):
        for ai in (0,1):(tmp_path/f'batch-{bi:03d}-attempt-{ai}-start.json').write_text('{}')
    file=tmp_path/'batch-000-attempt-0.json';before=set(tmp_path.iterdir())
    with pytest.raises(ValueError,match='BEFORE marker'):a.native_budget(tmp_path,file)
    assert set(tmp_path.iterdir())==before
    file.write_text('{}');a.native_budget(tmp_path,file)


def test_report_complete_and_fatal_override_keeps_observed_metrics(tmp_path,monkeypatch):
    s=SimpleNamespace(run=tmp_path,cfg=read_json(a.CONFIG),m={'sources':{}});a.save(tmp_path/'manifest.json',{})
    rows={arm:[{'sentence_id':i,'output':'Complete text.','hit_token_limit':False,'ended_eos':True,'generated_ids':[3,2]} for i in range(450)] for arm in p.ARMS}
    plans=p.plans();a.save(tmp_path/'inputs/plans.json',plans)
    grades={'ratings':{x['id']:{'sarcasm':1,'entailment':4,'grammaticality':4,'clarity':4} for batch in plans for x in batch},
            'batches':[{'actual_model_usage_ids':['claude-opus-5-5']} for _ in range(90)]}
    a.save(tmp_path/'grades/accepted.json',grades);monkeypatch.setattr(a,'read_grades',lambda s:grades);monkeypatch.setattr(a,'predictions',lambda s:rows)
    result=a.report(s);assert result['status']=='complete' and result['firstvalid_batches']==90
    a.save(tmp_path/'stopped.json',{'stop_cause':'fatal_execution_or_integrity'})
    assert a.report(s)['status']=='incomplete'
