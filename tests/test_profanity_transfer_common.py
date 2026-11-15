from contextlib import nullcontext
import copy
import importlib.util
from types import SimpleNamespace

import pytest
import torch

from scripts import profanity_transfer_common as p
from scripts import author_gcad_impolite_common as gcad
from scripts.common import path


def basis():
    return {'model_id':'test/tiny','revision':'a'*40,'model_source':'tiny-local-test',
            'tokenizer_sha256':'b'*64,'rendering_sha256':'c'*64,'coordinate_system':'native_hf'}


def tiny():
    from transformers import Qwen2Config,Qwen2ForCausalLM
    torch.manual_seed(8)
    cfg=Qwen2Config(vocab_size=31,hidden_size=16,intermediate_size=24,num_hidden_layers=2,
                   num_attention_heads=4,num_key_value_heads=2,max_position_embeddings=64,
                   pad_token_id=0,eos_token_id=2,bos_token_id=1,attn_implementation='sdpa')
    cfg._name_or_path=basis()['model_source']
    return Qwen2ForCausalLM(cfg).eval()


def bank(model,representation):
    cfg=model.config;g=torch.Generator().manual_seed(19)
    if representation=='last_prompt_token_contrast':
        units=torch.randn(cfg.num_hidden_layers,cfg.hidden_size,generator=g);units/=units.norm(dim=1,keepdim=True)
        tensors={'units':units,'targets':torch.tensor([2.,3.])}
    else:
        tensors={'delta':torch.randn(cfg.num_hidden_layers,cfg.hidden_size,generator=g)*.1,
                 'key':torch.randn(cfg.num_hidden_layers,cfg.num_key_value_heads,cfg.hidden_size//cfg.num_attention_heads,generator=g),
                 'demand':torch.zeros(cfg.num_hidden_layers)}
    return {'trait':'profanity','source_trait':'profanity','basis':basis(),'training_data_sha256':'d'*64,
            'representation':representation,'tensors':tensors,'tensor_sha256':{k:p.tensor_hash(v) for k,v in tensors.items()}}


def validation():return dict(expected_basis=basis(),expected_data_sha256='d'*64,forbidden_tensor_hashes=set())


def author_saps():
    file=path('outputs/persona-literature-audit-01/saps-steering/src/steering.py')
    spec=importlib.util.spec_from_file_location('_pf_test_source_saps',file)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def author_gcad():
    return gcad.load_author_class(path('outputs/persona-author-replication-audit-01/gcad/core/gated_steerer.py'),'MultiLayerGatedSteerer')


def test_author_sequential_prompt_contrast_and_independent_target_pass():
    torch.manual_seed(71);plain=torch.randn(5,2,16);styled=plain+torch.randn(5,2,16);second=torch.randn(5,2,16)
    actual=p.mean_prompt_direction(plain,styled,second)
    accum=torch.zeros(2,16)
    for h0,h1 in zip(plain,styled):accum+=h1-h0
    units=torch.nn.functional.normalize(accum/5,dim=1);targets=torch.zeros(2)
    for h in second:targets+=(h*units).sum(1)
    targets/=5
    assert torch.equal(actual['raw'],accum/5) and torch.equal(actual['units'],units) and torch.equal(actual['targets'],targets)
    assert not torch.equal(targets,(styled*units).sum(-1).mean(0))
    with pytest.raises(ValueError,match='zero contrast'):p.mean_prompt_direction(plain,plain,second)
    with pytest.raises(ValueError):p.mean_prompt_direction(plain.double(),styled,second)


@pytest.mark.parametrize('dtype',[torch.float32,torch.bfloat16])
@pytest.mark.parametrize('method',['fixed','saps'])
def test_projection_actual_pinned_source_parity_differing_states_and_sign(dtype,method):
    model=tiny().to(dtype);b=bank(model,'last_prompt_token_contrast');u=b['tensors']['units'][1];t=b['tensors']['targets'][1];h0=torch.arange(16).float()/7
    source=author_saps();alpha=1.4;audit={};c=(t.item()-(h0*u).sum().item())*alpha
    assert p.fixed_scalar(t,h0,u,alpha)==c
    reference=(source.make_fixed_projection_hook(1,u,c) if method=='fixed' else source.make_saps_hook(1,u,t,alpha))[1]
    context=p.projection_context(model,b,**validation(),block=1,alpha=alpha,method=method,h0=h0 if method=='fixed' else None,audit=audit)
    h=torch.arange(64).reshape(1,4,16).to(dtype)/11;decode=torch.flip(h[:,-1:,:],[-1])+2
    # Call the actual installed block-output hook on controlled identical tensors.
    with context:
        installed=next(iter(model.model.layers[1]._forward_hooks.values()))
        first=installed(None,(),h.clone());last=installed(None,(),decode.clone())
        assert torch.equal(first,reference(h.clone(),None)) and torch.equal(last,reference(decode.clone(),None))
        assert first.dtype==dtype and not torch.equal(first,h)
        assert (first.float()-h.float()).mul(u).sum()>0  # positive target correction for these controlled states
    assert audit['prefill_calls']==audit['decode_calls']==1 and audit['hooks_removed']
    assert not model.model.layers[1]._forward_hooks
    # Fixed c stays constant; adaptive recomputes it at the changed state.
    if method=='fixed':assert audit['shift_norms']['prefill']==audit['shift_norms']['decode']
    else:assert audit['shift_norms']['prefill']!=audit['shift_norms']['decode']


@pytest.mark.parametrize('method',['fixed','saps'])
def test_projection_actual_tiny_cached_zero_nonzero_and_exception_cleanup(method):
    model=tiny();b=bank(model,'last_prompt_token_contrast');ids=torch.tensor([[1,4,5]]);h0=torch.ones(16)
    before=gcad.hook_signature(model)
    with torch.no_grad():base=model(input_ids=ids,use_cache=True)
    for alpha in (0.,1.):
        audit={}
        with p.projection_context(model,b,**validation(),block=1,alpha=alpha,method=method,h0=h0 if method=='fixed' else None,audit=audit),torch.no_grad():
            first=model(input_ids=ids,use_cache=True)
            model(input_ids=torch.tensor([[6]]),past_key_values=first.past_key_values,use_cache=True)
        assert torch.equal(base.logits,first.logits) if alpha==0 else not torch.equal(base.logits,first.logits)
        assert audit['prefill_calls']==audit['decode_calls']==1 and audit['hooks_removed'] and gcad.hook_signature(model)==before
    with pytest.raises(RuntimeError):
        with p.projection_context(model,b,**validation(),block=1,alpha=1.,method=method,h0=h0 if method=='fixed' else None,audit={}):raise RuntimeError('injected')
    assert gcad.hook_signature(model)==before


def context_gcad(model,b,**overrides):
    kwargs=dict(author_factory=author_gcad(),layers=[0,1],base_coef=1.25,scale=1.5,inverse=False,debug=False)
    kwargs.update(overrides)
    return p.gcad_context(model,b,**validation(),**kwargs)


@pytest.mark.parametrize('dtype',[torch.float32,torch.bfloat16])
def test_actual_pinned_gcad_formula_sign_all_positions_and_no_normalization(dtype):
    model=tiny().to(dtype);b=bank(model,'system_attention_delta');ctx=context_gcad(model,b)
    assert torch.equal(ctx._v[0],b['tensors']['delta'][0].to(dtype))
    q=torch.arange(48).reshape(1,3,4,4).to(dtype)/19;hidden=torch.ones(1,3,16,dtype=dtype)
    ctx._q_captured[0]=q;ctx._pos_emb[0]=(torch.ones(1,3,4,dtype=dtype),torch.zeros(1,3,4,dtype=dtype))
    actual=ctx._make_attn_hook(0)(None,(),(hidden,None))[0]
    keys=b['tensors']['key'][0].repeat_interleave(2,dim=0).to(dtype)
    demand=(q*keys).sum(-1).mean(-1)/2
    expected=hidden+(2*1.25*torch.sigmoid(1.5*(demand-b['tensors']['demand'][0].item()))).unsqueeze(-1)*b['tensors']['delta'][0].to(dtype)
    assert torch.equal(actual,expected) and not torch.equal(actual,hidden)
    assert not ctx._q_captured and not ctx._pos_emb


def test_actual_pinned_gcad_tiny_qwen2_zero_nonzero_cached_cleanup():
    model=tiny();b=bank(model,'system_attention_delta');ids=torch.tensor([[0,4,5],[1,6,7]]);mask=ids.ne(0).long();before=gcad.hook_signature(model)
    with torch.no_grad():base=model(input_ids=ids,attention_mask=mask,use_cache=True)
    for coefficient in (0.,1.25):
        ctx=context_gcad(model,b,base_coef=coefficient);calls={0:[],1:[]};handles=[]
        with ctx,torch.no_grad():
            for block in (0,1):
                def record(module,args,value,block=block):calls[block].append(value[0].detach())
                handles.append(model.model.layers[block].self_attn.register_forward_hook(record))
            try:
                first=model(input_ids=ids,attention_mask=mask,use_cache=True)
                model(input_ids=torch.tensor([[8],[9]]),attention_mask=torch.cat([mask,torch.ones(2,1,dtype=torch.long)],dim=1),past_key_values=first.past_key_values,use_cache=True)
            finally:
                for h in handles:h.remove()
        assert torch.equal(first.logits,base.logits) if coefficient==0 else not torch.equal(first.logits,base.logits)
        assert all([tuple(v.shape[:2]) for v in values]==[(2,3),(2,1)] and all(torch.isfinite(v).all() for v in values) for values in calls.values())
        assert gcad.hook_signature(model)==before and not ctx._q_captured and not ctx._pos_emb
    with pytest.raises(RuntimeError):
        with context_gcad(model,b):raise RuntimeError('injected')
    assert gcad.hook_signature(model)==before


@pytest.mark.parametrize('mutation',['trait','source_trait','model','tokenizer','data','representation','dtype','shape','hash','unit','model_source'])
def test_reject_foreign_bank_basis_schema_and_tensor_changes(mutation):
    model=tiny();b=bank(model,'last_prompt_token_contrast')
    if mutation=='trait':b['trait']='impolite'
    elif mutation=='source_trait':b['source_trait']='impolite'
    elif mutation=='model':b['basis']['model_id']='wrong/model'
    elif mutation=='tokenizer':b['basis']['tokenizer_sha256']='f'*64
    elif mutation=='data':b['training_data_sha256']='e'*64
    elif mutation=='representation':b['representation']='response_mean'
    elif mutation=='dtype':b['tensors']['units']=b['tensors']['units'].double()
    elif mutation=='shape':b['tensors']['targets']=b['tensors']['targets'][:,None]
    elif mutation=='hash':b['tensor_sha256']['units']='0'*64
    elif mutation=='model_source':model.config._name_or_path='foreign-checkpoint'
    else:
        b['tensors']['units']*=2;b['tensor_sha256']['units']=p.tensor_hash(b['tensors']['units'])
    with pytest.raises(ValueError):p.validate_bank(model,b,**validation(),representation='last_prompt_token_contrast')


def test_known_impolite_tensor_hash_cannot_be_relabelled_and_no_implicit_settings():
    model=tiny();b=bank(model,'system_attention_delta');expected=validation();expected['forbidden_tensor_hashes']={p.tensor_hash(b['tensors']['delta'])}
    with pytest.raises(ValueError,match='impolite'):p.validate_bank(model,b,**expected,representation='system_attention_delta')
    with pytest.raises(ValueError):context_gcad(model,b,layers=[0,0])
    with pytest.raises(ValueError):context_gcad(model,b,layers=[2])
    with pytest.raises(ValueError):context_gcad(model,b,scale=float('nan'))
    with pytest.raises(TypeError):p.gcad_context(model,b)
    assert p.capture_last_prompt is p.saps.capture and p.fixed_scalar is p.saps.fixed_scalar
