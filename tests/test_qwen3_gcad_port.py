"""Tiny CPU source parity and the Qwen3 normalized-head port contract."""
from pathlib import Path
import copy
import pytest
import torch
from transformers import Qwen2Config,Qwen2ForCausalLM,Qwen3Config,Qwen3ForCausalLM
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb,repeat_kv
from scripts import qwen3_gcad_port as g
from scripts import profanity_method_transfer_common as p
from scripts import author_gcad_impolite_common as native

ROOT=Path('outputs/persona-author-replication-audit-01/gcad')

@pytest.fixture(autouse=True)
def cpu_thread():
    old=torch.get_num_threads();torch.set_num_threads(1)
    try:yield
    finally:torch.set_num_threads(old)


def tiny(kind='qwen3',layers=2):
    torch.manual_seed(19)
    common=dict(vocab_size=32,hidden_size=16,intermediate_size=24,num_hidden_layers=layers,num_attention_heads=4,num_key_value_heads=2,
        max_position_embeddings=512,pad_token_id=0,bos_token_id=1,eos_token_id=2,attention_bias=False,attn_implementation='eager',use_sliding_window=False)
    if kind=='qwen3':return Qwen3ForCausalLM(Qwen3Config(**common,head_dim=8)).eval()
    return Qwen2ForCausalLM(Qwen2Config(**common)).eval()

class Batch(dict):
    def to(self,device):return Batch({k:v.to(device) for k,v in self.items()})

class Tok:
    def encode(self,text,add_special_tokens=False):return [3+ord(c)%29 for c in text]
    def __call__(self,text,return_tensors=None,add_special_tokens=False):
        ids=self.encode(text)
        return Batch(input_ids=torch.tensor([ids]),attention_mask=torch.ones(1,len(ids),dtype=torch.long)) if return_tensors else {'input_ids':ids}


def vectors(model):
    cfg=model.config;hd=getattr(cfg,'head_dim',cfg.hidden_size//cfg.num_attention_heads)
    gen=torch.Generator().manual_seed(71)
    return {i:torch.randn(cfg.hidden_size,generator=gen)*.1 for i in range(cfg.num_hidden_layers)}, {i:torch.randn(cfg.num_key_value_heads,hd,generator=gen)*.1 for i in range(cfg.num_hidden_layers)}, {i:.1 for i in range(cfg.num_hidden_layers)}


def context(cls,model,base=1.):
    v,k,d=vectors(model)
    return cls(model,v,k,d,layer_indices=list(v),base_coef=base,scale=1.5,inverse=False,debug=False)


@pytest.mark.parametrize('dtype',[torch.float32,torch.bfloat16])
def test_qwen2_real_author_delta_key_demand_and_hook_bit_identical(dtype):
    model=tiny('qwen2').to(dtype);tok=Tok();prompts=['<|im_start|>system\ns<|im_end|>\nu','<|im_start|>system\nss<|im_end|>\nu'];answers=['abcd','ef']
    original=p.extraction_functions(ROOT/'core/generate_vec.py',ROOT/'scripts/generate_vec_prompt_attn_k.py')
    a=original['get_prompt_attn_delta_p_and_r'](model,tok,prompts,answers)[2]
    b=g.get_prompt_attn_delta_p_and_r(model,tok,prompts,answers)[2]
    for i in range(2):assert torch.equal(a[i],b[i])
    a=original['compute_k_pos_and_avg_demand'](model,tok,prompts,answers);b=g.compute_k_pos_and_avg_demand(model,tok,prompts,answers)
    assert all(torch.equal(x,y) for x,y in zip(a,b))
    cls=native.load_author_class(ROOT/'core/gated_steerer.py','MultiLayerGatedSteerer');ids=torch.tensor([[3,4,5]])
    with torch.no_grad(),context(cls,model):a=model(ids,use_cache=False).logits
    with torch.no_grad(),context(g.MultiLayerGatedSteerer,model):b=model(ids,use_cache=False).logits
    assert torch.equal(a,b)


def capture(model):
    values={};handles=[]
    for l,block in enumerate(model.model.layers):
        for name in ('q_norm','k_norm','v_proj'):
            handles.append(getattr(block.self_attn,name).register_forward_hook(lambda m,a,o,l=l,name=name:values.__setitem__((l,name),o.detach())))
        handles.append(block.self_attn.register_forward_pre_hook(lambda m,a,kw,l=l:values.__setitem__((l,'pe'),kw['position_embeddings']),with_kwargs=True))
        handles.append(block.self_attn.o_proj.register_forward_hook(lambda m,a,o,l=l:values.__setitem__((l,'o'),o.detach())))
    return values,handles


def test_qwen3_manual_normalized_attention_leftpad_then_cached_offset_and_decomposition():
    model=tiny();ids=torch.tensor([[0,0,3,4,5],[3,4,5,6,7]]);mask=(ids!=0).long();pos=(mask.cumsum(-1)-1).masked_fill(mask==0,1)
    values,handles=capture(model)
    try:
        with torch.no_grad():out=model(ids,attention_mask=mask,position_ids=pos,output_attentions=True,use_cache=True)
        keys={};vals={}
        for l in range(2):
            q=values[l,'q_norm'].transpose(1,2);k=values[l,'k_norm'].transpose(1,2);qr,kr=apply_rotary_pos_emb(q,k,*values[l,'pe']);keys[l]=kr
            v=values[l,'v_proj'].view(2,5,2,8).transpose(1,2);vals[l]=v
            logits=qr@repeat_kv(kr,2).transpose(-2,-1)/(8**.5)
            allowed=torch.ones(5,5,dtype=torch.bool).tril()[None,None]&mask[:,None,None,:].bool()
            logits=logits+torch.where(allowed,0.,torch.finfo(logits.dtype).min)
            weights=logits.softmax(-1)
            assert torch.allclose(weights,out.attentions[l],atol=2e-7,rtol=1e-5)
            actual_o=values[l,'o'].clone();projected=[]
            for left,right in [(0,2),(2,5)]:
                contribution=weights[:,:,:,left:right]@repeat_kv(v,2)[:,:,left:right,:]
                projected.append(model.model.layers[l].self_attn.o_proj(contribution.transpose(1,2).reshape(2,5,32)))
            assert torch.allclose(projected[0]+projected[1],actual_o,atol=2e-7,rtol=1e-5)
        mask2=torch.cat([mask,torch.ones(2,1,dtype=torch.long)],1);pos2=mask.sum(-1)[:,None]
        with torch.no_grad():step=model(torch.tensor([[8],[9]]),attention_mask=mask2,position_ids=pos2,past_key_values=out.past_key_values,output_attentions=True,use_cache=True)
        for l in range(2):
            qr,kr=apply_rotary_pos_emb(values[l,'q_norm'].transpose(1,2),values[l,'k_norm'].transpose(1,2),*values[l,'pe'])
            logits=qr@repeat_kv(torch.cat([keys[l],kr],2),2).transpose(-2,-1)/(8**.5)
            logits=logits+torch.where(mask2[:,None,None,:].bool(),0.,torch.finfo(logits.dtype).min)
            assert torch.allclose(logits.softmax(-1),step.attentions[l],atol=2e-7,rtol=1e-5)
    finally:
        for h in handles:h.remove()


def test_qwen3_port_pooling_key_and_mean_demand_use_normalized_states():
    model=tiny();tok=Tok();prompt='<|im_start|>system\ns<|im_end|>\nu';answer='abcd';v,h=capture(model)
    try:
        with torch.no_grad():out=model(**tok(prompt+answer,return_tensors='pt'),output_attentions=True,use_cache=False)
    finally:
        for handle in h:handle.remove()
    syslen=len(tok.encode(prompt[:prompt.index('<|im_end|>\n')+len('<|im_end|>\n')]));plen=len(tok.encode(prompt));length=len(tok.encode(prompt+answer))
    _,_,delta=g.get_prompt_attn_delta_p_and_r(model,tok,[prompt],[answer]);key,demand=g.compute_k_pos_and_avg_demand(model,tok,[prompt],[answer])
    for l in range(2):
        qr,kr=apply_rotary_pos_emb(v[l,'q_norm'].transpose(1,2),v[l,'k_norm'].transpose(1,2),*v[l,'pe'])
        k=kr[0,:,:syslen].float().sum(1)/syslen;q=qr[0,:,plen:].float().sum(1)/(length-plen)
        assert torch.equal(key[l],k)
        assert torch.equal(demand[l],(q*k.repeat_interleave(2,0)).sum(-1).mean()/(8**.5))
        val=repeat_kv(v[l,'v_proj'].view(1,length,2,8).transpose(1,2),2)
        projected=model.model.layers[l].self_attn.o_proj((out.attentions[l][:,:,:,:syslen]@val[:,:,:syslen]).transpose(1,2).reshape(1,length,32))
        assert torch.equal(delta[l],projected[:,plen:].mean(1))


def test_positive_gate_matches_manual_native_queries_and_zero_skip_counters():
    model=tiny();delta,keys,demands=vectors(model);capt={};handles=[];expected={}
    for l,block in enumerate(model.model.layers):
        handles.append(block.self_attn.q_norm.register_forward_hook(lambda m,a,o,l=l:capt.__setitem__((l,'q'),o.detach())))
        handles.append(block.self_attn.register_forward_pre_hook(lambda m,a,kw,l=l:capt.__setitem__((l,'pe'),kw['position_embeddings']),with_kwargs=True))
        def before(m,a,o,l=l):
            q=capt[l,'q'];qr,_=apply_rotary_pos_emb(q.transpose(1,2),q.transpose(1,2),*capt[l,'pe']);qr=qr.transpose(1,2)
            d=(qr*keys[l].repeat_interleave(2,0)).sum(-1).mean(-1)/(8**.5)
            expected[l]=o[0]+(2*1.3*torch.sigmoid(1.5*(d-demands[l])))[...,None]*delta[l]
        handles.append(block.self_attn.register_forward_hook(before))
    ctx=g.MultiLayerGatedSteerer(model,delta,keys,demands,layer_indices=[0,1],base_coef=1.3,scale=1.5)
    actual=[]
    try:
        with ctx:
            for l,b in enumerate(model.model.layers):handles.append(b.self_attn.register_forward_hook(lambda m,a,o,l=l:actual.append((l,o[0].detach().clone()))))
            with torch.no_grad():model(torch.tensor([[3,4,5]]),use_cache=False)
        assert all(torch.equal(value,expected[l]) for l,value in actual)
        assert all(x['prefill']['applied']==1 and x['prefill']['skipped']==0 for x in ctx.audit.values())
    finally:
        for h in handles:h.remove()


def test_true_zero_logits_tokens_cached_prefill_decode_kwargs_and_exception_cleanup():
    model=tiny();ids=torch.tensor([[3,4,5]]);before=native.hook_signature(model)
    with torch.no_grad():ordinary=model(ids).logits;tokens=model.generate(ids,max_new_tokens=3,do_sample=False)
    ctx=context(g.MultiLayerGatedSteerer,model,0.)
    with torch.no_grad(),ctx:zero=model(ids).logits
    assert torch.equal(ordinary,zero)
    ctx=context(g.MultiLayerGatedSteerer,model,0.)
    with torch.no_grad(),ctx:generated=model.generate(ids,max_new_tokens=3,do_sample=False)
    assert torch.equal(tokens,generated)
    assert all(v['prefill']['applied']==1 and v['decode']['applied']==generated.shape[1]-ids.shape[1]-1 and sum(x['skipped'] for x in v.values())==0 for v in ctx.audit.values())
    with pytest.raises(RuntimeError):
        with context(g.MultiLayerGatedSteerer,model):raise RuntimeError('CPU_EXCEPTION')
    assert native.hook_signature(model)==before


@pytest.mark.parametrize('mode',['missing_norm','bad_head_dim','bad_norm_shape','bias','rope','sliding','other','qwen2_norm'])
def test_geometry_fail_closed(mode):
    model=tiny('qwen2' if mode=='qwen2_norm' else 'qwen3')
    if mode=='missing_norm':del model.model.layers[0].self_attn.q_norm
    elif mode=='bad_head_dim':model.config.head_dim=4
    elif mode=='bad_norm_shape':model.model.layers[0].self_attn.k_norm.weight=torch.nn.Parameter(torch.ones(4))
    elif mode=='bias':model.model.layers[0].self_attn.o_proj.bias=torch.nn.Parameter(torch.zeros(16))
    elif mode=='rope':model.config.rope_scaling={}
    elif mode=='sliding':model.config.use_sliding_window=True
    elif mode=='other':model.config.model_type='unknown'
    else:model.model.layers[0].self_attn.q_norm=torch.nn.Identity()
    with pytest.raises(ValueError):g.validate_geometry(model)


def test_skip_instrumentation_and_non_nested_instances():
    model=tiny();ctx=context(g.MultiLayerGatedSteerer,model)
    out=(torch.zeros(1,3,16),None)
    ctx._make_attn_hook(0)(None,None,out)
    assert ctx.audit[0]['prefill']['skipped']==1
    with ctx:
        with pytest.raises(ValueError,match='nested'):ctx.__enter__()
