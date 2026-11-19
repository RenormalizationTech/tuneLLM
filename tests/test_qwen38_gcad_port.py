"""Native hybrid CPU mechanics; no weights/downloads/GPU calls."""
import pytest
import torch
from transformers import Qwen3_5TextConfig, Qwen3_5ForCausalLM
from transformers.models.qwen3_5.modeling_qwen3_5 import repeat_kv
from scripts import qwen38_gcad_port as g


@pytest.fixture(autouse=True)
def threads():
    old = torch.get_num_threads(); torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def tiny():
    torch.manual_seed(12)
    c = Qwen3_5TextConfig(vocab_size=40, hidden_size=16, intermediate_size=24,
        num_hidden_layers=3, layer_types=['linear_attention', 'full_attention', 'full_attention'],
        num_attention_heads=4, num_key_value_heads=2, head_dim=16,
        linear_num_key_heads=2, linear_num_value_heads=2, linear_key_head_dim=8,
        linear_value_head_dim=8, linear_conv_kernel_dim=4,
        rope_parameters={'rope_type': 'default', 'rope_theta': 10000.,
                         'partial_rotary_factor': .25, 'mrope_section': [1, 1, 0]},
        pad_token_id=0, eos_token_id=2, bos_token_id=1, attn_implementation='eager')
    return Qwen3_5ForCausalLM(c).eval()


def tensors(m):
    return {'1.delta': torch.randn(16)*.2, '1.key': torch.randn(2,16)*.1, '1.demand': torch.tensor(.2)}


def test_native_gate_rope_gqa_weights_and_cropped_decomposition():
    m = tiny(); ids = torch.tensor([[3,4,5,6,7]])
    native = {}
    a = m.model.layers[1].self_attn
    hook = a.register_forward_hook(lambda m,args,out: native.update(output=out[0].detach(), weights=out[1].detach()))
    try:
        with torch.no_grad(), g.Capture(m,[1]) as cap:
            m(ids, use_cache=False)
            q,k,gate = cap.qk_gate(1); s=cap.state[1]
            rawq, rawgate = s['q_proj'].view(1,5,4,32).chunk(2,-1)
            assert torch.equal(gate, rawgate.reshape(1,5,-1).sigmoid())
            assert not torch.equal(rawgate.reshape(1,5,-1), s['q_proj'].chunk(2,-1)[1])
            assert torch.equal(q[...,4:], s['q_norm'].transpose(1,2)[...,4:])
            assert torch.equal(k[...,4:], s['k_norm'].transpose(1,2)[...,4:])
            scores = (q@repeat_kv(k,2).transpose(-2,-1))/4
            scores = scores.masked_fill(~torch.ones(5,5,dtype=torch.bool).tril(), torch.finfo(q.dtype).min)
            w=scores.softmax(-1, dtype=torch.float32).to(q.dtype)
            torch.testing.assert_close(w, native['weights'], atol=1e-7, rtol=1e-5)
            v=repeat_kv(s['v_proj'].view(1,5,2,16).transpose(1,2),2)
            crop=lambda left,right:(w[...,left:right]@v[...,left:right,:]).transpose(1,2).reshape(1,5,-1)
            system, other=crop(0,2),crop(2,5)
            torch.testing.assert_close(a.o_proj(system*gate)+a.o_proj(other*gate),native['output'],atol=1e-7,rtol=1e-5)
            assert not torch.allclose(a.o_proj(system)+a.o_proj(other),native['output'],atol=1e-7,rtol=1e-5)
            expected=a.o_proj(system*gate)[0,3:].mean(0)
        extracted=g.extract(m,ids,2,3,[False]*3+[True]*2,[1])
        torch.testing.assert_close(extracted[1]['delta'],expected)
    finally: hook.remove()
    assert all(not x._forward_hooks and not x._forward_pre_hooks for x in m.modules())


def test_zero_cached_generation_logits_shift_and_cleanup():
    m=tiny(); ids=torch.tensor([[3,4,5]])
    b=tensors(m)
    kw=dict(use_cache=True, do_sample=False, max_new_tokens=4, eos_token_id=None, pad_token_id=0)
    with torch.no_grad():
        before=m(ids,use_cache=False).logits.clone()
        ordinary=m.generate(ids,**kw)
        with g.Steerer(m,b,[1],0) as zero:
            assert torch.equal(before,m(ids,use_cache=False).logits)
            assert torch.equal(ordinary,m.generate(ids,**kw))
            assert zero.audit[1]['prefill']>=2 and zero.audit[1]['decode']==3
            assert not any(x._forward_hooks or x._forward_pre_hooks for x in m.model.layers[0].modules())
            assert not any(x._forward_hooks or x._forward_pre_hooks for x in m.model.layers[2].modules())
        with g.Steerer(m,b,[1],1):
            assert not torch.equal(before,m(ids,use_cache=False).logits)
        assert torch.equal(before,m(ids,use_cache=False).logits)
    assert all(not x._forward_hooks and not x._forward_pre_hooks for x in m.modules())


def test_reject_linear_geometry_and_cleanup_on_exception():
    m=tiny()
    with pytest.raises(ValueError,match='linear-attention'):g.Capture(m,[0])
    with pytest.raises(RuntimeError):
        with g.Steerer(m,tensors(m),[1],1):raise RuntimeError('failure')
    assert all(not x._forward_hooks and not x._forward_pre_hooks for x in m.modules())


def test_bank_raw_pair_and_token_weighted_key_demand():
    def row(delta, nk, nq):
        return {1:dict(delta=torch.ones(16)*delta,key_sum=torch.ones(2,16)*nk*2,
            query_sum=torch.ones(4,16)*nq*3,key_count=nk,query_count=nq)}
    b=g.bank([row(5,2,3),row(7,4,5)],[row(1,2,3),row(3,4,5)],[1])
    assert torch.equal(b['1.delta'],torch.ones(16)*4)
    assert torch.equal(b['1.key'],torch.ones(2,16)*2)
    assert b['1.demand'].item()==24


def test_nonzero_cached_decode_matches_full_prefix_logits_and_tokens():
    m=tiny(); b=tensors(m);prefix=torch.tensor([[3,4,5]])
    with torch.no_grad(),g.Steerer(m,b,[1],1.):
        cached=m.generate(prefix,do_sample=False,use_cache=True,max_new_tokens=4,
            eos_token_id=None,pad_token_id=0,return_dict_in_generate=True,output_logits=True)
    current=prefix.clone()
    with torch.no_grad(),g.Steerer(m,b,[1],1.):
        for step in range(4):
            logits=m(current,use_cache=False).logits[:,-1]
            torch.testing.assert_close(logits,cached.logits[step],atol=2e-6,rtol=2e-5)
            token=logits.argmax(-1,keepdim=True)
            assert torch.equal(token,cached.sequences[:,prefix.shape[1]+step:prefix.shape[1]+step+1])
            current=torch.cat([current,token],1)


def test_distinct_gqa_mapping_and_manual_shift():
    m=tiny(); b=tensors(m); b['1.key']=torch.stack([torch.ones(16),torch.ones(16)*3])
    ids=torch.tensor([[3,4,5]]);reference={}
    hook=m.model.layers[1].self_attn.register_forward_hook(lambda m,a,o:reference.update(original=o[0].detach().clone()))
    try:
        with torch.no_grad(),g.Steerer(m,b,[1],1.) as ctx:
            # Save native Q before shift consumes capture state.
            qhook=m.model.layers[1].self_attn.q_norm.register_forward_hook(lambda m,a,o:reference.update(q=o.detach().clone()))
            pre=m.model.layers[1].self_attn.register_forward_pre_hook(lambda m,a,kw:reference.update(pe=kw['position_embeddings']),with_kwargs=True)
            try:
                captured={}
                final=m.model.layers[1].self_attn.register_forward_hook(lambda m,a,o:captured.update(shifted=o[0].detach().clone()))
                try:m(ids,use_cache=False)
                finally:final.remove()
                q,_=g.apply_rotary_pos_emb(reference['q'].transpose(1,2),reference['q'].transpose(1,2),*reference['pe'])
                manual=torch.stack([b['1.key'][h//2] for h in range(4)])
                d=(q.transpose(1,2)*manual).sum(-1).mean(-1)/4
                expected=reference['original']+2*torch.sigmoid(1.5*(d-b['1.demand'])).unsqueeze(-1)*b['1.delta']
                torch.testing.assert_close(captured['shifted'],expected)
            finally:qhook.remove();pre.remove()
    finally:hook.remove()
    # Distinct head weights also expose incorrect tiling in bank centering.
    r={1:dict(delta=torch.ones(16),key_sum=b['1.key'],key_count=1,
             query_sum=torch.stack([torch.ones(16)*h for h in [1,2,4,8]]),query_count=1)}
    result=g.bank([r],[r],[1])
    assert result['1.demand'].item()==(1+2+4*3+8*3)*4/4


def test_sdpa_and_eager_extraction_same_bounded_crop():
    m=tiny();ids=torch.tensor([[3,4,5,6,7]])
    eager=g.extract(m,ids,2,3,[False]*3+[True]*2,[1])
    m.set_attn_implementation('sdpa')
    sdpa=g.extract(m,ids,2,3,[False]*3+[True]*2,[1])
    for key in ['delta','key_sum','query_sum']:
        torch.testing.assert_close(eager[1][key],sdpa[1][key],atol=2e-6,rtol=2e-5)
