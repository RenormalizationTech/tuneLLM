"""Real tiny Qwen2 mechanics; no target weights, downloads, CUDA or judgments."""
import copy
import importlib.util
from pathlib import Path
import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM
from scripts.joint_gcad import JointGCAD
from scripts.author_gcad_impolite_common import hook_signature

PUBLIC = Path('outputs/steering-release-staging-03/runtime/gcad.py')
pytestmark = pytest.mark.artifact(str(PUBLIC))
if not PUBLIC.is_file():
    pytest.skip('requires local published GCAD runtime copy; no download during collection', allow_module_level=True)
spec = importlib.util.spec_from_file_location('published_gcad', PUBLIC)
published = importlib.util.module_from_spec(spec); spec.loader.exec_module(published)

@pytest.fixture(autouse=True)
def cpu_threads():
    old = torch.get_num_threads(); torch.set_num_threads(1)
    try: yield
    finally: torch.set_num_threads(old)

def tiny(dtype=torch.float32):
    with torch.random.fork_rng():
        torch.manual_seed(72)
        return Qwen2ForCausalLM(Qwen2Config(vocab_size=32, hidden_size=16,
            intermediate_size=24, num_hidden_layers=2, num_attention_heads=4,
            num_key_value_heads=2, max_position_embeddings=64, pad_token_id=0,
            bos_token_id=1, eos_token_id=None, attention_bias=False,
            use_sliding_window=False, attn_implementation='sdpa')).to(dtype).eval()

def components():
    gen = torch.Generator().manual_seed(193)
    return {name: {'delta': torch.randn(2,16,generator=gen)*.08,
        'key': torch.randn(2,2,4,generator=gen), 'demand': torch.randn(2,generator=gen),
        'base_coef': b, 'scale': 1.5, 'inverse': False}
        for name,b in [('profanity',5.),('impolite',3.5),('sarcasm',2.)]}

def reference(model,c):
    return published.MultiLayerGatedSteerer(model,
        {l:c['delta'][l] for l in [0,1]}, {l:c['key'][l] for l in [0,1]},
        {l:c['demand'][l].item() for l in [0,1]}, layer_indices=[0,1],
        base_coef=c['base_coef'],scale=c['scale'],inverse=c['inverse'])

@pytest.mark.parametrize('dtype',[torch.float32,torch.bfloat16])
@pytest.mark.parametrize('trait',['profanity','impolite','sarcasm'])
def test_single_trait_exact_published_core_prefill_and_cache(dtype,trait):
    model=tiny(dtype); c=components()[trait]; ids=torch.tensor([[3,4,5]])
    def run(ctx):
        with torch.no_grad(),ctx:
            a=model(ids,use_cache=True)
            b=model(torch.tensor([[6]]),past_key_values=a.past_key_values,use_cache=True)
        return a.logits,b.logits
    expected=run(reference(model,c)); ctx=JointGCAD(model,{trait:c},layer_indices=[0,1]); actual=run(ctx)
    assert all(torch.equal(a,b) for a,b in zip(expected,actual))
    snap=ctx.audit_snapshot(); assert snap['hooks_removed']
    for l in [0,1]:
        assert snap['coverage'][l]=={'prefill':{'applied':1,'skipped':0},'decode':{'applied':1,'skipped':0}}
        for phase,n in [('prefill',3),('decode',1)]:
            assert snap['measurements'][l][phase][trait]['positions']==n
            assert snap['measurements'][l][phase][trait]['finite']

@pytest.mark.parametrize('dtype',[torch.float32,torch.bfloat16])
def test_zero_all_three_exact_logits_tokens_and_cleanup(dtype):
    model=tiny(dtype); c=components(); ids=torch.tensor([[3,4,5]])
    for v in c.values():v['base_coef']=0.
    before=hook_signature(model)
    with torch.no_grad(): baseline=model(ids).logits; tokens=model.generate(ids,max_new_tokens=3,do_sample=False)
    with torch.no_grad(),JointGCAD(model,c,layer_indices=[0,1]):zero=model(ids).logits
    ctx=JointGCAD(model,c,layer_indices=[0,1])
    with torch.no_grad(),ctx:ztokens=model.generate(ids,max_new_tokens=3,do_sample=False)
    assert torch.equal(baseline,zero) and torch.equal(tokens,ztokens)
    assert hook_signature(model)==before
    for phases in ctx.audit_snapshot()['measurements'].values():
        for values in phases.values():assert values['sum']['norm_sum']==0

@pytest.mark.parametrize('dtype',[torch.float32,torch.bfloat16])
def test_sum_independent_source_gates_same_query_one_output_add(dtype):
    model=tiny(dtype); c=components(); ctx=JointGCAD(model,c,layer_indices=[0,1])
    expected={}; actual={}; handles=[]; source={}
    # Independent actual published hook evaluations all receive the SAME native
    # output/query/rotary state, never sequentially altered residual outputs.
    for l,b in enumerate(model.model.layers):
        source[l]=[reference(model,v) for _,v in sorted(c.items())]
        def before(m,args,out,l=l):
            shifts=[]
            for ref in source[l]:
                ref._q_captured[l]=ctx._q_captured[l]
                ref._pos_emb[l]=ctx._pos_emb[l]
                # Recover exact shift before source output addition (avoids
                # subtractive rounding); direct source formula on shared query.
                q=ref._q_captured[l]; qr=q.transpose(1,2);cos,sin=ref._pos_emb[l]
                qr,_=published.apply_rotary_pos_emb(qr,qr,cos.to(q),sin.to(q));qr=qr.transpose(1,2)
                demand=(qr*ref._k[l].to(q)).sum(-1).mean(-1)/(ref.head_dim**.5)
                coef=2.*ref.base_coef*torch.sigmoid(ref._sign*ref.scale*(demand-ref._avg_d[l]))
                shift=coef[...,None]*ref._v[l].to(q)
                assert torch.equal(ref._make_attn_hook(l)(m,args,out)[0],out[0]+shift)
                shifts.append(shift)
            total=shifts[0]
            for v in shifts[1:]:total=total+v
            expected[l]=out[0]+total
        handles.append(b.self_attn.register_forward_hook(before))
    try:
        with torch.no_grad(),ctx:
            for l,b in enumerate(model.model.layers):handles.append(b.self_attn.register_forward_hook(lambda m,a,o,l=l:actual.__setitem__(l,o[0].clone())))
            model(torch.tensor([[3,4,5]]))
        assert all(torch.equal(expected[l],actual[l]) for l in [0,1])
        assert any(v['sum']['norm_sum']>0 for phases in ctx.audit_snapshot()['measurements'].values() for v in phases.values() if v)
    finally:
        for h in handles:h.remove()


def test_missing_capture_fails_closed_exception_cleanup_no_state_leak():
    model=tiny(); before=hook_signature(model); ctx=JointGCAD(model,components(),layer_indices=[0,1])
    with pytest.raises(ValueError,match='missing shared'):
        with ctx:
            ctx._make_attn_hook(0)(model.model.layers[0].self_attn,(),(torch.zeros(1,2,16),None))
    assert hook_signature(model)==before and not ctx._q_captured and not ctx._pos_emb and not ctx._device_vectors
    with pytest.raises(ValueError,match='fresh context'):ctx.__enter__()

@pytest.mark.parametrize('bad',['nan','shape','dtype','strength','blocks','model','reserved'])
def test_explicit_invalid_geometry_or_parameters_rejected(bad):
    model=tiny(); c=components(); layers=[0,1]
    if bad=='nan':c['sarcasm']['key'][0,0,0]=float('nan')
    elif bad=='shape':c['sarcasm']['delta']=torch.ones(2,15)
    elif bad=='dtype':c['sarcasm']['demand']=torch.ones(2,dtype=torch.float64)
    elif bad=='strength':c['sarcasm']['base_coef']=float('inf')
    elif bad=='blocks':layers=[0,0]
    elif bad=='model':model.config.model_type='qwen3'
    else:c['sum']=c.pop('sarcasm')
    with pytest.raises(ValueError):JointGCAD(model,c,layer_indices=layers)
    assert all(not b.self_attn._forward_hooks for b in model.model.layers)

@pytest.mark.parametrize('dtype',[torch.float32,torch.bfloat16])
def test_canonical_order_permutation_bitwise_independent(dtype):
    import itertools
    model=tiny(dtype); c=components(); ids=torch.tensor([[3,4,5]])
    results=[]
    with torch.no_grad():
        for names in itertools.permutations(c):
            ctx=JointGCAD(model,{n:c[n] for n in names},layer_indices=[0,1])
            with ctx:
                a=model(ids,use_cache=True)
                b=model(torch.tensor([[6]]),past_key_values=a.past_key_values,use_cache=True)
            results.append((a.logits,b.logits))
    assert all(torch.equal(a,b) for result in results for a,b in zip(results[0],result))


def test_real_public_bank_hash_shapes_and_fake_attention_math():
    # Only small CPU bank tensors are loaded; no base model weights.
    import hashlib,json
    from safetensors.torch import load_file
    from types import SimpleNamespace
    from torch import nn
    root=PUBLIC.parent.parent
    manifest=json.loads((root/'release_manifest.json').read_text())
    c={}
    for trait in ('profanity','impolite','sarcasm'):
        bank=root/trait/'bank.safetensors'; profile=json.loads((root/trait/'profile.json').read_text())
        digest=hashlib.sha256(bank.read_bytes()).hexdigest()
        assert digest==manifest['files_sha256'][trait+'/bank.safetensors']==profile['bank']['sha256']
        v=load_file(str(bank),device='cpu')
        assert {k:tuple(t.shape) for k,t in v.items()}=={'delta':(28,3584),'key':(28,4,128),'demand':(28,)}
        for name,t in v.items():
            assert hashlib.sha256(t.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()==profile['bank']['tensor_sha256'][name]
        c[trait]={**v,**{k:profile['controller'][k] for k in ('base_coef','scale','inverse')}}
    # Real 7B geometry without allocating its large Linear weights.
    attention=lambda:SimpleNamespace(q_proj=SimpleNamespace(out_features=3584,weight=torch.zeros((),dtype=torch.bfloat16)),
        k_proj=SimpleNamespace(out_features=512),v_proj=SimpleNamespace(out_features=512),
        o_proj=SimpleNamespace(in_features=3584,out_features=3584,bias=None))
    cfg=SimpleNamespace(model_type='qwen2',hidden_size=3584,num_hidden_layers=28,
        num_attention_heads=28,num_key_value_heads=4,rope_scaling=None,use_sliding_window=False)
    model=SimpleNamespace(config=cfg,model=SimpleNamespace(layers=[SimpleNamespace(self_attn=attention()) for _ in range(28)]))
    ctx=JointGCAD(model,c,layer_indices=list(range(9,20)))
    gen=torch.Generator().manual_seed(91);q=torch.randn(1,2,28,128,generator=gen).bfloat16();out=torch.zeros(1,2,3584,dtype=torch.bfloat16)
    cos=torch.ones(1,2,128,dtype=torch.bfloat16);sin=torch.zeros_like(cos)
    for l in range(9,20):
        ctx._q_captured[l]=q;ctx._pos_emb[l]=(cos,sin)
        expected=None
        for trait in sorted(c):
            x=ctx.components[trait];d=(q*x._k[l]).sum(-1).mean(-1)/(128**.5)
            shift=(2.*x.base_coef*torch.sigmoid(1.5*(d-x._avg_d[l])))[...,None]*x._v[l]
            expected=shift if expected is None else expected+shift
        actual=ctx._make_attn_hook(l)(None,(),(out,None))[0]
        assert torch.equal(actual,out+expected) and torch.isfinite(actual).all()
    assert all(ctx.audit[l]['prefill']['applied']==1 for l in range(9,20))


def test_study_factory_baseline_null_and_single_omits_inactive():
    from scripts.joint_gcad import controller
    model=tiny();c=components();before=hook_signature(model)
    for v in c.values():v['base_coef']=0.
    with controller(model,c,layer_indices=[0,1]) as ctx:
        assert ctx is None and hook_signature(model)==before
    c['sarcasm']['base_coef']=2.
    ctx=controller(model,c,layer_indices=[0,1])
    assert list(ctx.components)==['sarcasm']
    with torch.no_grad(),ctx:actual=model(torch.tensor([[3,4,5]])).logits
    with torch.no_grad(),reference(model,c['sarcasm']):expected=model(torch.tensor([[3,4,5]])).logits
    assert torch.equal(actual,expected)
