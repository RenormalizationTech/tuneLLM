"""CPU-only packaging guards and actual tinyQwen2 zero hook; never live weights."""
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import copy
import pytest
import torch
from safetensors.torch import save_file
from scripts import stage_steering_release as a


@pytest.fixture(autouse=True)
def cpu(monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','');old=torch.get_num_threads();torch.set_num_threads(1)
    try:yield
    finally:torch.set_num_threads(old)


def values():return {k:torch.randn(shape,generator=torch.Generator().manual_seed(17)) for k,shape in a.SHAPES.items()}


def put(p,value):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value)+'\n')


def profile(trait,v):
    return {'trait':trait,'base':{'id':a.BASE,'revision':a.REV},
        'controller':{'blocks':list(range(9,20)),'base_coef':a.COEFS[trait],'scale':1.5,'inverse':False},
        'bank':{'tensor_sha256':{k:a.tensor_hash(x) for k,x in v.items()}},'limits':['CPU synthetic packaging fixture']}


@pytest.fixture
def bundle(tmp_path):
    v=values();root=tmp_path/'bundle';root.mkdir()
    for n,raw in a.runtime_files().items():
        p=root/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw)
    for n in ('README.md','LICENSE','NOTICE'):(root/n).write_text('CPU_TEST_ONLY\n')
    for trait in a.TRAITS:
        folder=root/trait;folder.mkdir();save_file(v,str(folder/'bank.safetensors'))
        p=profile(trait,v);p['bank']['sha256']=a.sha(folder/'bank.safetensors');put(folder/'profile.json',p)
        put(folder/'provenance.json',{'source_tensor_sha256':p['bank']['tensor_sha256']});(folder/'README.md').write_text('CPU_TEST_ONLY\n')
    m={'private':True,'status':'local_draft_not_uploaded','files_sha256':{p.relative_to(root).as_posix():a.sha(p) for p in root.rglob('*') if p.is_file()}}
    put(root/'release_manifest.json',m)
    return root,v


def rehash(root):
    m=a.read(root/'release_manifest.json');m['files_sha256']={p.relative_to(root).as_posix():a.sha(p) for p in root.rglob('*') if p.is_file() and p.name!='release_manifest.json'};put(root/'release_manifest.json',m)


def runtime(root):
    name='cpu_release_runtime_'+str(abs(hash(str(root))))
    spec=importlib.util.spec_from_file_location(name,root/'runtime/__init__.py',submodule_search_locations=[str(root/'runtime')]);pkg=importlib.util.module_from_spec(spec);sys.modules[name]=pkg;spec.loader.exec_module(pkg)
    spec=importlib.util.spec_from_file_location(name+'.loader',root/'runtime/loader.py');module=importlib.util.module_from_spec(spec);sys.modules[name+'.loader']=module;spec.loader.exec_module(module)
    return module


def test_bundle_roundtrip_and_exact_port_copy(bundle):
    root,v=bundle;checked=a.preflight(root);assert checked['file_count']==20
    loader=runtime(root);p,actual=loader.profile_and_tensors(root,'profanity',expected_manifest_sha256=a.sha(root/'release_manifest.json'))
    assert all(torch.equal(actual[k],v[k]) for k in v)
    assert (root/'runtime/gcad.py').read_bytes()==a.local(a.PORT).read_bytes()


@pytest.mark.parametrize('mode',['pickle','path','private','absolute','secret','runtime','private_flag'])
def test_preflight_rejects_even_rehashed_injections(bundle,mode):
    root,_=bundle
    if mode=='pickle':(root/'unexpected.pt').write_bytes(b'NOT_A_PICKLE')
    elif mode=='path':
        m=a.read(root/'release_manifest.json');m['files_sha256']['../escape']='0'*64;put(root/'release_manifest.json',m)
    elif mode=='runtime':
        f=root/'runtime/example.py';f.write_text(f.read_text()+'\nimport pickle\n');rehash(root)
    elif mode=='private_flag':
        m=a.read(root/'release_manifest.json');m['private']=False;put(root/'release_manifest.json',m)
    else:
        f=root/'profanity/profile.json';p=a.read(f)
        if mode=='private':p['answer']='CPU_PRIVATE_TEXT'
        elif mode=='absolute':p['limits']=['/home/cpu_private/cache']
        else:p['limits']=['hf_'+'x'*24]
        put(f,p);rehash(root)
    with pytest.raises(ValueError):a.preflight(root)


def test_symlink_and_tensor_file_hash_rejection(bundle):
    root,_=bundle;file=root/'NOTICE';file.unlink();file.symlink_to(root/'LICENSE')
    with pytest.raises(ValueError,match='symlink'):a.preflight(root)
    loader=runtime(root);f=root/'profanity/bank.safetensors';f.write_bytes(f.read_bytes()+b'CPU_CORRUPT')
    with pytest.raises(ValueError,match='hash'):loader.profile_and_tensors(root,'profanity',expected_manifest_sha256=a.sha(root/'release_manifest.json'))


@pytest.mark.parametrize('mode',['revision','shape','tensorhash'])
def test_runtime_rejects_revision_shape_and_tensor_hash_even_new_manifest(bundle,mode):
    root,v=bundle;p=a.read(root/'sarcasm/profile.json')
    if mode=='revision':p['base']['revision']='0'*40
    elif mode=='tensorhash':p['bank']['tensor_sha256']['delta']='0'*64
    else:
        v['key']=v['key'][:,:3].contiguous();save_file(v,str(root/'sarcasm/bank.safetensors'));p['bank']['sha256']=a.sha(root/'sarcasm/bank.safetensors');p['bank']['tensor_sha256']['key']=a.tensor_hash(v['key'])
    put(root/'sarcasm/profile.json',p);rehash(root);loader=runtime(root)
    with pytest.raises(ValueError):loader.profile_and_tensors(root,'sarcasm',expected_manifest_sha256=a.sha(root/'release_manifest.json'))


def test_source_hash_verified_before_weights_only_load(tmp_path,monkeypatch):
    v=values();p=profile('profanity',v);p['model']={'id':a.BASE,'revision':a.REV,'dtype':'bfloat16','attention':'sdpa'}
    p['controller'].update(bank={'path':'bank.pt','sha256':'0'*64},object_key='gcad',tensor_sha256={k:a.tensor_hash(x) for k,x in v.items()})
    p['basis']={'tokenizer_sha256':'b'*64,'rendering_sha256':'c'*64};p['measured_heldout']={'baseline':{},'selected':{},'random':{}}
    # Test-only old-format source; never put it in a staged/public bundle.
    torch.save({'gcad':{'tensors':v}},tmp_path/'bank.pt');put(tmp_path/'profile.json',p)
    monkeypatch.setattr(a,'ROOT',tmp_path);monkeypatch.setattr(a,'PROFILE_FILES',{'profanity':'profile.json'});monkeypatch.setattr(a,'committed',lambda n:None)
    monkeypatch.setattr(torch,'load',lambda *args,**kwargs:pytest.fail('mismatched hash must fail before deserialization'))
    with pytest.raises(ValueError,match='hash mismatch'):a.load_source_profile('profanity')


def test_source_tensor_only_positive_conversion_contract(tmp_path,monkeypatch):
    v=values();torch.save({'gcad':{'tensors':v}},tmp_path/'bank.pt');p=profile('profanity',v)
    p['model']={'id':a.BASE,'revision':a.REV,'dtype':'bfloat16','attention':'sdpa'}
    p['controller'].update(bank={'path':'bank.pt','sha256':a.sha(tmp_path/'bank.pt')},object_key='gcad',tensor_sha256={k:a.tensor_hash(x) for k,x in v.items()})
    p['basis']={'tokenizer_sha256':'b'*64,'rendering_sha256':'c'*64};p['measured_heldout']={'baseline':{},'selected':{},'random':{}};put(tmp_path/'profile.json',p)
    monkeypatch.setattr(a,'ROOT',tmp_path);monkeypatch.setattr(a,'PROFILE_FILES',{'profanity':'profile.json'});monkeypatch.setattr(a,'committed',lambda n:None)
    original=torch.load;calls=[]
    def load(*args,**kwargs):calls.append(kwargs);return original(*args,**kwargs)
    monkeypatch.setattr(torch,'load',load);actual,_,_,_=a.load_source_profile('profanity')
    assert all(torch.equal(v[k],actual[k]) for k in v) and calls==[{'map_location':'cpu','weights_only':True}]


def test_paths_and_finite_geometry_guards():
    with pytest.raises(ValueError):a.local('../escape')
    with pytest.raises(ValueError):a.local('/home/private')
    v=values();v['demand'][0]=float('nan')
    with pytest.raises(ValueError):a.validate_tensors(v)


def test_actual_staged_qwen2_zero_logits_tokens_and_cleanup(bundle):
    from transformers import Qwen2Config,Qwen2ForCausalLM
    loader=runtime(bundle[0]);module=sys.modules[loader.__package__+'.gcad'];torch.manual_seed(19)
    model=Qwen2ForCausalLM(Qwen2Config(vocab_size=20,hidden_size=16,intermediate_size=24,num_hidden_layers=2,
        num_attention_heads=4,num_key_value_heads=2,pad_token_id=0,eos_token_id=2,attn_implementation='sdpa')).to(torch.bfloat16).eval()
    ids=torch.tensor([[3,4,5]]);v={i:torch.randn(16) for i in range(2)};k={i:torch.randn(2,4) for i in range(2)};d={i:0. for i in range(2)}
    before={n:(len(m._forward_hooks),len(m._forward_pre_hooks)) for n,m in model.named_modules()}
    def ctx():return module.MultiLayerGatedSteerer(model,v,k,d,layer_indices=[0,1],base_coef=0.,scale=1.5)
    with torch.no_grad():ordinary=model(ids,use_cache=False).logits;tokens=model.generate(ids,max_new_tokens=3,do_sample=False)
    with torch.no_grad(),ctx():zero=model(ids,use_cache=False).logits
    c=ctx()
    with torch.no_grad(),c:generated=model.generate(ids,max_new_tokens=3,do_sample=False)
    assert torch.equal(ordinary,zero) and torch.equal(tokens,generated)
    assert all(sum(p['skipped'] for p in l.values())==0 for l in c.audit.values())
    assert before=={n:(len(m._forward_hooks),len(m._forward_pre_hooks)) for n,m in model.named_modules()}


def test_loader_base_id_and_revision_rejected_before_model_calls(bundle):
    loader=runtime(bundle[0]);profile=loader.profile_and_tensors(bundle[0],'profanity',expected_manifest_sha256=a.sha(bundle[0]/'release_manifest.json'))[0]
    model=SimpleNamespace(config=SimpleNamespace());tokenizer=SimpleNamespace()
    with pytest.raises(ValueError,match='exact base'):loader.validate_base(model,tokenizer,a.REV,profile,base_id='Qwen/Qwen3-4B',revision=a.REV)
    with pytest.raises(ValueError,match='exact base'):loader.validate_base(model,tokenizer,a.REV,profile,base_id=a.BASE,revision='0'*40)


def test_tampered_model_manifest_fails_before_read_use_or_any_write(tmp_path,monkeypatch):
    manifest=tmp_path/'model_manifest.json';put(manifest,[{'model':a.BASE,'revision':a.REV,'files':[]}])
    expected=a.sha(manifest)
    put(tmp_path/'pf_profile.json',{'model':{'manifest':{'path':'model_manifest.json','sha256':expected}}})
    put(manifest,[{'model':a.BASE,'revision':a.REV,'files':[{'path':'TAMPERED'}]}])
    dest=tmp_path/'new_bundle';monkeypatch.setattr(a,'ROOT',tmp_path);monkeypatch.setattr(a,'DEST',dest)
    monkeypatch.setattr(a,'PROFILE_FILES',{'profanity':'pf_profile.json'});monkeypatch.setattr(a,'committed',lambda n:None)
    monkeypatch.setattr(torch,'load',lambda *args,**kwargs:pytest.fail('do not load any bank after metadata hash failure'))
    before={p.name:p.read_bytes() for p in tmp_path.iterdir()}
    with pytest.raises(ValueError,match='hash mismatch'):a.stage()
    assert not dest.exists() and before=={p.name:p.read_bytes() for p in tmp_path.iterdir()}
