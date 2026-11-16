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
    m={'private':True,'status':'local_draft_not_uploaded','prior_validation':a.prior_validation(),'files_sha256':{p.relative_to(root).as_posix():a.sha(p) for p in root.rglob('*') if p.is_file()}}
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


@pytest.mark.parametrize('tamper',[None,'padding_state','merges','normalizer','added_token','template','wrong_pad','right_padding'])
def test_real_cached_pinned_tokenizer_reference_pass_and_semantic_tamper_rejection(bundle,tamper):
    from transformers import AutoTokenizer
    from tokenizers import Tokenizer
    snapshot=a.local('.hf/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/'+a.REV)
    names=('config.json','generation_config.json','tokenizer.json','tokenizer_config.json','vocab.json','merges.txt')
    metadata={name:a.sha(snapshot/name) for name in names}
    before={name:(snapshot/name).read_bytes() for name in names}
    tokenizer=AutoTokenizer.from_pretrained(snapshot,local_files_only=True,trust_remote_code=False)
    tokenizer.pad_token=tokenizer.eos_token;tokenizer.padding_side='left'
    loader=runtime(bundle[0])
    if tamper in ('merges','normalizer'):
        backend=json.loads(tokenizer.backend_tokenizer.to_str())
        if tamper=='merges':backend['model']['merges'][0],backend['model']['merges'][1]=backend['model']['merges'][1],backend['model']['merges'][0]
        else:backend['normalizer']={'type':'NFKC'}
        tokenizer._tokenizer=Tokenizer.from_str(json.dumps(backend))
        with pytest.raises(ValueError,match='actual tokenizer backend differs'):loader.validate_tokenizer(tokenizer,snapshot,metadata)
    elif tamper in ('added_token','template','wrong_pad','right_padding'):
        if tamper=='added_token':tokenizer.add_special_tokens({'additional_special_tokens':['<|CPU_EXTRA_SPECIAL|>']})
        elif tamper=='template':tokenizer.chat_template+='\nCPU_TEMPLATE_TAMPER'
        elif tamper=='wrong_pad':tokenizer.pad_token='<|im_start|>'
        else:tokenizer.padding_side='right'
        with pytest.raises(ValueError):loader.validate_tokenizer(tokenizer,snapshot,metadata)
    elif tamper=='padding_state':
        tokenizer(['a','bb'],padding=True)
        loader.validate_tokenizer(tokenizer,snapshot,metadata)
    else:loader.validate_tokenizer(tokenizer,snapshot,metadata)
    assert before=={name:(snapshot/name).read_bytes() for name in names}


@pytest.fixture
def public_fixture(tmp_path,monkeypatch):
    root=tmp_path/'public'
    monkeypatch.setattr(a,'PUBLIC_DEST',root)
    monkeypatch.setattr(a,'PUBLIC_UPLOAD_MANIFEST',tmp_path/'upload.json')
    monkeypatch.setattr(a,'resolve_source_ref',lambda ref:{'tag':ref,'commit':'a'*40})
    result=a.finalize_public('cpu-tested-tag')
    assert result['upload_performed'] is False and result['file_count']==23
    return root


def test_public_inference_core_byte_identity_and_strict_loader_cpu(public_fixture):
    root=public_fixture;result=a.preflight(root,public=True);assert result['pass']
    for name in a.core_names():assert (root/name).read_bytes()==(a.DEST/name).read_bytes()
    loader=runtime(root);p,v=loader.profile_and_tensors(root,'profanity',expected_manifest_sha256=a.sha(root/'release_manifest.json'))
    assert p['base']['revision']==a.REV and set(v)==set(a.SHAPES)
    assert a.read(root/'release_manifest.json')['validation']['public03_gpu_rerun'] is False


@pytest.mark.parametrize('tamper',['private','status','hash','secret','extra','core'])
def test_public_mode_rejects_visibility_status_hash_secret_extra_core_change(public_fixture,tamper):
    root=public_fixture;m=a.read(root/'release_manifest.json')
    if tamper=='private':m['private']=True
    elif tamper=='status':m['status']='local_draft_not_uploaded'
    elif tamper=='hash':m['files_sha256']['README.md']='0'*64
    elif tamper=='secret':
        (root/'README.md').write_text('Bearer '+'x'*24);m['files_sha256']['README.md']=a.sha(root/'README.md')
    elif tamper=='extra':(root/'unexpected.txt').write_text('CPU_TEST')
    else:
        file=root/'profanity/profile.json';p=a.read(file);p['runtime']['real_7b_reload_validation']='MUTATED_ARCHIVE';put(file,p);m['files_sha256']['profanity/profile.json']=a.sha(file)
    put(root/'release_manifest.json',m)
    with pytest.raises(ValueError):a.preflight(root,public=True)


def test_public_tag_guard_requires_committed_builder_and_resolves_exact_tag(monkeypatch):
    calls=[]
    def git(argv,**kwargs):
        calls.append(argv)
        if argv[1]=='rev-parse':return SimpleNamespace(stdout=('a'*40+'\n').encode())
        name=argv[2].split(':',1)[1]
        return SimpleNamespace(stdout=a.local(name).read_bytes())
    monkeypatch.setattr(a.subprocess,'run',git)
    assert a.resolve_source_ref('steering-controls-v1')=={'tag':'steering-controls-v1','commit':'a'*40}
    assert calls[0]==['git','rev-parse','--verify','refs/tags/steering-controls-v1^{commit}']
    calls.clear()
    with pytest.raises(ValueError):a.resolve_source_ref('../unsafe')
    assert not calls
    def stale(argv,**kwargs):
        if argv[1]=='rev-parse':return SimpleNamespace(stdout=('a'*40+'\n').encode())
        return SimpleNamespace(stdout=b'CPU_OLD_COMMITTED_BUILDER')
    monkeypatch.setattr(a.subprocess,'run',stale)
    with pytest.raises(ValueError,match='source tag does not bind'):a.resolve_source_ref('steering-controls-v1')


def test_public_upload_schema_sanitized_derivatives_and_no_self_hash(public_fixture):
    root=public_fixture;m=a.read(root/'release_manifest.json');upload=a.read(a.PUBLIC_UPLOAD_MANIFEST)
    assert set(upload)=={'repo_id','private','files'} and upload['repo_id']==a.PUBLIC_REPO and upload['private'] is False
    assert set(upload['files'])==a.allowlist(public=True) and len(upload['files'])==23
    for name,item in upload['files'].items():assert item=={'bytes':(root/name).stat().st_size,'sha256':a.sha(root/name)}
    assert 'release_manifest.json' not in m['files_sha256']
    public=a.read(root/'related_results/qwen3_4b_fixed_transfer_01.public.json');original=a.read(a.local(a.FOURB_RESULT))
    assert public['traits']==original['traits'] and public['costs']==original['costs']
    assert public['native_identity']['runtime_config']==Path(original['native_identity']['runtime_config']).name
    assert public['native_identity']['runtime_config_sha256']==original['native_identity']['runtime_config_sha256']
    card=(root/'README.md').read_text()
    assert a.sha(root/'release_manifest.json') not in card
    assert 'Release manifest SHA256: H' in card and 'revision=PINNED_HF_SHA' in card
    assert 'accelerate==1.7.0' in card and 'local_dir="controls"' in card
    assert 'github.com/' not in card


@pytest.mark.parametrize('tamper',['private_json_path','dead_relative_link','private_github_link'])
def test_public_path_and_link_scope_does_not_blindly_scan_runtime(public_fixture,tamper):
    root=public_fixture
    if tamper=='private_json_path':
        file=root/'related_results/qwen3_4b_fixed_transfer_01.public.json';data=a.read(file);data['native_identity']['runtime_config']='configs/CPU_PRIVATE.json';put(file,data)
    else:
        file=root/'README.md';file.write_text(file.read_text()+ ('\n[private](../../not-shipped.md)\n' if tamper=='dead_relative_link' else '\n[private](https://github.com/RenormalizationTech/tuneLLM)\n'))
    rehash(root)
    with pytest.raises(ValueError):a.preflight(root,public=True)
    # Existing exact-byte attribution paths in reviewed runtime remain allowed.
    assert b'outputs/' in (root/'runtime/gcad.py').read_bytes()


def test_public_closeout_publication_scope_and_expected_hash_wording(public_fixture):
    root=public_fixture;closeout=(root/'related_results/qwen3_4b_fixed_transfer_closeout.md').read_text();card=(root/'README.md').read_text()
    assert 'on GitHub' not in closeout and 'aggregates in this bundle' in closeout
    assert '[plan]' not in closeout and '[interpretation]' not in closeout
    assert 'private protocol (not included)' in closeout and 'private interpretation review (not included)' in closeout
    assert '[public method notes](../METHODS.md#qwen3-4b-fixed-transfer)' in closeout
    assert 'expected manifest hash' in ' '.join(card.split()) and 'trusted bank' not in card
    assert 'This verifies file consistency for the pinned revision you choose' in card
    assert 'does not independently prove who published them' in card
