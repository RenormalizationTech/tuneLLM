"""CPU tiny-model parity, local-only validation and scratch release staging."""
import ast
import hashlib
import json
from pathlib import Path
import shutil
import threading

import pytest
import torch
from scripts import qwen38_gcad_port as frozen
from scripts.qwen38_sarcasm_release_runtime import gcad as portable
from scripts.qwen38_sarcasm_release_runtime import loader
from scripts import stage_qwen38_sarcasm_release as stage
from scripts import verify_qwen38_sarcasm_release as verifier
from tests.test_qwen38_gcad_port import tiny,tensors


@pytest.fixture(autouse=True)
def threads():
    old=torch.get_num_threads();torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def test_port_copy_byte_identical_and_no_repository_imports():
    assert stage.sha(Path(portable.__file__))==stage.PORT==stage.sha(Path(frozen.__file__))
    for file in stage.RUNTIME.glob('*.py'):
        tree=ast.parse(file.read_text())
        for node in ast.walk(tree):
            if isinstance(node,ast.Import):assert all(not x.name.startswith('scripts') for x in node.names)
            if isinstance(node,ast.ImportFrom):assert not (node.module or '').startswith('scripts')


@pytest.mark.parametrize('b',[0.,.5,2.,8.,16.])
def test_portable_equal_to_frozen_prefill_cached_decode_and_cleanup(b):
    model=tiny();bank=tensors(model);ids=torch.tensor([[3,4,5]])
    from scripts.qwen38_sarcasm_followup_telemetry import signature
    before=signature(model);kwargs=dict(do_sample=False,use_cache=True,max_new_tokens=4,eos_token_id=None,pad_token_id=0,
                                       return_dict_in_generate=True,output_logits=True)
    with torch.no_grad():
        with frozen.Steerer(model,bank,[1],b):a=model.generate(ids,**kwargs)
        with portable.Steerer(model,bank,[1],b):z=model.generate(ids,**kwargs)
    assert torch.equal(a.sequences,z.sequences)
    assert all(torch.equal(x,y) for x,y in zip(a.logits,z.logits))
    assert signature(model)==before
    with pytest.raises(RuntimeError):
        with portable.Steerer(model,bank,[1],b):raise RuntimeError('CPU exception')
    assert signature(model)==before


@pytest.mark.parametrize('value',[True,False,-.1,16.1,float('nan'),float('inf'),-float('inf'),'8',None])
def test_invalid_strengths_refused(value):
    with pytest.raises(ValueError):loader.validate_strength(value)


def test_zero_bypasses_hooks_nested_concurrent_strength_changes_refused():
    model=tiny();runtime=loader.Runtime(model,None,{},Path('.'),{})
    ids=torch.tensor([[3,4,5]]);mask=torch.ones_like(ids)
    from scripts.qwen38_sarcasm_followup_telemetry import signature
    before=signature(model)
    with torch.no_grad():ordinary=model(ids,attention_mask=mask,use_cache=True,logits_to_keep=1).logits
    assert torch.equal(ordinary,runtime.forward(ids,mask,0).logits)
    with runtime.steering(ids,mask,0):
        assert signature(model)==before
        with pytest.raises(RuntimeError,match='nested/concurrent'):
            with runtime.steering(ids,mask,8):pass
        result=[]
        def other():
            try:
                with runtime.steering(ids,mask,0):pass
            except RuntimeError:result.append('refused')
        thread=threading.Thread(target=other);thread.start();thread.join()
        assert result==['refused']
    assert signature(model)==before
    with pytest.raises(AttributeError):runtime.strength=16


@pytest.mark.parametrize('ids,mask',[
    (torch.ones(2,3,dtype=torch.long),torch.ones(2,3)),
    (torch.ones(1,3,dtype=torch.long),torch.tensor([[0,1,1]])),
    (torch.ones(1,3,dtype=torch.long),torch.ones(1,2)),
    (torch.ones(1,3),torch.ones(1,3)),
    (torch.ones(1,1,dtype=torch.long),torch.ones(1,1)),
])
def test_batch_padding_and_input_geometry_refusal(ids,mask):
    with pytest.raises(ValueError):loader.validate_inputs(ids,mask)


def test_bank_keys_dtype_shape_and_finite_refusal():
    from safetensors.torch import load_file
    bank=load_file(str(stage.ROOT/'outputs/qwen38-sarcasm-gcad-01/extraction/bank.safetensors'))
    loader.validate_bank(bank)
    for invalid in [{**bank,'23.delta':torch.zeros(1)},
                    {**bank,'23.delta':bank['23.delta'].double()},
                    {k:v for k,v in bank.items() if k!='23.delta'},
                    {**bank,'23.demand':torch.tensor(float('nan'))}]:
        with pytest.raises(ValueError):loader.validate_bank(invalid)
    with pytest.raises(ValueError,match='geometry'):loader.validate_model(tiny())


@pytest.fixture(scope='module')
def staged(tmp_path_factory):
    root=tmp_path_factory.mktemp('release-cpu')/'fresh'
    record=stage.stage(root)
    return root,Path(record['payload']),record['manifest_sha256']


def test_scratch_staging_inventory_exact_bank_and_actual_environment(staged):
    root,payload,mf=staged
    manifest=loader.verify_bundle(payload,mf)
    loader.validate_environment(manifest)
    assert manifest['environment']['packages']['torch']==stage.importlib.metadata.version('torch')
    assert stage.sha(payload/'runtime/gcad.py')==stage.PORT
    assert stage.sha(payload/'bank.safetensors')==stage.BANK
    assert set(manifest['base']['files'])==set(stage.BASE_FILES)
    assert 'smoke_origin.json' not in manifest['base']['files']
    assert all(not p.is_symlink() for p in payload.rglob('*'))
    publication=json.loads((root/'publication_manifest.json').read_text())
    assert publication['repo_id']==stage.REPO and publication['private'] is False
    assert set(publication['files'])=={p.relative_to(payload).as_posix() for p in payload.rglob('*') if p.is_file()}


def test_public_aggregate_and_card_match_canonical_without_private_rows(staged):
    _,payload,_=staged;canonical=stage.read('results/qwen38_sarcasm_followup_01.json')
    exported=json.loads((payload/'evaluation.json').read_text())
    assert exported['part_B']['metrics']==canonical['part_B']['metrics']
    assert exported['part_B']['gates']==canonical['part_B']['gates']
    assert exported['outcome']=='exploratory_null_through_b16'
    assert 'paired_per_question' not in json.dumps(exported)
    assert not {'input_sha256','ledger','examples'}&set(exported)
    card=(payload/'README.md').read_text()
    assert stage.table(canonical) in card
    assert 'snapshot_download("'+stage.REPO+'"' in card
    assert 'Functional package verification is recorded separately' in card
    assert 'local_dir="../qwen38-base"' in card
    stage.privacy_scan(payload)
    for metrics in canonical['part_B']['metrics'].values():assert metrics['caps']==32


@pytest.mark.parametrize('mutation',['extra','missing','changed','bad_manifest','bank'])
def test_consumer_inventory_fails_closed(staged,tmp_path,mutation):
    _,source,mf=staged;payload=tmp_path/'copy';shutil.copytree(source,payload)
    if mutation=='extra':(payload/'extra.txt').write_text('unexpected')
    elif mutation=='missing':(payload/'METHODS.md').unlink()
    elif mutation=='changed':(payload/'runtime/example.py').write_text('modified')
    elif mutation=='bank':(payload/'bank.safetensors').write_bytes(b'bad bank')
    else:mf='0'*64
    with pytest.raises(ValueError):loader.verify_bundle(payload,mf)


def test_consumer_generated_python_metadata_is_allowed(staged,tmp_path):
    _,source,mf=staged;payload=tmp_path/'copy';shutil.copytree(source,payload)
    cache=payload/'runtime/__pycache__';cache.mkdir();(cache/'loader.cpython-313.pyc').write_bytes(b'consumer metadata')
    loader.verify_bundle(payload,mf)


@pytest.mark.parametrize('name',['../outside','/tmp/outside','a/../b','a//b','a/./b','C:\\bad',''])
def test_manifest_paths_refused(tmp_path,name):
    with pytest.raises(ValueError):loader.safe_file(tmp_path,name)


@pytest.mark.parametrize('hash_length',[40,64])
def test_safe_HF_cache_symlinks_and_rejected_other_links(tmp_path,hash_length):
    repo=tmp_path/'models--example';root=repo/'snapshots'/('a'*40);root.mkdir(parents=True)
    blobs=repo/'blobs';blobs.mkdir();blob=blobs/('b'*hash_length);blob.write_bytes(b'public immutable content')
    (root/'model.txt').symlink_to(blob)
    assert loader.safe_file(root,'model.txt').read_bytes()==blob.read_bytes()
    other=tmp_path/'other';other.write_text('private')
    (root/'escape').symlink_to(other)
    with pytest.raises(ValueError):loader.safe_file(root,'escape')
    (root/'directory').symlink_to(tmp_path,target_is_directory=True)
    with pytest.raises(ValueError):loader.safe_file(root,'directory/other')


@pytest.fixture
def shared_cache(tmp_path):
    from huggingface_hub.utils._shared_blobs import shared_blob_path
    cache=tmp_path/'hub';repo=cache/'models--RenormalizationTech--base'
    root=repo/'snapshots'/('a'*40);root.mkdir(parents=True)
    repo_blobs=repo/'blobs';repo_blobs.mkdir()
    content=b'Exact public runtime bytes; Xet and content hashes are distinct.'
    content_hash=hashlib.sha256(content).hexdigest();xet_hash='5f'+'6'*62
    final=shared_blob_path(cache,xet_hash);final.parent.mkdir(parents=True)
    marker=cache/'blobs/.huggingface-shared-blobs';marker.write_text('1\n')
    final.write_bytes(content)
    first=repo_blobs/content_hash
    first.symlink_to(Path('../../blobs')/xet_hash[:2]/xet_hash)
    (root/'model.safetensors').symlink_to(Path('../../blobs')/content_hash)
    return {'cache':cache,'repo':repo,'root':root,'first':first,'final':final,'marker':marker,
            'content':content,'content_hash':content_hash,'xet_hash':xet_hash}


def test_actual_Hub24_two_hop_layout_and_content_hash(shared_cache):
    c=shared_cache
    assert loader.safe_file(c['root'],'model.safetensors').read_bytes()==c['content']
    assert c['content_hash']!=c['xet_hash']
    loader.check_file(c['root'],'model.safetensors',{'bytes':len(c['content']),'sha256':c['content_hash']})
    c['final'].write_bytes(b'X'*len(c['content']))
    with pytest.raises(ValueError,match='bytes/hash'):
        loader.check_file(c['root'],'model.safetensors',{'bytes':len(c['content']),'sha256':c['content_hash']})


@pytest.mark.parametrize('mode',['wrong_prefix','missing_marker','bad_marker','marker_link','third_link',
                               'outside','bad_first_name','bad_final_name','prefix_dirlink','store_dirlink'])
def test_shared_cache_escape_marker_name_and_third_link_refusals(shared_cache,tmp_path,mode):
    c=shared_cache
    if mode=='missing_marker':c['marker'].unlink()
    elif mode=='bad_marker':c['marker'].write_text('2\n')
    elif mode=='marker_link':
        c['marker'].unlink();other=tmp_path/'marker';other.write_text('1\n');c['marker'].symlink_to(other)
    elif mode=='third_link':
        c['final'].unlink();other=c['cache']/'blobs'/'ff'/('f'*64)
        other.parent.mkdir();other.write_bytes(c['content']);c['final'].symlink_to(other)
    elif mode in ['outside','wrong_prefix','bad_final_name']:
        target=(tmp_path/'outside'/c['xet_hash'] if mode=='outside' else
                c['cache']/'blobs'/'aa'/c['xet_hash'] if mode=='wrong_prefix' else
                c['final'].parent/'NOT-A-64-HEX-HASH')
        target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(c['content'])
        c['first'].unlink();c['first'].symlink_to(target)
    elif mode=='bad_first_name':
        first=c['first'].with_name('NOT-A-CONTENT-HASH');c['first'].rename(first)
        f=c['root']/'model.safetensors';f.unlink();f.symlink_to(first)
    elif mode=='prefix_dirlink':
        parent=c['final'].parent;other=tmp_path/'prefix-copy';parent.rename(other);parent.symlink_to(other,target_is_directory=True)
    else:
        store=c['cache']/'blobs';other=tmp_path/'store-copy';store.rename(other);store.symlink_to(other,target_is_directory=True)
    with pytest.raises(ValueError):loader.safe_file(c['root'],'model.safetensors')


def test_regular_local_files_still_pass_without_cache_metadata(tmp_path):
    (tmp_path/'model.txt').write_bytes(b'local files need no HF marker')
    assert loader.safe_file(tmp_path,'model.txt')==tmp_path/'model.txt'


def test_changed_base_refused_before_any_model_or_download_call(staged,tmp_path,monkeypatch):
    import transformers,huggingface_hub
    _,payload,mf=staged;calls=[]
    for name in stage.BASE_FILES:(tmp_path/name).write_bytes(b'changed')
    monkeypatch.setattr(transformers.Qwen3_5ForCausalLM,'from_pretrained',lambda *a,**k:calls.append('model'))
    monkeypatch.setattr(huggingface_hub,'snapshot_download',lambda *a,**k:calls.append('download'))
    with pytest.raises(ValueError,match='bytes/hash'):loader.load_runtime(tmp_path,mf,bundle=payload)
    assert calls==[]


def test_publication_privacy_scan_and_verifier_aggregate_anchor(tmp_path):
    (tmp_path/'leak.json').write_text('{"session_id":"private"}')
    with pytest.raises(ValueError):stage.privacy_scan(tmp_path)
    canonical=stage.read('results/qwen38_sarcasm_followup_01.json')
    assert verifier.CANONICAL_SHA==stage.CANONICAL
    assert verifier.AGGREGATE_SHA==canonical['input_sha256']['generation/result.json']
    assert stage.sha(stage.ROOT/'outputs/qwen38-sarcasm-followup-01/generation/result.json')==verifier.AGGREGATE_SHA
