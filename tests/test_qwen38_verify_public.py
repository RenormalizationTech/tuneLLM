import json
import pytest
from scripts.qwen38_verify_public import publication,REPOS,validate_artifact,validate_listing,native_replay


def record():
    return {'repo_id':REPOS['nf4'],'status':'published_remote_hashes_verified','private':False,
            'revision':'a'*40,'files':{'config.json':{'bytes':1,'sha256':'b'*64}}}


@pytest.mark.parametrize('key,value', [('repo_id',REPOS['gguf']),('private',True),('private',0),
    ('status','uploaded_pending_hash_verification'),('revision','main'),
    ('files',{'../escape':{}}),('files',{})])
def test_wrong_or_unverified_publication_rejected(tmp_path,key,value):
    r=record();r[key]=value;p=tmp_path/'receipt.json';p.write_text(json.dumps(r))
    with pytest.raises(ValueError):publication(p,'nf4')


def test_exact_public_revision_accepted(tmp_path):
    p=tmp_path/'receipt.json';p.write_text(json.dumps(record()))
    assert publication(p,'nf4')==record()


def test_published_or_adapter_base_bytes_must_match_evaluated_manifest():
    expected={'config.json':{'bytes':1,'sha256':'b'*64}}
    validate_artifact(record(),expected)
    changed=record();changed['files']['config.json']['sha256']='c'*64
    with pytest.raises(ValueError,match='evaluated bytes'):validate_artifact(changed,expected)


def test_public_listing_rejects_extra_or_missing_file():
    validate_listing({'config.json','.gitattributes'},record()['files'])
    with pytest.raises(ValueError,match='unreviewed'):validate_listing({'config.json','unexpected.safetensors'},record()['files'])
    with pytest.raises(ValueError,match='missing'):validate_listing(set(),record()['files'])


def test_native_smoke_call_preserves_generation_config_stopping():
    class Model:
        def generate(self,**kwargs):self.kwargs=kwargs;return 'tokens'
    model=Model();inputs={'input_ids':[1],'attention_mask':[1]}
    assert native_replay(model,inputs)=='tokens'
    assert model.kwargs==dict(inputs,do_sample=False,num_beams=1,max_new_tokens=64,
        use_cache=True,temperature=None,top_p=None,top_k=None,repetition_penalty=1.0)
