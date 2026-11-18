import pytest
import argparse
import json
import sys
import types
from scripts.qwen38_publish_verification import evidence,validate_update_plan
from scripts.qwen38_verify_public import REPOS


def records(kind='nf4'):
    p={'repo_id':REPOS[kind],'revision':'a'*40}
    d={'kind':kind,'revision':'a'*40,'completed':True,'anonymous_access_verified':True,'all_file_hashes_verified':True}
    r={'kind':kind,'completed':True,'exact_smoke_token_replay':True,'adapter_tensor_equality':True,'public_route':'PeftModel.from_pretrained'}
    return p,d,r


@pytest.mark.parametrize('kind',list(REPOS))
def test_metadata_cites_actual_gpu_revision_not_a_new_evaluation(kind):
    result=evidence(kind,*records(kind))
    assert result['gpu_verified_revision']=='a'*40
    assert 'not another quality benchmark' in result['scope']
    assert 'failed' in result['known_limit']


@pytest.mark.parametrize('key',['completed','anonymous_access_verified','all_file_hashes_verified'])
def test_incomplete_download_cannot_be_documented_as_verified(key):
    p,d,r=records();d[key]=False
    with pytest.raises(ValueError):evidence('nf4',p,d,r)


def test_wrong_revision_or_missing_public_adapter_route_rejected():
    p,d,r=records();d['revision']='b'*40
    with pytest.raises(ValueError):evidence('nf4',p,d,r)
    p,d,r=records('prl');r['public_route']=None
    with pytest.raises(ValueError):evidence('prl',p,d,r)


@pytest.mark.parametrize('mutation',['runtime_operation','forged_parent','forged_runtime_hash'])
def test_publish_rejects_mutated_plan_before_any_hub_call(tmp_path,monkeypatch,mutation):
    import scripts.qwen38_publish_verification as module
    monkeypatch.setattr(module,'ROOT',tmp_path)
    payload=tmp_path/'artifacts/releases'/REPOS['nf4'].split('/')[1];payload.mkdir(parents=True)
    (payload/'README.md').write_text('# Model\n');(payload/'model.safetensors').write_bytes(b'weights')
    pub,d,r=records()
    pub.update(status='published_remote_hashes_verified',private=False,
        files={n:{'bytes':(payload/n).stat().st_size,'sha256':module.sha(payload/n)} for n in ['README.md','model.safetensors']})
    publication=tmp_path/'publication.json';publication.write_text(json.dumps(pub))
    verified=tmp_path/'outputs/verified';verified.mkdir(parents=True)
    d.update(publication_sha256=module.sha(publication),files=pub['files'])
    (verified/'download.json').write_text(json.dumps(d))
    r['download_sha256']=module.sha(verified/'download.json')
    (verified/'result.json').write_text(json.dumps(r))
    folder=tmp_path/'outputs/metadata'
    module.prepare(argparse.Namespace(output=folder,publication=publication,verification=verified,kind='nf4'))
    cfg=json.loads((folder/'prepared.json').read_text())
    if mutation=='runtime_operation':cfg['changed_files']['model.safetensors']='f'*64
    elif mutation=='forged_parent':cfg['parent_revision']='b'*40
    else:cfg['expected_files']['model.safetensors']['sha256']='f'*64
    (folder/'prepared.json').write_text(json.dumps(cfg))
    def forbidden():raise AssertionError('Hub client must not be created for invalid metadata')
    monkeypatch.setitem(sys.modules,'huggingface_hub',types.SimpleNamespace(HfApi=forbidden,CommitOperationAdd=forbidden))
    with pytest.raises(ValueError):module.publish(folder)
