import pytest
from scripts.qwen38_stage_base import publisher_manifest,REPOS


@pytest.mark.parametrize('kind',['nf4','gguf'])
def test_payload_filenames_cannot_change_release_destination(tmp_path,kind):
    (tmp_path/'requirements-qwen38-nf4.txt').write_text('torch==2.12.0\n')
    (tmp_path/'model.safetensors').write_bytes(b'fake weight bytes')
    manifest=publisher_manifest(kind,tmp_path)
    assert manifest['repo_id']==REPOS[kind]
    assert manifest['private'] is False
    assert manifest['files']['model.safetensors']['bytes']==17


def test_unapproved_release_kind_rejected(tmp_path):
    with pytest.raises(ValueError,match='approved'):publisher_manifest('requirements-qwen38-nf4.txt',tmp_path)
