"""Synthetic published-bundle bytes only; no network or original research inputs."""
import hashlib
from pathlib import Path
from unittest.mock import patch

import pytest

from scripts import setup_refusal_artifact as setup


def test_pinned_receipt_no_original_outputs_needed():
    assert setup.files()['direction.pt']['sha256'] == '671c6a839cd01a3767d6455b9a1b59a37059f712097ff1f8142324dfccbae28e'


def test_fake_download_is_anonymous_pinned_and_verified(tmp_path, monkeypatch):
    source = tmp_path/'cache'; source.mkdir(); (source/'runtime.py').write_bytes(b'synthetic')
    expected = {'runtime.py': {'bytes': 9, 'sha256': hashlib.sha256(b'synthetic').hexdigest()}}
    monkeypatch.setattr(setup, 'files', lambda: expected)
    def fetch(repo, **kwargs):
        assert repo == setup.REPO and kwargs['revision'] == setup.REVISION and kwargs['token'] is False
        return source/kwargs['filename']
    with patch('huggingface_hub.hf_hub_download', side_effect=fetch) as call:
        dest = setup.download(tmp_path/'bundle')
    assert call.call_count == 1 and setup.verify(dest) == dest
    (dest/'runtime.py').write_bytes(b'tampered!')
    with pytest.raises(ValueError, match='hash'): setup.verify(dest)
    with pytest.raises(FileExistsError): setup.download(dest)


def test_verify_does_not_download_missing_files(tmp_path):
    with patch('huggingface_hub.hf_hub_download', side_effect=AssertionError('no network')) as call:
        with pytest.raises(ValueError, match='missing'): setup.verify(tmp_path)
    assert not call.called
