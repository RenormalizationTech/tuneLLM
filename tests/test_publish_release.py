import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from scripts.publish_release import main


def payload(tmp_path):
    folder = tmp_path / "payload"
    folder.mkdir()
    weight = folder / "weights.safetensors"
    weight.write_bytes(b"reviewed bytes")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"repo_id": "example/test", "private": True,
        "files": {weight.name: {"bytes": weight.stat().st_size,
                  "sha256": hashlib.sha256(weight.read_bytes()).hexdigest()}}}))
    report = tmp_path / "report.json"
    argv = ["publish", "--folder", str(folder), "--manifest", str(manifest), "--output", str(report)]
    return folder, weight, manifest, report, argv


def test_changed_payload_never_contacts_hub(tmp_path):
    _, weight, _, _, argv = payload(tmp_path)
    weight.write_bytes(b"unreviewed changed bytes")
    with patch("sys.argv", argv), patch("huggingface_hub.HfApi") as api:
        with pytest.raises(ValueError, match="payload changed"):
            main()
        api.assert_not_called()


def test_private_publication_verifies_pinned_bytes_without_overwrite(tmp_path):
    _, weight, _, report, argv = payload(tmp_path)
    api = Mock()
    api.upload_folder.return_value = SimpleNamespace(oid="f" * 40, commit_url="https://example.invalid/commit")
    info = SimpleNamespace(sha="f" * 40, private=True,
        siblings=[SimpleNamespace(rfilename=weight.name, size=weight.stat().st_size, lfs=None)],
        card_data=SimpleNamespace(to_dict=lambda: {"license": "apache-2.0"}))
    api.model_info.return_value = info
    with patch("sys.argv", argv), patch("huggingface_hub.HfApi", return_value=api), \
            patch("huggingface_hub.hf_hub_download", return_value=str(weight)) as download:
        main()
    api.create_repo.assert_called_once_with("example/test", repo_type="model", private=True, exist_ok=False)
    assert download.call_args.kwargs["revision"] == "f" * 40
    result = json.loads(report.read_text())
    assert result["status"] == "published_remote_hashes_verified"
    assert result["private"] and not result["credentials_recorded"]


def test_resume_refuses_different_remote_visibility(tmp_path):
    _, _, _, _, argv = payload(tmp_path)
    api = Mock()
    api.model_info.return_value = SimpleNamespace(private=False, siblings=[])
    with patch("sys.argv", argv + ["--resume-created-repo"]), patch("huggingface_hub.HfApi", return_value=api):
        with pytest.raises(ValueError, match="visibility differs"):
            main()
    api.upload_folder.assert_not_called()
    api.create_repo.assert_not_called()


def test_manifest_cannot_escape_reviewed_payload(tmp_path):
    _, _, manifest, _, argv = payload(tmp_path)
    value = json.loads(manifest.read_text())
    value["files"]["../outside"] = {"bytes": 0, "sha256": "0" * 64}
    manifest.write_text(json.dumps(value))
    with patch("sys.argv", argv), patch("huggingface_hub.HfApi") as api:
        with pytest.raises(ValueError, match="paths must stay"):
            main()
        api.assert_not_called()
