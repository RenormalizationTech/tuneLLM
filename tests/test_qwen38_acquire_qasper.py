"""Local archive and tamper checks; no network or dataset execution."""
import io
import json
import tarfile

import pytest

from scripts.qwen38_acquire_qasper import unpack, sha, verify, REVISION


def test_archive_paths_and_unrelated_python_are_not_extracted(tmp_path):
    archive = tmp_path / "source.tgz"
    with tarfile.open(archive, "w:gz") as stream:
        for name, data in [("../qasper-train-v0.3.json", b"bad"),
                           ("release/qasper-train-v0.3.json", b"{}"),
                           ("release/setup.py", b"raise RuntimeError"),
                           ("release/qasper_evaluator.py", b"# source only")]:
            member = tarfile.TarInfo(name); member.size = len(data)
            stream.addfile(member, io.BytesIO(data))
    dest = tmp_path / "original"; dest.mkdir()
    result = unpack(archive, dest)
    assert set(result) == {"qasper-train-v0.3.json", "qasper_evaluator.py"}
    assert (dest / "qasper-train-v0.3.json").read_bytes() == b"{}"
    assert not (tmp_path / "qasper-train-v0.3.json").exists()


def test_tampered_original_file_fails_verification(tmp_path):
    (tmp_path / "original").mkdir()
    file = tmp_path / "original/train.json"; file.write_bytes(b"{}")
    manifest = {"hf_revision": REVISION, "release": "v0.3", "source_files": {},
                "extracted_files": {"train.json": {"bytes": 2, "sha256": sha(file)}}}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    file.write_bytes(b"[]")
    with pytest.raises(ValueError, match="input changed"):
        verify(tmp_path)
