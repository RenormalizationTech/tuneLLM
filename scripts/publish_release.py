"""Publish a reviewed explicit-file manifest and verify the pinned remote revision.

This command performs an external publication. The caller must already have
authorization for the repository and visibility recorded in the manifest.
Credentials are read from the ignored project HF cache and are never reported.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path, PurePosixPath

from scripts.common import path, read_json, write_json


def sha256(filename):
    value = hashlib.sha256()
    with Path(filename).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--resume-created-repo", action="store_true")
    parser.add_argument("--message", default="Publish reviewed tuneLLM model artifact")
    args = parser.parse_args()
    folder, output = path(args.folder).resolve(), path(args.output)
    manifest = read_json(args.manifest)
    repo, private = manifest["repo_id"], manifest["private"]
    if type(private) is not bool or len(repo.split("/")) != 2:
        raise ValueError("explicit repository owner/name and boolean private visibility required")
    for name, expected in manifest["files"].items():
        relative = PurePosixPath(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("manifest paths must stay inside reviewed payload")
        filename = folder / name
        if filename.stat().st_size != expected["bytes"] or sha256(filename) != expected["sha256"]:
            raise ValueError(f"payload changed after review: {name}")
    os.environ["HF_HOME"] = str(path(".hf").resolve())
    from huggingface_hub import HfApi, hf_hub_download
    api = HfApi()
    record = {"repo_id": repo, "url": "https://huggingface.co/" + repo, "private": private,
              "manifest_sha256": sha256(path(args.manifest)), "credentials_recorded": False}
    if output.exists():
        old = read_json(output)
        if (not args.resume_created_repo or old["repo_id"] != repo or old["private"] != private
                or old["manifest_sha256"] != record["manifest_sha256"]):
            raise ValueError("refusing to overwrite publication record without an exact resume match")
    if args.resume_created_repo:
        info = api.model_info(repo, files_metadata=True)
        if bool(info.private) != private:
            raise ValueError("existing repository visibility differs from reviewed manifest")
        names = {s.rfilename for s in info.siblings}
        if names - {".gitattributes"}:
            if names - {".gitattributes"} != set(manifest["files"]):
                raise ValueError("populated repository has different files; do not overwrite")
            # An upload may have completed before a local interruption. Verify it below.
            revision, commit_url = info.sha, "https://huggingface.co/" + repo + "/commit/" + info.sha
        else:
            revision = None
    else:
        api.create_repo(repo, repo_type="model", private=private, exist_ok=False)
        revision = None
    write_json(output, {**record, "status": "repository_created_or_resumed"})
    if revision is None:
        commit = api.upload_folder(repo_id=repo, repo_type="model", folder_path=str(folder),
            allow_patterns=list(manifest["files"]), commit_message=args.message)
        revision, commit_url = commit.oid, commit.commit_url
    write_json(output, {**record, "status": "uploaded_pending_hash_verification", "revision": revision,
                        "commit_url": commit_url})
    info = api.model_info(repo, revision=revision, files_metadata=True)
    if info.sha != revision or bool(info.private) != private:
        raise ValueError("published revision or visibility differs")
    remote = {s.rfilename: s for s in info.siblings}
    verified = {}
    for name, expected in manifest["files"].items():
        sibling = remote[name]
        if sibling.lfs:
            actual, method = sibling.lfs.sha256, "revision-pinned Hub LFS SHA256"
        else:
            downloaded = hf_hub_download(repo, name, revision=revision, cache_dir=str(path(".hf/published-verification/hub")))
            actual, method = sha256(downloaded), "revision-pinned downloaded SHA256"
        if actual != expected["sha256"] or sibling.size != expected["bytes"]:
            raise ValueError(f"remote file differs: {name}")
        verified[name] = {**expected, "method": method}
    write_json(output, {**record, "status": "published_remote_hashes_verified", "revision": revision,
        "commit_url": commit_url, "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "files": verified, "additional_remote_files": sorted(set(remote) - set(verified)),
        "card_metadata": info.card_data.to_dict(), "load_check": "separate pinned-download runtime verification required"})
    print(repo, revision, "visibility and remote hashes verified", flush=True)


if __name__ == "__main__":
    main()
