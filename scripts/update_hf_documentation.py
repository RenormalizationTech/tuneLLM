"""Validate staged documentation updates; remote writes require explicit --apply.

Only README.md and a NEW benchmark_summary.json may change. No model, config,
tokenizer, historical report, manifest, visibility or dependency changes.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re

from scripts.common import path, read_json, write_json

PREFIX = "RenormalizationTech/Qwen3-4B-tuneLLM-"
VISIBILITY = {PREFIX + name: private for name, private in
              (("GGUF", False), ("NF4", True), ("GSM8K-QLoRA", True), ("PRL-Writing-QLoRA", True))}
ALLOW = {"README.md", "benchmark_summary.json"}
BASE_PIN = {"base_model_name_or_path": PREFIX + "NF4",
            "revision": "e9af208e44e083b77702a71d8aa65b7d97f78ce6"}
DEPENDENCIES = {PREFIX + "PRL-Writing-QLoRA": BASE_PIN,
                PREFIX + "GSM8K-QLoRA": {"base_model_name_or_path": "Qwen/Qwen3-4B",
                                           "revision": "1cfa9a7208912126459214e8b04321603b3df60c"}}


def fingerprint(data):
    return {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def safe_text(data):
    if re.search(rb"hf_[A-Za-z0-9]{20,}|/(?:home|Users)/|artifacts/|outputs/|data/physics|-----BEGIN .*PRIVATE KEY", data):
        raise ValueError("documentation contains a local/private path or credential-like text")


def validate_stage(manifest, directory):
    repo = manifest["repo_id"]
    if (repo not in VISIBILITY or type(manifest.get("private")) is not bool
            or manifest["private"] != VISIBILITY[repo]
            or not re.fullmatch(r"[0-9a-f]{40}", manifest.get("parent_commit", ""))
            or manifest.get("allowed_changes") != sorted(ALLOW)
            or set(manifest.get("updates", {})) != ALLOW):
        raise ValueError("invalid documentation-only allowlist, repository, visibility or parent")
    expected = manifest["expected_inventory"]
    if "README.md" not in expected or "benchmark_summary.json" in expected:
        raise ValueError("require existing README and additive new benchmark report")
    if any(Path(n).is_absolute() or ".." in Path(n).parts or "\\" in n for n in expected):
        raise ValueError("unsafe inventory path")
    if any(not re.fullmatch(r"[0-9a-f]{64}", f.get("sha256", ""))
           for n, f in expected.items() if n != ".gitattributes"):
        raise ValueError("published inventory hashes required")
    updates = {}
    for name in sorted(ALLOW):
        source = directory / name
        if source.is_symlink() or not source.is_file():
            raise ValueError("staged documentation must be ordinary files")
        data = source.read_bytes()
        if fingerprint(data) != manifest["updates"][name]:
            raise ValueError("staged documentation changed after review")
        safe_text(data)
        updates[name] = data
    report = json.loads(updates["benchmark_summary.json"])
    if repo == PREFIX + "GGUF":
        if set(report["conditions"]) != {"bf16_gguf", "q8", "q4"}:
            raise ValueError("public GGUF report must contain only GGUF conditions")
        if re.search(rb"nf4|paper_text|writing|physics_adapter|paper_attributions", updates["benchmark_summary.json"], re.I):
            raise ValueError("private experiment metadata in public GGUF report")
    if repo in DEPENDENCIES and manifest.get("pinned_dependency") != DEPENDENCIES[repo]:
        raise ValueError("published adapter dependency pin must remain fixed")
    return updates


def check_before(manifest, head, private, inventory):
    if head != manifest["parent_commit"] or private != manifest["private"]:
        raise ValueError("remote head or visibility changed; restage for review")
    if set(inventory) != set(manifest["expected_inventory"]):
        raise ValueError("remote inventory changed; restage for review")
    for name, expected in manifest["expected_inventory"].items():
        for key in ("sha256", "bytes"):
            if expected.get(key) is not None and inventory[name][key] != expected[key]:
                raise ValueError("remote published bytes differ from reviewed inventory")


def check_after(before, after, updates):
    if set(after) != set(before) | {"benchmark_summary.json"}:
        raise ValueError("post-update inventory changed outside the allowlist")
    for name in after:
        expected = fingerprint(updates[name]) if name in updates else before[name]
        if after[name] != expected:
            raise ValueError("post-update bytes changed outside reviewed documentation")


def execute(manifest, directory, remote, apply=False):
    updates = validate_stage(manifest, directory)
    head, private, before = remote.inventory()
    check_before(manifest, head, private, before)
    if manifest.get("pinned_dependency"):
        if any(remote.json_file("adapter_config.json", head).get(k) != v
               for k, v in manifest["pinned_dependency"].items()):
            raise ValueError("remote adapter dependency pin differs")
    record = {"repo_id": manifest["repo_id"], "private": private, "parent_commit": head,
              "pre_inventory": before, "applied": False, "credentials_recorded": False}
    if not apply:
        return record
    # Remote implementation uses CommitOperationAdd only and parent_commit CAS.
    revision = remote.commit(updates, parent_commit=head)
    record.update({"applied": True, "revision": revision, "post_verified": False})
    try:
        new_head, new_private, after = remote.inventory()
        if new_head != revision or new_private != private:
            raise ValueError("post-update head or visibility differs")
        check_after(before, after, updates)
        record.update({"post_inventory": after, "post_verified": True})
        return record
    except Exception:
        # Preserve the returned commit ID if a later network/verification step fails.
        raise PostVerificationError(record) from None


class PostVerificationError(RuntimeError):
    def __init__(self, record):
        self.record = record
        super().__init__("documentation committed but post-verification failed; inspect audit before retry")


class HubRemote:
    def __init__(self, repo):
        # Explicit project-only credentials: never use ambient HF_TOKEN/default home.
        os.environ["HF_HOME"] = str(path(".hf"))
        os.environ["HF_HUB_OFFLINE"] = "0"
        os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
        from huggingface_hub import HfApi, hf_hub_download
        self.token = path(".hf/token").read_text().strip()
        if not self.token:
            raise ValueError("project HF authentication is missing")
        self.repo, self.api, self.download = repo, HfApi(token=self.token), hf_hub_download

    def file(self, name, revision):
        return Path(self.download(self.repo, name, revision=revision, token=self.token,
                    cache_dir=str(path(".hf/report-update-verification/hub"))))

    def json_file(self, name, revision):
        return json.loads(self.file(name, revision).read_text())

    def inventory(self):
        from scripts.save_nf4 import stream_hash
        info = self.api.model_info(self.repo, files_metadata=True)
        inventory = {}
        for sibling in info.siblings:
            if sibling.lfs:
                sha = sibling.lfs.sha256
            else:
                sha = stream_hash(self.file(sibling.rfilename, info.sha))
            inventory[sibling.rfilename] = {"bytes": sibling.size, "sha256": sha}
        return info.sha, bool(info.private), inventory

    def commit(self, updates, parent_commit):
        from huggingface_hub import CommitOperationAdd
        return self.api.create_commit(self.repo, repo_type="model", revision="main",
            operations=[CommitOperationAdd(path_in_repo=n, path_or_fileobj=b) for n, b in sorted(updates.items())],
            parent_commit=parent_commit, commit_message="Add benchmark summary and update documentation").oid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check-remote", action="store_true", help="read-only remote inventory preflight")
    mode.add_argument("--apply", action="store_true", help="explicit remote documentation-only commit")
    parser.add_argument("--audit", help="new local audit JSON required for remote checks/apply")
    args = parser.parse_args()
    filename = path(args.manifest)
    manifest = read_json(filename)
    validate_stage(manifest, filename.parent)
    if not (args.check_remote or args.apply):
        print("Local staging validated; no remote access or update.")
        return
    if not args.audit or path(args.audit).exists():
        raise ValueError("remote mode requires a fresh --audit path")
    try:
        record = execute(manifest, filename.parent, HubRemote(manifest["repo_id"]), apply=args.apply)
    except PostVerificationError as exc:
        write_json(args.audit, exc.record)
        raise
    write_json(args.audit, {**record, "update_manifest_sha256": hashlib.sha256(filename.read_bytes()).hexdigest()})
    print("Documentation update verified." if args.apply else "Remote preflight passed; no update.")


if __name__ == "__main__":
    main()
