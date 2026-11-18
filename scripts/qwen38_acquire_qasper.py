"""Fetch original QASPER v0.3 archives; inspect/verify locally by default.

No dataset scripts, model code or evaluator are executed. Downloads require
--download and a new destination. Interrupted destinations remain as evidence.
"""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import tarfile
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
REVISION = "fdc9d8214fbab5dd782958601db4d678e6934a54"
HF = f"https://huggingface.co/datasets/allenai/qasper/resolve/{REVISION}"
URLS = {
    "qasper-train-dev-v0.3.tgz": "https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-train-dev-v0.3.tgz",
    "qasper-test-and-evaluator-v0.3.tgz": "https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-test-and-evaluator-v0.3.tgz",
    "README.md": f"{HF}/README.md",
    "qasper.py": f"{HF}/qasper.py",
    "dataset_infos.json": f"{HF}/dataset_infos.json",
}
EXPECTED = {"qasper-train-v0.3.json", "qasper-dev-v0.3.json", "qasper-test-v0.3.json"}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def fetch(url, target):
    partial = target.with_name(target.name + ".part")
    size = 0
    with urlopen(url, timeout=60) as response, partial.open("xb") as output:
        for block in iter(lambda: response.read(1024 * 1024), b""):
            size += len(block)
            if size > 150_000_000:
                raise ValueError("QASPER source exceeds the registered 150 MB per-file cap")
            output.write(block)
    partial.rename(target)
    return {"url": url, "bytes": size, "sha256": sha(target)}


def unpack(archive, destination):
    """Only copy regular JSON/scorer/license members; never extract archive paths."""
    copied = {}
    with tarfile.open(archive, "r:gz") as source:
        for member in source.getmembers():
            path = PurePosixPath(member.name)
            if not member.isfile() or path.is_absolute() or ".." in path.parts:
                continue
            name = path.name
            if name not in EXPECTED and not (
                name.endswith(".py") and ("eval" in name.lower() or "score" in name.lower())
            ) and not name.lower().startswith(("license", "readme")):
                continue
            if member.size > 100_000_000:
                raise ValueError("oversized QASPER archive member")
            target = destination / name
            stream = source.extractfile(member)
            if stream is None:
                raise ValueError("missing archive member contents")
            with target.open("xb") as output:
                output.write(stream.read())
            copied[name] = {"archive_member": member.name, "bytes": target.stat().st_size, "sha256": sha(target)}
    return copied


def download(destination):
    destination.mkdir(parents=True, exist_ok=False)
    raw, extracted = destination / "raw", destination / "original"
    raw.mkdir(); extracted.mkdir()
    sources = {}
    members = {}
    for name, url in URLS.items():
        sources[name] = fetch(url, raw / name)
        if name.endswith(".tgz"):
            members.update(unpack(raw / name, extracted))
    if not EXPECTED.issubset(members):
        raise ValueError("original release is missing expected train/dev/test JSON files")
    counts, seen = {}, set()
    for name in sorted(EXPECTED):
        rows = json.loads((extracted / name).read_text())
        if not isinstance(rows, dict) or not rows or seen.intersection(rows):
            raise ValueError("nonempty, paper-disjoint original splits required")
        seen.update(rows)
        question_ids = [qa["question_id"] for paper in rows.values() for qa in paper["qas"]]
        if len(question_ids) != len(set(question_ids)):
            raise ValueError("duplicate question IDs in original release")
        counts[name] = {"papers": len(rows), "questions": len(question_ids)}
    manifest = {"dataset": "allenai/qasper", "hf_revision": REVISION,
                "release": "v0.3", "license": "CC-BY-4.0",
                "source_scope": "Official archive URLs from the pinned HF loader; archives identified by captured SHA256, not immutable URL revisions",
                "source_files": sources, "extracted_files": members, "counts": counts,
                "test_use": "download/schema validation only; no model outputs, prompt selection or tuning",
                "upstream_code_executed": False, "acquisition_script_sha256": sha(Path(__file__))}
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return verify(destination)


def verify(destination):
    manifest = json.loads((destination / "manifest.json").read_text())
    if manifest["hf_revision"] != REVISION or manifest["release"] != "v0.3":
        raise ValueError("unexpected QASPER source identity")
    for folder, field in (("raw", "source_files"), ("original", "extracted_files")):
        for name, expected in manifest[field].items():
            if Path(name).name != name:
                raise ValueError("manifest names must be basenames")
            target = destination / folder / name
            if target.stat().st_size != expected["bytes"] or sha(target) != expected["sha256"]:
                raise ValueError(f"QASPER input changed: {folder}/{name}")
    return {"verified": True, "destination": str(destination.relative_to(ROOT)),
            "manifest_sha256": sha(destination / "manifest.json"), "counts": manifest["counts"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", default="data/qasper-v03-01")
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    destination = (ROOT / args.destination).resolve()
    if not destination.is_relative_to(ROOT / "data"):
        parser.error("destination must stay in the ignored project data directory")
    print(json.dumps(download(destination) if args.download else verify(destination), indent=2))


if __name__ == "__main__":
    main()
