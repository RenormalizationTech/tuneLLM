"""Download or verify the small published refusal bundle; no base weights/models.

Network is opt-in (--download). No training/reconstruction, original experiment
mutation, authentication, upload, or automatic execution of downloaded Python.
"""
import argparse
import hashlib
from pathlib import Path, PurePosixPath
import shutil

from scripts.common import path, read_json

REPO = "RenormalizationTech/qwen2.5-7b-refusal-steering-experiment"
REVISION = "3e8a151667a8fef0ebba552a3f3d74317b1c093d"
RECEIPT = "results/refusal_7b_huggingface_publication_01.json"
DEST = "artifacts/public/refusal-7b-experiment"


def files():
    receipt = read_json(RECEIPT)
    if receipt['repo_id'] != REPO or receipt['revision'] != REVISION or receipt['private'] is not False:
        raise ValueError('pinned public publication receipt differs')
    for name in receipt['files']:
        p = PurePosixPath(name)
        if p.is_absolute() or '..' in p.parts or '\\' in name:
            raise ValueError('unsafe receipt filename')
    return receipt['files']


def verify(folder):
    folder = Path(folder)
    for name, expected in files().items():
        file = folder/name
        if not file.is_file() or file.is_symlink(): raise ValueError('missing/unsafe artifact: '+name)
        if file.stat().st_size != expected['bytes'] or hashlib.sha256(file.read_bytes()).hexdigest() != expected['sha256']:
            raise ValueError('published artifact hash/size differs: '+name)
    return folder


def download(folder):
    folder = Path(folder)
    if folder.exists(): raise FileExistsError('fresh public bundle destination required; use verify for existing files')
    expected = files()
    from huggingface_hub import hf_hub_download
    folder.mkdir(parents=True)
    for name in expected:
        cached = hf_hub_download(REPO, filename=name, revision=REVISION, token=False,
                                cache_dir=path('.hf/public-refusal-bundle'))
        target = folder/name; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(cached, target)
    return verify(folder)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination', default=DEST)
    parser.add_argument('--download', action='store_true')
    args = parser.parse_args(); dest = path(args.destination).resolve()
    allowed = path('artifacts/public').resolve()
    if dest == allowed or not dest.is_relative_to(allowed): raise ValueError('destination must stay inside ignored artifacts/public/')
    result = download(dest) if args.download else verify(dest)
    print(f'Verified {len(files())} files: {REPO}@{REVISION}\n{result}')


if __name__ == '__main__': main()
