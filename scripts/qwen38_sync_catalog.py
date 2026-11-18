"""Sync ONE completed Qwen3.8 metadata publication into the local release index.

CPU/filesystem only. Default validates and previews; --apply writes a compact
receipt, exact card mirror and catalog entry. Never publishes, loads a model,
rewrites raw receipts, or edits the visible index/historical card-polish receipt.
Conflicting existing entries/mirrors/receipts fail; identical reruns are no-ops.
"""
import argparse
import copy
from datetime import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import tempfile

from scripts import qwen38_publish_verification as metadata
from scripts.qwen38_verify_public import REPOS, publication
from scripts.qwen38_science_runtime import sha
from scripts.stage_physics_release import json_bytes, private_preflight

ROOT = Path(__file__).resolve().parents[1]
VERSION = 'qwen38-sync-catalog-v1'
LABELS = {'nf4': '27B NF4 base', 'gguf': '27B GGUF', 'qasper': '27B QASPER QLoRA', 'prl': '27B PRL abstract QLoRA'}
INVENTORY_RECIPE = {
    'payload': 'Map of each published relative path to {"bytes": integer_size, "sha256": lowercase_file_sha256}; exclude local cache/tool extras and mtimes.',
    'serialization': 'json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")',
    'digest': 'hashlib.sha256(serialized_bytes).hexdigest(); no trailing newline',
}


def load(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result: raise ValueError('duplicate JSON key')
            result[key] = value
        return result
    def invalid(value): raise ValueError('nonfinite JSON constant: ' + value)
    return json.loads(Path(path).read_text(), object_pairs_hook=unique, parse_constant=invalid)


def checked(path, expected):
    if not isinstance(expected, str) or not re.fullmatch('[0-9a-f]{64}', expected) or sha(path) != expected:
        raise ValueError('bound file hash differs: ' + Path(path).name)


def revision(value):
    if not isinstance(value, str) or not re.fullmatch('[0-9a-f]{40}', value):
        raise ValueError('immutable lowercase commit revision required')
    return value


def inventory(files):
    if not isinstance(files, dict) or not files: raise ValueError('nonempty full inventory required')
    result = {}
    for name, info in files.items():
        path = PurePosixPath(name)
        if (not name or path.is_absolute() or '..' in path.parts or name != path.as_posix()
                or '\\' in name or ':' in name or name == '.'):
            raise ValueError('unsafe inventory path')
        if (type(info.get('bytes')) is not int or info['bytes'] < 0
                or not isinstance(info.get('sha256'), str) or not re.fullmatch('[0-9a-f]{64}', info['sha256'])):
            raise ValueError('invalid inventory size/hash')
        result[name] = {'bytes': info['bytes'], 'sha256': info['sha256']}
    return result


def inventory_digest(files):
    payload = inventory(files)
    raw = json.dumps(payload, sort_keys=True, separators=(',', ':'), ensure_ascii=True).encode('utf-8')
    return hashlib.sha256(raw).hexdigest()


def local(root, value, prefix=None):
    value = Path(value)
    path = (root / value).resolve()
    if not path.is_relative_to(root) or prefix and not path.is_relative_to(root / prefix):
        raise ValueError('local path outside expected project scope')
    return path


def relative(root, path): return Path(path).relative_to(root).as_posix()


def timestamp(value):
    try: parsed = datetime.fromisoformat(value)
    except (ValueError, TypeError): raise ValueError('valid timezone-aware verification time required') from None
    if parsed.tzinfo is None: raise ValueError('timezone-aware verification time required')
    return parsed


def validated_release(root, kind, directory, result_path):
    """Replay the metadata publisher's evidence guards, without any service call."""
    if kind not in REPOS: raise ValueError('unsupported Qwen3.8 kind')
    folder = local(root, directory, 'outputs')
    report_path = local(root, result_path, 'results')
    if report_path.suffix != '.json' or not report_path.is_file() or report_path.stat().st_size > 5_000_000:
        raise ValueError('existing compact JSON study report inside results required')
    if not isinstance(load(report_path), dict): raise ValueError('study report must be a JSON object')
    result = load(folder / 'result.json')
    required = {'status': 'published_and_verified', 'private': False, 'metadata_only_update': True,
                'runtime_files_unchanged': True, 'quality_or_gpu_rerun_for_documentation_update': False, 'repo_id': REPOS[kind]}
    if any(result.get(k) != value or type(result.get(k)) is not type(value) for k, value in required.items()):
        raise ValueError('completed public metadata-only verification required')
    current = revision(result['revision'])
    tested = revision(result['gpu_verified_revision'])
    if current == tested: raise ValueError('metadata commit must distinguish the original GPU-tested revision')
    if result['url'] != 'https://huggingface.co/' + REPOS[kind]: raise ValueError('wrong publication URL')
    timestamp(result['verified_at_utc'])
    checked(folder / 'prepared.json', result['prepared_sha256'])
    cfg = load(folder / 'prepared.json')
    if cfg['kind'] != kind or cfg['repo_id'] != REPOS[kind] or cfg['parent_revision'] != tested:
        raise ValueError('metadata kind/destination/tested parent differ')
    checked(metadata.__file__, cfg['source_sha256'])
    if set(cfg['changed_files']) != metadata.CHANGED_FILES:
        raise ValueError('only two metadata changes allowed')
    payload = root / 'artifacts/releases' / REPOS[kind].split('/')[1]
    if local(root, cfg['canonical_payload']) != payload or local(root, result['canonical_payload']) != payload:
        raise ValueError('canonical payload destination differs')
    pub_path = local(root, cfg['publication'], 'results')
    checked(pub_path, cfg['publication_sha256'])
    original = publication(pub_path, kind)
    if original['revision'] != tested: raise ValueError('original GPU-tested publication differs')
    inventory(original['files'])
    verification = local(root, cfg['gpu_verification_dir'], 'outputs')
    downloaded = load(verification / 'download.json')
    reloaded = load(verification / 'result.json')
    checked(verification / 'result.json', cfg['gpu_result_sha256'])
    checked(verification / 'result.json', result['gpu_result_sha256'])
    if (downloaded['publication_sha256'] != sha(pub_path) or reloaded['download_sha256'] != sha(verification / 'download.json')
            or downloaded.get('repo_id') != REPOS[kind]
            or any(downloaded.get(k) is not True for k in ('completed', 'anonymous_access_verified', 'all_file_hashes_verified'))
            or reloaded.get('completed') is not True):
        raise ValueError('original publication/download/GPU evidence binding differs')
    inventory(downloaded['files'])
    evidence = load(folder / 'verification.json')
    checked(folder / 'verification.json', result['verification_evidence_sha256'])
    timestamp(evidence['checked_at_utc'])
    reconstructed = metadata.evidence(kind, original, downloaded, reloaded)
    reconstructed.update(checked_at_utc=evidence['checked_at_utc'], original_publication_sha256=sha(pub_path),
        download_sha256=sha(verification / 'download.json'), gpu_result_sha256=sha(verification / 'result.json'),
        metadata_update_source_sha256=sha(metadata.__file__))
    if evidence != json.loads(json_bytes(reconstructed, root)):
        raise ValueError('metadata evidence differs from saved GPU check')
    previous = folder / 'README.previous.md'
    checked(previous, original['files']['README.md']['sha256'])
    if (folder / 'README.md').read_bytes() != previous.read_bytes() + metadata.verification_section(kind, tested):
        raise ValueError('README is not the exact append-only update')
    changed = {}
    for name in metadata.CHANGED_FILES:
        checked(folder / name, cfg['changed_files'][name])
        changed[name] = {'bytes': (folder / name).stat().st_size, 'sha256': sha(folder / name)}
    metadata.validate_update_plan(cfg, original, downloaded, changed)
    files = inventory(result['files'])
    if result['files'] != files or files != cfg['expected_files']:
        raise ValueError('completed full inventory differs from original runtime plus two metadata files')
    # Every published file must exist locally and match; cache/tool extras are not
    # part of the inventory and cannot accidentally enter the catalog digest.
    for name, expected in files.items():
        path = (payload / name).resolve()
        if not path.is_relative_to(payload) or not path.is_file() or path.stat().st_size != expected['bytes']:
            raise ValueError('canonical published file missing/unsafe/wrong size: ' + name)
        checked(path, expected['sha256'])
    for name in metadata.CHANGED_FILES:
        if (payload / name).read_bytes() != (folder / name).read_bytes():
            raise ValueError('canonical metadata differs from verified preparation')
    raw_card = (payload / 'README.md').read_bytes()
    private_preflight(raw_card, root)
    bound = {str(path): sha(path) for path in (folder / 'result.json', folder / 'prepared.json', folder / 'verification.json',
        folder / 'README.md', previous, verification / 'download.json', verification / 'result.json', pub_path, report_path)}
    return {'folder': folder, 'result': result, 'files': files, 'payload': payload, 'publication': pub_path,
        'study': report_path, 'card': raw_card, 'bound': bound, 'download_sha256': sha(verification / 'download.json')}


def build_plan(root, kind, directory, result_path):
    root = Path(root).resolve()
    release = validated_release(root, kind, directory, result_path)
    catalog_path = root / 'releases/catalog.json'
    if catalog_path.resolve() != catalog_path: raise ValueError('catalog path must not traverse symlinks')
    original_bytes = catalog_path.read_bytes()
    catalog = load(catalog_path)
    if catalog.get('schema_version') != 1 or catalog.get('inventory_digest_recipe') != INVENTORY_RECIPE:
        raise ValueError('catalog schema/inventory digest recipe differs')
    public, retired = catalog['public_releases'], catalog['retired_releases']
    repo_ids = [row['repo_id'] for row in public]
    if len(set(repo_ids)) != len(repo_ids) or any(row['repo_id'] == REPOS[kind] for row in retired):
        raise ValueError('duplicate/retired destination is ambiguous')
    result, files, payload = release['result'], release['files'], release['payload']
    mirror = root / 'releases' / REPOS[kind].split('/')[1] / 'README.md'
    receipt_path = root / 'results' / f'qwen38_{kind}_public_verification_01.json'
    if receipt_path in map(Path, release['bound']): raise ValueError('compact receipt would overwrite original evidence/study report')
    compact = {'version': VERSION, 'status': 'published_and_verified', 'private': False, 'repo_id': REPOS[kind],
        'revision': result['revision'], 'gpu_verified_revision': result['gpu_verified_revision'],
        'url': result['url'], 'metadata_only_update': True, 'runtime_files_unchanged': True,
        'anonymous_pinned_download_verified': True, 'quality_or_gpu_rerun_for_documentation_update': False,
        'files': files, 'published_file_count': len(files), 'published_inventory_sha256': inventory_digest(files),
        'canonical_payload': relative(root, payload), 'original_publication': relative(root, release['publication']),
        'original_publication_sha256': sha(release['publication']),
        'metadata_result': relative(root, release['folder'] / 'result.json'), 'metadata_result_sha256': sha(release['folder'] / 'result.json'),
        'prepared_sha256': result['prepared_sha256'], 'verification_evidence_sha256': result['verification_evidence_sha256'],
        'gpu_result_sha256': result['gpu_result_sha256'], 'download_sha256': release['download_sha256'],
        'result_path': relative(root, release['study']), 'study_result_sha256': sha(release['study']),
        'verified_at_utc': result['verified_at_utc'], 'sync_source_sha256': sha(__file__),
        'metadata_publisher_source_sha256': sha(metadata.__file__),
        'scope': 'Local current-publication index; original GPU-tested commit distinguished from metadata commit; no new quality or runtime evaluation.'}
    receipt_bytes = json_bytes(compact, root)
    row = {'label': LABELS[kind], 'repo_id': REPOS[kind], 'status': 'public', 'revision': result['revision'],
        'gpu_verified_revision': result['gpu_verified_revision'], 'url': result['url'],
        'card_path': relative(root, mirror), 'card_sha256': files['README.md']['sha256'],
        'local_payload_path': relative(root, payload), 'published_file_count': len(files),
        'published_inventory_sha256': inventory_digest(files), 'verification_receipt': relative(root, receipt_path),
        'result_path': relative(root, release['study']), 'publication_receipt': relative(root, release['publication']),
        'historical_metadata_paths': []}
    new_catalog = copy.deepcopy(catalog)
    existing = [r for r in public if r['repo_id'] == REPOS[kind]]
    if existing:
        if existing[0] != row: raise ValueError('existing same-repo catalog entry has different data; reviewed reconciliation required')
    else:
        new_catalog['public_releases'].append(row)
        new_catalog['checked_at_utc'] = max((catalog['checked_at_utc'], result['verified_at_utc']), key=timestamp)
    desired = {mirror: release['card'], receipt_path: receipt_bytes}
    for path, raw in desired.items():
        if path.is_symlink() or path.exists() and path.read_bytes() != raw:
            raise ValueError('conflicting existing mirror/compact receipt: ' + relative(root, path))
        if path.resolve() != path: raise ValueError('output path must not traverse symlinks')
    # Keep existing catalog bytes on idempotent runs; do not churn historical fields.
    desired[catalog_path] = original_bytes if existing else (json.dumps(new_catalog, indent=2, ensure_ascii=True, allow_nan=False) + '\n').encode()
    for name, expected in release['bound'].items(): checked(name, expected)
    return {'root': root, 'repo_id': REPOS[kind], 'entry': row, 'files': desired, 'catalog_sha256': sha(catalog_path),
        'bound': release['bound'], 'no_op': all(path.exists() and path.read_bytes() == raw for path, raw in desired.items())}


def atomic_write(path, raw):
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix='.' + path.name + '.', dir=path.parent)
    try:
        with os.fdopen(descriptor, 'wb') as stream:
            stream.write(raw); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary): os.unlink(temporary)


def sync(kind, metadata_dir, result_path, apply=False, root=ROOT):
    root = Path(root).resolve()
    def perform():
        plan = build_plan(root, kind, metadata_dir, result_path)
        changed = [relative(root, p) for p, raw in plan['files'].items() if not p.exists() or p.read_bytes() != raw]
        if apply:
            checked(root / 'releases/catalog.json', plan['catalog_sha256'])
            for name, expected in plan['bound'].items(): checked(name, expected)
            for path, raw in plan['files'].items():
                if not path.exists() or path.read_bytes() != raw: atomic_write(path, raw)
        return {'repo_id': plan['repo_id'], 'applied': bool(apply), 'no_op': plan['no_op'], 'changed_paths': changed,
            'entry': plan['entry'], 'network_accessed': False, 'GPU_started': False,
            'docs_integration': 'Parent must extend docs catalog test beyond historical five-repo polish receipt and update releases/README.md separately.'}
    if not apply: return perform()
    # Serialize additions of different new kinds without touching historical receipts.
    lock = root / 'outputs/qwen38-catalog-sync.lock'
    lock.parent.mkdir(parents=True, exist_ok=True)
    with lock.open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        return perform()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kind', choices=list(REPOS), required=True)
    parser.add_argument('--metadata-dir', required=True, help='completed qwen38_publish_verification directory in outputs/')
    parser.add_argument('--result-path', required=True, help='existing compact study report inside results/')
    parser.add_argument('--apply', action='store_true', help='write locally after all checks; default is read-only preview')
    args = parser.parse_args()
    print(json.dumps(sync(args.kind, args.metadata_dir, args.result_path, args.apply), indent=2))


if __name__ == '__main__': main()
