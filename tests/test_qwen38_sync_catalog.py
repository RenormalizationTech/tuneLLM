"""Synthetic local publications only: never call a Hub client or model runtime."""
import argparse
import copy
import hashlib
import json
from pathlib import Path

import pytest

from scripts import qwen38_sync_catalog as s
from scripts import qwen38_publish_verification as metadata

ROOT = Path(__file__).resolve().parents[1]


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + '\n')


def info(path): return {'bytes': path.stat().st_size, 'sha256': s.sha(path)}


def snapshot(root):
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob('*') if p.is_file()}


def fake_release(root, monkeypatch, kind='nf4'):
    monkeypatch.setattr(metadata, 'ROOT', root)
    repo = s.REPOS[kind]
    payload = root / 'artifacts/releases' / repo.split('/')[1]
    payload.mkdir(parents=True)
    for name, value in {'README.md': b'# Synthetic release\n', 'model.safetensors': b'Never load these synthetic bytes',
                        'config.json': b'{"synthetic":true}\n', 'm\u00e9thode.json': b'{}\n', '.gitattributes': b'* binary\n'}.items():
        (payload / name).write_bytes(value)
    original = {'status': 'published_remote_hashes_verified', 'private': False, 'repo_id': repo, 'revision': 'a' * 40,
                'files': {p.name: info(p) for p in payload.iterdir() if p.name != '.gitattributes'}}
    raw = root / 'results' / f'{kind}-original-raw.json'
    write(raw, original)
    verified = root / 'outputs' / ('gpu-' + kind)
    downloaded = {'completed': True, 'kind': kind, 'repo_id': repo, 'revision': original['revision'],
                  'anonymous_access_verified': True, 'all_file_hashes_verified': True,
                  'publication_sha256': s.sha(raw), 'files': {p.name: info(p) for p in payload.iterdir()}}
    write(verified / 'download.json', downloaded)
    gpu = {'completed': True, 'kind': kind, 'exact_smoke_token_replay': True, 'adapter_tensor_equality': True,
           'public_route': 'PeftModel.from_pretrained', 'download_sha256': s.sha(verified / 'download.json')}
    write(verified / 'result.json', gpu)
    folder = root / 'outputs' / ('metadata-' + kind)
    metadata.prepare(argparse.Namespace(kind=kind, output=folder, publication=raw, verification=verified))
    cfg = s.load(folder / 'prepared.json')
    # Simulate only the already-completed metadata commit's local result.
    # No publisher function or external service is called.
    for name in metadata.CHANGED_FILES: (payload / name).write_bytes((folder / name).read_bytes())
    result = {'status': 'published_and_verified', 'repo_id': repo, 'private': False, 'revision': 'b' * 40,
              'url': 'https://huggingface.co/' + repo, 'gpu_verified_revision': original['revision'],
              'metadata_only_update': True, 'runtime_files_unchanged': True, 'files': copy.deepcopy(cfg['expected_files']),
              'verification_evidence_sha256': s.sha(folder / 'verification.json'), 'prepared_sha256': s.sha(folder / 'prepared.json'),
              'gpu_result_sha256': s.sha(verified / 'result.json'), 'quality_or_gpu_rerun_for_documentation_update': False,
              'canonical_payload': str(payload), 'verified_at_utc': '2026-11-17T19:20:00+00:00'}
    write(folder / 'result.json', result)
    study = root / 'results' / (kind + '-study.json')
    write(study, {'completed': True, 'scope': 'Synthetic compact study fixture'})
    return {'kind': kind, 'metadata_dir': folder, 'result_path': study, 'root': root}, raw, verified


@pytest.fixture
def bundle(tmp_path, monkeypatch):
    # A standalone five-entry historical fixture survives later real catalog syncs.
    old_repos = ['Qwen3-4B-tuneLLM-GGUF', 'Qwen3-4B-tuneLLM-NF4', 'Qwen3-4B-tuneLLM-PRL-Writing-QLoRA',
                 'qwen2.5-7b-gcad-steering-controls', 'qwen2.5-7b-refusal-steering-experiment']
    catalog = {'schema_version': 1, 'checked_at_utc': '2026-11-17T06:05:55+00:00',
        'scope': 'Synthetic historical publication index', 'storage': {'preserve_this': True},
        'inventory_digest_recipe': copy.deepcopy(s.INVENTORY_RECIPE),
        'public_releases': [{'repo_id': 'RenormalizationTech/' + name, 'label': name, 'status': 'public',
            'revision': 'c' * 40, 'card_path': f'releases/{name}/README.md', 'historical_metadata_paths': []} for name in old_repos],
        'retired_releases': [{'repo_id': 'RenormalizationTech/Qwen3-4B-tuneLLM-GSM8K-QLoRA',
            'status': 'deleted_after_verified_archival', 'card_path': 'releases/Qwen3-4B-tuneLLM-GSM8K-QLoRA/README.md'}]}
    write(tmp_path / 'releases/catalog.json', catalog)
    for row in catalog['public_releases'] + catalog['retired_releases']:
        p = tmp_path / row['card_path']; p.parent.mkdir(parents=True, exist_ok=True); p.write_text('Historical mirror: ' + row['repo_id'])
    args, raw, verified = fake_release(tmp_path, monkeypatch)
    return args, raw, verified, catalog


def test_preview_is_read_only_and_apply_preserves_history_and_raw_evidence(bundle):
    args, raw, verified, original = bundle
    before = snapshot(args['root'])
    preview = s.sync(**args)
    assert not preview['applied'] and not preview['no_op']
    assert snapshot(args['root']) == before
    assert set(preview['changed_paths']) == {'releases/catalog.json',
        'releases/Qwen3.8-27B-tuneLLM-NF4/README.md', 'results/qwen38_nf4_public_verification_01.json'}
    result = s.sync(**args, apply=True)
    assert result['applied'] and not result['network_accessed'] and not result['GPU_started']
    after = s.load(args['root'] / 'releases/catalog.json')
    assert after['public_releases'][:5] == original['public_releases']
    assert after['retired_releases'] == original['retired_releases']
    for field in original.keys() - {'public_releases', 'checked_at_utc'}: assert after[field] == original[field]
    for path, value in before.items():
        if path != 'releases/catalog.json': assert (args['root'] / path).read_bytes() == value
    row = after['public_releases'][-1]
    assert row == result['entry'] and row['revision'] == 'b' * 40 and row['gpu_verified_revision'] == 'a' * 40
    assert row['publication_receipt'] == raw.relative_to(args['root']).as_posix()
    assert (args['root'] / row['card_path']).read_bytes() == (Path(s.load(args['metadata_dir'] / 'result.json')['canonical_payload']) / 'README.md').read_bytes()
    receipt = s.load(args['root'] / row['verification_receipt'])
    assert receipt['files'] == s.load(args['metadata_dir'] / 'result.json')['files']
    assert receipt['published_file_count'] == len(receipt['files'])
    reference = hashlib.sha256(json.dumps(receipt['files'], sort_keys=True, separators=(',', ':'), ensure_ascii=True).encode('utf-8')).hexdigest()
    assert row['published_inventory_sha256'] == receipt['published_inventory_sha256'] == reference
    assert '\\u00e9' in json.dumps(receipt['files'], ensure_ascii=True)
    for field in ('result_path', 'publication_receipt', 'verification_receipt'): assert (args['root'] / row[field]).is_file()
    assert row['historical_metadata_paths'] == []
    written = snapshot(args['root'])
    again = s.sync(**args, apply=True)
    assert again['no_op'] and again['changed_paths'] == []
    assert snapshot(args['root']) == written


@pytest.mark.parametrize('kind', ['gguf', 'qasper', 'prl'])
def test_all_supported_new_destinations_and_independent_additions(bundle, monkeypatch, kind):
    args, _, _, original = bundle
    s.sync(**args, apply=True)
    next_args, _, _ = fake_release(args['root'], monkeypatch, kind)
    s.sync(**next_args, apply=True)
    public = s.load(args['root'] / 'releases/catalog.json')['public_releases']
    assert public[:5] == original['public_releases'] and len(public) == 7
    assert public[5]['repo_id'] == s.REPOS['nf4'] and public[6]['repo_id'] == s.REPOS[kind]


@pytest.mark.parametrize('change', ['status', 'private', 'metadata_only_update', 'runtime_files_unchanged',
    'gpu_verified_revision', 'revision', 'repo_id', 'canonical_payload', 'files', 'prepared_sha256', 'verification_evidence_sha256', 'gpu_result_sha256'])
def test_incomplete_or_mismatched_completed_metadata_rejected_without_writes(bundle, change):
    args, _, _, _ = bundle
    path = args['metadata_dir'] / 'result.json'
    result = s.load(path)
    changes = {'status': 'uploaded_pending_hash_verification', 'private': True, 'metadata_only_update': False,
               'runtime_files_unchanged': False, 'gpu_verified_revision': 'c' * 40, 'revision': 'main',
               'repo_id': s.REPOS['prl'], 'canonical_payload': str(args['root'] / 'wrong'),
               'prepared_sha256': '0' * 64, 'verification_evidence_sha256': '0' * 64, 'gpu_result_sha256': '0' * 64}
    if change == 'files': result['files']['config.json']['sha256'] = '0' * 64
    else: result[change] = changes[change]
    write(path, result)
    before = snapshot(args['root'])
    with pytest.raises(ValueError): s.sync(**args, apply=True)
    # The local coordination lock is the only permitted change on failed apply.
    after = snapshot(args['root']); after.pop('outputs/qwen38-catalog-sync.lock', None)
    assert after == before


@pytest.mark.parametrize('target', ['download', 'gpu', 'publication', 'runtime', 'card', 'previous', 'prepared'])
def test_broken_evidence_chain_or_canonical_file_rejected(bundle, target):
    args, raw, verified, _ = bundle
    payload = Path(s.load(args['metadata_dir'] / 'result.json')['canonical_payload'])
    targets = {'download': verified / 'download.json', 'gpu': verified / 'result.json', 'publication': raw,
               'runtime': payload / 'model.safetensors', 'card': payload / 'README.md',
               'previous': args['metadata_dir'] / 'README.previous.md', 'prepared': args['metadata_dir'] / 'prepared.json'}
    with targets[target].open('ab') as stream: stream.write(b' ')
    before = snapshot(args['root'])
    with pytest.raises(ValueError): s.sync(**args)
    assert snapshot(args['root']) == before


@pytest.mark.parametrize('conflict', ['entry', 'receipt', 'mirror', 'duplicates', 'retired', 'recipe'])
def test_ambiguous_existing_data_is_never_overwritten(bundle, conflict):
    args, _, _, _ = bundle
    root = args['root']
    if conflict in ('entry', 'receipt', 'mirror'):
        s.sync(**args, apply=True)
        if conflict == 'entry':
            catalog = s.load(root / 'releases/catalog.json'); catalog['public_releases'][-1]['revision'] = 'c' * 40
            write(root / 'releases/catalog.json', catalog)
        elif conflict == 'receipt': (root / 'results/qwen38_nf4_public_verification_01.json').write_text('{}')
        else: (root / 'releases/Qwen3.8-27B-tuneLLM-NF4/README.md').write_text('different mirror')
    else:
        catalog = s.load(root / 'releases/catalog.json')
        if conflict == 'duplicates': catalog['public_releases'].append(catalog['public_releases'][0])
        elif conflict == 'retired': catalog['retired_releases'].append({'repo_id': s.REPOS['nf4']})
        else: catalog['inventory_digest_recipe']['serialization'] += '\n'
        write(root / 'releases/catalog.json', catalog)
    before = snapshot(root)
    with pytest.raises(ValueError): s.sync(**args, apply=True)
    after = snapshot(root)
    before.pop('outputs/qwen38-catalog-sync.lock', None)
    after.pop('outputs/qwen38-catalog-sync.lock', None)
    assert after == before


@pytest.mark.parametrize('name', ['../model', '/absolute', 'a/../b', 'a\\b', 'a//b', '.'])
def test_unsafe_inventory_paths_rejected(name):
    with pytest.raises(ValueError): s.inventory({name: {'bytes': 1, 'sha256': 'a' * 64}})


def test_study_report_required_and_cannot_be_receipt_destination(bundle):
    args, _, _, _ = bundle
    args['result_path'] = args['root'] / 'results/missing.json'
    with pytest.raises(ValueError, match='existing compact'): s.sync(**args)
    args['result_path'] = args['root'] / 'results/qwen38_nf4_public_verification_01.json'
    write(args['result_path'], {'completed': True})
    with pytest.raises(ValueError, match='overwrite original'): s.sync(**args)


def test_inventory_recipe_matches_existing_catalog_definition():
    assert s.INVENTORY_RECIPE == s.load(ROOT / 'releases/catalog.json')['inventory_digest_recipe']
