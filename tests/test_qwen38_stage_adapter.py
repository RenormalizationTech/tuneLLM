"""CPU synthetic contracts; no tensors are loaded and no judge is invoked."""
import copy
import hashlib
import json
from pathlib import Path
import random
import shutil
import zipfile

import pytest

from scripts import qwen38_stage_adapter as s
from scripts.qwen38_retention import compare_scores
from scripts import qwen38_prl_review as review, tone_judge

ROOT = Path(__file__).resolve().parents[1]


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(s.canonical(value) + '\n')


def lines(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(s.canonical(r) + '\n' for r in value))


def copy_source(root, name):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((ROOT / name).read_bytes())
    return path


def training(root, task='prl', run='arbitrary-registered-prl-run'):
    folder = root / 'outputs' / run
    counts = s.TRAINING_COUNTS[task]
    nf4 = root / 'nf4.json'
    write(nf4, {'files': {k: {'bytes': 7, 'sha256': hashlib.sha256(k.encode()).hexdigest()} for k in s.COMPUTE}})
    cfg = {'task': task, 'base': str(root / 'original-local-base'), 'training_counts': counts,
           'versions': {'torch': 'synthetic-only'}, 'nf4_artifact_manifest': str(nf4),
           'nf4_artifact_manifest_sha256': s.sha(nf4)}
    prepared = root / 'outputs/qwen38-science-01' / s.PREPARED[task]
    manifest = {'task': task, 'output_sha256': {}}
    cfg['prepared_input_sha256'] = {}
    for split, n in counts.items():
        path = prepared / (split + '.jsonl')
        lines(path, [{'id': f'{split}-{i}', 'training_eligible': True} for i in range(n)])
        cfg['prepared_input_sha256'][split] = s.sha(path)
        manifest['output_sha256'][split + '.jsonl'] = s.sha(path)
    if task == 'prl':
        audit = root / 'outputs/qwen38-prl-claim-audit-01/result.json'
        write(audit, {'completed': True, 'eligible': counts, 'records': [
            {'id': f'{split}-{i}', 'split': split, 'eligible': True, 'pass_eligible': [True, True]}
            for split, n in counts.items() for i in range(n)]})
        evidence = prepared / 'claim_audit'
        evidence.mkdir(parents=True, exist_ok=True)
        (evidence / 'result.json').write_bytes(audit.read_bytes())
        write(evidence / 'judge_isolation_audit.json', {'synthetic': True, 'isolated': True})
        write(evidence / 'judge_runtime_binding.json', {'synthetic': True, 'runtime': 'CPU fixture only'})
        manifest['audit_application'] = {'eligible': counts, 'AI_audit_only': True,
              'audit_result_sha256': s.sha(audit),
              'audit_evidence_sha256': {p.name: s.sha(p) for p in evidence.iterdir()}}
    write(prepared / 'manifest.json', manifest)
    cfg['prepared_manifest_sha256'] = s.sha(prepared / 'manifest.json')
    protocol = {'config': cfg, 'selected_ids': {}, 'input_sha256': {}, 'source_sha256': {}, 'target_tokens': {}}
    for split, count in counts.items():
        records = [{'id': f'{split}-{i}', 'input_ids': [1, 2]} for i in range(count)]
        path = folder / 'inputs' / (split + '.jsonl')
        lines(path, records)
        protocol['selected_ids'][split] = [r['id'] for r in records]
        protocol['input_sha256'][split] = s.sha(path)
        protocol['target_tokens'][split] = count
    for name in ('scripts/qwen38_nf4_smoke.py', 'scripts/qwen38_train_science.py', 'scripts/qwen38_science_runtime.py'):
        path = folder / 'source' / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('# synthetic source, never run\n')
        protocol['source_sha256'][name] = s.sha(path)
    write(folder / 'config.json', cfg)
    write(folder / 'protocol.json', protocol)
    write(root / 'results/protocols' / (run + '.json'), protocol)
    adapter = folder / 'checkpoints/step-00024'
    write(adapter / 'adapter_config.json', {'r': 16, 'lora_alpha': 32, 'lora_dropout': .05,
          'bias': 'none', 'task_type': 'CAUSAL_LM', 'base_model_name_or_path': cfg['base'], 'revision': None,
          'target_modules': ['q_proj', 'v_proj']})
    (adapter / 'adapter_model.safetensors').write_bytes(b'SYNTHETIC-not-real-tensors\x00\xff')
    write(folder / 'result.json', {'completed': True, 'steps': 24, 'protocol_sha256': s.sha(folder / 'protocol.json'),
          'selected': {'path': str(adapter), 'adapter_files': {p.name: {'sha256': s.sha(p), 'bytes': p.stat().st_size} for p in adapter.iterdir()}}})
    return folder, s.training_identity(task, folder, root)


def retention(root, task, identity, broken=0):
    fixture = {t: [{'key' if t == 'ifeval' else 'id': f'{t}-{i}'} for i in range(n)]
               for t, n in [('general_mc', 128), ('gsm8k', 128), ('ifeval', 32)]}
    # Real layout: the immutable source has tasks excluded from a registered arm.
    source_fixture = {**fixture, 'physics_mc': [{'id': 'physics-diagnostic'}],
                      'writing': [{'id': 'unused-writing', 'user': 'Invented writing fixture'}]}
    write(root / 'data/benchmark-screen-01/test.json', source_fixture)
    fixture_hash = s.sha(root / 'data/benchmark-screen-01/test.json')
    write(root / 'data/benchmark-screen-01/provenance.json', {'fixture_sha256': {'test': fixture_hash}})
    source = 'scripts/qwen38_retention.py'
    source_hash = s.sha(ROOT / source)
    sources = {source: source_hash, **{k: v for k, v in identity['protocol']['source_sha256'].items() if k.endswith(('qwen38_science_runtime.py', 'qwen38_nf4_smoke.py'))}}
    binding = {'fixture_sha256': fixture_hash, 'base_manifest_sha256': identity['config']['nf4_artifact_manifest_sha256'],
               'source_sha256': sources, 'tasks': list(fixture),
               'prepared_sha256': hashlib.sha256(s.canonical([]).encode()).hexdigest()}
    digest = hashlib.sha256(s.canonical(binding).encode()).hexdigest()
    scores = {}
    for arm in ('base', task):
        folder = root / 'outputs' / ('retention-' + arm)
        write(folder / 'fixture.json', fixture)
        write(folder / 'prepared.json', [])
        path = folder / 'source' / source
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((ROOT / source).read_bytes())
        for name in sources:
            if name == source: continue
            p = folder / 'source' / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes((identity['adapter'].parents[1] / 'source' / name).read_bytes())
        cfg = {'arm': arm, 'adapter': None if arm == 'base' else str(identity['adapter']),
               'adapter_audit': None if arm == 'base' else {**identity['audit'],
                   'training_config_sha256': s.sha(identity['adapter'].parents[1] / 'config.json'), 'saved_config': identity['saved']},
               'comparison_binding': binding, 'comparison_sha256': digest,
               'fixture_copy_sha256': s.sha(folder / 'fixture.json'), 'prepared_copy_sha256': s.sha(folder / 'prepared.json')}
        write(folder / 'registration.json', cfg)
        lines(folder / 'predictions.jsonl', [{'id': r.get('id', r.get('key')), 'task': t, 'output': 'synthetic'} for t, group in fixture.items() for r in group])
        report = {'arm': arm, 'comparison_sha256': digest, 'registration_sha256': s.sha(folder / 'registration.json'),
                  'predictions_sha256': s.sha(folder / 'predictions.jsonl'), 'tasks': {}}
        for t, group in fixture.items():
            details = [{'id': r.get('id', r.get('key')), 'correct': arm == 'base' or i >= broken} for i, r in enumerate(group)]
            report['tasks'][t] = {'total': len(details), 'correct': sum(r['correct'] for r in details), 'details': details}
        if arm != 'base':
            report['baseline_scores_sha256'] = s.sha(root / 'outputs/retention-base/scores.json')
            report['comparison'] = compare_scores(scores['base'], report)
        write(folder / 'scores.json', report)
        write(folder / 'result.json', {'completed': True, 'registration_sha256': report['registration_sha256'], 'predictions_sha256': report['predictions_sha256']})
        scores[arm] = report
    return root / ('outputs/retention-' + task), root / 'outputs/retention-base'


def attribution(root):
    for split, n in [('train', 70), ('validation', 10), ('test', 20), ('fresh_test', 40)]:
        records = [{'id': f'{split}-{i}', 'paper_id': f'paper-{split}-{i}', 'title': 'Gap in a lattice', 'authors': 'A. Researcher',
             'doi': f'10.1103/example.{split}.{i}', 'source_url': f'https://arxiv.org/src/{split}-{i}',
             'source_sha256': '1' * 64, 'arxiv_version': 2, 'journal-ref': 'Phys. Rev. Lett.',
             'authoritative_license': {'verified': True, 'license': 'http://creativecommons.org/licenses/by/4.0/',
                                       'response_sha256': '2' * 64}} for i in range(n)]
        path = root / ('data/qwen38-prl-abstract-01/fresh-acquisition-01/fresh.jsonl' if split == 'fresh_test'
                       else f'data/qwen38-prl-abstract-01/historical-reextracted-02/{split}.jsonl')
        lines(path, records)


def prl_report(root, identity, complete=True, wins=True):
    evaluation = root / 'outputs/qwen38-prl-evaluation-01'
    source_hashes = {}
    for name in ('qwen38_prl_review.py', 'qwen38_prl_claim_audit.py', 'qwen38_judge_runtime.py', 'tone_judge.py'):
        path = evaluation / 'source' / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((ROOT / 'scripts' / name).read_bytes())
        source_hashes[name] = s.sha(path)
    copy_source(root, 'scripts/tone_judge.py')
    papers = [{'id': f'fresh_test-{i}', 'selected_body': 'We obtain a positive gap of two units.'} for i in range(40)]
    lines(evaluation / 'final.jsonl', papers)
    protocol = {'inputs': {'final': s.sha(evaluation / 'final.jsonl')}, 'source_sha256': source_hashes,
                'gates': {'preference_each_baseline': .6, 'paper_bootstrap_lower_bound_above': .5,
                          'unsupported_mean_increase_each_baseline_at_most': 0, 'complete_papers': 40},
                'bootstrap': {'draws': 10000, 'seed': 20261117, 'unit': 'paper', 'method': 'paired percentile95'}}
    write(evaluation / 'registration.json', protocol)
    write(root / 'results/protocols/qwen38-prl-evaluation-01.json', protocol)
    write(evaluation / 'prompt_selection.json', {'completed': True, 'chosen': 'base', 'final_used': False, 'identical_baselines': True})
    generations = {}
    for arm in ('base', 'adapter'):
        folder = root / 'outputs' / ('prl-final-' + arm)
        cfg = {'task': 'prl', 'purpose': 'final', 'input_sha256': protocol['inputs']['final'],
               'adapter': str(identity['adapter']) if arm == 'adapter' else None,
               'adapter_files': {k: v['sha256'] for k, v in identity['audit']['adapter_files'].items()} if arm == 'adapter' else {},
               'template': 'base', 'max_new_tokens': 512, 'max_context': 8192, 'seed': 42, 'batch_size': 1,
               'cached_generation': True, 'enable_thinking': False, 'do_sample': False,
               'base_manifest_sha256': identity['config']['nf4_artifact_manifest_sha256'], 'versions': identity['config']['versions']}
        write(folder / 'registration.json', cfg)
        write(folder / 'result.json', {'completed': True, 'generations': 40, 'registration_sha256': s.sha(folder / 'registration.json')})
        lines(folder / 'predictions.jsonl', [{'id': r['id'], 'output': 'We obtain a positive gap.', 'rendered_prompt': r['selected_body']} for r in papers])
        generations[arm] = {'run': str(folder.relative_to(root)), 'registration_sha256': s.sha(folder / 'registration.json'),
                            'predictions_sha256': s.sha(folder / 'predictions.jsonl')}
    final = evaluation / 'final'
    plans = []
    for p in (0, 1):
        order = list(papers)
        random.Random(20261117 + p).shuffle(order)
        for index, paper in enumerate(order):
            payload, mapping = review.packet(paper, {'base': 'We obtain a positive gap.', 'adapter': 'We obtain a positive gap.'}, p, index)
            directory = final / f'pass-{p + 1}' / payload['packet_id']
            write(directory / 'payload.json', payload)
            value = {'packet_id': payload['packet_id'], 'support': [{'sid': r['sid'], 'label': 'SUPPORTED',
                     'quote': paper['selected_body']} for c in payload['candidates'] for r in c['sentences']],
                     'preferences': [{'pair': pair, 'preferred': next(k for k, v in mapping.items() if v == 'adapter') if wins else 'tie', 'reason': 'synthetic preference'} for pair in payload['pairs']]}
            attempt = {'returncode': 0, 'canonical_stdin_sha256': tone_judge.canonical_hash(payload),
                       'isolation': {'git_not_repository': True, 'git_environment_clean': True, 'returncode': 128},
                       'stdout': s.canonical({'subtype': 'success', 'modelUsage': {tone_judge.MODEL: {}}, 'result': s.canonical(value)})}
            if complete or not (paper['id'] == 'fresh_test-0' and p == 1):
                write(directory / 'attempt.json', attempt)
                write(directory / 'checked.json', {'valid': True, **review.check(value, payload), 'model_ids': [tone_judge.MODEL]})
            else:
                (directory / 'checked.json').unlink(missing_ok=True)
                (directory / 'attempt.json').unlink(missing_ok=True)
            plans.append({'directory': str(directory.relative_to(final)), 'paper_id': paper['id'], 'pass': p,
                          'mapping': mapping, 'payload_sha256': s.sha(directory / 'payload.json')})
    write(final / 'plans.json', plans)
    write(final / 'binding.json', {'registration_sha256': s.sha(evaluation / 'registration.json'),
          'plans_sha256': s.sha(final / 'plans.json'), 'generations': generations})
    preference = 1. if wins else .5
    records = [{'id': r['id'], 'preferences': {'base': preference}, 'unsupported': {'base': 0, 'adapter': 0},
                'unsupported_fraction': {'base': 0.0, 'adapter': 0.0}, 'partial': {'base': 0, 'adapter': 0}} for r in sorted(papers, key=lambda r: r['id'])]
    report = {'completed': True, 'papers': 40, 'records': records, 'preference': {'base': {'mean': preference, 'ci95': [preference, preference]}},
              'unsupported_difference': {'base': {'mean': 0., 'ci95': [0., 0.]}}, 'science_gates_pass': wins,
              'unsupported_mean': {'base': 0., 'adapter': 0.}, 'unsupported_nonframing_fraction_macro': {'base': 0., 'adapter': 0.},
              'AI_review_only': True, 'identical_baselines': True, 'baseline_comparisons': 1, 'retention_gate_pending': True,
              'prompt_selection_sha256': s.sha(evaluation / 'prompt_selection.json')}
    if not complete:
        report = {'completed': False, 'complete_papers': 39, 'expected_papers': 40}
    write(final / 'scores.json', report)
    return final / 'scores.json'


@pytest.fixture
def package(tmp_path):
    root = tmp_path
    run, identity = training(root)
    candidate, base = retention(root, 'prl', identity)
    science = prl_report(root, identity)
    attribution(root)
    copy_source(root, 'scripts/stage_physics_release.py')
    (root / 'LICENSE').write_bytes((ROOT / 'LICENSE').read_bytes())
    path = root / 'outputs/qwen38-science-01/environment.freeze.txt'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('peft==synthetic\nlocal @ file:///home/example/package\n')
    write(root / 'data/qasper-v03-01/manifest.json', {'license': 'CC-BY-4.0', 'revision': 'synthetic'})
    receipt = root / 'publication.json'
    write(receipt, {'status': 'published_remote_hashes_verified', 'private': False, 'repo_id': s.BASE,
                    'revision': 'a' * 40, 'files': s.load(root / 'nf4.json')['files']})
    return {'task': 'prl', 'training': run, 'science_path': science, 'retention_dir': candidate,
            'retention_base': base, 'publication': receipt, 'root': root}


def test_complete_package_byte_copy_two_field_config_and_publisher_manifest(package):
    original = package['training'] / 'checkpoints/step-00024'
    before = {p.name: p.read_bytes() for p in original.iterdir()}
    result = s.stage(**package)
    assert result['recommended'] and not result['GPU_started'] and not result['upload_performed']
    out = Path(result['staged'])
    assert (out / 'adapter_model.safetensors').read_bytes() == before['adapter_model.safetensors']
    assert all(p.read_bytes() == before[p.name] for p in original.iterdir())
    old, new = json.loads(before['adapter_config.json']), s.load(out / 'adapter_config.json')
    assert {k for k in old | new if old.get(k) != new.get(k)} == {'base_model_name_or_path', 'revision'}
    assert new['base_model_name_or_path'] == s.BASE and new['revision'] == 'a' * 40
    manifest = s.load(result['manifest'])
    assert manifest['repo_id'].endswith(s.TARGETS['prl']) and manifest['private'] is False
    assert set(manifest['files']) == {p.name for p in out.iterdir()}
    for name, meta in manifest['files'].items():
        assert meta == {'bytes': (out / name).stat().st_size, 'sha256': s.sha(out / name)}
    assert s.load(out / 'config_diff.json')['weights_byte_identical']
    papers = s.load(out / 'paper_attributions.json')['papers']
    assert len(papers) == 140 and sum(p['used_for_optimization'] for p in papers) == 57
    assert sum(p['used_for_checkpoint_selection'] for p in papers) == 9
    assert sum(p['split'] == 'fresh_test' for p in papers) == 40
    assert b'/home/example' not in (out / 'environment.lock.txt').read_bytes()
    with zipfile.ZipFile(out / 'source.zip') as archive:
        assert 'evaluation/qwen38_prl_review.py' in archive.namelist()
        assert 'retention/scripts/qwen38_retention.py' in archive.namelist()
    card = (out / 'README.md').read_text()
    assert 'Qwen3_5ForCausalLM' in card and 'prepare_model_for_kbit_training' in card
    assert 'fresh to this project' in card and 'not proven absent from pretraining' in card
    assert 'All 57 selected training papers and all 40 fresh evaluation papers have cond-mat categories.' in card
    assert 'Author overlap was not excluded.' in card
    assert 'do not establish author-held-out or pretraining-clean generalization' in card
    with pytest.raises(FileExistsError): s.stage(**package)


@pytest.mark.parametrize('change', ['private', 'revision', 'status', 'repo_id', 'weights'])
def test_invalid_public_base_never_bypassed(package, change):
    receipt = s.load(package['publication'])
    if change == 'weights': receipt['files']['model.safetensors']['sha256'] = '0' * 64
    else: receipt[change] = {'private': True, 'revision': 'main', 'status': 'uploaded_pending_hash_verification', 'repo_id': 'Other/model'}[change]
    write(package['publication'], receipt)
    with pytest.raises(ValueError): s.stage(**package, experimental=True)
    assert not (package['root'] / 'artifacts').exists()


@pytest.mark.parametrize('filename', ['adapter_model.safetensors', 'adapter_config.json'])
def test_selected_checkpoint_tamper_rejected(package, filename):
    with (package['training'] / 'checkpoints/step-00024' / filename).open('ab') as stream: stream.write(b' ')
    with pytest.raises(ValueError, match='changed after save'): s.stage(**package)


def test_incomplete_judging_only_explicit_experimental(package):
    identity = s.training_identity('prl', package['training'], package['root'])
    package['science_path'] = prl_report(package['root'], identity, complete=False)
    with pytest.raises(ValueError, match='explicit --experimental'): s.stage(**package)
    result = s.stage(**package, experimental=True)
    assert not result['recommended'] and result['incomplete_science']
    card = (Path(result['staged']) / 'README.md').read_text()
    assert 'INCOMPLETE FINAL JUDGING' in card and '39/40' in card and 'NOT RECOMMENDED' in card


def test_failed_retention_requires_experimental(package):
    identity = s.training_identity('prl', package['training'], package['root'])
    retention(package['root'], 'prl', identity, broken=6)
    with pytest.raises(ValueError, match='explicit --experimental'): s.stage(**package)
    result = s.stage(**package, experimental=True)
    assert not result['recommended']


@pytest.mark.parametrize('kind', ['incomplete-generation', 'wrong-template', 'duplicate-id', 'wrong-input', 'forged-record', 'forged-ci', 'duplicate-plan', 'changed-rubric'])
def test_prl_report_adversarial_rejections(package, kind):
    root = package['root']
    final = package['science_path'].parent
    if kind in ('incomplete-generation', 'wrong-template', 'wrong-input', 'duplicate-id'):
        folder = root / 'outputs/prl-final-adapter'
        if kind == 'incomplete-generation':
            v = s.load(folder / 'result.json'); v['completed'] = False; write(folder / 'result.json', v)
        elif kind in ('wrong-template', 'wrong-input'):
            v = s.load(folder / 'registration.json'); v['template' if kind == 'wrong-template' else 'input_sha256'] = 'wrong'
            write(folder / 'registration.json', v)
        else:
            v = s.rows(folder / 'predictions.jsonl'); v[-1] = v[0]; lines(folder / 'predictions.jsonl', v)
        binding = s.load(final / 'binding.json')
        for name in ('registration', 'predictions'):
            binding['generations']['adapter'][name + '_sha256'] = s.sha(folder / (name + ('.json' if name == 'registration' else '.jsonl')))
        write(final / 'binding.json', binding)
    elif kind.startswith('forged'):
        v = s.load(package['science_path'])
        if kind == 'forged-record': v['records'][0]['unsupported']['adapter'] = 1
        else: v['preference']['base']['ci95'] = [.6, 1.]
        write(package['science_path'], v)
    else:
        plans = s.load(final / 'plans.json')
        if kind == 'duplicate-plan': plans[-1] = plans[0]
        else:
            p = final / plans[0]['directory'] / 'payload.json'
            v = s.load(p); v['rubric'] = {}; write(p, v); plans[0]['payload_sha256'] = s.sha(p)
        write(final / 'plans.json', plans)
        b = s.load(final / 'binding.json'); b['plans_sha256'] = s.sha(final / 'plans.json'); write(final / 'binding.json', b)
    with pytest.raises(ValueError): s.stage(**package, experimental=True)


@pytest.mark.parametrize('kind', ['incomplete', 'duplicate', 'wrong-base', 'forged-gate'])
def test_retention_contract_rejections(package, kind):
    folder = package['retention_dir']
    if kind == 'incomplete':
        v = s.load(folder / 'result.json'); v['completed'] = False; write(folder / 'result.json', v)
    else:
        v = s.load(folder / 'scores.json')
        if kind == 'duplicate': v['tasks']['general_mc']['details'][-1] = v['tasks']['general_mc']['details'][0]
        elif kind == 'forged-gate': v['comparison']['retention_pass'] = False
        else: v['comparison_sha256'] = '0' * 64
        write(folder / 'scores.json', v)
    with pytest.raises(ValueError): s.stage(**package, experimental=True)


@pytest.mark.parametrize('value', [{'token': 'secret'}, {'authorization': 'Bearer secret'}, {'x': 'hf_' + 'A' * 32}])
def test_credentials_rejected(value, tmp_path):
    with pytest.raises(ValueError): s.safe_json(value, tmp_path)


def test_metadata_sanitization_only(tmp_path):
    raw = {'path': str(tmp_path / 'data'), 'machine': '/home/alice/private/file', 'scratch': '/tmp/cache/model', 'url': 'https://arxiv.org/abs/1234'}
    sanitized = json.loads(s.safe_json(raw, tmp_path))
    assert sanitized['path'] == '<PROJECT_ROOT>/data' and sanitized['machine'] == '<HOME>/private/file'
    assert sanitized['scratch'] == '<LOCAL_PATH>' and sanitized['url'] == raw['url']


def test_generic_training_run_registry_and_missing_split(tmp_path):
    folder, identity = training(tmp_path, run='qwen38-prl-02')
    assert identity['config']['training_counts'] == {'train': 57, 'validation': 9}
    p = s.load(folder / 'protocol.json'); p['input_sha256'].pop('validation')
    write(folder / 'protocol.json', p); write(tmp_path / 'results/protocols/qwen38-prl-02.json', p)
    r = s.load(folder / 'result.json'); r['protocol_sha256'] = s.sha(folder / 'protocol.json'); write(folder / 'result.json', r)
    with pytest.raises(ValueError, match='both training'): s.training_identity('prl', folder, tmp_path)


def test_per_paper_license_and_selected_split_rejection(package):
    selected = s.load(package['training'] / 'protocol.json')['selected_ids']
    selected['train'][0] = 'test-0'
    with pytest.raises(ValueError, match='absent'): s.prl_attributions(package['root'], selected)
    path = package['root'] / 'data/qwen38-prl-abstract-01/fresh-acquisition-01/fresh.jsonl'
    records = s.rows(path); records[0]['authoritative_license']['verified'] = False; lines(path, records)
    with pytest.raises(ValueError, match='CC-BY'): s.prl_attributions(package['root'], {'train': [], 'validation': []})


def test_qasper_frozen_validation_and_card(tmp_path):
    # Reuse the independently tested CPU fixture builder, not any real final predictions.
    from test_qwen38_qasper_analysis import bundle, run_analysis
    b = bundle.__wrapped__(tmp_path)
    target = tmp_path / 'outputs/qwen38-science-01/prepared-qasper-tokenizer-01'
    shutil.copytree(b['prepared'], target)
    b['prepared'] = target
    # Preserve exact registered bytes; only registration input_path metadata points at old prepared.
    for arm in ('base', 'prompt', 'adapter'):
        from test_qwen38_qasper_analysis import rebind_run
        run = tmp_path / 'outputs' / ('final-' + arm)
        reg = s.load(run / 'registration.json'); reg['input_path'] = str(target / 'test.jsonl'); write(run / 'registration.json', reg)
        rebind_run(b, arm)
    report = run_analysis(b)
    copy_source(tmp_path, 'scripts/qwen38_qasper_analysis.py')
    identity = {'adapter': b['adapter'], 'audit': s.validate_adapter(b['adapter'], 'qasper', 'base')}
    replay, status = s.qasper_science(b['output'], identity, tmp_path, tokenizer=b['tokenizer'])
    assert status['passed'] and replay == report
    ret = {'comparison': {'tasks': {}}}
    card = s.card('qasper', status, report, ret, 'a' * 40, {}, False)
    assert 'Strict gain (95% paper CI)' in card and 'Lenient gain (95% paper CI)' in card and 'Invalid strict JSON / all403' in card
    modified = copy.deepcopy(report)
    for v in modified['analysis']['bootstrap']['lenient'].values():
        v['difference_a_minus_b'] = 0.; v['ci95'] = [0., 0.]
    assert 'FORMAT-DRIVEN RESULT' in s.card('qasper', status, modified, ret, 'a' * 40, {}, False)
    modified = copy.deepcopy(report); modified['analysis']['science_gates_pass'] = False; write(b['output'], modified)
    with pytest.raises(ValueError): s.qasper_science(b['output'], identity, tmp_path, tokenizer=b['tokenizer'])


def test_null_science_stages_only_experimental(package):
    identity = s.training_identity('prl', package['training'], package['root'])
    prl_report(package['root'], identity, wins=False)
    with pytest.raises(ValueError, match='explicit --experimental'): s.stage(**package)
    result = s.stage(**package, experimental=True)
    assert not result['recommended']
    provenance = s.load(Path(result['staged']) / 'provenance.json')
    assert not provenance['all_science_and_retention_gates_pass']
    card = (Path(result['staged']) / 'README.md').read_text()
    assert 'NOT RECOMMENDED' in card and '| adapter vs base | 0.500000 |' in card


@pytest.mark.parametrize('failure', [None, 'changed-data', 'changed-seed', 'old-optimized', 'wrong-hash'])
def test_registered_hardware_only_replacement(tmp_path, failure):
    old, _ = training(tmp_path, run='qwen38-prl-01')
    new, _ = training(tmp_path, run='qwen38-prl-02')
    for folder, gpu in ((old, '0'), (new, '1')):
        p = s.load(folder / 'protocol.json'); p['config']['gpu'] = gpu
        if failure == 'changed-seed' and folder == new: p['config']['seed'] = 99
        if failure == 'changed-data' and folder == new: p['target_tokens']['train'] += 1
        write(folder / 'protocol.json', p); write(folder / 'config.json', p['config'])
        write(tmp_path / 'results/protocols' / (folder.name + '.json'), p)
        r = s.load(folder / 'result.json'); r['protocol_sha256'] = s.sha(folder / 'protocol.json')
        if folder == old: r['steps'] = 1 if failure == 'old-optimized' else 0
        write(folder / 'result.json', r)
    identity = s.training_identity('prl', new, tmp_path)
    path = prl_report(tmp_path, identity)
    evaluation = path.parent.parent
    write(tmp_path / 'results/protocols/qwen38-prl-hardware-amendment-01.json', {
        'superseded_prepared_only_run': old.name, 'training_run': new.name,
        'old_training_protocol_sha256': s.sha(old / 'protocol.json'),
        'new_training_protocol_sha256': '0' * 64 if failure == 'wrong-hash' else s.sha(new / 'protocol.json'),
        'evaluation_registration_sha256': s.sha(evaluation / 'registration.json'),
        'optimization_started_at_amendment': False, 'config_changes': {'gpu': ['0', '1']}})
    if failure:
        with pytest.raises(ValueError): s.prl_science(path, identity, tmp_path)
    else:
        assert s.prl_science(path, identity, tmp_path)[1]['passed']


@pytest.mark.parametrize('kind', ['source-bytes', 'audit-incomplete', 'ineligible-id'])
def test_prepared_training_and_audit_integrity(package, kind):
    folder = package['training']
    if kind == 'source-bytes':
        with (folder / 'source/scripts/qwen38_science_runtime.py').open('ab') as stream: stream.write(b'# tamper\n')
    elif kind == 'audit-incomplete':
        write(package['root'] / 'outputs/qwen38-prl-claim-audit-01/result.json', {'completed': False})
    else:
        p = package['root'] / 'outputs/qwen38-science-01/prepared-prl-audited-01/train.jsonl'
        r = s.rows(p); r[0]['training_eligible'] = False; lines(p, r)
    with pytest.raises(ValueError): s.stage(**package, experimental=True)


def test_full_source_fixture_and_selected_task_copy_have_distinct_hashes(package):
    folder = package['retention_base']
    reg = s.load(folder / 'registration.json')
    full = s.load(package['root'] / 'data/benchmark-screen-01/test.json')
    assert {'physics_mc', 'writing'} <= set(full)
    assert reg['fixture_copy_sha256'] != reg['comparison_binding']['fixture_sha256']
    assert s.load(folder / 'fixture.json') == {task: full[task] for task in reg['comparison_binding']['tasks']}
    identity = s.training_identity('prl', package['training'], package['root'])
    assert s.retention_report(folder, 'base', identity, package['root'])[0]['arm'] == 'base'


@pytest.mark.parametrize('tamper', ['copy-bytes', 'rebound-copy-content', 'full-source'])
def test_projection_fix_preserves_both_hash_and_content_guards(package, tamper):
    folder = package['retention_base']
    path = package['root'] / 'data/benchmark-screen-01/test.json' if tamper == 'full-source' else folder / 'fixture.json'
    value = s.load(path)
    value['general_mc'][0]['prompt'] = 'Changed synthetic question'
    write(path, value)
    if tamper == 'rebound-copy-content':
        reg = s.load(folder / 'registration.json'); reg['fixture_copy_sha256'] = s.sha(path)
        write(folder / 'registration.json', reg)
        for name in ('result.json', 'scores.json'):
            data = s.load(folder / name); data['registration_sha256'] = s.sha(folder / 'registration.json')
            write(folder / name, data)
    identity = s.training_identity('prl', package['training'], package['root'])
    with pytest.raises(ValueError): s.retention_report(folder, 'base', identity, package['root'])


def test_cli_defaults_to_registered_base04(monkeypatch, capsys):
    import sys
    captured = {}
    def fake_stage(task, training, science, candidate, baseline, *args):
        captured['baseline'] = baseline
        return {'synthetic': True}
    monkeypatch.setattr(s, 'stage', fake_stage)
    monkeypatch.setattr(sys, 'argv', ['qwen38_stage_adapter', '--task', 'prl'])
    s.main()
    assert captured['baseline'] == s.ROOT / 'outputs/qwen38-retention-base-04'
    assert json.loads(capsys.readouterr().out) == {'synthetic': True}


def test_synthetic_examples_match_data_api_instructions():
    from scripts.qwen38_science_data import prepare_qasper, prepare_prl
    text = 'In a synthetic experiment, the blue sample cooled to 2 K.'
    question = 'What temperature did the blue sample reach?'
    paper = {'id': 'synthetic', 'abstract': '', 'full_text': [{'section_name': 'Results', 'paragraphs': [text]}],
             'qas': [{'question_id': 'q', 'question': question, 'answers': [{'answer': {
                 'unanswerable': False, 'yes_no': None, 'extractive_spans': [], 'free_form_answer': '2 K', 'evidence': [text]}}]}]}
    qa = prepare_qasper(paper, 'synthetic', len, 10000)[0]['user']
    # Numbering includes section/abstract entries; ensure instruction and paragraph rendering match.
    example = s.synthetic_prompt('qasper')
    assert example.split('\n\n')[0] == qa.split('\n\n')[0]
    assert '[p000000] Results\n' + text in example and example.endswith('Question: ' + question)
    prl = s.synthetic_prompt('prl')
    body = prl.split('\n\n', 1)[1]
    row = {'id': 'synthetic', 'paper_id': 'synthetic', 'text': body, 'abstract': 'A synthetic model is studied.'}
    prepared = prepare_prl(row, 'synthetic', len, 10000)
    assert prl.split('\n\n')[0] == prepared['user'].split('\n\n')[0]


def test_card_recipe_has_resolved_revision_and_runnable_invented_example(package):
    import ast
    identity = s.training_identity('prl', package['training'], package['root'])
    science, status = s.prl_science(package['science_path'], identity, package['root'])
    retention_report = s.paired_retention(package['retention_dir'], package['retention_base'], 'prl', identity, package['root'])
    card = s.card('prl', status, science, retention_report, 'a' * 40, identity['config'], False)
    recipe = card.split('```python\n', 1)[1].split('```', 1)[0]
    tree = ast.parse(recipe)
    assert 'REPLACE_WITH' not in card and 'Supply the registered' not in card
    assert recipe.count('HfApi().model_info(adapter_id).sha') == 1
    assert 'print("Resolved adapter release:", adapter_revision)' in recipe
    assert 'revision=adapter_revision' in recipe and 'enable_thinking=False' in recipe
    user = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == 'user' for t in n.targets))
    assert user == s.synthetic_prompt('prl')
    assert 'Modification notice (Apache-2.0)' in card and 'text-only NF4' in card and 'task-specific QLoRA' in card
    assert 'removing vision/MTP components' in card and 'quantizing its language weights' in card


def test_card_distinguishes_science_batch1_from_retention_batch4(package):
    identity = s.training_identity('prl', package['training'], package['root'])
    science, status = s.prl_science(package['science_path'], identity, package['root'])
    retention = s.paired_retention(package['retention_dir'], package['retention_base'], 'prl', identity, package['root'])
    retention['batching'] = {'generation_batch_size': 4, 'mc_batch_size': 1}
    card = s.card('prl', status, science, retention, 'a' * 40, identity['config'], False)
    assert 'Science generation uses batch1; retention generation uses batch4' in card
    assert 'retention MC likelihood scoring remains batch1' in card
    assert 'Batch4 is not token-equivalent to batch1: 2/4 invented preflight probes differed' in card
    assert 'exact same batch4 groups and padding across all retention arms' in card
    assert 'must not be pooled' in card


def test_enriched_retention_config_audit_is_bound_to_training(package):
    folder = package['retention_dir']
    reg = s.load(folder / 'registration.json')
    assert 'saved_config' in reg['adapter_audit']
    reg['adapter_audit']['saved_config']['r'] = 8
    write(folder / 'registration.json', reg)
    for name in ('scores.json', 'result.json'):
        value = s.load(folder / name); value['registration_sha256'] = s.sha(folder / 'registration.json')
        write(folder / name, value)
    identity = s.training_identity('prl', package['training'], package['root'])
    with pytest.raises(ValueError, match='different selected adapter/config'):
        s.retention_report(folder, 'prl', identity, package['root'])


@pytest.fixture
def secondary(package):
    root = package['root']
    final = package['science_path'].parent
    evaluation = final.parent
    records = s.rows(evaluation / 'final.jsonl')
    for i, r in enumerate(records):
        r['assistant'] = 'We measure a positive gap in the synthetic lattice model.'
        r['copy_baseline'] = {'text': '' if i == 0 else r['selected_body']}
    lines(evaluation / 'final.jsonl', records)
    protocol = s.load(evaluation / 'registration.json')
    protocol['inputs']['final'] = s.sha(evaluation / 'final.jsonl')
    write(evaluation / 'registration.json', protocol)
    write(root / 'results/protocols/qwen38-prl-evaluation-01.json', protocol)
    binding = s.load(final / 'binding.json')
    binding['registration_sha256'] = s.sha(evaluation / 'registration.json')
    for arm, mapping in binding['generations'].items():
        folder = root / mapping['run']
        cfg = s.load(folder / 'registration.json'); cfg['input_sha256'] = protocol['inputs']['final']
        write(folder / 'registration.json', cfg)
        mapping['registration_sha256'] = s.sha(folder / 'registration.json')
        status = s.load(folder / 'result.json'); status['registration_sha256'] = mapping['registration_sha256']
        write(folder / 'result.json', status)
    write(final / 'binding.json', binding)
    source = copy_source(root, 'scripts/qwen38_prl_diagnostics.py')
    copy_source(root, 'scripts/qwen38_science_score.py')
    write(root / 'results/protocols/qwen38-prl-diagnostics-01.json', {
        'source_sha256': s.sha(source), 'primary_evaluation_registration_sha256': s.sha(evaluation / 'registration.json'),
        'scope': 'Secondary descriptive diagnostics; never change gates'})
    from scripts import qwen38_prl_diagnostics as helper
    report = {'scope': 'secondary custom lexical diagnostics; no semantic faithfulness claim',
              'source_sha256': s.sha(source), 'input_sha256': protocol['inputs']['final'], 'arms': {}}
    for arm, mapping in binding['generations'].items():
        predicted = {r['id']: r['output'] for r in s.rows(root / mapping['run'] / 'predictions.jsonl')}
        report['arms'][arm] = {'records': [{'id': r['id'], **helper.describe(r, predicted[r['id']])} for r in records],
                              'predictions_sha256': mapping['predictions_sha256']}
    report['fixed_copy_baseline'] = [{'id': r['id'], **helper.describe(r, r['copy_baseline']['text'])} for r in records]
    report['reviewer_diagnostics'] = helper.agreement(final)
    path = root / 'results/secondary.json'
    write(path, report)
    return package, path


def test_secondary_replay_export_provenance_and_source_archive(secondary):
    package, path = secondary
    checked = s.prl_diagnostics(path, package['science_path'], package['root'])
    assert checked['summary']['base']['records'] == 40
    assert checked['summary']['fixed_copy']['means']['words']['denominator'] == 40
    assert checked['summary']['base']['means']['novel_unique_8gram_fraction'] == {'mean': None, 'denominator': 0}
    assert checked['summary']['fixed_copy']['means']['novel_unique_8gram_fraction']['denominator'] == 39
    assert checked['provenance']['gates_affected'] is False
    # Entirely synthetic package staging verifies plumbing, without actual model data.
    result = s.stage(**package, diagnostics=path)
    output = Path(result['staged'])
    exported = s.load(output / 'secondary_diagnostics.json')
    assert exported['reviewer_diagnostics'] == checked['report']['reviewer_diagnostics']
    assert exported['verified_summary'] == checked['summary']
    assert s.load(output / 'provenance.json')['secondary_diagnostics'] == json.loads(s.safe_json(checked['provenance'], package['root']))
    with zipfile.ZipFile(output / 'source.zip') as archive:
        assert archive.read('diagnostics/qwen38_prl_diagnostics.py') == checked['source'].read_bytes()
        assert archive.read('diagnostics/qwen38-prl-diagnostics-01.json') == checked['protocol'].read_bytes()
        assert 'diagnostics/qwen38_science_score.py' in archive.namelist()
    assert 'secondary_diagnostics.json' in s.load(result['manifest'])['files']


@pytest.mark.parametrize('change', ['source', 'input', 'prediction', 'record', 'duplicate', 'fixed-copy', 'agreement', 'registration'])
def test_secondary_forgery_and_wrong_bindings_rejected(secondary, change):
    package, path = secondary
    report = s.load(path)
    if change == 'source': report['source_sha256'] = '0' * 64
    elif change == 'input': report['input_sha256'] = '0' * 64
    elif change == 'prediction': report['arms']['adapter']['predictions_sha256'] = '0' * 64
    elif change == 'record': report['arms']['adapter']['records'][0]['words'] += 1
    elif change == 'duplicate': report['arms']['adapter']['records'][-1] = report['arms']['adapter']['records'][0]
    elif change == 'fixed-copy': report['fixed_copy_baseline'][0]['custom_whitespace_lcs_f1'] = 1.
    elif change == 'agreement': report['reviewer_diagnostics']['support_agreement'] = .123
    else:
        protocol_path = package['root'] / 'results/protocols/qwen38-prl-diagnostics-01.json'
        p = s.load(protocol_path); p['primary_evaluation_registration_sha256'] = '0' * 64; write(protocol_path, p)
    write(path, report)
    with pytest.raises(ValueError): s.prl_diagnostics(path, package['science_path'], package['root'])
    assert not (package['root'] / 'artifacts/releases').exists()


def test_negative_prl_card_secondary_overlap_and_preference_semantics(secondary):
    package, path = secondary
    diagnostics = s.prl_diagnostics(path, package['science_path'], package['root'])
    science = s.load(package['science_path'])
    science['science_gates_pass'] = False
    science['preference']['base'] = {'mean': .175, 'ci95': [.075, .2875]}
    science['unsupported_difference']['base'] = {'mean': .025, 'ci95': [0., .075]}
    # Card-only numeric examples match the reported result; validation is tested separately.
    for name, novelty in [('base', .985602701), ('adapter', .525500864)]:
        diagnostics['summary'][name]['means']['novel_unique_8gram_fraction'] = {'mean': novelty, 'denominator': 40}
    status = {'completed': True, 'passed': False, 'incomplete': False}
    card = s.card('prl', status, science, {'comparison': {'tasks': {}}}, 'a' * 40, {}, True, diagnostics=diagnostics)
    assert 'For this abstract task, recommend the unadapted' in card
    assert 'half credit to ties' in card and 'two passes per paper' in card and 'not a raw win rate' in card
    assert '| 0.175000 | [0.075, 0.2875] | 0.025000 | [0.0, 0.075]' in card
    assert 'does not demonstrate a large factual regression' in card
    assert 'not metrics from a ROUGE package' in card and 'Custom whitespace LCS F1' in card
    assert '0.525501 (n=40)' in card and 'n/a (n=0)' not in card  # Both novel denominators are now defined.
    assert 'Source overlap increased while judged quality fell' in card
    assert 'causal mechanism is unproven' in card
    assert 'lexical overlap does not establish support' in card


def test_secondary_argument_prl_only_and_default_compatibility(package):
    with pytest.raises(ValueError, match='PRL-only'):
        s.stage('qasper', package['training'], package['science_path'], package['retention_dir'], package['retention_base'],
                package['publication'], diagnostics=Path('not-read.json'))


def test_training_identity_reads_prepared_claim_audit_witnesses(tmp_path):
    folder, expected = training(tmp_path)
    prepared = tmp_path / 'outputs/qwen38-science-01' / s.PREPARED['prl']
    original = tmp_path / 'outputs/qwen38-prl-claim-audit-01'
    manifest = s.load(prepared / 'manifest.json')
    for name in ('judge_isolation_audit.json', 'judge_runtime_binding.json'):
        assert not (original / name).exists()
        assert s.sha(prepared / 'claim_audit' / name) == manifest['audit_application']['audit_evidence_sha256'][name]
    assert (original / 'result.json').is_file()
    assert s.training_identity('prl', folder, tmp_path) == expected


@pytest.mark.parametrize('name', ['judge_isolation_audit.json', 'judge_runtime_binding.json'])
def test_training_identity_rejects_tampered_prepared_witness(tmp_path, name):
    folder, _ = training(tmp_path)
    prepared = tmp_path / 'outputs/qwen38-science-01' / s.PREPARED['prl']
    original = tmp_path / 'outputs/qwen38-prl-claim-audit-01'
    # A matching file in the original audit dir must not mask a corrupt copied witness.
    (original / name).write_bytes((prepared / 'claim_audit' / name).read_bytes())
    with (prepared / 'claim_audit' / name).open('ab') as stream:
        stream.write(b' tampered prepared witness')
    with pytest.raises(ValueError, match='bound file hash differs: ' + name):
        s.training_identity('prl', folder, tmp_path)


@pytest.fixture
def qasper_completion(tmp_path, monkeypatch):
    # Reuse only synthetic completion fixtures; never inspect the active run.
    from test_qwen38_complete_science import fixture, complete
    from scripts import qwen38_complete_science as helper
    f = fixture.__wrapped__(tmp_path, monkeypatch)
    amendment = s.load(f['amendment'])
    amendment['original_run'] = f['original'].relative_to(tmp_path).as_posix()
    write(f['amendment'], amendment)
    run = complete(f)
    assembled = tmp_path / 'outputs/assembled'
    helper.assemble(run, assembled, root=tmp_path, tokenizer=f['tokenizer'])
    copy_source(tmp_path, 'scripts/qwen38_complete_science.py')
    science = {'arms': {'base': {'provenance': {'path': str(assembled),
        'registration_sha256': s.sha(assembled / 'registration.json'), 'result_sha256': s.sha(assembled / 'result.json'),
        'predictions_sha256': s.sha(assembled / 'predictions.jsonl')}}}}
    return f, science, assembled


def test_qasper_completion_replays_full_cpu_helper_and_preserves_receipt(qasper_completion, monkeypatch):
    from scripts import qwen38_complete_science as helper
    f, science, assembled = qasper_completion
    original_source = s.sha(helper.__file__)
    validate = helper.validate_assembly
    calls = []
    def spy(*args, **kwargs):
        calls.append((args, kwargs))
        return validate(*args, **kwargs)
    monkeypatch.setattr(helper, 'validate_assembly', spy)
    completion = s.qasper_baseline_completion(science, f['root'], tokenizer=f['tokenizer'])
    assert len(calls) == 1 and calls[0][1]['tokenizer'] is f['tokenizer']
    metadata = completion['metadata']
    assert metadata['proof']['prefix_count'] == 395 and metadata['proof']['suffix_count'] == 8
    assert metadata['original_failed_receipt'] == s.load(f['original'] / 'result.json')
    assert metadata['original_failed_receipt']['completed'] is False
    assert len(metadata['suffix']['ids']) == 8 and metadata['assembled']['status']['generations'] == 403
    assert metadata['raw_prediction_texts_exported'] is False
    assert s.load(completion['amendment'])['parent_pool_limit_gpu_hours'] == 12
    for key in ('scripts/qwen38_eval_science.py', 'scripts/qwen38_science_runtime.py'):
        assert str(f['root'] / key) in completion['bound']
    assert str(f['output'] / 'generation/predictions.jsonl') in completion['bound']
    assert s.sha(helper.__file__) == original_source
    exported = json.loads(s.safe_json(metadata, f['root']))
    assert str(f['root']) not in json.dumps(exported)
    assert 'output' not in metadata['suffix']['result']


@pytest.mark.parametrize('change', ['source-file', 'source-declaration', 'original', 'suffix', 'amendment', 'assembly'])
def test_qasper_completion_source_and_segment_binding_guards(qasper_completion, change):
    f, science, assembled = qasper_completion
    if change == 'source-file':
        path = f['root'] / 'scripts/qwen38_complete_science.py'
        with path.open('ab') as stream: stream.write(b'\n# tampered COPY, never real helper\n')
    elif change == 'source-declaration':
        p = assembled / 'registration.json'
        value = s.load(p); value['assembler_source_sha256'] = '0' * 64; write(p, value)
        science['arms']['base']['provenance']['registration_sha256'] = s.sha(p)
    elif change == 'amendment':
        with f['amendment'].open('ab') as stream: stream.write(b' ')
    else:
        path = (f['original'] / 'predictions.jsonl' if change == 'original' else
                f['output'] / 'generation/predictions.jsonl' if change == 'suffix' else assembled / 'predictions.jsonl')
        with path.open('ab') as stream: stream.write(b'\n')
        if change == 'assembly': science['arms']['base']['provenance']['predictions_sha256'] = s.sha(path)
    with pytest.raises(ValueError): s.qasper_baseline_completion(science, f['root'], tokenizer=f['tokenizer'])


def test_plain_qasper_baseline_completion_remains_optional(tmp_path):
    directory = tmp_path / 'outputs/plain-base'
    write(directory / 'registration.json', {'ids': [f'q{i}' for i in range(403)]})
    write(directory / 'result.json', {'completed': True, 'generations': 403})
    lines(directory / 'predictions.jsonl', [{'id': f'q{i}'} for i in range(403)])
    science = {'arms': {'base': {'provenance': {'path': str(directory),
        'registration_sha256': s.sha(directory / 'registration.json'), 'result_sha256': s.sha(directory / 'result.json'),
        'predictions_sha256': s.sha(directory / 'predictions.jsonl')}}}}
    assert s.qasper_baseline_completion(science, tmp_path) is None


def test_qasper_completion_card_disclosure_at_end_of_optional_signature(qasper_completion):
    f, report, _ = qasper_completion
    completion = s.qasper_baseline_completion(report, f['root'], tokenizer=f['tokenizer'])
    score = {mode: {'groups': {'primary': {'answer_f1': .1}, 'all_sampled': {'invalid_outputs': 0}}}
             for mode in ('strict', 'lenient')}
    science = {'arms': {arm: {'scores': score} for arm in ('base', 'prompt', 'adapter')},
        'protocol': {'success_gates': {'answer_f1_gain_each_baseline': .05}}, 'aliases': {'prompt': 'base'},
        'analysis': {'bootstrap': {mode: {arm: {'difference_a_minus_b': 0., 'ci95': [0., 0.]}
                                      for arm in ('base', 'prompt')} for mode in ('strict', 'lenient')}}}
    args = ('qasper', {'completed': True, 'passed': False, 'incomplete': False}, science,
            {'comparison': {'tasks': {}}}, 'a' * 40, {}, True)
    plain = s.card(*args)
    card = s.card(*args, None, None, completion)
    assert 'Baseline execution disclosure' not in plain
    assert 'original 10000-second phase failed after 395/403' in card
    assert 'eight never-previously-evaluated suffix questions were completed once' in card
    assert 'capped at 900 seconds' in card and 'two sequential GPU0 executions' in card
    assert 'original failed receipt is preserved' in card
    assert '12-hour QASPER evaluation pool and all quality gates remain unchanged' in card
    assert 'not an uninterrupted original-cap run' in card
