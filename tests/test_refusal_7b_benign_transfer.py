"""CPU fixtures only; no target-model/judge calls or study input generation."""
import copy
import json
from types import SimpleNamespace

import pytest

from scripts import refusal_7b_benign_transfer as run


def source_inputs():
    return [run.read_json(p) for p in ("results/xstest_baseline_01.json", "outputs/xstest-baseline-01/grades/accepted.json",
        "outputs/xstest-baseline-01/inputs/safe250.json", "outputs/xstest-stored-audit-01/review.json",
        "outputs/refusal-7b-transfer-01/inputs/reallocation.json", "data/refusal-01/prepared/development.json")]


def test_pinned_source13_seen4_reserved7_and_original67():
    run.transfer.check_hashes(run.SOURCE_HASHES)
    rows, inventory = run.derive_fixture(*source_inputs())
    assert [r['source_id'] for r in rows] == list(run.NEW) and len(rows) == 13
    assert inventory['seen4_ids'] == ['109', '162', '165', '171']
    assert inventory['new_reserved_ids_consumed'] == list(run.RESERVED)
    assert inventory['remaining_protected_count'] == 98
    assert inventory['questionable_id67_original_grade'] == 'P'
    assert inventory['review_supported_original_count'] == 16
    assert inventory['old_final_set_no_longer_untouched'] and inventory['consumed_old_final_ids'] == ['57', '64']
    assert all(r['label'] == 'safe' and r['family'] == r['type'] and r['id'] == 'xstest-'+r['source_id'] for r in rows)


def test_derive_rejects_source_label_or_review_changes():
    values = source_inputs()
    values[1]['ratings'][0]['grade'] = 'P' if values[1]['ratings'][0]['grade'] == 'C' else 'C'
    with pytest.raises(ValueError, match='accepted labels'): run.derive_fixture(*values)
    values = source_inputs()
    next(r for r in values[3]['rows'] if r['id'] == '67')['primary_label_interpretation'] = 'supported'
    with pytest.raises(ValueError, match='review mapping'): run.derive_fixture(*values)


@pytest.fixture
def synthetic_run(tmp_path, monkeypatch):
    original = run.read_json(run.transfer.CONFIG)
    cfg = run.profile(run.half.profile(original))
    monkeypatch.setattr(run, 'verify', lambda *a, **k: (cfg, {'raw_vector_sha256': 'synthetic'}))
    run.save(tmp_path/'manifest.json', {})
    rows = [{'id': 'xstest-'+i, 'source_id': i, 'instruction': 'Synthetic fixture '+i, 'label': 'safe', 'family': 'test'} for i in run.NEW]
    run.save(tmp_path/'inputs/fixture.json', {'items': rows})
    run.save(tmp_path/'inputs/inventory.json', {'synthetic': True})
    run.save(tmp_path/'inputs/rubric.json', run.read_json(original['rubric']))
    run.save(tmp_path/'inputs/generation_disclosure.json', {'synthetic': True})
    for name in ('m0_safe', 'subtract18_safe'):
        values = [{'id': row['id'], 'input_ids': [1, 2], 'rendered_prompt': row['instruction'], 'generated_ids': [3, 4],
                   'batch_ids': [r['id'] for r in rows[(i//8)*8:(i//8+1)*8]], 'batch_position': i%8,
                   'output': 'Synthetic response.', 'ended_eos': False, 'hit_token_limit': False}
                  for i, row in enumerate(rows)]
        run.save(tmp_path/'comparison'/(name+'.json'), {'predictions': values, 'settings': {'fixed': True}, 'pad_token_diagnostics': {}})
    run.save(tmp_path/'comparison/run.json', {'completed': True, 'manifest_sha256': run.stream_hash(tmp_path/'manifest.json'),
        'files': {str(p.relative_to(tmp_path/'comparison')): run.stream_hash(p) for p in (tmp_path/'comparison').rglob('*') if p.is_file()}})
    return tmp_path


def test_joint26_blind_packet_8_5_parity_and_no_reissue(synthetic_run):
    rows, predictions = run.comparison_inputs(synthetic_run)
    assert {len(p['batch_ids']) for p in predictions['m0_safe']} == {8, 5}
    run.packets(SimpleNamespace(), synthetic_run)
    packet = run.read_json(synthetic_run/'comparison/judging/packet.json')['items']
    keys = run.read_json(synthetic_run/'comparison/judging/keys.local.json')
    assert len(packet) == 26 and all(set(p) == {'id', 'prompt', 'response'} for p in packet)
    assert Counter_conditions(keys) == {'m0_safe': 13, 'subtract18_safe': 13}
    assert not {k['source_id'] for k in keys.values()} & {'xstest-'+i for i in run.SEEN}
    with pytest.raises(FileExistsError): run.packets(SimpleNamespace(), synthetic_run)


def Counter_conditions(keys):
    from collections import Counter
    return dict(Counter(k['condition'] for k in keys.values()))


@pytest.mark.parametrize('change', ['batch', 'input', 'settings'])
def test_input_or_8_5_membership_change_rejected(synthetic_run, monkeypatch, change):
    monkeypatch.setattr(run.transfer, 'completed', lambda *a: {})
    filename = synthetic_run/'comparison/subtract18_safe.json'; value = run.read_json(filename)
    if change == 'batch': value['predictions'][8]['batch_ids'] = ['different']
    elif change == 'input': value['predictions'][0]['input_ids'] = [99]
    else: value['settings'] = {'different': True}
    filename.write_text(json.dumps(value))
    with pytest.raises(ValueError, match='membership|parity'): run.comparison_inputs(synthetic_run)


def test_joint26_import_provenance_rejects_wrong_packet(synthetic_run, monkeypatch):
    run.packets(SimpleNamespace(), synthetic_run)
    out = synthetic_run/'comparison/judging'; keys = run.read_json(out/'keys.local.json')
    submission = {'judge': {'model_identifier': 'synthetic', 'exposed_settings': {}, 'fresh_context': True, 'advisor_overlap': False},
        'ratings': [{'id': i, 'refusal_class': 'full_compliance', 'unscorable': False, 'useful_and_correct': True} for i in keys]}
    judge = synthetic_run/'judge'; run.save(judge/'submission.json', submission)
    cfg, _ = run.verify(synthetic_run)
    monkeypatch.setattr(run.transfer.refusal_judge, 'verify_audit', lambda *a: {'binding': 'test'})
    run.save(judge/'provenance.json', {'source_sha256': {'wrong_packet': True}, 'submission_sha256': run.stream_hash(judge/'submission.json')})
    with pytest.raises(ValueError, match='provenance'):
        run.import_ratings(SimpleNamespace(ratings=str(judge/'submission.json')), synthetic_run)
    expected = {'binding': 'test', 'packet_file': run.stream_hash(out/'packet.json'),
                'judge_registration': run.stream_hash(run.path(cfg['judge_registration']))}
    (judge/'provenance.json').write_text(json.dumps({'source_sha256': expected, 'submission_sha256': run.stream_hash(judge/'submission.json')}))
    run.import_ratings(SimpleNamespace(ratings=str(judge/'submission.json')), synthetic_run)
    assert len(run.read_json(out/'ratings.json')['ratings']) == 26


def detail(identifier, refused=False, useful=True, invalid=False, capped=False):
    return {'id': identifier, 'class': 'full_refusal' if refused else 'full_compliance', 'refused': refused and not invalid,
            'valid': not invalid, 'valid_full_compliance': not refused and not invalid, 'useful_and_correct': useful and not invalid,
            'empty': invalid, 'degenerate': invalid, 'unscorable': invalid, 'truncated': capped}


def test_rescue_requires_useful_full_valid_and_harms_not_disjoint():
    before = [detail('a', True, False), detail('b', True, False), detail('c'), detail('d')]
    after = [detail('a'), detail('b', useful=False), detail('c', True, False), detail('d', invalid=True, capped=True)]
    report = run.paired_report(before, after)
    assert report['paired_useful_rescues']['usefully_rescued_ids'] == ['a']
    assert report['paired_useful_rescues']['newly_refused_ids'] == ['c']
    assert report['useful_to_not_useful_ids'] == ['c', 'd']
    assert report['candidate']['invalid_union'] == 1
    assert report['candidate']['empty'] == report['candidate']['degenerate'] == report['candidate']['unscorable'] == 1
    assert report['candidate']['truncated'] == 1


def test_summary_primary13_seen4_separate_no_rates_or_success(synthetic_run):
    run.packets(SimpleNamespace(), synthetic_run)
    out = synthetic_run/'comparison/judging'; keys = run.read_json(out/'keys.local.json')
    ratings = []
    for identifier, key in keys.items():
        baseline = key['condition'] == 'm0_safe'; first = key['source_id'] == 'xstest-14'
        refused = baseline and first
        ratings.append({'id': identifier, 'refusal_class': 'full_refusal' if refused else 'full_compliance',
                        'unscorable': False, 'useful_and_correct': not refused})
    run.save(out/'ratings.json', {'judge': {'model_identifier': 'synthetic', 'exposed_settings': {}, 'fresh_context': True, 'advisor_overlap': False}, 'ratings': ratings})
    before = [detail('xstest-'+i, i == '109', i != '109') for i in run.SEEN]
    after = [detail('xstest-'+i) for i in run.SEEN]
    run.save(synthetic_run/'reuse/seen4.json', {'m0_safe': before, 'subtract18_safe': after})
    run.summary(SimpleNamespace(output=str(synthetic_run/'summary.json')), synthetic_run)
    result = run.read_json(synthetic_run/'summary.json')
    assert result['primary13']['baseline']['total'] == 13
    assert result['seen4_descriptive_only']['baseline']['total'] == 4
    assert result['seen4_descriptive_only']['paired_useful_rescues']['usefully_rescued_ids'] == ['xstest-109']
    assert result['new_generations'] == result['new_judgments'] == 26
    text = (synthetic_run/'summary.json').read_text()
    assert 'benign_over_refusal_success' not in text and 'primary_refusal_rate' not in text
    assert 'Synthetic response.' not in text


def test_no_gpu_without_flag_and_fixed26_ceiling(monkeypatch, tmp_path):
    monkeypatch.setattr(run, 'verify', lambda *a, **k: pytest.fail('GPU/model verification before explicit flag'))
    with pytest.raises(ValueError, match='execute-gpu'): run.compare(SimpleNamespace(execute_gpu=False), tmp_path)
    cfg = run.profile(run.half.profile(run.read_json(run.transfer.CONFIG)))
    assert cfg['maximum_new_generations'] == 26 and cfg['raw_vector_scale'] == .5 and cfg['subtraction_coefficient'] == -.25


def test_no_stage_restart_preserves_interrupted_progress(tmp_path):
    run.save(tmp_path/'manifest.json', {})
    cfg = run.profile(run.half.profile(run.read_json(run.transfer.CONFIG)))
    with pytest.raises(RuntimeError):
        with run.transfer.stage(tmp_path, 'comparison', cfg) as out:
            run.save(out/'progress/first.result.json', {'synthetic': True}); raise RuntimeError('stop')
    assert run.read_json(tmp_path/'comparison/progress/first.result.json')['synthetic']
    with pytest.raises(FileExistsError):
        with run.transfer.stage(tmp_path, 'comparison', cfg): pass
