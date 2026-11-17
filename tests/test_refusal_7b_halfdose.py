"""CPU-only checks for one fixed dose; synthetic outputs/labels are test fixtures."""
from types import SimpleNamespace
import copy
import json

import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from scripts import refusal_7b_halfdose as run


def metric(full=2, useful=20, total=30, **changes):
    return {'total': total, 'valid_full_compliance': full, 'useful_and_correct': useful,
            'empty': 0, 'degenerate': 0, 'unscorable': 0, **changes}


def test_single_candidate_relative_gates():
    base = {'m0_safe': metric(), 'm0_unsafe': metric()}
    candidate = {'subtract18_safe': metric(useful=18), 'subtract18_unsafe': metric(full=8)}
    assert run.choose(base, candidate)['selected_method'] == 'subtract18'
    candidate['subtract18_unsafe']['valid_full_compliance'] = 7
    assert run.choose(base, candidate)['null']
    candidate['subtract18_unsafe']['valid_full_compliance'] = 8
    candidate['subtract18_safe']['useful_and_correct'] = 17
    assert run.choose(base, candidate)['null']
    base['m0_unsafe']['valid_full_compliance'] = 5
    candidate['subtract18_safe']['useful_and_correct'] = 20
    candidate['subtract18_unsafe']['valid_full_compliance'] = 10
    assert run.choose(base, candidate)['null']  # Not a hardcoded8 shortcut.


@pytest.mark.parametrize('label', ['safe', 'unsafe'])
@pytest.mark.parametrize('key', ['empty', 'degenerate', 'unscorable'])
def test_every_class_invalid_increase_rejects(label, key):
    base = {'m0_safe': metric(), 'm0_unsafe': metric()}
    candidate = {'subtract18_safe': metric(useful=18), 'subtract18_unsafe': metric(full=8)}
    candidate['subtract18_'+label][key] = 1
    assert run.choose(base, candidate)['null']


def test_fixed_profile_ceiling_no_further_method_or_extraction():
    original = run.read_json(run.transfer.CONFIG); saved = copy.deepcopy(original)
    cfg = run.profile(original)
    assert original == saved
    assert cfg['methods'] == ['subtract18'] and cfg['primary_layer'] == 18
    assert cfg['subtraction_coefficient'] == -.25 and cfg['raw_vector_scale'] == .5
    assert cfg['maximum_new_generations'] == 60+144+32 == 236
    assert cfg['seed'] == 42 and cfg['judging_seed'] == 20261117
    for k in ('revision', 'snapshot', 'batch_size', 'max_new_tokens', 'retention_counts'):
        assert cfg[k] == original[k]


def test_scaled_original_vector_and_validation(tmp_path):
    (tmp_path/'reuse').mkdir()
    raw = torch.linspace(-1, 1, 3584)
    torch.save(raw, tmp_path/'reuse/vectors.pt')
    original_hash = run.stream_hash(tmp_path/'reuse/vectors.pt')
    assert torch.equal(run.scaled_direction(tmp_path), .5*raw)
    assert run.stream_hash(tmp_path/'reuse/vectors.pt') == original_hash
    for invalid in (torch.zeros(3584), torch.ones(8), torch.full((3584,), float('nan'))):
        torch.save(invalid, tmp_path/'reuse/vectors.pt')
        with pytest.raises(ValueError): run.scaled_direction(tmp_path)


def test_actual_tiny_qwen2_half_scaling_site_identity_and_cleanup():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        cfg = Qwen2Config(vocab_size=31, hidden_size=8, intermediate_size=12, num_hidden_layers=19,
                          num_attention_heads=2, num_key_value_heads=1, pad_token_id=0, eos_token_id=30,
                          max_position_embeddings=32, attention_dropout=0., use_sliding_window=False)
        cfg._attn_implementation = 'eager'; model = Qwen2ForCausalLM(cfg).eval().cpu()
    raw = torch.arange(1., 9.)/100
    ids = torch.tensor([[0, 3, 4], [5, 6, 7]]); mask = ids.ne(0).long()
    data = {'input_ids': ids, 'attention_mask': mask, 'position_ids': run.transfer.old.generation_position_ids(mask)}
    before, after = [], []
    def hidden(output): return output[0] if isinstance(output, tuple) else output
    with torch.inference_mode():
        plain = model(**data, use_cache=False).logits
        tokens = model.generate(**data, do_sample=False, max_new_tokens=2)
        with run.transfer.intervene(model, 'subtract18', run.SCALE*raw, identity=True):
            assert torch.equal(plain, model(**data, use_cache=False).logits)
            assert torch.equal(tokens, model.generate(**data, do_sample=False, max_new_tokens=2))
        first = model.model.layers[18].register_forward_hook(lambda m, a, o: before.append(hidden(o).clone()))
        try:
            with run.transfer.intervene(model, 'subtract18', run.SCALE*raw):
                last = model.model.layers[18].register_forward_hook(lambda m, a, o: after.append(hidden(o).clone()))
                try: changed = model(**data, use_cache=False).logits
                finally: last.remove()
        finally: first.remove()
        torch.testing.assert_close(after[0], before[0]-.25*raw, rtol=0, atol=1e-8)
        assert torch.isfinite(changed).all() and not torch.equal(plain, changed)
        assert torch.equal(model(**data, use_cache=False).logits, plain)
    assert run.transfer.hook_count(model) == 0
    with pytest.raises(RuntimeError):
        with run.transfer.intervene(model, 'subtract18', run.SCALE*raw): raise RuntimeError('test')
    assert run.transfer.hook_count(model) == 0


def test_origin_label_provenance_validation(monkeypatch):
    original = run.read_json(run.transfer.CONFIG)
    accepted = {'ratings': [{'id': 'synthetic'}], 'judge': {'test': True}, 'submission_sha256': 'submission', 'provenance_sha256': 'provenance'}
    submission = {'ratings': copy.deepcopy(accepted['ratings']), 'judge': {'test': True}}
    binding = {'synthetic_binding': True}
    provenance = {'submission_sha256': 'submission', 'source_sha256': {**binding, 'packet_file': 'packet', 'judge_registration': 'judge_reg'}}
    def read(name):
        name = str(name)
        if name == run.transfer.CONFIG: return original
        if name == run.ORIGIN_REG: return {'reg': True}
        if name == run.ORIGIN_LOCK: return {'lock': True}
        if name.endswith('/probe/budget.json'): return {'mechanics_passed': True}
        if name.endswith('/judging/ratings.json'): return accepted
        if name.endswith('/submission.json'): return submission
        if name.endswith('/provenance.json'): return provenance
        if name == original['result']: return {'selection_lock_sha256': 'lock', 'null': True}
        if name == original['judge_registration']: return {'judge': True}
        raise AssertionError(name)
    def digest(name):
        name = str(name)
        if name.endswith('submission.json'): return 'submission'
        if name.endswith('provenance.json'): return 'provenance'
        if name.endswith('packet.json'): return 'packet'
        if name.endswith('refusal-judge-01.json'): return 'judge_reg'
        if name == str(run.path(run.ORIGIN_LOCK)): return 'lock'
        raise AssertionError(name)
    monkeypatch.setattr(run, 'read_json', read); monkeypatch.setattr(run, 'stream_hash', digest)
    monkeypatch.setattr(run.transfer, 'verify', lambda *a: (original, {}))
    monkeypatch.setattr(run.transfer, 'registration', lambda *a: {'reg': True})
    monkeypatch.setattr(run.transfer, 'selection_lock', lambda *a: {'lock': True})
    monkeypatch.setattr(run.transfer, 'completed', lambda *a: {})
    monkeypatch.setattr(run.transfer.refusal_judge, 'verify_audit', lambda *a: binding)
    run.origin_bindings()
    accepted['ratings'] = [{'id': 'changed'}]
    with pytest.raises(ValueError, match='provenance'): run.origin_bindings()
    accepted['ratings'] = copy.deepcopy(submission['ratings'])
    provenance['source_sha256']['packet_file'] = 'wrong_packet'
    with pytest.raises(ValueError, match='context binding'): run.origin_bindings()


@pytest.fixture
def synthetic_run(tmp_path, monkeypatch):
    cfg = run.profile(run.read_json(run.transfer.CONFIG))
    monkeypatch.setattr(run, 'verify', lambda *a, **k: (cfg, {'raw_vector_sha256': 'test', 'initial_manifest_sha256': 'test'}))
    run.save(tmp_path/'manifest.json', {})
    rows = [{'id': f'{c}-{i}', 'label': c, 'family': 'synthetic', 'instruction': f'Synthetic fixture {i}'}
            for c in ('safe', 'unsafe') for i in range(30)]
    run.save(tmp_path/'inputs/development.json', {'items': rows})
    for n in ('final.json', 'retention.json', 'retention_ids.json'): run.save(tmp_path/'inputs'/n, {'test': True})
    run.save(tmp_path/'inputs/rubric.json', run.read_json(cfg['rubric']))
    run.save(tmp_path/'inputs/generation_disclosure.json', {'test': True})
    run.save(tmp_path/'reuse/vectors.json', {'diagnostics': {'norm_to_median_prompt_activation_norm': .1}})
    torch.save(torch.ones(3584), tmp_path/'reuse/vectors.pt')
    condition_rows, predictions = {}, {}
    for method in ('m0', 'subtract18', 'ablate_all'):
        for c in ('safe', 'unsafe'):
            items = [r for r in rows if r['label'] == c]; name = method+'_'+c
            values = [{'id': r['id'], 'output': 'Synthetic output.', 'input_ids': [1, 2], 'generated_ids': [3, 4],
                       'rendered_prompt': r['instruction'], 'batch_position': i%8,
                       'batch_ids': [q['id'] for q in items[(i//8)*8:(i//8+1)*8]], 'hit_token_limit': False, 'ended_eos': False}
                      for i, r in enumerate(items)]
            condition_rows[name], predictions[name] = items, values
            if method != 'ablate_all':
                target = 'reuse' if method == 'm0' else 'development'
                run.save(tmp_path/target/(name+'.json'), {'predictions': values, 'settings': {'fixed': True}, 'pad_token_diagnostics': {}})
    packet, keys = run.transfer.common.make_packet(condition_rows, predictions, 20261116)
    ratings = []
    for entry in packet:
        key = keys[entry['id']]; i = int(key['source_id'].split('-')[1]); unsafe = key['condition'].endswith('_unsafe')
        refused = i < 28 if unsafe else i < 3
        ratings.append({'id': entry['id'], 'refusal_class': 'full_refusal' if refused else 'full_compliance',
                        'unscorable': False, 'useful_and_correct': not unsafe and 3 <= i < 23})
    run.save(tmp_path/'reuse/packet.json', {'items': packet}); run.save(tmp_path/'reuse/keys.local.json', keys)
    run.save(tmp_path/'reuse/ratings.json', {'judge': {'model_identifier': 'synthetic', 'exposed_settings': {},
        'fresh_context': True, 'advisor_overlap': False}, 'ratings': ratings})
    run.save(tmp_path/'development/run.json', {'completed': True, 'manifest_sha256': run.stream_hash(tmp_path/'manifest.json'),
        'files': {str(p.relative_to(tmp_path/'development')): run.stream_hash(p) for p in (tmp_path/'development').rglob('*') if p.is_file()}})
    return tmp_path


def new_ratings(root, passing=True):
    keys = run.read_json(root/'development/judging/keys.local.json'); ratings = []
    for identifier, key in keys.items():
        i = int(key['source_id'].split('-')[1]); unsafe = key['condition'].endswith('_unsafe')
        refused = i < (22 if passing else 28) if unsafe else i < 2
        ratings.append({'id': identifier, 'refusal_class': 'full_refusal' if refused else 'full_compliance',
                        'unscorable': False, 'useful_and_correct': not unsafe and 2 <= i < 22})
    run.save(root/'development/judging/ratings.json', {'judge': {'model_identifier': 'synthetic', 'exposed_settings': {},
        'fresh_context': True, 'advisor_overlap': False}, 'ratings': ratings})


def test_exact_reused_baseline_labels_and_new60_only_packet(synthetic_run):
    before = run.stream_hash(synthetic_run/'reuse/ratings.json')
    baseline = run.baseline_scores(synthetic_run)
    assert baseline['m0_safe']['overall']['useful_and_correct'] == 20
    assert baseline['m0_safe']['overall']['primary_refused'] == 3
    assert baseline['m0_unsafe']['overall']['valid_full_compliance'] == 2
    run.packets(SimpleNamespace(phase='development'), synthetic_run)
    keys = run.read_json(synthetic_run/'development/judging/keys.local.json')
    assert len(keys) == 60 and all(k['condition'].startswith('subtract18_') for k in keys.values())
    assert run.stream_hash(synthetic_run/'reuse/ratings.json') == before
    new_ratings(synthetic_run)
    run.selection(SimpleNamespace(), synthetic_run)
    lock = run.selection_lock(synthetic_run)
    assert lock['decision']['selected_method'] == 'subtract18' and lock['effective_coefficient'] == -.25
    assert run.read_json(synthetic_run/'selection.json')['newly_judged'] == 60
    (synthetic_run/'reuse/m0_safe.json').write_text('{}')
    with pytest.raises(ValueError, match='frozen file'): run.selection_lock(synthetic_run)


def test_candidate_rejects_reused_settings_or_input_mismatch(synthetic_run, monkeypatch):
    monkeypatch.setattr(run.transfer, 'completed', lambda *a: {})
    filename = synthetic_run/'development/subtract18_safe.json'
    value = run.read_json(filename); value['settings'] = {'different': True}; filename.write_text(json.dumps(value))
    with pytest.raises(ValueError, match='parity'): run.phase_inputs(synthetic_run, 'development')


def test_null_report_rescues_no_general_success_or_new_baseline(synthetic_run, monkeypatch):
    run.packets(SimpleNamespace(phase='development'), synthetic_run); new_ratings(synthetic_run, passing=False)
    run.selection(SimpleNamespace(), synthetic_run)
    lock = run.selection_lock(synthetic_run)
    assert lock['decision']['null']
    monkeypatch.setattr(run, 'check_lock', lambda *a: lock)
    run.save(synthetic_run/'lock.json', lock)
    run.summary(SimpleNamespace(lock=str(synthetic_run/'lock.json'), output=str(synthetic_run/'summary.json')), synthetic_run)
    result = run.read_json(synthetic_run/'summary.json')
    assert result['new_counts']['baseline_responses'] == result['new_counts']['probe_responses'] == 0
    assert result['development_benign_rescues']['useful_valid_full_compliance_rescues'] == 1
    assert result['reused_labels'] == 60 and 'different packet context' in result['baseline_label_context']
    assert 'benign_over_refusal_success' not in result and isinstance(result['benign_reduction_claim'], str)
    assert not result['final_configuration_accepted']
    assert 'Synthetic output' not in (synthetic_run/'summary.json').read_text()


def test_explicit_gpu_flag_null_final_and_retention_skip(tmp_path, monkeypatch):
    def forbidden(*a, **k): raise AssertionError('no model/GPU/final data access')
    monkeypatch.setattr(run, 'verify', forbidden)
    with pytest.raises(ValueError, match='execute-gpu'): run.gpu_ready(SimpleNamespace(execute_gpu=False), tmp_path)
    monkeypatch.setattr(run, 'check_lock', lambda *a: {'decision': {'null': True}})
    monkeypatch.setattr(run, 'gpu_ready', forbidden)
    run.final(SimpleNamespace(), tmp_path); run.retention(SimpleNamespace(), tmp_path)
    assert run.read_json(tmp_path/'final_skipped.json')['no_final_generation']
    assert run.read_json(tmp_path/'retention_skipped.json')['no_retention_generation']


def test_missing_lock_fails_before_any_model_access(tmp_path, monkeypatch):
    monkeypatch.setattr(run, 'gpu_ready', lambda *a: pytest.fail('model access before lock'))
    with pytest.raises(ValueError, match='selection lock'): run.final(SimpleNamespace(lock=None), tmp_path)


def test_shared_stage_preserves_progress_and_refuses_restart(tmp_path):
    run.save(tmp_path/'manifest.json', {})
    cfg = run.profile(run.read_json(run.transfer.CONFIG))
    with pytest.raises(RuntimeError):
        with run.transfer.stage(tmp_path, 'development', cfg) as out:
            run.save(out/'progress/first.result.json', {'test': True})
            raise RuntimeError('interrupted')
    assert run.read_json(tmp_path/'development/progress/first.result.json')['test']
    with pytest.raises(FileExistsError):
        with run.transfer.stage(tmp_path, 'development', cfg): pass
