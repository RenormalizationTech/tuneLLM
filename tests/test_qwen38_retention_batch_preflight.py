import copy
import json
from pathlib import Path
from types import SimpleNamespace
import subprocess
import sys

import pytest

from scripts import qwen38_retention_batch_preflight as p
from scripts import qwen38_retention as r


class Tokens:
    pad_token_id, eos_token_id = 0, 1
    def __call__(self, text, **kwargs): return {'input_ids': [ord(c) + 10 for c in text]}
    def apply_chat_template(self, messages, **kwargs):
        assert kwargs['return_dict'] is False and kwargs['enable_thinking'] is False
        return 'user:' + messages[0]['content'] + '\nassistant:'


@pytest.fixture
def registered(tmp_path, monkeypatch):
    from scripts import qwen38_science_runtime as runtime
    monkeypatch.setattr(p, 'ROOT', tmp_path)
    monkeypatch.setattr(runtime, 'versions', lambda: {'synthetic': 'CPU-only'})
    monkeypatch.setattr(runtime, 'load_runtime', lambda *a, **kw: pytest.fail('No model load in CPU tests'))
    source = tmp_path / 'reference'
    source.mkdir()
    prepared = [{'id': 'raw-mc', 'task': 'general_mc', 'input_ids': [8] * 7000},
                {'id': 'length-only-gsm', 'task': 'gsm8k', 'input_ids': [9] * 333},
                {'id': 'length-only-instruction', 'task': 'ifeval', 'input_ids': [9] * 401}]
    r.write_new(source / 'prepared.json', prepared)
    manifest = tmp_path / 'results/qwen38_27b_nf4_artifact_01.json'
    manifest.parent.mkdir(); r.write_new(manifest, {'files': {}})
    r.write_new(source / 'registration.json', {'prepared_copy_sha256': r.sha(source / 'prepared.json'),
        'comparison_binding': {'prepared_sha256': r.digest(prepared), 'base_manifest_sha256': r.sha(manifest)}})
    for name in [*r.SOURCES, p.SOURCE]:
        path = tmp_path / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_text('# synthetic bound source\n')
    base = tmp_path / 'base'; base.mkdir(); (base / 'tokenizer.json').write_text('{}')
    args = SimpleNamespace(output=tmp_path / 'preflight', retention_run=source, base=base, gpu='1', max_seconds=600)
    result = p.prepare(args, Tokens())
    return args, result


def test_cpu_registration_four_conditions_synthetic_length_coverage_and_hashes(registered):
    args, result = registered
    assert not result['gpu_launched'] and result['conditions'] == 4 and result['maximum_frozen_prompt_tokens'] == 7000
    cfg, rows = p.validate(args.output)
    assert [(c['adapter'], c['batch_size']) for c in cfg['conditions']] == p.CONDITIONS
    assert all(c['groups'][0]['max_new_tokens'] == 32 for c in cfg['conditions'])
    assert [len(c['groups']) for c in cfg['conditions']] == [4, 1, 4, 1]
    assert len(rows) == 4 and min(len(x['input_ids']) for x in rows) < 7000
    assert cfg['maximum_frozen_generation_prompt_tokens'] == 401
    assert all(len(rows[i]['input_ids']) >= 7000 for i in (2, 3))
    assert 'length-only-gsm' not in json.dumps(rows) and 'reference' not in json.dumps(rows)
    assert not (args.output / 'result.json').exists()
    with pytest.raises(FileExistsError): p.prepare(args, Tokens())


@pytest.mark.parametrize('change', ['source', 'tokens', 'groups', 'cap', 'reference'])
def test_preflight_hash_and_runtime_registration_rejections(registered, change):
    args, _ = registered
    cfg_path = args.output / 'registration.json'
    cfg = json.loads(cfg_path.read_text())
    if change == 'source':
        path = Path(p.ROOT) / p.SOURCE; path.write_text('# changed source\n')
    elif change == 'tokens':
        path = args.output / 'prompts.json'; data = json.loads(path.read_text()); data[0]['input_ids'][0] += 1
        path.write_text(r.canonical(data))
    elif change == 'reference':
        Path(cfg['reference_prepared']).write_text('[]')
    else:
        if change == 'groups': cfg['conditions'][1]['groups'][0]['ids'].reverse()
        else: cfg['max_seconds'] = 601
        cfg_path.write_text(r.canonical(cfg))
    with pytest.raises(ValueError): p.validate(args.output)


def test_watchdog_kills_once_and_records_actual_resource_usage(registered, monkeypatch):
    args, _ = registered
    killed, commands = [], []
    class Process:
        pid = 123456
        waits = 0
        def __init__(self, command, **kwargs):
            commands.append(command)
            assert '--execute-gpu' in command and '--execute-worker' in command
            assert kwargs['env']['QWEN38_PREFLIGHT_CHILD'] == r.sha(args.output / 'registration.json')
        def wait(self, timeout=None):
            self.waits += 1
            if self.waits == 1:
                assert timeout == 600
                raise subprocess.TimeoutExpired('synthetic never-launched child', timeout)
            return -9
        def poll(self): return None
    monkeypatch.setattr(p.subprocess, 'Popen', Process)
    monkeypatch.setattr(p.os, 'killpg', lambda *args: killed.append(args))
    with pytest.raises(subprocess.TimeoutExpired): p.execute(args.output)
    assert len(commands) == 1 and len(killed) == 1
    result = json.loads((args.output / 'result.json').read_text())
    receipt = json.loads((args.output / 'resource_usage.json').read_text())
    assert not result['completed'] and 'TimeoutExpired' in result['error']
    assert receipt['parent_ledger_must_debit'] and receipt['cap_seconds'] == 600
    assert receipt['result_sha256'] == r.sha(args.output / 'result.json')
    with pytest.raises(FileExistsError): p.execute(args.output)
    assert len(commands) == 1


def test_cli_requires_explicit_gpu_and_watchdog_for_child(monkeypatch, tmp_path):
    r.write_new(tmp_path / 'registration.json', {})
    monkeypatch.setattr(sys, 'argv', ['preflight', '--execute-worker', '--run', str(tmp_path)])
    monkeypatch.delenv('QWEN38_PREFLIGHT_CHILD', raising=False)
    with pytest.raises(SystemExit): p.main()


def test_all_four_conditions_zero_adapter_and_real_token_timing_aggregates_on_cpu(registered, monkeypatch):
    from contextlib import nullcontext
    from scripts import qwen38_science_runtime as runtime
    args, _ = registered
    class Parameter:
        value = 7
        def zero_(self): self.value = 0
    parameter = Parameter()
    class Model:
        disabled = False
        def named_parameters(self): return [('model.lora_A.weight', parameter)]
        def eval(self): pass
        def disable_adapter(self):
            from contextlib import contextmanager
            @contextmanager
            def disabled():
                self.disabled = True
                try: yield
                finally: self.disabled = False
            return disabled()
    model = Model()
    torch = SimpleNamespace(no_grad=nullcontext, inference_mode=nullcontext,
                            cuda=SimpleNamespace(get_device_name=lambda index: 'synthetic CPU stub'))
    transformers = SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **kw: Tokens()),
        StoppingCriteria=object, StoppingCriteriaList=list, set_seed=lambda _: None)
    monkeypatch.setitem(sys.modules, 'torch', torch)
    monkeypatch.setitem(sys.modules, 'transformers', transformers)
    def load(*args, **kwargs):
        assert kwargs['rank'] == 16 and kwargs['alpha'] == 32 and kwargs['dropout'] == .05
        return model, {'synthetic': True}
    monkeypatch.setattr(runtime, 'load_runtime', load)
    calls = []
    def generate(model, tokenizer, members, plan, stopping):
        assert parameter.value == 0
        calls.append((model.disabled, len(members)))
        predicted = [{'id': row['id'], 'input_ids': row['input_ids'], 'generated_ids': [1], 'output': '',
                      'ended_with_eos': True, 'hit_token_limit': False} for row in members]
        return predicted, {**plan, 'batch_seconds': 2., 'peak_allocated_bytes': 10, 'peak_reserved_bytes': 20}
    monkeypatch.setattr(r, 'generate_group', generate)
    result = p.worker(args.output)
    assert result['completed'] and result['zero_adapter_tensors'] == 1
    assert calls == [(True, 1)] * 4 + [(True, 4)] + [(False, 1)] * 4 + [(False, 4)]
    assert [c['generated_tokens'] for c in result['conditions']] == [4] * 4
    assert [c['generation_seconds'] for c in result['conditions']] == [8., 2., 8., 2.]
    assert [c['seconds_per_token'] for c in result['conditions']] == [2., .5, 2., .5]
    assert all(c['completed'] for c in result['conditions'])
    assert all(c['exact_items'] == 4 for c in result['token_agreement_vs_disabled_batch1'])
    assert not list(args.output.rglob('*.safetensors'))
    assert 'descriptive only' in result['agreement_scope']
