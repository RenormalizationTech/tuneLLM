"""CPU inspection/help/argument checks only; no model allocation or GPU calls."""
import copy
import json
from pathlib import Path
import subprocess
import sys

import pytest

from scripts import qwen38_nf4_smoke as smoke


def test_runtime_loading_and_dtype_sets_are_serializable(tmp_path):
    target = tmp_path/'record.json'
    smoke.write(target, {'loading': {'unexpected_keys': set()},
                         'keep_in_fp32_modules': {'dt_bias', 'A_log'}})
    assert json.loads(target.read_text()) == {
        'loading': {'unexpected_keys': []}, 'keep_in_fp32_modules': ['A_log', 'dt_bias']}


def test_cache_numerical_difference_is_preserved_as_diagnostic():
    from types import SimpleNamespace
    import torch
    outputs = [SimpleNamespace(logits=torch.tensor([[[2., 0.]]]), past_key_values=object()),
               SimpleNamespace(logits=torch.tensor([[[1., .1]]])),
               SimpleNamespace(logits=torch.tensor([[[1., .3]]]))]
    def model(**kwargs):
        return outputs.pop(0)
    inputs = {'input_ids': torch.ones(1, 2, dtype=torch.long),
              'attention_mask': torch.ones(1, 2, dtype=torch.long)}
    record = smoke.hybrid_check(model, inputs, {'cache_check': {'rtol': .02, 'atol': .05}})
    assert record['finite'] and record['argmax_matches']
    assert record['allclose'] is False and record['max_abs_delta'] > .19


def test_fixed_profile_and_hybrid_geometry():
    cfg = smoke.read(smoke.CONFIG); smoke.validate_config(cfg)
    plan = smoke.inspect(cfg)
    assert plan['text_layers'] == 64 and plan['hidden_size'] == 5120
    assert plan['layer_types'] == {'linear_attention': 48, 'full_attention': 16}
    assert plan['inspection_only'] and not plan['model_loaded']
    assert cfg['hf_home'] == '.hf' and cfg['max_new_tokens'] == 64 and cfg['batch_size'] == 1
    for key, value in [('revision', 'bad'), ('enable_thinking', True), ('batch_size', 8), ('gpu', '0,1'), ('max_new_tokens', 128)]:
        with pytest.raises(ValueError): smoke.validate_config({**cfg, key: value})


def test_installed_official_text_config_and_conversion_source():
    root = smoke.local('outputs/qwen38-27b-nf4-01/env/lib/python3.13/site-packages/transformers')
    if not root.is_dir(): pytest.skip('root-created isolated environment unavailable')
    support = smoke.official_support(root)
    assert support['available'] and support['loader'] == 'Qwen3_5ForCausalLM'
    assert support['config_class'] == 'Qwen3_5TextConfig'
    assert len(support['source_sha256']) == 3
    assert 'model.language_model.* -> model.*' in support['official_key_mapping']


def test_unsupported_source_is_reported_not_imported(tmp_path):
    assert smoke.official_support(tmp_path)['available'] is False


def test_expected_vision_mtp_skips_but_language_failures_rejected():
    info = {'missing_keys': [], 'mismatched_keys': [], 'unexpected_keys': ['model.visual.patch.weight', 'mtp.layers.0.weight'], 'error_msgs': []}
    assert smoke.validate_loading(info)['expected_skipped_vision_mtp'] == info['unexpected_keys']
    for key, value in [('missing_keys', ['model.layers.0.mlp.up_proj.weight']), ('mismatched_keys', ['lm_head.weight']),
                       ('unexpected_keys', ['model.language_model.layers.0.weight']), ('error_msgs', ['failure'])]:
        changed = copy.deepcopy(info); changed[key] = value
        with pytest.raises(ValueError, match='LANGUAGE'): smoke.validate_loading(changed)


def test_saved_reload_never_supplies_new_quantization_config():
    cfg = smoke.read(smoke.CONFIG)
    kwargs = smoke.loading_kwargs(cfg, saved=True)
    assert 'quantization_config' not in kwargs and 'config' not in kwargs
    assert kwargs['local_files_only'] and kwargs['trust_remote_code'] is False
    assert kwargs['device_map'] == {'': 'cuda:0'}


def test_artifact_paths_only_and_no_gpu_without_flag(tmp_path, monkeypatch):
    monkeypatch.setattr(smoke, 'ROOT', tmp_path)
    assert smoke.artifact('artifacts/experimental-nf4') == tmp_path/'artifacts/experimental-nf4'
    for value in ('outputs/nf4', 'artifacts', 'artifacts/../../outside'):
        with pytest.raises(ValueError): smoke.artifact(value)
    with pytest.raises(ValueError, match='execute-gpu'):
        smoke.execute(type('Args', (), {'execute_gpu': False})(), {})


def test_help_and_inspect_are_cpu_only_and_no_output_writes(tmp_path):
    code = "import runpy,sys; sys.argv=['qwen38_nf4_smoke','--inspect']; runpy.run_module('scripts.qwen38_nf4_smoke',run_name='__main__'); assert 'torch' not in sys.modules"
    inspected = subprocess.run([sys.executable, '-B', '-c', code], cwd=smoke.ROOT, capture_output=True, text=True, check=True)
    assert '"model_loaded": false' in inspected.stdout
    helped = subprocess.run([sys.executable, '-B', '-m', 'scripts.qwen38_nf4_smoke', '--help'], cwd=smoke.ROOT, capture_output=True, text=True, check=True)
    assert '--load-saved' in helped.stdout and '--compare-reload' in helped.stdout
