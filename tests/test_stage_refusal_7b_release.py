"""Essential CPU serialization, tensor-layout parity and cleanup checks."""
from types import SimpleNamespace
import importlib.util
import json

import pytest
import torch

from scripts import stage_refusal_7b_release as stage
from scripts import refusal_release_runtime as runtime
from scripts.refusal_7b_transfer_hooks import intervene


@pytest.fixture(scope='module')
def raw():
    file = stage.path(stage.read_json(stage.SETTING)['direction_file'])
    if not file.is_file():
        pytest.skip('artifact: original private vector absent; use downloaded-bundle tests for public inference artifacts')
    assert stage.sha256(file) == runtime.PT_SHA256
    return runtime.validate_direction(torch.load(file, map_location='cpu', weights_only=True))


class Block(torch.nn.Module):
    def __init__(self, tuple_output):
        super().__init__(); self.tuple_output = tuple_output
    def forward(self, x, marker=None):
        y = 3*x
        return (y, marker, None) if self.tuple_output else y


def model(tuple_output):
    base = torch.nn.Module(); base.layers = torch.nn.ModuleList([Block(tuple_output) for _ in range(28)])
    m = torch.nn.Module(); m.model = base
    m.config = SimpleNamespace(model_type='qwen2', num_hidden_layers=28, hidden_size=3584,
                               num_attention_heads=28, num_key_value_heads=4)
    return m


@pytest.mark.parametrize('tuple_output,dtype,width', [(False, torch.float32, 5), (True, torch.bfloat16, 5), (True, torch.bfloat16, 1)])
def test_exact_old_halfvector_hook_parity_and_zero_cleanup(raw, tuple_output, dtype, width):
    m = model(tuple_output); marker = object()
    x = torch.randn(2, width, 3584, generator=torch.Generator().manual_seed(42)).to(dtype); original = x.clone()
    def tensor(output): return output[0] if isinstance(output, tuple) else output
    with intervene(m, 'subtract18', .5*raw): old = m.model.layers[18](x, marker)
    with runtime.steering(m, raw): new = m.model.layers[18](x, marker)
    assert torch.equal(tensor(old), tensor(new))
    assert tensor(new).dtype == x.dtype and tensor(new).device == x.device
    if tuple_output: assert new[1] is marker and new[2] is None
    assert torch.equal(x, original)
    with runtime.steering(m, raw, strength=0): identity = m.model.layers[18](x, marker)
    assert torch.equal(tensor(identity), 3*x)
    with pytest.raises(RuntimeError):
        with runtime.steering(m, raw): raise RuntimeError('CPU exception')
    assert not m.model.layers[18]._forward_hooks
    assert torch.equal(tensor(m.model.layers[18](x, marker)), 3*x)


def test_model_revision_config_and_direction_rejection(raw):
    m = model(False)
    with pytest.raises(ValueError, match='revision'): runtime.validate_model(m, revision='wrong')
    with pytest.raises(ValueError, match='ID'): runtime.validate_model(m, model_id='wrong')
    m.config.hidden_size = 8
    with pytest.raises(ValueError, match='config'): runtime.validate_model(m)
    with pytest.raises(ValueError): runtime.validate_direction(raw[:2])
    with pytest.raises(ValueError): runtime.validate_direction(torch.zeros_like(raw))


def test_staging_exact_allowlist_safe_roundtrip_and_no_overwrite(tmp_path, raw):
    dest, mf = tmp_path/'payload', tmp_path/'manifest.json'
    manifest = stage.stage(dest, mf)
    assert manifest['repo_id'] == stage.REPO and manifest['private'] is False
    assert set(manifest['files']) == stage.ALLOWLIST and manifest['cpu_checks']['safetensors_exact']
    spec = importlib.util.spec_from_file_location('staged_refusal_runtime_cpu', dest/'refusal_runtime.py')
    packaged = importlib.util.module_from_spec(spec); spec.loader.exec_module(packaged)
    assert torch.equal(packaged.load_direction(dest), raw)
    assert torch.equal(packaged.load_direction(dest, use_pt=True), raw)
    assert stage.sha256(dest/'direction.pt') == runtime.PT_SHA256
    for target, source in stage.RESULTS.items():
        assert (dest/'results'/target).read_bytes() == stage.path(source).read_bytes()
    assert '/home/' not in (dest/'provenance.json').read_text()
    assert 'All-block projection' in (dest/'README.md').read_text()
    assert 'One repetitive output' in (dest/'README.md').read_text()
    with pytest.raises(FileExistsError): stage.stage(dest, mf)
    (dest/'extra.txt').write_text('unexpected')
    with pytest.raises(ValueError, match='allowlist'): stage.preflight(dest, manifest)


def test_pt_hash_checked_before_deserialization(tmp_path, monkeypatch):
    (tmp_path/'direction.pt').write_bytes(b'corrupt')
    monkeypatch.setattr(torch, 'load', lambda *a, **k: pytest.fail('hash failure must precede load'))
    with pytest.raises(ValueError, match='hash'): runtime.load_direction(tmp_path, use_pt=True)
