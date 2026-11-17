"""Optional CPU checks of the hash-verified public bundle, no original run inputs."""
import importlib.util

import pytest
import torch

from scripts import setup_refusal_artifact as setup

pytestmark = pytest.mark.artifact(setup.DEST+'/refusal_runtime.py')


def test_verified_downloaded_vector_formats_and_zero_hook():
    root = setup.verify(setup.path(setup.DEST))
    spec = importlib.util.spec_from_file_location('public_refusal_bundle_cpu', root/'refusal_runtime.py')
    runtime = importlib.util.module_from_spec(spec); spec.loader.exec_module(runtime)
    raw = runtime.load_direction(root)
    assert raw.shape == (3584,) and raw.dtype == torch.float32
    assert torch.equal(raw, runtime.load_direction(root, use_pt=True))
    from types import SimpleNamespace
    model = torch.nn.Module(); model.model = torch.nn.Module()
    model.model.layers = torch.nn.ModuleList([torch.nn.Identity() for _ in range(28)])
    model.config = SimpleNamespace(model_type='qwen2', num_hidden_layers=28, hidden_size=3584,
                                   num_attention_heads=28, num_key_value_heads=4)
    values = torch.ones(1, 2, 3584)
    with runtime.steering(model, raw, strength=0):
        assert torch.equal(model.model.layers[18](values), values)
    assert not model.model.layers[18]._forward_hooks
