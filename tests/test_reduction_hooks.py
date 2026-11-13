"""Actual random tiny Qwen3 on CPU, not downloaded models or experiment data."""
import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from scripts.reduction_hooks import intervene, hook_sites, projection_remove
from scripts.refusal_run import generation_position_ids


@pytest.fixture
def model():
    torch.manual_seed(42)
    config = Qwen3Config(vocab_size=47, hidden_size=16, intermediate_size=24,
                        num_hidden_layers=24, num_attention_heads=2, num_key_value_heads=1,
                        head_dim=8, max_position_embeddings=128, attention_dropout=0.0,
                        pad_token_id=0, eos_token_id=46, use_sliding_window=False)
    config._attn_implementation = "eager"
    return Qwen3ForCausalLM(config).eval().cpu()


def inputs():
    ids = torch.tensor([[0, 0, 3, 4], [5, 6, 7, 8]])
    mask = ids.ne(0).long()
    return {"input_ids": ids, "attention_mask": mask, "position_ids": generation_position_ids(mask)}


def forward(model, data):
    with torch.inference_mode():
        return model(**data, use_cache=False).logits


def hook_count(model):
    return sum(len(m._forward_hooks)+len(m._forward_pre_hooks) for m in model.modules())


def test_projection_algebra_fp32_and_activation_dtype():
    x = torch.tensor([[[1., 2., 3.], [4., 5., 6.]]])
    unit = torch.tensor([1., 1., 0.])/2**.5
    projected = projection_remove(x, unit)
    assert torch.max(torch.abs(projected @ unit)) < 1e-6
    torch.testing.assert_close(projection_remove(projected, unit), projected)
    assert projection_remove(x.bfloat16(), unit).dtype == torch.bfloat16
    assert torch.equal(x, torch.tensor([[[1., 2., 3.], [4., 5., 6.]]]))


@pytest.mark.parametrize("method", ["subtract23", "ablate_all"])
def test_real_hooks_identity_bitwise_and_exception_cleanup(model, method):
    data, vector = inputs(), torch.arange(1, 17).float()
    original = forward(model, data)
    state = {n: t.clone() for n, t in model.state_dict().items()}
    audit = {}
    with intervene(model, method, vector, identity=True, audit=audit):
        assert hook_count(model) > 0
        assert torch.equal(forward(model, data), original)
    assert hook_count(model) == 0 and audit["hooks_removed"]
    for name, value in state.items():
        assert torch.equal(model.state_dict()[name], value)
    with pytest.raises(RuntimeError, match="synthetic"):
        with intervene(model, method, vector):
            raise RuntimeError("synthetic exception")
    assert hook_count(model) == 0
    assert torch.equal(forward(model, data), original)


def test_subtract23_fractional_raw_output_all_positions_including_pads(model):
    vector, before, after = torch.arange(1, 17).float()/100, [], []
    def capture(values):
        return lambda m, a, o: values.append(o.detach().clone())
    first = model.model.layers[23].register_forward_hook(capture(before))
    try:
        with intervene(model, "subtract23", vector):
            last = model.model.layers[23].register_forward_hook(capture(after))
            try:
                forward(model, inputs())
            finally:
                last.remove()
    finally:
        first.remove()
    assert before[0].shape == (2, 4, 16)
    torch.testing.assert_close(after[0], before[0] - .5*vector)
    assert torch.all((before[0]-after[0]).abs().sum(-1) > 0)  # Pads too.


def test_projection_all_blocks_writers_mid_and_post_residuals(model):
    vector = torch.arange(1, 17).float()
    audit = {}
    with intervene(model, "ablate_all", vector, audit=audit):
        forward(model, inputs())
        forward(model, inputs())  # Audit samples once, counters still cover both.
    assert len(hook_sites(model, "ablate_all")) == 24*3
    assert len(audit["nodes"]) == 24*5
    assert all(r["prefill_calls"] == 2 for r in audit["nodes"].values())
    assert all(not r["nonfinite"] and r["max_abs_projection"] < 1e-5 for r in audit["nodes"].values())


@pytest.mark.parametrize("method", ["subtract23", "ablate_all"])
def test_actual_qwen3_left_padded_cache_vs_full_prefix(model, method):
    data, vector = inputs(), torch.linspace(-.1, .1, 16)
    audit = {}
    with intervene(model, method, vector, audit=audit), torch.inference_mode():
        cached = model(**data, use_cache=True)
        ids, mask = data["input_ids"], data["attention_mask"]
        for next_ids in (torch.tensor([[9], [10]]), torch.tensor([[11], [12]])):
            ids = torch.cat((ids, next_ids), -1)
            mask = torch.cat((mask, torch.ones_like(next_ids)), -1)
            positions = generation_position_ids(mask)
            decoded = model(input_ids=next_ids, attention_mask=mask, position_ids=positions[:, -1:],
                            past_key_values=cached.past_key_values, use_cache=True)
            full = model(input_ids=ids, attention_mask=mask, position_ids=positions, use_cache=False)
            torch.testing.assert_close(decoded.logits[:, -1], full.logits[:, -1], rtol=2e-5, atol=2e-6)
            cached = decoded
    assert all(r["prefill_calls"] == 3 and r["decode_calls"] == 2 for r in audit["nodes"].values())
    assert hook_count(model) == 0


def test_invalid_direction_and_failed_partial_install_cleanup(model, monkeypatch):
    with pytest.raises(ValueError):
        with intervene(model, "ablate_all", torch.zeros(16)):
            pass
    with pytest.raises(ValueError):
        with intervene(model, "subtract23", torch.ones(15)):
            pass
    def fail(*args, **kwargs):
        raise RuntimeError("synthetic registration failure")
    monkeypatch.setattr(model.model.layers[1], "register_forward_pre_hook", fail)
    with pytest.raises(RuntimeError):
        with intervene(model, "ablate_all", torch.ones(16)):
            pass
    assert hook_count(model) == 0
