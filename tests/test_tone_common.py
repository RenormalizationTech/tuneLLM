"""CPU-only synthetic hook and gate tests; no pretrained model generation."""
import copy
from types import SimpleNamespace

import pytest
import torch

from scripts.common import path, read_json
from scripts.tone_common import (decoded_ratings, degenerate, gates, grouped_interval, literal_checks,
    make_packets, mean_direction, measurements, response_encoding, response_means, select, steer, validate_ratings)
from scripts.tone_fixture import authored, validate


class CharacterTokenizer:
    eos_token, eos_token_id, all_special_ids = "!", 33, [33]
    def apply_chat_template(self, *args, **kwargs):
        return "T:"
    def __call__(self, text, **kwargs):
        result = {"input_ids": list(map(ord, text))}
        if kwargs.get("return_offsets_mapping"):
            result["offset_mapping"] = [(i, i+1) for i in range(len(text))]
        return result


def test_response_mask_excludes_template_eos_and_boundary_merge():
    encoded = response_encoding(CharacterTokenizer(), {"user": "test"}, "abc", "neutral")
    assert encoded["response_mask"] == [False, False, True, True, True, False]
    with pytest.raises(ValueError):
        response_encoding(CharacterTokenizer(), {"user": "test"}, "abc!", "neutral")
    class Crossing(CharacterTokenizer):
        def __call__(self, text, **kwargs):
            result = super().__call__(text, **kwargs)
            if kwargs.get("return_offsets_mapping"):
                result["offset_mapping"][1] = (1, 3)
            return result
    with pytest.raises(ValueError, match="crosses"):
        response_encoding(Crossing(), {"user": "test"}, "abc", "neutral")


def test_actual_cached_qwen_tokenizer_offsets_without_model():
    cfg = read_json("configs/tone_steering.json")
    snapshot = path(".hf/hub/models--Qwen--Qwen3-4B/snapshots") / cfg["revision"]
    if not (snapshot / "tokenizer.json").is_file():
        pytest.skip("optional local tokenizer integration probe; never downloads")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    for text in ("Boundary probe.", " Boundary probe.", "\nBoundary probe.", "café — 2!"):
        encoded = response_encoding(tokenizer, {"user": "Repeat supplied test text."}, text, cfg["neutral_system"])
        ids = [i for i, selected in zip(encoded["input_ids"], encoded["response_mask"]) if selected]
        assert tokenizer.decode(ids).strip() == text.strip()
        assert not encoded["response_mask"][-1]


def test_pair_means_equal_weight_and_raw_difference():
    pairs = [(torch.tensor([1., 3.]), torch.tensor([3., 5.])),
             (torch.tensor([2., 8.]), torch.tensor([8., 8.]))]
    vector, report = mean_direction(pairs)
    assert torch.equal(vector, torch.tensor([4., 1.]))
    assert report["pair_count"] == 2 and report["norm"] == vector.norm().item()


class IdentityModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList([torch.nn.Identity()])
    def forward(self, input_ids, attention_mask, cache_position=None):
        return self.model.layers[0](input_ids.float().unsqueeze(-1).expand(-1, -1, 2))


def test_hook_padding_decode_zero_and_exception_cleanup():
    model, vector = IdentityModel(), torch.tensor([1., 2.])
    ids, mask = torch.tensor([[0, 5, 6], [7, 8, 0]]), torch.tensor([[0, 1, 1], [1, 1, 0]])
    baseline = model(ids, mask)
    with steer(model, 0, vector, 2):
        result = model(input_ids=ids, attention_mask=mask, cache_position=torch.arange(3))
        delta = result-baseline
        assert torch.equal(delta[0, 2], vector*2) and torch.equal(delta[1, 1], vector*2)
        assert delta[0, :2].eq(0).all() and delta[1, 2].eq(0).all()
        decoded = model(input_ids=ids[:, :1], attention_mask=torch.ones(2, 4), cache_position=torch.tensor([3]))
        raw = ids[:, :1].float().unsqueeze(-1).expand(-1, -1, 2)
        assert torch.equal(decoded-raw, (vector*2).expand(2, 1, 2))
    assert not model._forward_pre_hooks and not model.model.layers[0]._forward_hooks
    with steer(model, 0, vector, 0):
        assert torch.equal(model(ids, mask), baseline)
        assert not model._forward_pre_hooks
    with pytest.raises(RuntimeError):
        with steer(model, 0, vector, 1):
            raise RuntimeError("synthetic failure")
    assert not model._forward_pre_hooks and not model.model.layers[0]._forward_hooks
    means = {}
    with response_means(model, [0], torch.tensor([[False, True, False], [True, False, False]]), means):
        model(ids, mask)
    assert torch.equal(means[0], torch.tensor([6., 6.]))


def test_random_qwen_cached_steering_matches_full_prefix_reference():
    from transformers import Qwen3Config, Qwen3ForCausalLM
    torch.manual_seed(4)
    config = Qwen3Config(vocab_size=32, hidden_size=24, intermediate_size=48, num_hidden_layers=3,
                        num_attention_heads=4, num_key_value_heads=2, head_dim=6,
                        max_position_embeddings=64, pad_token_id=0, eos_token_id=None, attention_dropout=0.)
    model = Qwen3ForCausalLM(config).eval()
    ids, mask = torch.tensor([[0, 0, 5, 6], [0, 7, 8, 9]]), torch.tensor([[0, 0, 1, 1], [0, 1, 1, 1]])
    weights = {n: p.detach().clone() for n, p in model.named_parameters()}
    vector, cache = torch.randn(24)*.02, None
    prefix, fullmask = ids.clone(), mask.clone()
    audit = {}
    for step in range(4):
        with torch.inference_mode(), steer(model, 1, vector, 1, audit):
            cached = model(input_ids=prefix if step == 0 else prefix[:, -1:], attention_mask=fullmask,
                           past_key_values=cache, use_cache=True)
        cache = cached.past_key_values
        def reference_hook(module, args, output):
            hidden = output[0] if isinstance(output, tuple) else output
            result = hidden.clone()
            result[:, 3:] += vector
            return (result, *output[1:]) if isinstance(output, tuple) else result
        handle = model.model.layers[1].register_forward_hook(reference_hook)
        try:
            with torch.inference_mode():
                reference = model(input_ids=prefix, attention_mask=fullmask, use_cache=False)
        finally:
            handle.remove()
        assert torch.allclose(cached.logits[:, -1], reference.logits[:, -1], atol=1e-6)
        next_id = cached.logits[:, -1].argmax(-1, keepdim=True)
        prefix = torch.cat([prefix, next_id], dim=1)
        fullmask = torch.cat([fullmask, torch.ones(2, 1, dtype=mask.dtype)], dim=1)
    assert audit["prefill_calls"] == 1 and audit["decode_calls"] == 3 and audit["max_abs_hidden_delta"] > 0
    with torch.inference_mode():
        default = model.generate(ids, attention_mask=mask, max_new_tokens=4, do_sample=False)
        zero_audit = {}
        with steer(model, 1, torch.zeros(24), 1, zero_audit):
            zero = model.generate(ids, attention_mask=mask, max_new_tokens=4, do_sample=False)
    assert torch.equal(default, zero)
    assert zero_audit["prefill_calls"] == 1 and zero_audit["decode_calls"] == 3
    assert all(torch.equal(p, weights[n]) for n, p in model.named_parameters())


def test_fixture_separation_semantic_event_variation_and_round_robin():
    splits = authored()
    report = validate(splits)
    assert report["counts"]["final"]["families"] == 12
    assert len({r["family"] for r in splits["extraction"][:16]}) == 16
    assert len({r["family"] for r in splits["extraction"][:50]}) == 16
    assert all({f["key"] for f in r["required_facts"]} >= {"person", "day", "event"} for rows in splits.values() for r in rows)
    family = splits["final"][:5]
    assert len({next(f["meaning"] for f in r["required_facts"] if f["key"] == "event") for r in family}) == 5
    broken = copy.deepcopy(splits)
    broken["final"][0]["family"] = splits["extraction"][0]["family"]
    with pytest.raises(ValueError, match="overlap"):
        validate(broken)
    broken = copy.deepcopy(splits)
    broken["extraction"].reverse()
    with pytest.raises(ValueError, match="round-robin"):
        validate(broken)


def test_literal_event_checks_count_and_object_not_action_semantics():
    row = authored()["extraction"][0]
    assert literal_checks(row, "Maya sorted two towels on Monday.")["complete"]
    assert not literal_checks(row, "Maya sorted two chairs on Monday.")["complete"]
    # Wrong action still has the literals: the independent AI event check must reject it.
    assert literal_checks(row, "Maya burned two towels on Monday.")["complete"]
    assert literal_checks(row, "Maya sorted 9 towels on Monday.")["unsupported_literal"]


def fixture_judgments():
    rows = authored()["development"][:2]
    predictions = [{"id": r["id"], "output": "dummy response used only in unit test", "generated_ids": [4, 5],
                    "ended_eos": False, "hit_token_limit": False} for r in rows]
    packet, keys = make_packets(rows, {"default": predictions, "hidden_layer17": predictions},
                                [("default", "hidden_layer17")], 42)
    judge = {"model_identifier": "test-judge", "exposed_settings": {"effort": None}, "fresh_context": True, "advisor_overlap": False}
    content = {"judge": judge, "ratings": [{"id": r["id"], "coverage": {f["key"]: True for f in r["required_facts"]},
        "task_completed": True, "unsupported_claims": 0, "contradicted_claims": 0, "content_score": 5, "warmth_score": 3} for r in packet["content"]]}
    warmth = {"judge": judge, "ratings": [{"id": r["id"], "preference": "tie", "warmth_A": 3, "warmth_B": 3,
                                          "content_A": 5, "content_B": 5} for r in packet["warmth"]]}
    return rows, predictions, packet, keys, content, warmth


def test_blind_packets_complete_schema_and_diagnostic_disagreements():
    rows, predictions, packet, keys, content, warmth = fixture_judgments()
    assert "hidden_layer17" not in str(packet) and "default" not in str(packet)
    validate_ratings(packet["content"], content, "content")
    validate_ratings(packet["warmth"], warmth, "warmth")
    decoded, preferences = decoded_ratings(keys, content, warmth)
    assert all(x == .5 for x in preferences["default__hidden_layer17"].values())
    metrics = measurements(rows, predictions, decoded["default"])
    assert metrics["literal_ai_coverage_disagreements"] == 2
    for bad in ({**content, "ratings": content["ratings"][:-1]},
                {**content, "ratings": content["ratings"] + [content["ratings"][0]]}):
        with pytest.raises(ValueError):
            validate_ratings(packet["content"], bad, "content")
    bad = copy.deepcopy(content)
    del bad["ratings"][0]["task_completed"]
    with pytest.raises(ValueError):
        validate_ratings(packet["content"], bad, "content")


def test_gates_and_selection_exact_registered_order():
    base = {"complete_facts": 30, "task_completion_failures": 0, "unsupported_or_contradicted": 0,
            "truncated": 0, "empty": 0, "degenerate": 0}
    assert gates(base, base, [1.]*30)["passed"]
    for field, value in (("complete_facts", 28), ("task_completion_failures", 2), ("unsupported_or_contradicted", 2),
                         ("truncated", 2), ("empty", 1), ("degenerate", 1)):
        assert not gates(base, {**base, field: value}, [1.]*30)["passed"]
    assert not gates(base, base, [1.51]*30)["passed"]
    assert degenerate("one two three four "*8)["degenerate"]
    assert not degenerate("one two three four")["degenerate"]
    def candidate(layer, multiplier, score, passed=True):
        return {"condition": f"L{layer}m{multiplier}", "layer": layer, "multiplier": multiplier,
                "warmth_score": score, "gates": {"passed": passed}}
    choice = select([candidate(11, 1, .7), candidate(17, 1, .7), candidate(23, 4, .95)])
    assert choice["layer"] == 17 and choice["multiplier"] == 1
    assert select([candidate(11, 1, .55), candidate(23, 2, .6)])["exploratory"]
    assert select([candidate(17, 1, .5), candidate(23, 1, 1, False)])["multiplier"] == 0
    rows = authored()["final"]
    ci = grouped_interval(rows, {r["id"]: .5 for r in rows}, samples=30)
    assert ci["families"] == 12 and ci["ci95"] == [.5, .5]
