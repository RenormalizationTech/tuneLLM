"""Small CPU checks for the sparse supervised-logit loss path."""
import pytest
import torch

from scripts.qwen38_science_runtime import answer_loss


class TinyModel:
    def __init__(self, logits):
        self.logits = logits

    def __call__(self, input_ids, attention_mask, logits_to_keep, use_cache):
        assert use_cache is False
        return type("Output", (), {"logits": self.logits[:, logits_to_keep]})()


def test_sparse_loss_matches_full_shifted_assistant_loss_and_gradients():
    torch.manual_seed(42)
    logits = torch.randn(1, 9, 17, requires_grad=True)
    labels = torch.tensor([[-100, -100, -100, -100, 4, 5, -100, 7, 8]])
    expected = torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, 17), labels[:, 1:].reshape(-1))
    expected_grad = torch.autograd.grad(expected, logits, retain_graph=True)[0]
    actual, count = answer_loss(TinyModel(logits), torch.zeros(1, 9, dtype=torch.long), labels, torch.ones(1, 9))
    actual_grad = torch.autograd.grad(actual, logits)[0]
    assert count == 4
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual_grad, expected_grad)


def test_empty_supervision_is_rejected_before_model_call():
    batch = torch.zeros(1, 5, dtype=torch.long)
    with pytest.raises(ValueError, match="nonempty assistant"):
        answer_loss(None, batch, torch.full_like(batch, -100), torch.ones_like(batch))


def test_first_input_token_is_never_a_causal_target():
    logits = torch.randn(1, 4, 9)
    labels = torch.tensor([[1, -100, -100, 3]])
    loss, count = answer_loss(TinyModel(logits), torch.zeros(1, 4, dtype=torch.long), labels, torch.ones(1, 4))
    assert count == 1
    torch.testing.assert_close(loss, torch.nn.functional.cross_entropy(logits[:, 2], torch.tensor([3])))
