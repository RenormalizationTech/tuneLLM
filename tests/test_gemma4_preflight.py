"""CPU checks for the new, opt-in Gemma mechanics preflight."""
from pathlib import Path

import pytest
import torch

from scripts.gemma4_preflight import encode, response_loss


@pytest.fixture(scope="module")
def tokenizer():
    from transformers import AutoTokenizer
    path = Path("outputs/gemma4-preflight-01/snapshot_path.txt")
    if not path.exists():
        pytest.skip("pinned Gemma tokenizer not downloaded")
    return AutoTokenizer.from_pretrained(path.read_text().strip(), local_files_only=True)


def test_native_no_thinking_prefix_and_turn_end_are_masked_correctly(tokenizer):
    user = "What does the measured sample show?"
    answer = '{"answer":"A larger response.","evidence":["P1"]}'
    row = encode(tokenizer, user, answer)
    n = row["prompt_tokens"]
    assert tokenizer.decode(row["input_ids"][:n]).endswith("<|channel>thought\n<channel|>")
    assert row["labels"][:n] == [-100] * n
    assert row["labels"][n:] == row["input_ids"][n:]
    assert row["labels"][-1] == 106 != tokenizer.eos_token_id
    assert tokenizer.decode(row["labels"][n:-1]) == answer
    # The native full-conversation formatter omits the forced empty thought.
    # This test prevents accidentally adopting that different SFT boundary.
    native = tokenizer.apply_chat_template([{"role":"user","content":user},
        {"role":"assistant","content":answer}], tokenize=True, enable_thinking=False, return_dict=False)
    assert isinstance(native,list) and all(isinstance(i,int) for i in native)
    assert native[:n] != row["input_ids"][:n]


@pytest.mark.parametrize("answer", ["", " ", "Bad <turn|> boundary", "<eos>"])
def test_empty_or_control_token_targets_rejected(tokenizer, answer):
    with pytest.raises(ValueError):
        encode(tokenizer, "A question.", answer)


def test_native_tiny_gemma_compact_loss_and_parameter_gradients_match_full_logits():
    from transformers import Gemma4ForCausalLM, Gemma4TextConfig
    torch.manual_seed(42)
    model = Gemma4ForCausalLM(Gemma4TextConfig(vocab_size=64, vocab_size_per_layer_input=64,
        hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=2,
        num_key_value_heads=1, num_global_key_value_heads=1, head_dim=16, global_head_dim=16,
        hidden_size_per_layer_input=0, num_kv_shared_layers=0, attention_k_eq_v=True,
        layer_types=["sliding_attention","full_attention"], sliding_window=8,
        max_position_embeddings=64, use_bidirectional_attention="vision"))
    model.eval()
    ids = torch.randint(3,64,(1,12)); labels = ids.clone(); labels[:,:7] = -100
    labels[:,9] = -100
    mask = torch.ones_like(ids)
    logits = model(input_ids=ids, attention_mask=mask, use_cache=False).logits.float()
    full = torch.nn.functional.cross_entropy(logits[:,:-1].reshape(-1,64), labels[:,1:].reshape(-1))
    expected = torch.autograd.grad(full, model.model.embed_tokens.weight, retain_graph=False)[0]
    compact = response_loss(model, ids, labels, mask)
    actual = torch.autograd.grad(compact, model.model.embed_tokens.weight)[0]
    torch.testing.assert_close(compact, full, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)


def test_empty_response_loss_rejected_without_a_model_call():
    ids = torch.ones((1,4),dtype=torch.long)
    with pytest.raises(ValueError,match="nonempty supervised"):
        response_loss(None, ids, torch.full_like(ids,-100), torch.ones_like(ids))
