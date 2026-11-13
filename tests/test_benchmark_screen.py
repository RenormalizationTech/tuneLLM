import copy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from scripts.benchmark_screen import (GGUFBackend, attach_adapter, choice_token_ids,
                                      hf_multiple_choice, ifeval_scores, run_screen)
from scripts.common import path
from scripts.run_support import snapshot_sources


class ChoiceTokenizer:
    pad_token_id, eos_token_id = 0, 15

    def __call__(self, text, **kwargs):
        assert kwargs["add_special_tokens"] is False
        if text in (" A", " B", " C", " D"):
            return {"input_ids": ["ABCD".index(text[-1]) + 1]}
        for prompt, tokens in (("PROMPT", [10, 11]), ("SHORT", [11])):
            if text == prompt:
                return {"input_ids": tokens}
            if text.startswith(prompt) and text[len(prompt):] in (" A", " B", " C", " D"):
                return {"input_ids": tokens + ["ABCD".index(text[-1]) + 1]}
        raise AssertionError(text)

    def __len__(self):
        return 1000

    def apply_chat_template(self, messages, **kwargs):
        return "PROMPT"


def mc_row(identifier="one", prompt="PROMPT"):
    return {"id": identifier, "subject": "physics", "prompt": prompt,
            "choices": [" A", " B", " C", " D"], "answer_index": 2}


def backend_without_process():
    backend = GGUFBackend.__new__(GGUFBackend)
    backend.tokenizer, backend.opener, backend.base = ChoiceTokenizer(), object(), "http://127.0.0.1:1234"
    return backend


class BenchmarkTests(unittest.TestCase):
    def test_official_ifeval_alignment_and_strict_scoring(self):
        rows = [{"key": 1, "prompt": "Do not use commas.", "instruction_id_list": ["punctuation:no_comma"], "kwargs": [{}]},
                {"key": 2, "prompt": "Mention orbit.", "instruction_id_list": ["keywords:existence"], "kwargs": [{"keywords": ["orbit"]}]}]
        result = ifeval_scores(rows, [{"id": "2", "output": "No keyword here."}, {"id": "1", "output": "A plain sentence."}])
        self.assertEqual(result["prompt_strict_correct"], 1)
        self.assertEqual(result["instruction_strict_correct"], 1)
        self.assertEqual(result["instruction_total"], 2)
        with self.assertRaises(ValueError):
            ifeval_scores(rows, [{"id": "1", "output": "x"}] * 2)

    def test_shared_choice_boundary_validation_and_cpu_likelihood(self):
        import torch
        tokenizer = ChoiceTokenizer()
        self.assertEqual(choice_token_ids(tokenizer, [mc_row()]), [1, 2, 3, 4])

        class BrokenBoundary(ChoiceTokenizer):
            def __call__(self, text, **kwargs):
                return {"input_ids": [9]} if text == "PROMPT A" else super().__call__(text, **kwargs)

        with self.assertRaisesRegex(ValueError, "boundary"):
            choice_token_ids(BrokenBoundary(), [mc_row()])
        tensor = torch.tensor

        class Model:
            def __call__(self, input_ids, attention_mask, logits_to_keep):
                assert input_ids.tolist() == [[10, 11], [0, 11]]
                assert attention_mask.tolist() == [[1, 1], [0, 1]]
                assert logits_to_keep == 1
                logits = torch.zeros(2, 1, 16)
                logits[:, :, 3] = 5
                return SimpleNamespace(logits=logits)

        with patch("torch.tensor", side_effect=lambda value, **kwargs: tensor(value, device="cpu")):
            records = hf_multiple_choice(Model(), tokenizer, [mc_row(), mc_row("two", "SHORT")])
        self.assertTrue(all(record["correct"] for record in records))
        self.assertTrue(all(record["choice_loglikelihood"][2] > record["choice_loglikelihood"][0] for record in records))

    def test_gguf_completion_uses_verified_ids_and_rejects_context_loss(self):
        backend = backend_without_process()
        response = {"content": "C", "tokens": [3], "tokens_evaluated": 2, "truncated": False}
        with patch("scripts.benchmark_screen.request_json", side_effect=[{"tokens": [10, 11]}, response]) as request:
            self.assertEqual(backend.completion("PROMPT", 1, 512), response)
            self.assertTrue(request.call_args_list[0].args[3]["parse_special"])
            payload = request.call_args_list[1].args[3]
            self.assertEqual(payload["prompt"], [10, 11])
            self.assertFalse(payload["post_sampling_probs"])
            self.assertFalse(payload["cache_prompt"])
        with patch("scripts.benchmark_screen.request_json", side_effect=[{"tokens": [10, 11]}, {**response, "truncated": True}]):
            with self.assertRaisesRegex(ValueError, "truncated"):
                backend.completion("PROMPT", 1)

    def test_gguf_limit_field_and_mc_full_vocabulary_retry(self):
        backend = backend_without_process()
        backend.completion = Mock(return_value={"content": "x", "tokens": [3], "stop_type": "limit", "timings": {}})
        self.assertTrue(backend.generate([{"id": "one", "user": "q"}], 8)[0]["hit_token_limit"])
        response = lambda entries: {"completion_probabilities": [{"top_logprobs": [{"id": i, "logprob": value} for i, value in entries]}]}
        backend.completion = Mock(side_effect=[response([(1, -1)]), response([(1, -4), (2, -3), (3, -1), (4, -2)])])
        self.assertTrue(backend.multiple_choice([mc_row()])[0]["correct"])
        self.assertEqual(backend.completion.call_args_list[-1].args[2], len(backend.tokenizer))

    def test_server_is_stopped_if_readiness_fails(self):
        backend = GGUFBackend.__new__(GGUFBackend)
        child = Mock()
        with tempfile.TemporaryDirectory() as temp, patch("scripts.benchmark_screen.socket.socket") as socket, \
                patch("scripts.benchmark_screen.subprocess.Popen", return_value=child), \
                patch("scripts.benchmark_screen.wait_ready", side_effect=TimeoutError("load timeout")), \
                patch("scripts.benchmark_screen.stop_child") as stop:
            socket.return_value.__enter__.return_value.getsockname.return_value = ("127.0.0.1", 1234)
            with self.assertRaises(TimeoutError):
                backend.__init__("unused.gguf", ChoiceTokenizer(), Path(temp))
            stop.assert_called_once_with(child)
            self.assertTrue(backend.log.closed)

    def test_hf_generation_uses_batch8_and_records_token_diagnostic(self):
        fixture = {"gsm8k": [{"id": "one", "user": "q", "assistant": "#### 2"}]}
        generated = {"predictions": [{"id": "one", "output": "#### 2", "generated_ids": [2, 15], "hit_token_limit": False}],
                     "timing": {"generation_wall_seconds": 1.0}}
        with tempfile.TemporaryDirectory() as temp, patch("scripts.benchmark_screen.generate_batched", return_value=generated) as generate:
            diagnostics = []
            run_screen(None, None, fixture, Path(temp), tasks=["gsm8k"], diagnostics=diagnostics)
            self.assertEqual(generate.call_args.kwargs["batch_size"], 8)
            self.assertEqual(diagnostics[0]["returned_generated_token_ids"], 2)

    def test_source_snapshot_includes_benchmark_scripts(self):
        with tempfile.TemporaryDirectory() as temp:
            snapshot_sources(Path(temp) / "source")
            for name in ("benchmark_screen.py", "prepare_benchmark_screen.py"):
                self.assertEqual((Path(temp) / "source/scripts" / name).read_bytes(), path("scripts/" + name).read_bytes())

    def test_bf16_control_and_adapter_reload_preserve_base_dtype(self):
        import torch
        from transformers import Qwen3Config, Qwen3ForCausalLM
        from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
        from scripts.physics_train import load_base
        config = Qwen3Config(vocab_size=16, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                             num_attention_heads=2, num_key_value_heads=2, head_dim=8, max_position_embeddings=32)
        original = Qwen3ForCausalLM(config).to(torch.bfloat16)
        weights = copy.deepcopy(original.state_dict())
        with patch("scripts.physics_train.local_snapshot", return_value="unused"), \
                patch("transformers.AutoModelForCausalLM.from_pretrained", return_value=original) as load:
            self.assertIs(load_base({"representation": "bf16"}, training=False), original)
            self.assertEqual(load.call_args.kwargs["torch_dtype"], torch.bfloat16)
        adapted = get_peft_model(original, LoraConfig(task_type="CAUSAL_LM", r=2, lora_alpha=4, target_modules=["q_proj", "v_proj"]))
        for name, parameter in adapted.named_parameters():
            if "lora_B" in name:
                parameter.data.fill_(0.01)
        with tempfile.TemporaryDirectory() as temp:
            adapted.save_pretrained(temp, safe_serialization=True)
            fresh = Qwen3ForCausalLM(config).to(torch.bfloat16)
            fresh.load_state_dict(weights)
            reloaded = attach_adapter(fresh, temp)
            for name, parameter in reloaded.named_parameters():
                self.assertEqual(parameter.dtype, torch.float32 if "lora_" in name else torch.bfloat16)
                self.assertFalse(parameter.requires_grad)
            before, after = get_peft_model_state_dict(adapted), get_peft_model_state_dict(reloaded)
            self.assertEqual(set(before), set(after))
            self.assertTrue(all(torch.equal(value, after[name]) for name, value in before.items()))
            adapted.eval()
            reloaded.eval()
            with torch.inference_mode():
                tokens = torch.tensor([[1, 2, 3]])
                self.assertTrue(torch.equal(adapted(tokens).logits, reloaded(tokens).logits))


if __name__ == "__main__":
    unittest.main()
