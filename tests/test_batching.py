from types import SimpleNamespace
import unittest
from scripts.batched_generation import generate_batched, generated_suffix, left_pad


class PaddingTests(unittest.TestCase):
    def test_left_padding_attention_and_eos(self):
        padded = left_pad([[4, 5], [6]], 0)
        self.assertEqual(padded["input_ids"], [[4, 5], [0, 6]])
        self.assertEqual(padded["attention_mask"], [[1, 1], [0, 1]])
        self.assertEqual(generated_suffix([0, 6, 9, 1, 0, 0], 2, 1), [9, 1])
        self.assertEqual(generated_suffix([4, 5, 9, 10], 2, 1), [9, 10])
        self.assertEqual(generated_suffix([4, 5, 9, 0, 0], 2, 1), [9, 0, 0])
        self.assertEqual(generated_suffix([4, 5, 9, 0, 1, 0], 2, 1), [9, 0, 1])

    def test_cpu_fake_model_uses_padded_width_and_preserves_order(self):
        import torch

        class Tokenizer:
            pad_token_id, eos_token_id, padding_side = 0, 1, "right"

            def apply_chat_template(self, messages, **kwargs):
                assert kwargs["enable_thinking"] is False
                return messages[-1]["content"]

            def __call__(self, prompt, **kwargs):
                return {"input_ids": [int(value) for value in prompt.split()]}

            def decode(self, ids, **kwargs):
                return " ".join(str(value) for value in ids if value != 1)

        class Model:
            config = SimpleNamespace(max_position_embeddings=100)

            def eval(self):
                return self

            def generate(self, input_ids, attention_mask, **kwargs):
                assert input_ids.tolist() == [[4, 5], [0, 6]]
                assert attention_mask.tolist() == [[1, 1], [0, 1]]
                assert kwargs["do_sample"] is False
                return torch.cat((input_ids, torch.tensor([[9, 1, 0], [8, 7, 0]])), dim=1)

        tokenizer = Tokenizer()
        result = generate_batched(Model(), tokenizer, [{"id": "a", "user": "4 5"}, {"id": "b", "user": "6"}],
                                  batch_size=8, max_new_tokens=3, device="cpu")
        self.assertEqual(tokenizer.padding_side, "right")
        self.assertEqual([p["generated_ids"] for p in result["predictions"]], [[9, 1], [8, 7, 0]])
        self.assertFalse(result["predictions"][0]["hit_token_limit"])
        self.assertTrue(result["predictions"][1]["hit_token_limit"])
        self.assertEqual([p["id"] for p in result["predictions"]], ["a", "b"])
        self.assertEqual(result["predictions"][1]["batch_ids"], ["a", "b"])
        self.assertEqual(result["batches"][0]["padded_prompt_width"], 2)


if __name__ == "__main__":
    unittest.main()
