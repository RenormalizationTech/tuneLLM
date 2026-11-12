"""Actual Qwen tokenizer boundary tests; skip only when local cache/dependencies absent."""
import importlib.util
import unittest
from scripts.common import load_config, local_snapshot, path, setup_hf
from scripts.training_data import encode_row, pad_batch

CFG = load_config("configs/baseline.json")
SNAPSHOT = path(CFG["hf_home"]) / "hub" / "models--Qwen--Qwen3-4B" / "snapshots" / CFG["revision"]


@unittest.skipUnless((SNAPSHOT / "tokenizer_config.json").is_file() and importlib.util.find_spec("transformers"),
                     "requires manager-provisioned Transformers and cached Qwen tokenizer; never downloads")
class LocalTokenizerTests(unittest.TestCase):
    def test_actual_boundary_eos_and_padding(self):
        setup_hf(CFG)
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(SNAPSHOT, local_files_only=True, trust_remote_code=False)
        self.assertNotEqual(tokenizer.pad_token_id, tokenizer.eos_token_id)
        self.assertEqual(tokenizer.eos_token, "<|im_end|>")
        encoded = []
        for i, answer in enumerate(("42", "Reasoning here.\n#### -1,234.5", "π = 3.14", "Unicode: café")):
            row = {"id": str(i), "user": "What is the answer?", "assistant": answer, "system": "Be concise."}
            example = encode_row(row, tokenizer, 1024)
            supervised = [token for token in example["labels"] if token != -100]
            self.assertEqual(tokenizer.decode(supervised), answer + tokenizer.eos_token)
            prompt = tokenizer.decode(example["input_ids"][:example["labels"].count(-100)])
            self.assertIn("Be concise.", prompt)
            self.assertTrue(prompt.endswith("<think>\n\n</think>\n\n"))
            encoded.append(example)
        batch = pad_batch(encoded, tokenizer.pad_token_id)
        for index, row in enumerate(encoded):
            self.assertEqual(batch["labels"][index][len(row["input_ids"]) - 1], tokenizer.eos_token_id)
            self.assertTrue(all(label == -100 for label in batch["labels"][index][len(row["input_ids"]):]))
        # Leading newlines can merge with the inference prefix. Fail rather than
        # assign a boundary token partly containing the prompt to assistant loss.
        with self.assertRaisesRegex(ValueError, "merges across"):
            encode_row({"id": "leading-newline", "user": "Question?", "assistant": "\n42"}, tokenizer, 1024)


if __name__ == "__main__":
    unittest.main()
