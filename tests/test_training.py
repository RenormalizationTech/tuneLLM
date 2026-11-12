import json
import tempfile
import unittest
from pathlib import Path

from evaluation.gsm8k import score_gsm8k
from scripts.prepare_gsm8k import INSTRUCTION, prepare_records
from scripts.training_data import encode_row, load_splits, pad_batch


class FakeTokenizer:
    """Character tokens, plus dedicated EOS; exposes masking independent of ML packages."""
    pad_token_id = 0
    eos_token_id = 1
    eos_token = "<EOS>"
    all_special_ids = [0, 1]

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs == {"tokenize": False, "add_generation_prompt": True, "enable_thinking": False}
        return "USER:" + messages[-1]["content"] + "\nASSISTANT:<think>\n\n</think>\n\n"

    def __call__(self, text, **kwargs):
        assert kwargs == {"add_special_tokens": False}
        ids = []
        while text:
            if text.startswith(self.eos_token):
                ids.append(1)
                text = text[len(self.eos_token):]
            else:
                ids.append(ord(text[0]) + 2)
                text = text[1:]
        return {"input_ids": ids}


class LabelTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = FakeTokenizer()
        self.row = {"id": "one", "user": "Question?", "assistant": "Answer."}

    def test_only_answer_and_eos_supervised(self):
        example = encode_row(self.row, self.tokenizer, 256)
        supervised = [x for x in example["labels"] if x != -100]
        self.assertEqual(supervised, self.tokenizer("Answer.<EOS>", add_special_tokens=False)["input_ids"])
        self.assertGreater(example["labels"].count(-100), len(self.row["user"]))
        self.assertEqual(example["labels"][-1], self.tokenizer.eos_token_id)

    def test_padding_masks_positions_and_preserves_real_eos(self):
        long = encode_row(self.row, self.tokenizer, 256)
        short = encode_row({**self.row, "assistant": "A"}, self.tokenizer, 256)
        batch = pad_batch([long, short], self.tokenizer.pad_token_id)
        end = len(short["input_ids"]) - 1
        self.assertEqual(batch["labels"][1][end], self.tokenizer.eos_token_id)
        self.assertEqual(batch["attention_mask"][1][end], 1)
        self.assertTrue(all(x == -100 for x in batch["labels"][1][end + 1:]))
        self.assertTrue(all(x == 0 for x in batch["attention_mask"][1][end + 1:]))

    def test_reject_truncation_special_answer_and_pad_eos_alias(self):
        with self.assertRaisesRegex(ValueError, "never silently truncate"):
            encode_row(self.row, self.tokenizer, 2)
        with self.assertRaises(ValueError):
            encode_row({**self.row, "assistant": "x<EOS>"}, self.tokenizer, 256)
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        with self.assertRaises(ValueError):
            encode_row(self.row, self.tokenizer, 256)

    def test_reject_cross_boundary_merge(self):
        class MergingTokenizer(FakeTokenizer):
            def __call__(self, text, **kwargs):
                result = super().__call__(text, **kwargs)
                if text.endswith(self.eos_token):
                    result["input_ids"][0] = 999
                return result
        with self.assertRaisesRegex(ValueError, "merges across"):
            encode_row(self.row, MergingTokenizer(), 256)

    def test_split_leakage_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            paths = {}
            for i, name in enumerate(("train", "validation", "test")):
                p = Path(temp) / (name + ".jsonl")
                p.write_text(json.dumps({**self.row, "id": str(i)}))
                paths[name] = str(p)
            with self.assertRaisesRegex(ValueError, "duplicate"):
                load_splits(paths)

    def test_jsonl_unicode_separators_inside_strings_are_not_records(self):
        separators = "\u0085\u2028\u2029"
        with tempfile.TemporaryDirectory() as temp:
            paths, expected = {}, {}
            for index, name in enumerate(("train", "validation", "test")):
                rows = [{"id": f"{name}-{j}", "user": f"{name} question {j}{separators}more",
                         "assistant": f"answer{separators}continued"} for j in range(2)]
                p = Path(temp) / (name + ".jsonl")
                p.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")
                paths[name], expected[name] = str(p), rows
            self.assertEqual(load_splits(paths), expected)


class GSMTests(unittest.TestCase):
    def test_overlap_dedup_annotations_and_deterministic_split(self):
        train = [{"question": q, "answer": "calc <<2+2=4>>4\n#### 4"}
                 for q in ("A", "B", "C", " a ", " TEST ")]
        test = [{"question": "test", "answer": "#### 7"}]
        splits, counts = prepare_records(train, test, 42, 1)
        self.assertEqual(counts["train_test_overlap_removed"], 1)
        self.assertEqual(counts["train_duplicates_removed"], 1)
        self.assertEqual(counts["calculator_annotations_removed"], 3)
        self.assertEqual(splits, prepare_records(train, test, 42, 1)[0])
        self.assertEqual(len(splits["test"]), len(test))
        for row in splits["train"] + splits["validation"] + splits["test"]:
            self.assertTrue(row["user"].endswith(INSTRUCTION))
            self.assertNotIn("<<", row["assistant"])
            self.assertIn("source_id", row)

    def test_marker_fallback_and_truncation_separate(self):
        references = [{"id": "a", "assistant": "#### 1200"}, {"id": "b", "assistant": "#### -2.5"}]
        predictions = [{"id": "a", "output": "#### 1,200.0", "hit_token_limit": False},
                       {"id": "b", "output": "The answer is -2.50", "hit_token_limit": True}]
        scores = score_gsm8k(references, predictions)
        self.assertEqual(scores["strict_correct"], 1)
        self.assertEqual(scores["fallback_correct"], 2)
        self.assertEqual(scores["format_compliant"], 1)
        self.assertEqual(scores["hit_token_limit"], 1)
        predictions[1]["output"] = "#### -2.5\nMore prose"
        self.assertEqual(score_gsm8k(references, predictions)["format_compliant"], 1)


if __name__ == "__main__":
    unittest.main()
