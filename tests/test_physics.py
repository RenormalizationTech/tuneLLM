import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from scripts.physics_data import aggregate_nll, chunk_paper, chunk_papers, exposure, read_papers, validate_splits
from scripts.physics_eval import evaluate_chunks
from scripts.training_data import pad_batch


class NumberTokenizer:
    eos_token_id, pad_token_id = 99, 0

    def __call__(self, text, **kwargs):
        assert kwargs == {"add_special_tokens": False}
        return {"input_ids": [int(value) for value in text.split()]}


class PaperTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = NumberTokenizer()
        self.row = {"id": "id1", "paper_id": "paper1", "text": "1 2 3 4 5 6 7 8 9 10"}

    def test_chunks_preserve_each_target_and_tail_with_paper_end_eos_only(self):
        chunks = chunk_paper(self.row, self.tokenizer, 4)
        targets = [token for row in chunks for token in row["labels"][1:]]
        self.assertEqual(targets, list(range(2, 11)) + [99])
        self.assertEqual([row["input_ids"] for row in chunks], [[1, 2, 3, 4], [4, 5, 6, 7], [7, 8, 9, 10], [10, 99]])
        self.assertEqual(sum(row["input_ids"].count(99) for row in chunks), 1)
        stats = exposure(chunks, 4)
        self.assertEqual(stats["loss_tokens_per_epoch"], 10)
        self.assertEqual(stats["remainder_chunks"], 1)
        self.assertEqual(stats["repeated_context_tokens_per_epoch"], 3)
        batch = pad_batch([chunks[0], chunks[-1]], 0)
        self.assertEqual(batch["labels"][1], [-100, 99, -100, -100])
        self.assertEqual(batch["attention_mask"][1], [1, 1, 0, 0])

    def test_never_pack_across_papers_and_reject_leakage(self):
        second = {"id": "id2", "paper_id": "paper2", "text": "11 12"}
        chunks = chunk_papers([self.row, second], self.tokenizer, 4)
        self.assertEqual(chunks[-1]["input_ids"], [11, 12, 99])
        self.assertEqual(chunks[-1]["paper_id"], "paper2")
        splits = {"train": [self.row], "validation": [second],
                  "test": [{"id": "id3", "paper_id": "paper3", "text": "13 14"}]}
        validate_splits(splits)
        splits["test"][0]["paper_id"] = "paper1"
        with self.assertRaisesRegex(ValueError, "duplicate"):
            validate_splits(splits)
        with self.assertRaisesRegex(ValueError, "embedded EOS"):
            chunk_paper({**self.row, "text": "2 99 3"}, self.tokenizer, 4)

    def test_jsonl_preserves_unicode_and_nll_weights_by_target_not_paper(self):
        with tempfile.TemporaryDirectory() as temp:
            filename = Path(temp) / "paper.jsonl"
            row = {**self.row, "text": "line\u2028not a new record\u0085\u2029"}
            filename.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
            self.assertEqual(read_papers(filename), [row])
        report = aggregate_nll([{"paper_id": "p1", "loss_tokens": 4, "nll_sum": 4.0},
                                {"paper_id": "p1", "loss_tokens": 6, "nll_sum": 6.0},
                                {"paper_id": "p2", "loss_tokens": 3, "nll_sum": 9.0}])
        self.assertAlmostEqual(report["token_weighted_nll"], 19 / 13)
        self.assertEqual(report["mean_per_paper_nll"], 2)
        self.assertEqual(report["per_paper"]["p1"]["loss_tokens"], 10)
        with self.assertRaises(ValueError):
            aggregate_nll([{"paper_id": "bad", "loss_tokens": 0, "nll_sum": 1.0}])

    def test_cpu_uniform_logits_evaluation_counts_eos_masks_pad_and_restores_mode(self):
        import torch

        class UniformModel(torch.nn.Module):
            def forward(self, input_ids, attention_mask, use_cache):
                assert not use_cache
                return SimpleNamespace(logits=torch.zeros((*input_ids.shape, 128)))

        model = UniformModel()
        chunks = chunk_paper(self.row, self.tokenizer, 4)
        result = evaluate_chunks(model, chunks, self.tokenizer, batch_size=3, device="cpu")
        self.assertTrue(model.training)
        self.assertEqual(result["loss_tokens"], 10)
        self.assertAlmostEqual(result["token_weighted_nll"], math.log(128), places=5)
        self.assertAlmostEqual(result["per_paper"]["paper1"]["nll"], math.log(128), places=5)


if __name__ == "__main__":
    unittest.main()
