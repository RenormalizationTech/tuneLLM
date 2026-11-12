import copy
import unittest

from evaluation.score import score
from scripts.common import read_json, required_weights


class ScoreTests(unittest.TestCase):
    def setUp(self):
        self.tasks = read_json("evaluation/small_tasks.json")["tasks"]
        self.predictions = [{"id": t["id"], "output": t["answers"][0]} for t in self.tasks]

    def test_correct_and_order_independent(self):
        self.predictions[0]["output"] = " \n４２\t"
        self.assertEqual(score(self.tasks, self.predictions[::-1])["accuracy"], 1)

    def test_no_substring_credit(self):
        self.predictions[0]["output"] = "The answer is 42, or possibly 43."
        result = score(self.tasks, self.predictions)
        self.assertEqual(result["correct"], len(self.tasks) - 1)
        self.assertEqual(result["categories"]["general"]["correct"], 3)

    def test_reject_incomplete_duplicate_or_extra_predictions(self):
        for predictions in (self.predictions[:-1], self.predictions + [self.predictions[0]],
                            self.predictions + [{"id": "extra", "output": "42"}]):
            with self.assertRaises(ValueError):
                score(self.tasks, predictions)

    def test_defensive_download_selection(self):
        self.assertEqual(required_weights(["pytorch_model.bin", "model.safetensors", "adapter_model.safetensors"]), ["model.safetensors"])
        selected = required_weights(["model.safetensors.index.json", "model-00001-of-00002.safetensors",
                                     "model-00002-of-00002.safetensors", "pytorch_model-00001.bin"])
        self.assertEqual(len(selected), 3)
        self.assertFalse(any(name.endswith(".bin") for name in selected))
        with self.assertRaises(ValueError):
            required_weights(["pytorch_model.bin"])


if __name__ == "__main__":
    unittest.main()
