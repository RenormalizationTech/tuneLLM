import unittest
from scripts.prepare_retention import eligible_subject, select_retention
from scripts.score_retention import score_retention


class RetentionTests(unittest.TestCase):
    def test_strict_and_case_preserved_fallback(self):
        outputs = ["it is a cat", "Answer: B", "B.", "B or C", " b "]
        rows = [{"id": str(i), "answer": "B"} for i in range(len(outputs))]
        predictions = [{"id": str(i), "output": value, "hit_token_limit": False} for i, value in enumerate(outputs)]
        scores = score_retention(rows, predictions)
        self.assertTrue(scores["details"][0]["fallback_invalid"])
        self.assertIsNone(scores["details"][0]["fallback_choice"])
        self.assertEqual(scores["strict_correct"], 1)
        self.assertEqual(scores["fallback_correct"], 4)
        self.assertTrue(scores["details"][3]["fallback_ambiguous"])
        self.assertEqual(scores["details"][3]["fallback_choice"], "B")
        self.assertEqual(scores["invalid"], 4)

    def test_stratification_source_indices_and_determinism(self):
        rows = [{"subject": subject, "question": f"question {i}", "choices": ["one", "two", "three", "four"], "answer": i % 4}
                for subject in ("world_history", "astronomy", "formal_logic", "college_mathematics", "abstract_algebra", "high_school_statistics")
                for i in range(5)]
        selected, metadata = select_retention(rows, seed=1234, count=8)
        self.assertEqual((selected, metadata), select_retention(rows, seed=1234, count=8))
        self.assertEqual(sorted(metadata["allocation"].values()), [2, 3, 3])
        for row in selected:
            self.assertTrue(eligible_subject(row["subject"]))
            self.assertEqual(rows[row["source_row"]]["subject"], row["subject"])
            self.assertEqual(row["answer"], "ABCD"[rows[row["source_row"]]["answer"]])
            self.assertTrue(row["user"].endswith("Answer with only the letter."))


if __name__ == "__main__":
    unittest.main()
