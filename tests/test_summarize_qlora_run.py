import json
from pathlib import Path
import tempfile
import unittest
from scripts.summarize_qlora_run import (condition_summary, exact_mcnemar, learning_curve,
                                         paired_counts, summarize, verify_selection_history)


class PairedSummaryTests(unittest.TestCase):
    def test_selection_losses_must_match_validation_history(self):
        curve = {"validation_loss_history": [{"step": 32, "validation_loss": 1.2},
                                             {"step": 64, "validation_loss": 1.1}]}
        selection = {"candidate_validation_losses": {"32": 1.2, "64": 1.1}}
        verify_selection_history(selection, curve)
        selection["candidate_validation_losses"]["64"] = 1.0
        with self.assertRaisesRegex(ValueError, "disagree with validation history"):
            verify_selection_history(selection, curve)

    def test_learning_curve_keeps_all_validation_points_and_only_compact_log_fields(self):
        history = [{"step": 0, "eval_loss": 3.0}]
        for step in (16, 32, 48, 64):
            history.extend([{"step": step, "loss": 1.5, "grad_norm": 0.2, "learning_rate": 0.0001,
                             "epoch": step / 64, "extra_text": "must be omitted"},
                            {"step": step, "eval_loss": 2 - step / 100, "eval_runtime": 5}])
        curve = learning_curve({"eval_loss": 3.0, "eval_runtime": 5}, history)
        self.assertEqual(curve["baseline_validation_loss"], 3.0)
        self.assertEqual([row["step"] for row in curve["validation_loss_history"]], [16, 32, 48, 64])
        self.assertEqual(set(curve["training_log_points"][0]), {"step", "loss", "grad_norm", "learning_rate"})
        self.assertNotIn("extra_text", json.dumps(curve))
        with self.assertRaisesRegex(ValueError, "exactly steps"):
            learning_curve({"eval_loss": 3.0}, history[:-1])

    def test_paired_alignment_by_id_not_position(self):
        base = [{"id": "fixed", "correct": False}, {"id": "broken", "correct": True},
                {"id": "both", "correct": True}, {"id": "neither", "correct": False}]
        after = [{"id": "neither", "correct": False}, {"id": "both", "correct": True},
                 {"id": "broken", "correct": False}, {"id": "fixed", "correct": True}]
        result = paired_counts(base, after, "correct")
        self.assertEqual({k: result[k] for k in ("fixed", "broken", "both", "neither")},
                         {"fixed": 1, "broken": 1, "both": 1, "neither": 1})
        self.assertEqual(result["accuracy_delta"], 0)
        self.assertEqual(result["mcnemar_exact_two_sided_p"], 1)
        self.assertEqual((result["b"], result["c"]), (1, 1))

    def test_missing_duplicate_and_nonboolean_pairs_fail(self):
        base = [{"id": "a", "correct": False}]
        for after in ([], [{"id": "other", "correct": True}],
                      [{"id": "a", "correct": True}] * 2,
                      [{"id": "a", "correct": 1}]):
            with self.assertRaises(ValueError):
                paired_counts(base, after, "correct")

    def test_binomial_edges_and_symmetry(self):
        for fixed, broken, expected in ((0, 0, 1), (1, 0, 1), (2, 0, 0.5),
                                        (5, 0, 0.0625), (5, 1, 0.21875), (10, 10, 1)):
            self.assertEqual(exact_mcnemar(fixed, broken), expected)
            self.assertEqual(exact_mcnemar(broken, fixed), expected)
        for counts in ((-1, 0), (1.5, 1), (True, 0)):
            with self.assertRaises(ValueError):
                exact_mcnemar(*counts)

    def test_condition_summary_uses_lengths_and_rejects_misalignment(self):
        saved = {"scores": {"total": 2, "strict_correct": 1, "fallback_correct": 2,
                            "details": [{"id": "b", "strict_correct": False, "fallback_correct": True,
                                         "format_compliant": False, "hit_token_limit": True},
                                        {"id": "a", "strict_correct": True, "fallback_correct": True,
                                         "format_compliant": True, "hit_token_limit": False}]},
                 "predictions": [{"id": "a", "generated_ids": [4, 1], "hit_token_limit": False},
                                 {"id": "b", "generated_ids": [4, 5, 6, 7], "hit_token_limit": True}]}
        result = condition_summary(saved, ["a", "b"], "gsm8k")
        self.assertEqual(result["mean_generated_tokens_including_eos"], 3)
        self.assertEqual(result["invalid"], 1)
        self.assertEqual(result["truncated"], 1)
        with self.assertRaises(ValueError):
            condition_summary(saved, ["a", "other"], "gsm8k")

    def test_incomplete_run_and_failed_reload_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp)
            (run / "protocol.json").write_text("{}")
            (run / "run.json").write_text(json.dumps({"completed": False}))
            with self.assertRaisesRegex(ValueError, "must be completed"):
                summarize(run)
            metadata = {"completed": True, "status": "complete", "phase": "complete", "reload": {
                "exact_state_tensors_equal": True, "gsm8k_first16_tokens_equal": False,
                "retention_all128_tokens_equal": True}}
            (run / "run.json").write_text(json.dumps(metadata))
            with self.assertRaisesRegex(ValueError, "reload check not passed"):
                summarize(run)


if __name__ == "__main__":
    unittest.main()
