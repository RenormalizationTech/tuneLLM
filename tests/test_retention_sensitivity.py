import copy
import unittest
from scripts.retention_sensitivity import compare_caps, verify_settings


def saved(cap):
    details = [{"id": identifier, "strict_correct": False, "fallback_correct": True,
                "invalid": True, "fallback_invalid": False, "fallback_ambiguous": False,
                "hit_token_limit": True} for identifier in ("a", "b")]
    return {"settings": {"batch_size": 8, "max_new_tokens": cap, "seed": 42, "do_sample": False,
                         "num_beams": 1, "enable_thinking": False},
            "batches": [{"ids": ["a", "b"]}],
            "predictions": [{"id": identifier, "generated_ids": list(range(cap))} for identifier in ("a", "b")],
            "scores": {"details": details, "total": 2, "strict_correct": 0, "fallback_correct": 2,
                       "invalid": 2, "fallback_invalid": 0, "fallback_ambiguous": 0, "hit_token_limit": 2}}


class SensitivityTests(unittest.TestCase):
    def test_comparison_aligns_scores_and_lengths_by_id(self):
        primary, secondary = saved(8), saved(64)
        secondary["scores"]["details"].reverse()
        secondary["predictions"].reverse()
        secondary["scores"]["details"][0]["fallback_correct"] = False
        comparison = compare_caps(primary, secondary, ["a", "b"])
        self.assertEqual(comparison["details"][0]["fallback_correct_delta"], 0)
        self.assertEqual(comparison["details"][1]["fallback_correct_delta"], -1)
        self.assertEqual(comparison["details"][0]["cap8"]["token_budget"], 8)
        self.assertEqual(comparison["details"][0]["cap64"]["generated_tokens_including_eos"], 64)

    def test_settings_order_and_ids_cannot_drift(self):
        for key, value in (("max_new_tokens", 32), ("seed", 1234), ("batch_size", 1), ("do_sample", True)):
            result = saved(64)
            result["settings"][key] = value
            with self.assertRaisesRegex(ValueError, "settings must match"):
                verify_settings(result, 64, ["a", "b"])
        result = saved(64)
        result["batches"][0]["ids"].reverse()
        with self.assertRaisesRegex(ValueError, "membership/order"):
            verify_settings(result, 64, ["a", "b"])
        result = saved(64)
        result["scores"]["details"].pop()
        with self.assertRaisesRegex(ValueError, "IDs differ"):
            verify_settings(result, 64, ["a", "b"])


if __name__ == "__main__":
    unittest.main()
