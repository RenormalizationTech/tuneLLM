import copy
import json
from pathlib import Path
import tempfile
import unittest

from scripts.physics_selection import (best_epoch, load_epochs, paired_bootstrap,
                                       screen_gates, select_candidates, selection_report)


def nll_report(losses, counts=None):
    counts = counts or [1] * len(losses)
    papers = {f"paper{i}": {"nll_sum": loss * count, "loss_tokens": count}
              for i, (loss, count) in enumerate(zip(losses, counts))}
    return {"token_weighted_nll": sum(row["nll_sum"] for row in papers.values()) / sum(counts), "per_paper": papers}


def candidate(identifier, rank, rate, losses, passed=True):
    report = nll_report(losses)
    return {"id": identifier, "rank": rank, "learning_rate": rate, "gate": {"passed": passed},
            "epoch": {"epoch": 2, "checkpoint": "unused/checkpoint-2", "report": report,
                      "token_weighted_nll": report["token_weighted_nll"]}}


def screen():
    return {"general_mc": {"total": 32, "correct": 20, "truncated": None},
            "physics_mc": {"total": 32, "correct": 20, "truncated": None},
            "gsm8k": {"total": 32, "correct": 20, "truncated": 3},
            "ifeval": {"total": 16, "correct": 12, "truncated": 3},
            "writing": {"total": 8, "nonempty": 8, "flags": 2, "truncated": 5}}


class SelectionTests(unittest.TestCase):
    def test_bootstrap_uses_paired_token_weighted_ratios_not_mean_paper_nll(self):
        a = nll_report([0, 10], [1, 9])
        b = nll_report([2, 2], [1, 9])
        result = paired_bootstrap(a, b)
        self.assertEqual(result["candidate_minus_reference_nll"], 7)
        self.assertEqual(result["ci95"], [-2, 8])
        self.assertEqual(result["resamples"], 10000)
        self.assertEqual(result["seed"], 20261112)
        self.assertEqual(result, paired_bootstrap(a, b))
        with self.assertRaisesRegex(ValueError, "target counts"):
            paired_bootstrap(a, nll_report([2, 2], [2, 8]))
        other = copy.deepcopy(b)
        other["per_paper"]["different"] = other["per_paper"].pop("paper0")
        with self.assertRaisesRegex(ValueError, "paper IDs"):
            paired_bootstrap(a, other)

    def test_lr_prefers_lower_if_ci_includes_zero_otherwise_best(self):
        high = candidate("high", 32, 1e-4, [1] * 10)
        low = candidate("low", 32, 5e-5, [0.5] * 5 + [2] * 5)
        result = select_candidates([low, high], "lr_pilot")
        self.assertTrue(result["comparisons"]["low"]["includes_zero"])
        self.assertEqual(result["selected_id"], "low")
        low = candidate("low", 32, 5e-5, [2] * 10)
        self.assertEqual(select_candidates([low, high], "lr_pilot")["selected_id"], "high")

    def test_rank_smallest_passing_compatible_and_all_fail_base(self):
        rows = [candidate("small", 8, 5e-5, [2] * 10),
                candidate("medium", 32, 5e-5, [1] * 10), candidate("large", 128, 5e-5, [1] * 10)]
        self.assertEqual(select_candidates(rows, "rank")["selected_id"], "medium")
        rows[0] = candidate("small", 8, 5e-5, [1] * 10)
        self.assertEqual(select_candidates(rows, "rank")["selected_id"], "small")
        for row in rows:
            row["gate"]["passed"] = False
        result = select_candidates(rows, "rank")
        self.assertTrue(result["fallback_base"])
        self.assertIsNone(result["selected_id"])

    def test_gate_boundaries_and_nonzero_base_truncation(self):
        base, after = screen(), screen()
        for task in ("general_mc", "physics_mc", "gsm8k"):
            after[task]["correct"] -= 4
        after["ifeval"]["correct"] -= 3
        for task in ("gsm8k", "ifeval", "writing"):
            after[task]["truncated"] += 2
        after["writing"]["flags"] += 2
        self.assertTrue(screen_gates(after, base, metadata={"note": "audit only"})["passed"])
        for task, key in (("gsm8k", "truncated"), ("ifeval", "truncated"), ("writing", "truncated"), ("writing", "flags")):
            failing = copy.deepcopy(after)
            failing[task][key] += 1
            self.assertFalse(screen_gates(failing, base)["passed"])
        failing = copy.deepcopy(after)
        failing["general_mc"]["correct"] -= 1
        self.assertFalse(screen_gates(failing, base)["passed"])
        failing = copy.deepcopy(after)
        failing["writing"]["nonempty"] -= 1
        self.assertFalse(screen_gates(failing, base)["passed"])

    def test_epoch_mapping_uses_learning_curve_and_earlier_tie(self):
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp)
            (run / "validation").mkdir()
            history = []
            for epoch, step, loss in ((1, 5, 2.0), (2, 10, 1.5), (3, 15, 1.5)):
                history.append({"step": step, "epoch": float(epoch), "eval_loss": loss})
                (run / f"validation/step-{step:06d}.json").write_text(json.dumps(nll_report([loss] * 10)))
                checkpoint = run / f"checkpoints/checkpoint-{step}"
                checkpoint.mkdir(parents=True)
                (checkpoint / "adapter_model.safetensors").write_bytes(b"test fixture, not model data")
            (run / "learning_curve.json").write_text(json.dumps(history))
            (run / "config.json").write_text(json.dumps({"epochs": 3}))
            chosen = best_epoch(load_epochs(run))
            self.assertEqual(chosen["epoch"], 2)
            self.assertTrue(chosen["checkpoint"].endswith("checkpoint-10"))
            history[-1]["eval_loss"] = 0.1
            (run / "learning_curve.json").write_text(json.dumps(history))
            with self.assertRaisesRegex(ValueError, "disagree"):
                load_epochs(run)

    def test_final_test_mapping_rejected_without_opening_inputs(self):
        with self.assertRaisesRegex(ValueError, "development inputs only"):
            selection_report({"split": "test"})


if __name__ == "__main__":
    unittest.main()
