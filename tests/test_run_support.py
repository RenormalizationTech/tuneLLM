import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.run_support import (choose_checkpoint, seal_snapshot, seeded_subset,
                                 select_training_rows, snapshot_sources, verify_snapshot)
from scripts.training_data import OverlengthError


class RunSelectionTests(unittest.TestCase):
    def test_final_subset_excludes_smoke_and_is_reproducible(self):
        rows = [{"id": str(i)} for i in range(200)]
        excluded = [str(i) for i in range(16)]
        selected = seeded_subset(rows, 128, 42, excluded)
        self.assertEqual(selected, seeded_subset(rows, 128, 42, excluded))
        self.assertEqual(len(selected), 128)
        self.assertFalse({r["id"] for r in selected} & set(excluded))
        self.assertEqual([selected[i:i + 8] for i in range(0, 16, 8)],
                         [selected[:16][i:i + 8] for i in range(0, 16, 8)])
        with self.assertRaises(ValueError):
            seeded_subset(rows, 200, 42, excluded)

    def test_overlength_excluded_before_seeded_train_subset(self):
        rows = [{"id": str(i)} for i in range(10)]

        def encode(row, *args):
            if row["id"] in {"1", "5"}:
                raise OverlengthError("too long")
            return {"labels": [1]}

        with patch("scripts.run_support.encode_row", side_effect=encode):
            selected, stats = select_training_rows(rows, None, 100, 4, 42)
        eligible = [row for row in rows if row["id"] not in {"1", "5"}]
        self.assertEqual(selected, seeded_subset(eligible, 4, 42))
        self.assertEqual(stats["eligible_rows"], 8)
        self.assertEqual(stats["overlength_excluded_ids"], ["1", "5"])
        with patch("scripts.run_support.encode_row", side_effect=ValueError("unsafe masking")):
            with self.assertRaisesRegex(ValueError, "unsafe masking"):
                select_training_rows(rows, None, 100, 4, 42)

    def test_only_32_64_validation_losses_select_with_earlier_tie(self):
        history = [{"step": 16, "eval_loss": 0.01}, {"step": 32, "eval_loss": 2},
                   {"step": 48, "eval_loss": 0.001}, {"step": 64, "eval_loss": 2}]
        self.assertEqual(choose_checkpoint(history)["selected_step"], 32)
        history[-1]["eval_loss"] = 1.9
        self.assertEqual(choose_checkpoint(history)["selected_step"], 64)
        with self.assertRaises(ValueError):
            choose_checkpoint(history[:-1])
        history[-1]["eval_loss"] = float("nan")
        with self.assertRaises(ValueError):
            choose_checkpoint(history)

    def test_snapshot_freezes_bytes_and_detects_source_or_input_drift(self):
        with tempfile.TemporaryDirectory() as temp:
            root, out = Path(temp) / "original", Path(temp) / "run"
            (root / "scripts").mkdir(parents=True)
            original = root / "scripts/example.py"
            original.write_text("value = 1\n")
            (out / "inputs").mkdir(parents=True)
            (out / "inputs/train.jsonl").write_text('{"id":"one"}\n')
            (out / "config.json").write_text('{"seed":42}\n')
            (out / "protocol.json").write_text('{"selected_ids":["one"]}\n')
            snapshot_sources(out / "source", root)
            try:
                seal_snapshot(out)
                verify_snapshot(out)
                original.write_text("value = 2\n")
                verify_snapshot(out)  # Live edits cannot change the frozen executable.
                frozen = out / "source/scripts/example.py"
                self.assertEqual(frozen.read_text(), "value = 1\n")
                frozen.chmod(0o644)
                frozen.write_text("value = 3\n")
                with self.assertRaisesRegex(ValueError, "source changed"):
                    verify_snapshot(out)
            finally:
                for directory, dirs, files in os.walk(out):
                    Path(directory).chmod(0o755)
                    for name in files:
                        (Path(directory) / name).chmod(0o644)


if __name__ == "__main__":
    unittest.main()
