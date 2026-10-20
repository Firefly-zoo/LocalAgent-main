from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "training" / "scripts"))

from make_validation_config import _is_visibly_test_path  # noqa: E402
from select_checkpoint import collect_metrics, select_best  # noqa: E402


class CheckpointSelectionTests(unittest.TestCase):
    def test_highest_validation_accuracy_wins(self) -> None:
        records = [
            {"step": 50, "metric": 0.81},
            {"step": 150, "metric": 0.83},
            {"step": 250, "metric": 0.82},
        ]
        self.assertEqual(select_best(records)["step"], 150)

    def test_earliest_step_breaks_tie(self) -> None:
        records = [{"step": 50, "metric": 0.83}, {"step": 150, "metric": 0.83}]
        self.assertEqual(select_best(records)["step"], 50)

    def test_test_named_paths_are_rejected(self) -> None:
        self.assertTrue(_is_visibly_test_path(Path("data/fgvqa/test-00000.parquet")))
        self.assertFalse(_is_visibly_test_path(Path("data/fgvqa/validation.parquet")))

    def test_metrics_require_validation_marker(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            metrics_path = Path(tmp_dir) / "checkpoint-50" / "metrics.json"
            metrics_path.parent.mkdir()
            metrics_path.write_text(
                json.dumps({
                    "status": "completed",
                    "num_samples": 10,
                    "acc_plain": 0.8,
                    "config": {"dataset_path": "validation.parquet", "question_scope": "pairwise"},
                }),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "validation-only"):
                collect_metrics(Path(tmp_dir), "acc_plain")


if __name__ == "__main__":
    unittest.main()
