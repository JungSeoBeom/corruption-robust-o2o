from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from plot_results import (
    plot_aggregate,
    update_comparison_plots,
    write_final_score_summary,
)
from robust_o2o.logging_utils import METRIC_FIELDS


class AggregateResultsTest(unittest.TestCase):
    def _make_run(
        self,
        root: Path,
        algorithm: str,
        seed: int,
        final_score: float,
        elapsed: float,
    ) -> None:
        run_dir = root / algorithm / f"seed_{seed}"
        run_dir.mkdir(parents=True)
        config = {
            "algorithm": algorithm,
            "env_name": "hopper-medium-replay-v2",
            "corruption": "random",
            "corruption_target": "mixed",
            "seed": seed,
        }
        (run_dir / "config.json").write_text(
            json.dumps(config), encoding="utf-8"
        )
        (run_dir / "summary.json").write_text(
            json.dumps({"status": "completed", "elapsed_seconds": elapsed}),
            encoding="utf-8",
        )
        with (run_dir / "metrics.csv").open(
            "w", newline="", encoding="utf-8"
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=METRIC_FIELDS)
            writer.writeheader()
            for step, score in ((100, final_score - 1.0), (200, final_score)):
                writer.writerow(
                    {
                        "phase": "online",
                        "step": step,
                        "env_steps": step,
                        "updates": step,
                        "return_mean": score * 10.0,
                        "return_std": 1.0,
                        "normalized_return_mean": score,
                        "normalized_return_std": 0.5,
                    }
                )

    def test_plot_and_final_score_csv_cover_all_algorithms(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._make_run(root, "rpex", 0, 80.0, 12.0)
            self._make_run(root, "uwmsg", 0, 70.0, 10.0)
            plot = plot_aggregate(
                root,
                root / "comparison.png",
                env_name="hopper-medium-replay-v2",
                corruption="random",
                target="mixed",
            )
            final_scores = write_final_score_summary(
                root,
                root / "final_scores.csv",
                env_name="hopper-medium-replay-v2",
                corruption="random",
                target="mixed",
            )
            self.assertTrue(plot.exists())
            self.assertTrue(plot.with_suffix(".csv").exists())
            with final_scores.open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual([row["algorithm"] for row in rows], ["rpex", "uwmsg"])
            self.assertEqual(float(rows[0]["elapsed_seconds_mean"]), 12.0)

    def test_three_comparison_plots_are_written(self):
        with tempfile.TemporaryDirectory() as directory:
            comparison_dir = Path(directory)
            self._make_run(comparison_dir / "runs", "rpex", 0, 80.0, 12.0)
            outputs = update_comparison_plots(
                comparison_dir,
                "hopper-medium-replay-v2",
                "random",
                "mixed",
            )
            self.assertEqual(set(outputs), {"offline_online", "offline", "online"})
            for output in outputs.values():
                self.assertTrue(output.exists())
                self.assertTrue(output.with_suffix(".csv").exists())


if __name__ == "__main__":
    unittest.main()
