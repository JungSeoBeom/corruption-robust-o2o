from __future__ import annotations

import csv
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import Mock, patch

from robust_o2o.logging_utils import format_duration, format_timestamp
from run_all_algorithms import main, summarize_algorithm_timings, write_timing_csv


class TimingTest(unittest.TestCase):
    def test_timestamp_is_clean_to_seconds(self):
        value = datetime(
            2026,
            7,
            31,
            15,
            25,
            59,
            299187,
            tzinfo=timezone(timedelta(hours=9)),
        )
        self.assertEqual(format_timestamp(value), "2026-07-31 15:25:59")
        self.assertEqual(format_duration(3661.4), "01:01:01")

    def test_algorithm_summary_and_timing_csv(self):
        records = [
            {
                "algorithm": "rpex",
                "algorithm_name": "RPEX",
                "seed": 0,
                "status": "completed",
                "start_time": "2026-07-31 10:00:00",
                "end_time": "2026-07-31 10:00:10",
                "elapsed_hms": "00:00:10",
                "elapsed_seconds": 10.0,
                "returncode": 0,
            },
            {
                "algorithm": "rpex",
                "algorithm_name": "RPEX",
                "seed": 1,
                "status": "completed",
                "start_time": "2026-07-31 10:00:10",
                "end_time": "2026-07-31 10:00:22",
                "elapsed_hms": "00:00:12",
                "elapsed_seconds": 12.0,
                "returncode": 0,
            },
        ]
        summary = summarize_algorithm_timings(records, ("rpex",))[0]
        self.assertEqual(summary["elapsed_seconds"], 22.0)
        self.assertEqual(summary["elapsed_hms"], "00:00:22")
        self.assertEqual(summary["completed_runs"], 2)

        with tempfile.TemporaryDirectory() as directory:
            output = write_timing_csv(Path(directory) / "timing.csv", records)
            with output.open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["algorithm_name"], "RPEX")
        self.assertEqual(rows[1]["elapsed_hms"], "00:00:12")

    def test_all_algorithm_runner_prints_and_saves_timing_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            arguments = [
                "run_all_algorithms.py",
                "--env-name",
                "hopper-medium-replay-v2",
                "--corruption",
                "clean",
                "--algorithms",
                "rpex,riql_pex",
                "--seeds",
                "0",
                "--output-root",
                directory,
            ]
            output = io.StringIO()
            with (
                patch.object(sys, "argv", arguments),
                patch(
                    "run_all_algorithms.subprocess.run",
                    return_value=Mock(returncode=0),
                ),
                patch(
                    "run_all_algorithms.preflight_runtime",
                    return_value={
                        "protocol": "rpex_d4rl_v2_legacy",
                        "d4rl_env_id": "hopper-medium-replay-v2",
                        "environment_backend": "gym-0.23.1+d4rl-v2+mujoco_py",
                        "dataset_backend": (
                            "d4rl.qlearning_dataset(terminate_on_end=False)"
                        ),
                        "dataset_path": str(Path(directory) / "dataset.hdf5"),
                    },
                ),
                patch("run_all_algorithms.update_comparison_plots", return_value={}),
                patch(
                    "run_all_algorithms.write_final_score_summary",
                    side_effect=lambda _root, path, *_args: path,
                ),
                redirect_stdout(output),
            ):
                returncode = main()

            self.assertEqual(returncode, 0)
            text = output.getvalue()
            self.assertIn("ALGORITHM_FINISHED: RPEX (rpex)", text)
            self.assertIn("ALGORITHM_FINISHED: RIQL+PEX (riql_pex)", text)
            self.assertIn("ALGORITHM_TIMING_SUMMARY:", text)
            self.assertRegex(text, r"START_TIME: \d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}")
            self.assertNotIn("T15:", text)
            timing_paths = list(Path(directory).rglob("timing.csv"))
            self.assertEqual(len(timing_paths), 1)
            with timing_paths[0].open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(
                [row["algorithm"] for row in rows], ["rpex", "riql_pex"]
            )
            manifest_paths = list(Path(directory).rglob("manifest.json"))
            self.assertEqual(len(manifest_paths), 1)
            manifest = json.loads(manifest_paths[0].read_text(encoding="utf-8"))
            self.assertEqual(manifest["protocol"], "rpex_d4rl_v2_legacy")
            self.assertEqual(
                manifest["environment_backend"],
                "gym-0.23.1+d4rl-v2+mujoco_py",
            )


if __name__ == "__main__":
    unittest.main()
