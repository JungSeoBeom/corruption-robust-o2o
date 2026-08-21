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

from robust_o2o.fidelity import MAIN_BASELINES, OPTIONAL_BASELINES
from robust_o2o.logging_utils import format_duration, format_timestamp
from run_all_algorithms import (
    _validate_args as validate_run_all_args,
    build_parser as build_run_all_parser,
    main,
    summarize_algorithm_timings,
    write_timing_csv,
)


class TimingTest(unittest.TestCase):
    def test_research_dry_run_defaults_to_main_three_without_side_effects(self):
        with tempfile.TemporaryDirectory() as directory:
            arguments = [
                "run_all_algorithms.py",
                "--env-name",
                "hopper-medium-replay-v2",
                "--corruption",
                "clean",
                "--run-purpose",
                "research_benchmark",
                "--suite-profile",
                "research_benchmark",
                "--output-root",
                directory,
                "--comparison-name",
                "research_dry_run",
                "--dry-run",
            ]
            output = io.StringIO()
            with (
                patch.object(sys, "argv", arguments),
                patch("run_all_algorithms.preflight_runtime") as preflight,
                patch("run_all_algorithms.subprocess.run") as run,
                patch("run_all_algorithms.write_reproduction_summaries") as reports,
                redirect_stdout(output),
            ):
                returncode = main()

            self.assertEqual(returncode, 0)
            preflight.assert_not_called()
            run.assert_not_called()
            reports.assert_not_called()
            command_lines = [
                line
                for line in output.getvalue().splitlines()
                if line.startswith("[")
            ]
            self.assertEqual(len(command_lines), len(MAIN_BASELINES))
            for line, algorithm in zip(command_lines, MAIN_BASELINES):
                self.assertIn(f"--algorithm {algorithm}", line)
                self.assertIn("--run-purpose research_benchmark", line)
                self.assertIn("--suite-profile research_benchmark", line)
                self.assertIn(
                    "--implementation-profile research_benchmark", line
                )
            for algorithm in OPTIONAL_BASELINES:
                self.assertNotIn(f"--algorithm {algorithm}", output.getvalue())
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_research_optional_algorithms_publish_role_summary_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            algorithms = (*MAIN_BASELINES, *OPTIONAL_BASELINES)
            arguments = [
                "run_all_algorithms.py",
                "--env-name",
                "hopper-medium-replay-v2",
                "--corruption",
                "clean",
                "--run-purpose",
                "research_benchmark",
                "--suite-profile",
                "research_benchmark",
                "--algorithms",
                ",".join(algorithms),
                "--output-root",
                directory,
                "--comparison-name",
                "research_roles",
            ]
            preflight = {
                "protocol": "rpex_d4rl_v2_legacy",
                "d4rl_env_id": "hopper-medium-replay-v2",
                "environment_backend": "gym-0.23.1+d4rl-v2+mujoco_py",
                "dataset_backend": (
                    "d4rl.qlearning_dataset(terminate_on_end=False)"
                ),
                "dataset_path": str(Path(directory) / "dataset.hdf5"),
            }

            def role_outputs(_root, output_dir, *_args, **_kwargs):
                return {
                    "research_summary": output_dir / "research_summary.csv",
                    "adapted_baselines_summary": output_dir
                    / "adapted_baselines_summary.csv",
                    "diagnostic_summary": output_dir / "diagnostic_summary.csv",
                }

            with (
                patch.object(sys, "argv", arguments),
                patch(
                    "run_all_algorithms.subprocess.run",
                    return_value=Mock(returncode=0),
                ) as run,
                patch(
                    "run_all_algorithms.preflight_runtime",
                    return_value=preflight,
                ),
                patch("run_all_algorithms.update_comparison_plots", return_value={}),
                patch(
                    "run_all_algorithms.write_final_score_summary",
                    side_effect=lambda _root, path, *_args: path,
                ),
                patch(
                    "run_all_algorithms.write_reproduction_summaries",
                    side_effect=role_outputs,
                ) as reports,
                redirect_stdout(io.StringIO()),
            ):
                returncode = main()

            self.assertEqual(returncode, 0)
            self.assertEqual(run.call_count, len(algorithms))
            reports.assert_called_once()
            self.assertFalse(reports.call_args.kwargs["strict"])
            self.assertEqual(reports.call_args.kwargs["phase"], "online")
            manifest_path = next(Path(directory).rglob("manifest.json"))
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["algorithms"], list(algorithms))
            self.assertEqual(
                set(manifest["artifacts"]["reporting_csvs"]),
                {
                    "research_summary",
                    "adapted_baselines_summary",
                    "diagnostic_summary",
                },
            )

    def test_final_adversarial_controller_rejects_non_hopper_fixture(self):
        parser = build_run_all_parser()
        args = parser.parse_args(
            [
                "--env-name",
                "halfcheetah-medium-replay-v2",
                "--corruption",
                "adversarial",
                "--corruption-target",
                "observations",
                "--algorithms",
                "rpex,riql_naive",
                "--seeds",
                "0,1,2,3,4",
                "--stage",
                "both",
                "--suite-profile",
                "primary_research_benchmark",
                "--run-purpose",
                "final_benchmark",
            ]
        )
        with self.assertRaises(SystemExit):
            validate_run_all_args(parser, args, ())

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
                "--stage",
                "offline",
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
                patch(
                    "run_all_algorithms.write_reproduction_summaries",
                    return_value={},
                ) as reporting_mock,
                redirect_stdout(output),
            ):
                returncode = main()

            self.assertEqual(returncode, 0)
            self.assertEqual(reporting_mock.call_args.kwargs["phase"], "offline")
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

    def test_failed_diagnostic_suite_does_not_publish_canonical_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            arguments = [
                "run_all_algorithms.py",
                "--env-name",
                "hopper-medium-replay-v2",
                "--corruption",
                "clean",
                "--algorithms",
                "rpex",
                "--seeds",
                "0",
                "--stage",
                "offline",
                "--output-root",
                directory,
            ]
            preflight = {
                "protocol": "rpex_d4rl_v2_legacy",
                "d4rl_env_id": "hopper-medium-replay-v2",
                "environment_backend": "gym-0.23.1+d4rl-v2+mujoco_py",
                "dataset_backend": "d4rl.qlearning_dataset(terminate_on_end=False)",
                "dataset_path": str(Path(directory) / "dataset.hdf5"),
            }
            canonical_names = (
                "final_scores.csv",
                "research_summary.csv",
                "adapted_baselines_summary.csv",
                "diagnostic_summary.csv",
                "comparison_online.png",
                "comparison_online.csv",
            )

            def failed_run(command, check):
                self.assertFalse(check)
                runs_dir = Path(command[command.index("--output-dir") + 1])
                comparison_dir = runs_dir.parent
                for name in canonical_names:
                    (comparison_dir / name).write_text(
                        "partial", encoding="utf-8"
                    )
                return Mock(returncode=1)

            with (
                patch.object(sys, "argv", arguments),
                patch(
                    "run_all_algorithms.subprocess.run",
                    side_effect=failed_run,
                ),
                patch(
                    "run_all_algorithms.preflight_runtime",
                    return_value=preflight,
                ),
                patch("run_all_algorithms.update_comparison_plots") as plots,
                patch("run_all_algorithms.write_final_score_summary") as scores,
                patch("run_all_algorithms.write_reproduction_summaries") as reports,
                redirect_stdout(io.StringIO()),
            ):
                returncode = main()

            self.assertEqual(returncode, 1)
            plots.assert_not_called()
            scores.assert_not_called()
            reports.assert_not_called()
            manifest_path = next(Path(directory).rglob("manifest.json"))
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertFalse(manifest["benchmark_valid"])
            self.assertIn("suite is incomplete", manifest["aggregation_error"])
            self.assertEqual(
                list(manifest_path.parent.glob("comparison_*")), []
            )
            for name in canonical_names:
                self.assertFalse((manifest_path.parent / name).exists())


if __name__ == "__main__":
    unittest.main()
