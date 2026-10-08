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

from robust_o2o.config import DEFAULT_PROTOCOL
from robust_o2o.fidelity import MAIN_BASELINES
from robust_o2o.logging_utils import format_duration, format_timestamp
from run_all_algorithms import (
    _comparison_directory,
    _validate_args as validate_run_all_args,
    build_parser as build_run_all_parser,
    main,
    summarize_algorithm_timings,
    write_timing_csv,
)


def _preflight(root: str) -> dict:
    return {
        "protocol": DEFAULT_PROTOCOL,
        "environment_id": "hopper-medium-replay-v2",
        "d4rl_env_id": "hopper-medium-replay-v2",
        "environment_backend": "gymnasium-v4+native-mujoco",
        "dataset_backend": "d4rl-v2-hdf5+index-aware-qlearning-conversion",
        "dataset_path": str(Path(root) / "dataset.hdf5"),
    }


class TimingTest(unittest.TestCase):
    def test_run_all_uses_compact_comparison_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            args = build_run_all_parser().parse_args(
                [
                    "--env-name",
                    "halfcheetah-medium-replay-v2",
                    "--corruption",
                    "random",
                    "--corruption-target",
                    "observations",
                    "--output-root",
                    directory,
                    "--comparison-name",
                    "halfcheetah_5x5_test",
                ]
            )
            self.assertEqual(
                _comparison_directory(args),
                Path(directory).resolve()
                / "comparisons"
                / "halfcheetah-medium-replay-v2"
                / "random"
                / "observations"
                / "halfcheetah_5x5_test",
            )

    def test_legacy_protocol_controls_are_not_public_options(self):
        parser = build_run_all_parser()
        help_text = parser.format_help()
        self.assertNotIn("--protocol", help_text)
        self.assertNotIn("--run-purpose", help_text)
        self.assertNotIn("--allow-diagnostic-protocol", help_text)
        self.assertNotIn("--evaluation-mode", help_text)
        args, passthrough = parser.parse_known_args(
            [
                "--env-name",
                "hopper-medium-replay-v2",
                "--corruption",
                "clean",
                "--protocol",
                "legacy",
            ]
        )
        with self.assertRaises(SystemExit):
            validate_run_all_args(parser, args, passthrough)

        args, passthrough = parser.parse_known_args(
            [
                "--env-name",
                "hopper-medium-replay-v2",
                "--corruption",
                "clean",
                "--evaluation-mode",
                "deterministic",
            ]
        )
        with self.assertRaises(SystemExit):
            validate_run_all_args(parser, args, passthrough)

    def test_research_dry_run_defaults_to_main_five_without_side_effects(self):
        with tempfile.TemporaryDirectory() as directory:
            arguments = [
                "run_all_algorithms.py",
                "--env-name",
                "hopper-medium-replay-v2",
                "--corruption",
                "clean",
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
                redirect_stdout(output),
            ):
                returncode = main()

            self.assertEqual(returncode, 0)
            preflight.assert_not_called()
            run.assert_not_called()
            command_lines = [
                line for line in output.getvalue().splitlines() if line.startswith("[")
            ]
            self.assertEqual(len(command_lines), len(MAIN_BASELINES))
            for line, algorithm in zip(command_lines, MAIN_BASELINES):
                self.assertIn(f"--algorithm {algorithm}", line)
                self.assertIn("--suite-profile research_benchmark", line)
                self.assertIn("--implementation-profile research_benchmark", line)
                self.assertIn("--corruption-profile riql_rpex_code", line)
                self.assertNotIn("--protocol", line)
                self.assertNotIn("--run-purpose", line)
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_run_all_accepts_space_comma_mix_and_canonical_aliases(self):
        parser = build_run_all_parser()
        args = parser.parse_args(
            [
                "--env-name",
                "hopper-medium-replay-v2",
                "--corruption",
                "clean",
                "--suite-profile",
                "research_benchmark",
                "--algorithms",
                "rpex,riql_naive",
                "wsrl",
                "cal-ql",
                "pessimistic-q-ensemble",
            ]
        )
        validate_run_all_args(parser, args, ())
        self.assertEqual(tuple(args.algorithms), MAIN_BASELINES)

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
                    return_value=_preflight(directory),
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
            timing_path = next(Path(directory).rglob("timing.csv"))
            with timing_path.open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual([row["algorithm"] for row in rows], ["rpex", "riql_pex"])
            manifest_path = next(Path(directory).rglob("manifest.json"))
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["protocol"], DEFAULT_PROTOCOL)
            self.assertEqual(
                manifest["environment_backend"], "gymnasium-v4+native-mujoco"
            )

    def test_failed_suite_keeps_timing_but_skips_aggregation(self):
        with tempfile.TemporaryDirectory() as directory:
            arguments = [
                "run_all_algorithms.py",
                "--env-name",
                "hopper-medium-replay-v2",
                "--corruption",
                "clean",
                "--algorithms",
                "rpex",
                "--stage",
                "offline",
                "--output-root",
                directory,
            ]
            with (
                patch.object(sys, "argv", arguments),
                patch(
                    "run_all_algorithms.subprocess.run",
                    return_value=Mock(returncode=1),
                ),
                patch(
                    "run_all_algorithms.preflight_runtime",
                    return_value=_preflight(directory),
                ),
                patch("run_all_algorithms.update_comparison_plots") as plots,
                patch("run_all_algorithms.write_final_score_summary") as scores,
                redirect_stdout(io.StringIO()),
            ):
                returncode = main()

            self.assertEqual(returncode, 1)
            plots.assert_not_called()
            scores.assert_not_called()
            manifest_path = next(Path(directory).rglob("manifest.json"))
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertFalse(manifest["benchmark_valid"])
            self.assertTrue((manifest_path.parent / "timing.csv").is_file())


if __name__ == "__main__":
    unittest.main()
