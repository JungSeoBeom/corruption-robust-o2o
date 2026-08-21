from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from plot_results import (
    load_runs,
    plot_aggregate,
    update_comparison_plots,
    update_live_comparison_plots,
    write_final_score_summary,
    write_reproduction_summaries,
)
from robust_o2o.config import ExperimentConfig
from robust_o2o.fidelity import canonical_json_sha256
from robust_o2o.logging_utils import METRIC_FIELDS
from robust_o2o.manifest import build_experiment_manifest
from robust_o2o.reporting import ReportingValidationError


class AggregateResultsTest(unittest.TestCase):
    def _make_run(
        self,
        root: Path,
        algorithm: str,
        seed: int,
        final_score: float,
        elapsed: float,
        phase: str = "online",
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
                        "phase": phase,
                        "step": step,
                        "env_steps": step,
                        "updates": step,
                        "return_mean": score * 10.0,
                        "return_std": 1.0,
                        "normalized_return_mean": score,
                        "normalized_return_std": 0.5,
                    }
                )

    def _make_manifested_run(self, root: Path) -> Path:
        self._make_run(root, "rpex", 0, 80.0, 12.0)
        run_dir = root / "rpex" / "seed_0"
        config = ExperimentConfig(
            "rpex",
            "hopper-medium-replay-v2",
            corruption="random",
            corruption_target="observations",
            implementation_profile="official_code_reference",
            offline_steps=40_000,
            online_steps=40_000,
            eval_period=10_000,
        ).to_dict()
        config.update(
            dataset_id="hopper-medium-replay-v2",
            evaluation_env_id="hopper-medium-replay-v2",
            online_env_id="hopper-medium-replay-v2",
            environment_protocol="rpex_d4rl_v2_legacy",
            environment_max_episode_steps=1_000,
            dataset_sha256="dataset",
            normalizer_sha256="normalizer",
            score_semantics="d4rl_normalized_return",
        )
        (run_dir / "config.json").write_text(
            json.dumps(config), encoding="utf-8"
        )
        launch = build_experiment_manifest(config)
        (run_dir / "experiment_manifest.json").write_text(
            json.dumps(launch), encoding="utf-8"
        )
        completion = {
            **launch,
            "launch_manifest_sha256": launch["manifest_sha256"],
            "requested_online_steps": 40_000,
            "actual_online_steps": 40_001,
            "episode_boundary_overshoot": 1,
            "online_budget_semantics": (
                "rpex_official_episode_boundary_strict_greater_than"
            ),
        }
        completion["completion_manifest_sha256"] = canonical_json_sha256(
            completion
        )
        (run_dir / "completed_experiment_manifest.json").write_text(
            json.dumps(completion), encoding="utf-8"
        )
        return run_dir

    def _make_calql_manifested_run(self, root: Path) -> Path:
        self._make_run(root, "cal_ql", 0, 70.0, 8.0)
        run_dir = root / "cal_ql" / "seed_0"
        config = ExperimentConfig(
            "cal_ql",
            "hopper-medium-replay-v2",
            corruption="random",
            corruption_target="observations",
            run_purpose="research_benchmark",
            suite_profile="research_benchmark",
            implementation_profile="research_benchmark",
            offline_steps=200,
            online_steps=200,
            eval_period=100,
            final_window_size=2,
        ).to_dict()
        config.update(
            dataset_id="hopper-medium-replay-v2",
            evaluation_env_id="hopper-medium-replay-v2",
            online_env_id="hopper-medium-replay-v2",
            environment_protocol="rpex_d4rl_v2_legacy",
            environment_max_episode_steps=1_000,
            dataset_sha256="dataset",
            normalizer_sha256="normalizer",
            score_semantics="d4rl_normalized_return",
        )
        (run_dir / "config.json").write_text(
            json.dumps(config), encoding="utf-8"
        )
        launch = build_experiment_manifest(config)
        (run_dir / "experiment_manifest.json").write_text(
            json.dumps(launch), encoding="utf-8"
        )
        completion = {
            **launch,
            "launch_manifest_sha256": launch["manifest_sha256"],
            "requested_online_steps": 200,
            "actual_online_steps": 203,
            "episode_boundary_overshoot": 3,
            "online_budget_semantics": (
                "calql_complete_current_episode_at_or_after_requested"
            ),
            "completed_online_transitions": 203,
            "pending_episode_length": 0,
            "effective_calql_training_transitions": 203,
        }
        completion["completion_manifest_sha256"] = canonical_json_sha256(
            completion
        )
        (run_dir / "completed_experiment_manifest.json").write_text(
            json.dumps(completion), encoding="utf-8"
        )
        return run_dir

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
            self.assertEqual(
                rows[0]["aggregation_rule"],
                "common_mean_last_3_online_evaluations_per_seed_then_population_mean_std__partial_2_of_3",
            )

    def test_historical_result_algorithm_names_load_as_canonical(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._make_run(
                root, "cal_ql_locomotion_adaptation", 0, 70.0, 2.0
            )
            self._make_run(
                root, "pqe_shared_actor_approx", 0, 60.0, 3.0
            )
            frame = load_runs(root)
        self.assertEqual(
            set(frame["algorithm"]),
            {"cal_ql", "pessimistic_q_ensemble"},
        )

    def test_completed_nan_curve_fails_instead_of_being_silently_dropped(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._make_run(root, "rpex", 0, float("nan"), 2.0)
            with self.assertRaisesRegex(RuntimeError, "NaN or Inf"):
                plot_aggregate(
                    root,
                    root / "comparison.png",
                    env_name="hopper-medium-replay-v2",
                    corruption="random",
                    target="mixed",
                )

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

    def test_live_refresh_never_publishes_canonical_plot_names(self):
        with tempfile.TemporaryDirectory() as directory:
            comparison_dir = Path(directory)
            self._make_run(comparison_dir / "runs", "rpex", 0, 80.0, 12.0)
            outputs = update_live_comparison_plots(
                comparison_dir,
                "hopper-medium-replay-v2",
                "random",
                "mixed",
            )
            for output in outputs.values():
                self.assertTrue(output.name.startswith("diagnostic_running_"))
                self.assertTrue(output.exists())
            self.assertFalse((comparison_dir / "comparison_online.png").exists())

    def test_verified_completion_is_source_for_actual_online_steps(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._make_manifested_run(root)
            frame = load_runs(root)
            self.assertEqual(set(frame["actual_online_steps"]), {40_001})

    def test_calql_fairness_outcomes_load_from_verified_completion(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._make_calql_manifested_run(root)
            frame = load_runs(root)
        expected = {
            "requested_online_steps": 200,
            "actual_online_steps": 203,
            "episode_boundary_overshoot": 3,
            "completed_online_transitions": 203,
            "pending_episode_length": 0,
            "effective_calql_training_transitions": 203,
        }
        for field, value in expected.items():
            self.assertEqual(set(frame[field]), {value}, field)

    def test_manifest_config_and_online_outcome_mismatches_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = self._make_manifested_run(root)
            config = json.loads((run_dir / "config.json").read_text())
            config["seed"] = 9
            (run_dir / "config.json").write_text(json.dumps(config))
            with self.assertRaisesRegex(RuntimeError, "config/manifest"):
                load_runs(root)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = self._make_manifested_run(root)
            (run_dir / "online_corruption_manifest.json").write_text(
                json.dumps({"actual_online_steps": 40_002})
            )
            with self.assertRaisesRegex(RuntimeError, "actual-step mismatch"):
                load_runs(root)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = self._make_manifested_run(root)
            completion_path = run_dir / "completed_experiment_manifest.json"
            completion = json.loads(completion_path.read_text())
            completion["publication_eligible"] = not bool(
                completion["publication_eligible"]
            )
            completion.pop("completion_manifest_sha256")
            completion["completion_manifest_sha256"] = canonical_json_sha256(
                completion
            )
            completion_path.write_text(json.dumps(completion))
            with self.assertRaisesRegex(RuntimeError, "immutable launch fields"):
                load_runs(root)

    def test_completed_current_run_requires_completion_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = self._make_manifested_run(root)
            (run_dir / "completed_experiment_manifest.json").unlink()
            with self.assertRaisesRegex(RuntimeError, "no verified completion"):
                load_runs(root)

    def test_offline_only_reproduction_summary_does_not_require_online_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._make_run(
                root / "runs", "rpex", 0, 80.0, 12.0, phase="offline"
            )
            outputs = write_reproduction_summaries(
                root / "runs",
                root,
                env_name="hopper-medium-replay-v2",
                corruption="random",
                target="mixed",
                phase="offline",
            )
            with outputs["paper_reproduction_summary"].open(
                newline="", encoding="utf-8"
            ) as stream:
                paper_rows = list(csv.DictReader(stream))
            with outputs["common_benchmark_summary"].open(
                newline="", encoding="utf-8"
            ) as stream:
                common_rows = list(csv.DictReader(stream))
        self.assertEqual(paper_rows, [])
        self.assertEqual(len(common_rows), 1)
        self.assertIn("last_3_offline", common_rows[0]["aggregation_rule"])

    def test_empty_reproduction_summary_is_allowed_only_non_strict(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            outputs = write_reproduction_summaries(
                root / "missing-runs",
                root,
                env_name="hopper-medium-replay-v2",
                corruption="clean",
                target="none",
                strict=False,
            )
            with outputs["common_benchmark_summary"].open(
                newline="", encoding="utf-8"
            ) as stream:
                self.assertEqual(list(csv.DictReader(stream)), [])
            with self.assertRaisesRegex(
                ReportingValidationError, "no evaluation rows"
            ):
                write_reproduction_summaries(
                    root / "missing-runs",
                    root,
                    strict=True,
                    expected_seeds=[0],
                )


if __name__ == "__main__":
    unittest.main()
