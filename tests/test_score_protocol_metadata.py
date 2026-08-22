from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from robust_o2o.config import (
    LEGACY_PROTOCOL,
    LEGACY_SCORE_SEMANTICS,
    LOCAL_PROTOCOL,
    LOCAL_SCORE_SEMANTICS,
    ExperimentConfig,
    build_parser,
    config_from_args,
)
from robust_o2o.logging_utils import RunLogger
from robust_o2o.manifest import build_experiment_manifest


def _research_config(protocol: str = LEGACY_PROTOCOL, **overrides) -> ExperimentConfig:
    values = {
        "algorithm": "rpex",
        "env_name": "hopper-medium-replay-v2",
        "run_purpose": "research_benchmark",
        "suite_profile": "research_benchmark",
        "implementation_profile": "research_benchmark",
        "protocol": protocol,
        "allow_diagnostic_protocol": protocol == LOCAL_PROTOCOL,
    }
    values.update(overrides)
    return ExperimentConfig(**values)


def _runtime_config(config: ExperimentConfig) -> dict:
    resolved = config.to_dict()
    resolved.update(
        environment_protocol=config.protocol,
        dataset_id=config.env_name,
        evaluation_env_id=(
            config.env_name if config.protocol == LEGACY_PROTOCOL else "Hopper-v4"
        ),
        online_env_id=(
            config.env_name if config.protocol == LEGACY_PROTOCOL else "Hopper-v4"
        ),
        dataset_sha256="dataset",
        normalizer_sha256="normalizer",
    )
    return resolved


class ScoreProtocolMetadataTest(unittest.TestCase):
    def test_single_run_cli_rejects_fresh_research_local_protocol(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            existing_parent = Path(directory) / "existing_comparison"
            existing_parent.mkdir()
            args = build_parser().parse_args(
                [
                    "--algorithm",
                    "rpex",
                    "--env-name",
                    "hopper-medium-replay-v2",
                    "--run-purpose",
                    "research_benchmark",
                    "--suite-profile",
                    "research_benchmark",
                    "--implementation-profile",
                    "research_benchmark",
                    "--protocol",
                    LOCAL_PROTOCOL,
                    "--allow-diagnostic-protocol",
                    "--output-dir",
                    str(existing_parent / "runs"),
                ]
            )
            with patch(
                "robust_o2o.config.is_inflight_pre_gate_run55_descendant",
                return_value=False,
            ), self.assertRaisesRegex(
                ValueError, "ResearchBenchmarkProtocolError"
            ):
                config_from_args(args)

    def test_single_run_cli_allows_only_pre_gate_process_continuation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            args = build_parser().parse_args(
                [
                    "--algorithm",
                    "rpex",
                    "--env-name",
                    "hopper-medium-replay-v2",
                    "--run-purpose",
                    "research_benchmark",
                    "--suite-profile",
                    "research_benchmark",
                    "--implementation-profile",
                    "research_benchmark",
                    "--protocol",
                    LOCAL_PROTOCOL,
                    "--allow-diagnostic-protocol",
                    "--output-dir",
                    str(Path(directory) / "runs"),
                ]
            )
            with patch(
                "robust_o2o.config.is_inflight_pre_gate_run55_descendant",
                return_value=True,
            ):
                config = config_from_args(args)
        self.assertEqual(config.protocol, LOCAL_PROTOCOL)

    def test_config_classifies_legacy_and_local_scores_explicitly(self) -> None:
        legacy = _research_config().to_dict()
        self.assertEqual(legacy["protocol"], LEGACY_PROTOCOL)
        self.assertEqual(legacy["run_purpose"], "research_benchmark")
        self.assertEqual(legacy["score_semantics"], LEGACY_SCORE_SEMANTICS)
        self.assertEqual(legacy["normalized_score_rule"], LEGACY_SCORE_SEMANTICS)
        self.assertTrue(legacy["benchmark_eligible"])

        local = ExperimentConfig(
            "rpex",
            "hopper-medium-replay-v2",
            protocol=LOCAL_PROTOCOL,
            allow_diagnostic_protocol=True,
        ).to_dict()
        self.assertEqual(local["protocol"], LOCAL_PROTOCOL)
        self.assertEqual(local["score_semantics"], LOCAL_SCORE_SEMANTICS)
        self.assertEqual(local["normalized_score_rule"], LOCAL_SCORE_SEMANTICS)
        self.assertFalse(local["benchmark_eligible"])

        with self.assertRaisesRegex(ValueError, "ResearchBenchmarkProtocolError"):
            _research_config(LOCAL_PROTOCOL)

    def test_manifest_records_ids_and_fails_closed_for_local_protocol(self) -> None:
        legacy = build_experiment_manifest(_runtime_config(_research_config()))
        self.assertEqual(legacy["protocol"], LEGACY_PROTOCOL)
        self.assertEqual(legacy["environment_protocol"], LEGACY_PROTOCOL)
        self.assertEqual(legacy["score_semantics"], LEGACY_SCORE_SEMANTICS)
        self.assertTrue(legacy["benchmark_eligible"])
        self.assertEqual(legacy["dataset_id"], "hopper-medium-replay-v2")
        self.assertEqual(legacy["evaluation_env_id"], "hopper-medium-replay-v2")

        local_config = ExperimentConfig(
            "rpex",
            "hopper-medium-replay-v2",
            protocol=LOCAL_PROTOCOL,
            allow_diagnostic_protocol=True,
        )
        local_resolved = _runtime_config(local_config)
        local_resolved["run_purpose"] = "research_benchmark"
        local_resolved["benchmark_eligible"] = True
        local = build_experiment_manifest(local_resolved)
        self.assertEqual(local["protocol"], LOCAL_PROTOCOL)
        self.assertEqual(local["score_semantics"], LOCAL_SCORE_SEMANTICS)
        self.assertFalse(local["benchmark_eligible"])
        self.assertEqual(local["dataset_id"], "hopper-medium-replay-v2")
        self.assertEqual(local["evaluation_env_id"], "Hopper-v4")

    def test_logger_labels_legacy_and_diagnostic_scores(self) -> None:
        metrics = {
            "return_mean": 1.0,
            "return_std": 0.5,
            "normalized_return_mean": 2.0,
            "normalized_return_std": 0.25,
            "evaluation_mode": "method_faithful",
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch("plot_results.update_live_comparison_plots"):
                legacy_config = _research_config(output_dir=str(root / "legacy"))
                legacy_logger = RunLogger(legacy_config)
                legacy_logger.write_config(_runtime_config(legacy_config))
                legacy_logger.log_evaluation("offline", 1, 0, 1, metrics)
                legacy_log_path = legacy_logger.run_dir / "result.log"
                legacy_metrics_path = legacy_logger.metrics_path
                legacy_logger.close()

                local_config = ExperimentConfig(
                    "rpex",
                    "hopper-medium-replay-v2",
                    protocol=LOCAL_PROTOCOL,
                    allow_diagnostic_protocol=True,
                    output_dir=str(root / "local"),
                )
                local_logger = RunLogger(local_config)
                local_resolved = _runtime_config(local_config)
                local_logger.write_config(local_resolved)
                # Rewriting the resolved config must not duplicate the banner.
                local_logger.write_config(local_resolved)
                local_logger.log_evaluation("online", 1, 1, 1, metrics)
                local_log_path = local_logger.run_dir / "result.log"
                local_metrics_path = local_logger.metrics_path
                local_logger.close()

            legacy_log = legacy_log_path.read_text(encoding="utf-8")
            self.assertIn("normalized=2.0±0.2 benchmark_eligible=true", legacy_log)

            local_log = local_log_path.read_text(encoding="utf-8")
            self.assertEqual(local_log.count("DIAGNOSTIC EVALUATION ONLY"), 1)
            self.assertIn(LOCAL_SCORE_SEMANTICS, local_log)
            self.assertIn(
                "Scores from this run are not comparable to the legacy D4RL-v2 "
                "research benchmark and will be excluded from research_summary.csv.",
                local_log,
            )
            self.assertIn(
                "diagnostic_scaled=2.0±0.2 benchmark_eligible=false", local_log
            )
            with legacy_metrics_path.open(newline="", encoding="utf-8") as stream:
                legacy_row = next(csv.DictReader(stream))
            with local_metrics_path.open(newline="", encoding="utf-8") as stream:
                local_row = next(csv.DictReader(stream))
            self.assertEqual(legacy_row["normalized_return_mean"], "2.0")
            self.assertEqual(
                legacy_row["diagnostic_d4rl_reference_scaled_return_mean"], ""
            )
            self.assertEqual(local_row["normalized_return_mean"], "")
            self.assertEqual(local_row["normalized_return_std"], "")
            self.assertEqual(
                local_row["diagnostic_d4rl_reference_scaled_return_mean"], "2.0"
            )
            self.assertEqual(
                local_row["diagnostic_d4rl_reference_scaled_return_std"], "0.25"
            )


if __name__ == "__main__":
    unittest.main()
