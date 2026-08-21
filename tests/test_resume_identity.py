from __future__ import annotations

import importlib
import json
import os
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock, patch

import torch

from robust_o2o.config import ExperimentConfig, STRICT_FINAL_SEEDS
from robust_o2o.experiment import resolve_resume_checkpoint
from robust_o2o.fidelity import BASELINE_REPRODUCTION_REGISTRY
from robust_o2o.final_gate import AUDIT_RECEIPT_SHA256_ENV, RECEIPT_SCHEMA
from robust_o2o.logging_utils import METRIC_FIELDS, RunLogger
from robust_o2o.manifest import (
    aggregation_signature,
    build_experiment_manifest,
    resume_identity_signature,
)
from run_experiment import (
    _receipt_content_sha256,
    inherit_verified_final_resume_attestation,
)


def _resolved_config(config: ExperimentConfig) -> dict:
    resolved = config.to_dict()
    resolved.update(
        dataset_id=config.env_name,
        evaluation_env_id=config.env_name,
        online_env_id=config.env_name,
        dataset_sha256="dataset",
        normalizer_sha256="normalizer",
    )
    return resolved


def _verified_rpex_registry():
    """Install a test-only verified record to reach resume-specific gates."""

    verified = replace(
        BASELINE_REPRODUCTION_REGISTRY["rpex"],
        implementation_type="pinned upstream subprocess adapter",
        reproduction_status="official_adapter_verified",
        parity_status="official_adapter_verified",
        strict_final_eligible=True,
        remaining_deviation="none in synthetic strict-resume fixture",
    )
    return patch.dict(BASELINE_REPRODUCTION_REGISTRY, {"rpex": verified})


def _receipt(context_token: str, context: dict, origin_pid: int = 123) -> dict:
    return {
        "schema": RECEIPT_SCHEMA,
        "issued_at_utc": "2026-08-21T00:00:00+00:00",
        "origin_pid": origin_pid,
        "context_token": context_token,
        "context": context,
        "audit_command": ["python", "audit_reproducibility.py", "--json"],
        "audit_returncode": 0,
        "audit_stdout_sha256": "stdout",
        "audit_stderr_sha256": "stderr",
        "audit_result": {
            "reproducibility_audit": "PASS",
            "final_benchmark_status": "READY",
        },
    }


def _tree_bytes(root: Path) -> dict[str, bytes]:
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


class ResumeIdentityTest(unittest.TestCase):
    def test_receipt_is_hashed_provenance_but_not_resume_or_aggregation_identity(self):
        config = ExperimentConfig("wsrl", "hopper-medium-replay-v2")
        config._controller_seed_cohort_attested = True
        config._final_audit_context_token = "stable-context"
        config._final_audit_receipt_sha256 = "first-receipt"
        first = build_experiment_manifest(_resolved_config(config))

        config._final_audit_receipt_sha256 = "fresh-receipt"
        second = build_experiment_manifest(_resolved_config(config))

        self.assertNotEqual(first["manifest_sha256"], second["manifest_sha256"])
        self.assertEqual(
            resume_identity_signature(first), resume_identity_signature(second)
        )
        self.assertEqual(aggregation_signature(first), aggregation_signature(second))

        second["final_audit_context_token"] = "different-context"
        self.assertNotEqual(
            resume_identity_signature(first), resume_identity_signature(second)
        )

    def test_logger_accepts_fresh_receipt_without_rewriting_launch_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            original = ExperimentConfig(
                "wsrl",
                "hopper-medium-replay-v2",
                output_dir=directory,
                comparison_name="receipt_resume",
            )
            original._controller_seed_cohort_attested = True
            original._final_audit_context_token = "stable-context"
            original._final_audit_receipt_sha256 = "original-receipt"
            logger = RunLogger(original)
            logger.write_config(_resolved_config(original))
            launch_path = logger.run_dir / "experiment_manifest.json"
            original_launch = launch_path.read_bytes()
            prior_evaluation = {
                "return_mean": 12.0,
                "normalized_return_mean": 34.0,
            }
            (logger.run_dir / "summary.json").write_text(
                json.dumps(
                    {
                        "status": "completed",
                        "elapsed_seconds": 10.0,
                        "last_evaluation": prior_evaluation,
                    }
                ),
                encoding="utf-8",
            )
            logger.close()

            resumed = ExperimentConfig(
                "wsrl",
                "hopper-medium-replay-v2",
                output_dir=directory,
                comparison_name="receipt_resume",
                resume_run=str(logger.run_dir),
            )
            resumed._controller_seed_cohort_attested = True
            resumed._final_audit_context_token = "stable-context"
            resumed._final_audit_receipt_sha256 = "fresh-receipt"
            resumed_logger = RunLogger(resumed)
            resumed_logger.write_config(_resolved_config(resumed))

            self.assertTrue(resumed_logger.resume_committed)
            self.assertEqual(resumed_logger.last_eval, prior_evaluation)
            self.assertEqual(launch_path.read_bytes(), original_launch)
            event = json.loads(
                (logger.run_dir / "resume_events.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()[-1]
            )
            self.assertEqual(
                event["resume_final_audit_receipt_sha256"], "fresh-receipt"
            )
            resumed_logger.close()

    def test_foreign_or_legacy_exact_resume_checkpoint_is_read_only(self):
        with tempfile.TemporaryDirectory() as directory:
            config = ExperimentConfig(
                "wsrl",
                "hopper-medium-replay-v2",
                output_dir=directory,
                comparison_name="foreign_checkpoint",
            )
            logger = RunLogger(config)
            logger.write_config(_resolved_config(config))
            logger.close()
            checkpoint_dir = logger.run_dir / "checkpoints" / "offline"
            checkpoint_dir.mkdir(parents=True)

            for label, checkpoint_manifest in (
                ("foreign", "different-run-manifest"),
                ("legacy", None),
            ):
                with self.subTest(label=label):
                    for old_checkpoint in checkpoint_dir.glob("*.pt"):
                        old_checkpoint.unlink()
                    payload = {
                        "exact_resume_available": True,
                        "env_steps": 0,
                        "step": 100,
                    }
                    if checkpoint_manifest is not None:
                        payload["manifest_sha256"] = checkpoint_manifest
                    torch.save(payload, checkpoint_dir / f"{label}.pt")
                    before = _tree_bytes(logger.run_dir)

                    expected = (
                        "different run manifest"
                        if checkpoint_manifest is not None
                        else "no launch manifest SHA256"
                    )
                    with self.assertRaisesRegex(ValueError, expected):
                        resolve_resume_checkpoint(
                            str(logger.run_dir), torch.device("cpu")
                        )

                    self.assertEqual(_tree_bytes(logger.run_dir), before)

    def test_invalid_resume_identity_is_read_only(self):
        with tempfile.TemporaryDirectory() as directory:
            original = ExperimentConfig(
                "wsrl",
                "hopper-medium-replay-v2",
                output_dir=directory,
                comparison_name="invalid_resume",
            )
            logger = RunLogger(original)
            logger.write_config(_resolved_config(original))
            (logger.run_dir / "summary.json").write_text(
                json.dumps({"status": "completed", "marker": "untouched"}),
                encoding="utf-8",
            )
            logger.write_completion_manifest({"actual_online_steps": 500_000})
            logger.close()
            before = _tree_bytes(logger.run_dir)

            changed = ExperimentConfig(
                "wsrl",
                "hopper-medium-replay-v2",
                output_dir=directory,
                comparison_name="invalid_resume",
                resume_run=str(logger.run_dir),
                batch_size=257,
            )
            resumed_logger = RunLogger(changed)
            with self.assertRaisesRegex(ValueError, "does not match"):
                resumed_logger.write_config(_resolved_config(changed))
            with self.assertRaisesRegex(RuntimeError, "uncommitted resume"):
                resumed_logger.finish("failed")
            resumed_logger.close()

            self.assertEqual(_tree_bytes(logger.run_dir), before)

    def test_cli_precommit_failure_does_not_overwrite_original_run(self):
        with tempfile.TemporaryDirectory() as directory:
            original = ExperimentConfig(
                "wsrl",
                "hopper-medium-replay-v2",
                output_dir=directory,
                comparison_name="cli_resume",
            )
            logger = RunLogger(original)
            logger.write_config(_resolved_config(original))
            (logger.run_dir / "summary.json").write_text(
                json.dumps({"status": "completed", "marker": "untouched"}),
                encoding="utf-8",
            )
            logger.write_completion_manifest({"actual_online_steps": 500_000})
            logger.close()
            before = _tree_bytes(logger.run_dir)

            resumed = ExperimentConfig(
                "wsrl",
                "hopper-medium-replay-v2",
                output_dir=directory,
                comparison_name="cli_resume",
                resume_run=str(logger.run_dir),
            )
            cli = importlib.import_module("run_experiment")
            parser = Mock()
            parser.parse_args.return_value = Mock()
            with (
                patch.object(cli, "build_parser", return_value=parser),
                patch.object(cli, "config_from_args", return_value=resumed),
                patch.object(
                    cli,
                    "run_experiment",
                    side_effect=ValueError("precommit writer-position mismatch"),
                ),
            ):
                self.assertEqual(cli.main(), 1)

            self.assertEqual(_tree_bytes(logger.run_dir), before)

    def test_new_direct_final_run_is_rejected_before_output_creation(self):
        with tempfile.TemporaryDirectory() as directory:
            with _verified_rpex_registry():
                config = ExperimentConfig(
                    "rpex",
                    "hopper-medium-replay-v2",
                    corruption="random",
                    corruption_target="observations",
                    suite_profile="primary_research_benchmark",
                    run_purpose="final_benchmark",
                    benchmark_seed_set=STRICT_FINAL_SEEDS,
                    output_dir=directory,
                    comparison_name="must_not_exist",
                )
                context = {"git_head": "abc", "python_executable": "/python"}
                receipt = _receipt("stable-context", context, origin_pid=os.getpid())
                receipt_digest = _receipt_content_sha256(receipt)
                cli = importlib.import_module("run_experiment")
                parser = Mock()
                parser.parse_args.return_value = Mock()
                with (
                    patch.object(cli, "build_parser", return_value=parser),
                    patch.object(cli, "config_from_args", return_value=config),
                    patch(
                        "robust_o2o.final_gate.require_final_benchmark_audit",
                        return_value=receipt,
                    ),
                    patch.dict(
                        os.environ,
                        {AUDIT_RECEIPT_SHA256_ENV: receipt_digest},
                        clear=False,
                    ),
                ):
                    self.assertEqual(cli.main(), 1)

            self.assertEqual(list(Path(directory).iterdir()), [])

    def _write_attested_final_run(
        self, root: Path, *, context_token: str, context: dict
    ) -> tuple[Path, dict]:
        run_dir = root / "comparison" / "runs" / "rpex" / "run"
        run_dir.mkdir(parents=True)
        (run_dir / "metrics.csv").write_text(
            ",".join(METRIC_FIELDS) + "\n", encoding="utf-8"
        )
        launch_receipt = _receipt(context_token, context)
        receipt_digest = _receipt_content_sha256(launch_receipt)
        with _verified_rpex_registry():
            config = ExperimentConfig(
                "rpex",
                "hopper-medium-replay-v2",
                corruption="random",
                corruption_target="observations",
                suite_profile="primary_research_benchmark",
                run_purpose="final_benchmark",
                benchmark_seed_set=STRICT_FINAL_SEEDS,
            )
            config._controller_seed_cohort_attested = True
            config._final_audit_context_token = context_token
            config._final_audit_receipt_sha256 = receipt_digest
            resolved = _resolved_config(config)
            resolved.update(
                condition_certificate_verified=True,
                strict_runtime_fixture_verified=True,
                strict_environment_preflight_verified=True,
                save_resume_certificate_verified=True,
                audit_receipt_verified=True,
                repository_worktree_clean=True,
                dataset_hash_recorded=True,
                required_checkpoint_hashes_verified=True,
                paper_reproduction_eligible=True,
                publication_eligible=True,
            )
            manifest = build_experiment_manifest(resolved)
        (run_dir / "experiment_manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )
        evidence = {
            **launch_receipt,
            "receipt_source": "/tmp/original/receipt.json",
            "receipt_sha256": receipt_digest,
        }
        (run_dir / "final_audit_evidence.json").write_text(
            json.dumps(evidence), encoding="utf-8"
        )
        return run_dir, launch_receipt

    def test_final_resume_safely_inherits_original_cohort_attestation(self):
        with tempfile.TemporaryDirectory() as directory:
            context = {"git_head": "abc", "python_executable": "/python"}
            run_dir, _ = self._write_attested_final_run(
                Path(directory), context_token="stable-context", context=context
            )
            with _verified_rpex_registry():
                resumed = ExperimentConfig(
                    "rpex",
                    "hopper-medium-replay-v2",
                    corruption="random",
                    corruption_target="observations",
                    suite_profile="primary_research_benchmark",
                    run_purpose="final_benchmark",
                    benchmark_seed_set=STRICT_FINAL_SEEDS,
                    resume_run=str(run_dir),
                )
            resumed._controller_seed_cohort_attested = False
            resumed._final_audit_context_token = "stable-context"
            fresh = _receipt("stable-context", context, origin_pid=999)
            fresh_digest = _receipt_content_sha256(fresh)
            resumed._final_audit_receipt_sha256 = fresh_digest

            inherit_verified_final_resume_attestation(resumed, fresh)

            self.assertTrue(resumed._controller_seed_cohort_attested)
            self.assertEqual(
                resumed._final_audit_context_token, "stable-context"
            )
            self.assertEqual(
                resumed._final_audit_receipt_sha256, fresh_digest
            )

    def test_final_resume_rejects_context_change_without_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            context = {"git_head": "abc", "python_executable": "/python"}
            run_dir, _ = self._write_attested_final_run(
                Path(directory), context_token="stable-context", context=context
            )
            with _verified_rpex_registry():
                resumed = ExperimentConfig(
                    "rpex",
                    "hopper-medium-replay-v2",
                    corruption="random",
                    corruption_target="observations",
                    suite_profile="primary_research_benchmark",
                    run_purpose="final_benchmark",
                    benchmark_seed_set=STRICT_FINAL_SEEDS,
                    resume_run=str(run_dir),
                )
            before = _tree_bytes(run_dir)
            changed_receipt = _receipt(
                "different-context",
                {"git_head": "changed", "python_executable": "/python"},
            )
            resumed._final_audit_receipt_sha256 = _receipt_content_sha256(
                changed_receipt
            )
            with self.assertRaisesRegex(ValueError, "context differs"):
                inherit_verified_final_resume_attestation(
                    resumed,
                    changed_receipt,
                )
            self.assertEqual(_tree_bytes(run_dir), before)


if __name__ == "__main__":
    unittest.main()
