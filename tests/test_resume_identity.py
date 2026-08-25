from __future__ import annotations

import importlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import torch

from robust_o2o.config import ExperimentConfig
from robust_o2o.experiment import resolve_resume_checkpoint
from robust_o2o.logging_utils import RunLogger


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


def _tree_bytes(root: Path) -> dict[str, bytes]:
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


class ResumeIdentityTest(unittest.TestCase):
    def test_single_run_cli_exposes_only_the_fixed_modern_runtime(self):
        cli = importlib.import_module("run_experiment")
        help_text = cli.build_parser().format_help()
        self.assertNotIn("--protocol", help_text)
        self.assertNotIn("--run-purpose", help_text)
        self.assertNotIn("--allow-diagnostic-protocol", help_text)
        self.assertNotIn("--benchmark-seed-set", help_text)

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


if __name__ == "__main__":
    unittest.main()
