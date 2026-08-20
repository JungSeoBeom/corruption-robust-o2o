from __future__ import annotations

import tempfile
import unittest
import logging
from pathlib import Path

from robust_o2o.config import ExperimentConfig
from robust_o2o.experiment import (
    _prune_periodic_checkpoints,
    _save_phase_checkpoint,
)


class _Agent:
    def checkpoint_state(self):
        return {"weight": 1.0}


class _Normalizer:
    def state_dict(self):
        return {"enabled": False}


class _Logger:
    def __init__(self, run_dir: Path):
        self.run_dir = run_dir
        self.logger = logging.getLogger("checkpoint-test")


class CheckpointConfigurationTest(unittest.TestCase):
    def test_phase_specific_periods_override_shared_period(self):
        config = ExperimentConfig(
            "rpex",
            "hopper-medium-replay-v2",
            checkpoint_period=100,
            offline_checkpoint_period=20,
            online_checkpoint_period=30,
        )
        self.assertEqual(config.effective_offline_checkpoint_period, 20)
        self.assertEqual(config.effective_online_checkpoint_period, 30)

    def test_periodic_checkpoint_retention(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for step in (10, 20, 30, 40):
                (root / f"step_{step:09d}.pt").touch()
            (root / "final.pt").touch()
            _prune_periodic_checkpoints(root, keep_last=2)
            self.assertEqual(
                [path.name for path in sorted(root.glob("step_*.pt"))],
                ["step_000000030.pt", "step_000000040.pt"],
            )
            self.assertTrue((root / "final.pt").exists())

    def test_checkpoints_are_separated_by_phase(self):
        config = ExperimentConfig(
            "rpex",
            "hopper-medium-replay-v2",
            keep_last_checkpoints=1,
        )
        with tempfile.TemporaryDirectory() as directory:
            logger = _Logger(Path(directory))
            for step in (10, 20):
                _save_phase_checkpoint(
                    logger,
                    _Agent(),
                    config,
                    _Normalizer(),
                    "offline",
                    step,
                    0,
                    5,
                    2,
                    final=False,
                )
            _save_phase_checkpoint(
                logger,
                _Agent(),
                config,
                _Normalizer(),
                "online",
                30,
                30,
                5,
                2,
                final=True,
            )
            self.assertFalse(
                (logger.run_dir / "checkpoints/offline/step_000000010.pt").exists()
            )
            self.assertTrue(
                any((logger.run_dir / "checkpoints/offline").glob("step_000000020_manifest_*.pt"))
            )
            self.assertTrue(
                any((logger.run_dir / "checkpoints/online").glob("final_manifest_*.pt"))
            )


if __name__ == "__main__":
    unittest.main()
