from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from robust_o2o.config import ExperimentConfig
from robust_o2o.corruption import corrupt_offline_dataset


class CorruptionTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(1)
        self.dataset = {
            "observations": rng.normal(size=(100, 3)).astype(np.float32),
            "actions": rng.normal(size=(100, 2)).astype(np.float32),
            "rewards": rng.normal(size=100).astype(np.float32),
            "next_observations": rng.normal(size=(100, 3)).astype(np.float32),
            "terminals": np.zeros(100, dtype=np.float32),
            "mc_returns": np.zeros(100, dtype=np.float32),
            "episode_id": np.arange(100, dtype=np.int64),
        }

    def test_clean_is_unchanged(self):
        config = ExperimentConfig("rpex", "hopper-medium-replay-v2")
        with tempfile.TemporaryDirectory() as directory:
            result, stats = corrupt_offline_dataset(
                self.dataset, config, None, Path(directory)
            )
        np.testing.assert_array_equal(result["rewards"], self.dataset["rewards"])
        self.assertEqual(stats["corrupted_count"], 0)

    def test_random_reward_is_reproducible(self):
        config = ExperimentConfig(
            "rpex",
            "hopper-medium-replay-v2",
            corruption="random",
            corruption_target="rewards",
            seed=7,
        )
        with tempfile.TemporaryDirectory() as directory:
            first, _ = corrupt_offline_dataset(
                self.dataset, config, None, Path(directory)
            )
            second, stats = corrupt_offline_dataset(
                self.dataset, config, None, Path(directory)
            )
        np.testing.assert_array_equal(first["rewards"], second["rewards"])
        self.assertEqual(stats["loaded_from_cache"], 1.0)

    def test_corruption_seed_is_independent_of_learner_seed(self):
        common = dict(
            algorithm="rpex",
            env_name="hopper-medium-replay-v2",
            corruption="random",
            corruption_target="observations",
            corruption_seed=19,
        )
        first_config = ExperimentConfig(**common, learner_seed=1)
        second_config = ExperimentConfig(**common, learner_seed=999)
        with tempfile.TemporaryDirectory() as first_dir, tempfile.TemporaryDirectory() as second_dir:
            first, _ = corrupt_offline_dataset(
                self.dataset, first_config, None, Path(first_dir)
            )
            second, _ = corrupt_offline_dataset(
                self.dataset, second_config, None, Path(second_dir)
            )
        np.testing.assert_array_equal(first["observations"], second["observations"])
        np.testing.assert_array_equal(
            first["mc_calibration_valid"], second["mc_calibration_valid"]
        )

    def test_different_corruption_seed_changes_realization(self):
        configs = [
            ExperimentConfig(
                "rpex",
                "hopper-medium-replay-v2",
                corruption="random",
                corruption_target="observations",
                learner_seed=7,
                corruption_seed=seed,
            )
            for seed in (11, 12)
        ]
        outputs = []
        for config in configs:
            with tempfile.TemporaryDirectory() as directory:
                output, _ = corrupt_offline_dataset(
                    self.dataset, config, None, Path(directory)
                )
                outputs.append(output)
        self.assertFalse(
            np.array_equal(
                outputs[0]["mc_calibration_valid"],
                outputs[1]["mc_calibration_valid"],
            )
        )

    def test_random_mixed_allocates_each_row_to_one_target(self):
        config = ExperimentConfig(
            "rpex",
            "hopper-medium-replay-v2",
            corruption="random",
            corruption_target="mixed",
            mixed_ratios=(0.1, 0.2, 0.3, 0.4),
            offline_corruption_rate=1.0,
            seed=11,
        )
        with tempfile.TemporaryDirectory() as directory:
            first, first_stats = corrupt_offline_dataset(
                self.dataset, config, None, Path(directory)
            )
            second, second_stats = corrupt_offline_dataset(
                self.dataset, config, None, Path(directory)
            )

        changed = {
            "observations": int(
                np.any(first["observations"] != self.dataset["observations"], axis=1).sum()
            ),
            "actions": int(
                np.any(first["actions"] != self.dataset["actions"], axis=1).sum()
            ),
            "rewards": int(
                np.count_nonzero(first["rewards"] != self.dataset["rewards"])
            ),
            "dynamics": int(
                np.any(
                    first["next_observations"]
                    != self.dataset["next_observations"],
                    axis=1,
                ).sum()
            ),
        }
        self.assertEqual(sum(changed.values()), len(self.dataset["rewards"]))
        for target, count in changed.items():
            self.assertEqual(first_stats[f"{target}_corrupted_count"], count)
        for key in self.dataset:
            np.testing.assert_array_equal(first[key], second[key])
        self.assertEqual(second_stats["loaded_from_cache"], 1.0)

    def test_mixed_ratios_must_sum_to_one(self):
        with self.assertRaisesRegex(ValueError, "sum to 1.0"):
            ExperimentConfig(
                "rpex",
                "hopper-medium-replay-v2",
                corruption="random",
                corruption_target="mixed",
                mixed_ratios=(0.1, 0.2, 0.3, 0.3),
            )


if __name__ == "__main__":
    unittest.main()
