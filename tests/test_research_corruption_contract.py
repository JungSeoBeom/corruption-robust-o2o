from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from robust_o2o.config import ExperimentConfig
from robust_o2o.corruption import (
    AttackOracle,
    EDACActor,
    EDACCritic,
    SUPPORTED_ADVERSARIAL_TARGETS,
    corrupt_offline_dataset,
    corrupt_online_transition,
    validate_adversarial_target,
)
from robust_o2o.dataset import CORRUPTION_LABEL_KEYS
from robust_o2o.experiment import _poison_replay_in_learner_coordinates
from robust_o2o.replay import OfflineDataset


def synthetic_dataset(size: int = 256):
    rng = np.random.default_rng(13)
    return {
        "observations": rng.normal(size=(size, 4)).astype(np.float32),
        "actions": rng.normal(size=(size, 2)).astype(np.float32),
        "rewards": rng.normal(size=size).astype(np.float32),
        "next_observations": rng.normal(size=(size, 4)).astype(np.float32),
        "terminals": np.zeros(size, dtype=np.float32),
        "mc_returns": np.zeros(size, dtype=np.float32),
        "episode_id": np.repeat(np.arange((size + 7) // 8), 8)[:size].astype(
            np.float32
        ),
        "mc_calibration_valid": np.ones(size, dtype=np.float32),
    }


class AdditiveOracle:
    def __init__(self, checkpoint: Path):
        self.checkpoint = checkpoint
        self.checkpoint_sha256 = "synthetic-oracle"
        self.device = torch.device("cpu")
        self.env_name = "synthetic"
        self.strict_checkpoint_load_verified = True

    def attack(
        self,
        original,
        std,
        observations,
        actions,
        target,
        scale,
        steps,
        step_size,
        *,
        online=False,
    ):
        del std, observations, actions, target, scale, steps, step_size, online
        return np.asarray(original, dtype=np.float32) + np.float32(0.25)


class RaisingOracle(AdditiveOracle):
    def attack(self, *args, **kwargs):
        del args, kwargs
        raise RuntimeError("synthetic adversarial optimizer failure")


class ResearchCorruptionContractTest(unittest.TestCase):
    def test_research_poisoning_uses_learner_replay_coordinates(self):
        research = ExperimentConfig(
            "rpex",
            "hopper-medium-replay-v2",
            run_purpose="research_benchmark",
            suite_profile="research_benchmark",
            implementation_profile="research_benchmark",
        )
        legacy_diagnostic = ExperimentConfig(
            "rpex", "hopper-medium-replay-v2"
        )
        self.assertTrue(_poison_replay_in_learner_coordinates(research))
        self.assertFalse(
            _poison_replay_in_learner_coordinates(legacy_diagnostic)
        )

    def test_supported_adversarial_targets_are_explicit(self):
        self.assertEqual(
            tuple(SUPPORTED_ADVERSARIAL_TARGETS),
            ("observations", "actions", "rewards", "dynamics"),
        )
        unsupported = SimpleNamespace(
            corruption="adversarial",
            corruption_target="unverified_target",
            mixed_ratios=(0.25, 0.25, 0.25, 0.25),
        )
        with self.assertRaisesRegex(ValueError, "unsupported adversarial"):
            validate_adversarial_target(unsupported)

    def test_random_online_poisoning_changes_only_selected_replay_field(self):
        clean_state = np.asarray([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        clean_action = np.asarray([0.2, -0.4], dtype=np.float32)
        clean_reward = 1.5
        clean_next_state = np.asarray([5.0, 6.0, 7.0, 8.0], dtype=np.float32)
        originals = {
            "observations": clean_state.copy(),
            "actions": clean_action.copy(),
            "rewards": clean_reward,
            "dynamics": clean_next_state.copy(),
        }
        fields = {
            "observations": 0,
            "actions": 1,
            "rewards": 2,
            "dynamics": 3,
        }
        for target, target_index in fields.items():
            with self.subTest(target=target):
                config = ExperimentConfig(
                    "rpex",
                    "hopper-medium-replay-v2",
                    corruption="random",
                    corruption_target=target,
                    online_corruption_rate=1.0,
                )
                result = corrupt_online_transition(
                    clean_state,
                    clean_action,
                    clean_reward,
                    clean_next_state,
                    config,
                    None,
                    np.random.default_rng(101),
                    np.ones(4, dtype=np.float32),
                    np.ones(2, dtype=np.float32),
                    selected_target=target,
                    selection_already_sampled=True,
                )
                self.assertTrue(result[-1])
                clean_fields = (
                    clean_state,
                    clean_action,
                    clean_reward,
                    clean_next_state,
                )
                for index, (actual, expected) in enumerate(
                    zip(result[:4], clean_fields)
                ):
                    if index == target_index:
                        self.assertFalse(np.array_equal(actual, expected))
                    else:
                        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(clean_state, originals["observations"])
        np.testing.assert_array_equal(clean_action, originals["actions"])
        self.assertEqual(clean_reward, originals["rewards"])
        np.testing.assert_array_equal(clean_next_state, originals["dynamics"])

    def test_adversarial_offline_and_online_modify_only_target_field(self):
        dataset = synthetic_dataset(32)
        clean = {key: value.copy() for key, value in dataset.items()}
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "oracle.pt"
            checkpoint.write_bytes(b"synthetic")
            oracle = AdditiveOracle(checkpoint)
            config = ExperimentConfig(
                "rpex",
                "hopper-medium-replay-v2",
                corruption="adversarial",
                corruption_target="actions",
                offline_corruption_rate=1.0,
                online_corruption_rate=1.0,
            )
            corrupted, stats = corrupt_offline_dataset(
                dataset, config, oracle, Path(directory) / "cache"
            )
            self.assertEqual(stats["selected_transition_count"], 32)
            self.assertFalse(np.array_equal(corrupted["actions"], clean["actions"]))
            for key in (
                "observations",
                "rewards",
                "next_observations",
                "terminals",
            ):
                np.testing.assert_array_equal(corrupted[key], clean[key])
            for key in clean:
                np.testing.assert_array_equal(dataset[key], clean[key])

            online = corrupt_online_transition(
                clean["observations"][0],
                clean["actions"][0],
                float(clean["rewards"][0]),
                clean["next_observations"][0],
                config,
                oracle,
                np.random.default_rng(0),
                np.ones(4, dtype=np.float32),
                np.ones(2, dtype=np.float32),
                selected_target="actions",
                selection_already_sampled=True,
            )
            np.testing.assert_array_equal(online[0], clean["observations"][0])
            self.assertFalse(np.array_equal(online[1], clean["actions"][0]))
            self.assertEqual(online[2], float(clean["rewards"][0]))
            np.testing.assert_array_equal(
                online[3], clean["next_observations"][0]
            )

    def test_adversarial_optimizer_failure_never_falls_back_to_random(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "oracle.pt"
            checkpoint.write_bytes(b"synthetic")
            config = ExperimentConfig(
                "rpex",
                "hopper-medium-replay-v2",
                corruption="adversarial",
                corruption_target="observations",
                online_corruption_rate=1.0,
            )
            with self.assertRaisesRegex(RuntimeError, "optimizer failure"):
                corrupt_online_transition(
                    np.zeros(4, dtype=np.float32),
                    np.zeros(2, dtype=np.float32),
                    0.0,
                    np.ones(4, dtype=np.float32),
                    config,
                    RaisingOracle(checkpoint),
                    np.random.default_rng(0),
                    np.ones(4, dtype=np.float32),
                    np.ones(2, dtype=np.float32),
                    selected_target="observations",
                    selection_already_sampled=True,
                )

    def test_main_baselines_reuse_the_same_random_offline_artifact(self):
        dataset = synthetic_dataset()
        common = {
            "env_name": "hopper-medium-replay-v2",
            "run_purpose": "research_benchmark",
            "suite_profile": "research_benchmark",
            "corruption": "random",
            "corruption_target": "observations",
            "corruption_seed": 91,
        }
        rpex = ExperimentConfig(algorithm="rpex", **common)
        wsrl = ExperimentConfig(algorithm="wsrl", **common)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first, first_stats = corrupt_offline_dataset(
                dataset, rpex, None, root
            )
            second, second_stats = corrupt_offline_dataset(
                dataset, wsrl, None, root
            )
        self.assertEqual(first_stats["cache_key"], second_stats["cache_key"])
        self.assertEqual(first_stats["cache_file"], second_stats["cache_file"])
        self.assertFalse(first_stats["cache_hit"])
        self.assertTrue(second_stats["cache_hit"])
        for key in first:
            np.testing.assert_array_equal(first[key], second[key])

    def test_corruption_diagnostics_never_enter_learner_batch(self):
        offline = OfflineDataset(synthetic_dataset(32), seed=5)
        batch = offline.sample(8, torch.device("cpu"))
        self.assertFalse(CORRUPTION_LABEL_KEYS.intersection(batch))
        self.assertNotIn("episode_id", batch)
        self.assertIn("mc_returns", batch)

    def test_oracle_checkpoint_load_is_strict(self):
        actor = EDACActor(4, 2, 1.0).state_dict()
        critic = EDACCritic(4, 2).state_dict()
        actor.pop("trunk.2.bias")
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "missing_key.pt"
            torch.save({"actor": actor, "critic": critic}, checkpoint)
            with self.assertRaisesRegex(RuntimeError, "Missing key"):
                AttackOracle(
                    4,
                    2,
                    1.0,
                    checkpoint,
                    torch.device("cpu"),
                )


if __name__ == "__main__":
    unittest.main()
