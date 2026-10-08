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
from robust_o2o.experiment import (
    capture_global_rng_state,
    restore_global_rng_state,
)
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


class FixedPoisonedActionOracle(AdditiveOracle):
    def attack(self, original, *args, **kwargs):
        del args, kwargs
        return np.full_like(np.asarray(original, dtype=np.float32), -1.7)


class ResearchCorruptionContractTest(unittest.TestCase):
    def test_zero_rate_artifact_is_bitwise_clean_and_reports_no_change(self):
        dataset = synthetic_dataset(32)
        config = ExperimentConfig(
            "rpex",
            "hopper-medium-replay-v2",
            corruption="random",
            corruption_target="observations",
            offline_corruption_rate=0.0,
        )
        with tempfile.TemporaryDirectory() as directory:
            corrupted, stats = corrupt_offline_dataset(
                dataset, config, None, Path(directory)
            )
        self.assertEqual(dataset.keys(), corrupted.keys())
        for key in dataset:
            self.assertEqual(dataset[key].tobytes(), corrupted[key].tobytes())
        self.assertEqual(stats["selected_transition_count"], 0)
        self.assertEqual(stats["actual_changed_transition_count"], 0)

    def test_selected_and_actual_change_rates_are_distinct(self):
        dataset = synthetic_dataset(32)
        with tempfile.TemporaryDirectory() as directory:
            vector_config = ExperimentConfig(
                "rpex",
                "hopper-medium-replay-v2",
                corruption="random",
                corruption_target="observations",
                offline_corruption_rate=1.0,
                corruption_range=0.0,
            )
            _, vector_stats = corrupt_offline_dataset(
                dataset, vector_config, None, Path(directory) / "vector"
            )
            reward_config = ExperimentConfig(
                "rpex",
                "hopper-medium-replay-v2",
                corruption="random",
                corruption_target="rewards",
                offline_corruption_rate=1.0,
                corruption_range=0.0,
            )
            reward_result, reward_stats = corrupt_offline_dataset(
                dataset, reward_config, None, Path(directory) / "reward"
            )
        self.assertEqual(vector_stats["selected_transition_count"], 32)
        self.assertEqual(vector_stats["actual_changed_transition_count"], 0)
        self.assertEqual(reward_stats["selected_transition_count"], 32)
        self.assertGreater(reward_stats["actual_changed_transition_count"], 0)
        self.assertFalse(
            np.array_equal(reward_result["rewards"], dataset["rewards"])
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
                    normalizer_mean=np.zeros(4, dtype=np.float32),
                    normalizer_std=np.ones(4, dtype=np.float32),
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
                normalizer_mean=np.zeros(4, dtype=np.float32),
                normalizer_std=np.ones(4, dtype=np.float32),
            )
            np.testing.assert_array_equal(online[0], clean["observations"][0])
            self.assertFalse(np.array_equal(online[1], clean["actions"][0]))
            self.assertEqual(online[2], float(clean["rewards"][0]))
            np.testing.assert_array_equal(
                online[3], clean["next_observations"][0]
            )

    def test_poisoned_action_label_is_not_clipped_before_replay(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "oracle.pt"
            checkpoint.write_bytes(b"synthetic")
            config = ExperimentConfig(
                "rpex",
                "hopper-medium-replay-v2",
                corruption="adversarial",
                corruption_target="actions",
                online_corruption_rate=1.0,
            )
            _, poisoned_action, _, _, selected = corrupt_online_transition(
                np.zeros(4, dtype=np.float32),
                np.asarray([0.3, -0.2], dtype=np.float32),
                1.0,
                np.ones(4, dtype=np.float32),
                config,
                FixedPoisonedActionOracle(checkpoint),
                np.random.default_rng(0),
                np.ones(4, dtype=np.float32),
                np.ones(2, dtype=np.float32),
                selected_target="actions",
                selection_already_sampled=True,
                normalizer_mean=np.zeros(4, dtype=np.float32),
                normalizer_std=np.ones(4, dtype=np.float32),
            )
        self.assertTrue(selected)
        np.testing.assert_array_equal(
            poisoned_action, np.asarray([-1.7, -1.7], dtype=np.float32)
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
                    normalizer_mean=np.zeros(4, dtype=np.float32),
                    normalizer_std=np.ones(4, dtype=np.float32),
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

    def test_oracle_objectives_use_checkpoint_declared_state_coordinates(self):
        state_mean = np.asarray([10.0, -4.0, 3.0, 8.0], dtype=np.float32)
        state_std = np.asarray([2.0, 4.0, 0.5, 8.0], dtype=np.float32)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "preprocessed_oracle.pt"
            torch.save(
                {
                    "actor": EDACActor(4, 2, 1.0).state_dict(),
                    "critic": EDACCritic(4, 2).state_dict(),
                    "metadata": {
                        "state_mean": torch.as_tensor(state_mean),
                        "state_std": torch.as_tensor(state_std),
                    },
                },
                checkpoint,
            )
            oracle = AttackOracle(
                4,
                2,
                1.0,
                checkpoint,
                torch.device("cpu"),
                seed=17,
                implementation_profile="rpex_official_adam",
                benchmark_profile="research_benchmark",
                record_trace=True,
            )
            self.assertTrue(oracle.preprocessing_verified)
            self.assertIn("state_mean_state_std", oracle.preprocessing_source)

            observations = np.asarray([[12.0, 4.0, 4.0, 0.0]], dtype=np.float32)
            actions = np.asarray([[0.2, -0.3]], dtype=np.float32)
            state_scale = np.asarray([[1.5, 2.0, 0.25, 4.0]], dtype=np.float32)
            action_scale = np.asarray([[0.5, 2.0]], dtype=np.float32)
            critic_states: list[torch.Tensor] = []
            actor_states: list[torch.Tensor] = []
            critic_hook = oracle.critic.register_forward_pre_hook(
                lambda _module, args: critic_states.append(args[0].detach().clone())
            )
            actor_hook = oracle.actor.register_forward_pre_hook(
                lambda _module, args: actor_states.append(args[0].detach().clone())
            )
            try:
                for target in ("observations", "actions", "dynamics"):
                    critic_states.clear()
                    actor_states.clear()
                    oracle.attack_traces.clear()
                    original = actions if target == "actions" else observations
                    scale = action_scale if target == "actions" else state_scale
                    oracle.attack(
                        original,
                        scale,
                        observations,
                        actions,
                        target,
                        0.2,
                        1,
                        0.01,
                        online=False,
                    )
                    if target == "actions":
                        expected_state = (observations - state_mean) / state_std
                    else:
                        initial_raw = (
                            original
                            + oracle.attack_traces[0][
                                "initial_effective_perturbation"
                            ]
                        )
                        expected_state = (initial_raw - state_mean) / state_std
                    torch.testing.assert_close(
                        critic_states[0], torch.as_tensor(expected_state)
                    )
                    if target == "dynamics":
                        torch.testing.assert_close(
                            actor_states[0], torch.as_tensor(expected_state)
                        )
            finally:
                critic_hook.remove()
                actor_hook.remove()
                oracle.close()

    def test_research_adversary_uses_resumable_private_stream_and_single_std(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "oracle.pt"
            torch.save(
                {
                    "actor": EDACActor(3, 2, 1.0).state_dict(),
                    "critic": EDACCritic(3, 2).state_dict(),
                },
                checkpoint,
            )

            def make_oracle():
                return AttackOracle(
                    3,
                    2,
                    1.0,
                    checkpoint,
                    torch.device("cpu"),
                    seed=23,
                    implementation_profile="rpex_official_adam",
                    benchmark_profile="research_benchmark",
                )

            original = np.zeros((1, 3), dtype=np.float32)
            std = np.asarray([[2.0, 3.0, 4.0]], dtype=np.float32)
            actions = np.zeros((1, 2), dtype=np.float32)
            first = make_oracle()
            second = make_oracle()
            try:
                first_sequence = [
                    first.attack(
                        original,
                        std,
                        original,
                        actions,
                        "observations",
                        0.25,
                        0,
                        0.1,
                        online=True,
                    )
                    for _ in range(2)
                ]
                second_sequence = [
                    second.attack(
                        original,
                        std,
                        original,
                        actions,
                        "observations",
                        0.25,
                        0,
                        0.1,
                        online=True,
                    )
                    for _ in range(2)
                ]
                for left, right in zip(first_sequence, second_sequence):
                    np.testing.assert_array_equal(left, right)
                    self.assertTrue(np.all(np.abs(left) <= 0.25 * std + 1e-7))
                self.assertFalse(np.array_equal(*first_sequence))

                corruption_rng = np.random.default_rng(41)
                restored_corruption_rng = np.random.default_rng(999)
                resume_state = capture_global_rng_state(corruption_rng, first)
                expected_after_resume = first.attack(
                    original,
                    std,
                    original,
                    actions,
                    "observations",
                    0.25,
                    0,
                    0.1,
                    online=True,
                )
                # Perturb both private streams before restoring the production
                # exact-resume state.
                second.attack(
                    original,
                    std,
                    original,
                    actions,
                    "observations",
                    0.25,
                    0,
                    0.1,
                    online=True,
                )
                restored_corruption_rng.random(3)
                restore_global_rng_state(
                    resume_state, restored_corruption_rng, second
                )
                actual_after_resume = second.attack(
                    original,
                    std,
                    original,
                    actions,
                    "observations",
                    0.25,
                    0,
                    0.1,
                    online=True,
                )
                np.testing.assert_array_equal(
                    expected_after_resume, actual_after_resume
                )
                self.assertEqual(
                    corruption_rng.random(), restored_corruption_rng.random()
                )
                torch.manual_seed(71)
                expected_global = torch.rand(5)
                torch.manual_seed(71)
                first.attack(
                    original,
                    std,
                    original,
                    actions,
                    "dynamics",
                    0.25,
                    2,
                    0.1,
                    online=True,
                )
                self.assertTrue(torch.equal(expected_global, torch.rand(5)))
                self.assertFalse(first.preprocessing_verified)
                self.assertEqual(
                    first.preprocessing_source,
                    "checkpoint_metadata_missing_identity_unverified",
                )
            finally:
                first.close()
                second.close()


if __name__ == "__main__":
    unittest.main()
