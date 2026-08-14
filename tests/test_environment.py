from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from robust_o2o.config import DEFAULT_PROTOCOL, LOCAL_PROTOCOL, ExperimentConfig
from robust_o2o.environment import (
    EXPECTED_D4RL_COMMIT,
    RPEXProtocolError,
    _index_aware_qlearning_dataset,
    environment_metadata,
    local_dataset_path,
    load_d4rl_dataset,
    make_env,
    normalized_d4rl_scores,
    qlearning_valid_indices,
    raw_monte_carlo_returns,
    reset_env,
    step_env,
    validate_dataset,
)


class FakeSpace:
    def __init__(self, shape: tuple[int, ...]):
        self.shape = shape
        self.high = np.ones(shape, dtype=np.float32)
        self.seed_calls: list[int] = []

    def seed(self, seed: int) -> None:
        self.seed_calls.append(seed)

    def sample(self) -> np.ndarray:
        return np.zeros(self.shape, dtype=np.float32)


class FakeEnv:
    def __init__(
        self,
        env_id: str = "walker2d-medium-replay-v2",
        raw: dict[str, np.ndarray] | None = None,
    ):
        self.spec = SimpleNamespace(id=env_id, max_episode_steps=1_000)
        self._max_episode_steps = 1_000
        self.observation_space = FakeSpace((3,))
        self.action_space = FakeSpace((2,))
        self.unwrapped = self
        self.dataset_url = "https://example.invalid/dataset.hdf5"
        self.dataset_filepath = None
        self.raw = raw
        self.close_calls = 0
        self.seed_calls: list[int] = []

    def seed(self, seed: int) -> None:
        self.seed_calls.append(seed)

    def reset(self) -> np.ndarray:
        return np.zeros(3, dtype=np.float32)

    def step(self, _action: np.ndarray):
        return np.ones(3, dtype=np.float32), 1.0, False, {}

    def get_dataset(self) -> dict[str, np.ndarray]:
        if self.raw is None:
            raise RuntimeError("No synthetic dataset configured")
        return self.raw

    def close(self) -> None:
        self.close_calls += 1


def small_raw_dataset() -> dict[str, np.ndarray]:
    observations = np.arange(18, dtype=np.float32).reshape(6, 3)
    return {
        "observations": observations,
        "actions": np.arange(12, dtype=np.float32).reshape(6, 2),
        "rewards": np.arange(6, dtype=np.float32),
        "terminals": np.asarray([0, 0, 1, 0, 0, 0], dtype=np.float32),
        "timeouts": np.asarray([0, 1, 0, 0, 0, 1], dtype=np.float32),
    }


class StrictRPEXEnvironmentTest(unittest.TestCase):
    def test_config_preserves_complete_d4rl_id(self):
        config = ExperimentConfig("rpex", "half-cheetah-medium-replay-v2")
        self.assertEqual(config.env_name, "halfcheetah-medium-replay-v2")
        self.assertEqual(config.protocol, DEFAULT_PROTOCOL)

    def test_full_d4rl_id_is_passed_unchanged_to_gym_make(self):
        env = FakeEnv()
        gym = SimpleNamespace(make=Mock(return_value=env))
        with patch(
            "robust_o2o.environment._import_legacy_backend",
            return_value=(gym, object()),
        ):
            result = make_env("walker2d-medium-replay-v2")
        self.assertIs(result, env)
        gym.make.assert_called_once_with("walker2d-medium-replay-v2")
        requested = gym.make.call_args.args[0]
        self.assertNotIn(requested, ("Walker2d-v2", "Walker2d-v4", "Walker2d-v5"))

    def test_strict_mode_rejects_gymnasium_ids(self):
        for name in ("Walker2d-v4", "Hopper-v5", "HalfCheetah-v4"):
            with self.subTest(name=name), self.assertRaises(RPEXProtocolError):
                make_env(name)

    def test_qlearning_dataset_is_official_and_terminate_on_end_false(self):
        raw = small_raw_dataset()
        env = FakeEnv(raw=raw)
        d4rl = SimpleNamespace()
        d4rl.set_dataset_path = Mock()

        def qlearning_dataset(received_env, *, dataset, terminate_on_end):
            self.assertIs(received_env, env)
            self.assertIs(dataset, raw)
            self.assertFalse(terminate_on_end)
            indices = qlearning_valid_indices(raw, env._max_episode_steps)
            return _index_aware_qlearning_dataset(raw, indices)

        d4rl.qlearning_dataset = Mock(side_effect=qlearning_dataset)
        with patch(
            "robust_o2o.environment._import_legacy_backend",
            return_value=(object(), d4rl),
        ):
            dataset = load_d4rl_dataset(env, "/tmp/d4rl-test", discount=1.0)

        d4rl.set_dataset_path.assert_called_once_with(
            str(Path("/tmp/d4rl-test").resolve())
        )
        self.assertEqual(d4rl.qlearning_dataset.call_count, 1)
        np.testing.assert_array_equal(dataset["rewards"], [0, 2, 3, 4])
        self.assertEqual(len(dataset["mc_returns"]), len(dataset["rewards"]))

    def test_official_normalization_is_called_for_reference_returns(self):
        random_return = -20.272305
        expert_return = 3234.3
        d4rl = SimpleNamespace()

        def get_normalized_score(env_name, values):
            self.assertEqual(env_name, "hopper-medium-replay-v2")
            return (values - random_return) / (expert_return - random_return)

        d4rl.get_normalized_score = Mock(side_effect=get_normalized_score)
        with patch(
            "robust_o2o.environment._import_legacy_backend",
            return_value=(object(), d4rl),
        ):
            scores = normalized_d4rl_scores(
                "hopper-medium-replay-v2",
                np.asarray([random_return, expert_return]),
            )
        np.testing.assert_allclose(scores, [0.0, 100.0])
        d4rl.get_normalized_score.assert_called_once()

    def test_local_hopper_dataset_path_and_normalization(self):
        path = local_dataset_path(
            "hopper-medium-replay-v2", "/tmp/local-d4rl-datasets"
        )
        self.assertEqual(path.name, "hopper_medium_replay-v2.hdf5")
        scores = normalized_d4rl_scores(
            "hopper-medium-replay-v2",
            np.asarray([-20.272305, 3234.3]),
            LOCAL_PROTOCOL,
        )
        np.testing.assert_allclose(scores, [0.0, 100.0])

    def test_local_gymnasium_reset_and_step_results_are_accepted(self):
        env = FakeEnv()
        env.reset = Mock(return_value=(np.zeros(3), {"seeded": True}))
        env.step = Mock(return_value=(np.ones(3), 1.5, True, False, {}))
        observation = reset_env(env, seed=7, protocol=LOCAL_PROTOCOL)
        transition = step_env(env, np.zeros(2), protocol=LOCAL_PROTOCOL)
        np.testing.assert_array_equal(observation, np.zeros(3))
        env.reset.assert_called_once_with(seed=7)
        self.assertEqual(transition[1:4], (1.5, True, False))

    def test_dataset_shape_and_finiteness_validation(self):
        env = FakeEnv()
        valid = {
            "observations": np.zeros((4, 3), dtype=np.float32),
            "actions": np.zeros((4, 2), dtype=np.float32),
            "next_observations": np.ones((4, 3), dtype=np.float32),
            "rewards": np.zeros(4, dtype=np.float32),
            "terminals": np.zeros(4, dtype=np.float32),
            "mc_returns": np.zeros(4, dtype=np.float32),
        }
        self.assertEqual(len(validate_dataset(valid, env)["rewards"]), 4)
        malformed = {key: value.copy() for key, value in valid.items()}
        malformed["actions"] = np.zeros((4, 3), dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "action dimension"):
            validate_dataset(malformed, env)
        nonfinite = {key: value.copy() for key, value in valid.items()}
        nonfinite["observations"][0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "non-finite"):
            validate_dataset(nonfinite, env)

    def test_legacy_timeout_and_terminal_masks(self):
        timeout_env = FakeEnv()
        timeout_env.step = Mock(
            return_value=(
                np.zeros(3, dtype=np.float32),
                1.0,
                True,
                {"TimeLimit.truncated": True},
            )
        )
        _, _, terminated, truncated, _ = step_env(
            timeout_env, np.zeros(2, dtype=np.float32)
        )
        self.assertFalse(terminated)
        self.assertTrue(truncated)

        terminal_env = FakeEnv()
        terminal_env.step = Mock(
            return_value=(np.zeros(3, dtype=np.float32), 1.0, True, {})
        )
        _, _, terminated, truncated, _ = step_env(
            terminal_env, np.zeros(2, dtype=np.float32)
        )
        self.assertTrue(terminated)
        self.assertFalse(truncated)

    def test_gymnasium_reset_and_step_results_are_rejected(self):
        env = FakeEnv()
        env.reset = Mock(return_value=(np.zeros(3), {}))
        with self.assertRaisesRegex(RPEXProtocolError, "reset"):
            reset_env(env)
        env.step = Mock(return_value=(np.zeros(3), 0.0, False, False, {}))
        with self.assertRaisesRegex(RPEXProtocolError, "five values"):
            step_env(env, np.zeros(2))

    def test_two_consecutive_thousand_step_timeouts_do_not_leak(self):
        size = 2_000
        raw = {
            "observations": np.zeros((size, 3), dtype=np.float32),
            "actions": np.zeros((size, 2), dtype=np.float32),
            "rewards": np.ones(size, dtype=np.float32),
            "terminals": np.zeros(size, dtype=np.float32),
            "timeouts": np.zeros(size, dtype=np.float32),
        }
        raw["timeouts"][[999, 1999]] = 1.0
        returns = raw_monte_carlo_returns(raw, 1.0, 1_000)
        indices = qlearning_valid_indices(raw, 1_000)
        mc_returns = returns[indices]
        self.assertEqual(len(indices), 1_998)
        self.assertEqual(len(mc_returns), len(indices))
        self.assertEqual(returns[0], 1_000.0)
        self.assertEqual(returns[998], 2.0)
        self.assertEqual(returns[1_000], 1_000.0)
        self.assertEqual(mc_returns[999], 1_000.0)

    def test_missing_timeouts_uses_d4rl_max_episode_fallback(self):
        raw = {
            "observations": np.zeros((7, 3), dtype=np.float32),
            "actions": np.zeros((7, 2), dtype=np.float32),
            "rewards": np.ones(7, dtype=np.float32),
            "terminals": np.zeros(7, dtype=np.float32),
        }
        indices = qlearning_valid_indices(raw, max_episode_steps=3)
        np.testing.assert_array_equal(indices, [0, 1, 3, 4])
        returns = raw_monte_carlo_returns(raw, 1.0, max_episode_steps=3)
        np.testing.assert_array_equal(returns, [3, 2, 1, 3, 2, 1, 1])
        np.testing.assert_array_equal(returns[indices], [3, 2, 3, 2])

    def test_natural_and_timeout_episodes_for_two_discounts(self):
        raw = {
            "observations": np.zeros((5, 3), dtype=np.float32),
            "actions": np.zeros((5, 2), dtype=np.float32),
            "rewards": np.asarray([1, 2, 3, 4, 999], dtype=np.float32),
            "terminals": np.asarray([0, 1, 0, 0, 0], dtype=np.float32),
            "timeouts": np.asarray([0, 0, 0, 1, 0], dtype=np.float32),
        }
        indices = qlearning_valid_indices(raw, 1_000)
        np.testing.assert_array_equal(indices, [0, 1, 2])
        gamma_one = raw_monte_carlo_returns(raw, 1.0, 1_000)
        np.testing.assert_allclose(gamma_one[indices], [3.0, 2.0, 7.0])
        gamma_099 = raw_monte_carlo_returns(raw, 0.99, 1_000)
        np.testing.assert_allclose(
            gamma_099[indices], [2.98, 2.0, 6.96], rtol=1e-6
        )
        self.assertLess(gamma_099[0], 3.0)
        self.assertLess(gamma_099[2], 7.0)

    def test_runtime_metadata_identifies_protocol_and_full_environment(self):
        env = FakeEnv()
        dataset = {
            "observations": np.zeros((4, 3), dtype=np.float32),
            "actions": np.zeros((4, 2), dtype=np.float32),
            "next_observations": np.zeros((4, 3), dtype=np.float32),
            "rewards": np.zeros(4, dtype=np.float32),
            "terminals": np.zeros(4, dtype=np.float32),
        }
        with patch(
            "robust_o2o.environment.installed_d4rl_commit",
            return_value=EXPECTED_D4RL_COMMIT,
        ):
            metadata = environment_metadata(
                env, "walker2d-medium-replay-v2", dataset, seed=42
            )
        self.assertEqual(metadata["protocol"], DEFAULT_PROTOCOL)
        self.assertEqual(metadata["d4rl_env_id"], "walker2d-medium-replay-v2")
        self.assertEqual(metadata["env_spec_id"], "walker2d-medium-replay-v2")
        self.assertEqual(metadata["expected_d4rl_commit"], EXPECTED_D4RL_COMMIT)
        self.assertEqual(metadata["seed"], 42)
        for package in ("Python", "numpy", "torch", "gym", "d4rl", "mujoco-py", "h5py"):
            self.assertIn(package, metadata["runtime_package_versions"])


if __name__ == "__main__":
    unittest.main()
