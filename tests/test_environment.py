from __future__ import annotations

import io
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import h5py
import numpy as np

from robust_o2o.config import DEFAULT_PROTOCOL, ExperimentConfig
from robust_o2o.environment import (
    MODERN_PROTOCOL,
    EnvironmentSetupError,
    _index_aware_qlearning_dataset,
    d4rl_dataset_url,
    download_d4rl_dataset,
    environment_metadata,
    expected_env_spec_id,
    local_dataset_path,
    load_d4rl_dataset,
    make_env,
    normalized_d4rl_scores,
    qlearning_valid_indices,
    raw_monte_carlo_returns,
    reset_env,
    runtime_package_versions,
    step_env,
    validate_dataset,
)


class FakeSpace:
    def __init__(self, shape: tuple[int, ...]):
        self.shape = shape
        self.low = -np.ones(shape, dtype=np.float32)
        self.high = np.ones(shape, dtype=np.float32)


class FakeEnv:
    def __init__(
        self,
        env_id: str = "Walker2d-v4",
        observation_dim: int = 3,
        action_dim: int = 2,
    ):
        self.spec = SimpleNamespace(id=env_id, max_episode_steps=1_000)
        self._max_episode_steps = 1_000
        self.observation_space = FakeSpace((observation_dim,))
        self.action_space = FakeSpace((action_dim,))
        self.unwrapped = self
        self.close_calls = 0

    def reset(self, *, seed: int | None = None):
        return np.zeros(self.observation_space.shape, dtype=np.float32), {
            "seed": seed
        }

    def step(self, _action: np.ndarray):
        return (
            np.ones(self.observation_space.shape, dtype=np.float32),
            1.0,
            False,
            False,
            {},
        )

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


def write_hdf5(path: Path, raw: dict[str, np.ndarray]) -> None:
    with h5py.File(path, "w") as stream:
        for key, value in raw.items():
            stream.create_dataset(key, data=value)


class ModernEnvironmentTest(unittest.TestCase):
    def test_config_and_environment_use_one_modern_protocol(self):
        config = ExperimentConfig("rpex", "half-cheetah-medium-replay-v2")
        self.assertEqual(config.env_name, "halfcheetah-medium-replay-v2")
        self.assertEqual(config.protocol, MODERN_PROTOCOL)
        self.assertEqual(DEFAULT_PROTOCOL, MODERN_PROTOCOL)

    def test_d4rl_ids_map_to_v4_without_changing_dataset_id(self):
        expected = {
            "halfcheetah-medium-v2": "HalfCheetah-v4",
            "hopper-medium-replay-v2": "Hopper-v4",
            "walker2d-medium-expert-v2": "Walker2d-v4",
        }
        for dataset_id, gymnasium_id in expected.items():
            with self.subTest(dataset_id=dataset_id):
                self.assertEqual(expected_env_spec_id(dataset_id), gymnasium_id)

    def test_make_env_calls_gymnasium_with_v4_id(self):
        env = FakeEnv("Walker2d-v4")
        gymnasium = SimpleNamespace(make=Mock(return_value=env))
        with patch.dict(sys.modules, {"gymnasium": gymnasium}):
            result = make_env("walker2d-medium-replay-v2")
        self.assertIs(result, env)
        gymnasium.make.assert_called_once_with("Walker2d-v4")

    def test_legacy_protocol_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "only supported protocol"):
            expected_env_spec_id(
                "hopper-medium-replay-v2", "rpex_d4rl_v2_legacy"
            )

    def test_qlearning_conversion_matches_terminate_on_end_false(self):
        raw = small_raw_dataset()
        indices = qlearning_valid_indices(raw, max_episode_steps=1_000)
        np.testing.assert_array_equal(indices, [0, 2, 3, 4])
        dataset = _index_aware_qlearning_dataset(raw, indices)
        np.testing.assert_array_equal(dataset["rewards"], [0, 2, 3, 4])
        np.testing.assert_array_equal(
            dataset["next_observations"], raw["observations"][[1, 3, 4, 5]]
        )

    def test_direct_hdf5_loader_keeps_qlearning_and_calql_semantics(self):
        raw = small_raw_dataset()
        env = FakeEnv()
        with tempfile.TemporaryDirectory() as directory:
            path = local_dataset_path(
                "walker2d-medium-replay-v2", directory
            )
            write_hdf5(path, raw)
            dataset = load_d4rl_dataset(
                env,
                directory,
                discount=1.0,
                env_name="walker2d-medium-replay-v2",
            )
        np.testing.assert_array_equal(dataset["rewards"], [0, 2, 3, 4])
        np.testing.assert_array_equal(dataset["episode_id"], [0, 1, 2, 2])
        np.testing.assert_array_equal(dataset["mc_returns"], [0, 2, 7, 4])
        np.testing.assert_array_equal(
            dataset["mc_calibration_valid"], np.ones(4, dtype=np.float32)
        )

    def test_dataset_path_and_official_url(self):
        path = local_dataset_path(
            "hopper-medium-replay-v2", "/tmp/d4rl-datasets"
        )
        self.assertEqual(path.name, "hopper_medium_replay-v2.hdf5")
        self.assertEqual(
            d4rl_dataset_url("hopper-medium-replay-v2"),
            "https://rail.eecs.berkeley.edu/datasets/offline_rl/"
            "gym_mujoco_v2/hopper_medium_replay-v2.hdf5",
        )

    def test_dataset_download_is_validated_and_atomically_installed(self):
        with tempfile.TemporaryDirectory() as source_directory:
            source = Path(source_directory) / "source.hdf5"
            write_hdf5(source, small_raw_dataset())
            payload = source.read_bytes()
        with tempfile.TemporaryDirectory() as destination_directory:
            response = io.BytesIO(payload)
            with patch(
                "robust_o2o.environment.urllib.request.urlopen",
                return_value=response,
            ) as urlopen:
                result = download_d4rl_dataset(
                    "hopper-medium-replay-v2", destination_directory
                )
            self.assertTrue(result.is_file())
            self.assertEqual(result.read_bytes(), payload)
            self.assertEqual(list(result.parent.glob("*.part")), [])
            request = urlopen.call_args.args[0]
            self.assertEqual(
                request.full_url,
                d4rl_dataset_url("hopper-medium-replay-v2"),
            )

    def test_existing_dataset_is_not_downloaded_without_force(self):
        with tempfile.TemporaryDirectory() as directory:
            path = local_dataset_path("hopper-medium-v2", directory)
            write_hdf5(path, small_raw_dataset())
            with patch(
                "robust_o2o.environment.urllib.request.urlopen"
            ) as urlopen:
                result = download_d4rl_dataset("hopper-medium-v2", directory)
            self.assertEqual(result, path)
            urlopen.assert_not_called()

    def test_failed_download_leaves_no_partial_or_target_file(self):
        with tempfile.TemporaryDirectory() as directory:
            with (
                patch(
                    "robust_o2o.environment.urllib.request.urlopen",
                    return_value=io.BytesIO(b"not an hdf5 file"),
                ),
                self.assertRaises(OSError),
            ):
                download_d4rl_dataset("hopper-medium-v2", directory)
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_reference_score_scaling_is_unchanged(self):
        scores = normalized_d4rl_scores(
            "hopper-medium-replay-v2",
            np.asarray([-20.272305, 3234.3]),
        )
        np.testing.assert_allclose(scores, [0.0, 100.0])

    def test_gymnasium_reset_and_step_api(self):
        env = FakeEnv()
        observation = reset_env(env, seed=7)
        transition = step_env(env, np.zeros(2, dtype=np.float32))
        np.testing.assert_array_equal(observation, np.zeros(3))
        self.assertEqual(transition[1:4], (1.0, False, False))

    def test_legacy_step_api_is_rejected(self):
        env = FakeEnv()
        env.step = Mock(return_value=(np.zeros(3), 1.0, False, {}))
        with self.assertRaisesRegex(EnvironmentSetupError, "five values"):
            step_env(env, np.zeros(2, dtype=np.float32))

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
        self.assertEqual(len(indices), 1_998)
        self.assertEqual(returns[0], 1_000.0)
        self.assertEqual(returns[998], 2.0)
        self.assertEqual(returns[1_000], 1_000.0)

    def test_missing_timeouts_uses_d4rl_horizon_fallback(self):
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

    def test_terminal_keeps_official_d4rl_counter_order(self):
        raw = {
            "observations": np.zeros((7, 3), dtype=np.float32),
            "actions": np.zeros((7, 2), dtype=np.float32),
            "rewards": np.ones(7, dtype=np.float32),
            "terminals": np.asarray([1, 0, 0, 0, 0, 0, 0], dtype=np.float32),
        }
        indices = qlearning_valid_indices(raw, max_episode_steps=3)
        np.testing.assert_array_equal(indices, [0, 1, 3, 4])

    def test_metadata_contains_only_modern_runtime_identity(self):
        env = FakeEnv()
        dataset = {
            "observations": np.zeros((4, 3), dtype=np.float32),
            "actions": np.zeros((4, 2), dtype=np.float32),
            "next_observations": np.zeros((4, 3), dtype=np.float32),
            "rewards": np.zeros(4, dtype=np.float32),
            "terminals": np.zeros(4, dtype=np.float32),
        }
        with tempfile.TemporaryDirectory() as directory:
            path = local_dataset_path(
                "walker2d-medium-replay-v2", directory
            )
            write_hdf5(path, small_raw_dataset())
            metadata = environment_metadata(
                env,
                "walker2d-medium-replay-v2",
                dataset,
                seed=42,
                dataset_dir=directory,
            )
        self.assertEqual(metadata["protocol"], MODERN_PROTOCOL)
        self.assertEqual(
            metadata["environment_id"], "walker2d-medium-replay-v2"
        )
        self.assertEqual(metadata["env_spec_id"], "Walker2d-v4")
        self.assertEqual(
            metadata["environment_backend"], "gymnasium-v4+native-mujoco"
        )
        self.assertEqual(metadata["mujoco_backend"], "native_mujoco")
        self.assertNotIn("mujoco_py_version", metadata)
        self.assertNotIn("diagnostic_reason", metadata)
        self.assertEqual(len(metadata["repository_status_sha256"]), 64)
        for package in (
            "Python",
            "numpy",
            "torch",
            "gymnasium",
            "mujoco",
            "h5py",
        ):
            self.assertIn(package, metadata["runtime_package_versions"])
        for legacy_package in ("gym", "d4rl", "mujoco-py"):
            self.assertNotIn(legacy_package, runtime_package_versions())


if __name__ == "__main__":
    unittest.main()
