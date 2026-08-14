from __future__ import annotations

import random
import unittest
from unittest.mock import patch

import numpy as np
import torch

from robust_o2o.agents.registry import build_agent
from robust_o2o.config import ExperimentConfig, LOCAL_PROTOCOL
from robust_o2o.corruption import (
    _apply_mc_return_semantics,
    mc_returns_from_reward_deltas,
)
from robust_o2o.device import seed_env_only, seed_everything
from robust_o2o.environment import StateNormalizer, evaluate_agent
from robust_o2o.experiment import _make_evaluation_env
from robust_o2o.replay import (
    OfflineDataset,
    ReplayBuffer,
    sample_pqe_update_batches,
    update_sample_priorities,
)


def synthetic_dataset(size: int = 64) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(17)
    return {
        "observations": rng.normal(size=(size, 3)).astype(np.float32),
        "actions": rng.uniform(-0.9, 0.9, size=(size, 2)).astype(np.float32),
        "rewards": rng.normal(size=size).astype(np.float32),
        "next_observations": rng.normal(size=(size, 3)).astype(np.float32),
        "terminals": np.zeros(size, dtype=np.float32),
        "mc_returns": np.zeros(size, dtype=np.float32),
        "episode_id": np.repeat(np.arange((size + 7) // 8), 8)[:size].astype(
            np.float32
        ),
        "mc_calibration_valid": np.ones(size, dtype=np.float32),
    }


class SeedSpace:
    def __init__(self, shape: tuple[int, ...]):
        self.shape = shape
        self.low = -np.ones(shape, dtype=np.float32)
        self.high = np.ones(shape, dtype=np.float32)
        self.seed_calls: list[int] = []

    def seed(self, seed: int) -> None:
        self.seed_calls.append(seed)


class RngConsumingEvalEnv:
    def __init__(self):
        random.random()
        np.random.random()
        torch.rand(1)
        self.action_space = SeedSpace((2,))
        self.observation_space = SeedSpace((3,))
        self.steps = 0

    def reset(self, seed=None):
        del seed
        random.random()
        np.random.random()
        torch.rand(1)
        self.steps = 0
        return np.zeros(3, dtype=np.float32), {}

    def step(self, action):
        self.steps += 1
        return np.zeros(3, dtype=np.float32), float(np.sum(action)), True, False, {}

    def close(self):
        pass


class RandomPolicyAgent:
    def select_action(self, state, evaluate=False, evaluation_mode=None):
        del evaluate, evaluation_mode
        return torch.rand(2, device=state.device)


def flattened_parameters(agent: torch.nn.Module) -> torch.Tensor:
    return torch.cat([parameter.detach().reshape(-1) for parameter in agent.parameters()])


class EvaluationSeedRegressionTest(unittest.TestCase):
    def test_eval_environment_creation_and_seeding_preserve_all_global_rngs(self):
        seed_everything(123)
        expected = (random.random(), np.random.random(), torch.rand(4))
        seed_everything(123)
        with patch(
            "robust_o2o.experiment.make_env",
            side_effect=lambda *_: RngConsumingEvalEnv(),
        ):
            env = _make_evaluation_env("hopper-medium-replay-v2", LOCAL_PROTOCOL, 77)
        actual = (random.random(), np.random.random(), torch.rand(4))
        self.assertEqual(expected[0], actual[0])
        self.assertEqual(expected[1], actual[1])
        self.assertTrue(torch.equal(expected[2], actual[2]))
        self.assertEqual(env.action_space.seed_calls, [77])
        self.assertEqual(env.observation_space.seed_calls, [77])

    def test_seed_env_only_does_not_touch_global_rngs(self):
        env = RngConsumingEvalEnv()
        seed_everything(5)
        expected = (random.random(), np.random.random(), torch.rand(3))
        seed_everything(5)
        seed_env_only(env, 99)
        actual = (random.random(), np.random.random(), torch.rand(3))
        self.assertEqual(expected[0], actual[0])
        self.assertEqual(expected[1], actual[1])
        self.assertTrue(torch.equal(expected[2], actual[2]))

    def test_config_seed_matches_explicit_seed_and_is_repeatable(self):
        config = ExperimentConfig(
            "riql_naive",
            "hopper-medium-replay-v2",
            seed=41,
            hidden_dim=16,
            hidden_layers=2,
            num_critics=3,
        )
        seed_everything(config.seed)
        from_config_seed = build_agent(config, 3, 2, 1.0, torch.device("cpu"))
        seed_everything(41)
        from_explicit_seed = build_agent(config, 3, 2, 1.0, torch.device("cpu"))
        seed_everything(config.seed)
        repeated = build_agent(config, 3, 2, 1.0, torch.device("cpu"))
        self.assertTrue(
            torch.equal(
                flattened_parameters(from_config_seed),
                flattened_parameters(from_explicit_seed),
            )
        )
        self.assertTrue(
            torch.equal(
                flattened_parameters(from_config_seed),
                flattened_parameters(repeated),
            )
        )

    def test_different_training_seeds_change_initial_parameters(self):
        config = ExperimentConfig(
            "riql_naive",
            "hopper-medium-replay-v2",
            hidden_dim=16,
            hidden_layers=2,
            num_critics=3,
        )
        seed_everything(11)
        first = build_agent(config, 3, 2, 1.0, torch.device("cpu"))
        seed_everything(12)
        second = build_agent(config, 3, 2, 1.0, torch.device("cpu"))
        self.assertFalse(
            torch.equal(flattened_parameters(first), flattened_parameters(second))
        )

    def test_clean_iql_tiny_same_seed_runs_match_initialization_and_outputs(self):
        config = ExperimentConfig(
            "riql_naive",
            "hopper-medium-replay-v2",
            seed=23,
            hidden_dim=16,
            hidden_layers=2,
            num_critics=3,
            batch_size=8,
        )
        data = synthetic_dataset(8)
        batch = {
            key: torch.as_tensor(value, dtype=torch.float32)
            for key, value in data.items()
        }

        def tiny_run():
            seed_everything(config.seed)
            agent = build_agent(config, 3, 2, 1.0, torch.device("cpu"))
            initial = flattened_parameters(agent).clone()
            initial_action = agent.select_action(
                torch.zeros(3), evaluate=True
            ).clone()
            for _ in range(3):
                agent.update(batch)
            final_action = agent.select_action(
                torch.zeros(3), evaluate=True
            ).clone()
            return initial, initial_action, final_action

        first = tiny_run()
        second = tiny_run()
        for first_value, second_value in zip(first, second):
            self.assertTrue(torch.equal(first_value, second_value))

    def test_evaluation_preserves_next_training_draw_and_policy_sample(self):
        env = RngConsumingEvalEnv()
        agent = RandomPolicyAgent()
        normalizer = StateNormalizer(np.zeros(3), np.ones(3))
        state = torch.zeros(3)
        seed_everything(88)
        expected_random = (random.random(), np.random.random())
        expected_action = agent.select_action(state, evaluate=False)
        seed_everything(88)
        evaluate_agent(
            env,
            "hopper-medium-replay-v2",
            agent,
            normalizer,
            torch.device("cpu"),
            episodes=2,
            max_episode_steps=1,
            seed=4,
            protocol=LOCAL_PROTOCOL,
            evaluation_mode="method_faithful",
        )
        actual_random = (random.random(), np.random.random())
        actual_action = agent.select_action(state, evaluate=False)
        self.assertEqual(expected_random, actual_random)
        self.assertTrue(torch.equal(expected_action, actual_action))


class CalQLReturnRegressionTest(unittest.TestCase):
    def adjusted(self, clean, corrupted, returns, episodes, gamma=0.5):
        return mc_returns_from_reward_deltas(
            np.asarray(clean, dtype=np.float32),
            np.asarray(corrupted, dtype=np.float32),
            np.asarray(returns, dtype=np.float32),
            np.asarray(episodes, dtype=np.int64),
            gamma,
        )

    def test_timeout_tail_is_preserved(self):
        actual = self.adjusted([1], [3], [6], [0])
        np.testing.assert_array_equal(actual, np.asarray([8], dtype=np.float32))

    def test_multiple_episodes_do_not_leak(self):
        actual = self.adjusted(
            [1, 2, 10, 20], [2, 2, 10, 22], [2, 2, 20, 20], [0, 0, 1, 1]
        )
        np.testing.assert_array_equal(
            actual, np.asarray([3, 2, 21, 22], dtype=np.float32)
        )

    def test_terminal_episode_reward_change(self):
        actual = self.adjusted([1, 2], [1, 4], [2, 2], [0, 0])
        np.testing.assert_array_equal(actual, np.asarray([3, 4], dtype=np.float32))

    def test_timeout_episode_reward_change(self):
        actual = self.adjusted([1], [2], [6], [0])
        np.testing.assert_array_equal(actual, np.asarray([7], dtype=np.float32))

    def test_first_middle_and_final_retained_reward_changes(self):
        clean = [1, 2, 3]
        returns = [2.75, 3.5, 3]
        episodes = [0, 0, 0]
        cases = (
            ([3, 2, 3], [4.75, 3.5, 3]),
            ([1, 4, 3], [3.75, 5.5, 3]),
            ([1, 2, 5], [3.25, 4.5, 5]),
        )
        for corrupted, expected in cases:
            with self.subTest(corrupted=corrupted):
                np.testing.assert_array_equal(
                    self.adjusted(clean, corrupted, returns, episodes),
                    np.asarray(expected, dtype=np.float32),
                )

    def test_no_reward_change_keeps_clean_returns_exactly(self):
        returns = np.asarray([6.0, 4.0], dtype=np.float32)
        actual = self.adjusted([1, 2], [1, 2], returns, [0, 0])
        np.testing.assert_array_equal(actual, returns)

    def test_state_action_and_next_state_corruption_invalidate_calibration(self):
        clean = synthetic_dataset(4)
        clean_returns = np.asarray([4.0, 3.0, 2.0, 1.0], dtype=np.float32)
        clean["mc_returns"] = clean_returns.copy()
        for target in ("observations", "actions", "dynamics"):
            with self.subTest(target=target):
                result = {key: value.copy() for key, value in clean.items()}
                config = ExperimentConfig(
                    "cal_ql",
                    "hopper-medium-replay-v2",
                    corruption="random",
                    corruption_target=target,
                )
                _apply_mc_return_semantics(
                    clean,
                    result,
                    {target: np.asarray([1], dtype=np.int64)},
                    config,
                )
                np.testing.assert_array_equal(result["mc_returns"], clean_returns)
                np.testing.assert_array_equal(
                    result["mc_calibration_valid"],
                    np.asarray([1, 0, 1, 1], dtype=np.float32),
                )


class PQEBatchRoutingRegressionTest(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cpu")
        self.offline = OfflineDataset(synthetic_dataset(96), seed=2)
        self.online = ReplayBuffer(3, 2, 128, seed=3)
        data = synthetic_dataset(64)
        for index in range(64):
            self.online.add(
                data["observations"][index],
                data["actions"][index],
                float(data["rewards"][index]),
                data["next_observations"][index],
                0.0,
            )

    def sample(self, prioritized=True):
        return sample_pqe_update_batches(
            self.offline,
            self.online,
            16,
            0.5,
            self.device,
            prioritized_rl=prioritized,
        )

    def test_density_sampling_is_uniform_and_rl_sampling_is_prioritized(self):
        with patch.object(
            self.offline, "sample", wraps=self.offline.sample
        ) as offline_sample, patch.object(
            self.online, "sample", wraps=self.online.sample
        ) as online_sample:
            self.sample(prioritized=True)
        self.assertFalse(offline_sample.call_args_list[0].kwargs["prioritized"])
        self.assertFalse(online_sample.call_args_list[0].kwargs["prioritized"])
        self.assertTrue(offline_sample.call_args_list[1].kwargs["prioritized"])
        self.assertTrue(online_sample.call_args_list[1].kwargs["prioritized"])

    def test_density_and_rl_batches_are_distinct(self):
        rl_batch, density_offline, density_online = self.sample()
        self.assertIsNot(rl_batch, density_offline)
        self.assertIsNot(rl_batch, density_online)
        rl_offline_indices = rl_batch["_indices"][rl_batch["_source"] == 0]
        rl_online_indices = rl_batch["_indices"][rl_batch["_source"] == 1]
        self.assertFalse(torch.equal(rl_offline_indices, density_offline["_indices"]))
        self.assertFalse(torch.equal(rl_online_indices, density_online["_indices"]))

    def test_density_updates_and_priorities_write_to_aligned_indices(self):
        seed_everything(9)
        config = ExperimentConfig(
            "pessimistic_q_ensemble",
            "hopper-medium-replay-v2",
            hidden_dim=16,
            hidden_layers=2,
            sac_num_critics=2,
            cql_n_actions=2,
            batch_size=16,
        )
        agent = build_agent(config, 3, 2, 1.0, self.device)
        agent.begin_online()
        before = flattened_parameters(agent.density_ratio).clone()
        rl_batch, density_offline, density_online = self.sample()
        metrics = agent.update(
            rl_batch=rl_batch,
            density_offline_batch=density_offline,
            density_online_batch=density_online,
            rl_batch_prioritized=True,
        )
        after = flattened_parameters(agent.density_ratio)
        self.assertFalse(torch.equal(before, after))
        priorities = agent.consume_priority_values()
        self.assertIsNotNone(priorities)
        self.assertGreater(float(priorities.std(unbiased=False)), 0.0)
        stats = update_sample_priorities(
            self.offline, self.online, rl_batch, priorities
        )
        offline_indices = rl_batch["_indices"][rl_batch["_source"] == 0].cpu().numpy()
        online_indices = rl_batch["_indices"][rl_batch["_source"] == 1].cpu().numpy()
        self.assertTrue(np.any(self.offline.priorities[offline_indices] != 1.0))
        self.assertTrue(np.any(self.online.priorities[online_indices] != 1.0))
        self.assertEqual(self.offline.priority_updates, len(offline_indices))
        self.assertEqual(self.online.priority_updates, len(online_indices))
        self.assertGreater(stats["number_of_priority_updates"], 0.0)
        self.assertEqual(metrics["density_batches_prioritized"], 0.0)
        self.assertEqual(metrics["rl_batch_prioritized"], 1.0)
        self.assertEqual(metrics["density_offline_count"], 16.0)
        self.assertEqual(metrics["density_online_count"], 16.0)

    def test_resolved_configuration_labels_shared_actor_approximation(self):
        config = ExperimentConfig(
            "pqe_shared_actor", "hopper-medium-replay-v2"
        )
        self.assertEqual(config.algorithm, "pessimistic_q_ensemble")
        self.assertEqual(config.to_dict()["implementation_variant"], "shared_actor_approx")


if __name__ == "__main__":
    unittest.main()
