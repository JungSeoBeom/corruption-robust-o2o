from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from plot_results import add_global_plot_steps
from robust_o2o.agents.calql import calql_td_target
from robust_o2o.agents.registry import build_agent
from robust_o2o.config import ExperimentConfig, LOCAL_PROTOCOL
from robust_o2o.corruption import (
    corrupt_offline_dataset,
    corrupt_online_transition,
    corruption_cache_fingerprint,
    mc_returns_from_reward_deltas,
    recompute_mc_returns,
)
from robust_o2o.environment import StateNormalizer, evaluate_agent
from robust_o2o.experiment import _validate_checkpoint, bounded_executed_action
from robust_o2o.networks import ExpansionGaussianPolicy
from robust_o2o.replay import (
    OfflineDataset,
    ReplayBuffer,
    balanced_priority_batch,
    mixed_batch,
    update_sample_priorities,
)


def synthetic_dataset(size: int = 64, state_dim: int = 3, action_dim: int = 2):
    rng = np.random.default_rng(5)
    return {
        "observations": rng.normal(size=(size, state_dim)).astype(np.float32),
        "actions": rng.uniform(-0.8, 0.8, size=(size, action_dim)).astype(
            np.float32
        ),
        "rewards": rng.normal(size=size).astype(np.float32),
        "next_observations": rng.normal(size=(size, state_dim)).astype(np.float32),
        "terminals": np.zeros(size, dtype=np.float32),
        "mc_returns": np.zeros(size, dtype=np.float32),
        "episode_id": np.repeat(np.arange((size + 7) // 8), 8)[:size].astype(
            np.float32
        ),
        "mc_calibration_valid": np.ones(size, dtype=np.float32),
    }


class FakeEvalEnv:
    def __init__(self):
        self.action_space = SimpleNamespace(
            low=np.array([-2.0, 0.5], dtype=np.float32),
            high=np.array([1.0, 3.0], dtype=np.float32),
        )
        self.observation_space = SimpleNamespace(shape=(3,))
        self.steps = 0

    def reset(self, seed=None):
        del seed
        self.steps = 0
        return np.zeros(3, dtype=np.float32), {}

    def step(self, action):
        self.steps += 1
        return np.zeros(3, dtype=np.float32), float(np.sum(action)), True, False, {}


class RandomAgent:
    def select_action(self, state, evaluate=False, evaluation_mode=None):
        del evaluate, evaluation_mode
        return torch.rand(2, device=state.device) * 10.0 - 5.0


class TrainingInvariantTest(unittest.TestCase):
    def test_expansion_policy_actions_are_bounded_and_density_finite(self):
        torch.manual_seed(1)
        low = torch.tensor([-2.0, 0.5])
        high = torch.tensor([1.0, 3.0])
        policy = ExpansionGaussianPolicy(
            3, 2, hidden_dim=16, hidden_layers=2, action_low=low, action_high=high
        )
        states = torch.randn(10_000, 3)
        actions, log_prob, _, _ = policy(states, need_log_prob=True)
        self.assertEqual(tuple(actions.shape), (10_000, 2))
        self.assertTrue(torch.all(actions >= low - 1e-6))
        self.assertTrue(torch.all(actions <= high + 1e-6))
        self.assertTrue(torch.isfinite(log_prob).all())
        deterministic = policy.act(states[:17], deterministic=True)
        self.assertTrue(torch.all(deterministic >= low - 1e-6))
        self.assertTrue(torch.all(deterministic <= high + 1e-6))
        boundary = torch.stack((low + 1e-7, high - 1e-7))
        self.assertTrue(torch.isfinite(policy.log_prob(states[:2], boundary)).all())

    def test_online_replay_stores_executed_action(self):
        raw = np.array([2.5, -4.0], dtype=np.float32)
        low = np.array([-1.0, -2.0], dtype=np.float32)
        high = np.array([1.0, 0.5], dtype=np.float32)
        executed = bounded_executed_action(raw, low, high)
        config = ExperimentConfig("rpex", "hopper-medium-replay-v2")
        _, stored_action, _, _, was_corrupted = corrupt_online_transition(
            np.zeros(3, dtype=np.float32),
            executed,
            0.0,
            np.ones(3, dtype=np.float32),
            config,
            None,
            np.random.default_rng(0),
            np.ones(3, dtype=np.float32),
            np.ones(2, dtype=np.float32),
        )
        self.assertFalse(was_corrupted)
        np.testing.assert_allclose(stored_action, executed)
        replay = ReplayBuffer(3, 2, 4, 0)
        replay.add(np.zeros(3), stored_action, 0.0, np.ones(3), 0.0)
        np.testing.assert_allclose(replay.actions[0], executed)
        self.assertTrue(np.all(executed >= low))
        self.assertTrue(np.all(executed <= high))

    def test_incompatible_legacy_expansion_checkpoint_fails(self):
        config = ExperimentConfig("rpex", "hopper-medium-replay-v2")
        payload = {
            "algorithm": "rpex",
            "env_name": config.env_name,
            "protocol": config.protocol,
            "state_dim": 3,
            "action_dim": 2,
            "config": {},
        }
        with self.assertRaisesRegex(ValueError, "action_distribution"):
            _validate_checkpoint(payload, config, 3, 2)

    def test_priority_metadata_survives_and_updates_both_sources(self):
        device = torch.device("cpu")
        offline = OfflineDataset(synthetic_dataset(), seed=2)
        online = ReplayBuffer(3, 2, 128, seed=3)
        data = synthetic_dataset(32)
        for index in range(32):
            online.add(
                data["observations"][index],
                data["actions"][index],
                float(data["rewards"][index]),
                data["next_observations"][index],
                0.0,
            )
        batch = mixed_batch(
            offline,
            online,
            16,
            0.5,
            device,
            prioritized_online=True,
            prioritized_offline=True,
        )
        self.assertEqual(tuple(batch["_indices"].shape), (16,))
        self.assertEqual(tuple(batch["_source"].shape), (16,))
        priorities = torch.linspace(0.1, 2.0, 16)
        stats = update_sample_priorities(offline, online, batch, priorities)
        self.assertGreater(stats["priority_std"], 0.0)
        self.assertGreater(stats["online_priority_std"], 0.0)
        self.assertGreater(stats["number_of_priority_updates"], 0.0)

    def test_high_priority_items_are_sampled_more_often(self):
        offline = OfflineDataset(synthetic_dataset(size=10), seed=7)
        offline.priorities[:] = 1.0
        offline.priorities[4] = 200.0
        counts = np.zeros(10, dtype=np.int64)
        for _ in range(1000):
            sample = offline.sample(1, torch.device("cpu"), prioritized=True)
            counts[int(sample["_indices"].item())] += 1
        self.assertGreater(counts[4], counts[np.arange(10) != 4].max() * 5)

    def test_balanced_priority_mass_controls_source_fraction(self):
        offline = OfflineDataset(synthetic_dataset(size=100), seed=1)
        online = ReplayBuffer(3, 2, 20, seed=2)
        data = synthetic_dataset(size=10)
        for index in range(10):
            online.add(
                data["observations"][index],
                data["actions"][index],
                float(data["rewards"][index]),
                data["next_observations"][index],
                0.0,
                priority=10.0,
            )
        batch = balanced_priority_batch(
            offline, online, 20, torch.device("cpu")
        )
        self.assertEqual(int((batch["_source"] == 0).sum()), 10)
        self.assertEqual(int((batch["_source"] == 1).sum()), 10)

    def test_uniform_pqe_replay_is_explicit_ablation(self):
        config = ExperimentConfig(
            "pessimistic_q_ensemble",
            "hopper-medium-replay-v2",
            pqe_replay_mode="uniform",
        )
        self.assertEqual(config.pqe_replay_mode, "uniform")

    def test_pqe_priorities_change_after_update(self):
        torch.manual_seed(4)
        device = torch.device("cpu")
        config = ExperimentConfig(
            "pessimistic_q_ensemble",
            "hopper-medium-replay-v2",
            hidden_dim=16,
            hidden_layers=2,
            sac_num_critics=2,
            cql_n_actions=2,
            batch_size=16,
        )
        agent = build_agent(config, 3, 2, 1.0, device)
        agent.begin_online()
        offline = OfflineDataset(synthetic_dataset(), 1)
        online = ReplayBuffer(3, 2, 64, 2)
        data = synthetic_dataset(32)
        for index in range(32):
            online.add(
                data["observations"][index], data["actions"][index],
                float(data["rewards"][index]), data["next_observations"][index], 0.0
            )
        batch = mixed_batch(offline, online, 16, 0.5, device)
        density_offline = offline.sample(16, device, prioritized=False)
        density_online = online.sample(16, device, prioritized=False)
        metrics = agent.update(
            rl_batch=batch,
            density_offline_batch=density_offline,
            density_online_batch=density_online,
        )
        priorities = agent.consume_priority_values()
        self.assertIsNotNone(priorities)
        self.assertTrue(torch.isfinite(priorities).all())
        stats = update_sample_priorities(offline, online, batch, priorities)
        self.assertGreater(stats["priority_std"] + stats["online_priority_std"], 0.0)
        self.assertTrue(all(np.isfinite(value) for value in metrics.values()))

    def test_reward_corruption_recomputes_mc_returns_and_validity(self):
        dataset = synthetic_dataset(size=8)
        dataset["episode_id"] = np.array([0, 0, 0, 1, 1, 1, 1, 1], np.float32)
        config = ExperimentConfig(
            "cal_ql",
            "hopper-medium-replay-v2",
            corruption="random",
            corruption_target="rewards",
            offline_corruption_rate=1.0,
            discount=0.5,
        )
        with tempfile.TemporaryDirectory() as directory:
            result, _ = corrupt_offline_dataset(
                dataset, config, None, Path(directory)
            )
        expected = mc_returns_from_reward_deltas(
            dataset["rewards"],
            result["rewards"],
            dataset["mc_returns"],
            dataset["episode_id"],
            0.5,
        )
        np.testing.assert_allclose(result["mc_returns"], expected)
        np.testing.assert_array_equal(result["mc_calibration_valid"], 1.0)

        config = ExperimentConfig(
            "cal_ql",
            "hopper-medium-replay-v2",
            corruption="random",
            corruption_target="actions",
            offline_corruption_rate=1.0,
        )
        with tempfile.TemporaryDirectory() as directory:
            invalid, _ = corrupt_offline_dataset(
                dataset, config, None, Path(directory)
            )
        np.testing.assert_array_equal(invalid["mc_calibration_valid"], 0.0)

    def test_mc_returns_do_not_leak_across_episode_boundaries(self):
        dataset = synthetic_dataset(size=4)
        dataset["rewards"] = np.array([1.0, 2.0, 10.0, 20.0], np.float32)
        dataset["episode_id"] = np.array([0, 0, 1, 1], np.float32)
        np.testing.assert_allclose(
            recompute_mc_returns(dataset, 0.5),
            np.array([2.0, 2.0, 20.0, 20.0], np.float32),
        )

    def test_legacy_mc_returns_require_explicit_mode(self):
        dataset = synthetic_dataset(size=8)
        original = np.arange(8, dtype=np.float32)
        dataset["mc_returns"] = original.copy()
        config = ExperimentConfig(
            "cal_ql",
            "hopper-medium-replay-v2",
            corruption="random",
            corruption_target="rewards",
            offline_corruption_rate=1.0,
            mc_return_source="legacy_pre_corruption",
        )
        with tempfile.TemporaryDirectory() as directory:
            result, _ = corrupt_offline_dataset(
                dataset, config, None, Path(directory)
            )
        np.testing.assert_array_equal(result["mc_returns"], original)

    def test_calql_backup_entropy_switch(self):
        rewards = torch.tensor([1.0])
        terminals = torch.tensor([0.0])
        next_q = torch.tensor([4.0])
        next_log_prob = torch.tensor([-2.0])
        alpha = torch.tensor(0.5)
        without = calql_td_target(
            rewards, terminals, next_q, next_log_prob, 0.9, alpha, False
        )
        with_entropy = calql_td_target(
            rewards, terminals, next_q, next_log_prob, 0.9, alpha, True
        )
        self.assertTrue(torch.allclose(without, torch.tensor([4.6])))
        self.assertTrue(torch.allclose(with_entropy, torch.tensor([5.5])))

    def test_calql_validity_mask_all_some_none_is_finite(self):
        device = torch.device("cpu")
        for expected_fraction, mask in (
            (1.0, torch.ones(8)),
            (0.5, torch.tensor([1, 0, 1, 0, 1, 0, 1, 0], dtype=torch.float32)),
            (0.0, torch.zeros(8)),
        ):
            torch.manual_seed(3)
            config = ExperimentConfig(
                "cal_ql",
                "hopper-medium-replay-v2",
                hidden_dim=16,
                hidden_layers=2,
                cql_n_actions=2,
                bc_steps=0,
            )
            agent = build_agent(config, 3, 2, 1.0, device)
            data = synthetic_dataset(size=8)
            batch = {
                key: torch.as_tensor(value, dtype=torch.float32)
                for key, value in data.items()
            }
            batch["mc_calibration_valid"] = mask
            metrics = agent.update(batch)
            self.assertAlmostEqual(
                metrics["mc_calibration_valid_fraction"], expected_fraction
            )
            self.assertTrue(all(np.isfinite(value) for value in metrics.values()))

    def test_online_curve_is_offset_after_offline_phase(self):
        import pandas as pd

        frame = pd.DataFrame(
            {
                "phase": ["offline", "offline", "online", "online"],
                "step": [10, 20, 1, 2],
                "env_steps": [0, 0, 1, 2],
            }
        )
        plotted = add_global_plot_steps(frame)
        self.assertGreaterEqual(
            plotted.loc[plotted.phase == "online", "global_step"].min(),
            plotted.loc[plotted.phase == "offline", "global_step"].max(),
        )

    def test_evaluation_does_not_mutate_training_rng(self):
        env = FakeEvalEnv()
        normalizer = StateNormalizer(
            np.zeros(3, dtype=np.float32), np.ones(3, dtype=np.float32)
        )
        torch.manual_seed(123)
        expected_next = torch.rand(5)
        torch.manual_seed(123)
        evaluate_agent(
            env,
            "hopper-medium-replay-v2",
            RandomAgent(),
            normalizer,
            torch.device("cpu"),
            episodes=3,
            max_episode_steps=1,
            seed=0,
            protocol=LOCAL_PROTOCOL,
            evaluation_mode="method_faithful",
        )
        actual_next = torch.rand(5)
        self.assertTrue(torch.equal(expected_next, actual_next))

    def test_rpex_deterministic_diagnostic_is_repeatable(self):
        torch.manual_seed(6)
        config = ExperimentConfig(
            "rpex",
            "hopper-medium-replay-v2",
            hidden_dim=16,
            hidden_layers=2,
            num_critics=3,
        )
        agent = build_agent(config, 3, 2, 1.0, torch.device("cpu"))
        agent.begin_online()
        state = torch.tensor([0.1, -0.2, 0.3])
        first = agent.select_action(
            state, evaluate=True, evaluation_mode="deterministic_diagnostic"
        )
        second = agent.select_action(
            state, evaluate=True, evaluation_mode="deterministic_diagnostic"
        )
        self.assertTrue(torch.equal(first, second))

    def test_cache_fingerprint_covers_behavior_inputs(self):
        dataset = synthetic_dataset(size=8)
        base = ExperimentConfig(
            "rpex", "hopper-medium-replay-v2",
            corruption="adversarial", corruption_target="rewards"
        )
        fingerprint, _ = corruption_cache_fingerprint(dataset, base, None)
        changed_steps = ExperimentConfig(**{**base.__dict__, "offline_attack_steps": 7})
        changed_size = ExperimentConfig(**{**base.__dict__, "attack_step_size": 0.2})
        changed_target = ExperimentConfig(**{**base.__dict__, "corruption_target": "actions"})
        changed_dataset = {key: value.copy() for key, value in dataset.items()}
        changed_dataset["rewards"][0] += 1.0
        self.assertNotEqual(fingerprint, corruption_cache_fingerprint(dataset, changed_steps, None)[0])
        self.assertNotEqual(fingerprint, corruption_cache_fingerprint(dataset, changed_size, None)[0])
        self.assertNotEqual(fingerprint, corruption_cache_fingerprint(dataset, changed_target, None)[0])
        self.assertNotEqual(fingerprint, corruption_cache_fingerprint(changed_dataset, base, None)[0])
        with tempfile.TemporaryDirectory() as directory:
            first_path = Path(directory) / "first.pt"
            second_path = Path(directory) / "second.pt"
            first_path.write_bytes(b"checkpoint-a")
            second_path.write_bytes(b"checkpoint-b")
            first_oracle = SimpleNamespace(checkpoint=first_path)
            second_oracle = SimpleNamespace(checkpoint=second_path)
            self.assertNotEqual(
                corruption_cache_fingerprint(dataset, base, first_oracle)[0],
                corruption_cache_fingerprint(dataset, base, second_oracle)[0],
            )

    def test_robust_normalizer_resists_extreme_outlier(self):
        dataset = synthetic_dataset(size=32)
        dataset["observations"][0, 0] = 1e9
        standard = StateNormalizer.fit(dataset, mode="standard")
        robust = StateNormalizer.fit(dataset, mode="robust_median_mad")
        self.assertLess(abs(float(robust.mean[0])), abs(float(standard.mean[0])))
        self.assertTrue(np.all(robust.std >= 1e-3))
        restored = StateNormalizer.from_state_dict(robust.state_dict())
        self.assertEqual(restored.mode, "robust_median_mad")

    def test_tiny_clean_learning_changes_parameters_and_improves_action(self):
        torch.manual_seed(9)
        config = ExperimentConfig(
            "pex",
            "hopper-medium-replay-v2",
            hidden_dim=32,
            hidden_layers=2,
            learning_rate=1e-3,
            batch_size=64,
        )
        agent = build_agent(config, 1, 1, 1.0, torch.device("cpu"))
        states = torch.zeros(64, 1)
        optimal_action = 0.7
        batch = {
            "observations": states,
            "actions": torch.full((64, 1), optimal_action),
            "rewards": torch.ones(64),
            "next_observations": states,
            "terminals": torch.ones(64),
            "mc_returns": torch.ones(64),
        }
        before_action = float(agent.select_action(torch.zeros(1), evaluate=True))
        before_parameters = [parameter.detach().clone() for parameter in agent.parameters()]
        with torch.no_grad():
            before_q = float(agent.critic(states[:1], batch["actions"][:1]).mean())
        metrics = {}
        for _ in range(150):
            metrics = agent.update(batch)
        after_action = float(agent.select_action(torch.zeros(1), evaluate=True))
        with torch.no_grad():
            after_q = float(agent.critic(states[:1], batch["actions"][:1]).mean())
        parameter_delta = sum(
            float((after.detach() - before).abs().sum())
            for after, before in zip(agent.parameters(), before_parameters)
        )
        self.assertGreater(parameter_delta, 0.0)
        self.assertLess(abs(after_action - optimal_action), abs(before_action - optimal_action))
        before_return = -(before_action - optimal_action) ** 2
        after_return = -(after_action - optimal_action) ** 2
        self.assertGreater(after_return, before_return + 0.1)
        self.assertLess(abs(after_q - 1.0), abs(before_q - 1.0))
        self.assertTrue(all(np.isfinite(value) for value in metrics.values()))
        self.assertTrue(
            all(torch.isfinite(parameter).all() for parameter in agent.parameters())
        )


if __name__ == "__main__":
    unittest.main()
