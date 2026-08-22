from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np
import torch

from robust_o2o.agents.calql import (
    CalQLAgent,
    CalQLQNetwork,
    CalQLTanhGaussianPolicy,
)
from robust_o2o.calql_online import (
    CALQL_TRANSITION_KEYS,
    CalQLTrajectoryAccumulator,
    discounted_return_to_go,
    dynamic_batch_counts,
    episodic_return_to_go,
)
from robust_o2o.config import ExperimentConfig, LEGACY_PROTOCOL


def calql_config(**overrides):
    values = {
        "algorithm": "cal_ql",
        "hidden_dim": 16,
        "hidden_layers": 2,
        "actor_learning_rate": 1e-4,
        "critic_learning_rate": 3e-4,
        "temperature_learning_rate": 1e-4,
        "target_entropy": -2.0,
        "cql_n_actions": 3,
        "cql_temperature": 1.0,
        "cql_alpha": 5.0,
        "cql_alpha_online": 5.0,
        "calibration_mask_mode": "all",
        "enable_calql": True,
        "calql_bc_warmup_steps": 0,
        "cql_max_target_backup": True,
        "backup_entropy": False,
        "discount": 0.99,
        "target_update_rate": 0.005,
        "max_grad_norm": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def tensor_batch(size: int = 8, mc_return: float = 100.0):
    return {
        "observations": torch.randn(size, 3),
        "actions": torch.tanh(torch.randn(size, 2)),
        "rewards": torch.randn(size),
        "next_observations": torch.randn(size, 3),
        "terminals": torch.zeros(size),
        "mc_returns": torch.full((size,), mc_return),
    }


class CalQLMainConfigTest(unittest.TestCase):
    @staticmethod
    def config(**overrides):
        values = {
            "algorithm": "cal_ql",
            "env_name": "hopper-medium-replay-v2",
            "run_purpose": "research_benchmark",
            "suite_profile": "research_benchmark",
            "implementation_profile": "research_benchmark",
            "protocol": LEGACY_PROTOCOL,
        }
        values.update(overrides)
        return ExperimentConfig(**values)

    def test_frozen_source_aligned_locomotion_defaults(self):
        config = self.config()
        self.assertEqual((config.hidden_dim, config.hidden_layers), (256, 2))
        self.assertEqual(config.actor_learning_rate, 1e-4)
        self.assertEqual(config.critic_learning_rate, 3e-4)
        self.assertEqual(config.temperature_learning_rate, 1e-4)
        self.assertEqual(config.cql_alpha, 5.0)
        self.assertEqual(config.cql_alpha_online, 5.0)
        self.assertEqual(config.cql_n_actions, 10)
        self.assertEqual(config.cql_temperature, 1.0)
        self.assertTrue(config.cql_importance_sample)
        self.assertTrue(config.cql_max_target_backup)
        self.assertFalse(config.backup_entropy)
        self.assertTrue(config.enable_calql)
        self.assertEqual(config.calibration_mask_mode, "all")
        self.assertEqual(config.calql_bc_warmup_steps, 0)
        self.assertEqual(config.updates_per_step, 1)
        self.assertIsNone(config.offline_ratio)
        self.assertEqual(config.online_replay_profile, "dynamic_offline_online_mixture")

    def test_main_rejects_bc_and_disabled_online_calibration(self):
        with self.assertRaisesRegex(ValueError, "Cal-QL main frozen config"):
            self.config(calql_bc_warmup_steps=1)
        with self.assertRaisesRegex(ValueError, "Cal-QL main frozen config"):
            self.config(enable_calql=False)


class CalQLNetworkAndLossTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.agent = CalQLAgent(calql_config(), 3, 2, 1.0, torch.device("cpu"))

    def test_source_aligned_two_by_hidden_networks_and_optimizers(self):
        self.assertIsInstance(self.agent.actor, CalQLTanhGaussianPolicy)
        self.assertIsInstance(self.agent.q1, CalQLQNetwork)
        actor_hidden = [
            module
            for module in self.agent.actor.trunk.modules()
            if isinstance(module, torch.nn.Linear)
        ]
        q_linears = [
            module
            for module in self.agent.q1.net.modules()
            if isinstance(module, torch.nn.Linear)
        ]
        self.assertEqual(len(actor_hidden), 2)
        self.assertEqual(len(q_linears), 3)  # two hidden layers plus Q head
        self.assertEqual(self.agent.actor.output.out_features, 4)
        self.assertAlmostEqual(
            torch.linalg.vector_norm(self.agent.actor.output.weight).item(),
            np.sqrt(self.agent.actor.output.out_features) * 1e-2,
            places=6,
        )
        self.assertEqual(float(self.agent.actor.log_std_multiplier.detach()), 1.0)
        self.assertEqual(float(self.agent.actor.log_std_offset.detach()), -1.0)
        self.assertEqual(self.agent.actor_optimizer.param_groups[0]["lr"], 1e-4)
        self.assertEqual(self.agent.q1_optimizer.param_groups[0]["lr"], 3e-4)
        self.assertEqual(self.agent.alpha_optimizer.param_groups[0]["lr"], 1e-4)

    def test_agent_does_not_expand_or_shrink_frozen_two_layer_network(self):
        agent = CalQLAgent(
            calql_config(hidden_layers=7), 3, 2, 1.0, torch.device("cpu")
        )
        actor_hidden = [
            module
            for module in agent.actor.trunk.modules()
            if isinstance(module, torch.nn.Linear)
        ]
        self.assertEqual(len(actor_hidden), 2)

    def test_offline_and_online_calibration_bound_current_and_next_proposals(self):
        offline_metrics = self.agent.update(tensor_batch())
        self.assertEqual(offline_metrics["calql_calibration_enabled"], 1.0)
        self.assertEqual(offline_metrics["calql_current_calibration_bound_rate"], 1.0)
        self.assertEqual(offline_metrics["calql_next_calibration_bound_rate"], 1.0)
        self.assertEqual(offline_metrics["cql_weight"], 5.0)

        self.agent.begin_online()
        online_batch = tensor_batch()
        # A diagnostic mask must not alter main Cal-QL's all-sample bound.
        online_batch["mc_calibration_valid"] = torch.zeros(8)
        online_metrics = self.agent.update(online_batch)
        self.assertEqual(online_metrics["calql_calibration_enabled"], 1.0)
        self.assertEqual(online_metrics["mc_calibration_valid_fraction"], 1.0)
        self.assertEqual(online_metrics["calql_current_calibration_bound_rate"], 1.0)
        self.assertEqual(online_metrics["calql_next_calibration_bound_rate"], 1.0)
        self.assertEqual(online_metrics["online_calibration_bound_rate"], 1.0)
        self.assertEqual(online_metrics["cql_weight"], 5.0)
        self.assertGreater(abs(online_metrics["cql_loss"]), 0.0)

    def test_missing_or_nonfinite_mc_return_is_rejected(self):
        missing = tensor_batch()
        missing.pop("mc_returns")
        with self.assertRaisesRegex(ValueError, "exact mc_returns"):
            self.agent.update(missing)
        invalid = tensor_batch()
        invalid["mc_returns"][3] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite"):
            self.agent.update(invalid)

    def test_sac_actor_loss_has_no_bc_warmup_branch(self):
        agent = CalQLAgent(
            calql_config(calql_bc_warmup_steps=100_000),
            3,
            2,
            1.0,
            torch.device("cpu"),
        )
        before = [parameter.detach().clone() for parameter in agent.actor.parameters()]
        metrics = agent.update(tensor_batch(mc_return=1.0))
        delta = sum(
            float((after.detach() - previous).abs().sum())
            for after, previous in zip(agent.actor.parameters(), before)
        )
        self.assertEqual(metrics["actor_update_mode_bc_warmup"], 0.0)
        self.assertGreater(delta, 0.0)

    def test_deterministic_evaluation_uses_tanh_policy_mean(self):
        states = torch.randn(5, 3)
        with torch.no_grad():
            distribution = self.agent.actor.distribution(states)
            expected = torch.tanh(distribution.mean)
            actual = self.agent.select_action(states, evaluate=True)
        torch.testing.assert_close(actual, expected)


class CalQLTrajectorySemanticsTest(unittest.TestCase):
    @staticmethod
    def append_episode(
        target: str,
        clean_rewards: tuple[float, ...],
        corrupted_rewards: tuple[float, ...],
    ):
        accumulator = CalQLTrajectoryAccumulator(discount=0.5)
        completed = None
        for index, (clean_reward, corrupted_reward) in enumerate(
            zip(clean_rewards, corrupted_rewards)
        ):
            stored_reward = corrupted_reward if target == "rewards" else clean_reward
            completed = accumulator.append(
                observation=np.asarray([index, 0.0], dtype=np.float32),
                action=np.asarray([0.25], dtype=np.float32),
                reward=stored_reward,
                next_observation=np.asarray([index + 1, 0.0], dtype=np.float32),
                terminal=index == len(clean_rewards) - 1,
            )
        assert completed is not None
        return completed

    def test_reward_corruption_uses_corrupted_trajectory_rewards(self):
        completed = self.append_episode("rewards", (1.0, 2.0), (4.0, 8.0))
        np.testing.assert_allclose(completed.batch["rewards"], [4.0, 8.0])
        np.testing.assert_allclose(completed.batch["mc_returns"], [8.0, 8.0])

    def test_other_corruption_targets_preserve_reward_return_sequence(self):
        for target in ("observations", "actions", "dynamics"):
            with self.subTest(target=target):
                completed = self.append_episode(target, (1.0, 2.0), (100.0, 200.0))
                np.testing.assert_allclose(completed.batch["rewards"], [1.0, 2.0])
                np.testing.assert_allclose(completed.batch["mc_returns"], [2.0, 2.0])

    def test_pending_episode_never_emits_fake_mc_return(self):
        accumulator = CalQLTrajectoryAccumulator(discount=0.99)
        result = accumulator.append(
            observation=np.zeros(2),
            action=np.zeros(1),
            reward=3.0,
            next_observation=np.ones(2),
            terminal=False,
            timeout=False,
        )
        self.assertIsNone(result)
        self.assertEqual(accumulator.pending_episode_length, 1)
        self.assertEqual(accumulator.online_mc_return_valid_fraction, 0.0)

    def test_timeout_is_mc_boundary_but_not_bellman_terminal(self):
        accumulator = CalQLTrajectoryAccumulator(discount=0.5)
        self.assertIsNone(
            accumulator.append(
                observation=np.zeros(2),
                action=np.zeros(1),
                reward=2.0,
                next_observation=np.ones(2),
                terminal=False,
            )
        )
        completed = accumulator.append(
            observation=np.ones(2),
            action=np.ones(1),
            reward=4.0,
            next_observation=np.full(2, 2.0),
            terminal=False,
            timeout=True,
        )
        self.assertIsNotNone(completed)
        np.testing.assert_allclose(completed.batch["mc_returns"], [4.0, 4.0])
        np.testing.assert_array_equal(completed.batch["terminals"], [0.0, 0.0])
        self.assertEqual(set(completed.batch), set(CALQL_TRANSITION_KEYS))
        self.assertFalse(any("corrupt" in key for key in completed.batch))
        self.assertEqual(completed.update_count(1), 2)
        self.assertEqual(
            accumulator.metadata(),
            {
                "completed_online_trajectories": 1,
                "completed_online_transitions": 2,
                "pending_episode_length": 0,
                "online_mc_return_valid_fraction": 1.0,
            },
        )

    def test_terminal_and_timeout_boundaries_do_not_leak_returns(self):
        actual = episodic_return_to_go(
            rewards=np.asarray([1.0, 2.0, 10.0, 20.0]),
            terminals=np.asarray([0, 1, 0, 0]),
            timeouts=np.asarray([0, 0, 0, 1]),
            discount=0.5,
        )
        np.testing.assert_allclose(actual, [2.0, 2.0, 20.0, 20.0])

    def test_pending_trajectory_round_trips_for_resume(self):
        first = CalQLTrajectoryAccumulator(discount=0.9)
        first.append(
            observation=np.asarray([1.0, 2.0]),
            action=np.asarray([0.3]),
            reward=5.0,
            next_observation=np.asarray([2.0, 3.0]),
            terminal=False,
        )
        second = CalQLTrajectoryAccumulator(discount=0.9)
        second.load_state_dict(first.state_dict())
        self.assertEqual(second.pending_episode_length, 1)
        completed = second.append(
            observation=np.asarray([2.0, 3.0]),
            action=np.asarray([0.4]),
            reward=10.0,
            next_observation=np.asarray([3.0, 4.0]),
            terminal=True,
        )
        np.testing.assert_allclose(completed.batch["mc_returns"], [14.0, 10.0])

    def test_dynamic_replay_mixing_is_not_fixed_half(self):
        offline_count, online_count, ratio = dynamic_batch_counts(256, 1_000, 100)
        self.assertAlmostEqual(ratio, 1_000 / 1_100)
        self.assertEqual(offline_count, int(256 * ratio))
        self.assertEqual(online_count, 256 - offline_count)
        self.assertNotEqual((offline_count, online_count), (128, 128))
        self.assertEqual(dynamic_batch_counts(256, 1_000, 0), (256, 0, 1.0))

    def test_single_trajectory_return_helper(self):
        np.testing.assert_allclose(
            discounted_return_to_go([1.0, 2.0, 4.0], 0.5),
            [3.0, 4.0, 4.0],
        )


if __name__ == "__main__":
    unittest.main()
