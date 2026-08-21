from __future__ import annotations

import copy
import math
import unittest
from types import SimpleNamespace

import torch
import torch.nn as nn

from robust_o2o.agents.pessimistic_q_ensemble import (
    PQE_CHECKPOINT_FORMAT,
    PessimisticQEnsembleAgent,
)
from robust_o2o.agents.sac_family import SACEnsembleAgent


def _config(**overrides):
    values = {
        "seed": 3,
        "pqe_ensemble_size": 5,
        "pqe_replay_mode": "balanced_density",
        "pqe_hidden_dim": 8,
        "pqe_hidden_layers": 2,
        "actor_learning_rate": 3e-4,
        "critic_learning_rate": 3e-4,
        "temperature_learning_rate": 3e-4,
        "learning_rate": 3e-4,
        "discount": 0.99,
        "target_update_rate": 0.005,
        "target_entropy": -2.0,
        "cql_alpha": 1.0,
        "cql_n_actions": 2,
        "cql_temperature": 1.0,
        "backup_entropy": True,
        "max_grad_norm": None,
        "pqe_priority_temperature": 5.0,
        "priority_floor": 1e-3,
        "pqe_priority_ceiling": 1e3,
        "pqe_init_online_fraction": 0.75,
        "pqe_first_epoch_multiplier": 5,
        "pqe_first_online_block_steps": 1_000,
        "pqe_online_buffer_size": 250_000,
        "pqe_weight_batch_size": 256,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _batch(seed: int, size: int = 6) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    return {
        "observations": torch.randn(size, 3, generator=generator),
        "actions": torch.empty(size, 2).uniform_(
            -0.8, 0.8, generator=generator
        ),
        "rewards": torch.randn(size, generator=generator),
        "next_observations": torch.randn(size, 3, generator=generator),
        "terminals": torch.zeros(size),
    }


def _agent(**overrides) -> PessimisticQEnsembleAgent:
    return PessimisticQEnsembleAgent(
        _config(**overrides),
        state_dim=3,
        action_dim=2,
        max_action=1.0,
        device=torch.device("cpu"),
    )


class _ConstantQ(nn.Module):
    def __init__(self, value: float) -> None:
        super().__init__()
        self.register_buffer("value", torch.tensor(float(value)))

    def forward(
        self, states: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        del actions
        return self.value.expand(len(states))


class PessimisticQEnsembleStructureTest(unittest.TestCase):
    def test_five_independent_actor_and_twin_critic_members(self) -> None:
        agent = _agent()

        self.assertEqual(agent.ensemble_size, 5)
        self.assertEqual(agent.member_seeds, (3, 7, 11, 15, 19))
        self.assertEqual(len(agent.actors), 5)
        self.assertEqual(len(agent.q1_members), 5)
        self.assertEqual(len(agent.q2_members), 5)
        self.assertEqual(len(agent.target_q1_members), 5)
        self.assertEqual(len(agent.target_q2_members), 5)
        self.assertNotIsInstance(agent, SACEnsembleAgent)
        agent.assert_independent_parameter_storage()

        actor_one_before = next(agent.actors[1].parameters()).detach().clone()
        q1_one_before = next(agent.q1_members[1].parameters()).detach().clone()
        with torch.no_grad():
            next(agent.actors[0].parameters()).add_(1.0)
            next(agent.q1_members[0].parameters()).add_(1.0)
        self.assertTrue(
            torch.equal(
                next(agent.actors[1].parameters()), actor_one_before
            )
        )
        self.assertTrue(
            torch.equal(next(agent.q1_members[1].parameters()), q1_one_before)
        )

    def test_non_main_ensemble_size_and_uniform_replay_fail_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "exactly five"):
            _agent(pqe_ensemble_size=4)
        with self.assertRaisesRegex(ValueError, "uniform replay"):
            _agent(pqe_replay_mode="uniform")

    def test_moment_matched_pre_tanh_policy_and_deterministic_action(self) -> None:
        agent = _agent()
        means = torch.tensor(
            [
                [-1.0, 0.5],
                [-0.5, 0.4],
                [0.0, 0.3],
                [0.5, 0.2],
                [1.0, 0.1],
            ]
        )
        stds = torch.tensor(
            [
                [0.4, 0.8],
                [0.5, 0.7],
                [0.6, 0.6],
                [0.7, 0.5],
                [0.8, 0.4],
            ]
        )
        with torch.no_grad():
            for index, actor in enumerate(agent.actors):
                for parameter in actor.parameters():
                    parameter.zero_()
                actor.mean.bias.copy_(means[index])
                actor.log_std.bias.copy_(stds[index].log())

        states = torch.zeros(4, 3)
        average_mean, average_std = agent.moment_parameters(states)
        expected_mean = means.mean(dim=0)
        expected_variance = (
            (stds.square() + means.square()).mean(dim=0)
            - expected_mean.square()
        )
        self.assertTrue(
            torch.allclose(average_mean, expected_mean.expand_as(average_mean))
        )
        self.assertTrue(
            torch.allclose(
                average_std,
                expected_variance.sqrt().expand_as(average_std),
                atol=1e-6,
            )
        )
        deterministic = agent.select_action(states[0], evaluate=True)
        self.assertTrue(torch.allclose(deterministic, expected_mean.tanh()))

    def test_actor_value_uses_mean_of_member_clipped_double_q(self) -> None:
        agent = _agent()
        q1_values = (1.0, 2.0, 3.0, 4.0, 5.0)
        q2_values = (0.0, 10.0, 3.0, 2.0, 6.0)
        agent.q1_members = nn.ModuleList(_ConstantQ(x) for x in q1_values)
        agent.q2_members = nn.ModuleList(_ConstantQ(x) for x in q2_values)

        values = agent.ensemble_clipped_q(
            torch.zeros(3, 3), torch.zeros(3, 2)
        )

        self.assertTrue(torch.allclose(values, torch.full((3,), 2.4)))


class PessimisticQEnsembleReplayObjectiveTest(unittest.TestCase):
    def test_density_ratio_objective_matches_public_equation(self) -> None:
        offline = torch.tensor([0.5, 1.0, 2.0])
        online = torch.tensor([0.25, 1.5, 3.0])
        actual = PessimisticQEnsembleAgent.density_ratio_objective_from_weights(
            offline, online
        )
        expected = (
            -torch.log(2.0 / (offline + 1.0) + 1e-10).mean()
            - torch.log(2.0 * online / (online + 1.0) + 1e-10).mean()
        )
        self.assertTrue(torch.allclose(actual, expected))

    def test_priority_temperature_normalization_and_clipping(self) -> None:
        weights = torch.tensor([0.0, 1.0, 1e30])
        offline_weights = torch.ones(8)
        priorities = PessimisticQEnsembleAgent.priority_values_from_weights(
            weights,
            offline_weights,
            temperature=5.0,
            floor=1e-3,
            ceiling=1e3,
        )

        self.assertAlmostEqual(priorities[0].item(), 1e-3, places=7)
        self.assertAlmostEqual(priorities[1].item(), 1.0, places=6)
        self.assertAlmostEqual(priorities[2].item(), 1e3, places=3)

    def test_source_initial_priority_and_first_block_multiplier(self) -> None:
        agent = _agent()

        self.assertAlmostEqual(
            agent.initial_online_priority(1_000_000),
            3_000.0,
        )
        self.assertEqual(agent.online_update_count_for_block(0, 1_000), 5_000)
        self.assertEqual(agent.online_update_count_for_block(1, 1_000), 1_000)
        metadata = agent.algorithm_metadata()
        self.assertEqual(metadata["init_online_fraction"], 0.75)
        self.assertEqual(metadata["first_epoch_multiplier"], 5)
        self.assertEqual(metadata["first_online_block_steps"], 1_000)

    def test_offline_members_use_independent_batches_and_plain_cql(self) -> None:
        agent = _agent()
        agent.bind_offline_artifact("cache-key", "a" * 64)
        agent.bind_offline_artifact("cache-key", "a" * 64)
        with self.assertRaisesRegex(ValueError, "cannot switch"):
            agent.bind_offline_artifact("other-key", "b" * 64)
        with self.assertRaisesRegex(ValueError, "single shared minibatch"):
            agent.update(_batch(1))

        metrics = agent.update(
            member_batches=[_batch(20 + index) for index in range(5)]
        )

        self.assertEqual(agent.offline_updates_per_member, [1, 1, 1, 1, 1])
        self.assertEqual(agent.total_offline_gradient_updates, 5)
        self.assertEqual(metrics["number_of_critic_updates"], 5.0)
        self.assertEqual(metrics["cql_loss_enabled"], 1.0)
        self.assertTrue(math.isfinite(metrics["cql_loss"]))

    def test_online_sac_requires_and_returns_priority_replay_updates(self) -> None:
        agent = _agent()
        agent.begin_online()
        rl_batch = _batch(41)
        rl_batch["_source"] = torch.tensor([0, 0, 0, 1, 1, 1])
        offline_batch = _batch(42)
        online_batch = _batch(43)
        with self.assertRaisesRegex(ValueError, "proportional priority"):
            agent.update(
                rl_batch=rl_batch,
                density_offline_batch=offline_batch,
                density_online_batch=online_batch,
            )

        actor_before = [
            parameter.detach().clone()
            for actor in agent.actors
            for parameter in actor.parameters()
        ]
        metrics = agent.update(
            rl_batch=rl_batch,
            density_offline_batch=offline_batch,
            density_online_batch=online_batch,
            rl_batch_prioritized=True,
        )
        actor_after = [
            parameter.detach()
            for actor in agent.actors
            for parameter in actor.parameters()
        ]

        self.assertTrue(any(not torch.equal(x, y) for x, y in zip(actor_before, actor_after)))
        self.assertTrue(math.isfinite(metrics["critic_loss"]))
        self.assertTrue(math.isfinite(metrics["density_loss"]))
        self.assertEqual(metrics["cql_loss_enabled"], 0.0)
        self.assertEqual(metrics["rl_offline_count"], 3.0)
        self.assertEqual(metrics["rl_online_count"], 3.0)
        priorities = agent.consume_priority_values()
        self.assertIsNotNone(priorities)
        self.assertEqual(len(priorities), len(rl_batch["rewards"]))
        self.assertIsNone(agent.consume_priority_values())

    def test_online_boundary_resets_cql_optimizers_and_hard_copies_targets(self) -> None:
        agent = _agent()
        agent.update(
            member_batches=[_batch(100 + index) for index in range(5)]
        )
        actor_weights = [
            parameter.detach().clone()
            for actor in agent.actors
            for parameter in actor.parameters()
        ]
        old_actor_optimizers = list(agent.actor_optimizers)
        self.assertTrue(all(optimizer.state for optimizer in old_actor_optimizers))

        agent.begin_online()

        self.assertTrue(agent.online_phase)
        self.assertTrue(all(not optimizer.state for optimizer in agent.actor_optimizers))
        self.assertTrue(
            all(
                old is not new
                for old, new in zip(old_actor_optimizers, agent.actor_optimizers)
            )
        )
        self.assertTrue(
            all(
                torch.equal(before, after)
                for before, after in zip(
                    actor_weights,
                    (
                        parameter
                        for actor in agent.actors
                        for parameter in actor.parameters()
                    ),
                )
            )
        )
        for q1, target_q1, q2, target_q2 in zip(
            agent.q1_members,
            agent.target_q1_members,
            agent.q2_members,
            agent.target_q2_members,
        ):
            self.assertTrue(
                all(
                    torch.equal(source, target)
                    for source, target in zip(
                        q1.parameters(), target_q1.parameters()
                    )
                )
            )
            self.assertTrue(
                all(
                    torch.equal(source, target)
                    for source, target in zip(
                        q2.parameters(), target_q2.parameters()
                    )
                )
            )


class PessimisticQEnsembleCheckpointTest(unittest.TestCase):
    def test_five_unique_member_checkpoints_round_trip(self) -> None:
        source = _agent()
        with torch.no_grad():
            next(source.actors[0].parameters()).add_(0.25)
        checkpoints = source.member_checkpoint_states()
        self.assertTrue(
            all(
                checkpoint["format"] == PQE_CHECKPOINT_FORMAT
                for checkpoint in checkpoints
            )
        )

        restored = _agent()
        restored.load_member_checkpoint_states(checkpoints)

        for source_actor, restored_actor in zip(
            source.actors, restored.actors
        ):
            for expected, actual in zip(
                source_actor.parameters(), restored_actor.parameters()
            ):
                self.assertTrue(torch.equal(expected, actual))
        self.assertEqual(len(set(restored.member_checkpoint_hashes)), 5)

    def test_duplicate_member_checkpoint_is_rejected(self) -> None:
        agent = _agent()
        checkpoints = agent.member_checkpoint_states()
        duplicates = [copy.deepcopy(checkpoints[0]) for _ in range(5)]
        with self.assertRaisesRegex(ValueError, "duplicate.*content"):
            _agent().load_member_checkpoint_states(duplicates)
        with self.assertRaisesRegex(ValueError, "duplicate.*hashes"):
            _agent().load_member_checkpoint_states(
                checkpoints,
                checkpoint_hashes=["a" * 64] * 5,
            )

    def test_member_seed_and_structure_are_strict(self) -> None:
        checkpoints = _agent().member_checkpoint_states()
        wrong_seed = copy.deepcopy(checkpoints)
        wrong_seed[2]["member_seed"] = 999
        with self.assertRaisesRegex(ValueError, "wrong seed"):
            _agent().load_member_checkpoint_states(wrong_seed)

        missing_q = copy.deepcopy(checkpoints)
        del missing_q[4]["q2"]
        with self.assertRaisesRegex(ValueError, "missing q2"):
            _agent().load_member_checkpoint_states(missing_q)

    def test_full_training_checkpoint_preserves_members_and_counters(self) -> None:
        source = _agent()
        source.update(
            member_batches=[_batch(60 + index) for index in range(5)]
        )
        source.record_member_checkpoint_hash(0, "0" * 64)
        source.record_member_checkpoint_hash(1, "1" * 64)
        source.bind_offline_artifact("cache", "f" * 64)
        state = copy.deepcopy(source.checkpoint_state())

        restored = _agent()
        restored.load_checkpoint_state(state)

        self.assertEqual(restored.offline_updates_per_member, [1] * 5)
        self.assertEqual(restored.total_updates, 5)
        self.assertEqual(restored.actor_updates, 5)
        self.assertEqual(restored.member_checkpoint_hashes[:2], ["0" * 64, "1" * 64])
        self.assertEqual(
            restored.offline_artifact_identity,
            ("cache", "f" * 64),
        )


if __name__ == "__main__":
    unittest.main()
