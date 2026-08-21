from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np
import torch

from robust_o2o.agents import build_agent
from robust_o2o.config import ExperimentConfig
from robust_o2o.experiment import (
    _run_wsrl_high_utd_update,
    _split_wsrl_high_utd_batch,
    _wsrl_first_update_env_step,
    _wsrl_runtime_metadata,
)
from robust_o2o.replay import ReplayBuffer


def _tensor_batch(size: int = 4) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(17)
    return {
        "observations": torch.randn(size, 3, generator=generator),
        "actions": torch.empty(size, 2).uniform_(
            -0.8, 0.8, generator=generator
        ),
        "rewards": torch.randn(size, generator=generator),
        "next_observations": torch.randn(size, 3, generator=generator),
        "terminals": torch.zeros(size),
    }


class _RecordingAgent:
    def __init__(self) -> None:
        self.calls: list[tuple[str, list[int]]] = []
        self.actor_updates = 0
        self.critic_updates = 0
        self.temperature_updates = 0

    def update(
        self,
        batch: dict[str, torch.Tensor],
        *,
        update_actor_temperature: bool,
        update_critic: bool,
    ) -> dict[str, float]:
        mode = "critic" if update_critic else "actor_temperature"
        self.calls.append((mode, batch["_indices"].tolist()))
        self.critic_updates += int(update_critic)
        self.actor_updates += int(update_actor_temperature)
        self.temperature_updates += int(update_actor_temperature)
        return {
            "critic_loss": float(self.critic_updates),
            "actor_loss": float(self.actor_updates),
            "number_of_actor_updates": float(update_actor_temperature),
            "number_of_critic_updates": float(update_critic),
            "number_of_temperature_updates": float(
                update_actor_temperature
            ),
            "total_actor_updates": float(self.actor_updates),
            "total_critic_updates": float(self.critic_updates),
            "total_temperature_updates": float(
                self.temperature_updates
            ),
        }


class WSRLProtocolTest(unittest.TestCase):
    def test_locomotion_defaults_match_pinned_launchers(self) -> None:
        config = ExperimentConfig("wsrl", "hopper-medium-replay-v2")

        self.assertEqual(config.batch_size, 256)
        self.assertEqual(config.cql_alpha, 5.0)
        self.assertEqual(config.cql_n_actions, 10)
        self.assertTrue(config.cql_max_target_backup)
        self.assertFalse(config.backup_entropy)
        self.assertEqual(config.effective_offline_ratio, 0.0)
        self.assertEqual(config.initial_collection_steps, 5_000)
        self.assertEqual(config.warmup_steps, 5_000)
        self.assertEqual(config.sac_num_critics, 10)
        self.assertEqual(config.wsrl_target_critic_subsample_size, 2)
        self.assertEqual(config.wsrl_utd_ratio, 4)
        self.assertEqual(config.wsrl_per_critic_batch_size, 256)
        self.assertEqual(
            config.wsrl_utd_ratio * config.wsrl_per_critic_batch_size,
            1_024,
        )

    def test_high_utd_batch_is_split_contiguously(self) -> None:
        full_batch = {
            "_indices": torch.arange(8),
            "observations": torch.arange(24).reshape(8, 3),
        }

        chunks = _split_wsrl_high_utd_batch(full_batch, 4)

        self.assertEqual(
            [chunk["_indices"].tolist() for chunk in chunks],
            [[0, 1], [2, 3], [4, 5], [6, 7]],
        )
        with self.assertRaisesRegex(ValueError, "divisible"):
            _split_wsrl_high_utd_batch({"x": torch.arange(7)}, 4)
        with self.assertRaisesRegex(ValueError, "shared leading"):
            _split_wsrl_high_utd_batch(
                {"x": torch.arange(8), "y": torch.arange(7)}, 4
            )

    def test_high_utd_samples_once_and_reuses_full_actor_batch(self) -> None:
        config = ExperimentConfig(
            "wsrl",
            "hopper-medium-replay-v2",
            wsrl_utd_ratio=4,
            wsrl_per_critic_batch_size=2,
        )
        agent = _RecordingAgent()
        sampled = {"_indices": torch.arange(8)}
        offline = object()
        replay = object()
        device = torch.device("cpu")

        with patch(
            "robust_o2o.experiment.mixed_batch", return_value=sampled
        ) as sample_batch:
            metrics = _run_wsrl_high_utd_update(
                agent, offline, replay, config, device
            )

        sample_batch.assert_called_once_with(
            offline, replay, 8, 0.0, device, online_replace=True
        )
        self.assertEqual(
            agent.calls,
            [
                ("critic", [0, 1]),
                ("critic", [2, 3]),
                ("critic", [4, 5]),
                ("critic", [6, 7]),
                ("actor_temperature", list(range(8))),
            ],
        )
        self.assertEqual(metrics["number_of_critic_updates"], 4.0)
        self.assertEqual(metrics["number_of_actor_updates"], 1.0)
        self.assertEqual(metrics["number_of_temperature_updates"], 1.0)
        self.assertEqual(agent.critic_updates, 4)
        self.assertEqual(agent.actor_updates, 1)
        self.assertEqual(agent.temperature_updates, 1)

    def test_wsrl_online_replay_samples_with_replacement(self) -> None:
        replay = ReplayBuffer(3, 2, 16, seed=5)
        for index in range(3):
            replay.add(
                np.full(3, index, dtype=np.float32),
                np.full(2, index, dtype=np.float32),
                float(index),
                np.full(3, index + 1, dtype=np.float32),
                0.0,
            )

        with self.assertRaisesRegex(ValueError, "need 8"):
            replay.sample(8, torch.device("cpu"))
        batch = replay.sample(8, torch.device("cpu"), replace=True)
        self.assertEqual(len(batch["rewards"]), 8)
        self.assertLessEqual(len(torch.unique(batch["_indices"])), 3)

    def test_high_utd_rejects_offline_replay_retention(self) -> None:
        config = ExperimentConfig(
            "wsrl",
            "hopper-medium-replay-v2",
            offline_ratio=0.25,
            wsrl_per_critic_batch_size=2,
        )
        with self.assertRaisesRegex(ValueError, "offline data"):
            _run_wsrl_high_utd_update(
                _RecordingAgent(),
                object(),
                object(),
                config,
                torch.device("cpu"),
            )

    def test_warmup_transition_matches_pinned_zero_based_loop(self) -> None:
        self.assertEqual(
            _wsrl_first_update_env_step(
                warmup_steps=5_000, total_batch_size=1_024
            ),
            5_002,
        )

    def test_completion_metadata_uses_measured_optimizer_counts(self) -> None:
        config = ExperimentConfig("wsrl", "hopper-medium-replay-v2")
        agent = _RecordingAgent()
        agent.actor_updates = 13
        agent.critic_updates = 49
        agent.temperature_updates = 13

        metadata = _wsrl_runtime_metadata(
            config,
            agent,
            online_initial_update_counts={
                "actor": 3,
                "critic": 9,
                "temperature": 3,
            },
        )

        self.assertEqual(metadata["wsrl_first_update_env_step"], 5_002)
        self.assertEqual(metadata["wsrl_total_sampled_batch_size"], 1_024)
        self.assertEqual(metadata["wsrl_online_actor_updates"], 10)
        self.assertEqual(metadata["wsrl_online_critic_updates"], 40)
        self.assertEqual(metadata["wsrl_online_temperature_updates"], 10)
        self.assertFalse(metadata["wsrl_online_cql_enabled"])

    def test_cql_is_offline_only_and_update_counts_are_explicit(self) -> None:
        torch.manual_seed(3)
        config = ExperimentConfig(
            "wsrl",
            "hopper-medium-replay-v2",
            hidden_dim=16,
            cql_n_actions=2,
        )
        agent = build_agent(config, 3, 2, 1.0, torch.device("cpu"))
        batch = _tensor_batch()

        actor_before = {
            key: value.detach().clone()
            for key, value in agent.actor.state_dict().items()
        }
        original_cql_penalty = agent._cql_penalty
        cql_saw_pre_update_actor: list[bool] = []

        def checked_cql_penalty(*args):
            cql_saw_pre_update_actor.append(
                all(
                    torch.equal(agent.actor.state_dict()[key], value)
                    for key, value in actor_before.items()
                )
            )
            return original_cql_penalty(*args)

        with patch.object(
            agent, "_cql_penalty", side_effect=checked_cql_penalty
        ):
            offline_metrics = agent.update(batch)
        self.assertEqual(cql_saw_pre_update_actor, [True])
        self.assertEqual(offline_metrics["cql_loss_enabled"], 1.0)
        self.assertEqual(offline_metrics["wsrl_online_cql_disabled"], 0.0)
        self.assertTrue(np.isfinite(offline_metrics["cql_penalty"]))

        agent.begin_online()
        critic_metrics = agent.update(
            batch,
            update_actor_temperature=False,
            update_critic=True,
        )
        self.assertEqual(critic_metrics["cql_loss_enabled"], 0.0)
        self.assertEqual(critic_metrics["wsrl_online_cql_disabled"], 1.0)
        self.assertEqual(critic_metrics["number_of_actor_updates"], 0.0)
        self.assertEqual(critic_metrics["number_of_critic_updates"], 1.0)

        actor_metrics = agent.update(
            batch,
            update_actor_temperature=True,
            update_critic=False,
        )
        self.assertEqual(actor_metrics["number_of_actor_updates"], 1.0)
        self.assertEqual(actor_metrics["number_of_critic_updates"], 0.0)
        self.assertEqual(actor_metrics["number_of_temperature_updates"], 1.0)
        self.assertEqual(agent.critic_updates, 2)
        self.assertEqual(agent.actor_updates, 2)
        self.assertEqual(agent.temperature_updates, 2)


if __name__ == "__main__":
    unittest.main()
