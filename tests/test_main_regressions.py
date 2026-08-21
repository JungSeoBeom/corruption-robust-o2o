from __future__ import annotations

import unittest

import torch

from robust_o2o.agents import build_agent
from robust_o2o.config import ExperimentConfig


def _batch(size: int = 8) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(913)
    return {
        "observations": torch.randn(size, 3, generator=generator),
        "actions": torch.empty(size, 2).uniform_(
            -0.75, 0.75, generator=generator
        ),
        "rewards": torch.randn(size, generator=generator),
        "next_observations": torch.randn(size, 3, generator=generator),
        "terminals": torch.zeros(size),
    }


def _research_config(algorithm: str, **overrides: object) -> ExperimentConfig:
    values: dict[str, object] = {
        "algorithm": algorithm,
        "env_name": "hopper-medium-replay-v2",
        "suite_profile": "research_benchmark",
        "run_purpose": "research_benchmark",
        "hidden_dim": 8,
        "hidden_layers": 1,
        "batch_size": 8,
        "offline_steps": 1,
        "online_steps": 1,
    }
    values.update(overrides)
    return ExperimentConfig(**values)


class MainBaselineRegressionTest(unittest.TestCase):
    def test_iql_family_online_phase_uses_fresh_optimizers(self) -> None:
        for algorithm in ("rpex", "riql_naive", "riql_pex"):
            with self.subTest(algorithm=algorithm):
                torch.manual_seed(31)
                config = (
                    _research_config(algorithm)
                    if algorithm != "riql_pex"
                    else ExperimentConfig(
                        algorithm=algorithm,
                        env_name="hopper-medium-replay-v2",
                        hidden_dim=8,
                        hidden_layers=1,
                        batch_size=8,
                        offline_steps=1,
                        online_steps=1,
                    )
                )
                agent = build_agent(
                    config, state_dim=3, action_dim=2, max_action=1.0,
                    device=torch.device("cpu")
                )
                agent.update(_batch())
                optimizer_ids = (
                    id(agent.q_optimizer),
                    id(agent.value_optimizer),
                    id(agent.actor_optimizer),
                )
                self.assertTrue(agent.q_optimizer.state)
                self.assertTrue(agent.value_optimizer.state)
                self.assertTrue(agent.actor_optimizer.state)

                agent.begin_online()

                self.assertNotEqual(id(agent.q_optimizer), optimizer_ids[0])
                self.assertNotEqual(id(agent.value_optimizer), optimizer_ids[1])
                self.assertNotEqual(id(agent.actor_optimizer), optimizer_ids[2])
                self.assertFalse(agent.q_optimizer.state)
                self.assertFalse(agent.value_optimizer.state)
                self.assertFalse(agent.actor_optimizer.state)
                self.assertIsNone(agent.actor_scheduler)
                self.assertGreater(
                    agent.actor_optimizer.param_groups[0]["lr"], 0.0
                )
                self.assertEqual(
                    agent.actor_optimizer.param_groups[0]["lr"],
                    config.actor_learning_rate,
                )

    def test_riql_naive_keeps_weights_and_first_online_update_moves_actor(self) -> None:
        torch.manual_seed(47)
        config = _research_config(
            "riql_naive", actor_learning_rate=3e-4
        )
        agent = build_agent(
            config, state_dim=3, action_dim=2, max_action=1.0,
            device=torch.device("cpu")
        )
        agent.update(_batch())
        self.assertAlmostEqual(
            agent.actor_optimizer.param_groups[0]["lr"], 0.0
        )
        offline_weights = {
            name: parameter.detach().clone()
            for name, parameter in agent.actor.named_parameters()
        }

        agent.begin_online()

        for name, parameter in agent.actor.named_parameters():
            torch.testing.assert_close(parameter, offline_weights[name])
        self.assertGreater(agent.actor_optimizer.param_groups[0]["lr"], 0.0)

        before_online_update = {
            name: parameter.detach().clone()
            for name, parameter in agent.actor.named_parameters()
        }
        metrics = agent.update(_batch())
        actor_delta = sum(
            (parameter.detach() - before_online_update[name]).abs().sum().item()
            for name, parameter in agent.actor.named_parameters()
        )

        self.assertGreater(actor_delta, 0.0)
        self.assertGreater(metrics["gradient_norm_actor"], 0.0)
        self.assertEqual(metrics["number_of_actor_updates"], 1.0)

    def test_wsrl_research_protocol_invariants_remain_source_aligned(self) -> None:
        config = _research_config("wsrl", offline_steps=2, online_steps=2)

        self.assertEqual(config.effective_offline_ratio, 0.0)
        self.assertEqual(config.initial_collection_steps, 5_000)
        self.assertEqual(config.warmup_steps, 5_000)
        self.assertEqual(config.sac_num_critics, 10)
        self.assertEqual(config.wsrl_target_critic_subsample_size, 2)
        self.assertEqual(config.wsrl_utd_ratio, 4)
        self.assertEqual(config.target_entropy, -3.0)

        agent = build_agent(
            config, state_dim=3, action_dim=2, max_action=1.0,
            device=torch.device("cpu")
        )
        agent.begin_online()
        metrics = agent.update(
            _batch(), update_actor_temperature=False, update_critic=True
        )
        self.assertEqual(metrics["cql_loss_enabled"], 0.0)
        self.assertEqual(metrics["wsrl_online_cql_disabled"], 1.0)


if __name__ == "__main__":
    unittest.main()
