from __future__ import annotations

import unittest

import torch

from robust_o2o.agents import build_agent
from robust_o2o.config import ALGORITHMS, ExperimentConfig
from robust_o2o.networks import OfficialRPEXGaussianPolicy


class AgentSmokeTest(unittest.TestCase):
    def _config(self, algorithm: str) -> ExperimentConfig:
        return ExperimentConfig(
            algorithm=algorithm,
            env_name="hopper-medium-replay-v2",
            corruption="clean",
            hidden_dim=16,
            hidden_layers=1,
            batch_size=8,
            num_critics=3,
            sac_num_critics=3,
            cql_n_actions=2,
            pqe_weight_batch_size=8,
            ro2o_sample_size=2,
            offline_steps=1,
            online_steps=1,
        )

    def _batch(self):
        batch = {
            "observations": torch.randn(8, 5),
            "actions": torch.tanh(torch.randn(8, 2)),
            "rewards": torch.randn(8),
            "next_observations": torch.randn(8, 5),
            "terminals": torch.zeros(8),
            "mc_returns": torch.randn(8),
            "_source": torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.float32),
        }
        return batch

    def test_all_agents_update_and_act(self):
        for algorithm in ALGORITHMS:
            with self.subTest(algorithm=algorithm):
                config = self._config(algorithm)
                agent = build_agent(config, 5, 2, 1.0, torch.device("cpu"))
                if algorithm == "pessimistic_q_ensemble":
                    metrics = agent.update(
                        member_batches=[self._batch() for _ in range(5)]
                    )
                else:
                    metrics = agent.update(self._batch())
                self.assertTrue(metrics)
                action = agent.select_action(torch.randn(5), evaluate=True)
                self.assertEqual(tuple(action.shape), (2,))
                agent.begin_online()
                if algorithm == "pessimistic_q_ensemble":
                    online_metrics = agent.update(
                        rl_batch=self._batch(),
                        density_offline_batch=self._batch(),
                        density_online_batch=self._batch(),
                        rl_batch_prioritized=True,
                    )
                else:
                    online_metrics = agent.update(self._batch())
                self.assertTrue(online_metrics)

    def test_rpex_uses_official_style_online_expansion_gaussian(self):
        agent = build_agent(
            self._config("rpex"), 5, 2, 1.0, torch.device("cpu")
        )
        self.assertIsInstance(agent.actor, OfficialRPEXGaussianPolicy)
        agent.begin_online()
        self.assertIsInstance(agent.actor, OfficialRPEXGaussianPolicy)
        states = torch.randn(4, 5)
        distribution = agent.actor.distribution(states)
        self.assertEqual(tuple(distribution.mean.shape), (4, 2))
        self.assertEqual(tuple(agent.actor.log_std.shape), (2,))
        self.assertTrue(torch.all(agent.actor.log_std == 0.0))


if __name__ == "__main__":
    unittest.main()
