from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np
import torch

from robust_o2o.agents import build_agent
from robust_o2o.agents.calql import calql_max_target_backup
from robust_o2o.cql import importance_sampled_cql
from robust_o2o.config import (
    LOCAL_PROTOCOL,
    ExperimentConfig,
    build_parser,
    config_from_args,
)
from robust_o2o.networks import EnsembleLayerNorm, VectorizedLinear
from robust_o2o.experiment import _validate_checkpoint
from robust_o2o.device import seed_everything


def tensor_batch(size: int = 8):
    rng = np.random.default_rng(4)
    return {
        "observations": torch.tensor(rng.normal(size=(size, 3)), dtype=torch.float32),
        "actions": torch.tensor(rng.uniform(-0.8, 0.8, size=(size, 2)), dtype=torch.float32),
        "next_observations": torch.tensor(rng.normal(size=(size, 3)), dtype=torch.float32),
        "rewards": torch.tensor(rng.normal(size=size), dtype=torch.float32),
        "terminals": torch.zeros(size),
        "mc_returns": torch.tensor(rng.normal(size=size), dtype=torch.float32),
        "mc_calibration_valid": torch.zeros(size),
    }


class ZeroPolicy:
    max_action = 1.0

    def __call__(self, states, need_log_prob=False):
        actions = states.new_zeros((len(states), 2))
        log_prob = states.new_zeros(len(states))
        std = states.new_ones((len(states), 2))
        return actions, log_prob if need_log_prob else None, actions, std


class ReferenceProfileTest(unittest.TestCase):
    def test_calql_max_target_backup_uses_clipped_double_q_argmax(self):
        q1 = torch.tensor([[1.0, 5.0, 2.0], [7.0, 2.0, 1.0]])
        q2 = torch.tensor([[2.0, 4.0, 3.0], [6.0, 3.0, 4.0]])
        log_probs = torch.tensor([[-1.0, -2.0, -3.0], [-4.0, -5.0, -6.0]])
        values, selected_log_probs = calql_max_target_backup(q1, q2, log_probs)
        self.assertTrue(torch.equal(values, torch.tensor([4.0, 6.0])))
        self.assertTrue(
            torch.equal(selected_log_probs, torch.tensor([-2.0, -4.0]))
        )

    def test_common_cql_head_shape_sign_and_reduction(self):
        states = torch.zeros(3, 4)
        actions = torch.zeros(3, 2)

        def evaluator(sample_states, sample_actions):
            count = sample_actions.shape[1]
            return sample_states.new_zeros((2, len(sample_states), count))

        result = importance_sampled_cql(
            policy=ZeroPolicy(),
            evaluators=(evaluator,),
            states=states,
            next_states=states,
            data_actions=actions,
            data_values=(torch.zeros(2, 3),),
            num_actions=2,
        )
        self.assertEqual(tuple(result.differences[0].shape), (2, 3))
        self.assertTrue((result.differences[0] > 0).all())
        self.assertAlmostEqual(result.loss.item(), np.log(12.0), places=5)

    def test_calql_reference_defaults_and_first_actor_update(self):
        config = ExperimentConfig(
            "cal_ql",
            "hopper-medium-replay-v2",
            hidden_dim=16,
            hidden_layers=2,
            cql_n_actions=2,
        )
        self.assertEqual(config.calql_bc_warmup_steps, 0)
        self.assertTrue(config.cql_max_target_backup)
        self.assertEqual(config.resolved_algorithm_profile, "calql_locomotion_port")
        agent = build_agent(config, 3, 2, 1.0, torch.device("cpu"))
        self.assertEqual(agent.actor_optimizer.param_groups[0]["lr"], 1e-4)
        self.assertEqual(agent.q1_optimizer.param_groups[0]["lr"], 3e-4)
        metrics = agent.update(tensor_batch())
        self.assertEqual(metrics["actor_update_mode_bc_warmup"], 0.0)

    def test_calql_legacy_keeps_bc100k(self):
        config = ExperimentConfig(
            "cal_ql", "hopper-medium-replay-v2", algorithm_profile="legacy_current"
        )
        self.assertEqual(config.calql_bc_warmup_steps, 100_000)
        self.assertFalse(config.cql_max_target_backup)
        self.assertEqual(config.resolved_algorithm_profile, "calql_legacy_bc100k")

    def test_oracle_mode_is_never_plain_calql_profile(self):
        config = ExperimentConfig(
            "cal_ql",
            "hopper-medium-replay-v2",
            calibration_mask_mode="oracle_exclude_corrupted",
        )
        self.assertIn("oracle", config.resolved_algorithm_profile)

    def test_wsrl_reference_architecture_and_schedule(self):
        config = ExperimentConfig(
            "wsrl",
            "hopper-medium-replay-v2",
            hidden_dim=16,
            hidden_layers=2,
            cql_n_actions=2,
        )
        self.assertEqual(config.sac_num_critics, 10)
        self.assertEqual(config.wsrl_target_critic_subsample_size, 2)
        self.assertEqual(config.wsrl_utd_ratio, 4)
        self.assertEqual(config.wsrl_total_sampled_batch_size if hasattr(config, "wsrl_total_sampled_batch_size") else config.wsrl_utd_ratio * config.wsrl_per_critic_batch_size, 1024)
        agent = build_agent(config, 3, 2, 1.0, torch.device("cpu"))
        actor_linears = [
            module
            for module in agent.actor.trunk.modules()
            if isinstance(module, torch.nn.Linear)
        ]
        self.assertEqual(len(actor_linears), 2)
        self.assertAlmostEqual(
            torch.linalg.matrix_norm(actor_linears[-1].weight, ord=2).item(),
            1e-2,
            places=6,
        )
        self.assertEqual(config.actor_learning_rate, 1e-4)
        self.assertEqual(config.critic_learning_rate, 3e-4)
        self.assertEqual(config.temperature_learning_rate, 1e-4)
        self.assertTrue(any(isinstance(module, torch.nn.LayerNorm) for module in agent.actor.modules()))
        self.assertTrue(any(isinstance(module, EnsembleLayerNorm) for module in agent.critic.modules()))
        torch.manual_seed(config.learner_seed)
        first_indices = agent._sample_target_critic_indices(10)
        torch.manual_seed(config.learner_seed)
        second_indices = agent._sample_target_critic_indices(10)
        self.assertTrue(torch.equal(first_indices, second_indices))
        fixed_q = torch.tensor(
            [[9.0, -1.0], [4.0, 8.0], [2.0, 7.0], [6.0, 3.0]]
        )
        fixed_indices = torch.tensor([2, 2])
        self.assertTrue(
            torch.equal(
                agent._wsrl_subsampled_min(fixed_q, fixed_indices),
                torch.tensor([2.0, 7.0]),
            )
        )
        cql_batch = tensor_batch()
        agent.zero_grad(set_to_none=True)
        with patch.object(
            agent,
            "_sample_target_critic_indices",
            return_value=torch.tensor([1, 1]),
        ):
            cql_penalty = agent._cql_penalty(
                cql_batch["observations"],
                cql_batch["next_observations"],
                cql_batch["actions"],
            )
        cql_penalty.backward()
        output_layer = [
            module
            for module in agent.critic.modules()
            if isinstance(module, VectorizedLinear)
        ][-1]
        per_head_gradient = output_layer.weight.grad.abs().sum(dim=(1, 2))
        self.assertGreater(per_head_gradient[1].item(), 0.0)
        self.assertEqual(torch.count_nonzero(per_head_gradient).item(), 1)
        agent.begin_online()
        batches = []
        for _ in range(config.wsrl_utd_ratio):
            batch = tensor_batch()
            batches.append(batch)
            agent.update(
                batch, update_actor_temperature=False, update_critic=True
            )
        combined = {
            key: torch.cat([batch[key] for batch in batches], dim=0)
            for key in batches[0]
        }
        agent.update(
            combined, update_actor_temperature=True, update_critic=False
        )
        self.assertEqual(agent.critic_updates, 4)
        self.assertEqual(agent.actor_updates, 1)
        self.assertEqual(agent.temperature_updates, 1)

    def test_role_seed_derivation_is_stable_and_distinct(self):
        first = ExperimentConfig("rpex", "hopper-medium-replay-v2", seed=7)
        second = ExperimentConfig("rpex", "hopper-medium-replay-v2", seed=7)
        values = (
            first.learner_seed,
            first.corruption_seed,
            first.replay_seed,
            first.train_env_seed,
            first.eval_seed,
        )
        self.assertEqual(values, (
            second.learner_seed,
            second.corruption_seed,
            second.replay_seed,
            second.train_env_seed,
            second.eval_seed,
        ))
        self.assertEqual(len(set(values)), len(values))

    def test_learner_initialization_ignores_corruption_seed(self):
        configs = [
            ExperimentConfig(
                "cal_ql",
                "hopper-medium-replay-v2",
                learner_seed=5,
                corruption_seed=seed,
                hidden_dim=16,
                hidden_layers=2,
            )
            for seed in (1, 99)
        ]
        agents = []
        for config in configs:
            seed_everything(config.learner_seed)
            agents.append(
                build_agent(config, 3, 2, 1.0, torch.device("cpu"))
            )
        for first, second in zip(agents[0].parameters(), agents[1].parameters()):
            self.assertTrue(torch.equal(first, second))

    def test_local_cli_requires_explicit_acknowledgement(self):
        parser = build_parser()
        args = parser.parse_args(
            [
                "--algorithm", "rpex",
                "--env-name", "hopper-medium-replay-v2",
                "--protocol", LOCAL_PROTOCOL,
            ]
        )
        with self.assertRaisesRegex(ValueError, "diagnostic-only"):
            config_from_args(args)
        args.allow_diagnostic_protocol = True
        config = config_from_args(args)
        self.assertEqual(config.protocol, LOCAL_PROTOCOL)

    def test_checkpoint_fingerprint_and_profile_mismatch_are_hard_errors(self):
        config = ExperimentConfig("cal_ql", "hopper-medium-replay-v2")
        config._environment_fingerprint = "current"
        config._environment_fingerprint_payload = {"dataset_sha256": "new"}
        payload = {
            "algorithm": "cal_ql",
            "algorithm_profile": config.implementation_profile,
            "implementation_profile": config.implementation_profile,
            "env_name": config.env_name,
            "protocol": config.protocol,
            "state_dim": 3,
            "action_dim": 2,
            "environment_fingerprint": "old",
            "environment_fingerprint_payload": {"dataset_sha256": "old"},
            "config": {},
        }
        with self.assertRaisesRegex(ValueError, "fingerprint mismatch"):
            _validate_checkpoint(payload, config, 3, 2)
        payload["environment_fingerprint"] = "current"
        payload["algorithm_profile"] = "legacy_current"
        payload["implementation_profile"] = "legacy_current"
        with self.assertRaisesRegex(ValueError, "implementation_profile"):
            _validate_checkpoint(payload, config, 3, 2)


if __name__ == "__main__":
    unittest.main()
