from __future__ import annotations

import copy
import csv
import json
import math
import multiprocessing
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch.distributions import Independent, Normal, TransformedDistribution

from robust_o2o.config import ExperimentConfig
from robust_o2o.corruption import (
    AttackOracle,
    EDACActor,
    EDACCritic,
    corrupt_offline_dataset,
)
from robust_o2o.fidelity import canonical_json_sha256
from robust_o2o.agents import build_agent
from robust_o2o.agents.iql_family import robust_huber
from robust_o2o.logging_utils import RunLogger
from robust_o2o.manifest import aggregation_signature, build_experiment_manifest
from robust_o2o.networks import OfficialRPEXGaussianPolicy
from robust_o2o.replay import ReplayBuffer


def _small_corruption_dataset():
    return {
        "observations": np.arange(384, dtype=np.float32).reshape(128, 3),
        "actions": np.arange(256, dtype=np.float32).reshape(128, 2),
        "rewards": np.arange(128, dtype=np.float32),
        "next_observations": np.arange(384, dtype=np.float32).reshape(128, 3),
        "terminals": np.zeros(128, dtype=np.float32),
    }


def _concurrent_cache_worker(cache_root: str, result_queue) -> None:
    config = ExperimentConfig(
        "riql_naive",
        "hopper-medium-replay-v2",
        corruption="random",
        corruption_target="rewards",
        corruption_seed=31,
    )
    result, stats = corrupt_offline_dataset(
        _small_corruption_dataset(), config, None, Path(cache_root)
    )
    result_queue.put((result["rewards"].tolist(), stats["cache_hit"]))


def _assert_nested_equal(test: unittest.TestCase, first, second) -> None:
    if isinstance(first, torch.Tensor):
        test.assertTrue(torch.equal(first, second))
    elif isinstance(first, np.ndarray):
        test.assertTrue(np.array_equal(first, second))
    elif isinstance(first, dict):
        test.assertEqual(first.keys(), second.keys())
        for key in first:
            _assert_nested_equal(test, first[key], second[key])
    elif isinstance(first, (list, tuple)):
        test.assertEqual(len(first), len(second))
        for left, right in zip(first, second):
            _assert_nested_equal(test, left, right)
    else:
        test.assertEqual(first, second)


class FidelityProfileTest(unittest.TestCase):
    def test_generic_reference_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "generic.*reference"):
            ExperimentConfig(
                "rpex",
                "hopper-medium-replay-v2",
                algorithm_profile="reference",
            )

    def test_riql_table_literal_rows(self):
        cases = {
            "observations": (0.1, 0.25, 3),
            "actions": (0.1, 0.25, 5),
            "rewards": (1.0, 0.25, 3),
            "dynamics": (1.0, 0.5, 5),
        }
        for target, expected in cases.items():
            config = ExperimentConfig(
                "rpex",
                "hopper-medium-replay-v2",
                corruption="random",
                corruption_target=target,
                suite_profile="method_fidelity",
            )
            self.assertEqual(
                (config.riql_sigma, config.riql_quantile, config.num_critics),
                expected,
            )
            self.assertFalse(config.riql_config_extension)
            self.assertEqual(config.offline_steps, 2_000_001)
            self.assertEqual(config.online_steps, 1_000_001)

    def test_riql_fixed_tensor_golden_and_align_iql_selection(self):
        q_ensemble = torch.tensor([[1.0, 4.0], [3.0, 2.0], [5.0, 0.0]])
        q_quantile = torch.quantile(q_ensemble, 0.25, dim=0)
        next_value = torch.tensor([0.5, 2.0])
        rewards = torch.tensor([1.0, -3.0])
        terminals = torch.tensor([0.0, 1.0])
        target_q = (
            rewards + (1.0 - terminals) * 0.99 * next_value
        ).clamp(-100.0, 1_000.0)
        predicted = torch.tensor(
            [[1.0, -2.0], [2.0, -4.0], [1.5, -3.5]]
        )
        critic_loss = robust_huber(
            target_q.unsqueeze(0) - predicted, 3.0
        ).mean()
        advantage = q_quantile - torch.tensor([1.25, 0.5])
        awr_weight = torch.exp(3.0 * advantage).clamp(max=100.0)
        actor_loss = (awr_weight * torch.tensor([1.2, 0.7])).mean()

        self.assertTrue(torch.equal(q_quantile, torch.tensor([2.0, 1.0])))
        self.assertTrue(torch.allclose(target_q, torch.tensor([1.495, -3.0])))
        self.assertAlmostEqual(critic_loss.item(), 0.5370557904, places=7)
        self.assertTrue(torch.equal(advantage, torch.tensor([0.75, 0.5])))
        self.assertTrue(
            torch.allclose(awr_weight, torch.tensor([9.4877357483, 4.4816889763]))
        )
        self.assertAlmostEqual(actor_loss.item(), 7.2612328529, places=6)

        observation_config = ExperimentConfig(
            "rpex",
            "hopper-medium-replay-v2",
            corruption="random",
            corruption_target="observations",
            suite_profile="method_fidelity",
        )
        self.assertEqual(observation_config.policy_extraction, "align_iql")

        scheduler_config = ExperimentConfig(
            "riql_naive",
            "hopper-medium-replay-v2",
            hidden_dim=4,
            hidden_layers=1,
            offline_steps=4,
            actor_learning_rate=4e-4,
            critic_learning_rate=3e-4,
        )
        scheduler_agent = build_agent(
            scheduler_config, 3, 2, 1.0, torch.device("cpu")
        )
        batch = {
            "observations": torch.zeros(2, 3),
            "actions": torch.zeros(2, 2),
            "rewards": torch.zeros(2),
            "next_observations": torch.ones(2, 3),
            "terminals": torch.zeros(2),
        }
        scheduler_agent.update(batch)
        expected_lr = 4e-4 * (1.0 + math.cos(math.pi / 4.0)) / 2.0
        self.assertAlmostEqual(
            scheduler_agent.actor_optimizer.param_groups[0]["lr"], expected_lr
        )
        self.assertEqual(scheduler_agent.q_optimizer.param_groups[0]["lr"], 3e-4)

    def test_missing_official_row_and_pqe_method_suite_fail_fast(self):
        with self.assertRaisesRegex(ValueError, "no official row"):
            ExperimentConfig(
                "rpex",
                "hopper-medium-replay-v2",
                suite_profile="method_fidelity",
            )

    def test_official_adversarial_optimizer_schedule_is_separate(self):
        config = ExperimentConfig(
            "rpex",
            "hopper-medium-replay-v2",
            corruption="adversarial",
            corruption_target="observations",
            suite_profile="method_fidelity",
            implementation_profile="official_code_reference",
        )
        self.assertEqual(config.adversarial_attack_profile, "rpex_official_adam")
        self.assertEqual(config.offline_attack_steps, 100)
        self.assertEqual(config.attack_step_size, 0.01)
        self.assertEqual(config.online_attack_steps, 2)
        self.assertEqual(config.online_attack_step_size, 0.1)

    def test_attack_oracle_rng_is_dedicated_and_seeded(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "oracle.pt"
            torch.save(
                {
                    "actor": EDACActor(3, 2, 1.0).state_dict(),
                    "critic": EDACCritic(3, 2).state_dict(),
                },
                checkpoint,
            )
            oracles = [
                AttackOracle(
                    3,
                    2,
                    1.0,
                    checkpoint,
                    torch.device("cpu"),
                    seed=seed,
                    implementation_profile="rpex_official_adam",
                )
                for seed in (13, 13, 14)
            ]
            original = np.zeros((2, 3), dtype=np.float32)
            std = np.ones((1, 3), dtype=np.float32)
            observations = np.zeros((2, 3), dtype=np.float32)
            actions = np.zeros((2, 2), dtype=np.float32)
            outputs = [
                oracle.attack(
                    original,
                    std,
                    observations,
                    actions,
                    "observations",
                    1.0,
                    0,
                    0.01,
                )
                for oracle in oracles
            ]
            self.assertTrue(np.array_equal(outputs[0], outputs[1]))
            self.assertFalse(np.array_equal(outputs[0], outputs[2]))

            torch.manual_seed(71)
            expected = torch.rand(5)
            torch.manual_seed(71)
            oracles[0].attack(
                original,
                std,
                observations,
                actions,
                "observations",
                1.0,
                0,
                0.01,
            )
            actual = torch.rand(5)
            self.assertTrue(torch.equal(expected, actual))
            for oracle in oracles:
                oracle.close()
        with self.assertRaisesRegex(ValueError, "pqe_shared_actor_approx"):
            ExperimentConfig(
                "pessimistic_q_ensemble",
                "hopper-medium-replay-v2",
                suite_profile="method_fidelity",
            )

    def test_official_unsquashed_policy(self):
        torch.manual_seed(3)
        policy = OfficialRPEXGaussianPolicy(3, 2, hidden_dim=8, hidden_layers=2)
        policy_linears = [
            module for module in policy.modules() if isinstance(module, torch.nn.Linear)
        ]
        self.assertEqual(len(policy_linears), 3)  # two hidden + one mean head
        states = torch.randn(7, 3)
        distribution = policy.distribution(states)
        self.assertIsInstance(distribution, Independent)
        self.assertIsInstance(distribution.base_dist, Normal)
        self.assertNotIsInstance(distribution, TransformedDistribution)
        self.assertTrue(
            torch.equal(distribution.stddev[0], distribution.stddev[-1])
        )
        manual = distribution.base_dist.log_prob(distribution.mean).sum(-1)
        self.assertTrue(torch.allclose(policy.log_prob(states, distribution.mean), manual))
        with torch.no_grad():
            policy.log_std.fill_(2.0)
        samples = policy.distribution(states.repeat(256, 1)).sample()
        self.assertTrue((samples.abs() > 1.0).any())

    def test_replay_resume_round_trip(self):
        replay = ReplayBuffer(2, 1, 8, seed=17)
        for index in range(5):
            replay.add(
                np.array([index, -index], dtype=np.float32),
                np.array([index / 10], dtype=np.float32),
                float(index),
                np.array([index + 1, -index - 1], dtype=np.float32),
                0.0,
            )
        state = replay.state_dict()
        restored = ReplayBuffer(2, 1, 8, seed=999)
        restored.load_state_dict(state)
        first = replay.sample(3, torch.device("cpu"))
        second = restored.sample(3, torch.device("cpu"))
        for key in first:
            self.assertTrue(torch.equal(first[key], second[key]), key)

    def test_cache_is_checksummed_and_validated(self):
        dataset = {
            "observations": np.arange(24, dtype=np.float32).reshape(8, 3),
            "actions": np.arange(16, dtype=np.float32).reshape(8, 2),
            "rewards": np.arange(8, dtype=np.float32),
            "next_observations": np.arange(24, dtype=np.float32).reshape(8, 3),
            "terminals": np.zeros(8, dtype=np.float32),
        }
        config = ExperimentConfig(
            "riql_naive",
            "hopper-medium-replay-v2",
            corruption="random",
            corruption_target="rewards",
            corruption_seed=23,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first, first_stats = corrupt_offline_dataset(dataset, config, None, root)
            second, second_stats = corrupt_offline_dataset(dataset, config, None, root)
            self.assertFalse(first_stats["cache_hit"])
            self.assertTrue(second_stats["cache_hit"])
            self.assertTrue(np.array_equal(first["rewards"], second["rewards"]))
            self.assertEqual(
                first_stats["corruption_value_sha256"],
                second_stats["corruption_value_sha256"],
            )
            caches = list(root.rglob("*.npz"))
            self.assertEqual(len(caches), 1)
            self.assertTrue(caches[0].with_suffix(".npz.sha256").exists())

    def test_two_processes_generate_one_atomic_cache(self):
        context = multiprocessing.get_context("fork")
        with tempfile.TemporaryDirectory() as directory:
            result_queue = context.Queue()
            processes = [
                context.Process(
                    target=_concurrent_cache_worker,
                    args=(directory, result_queue),
                )
                for _ in range(2)
            ]
            for process in processes:
                process.start()
            for process in processes:
                process.join(15)
                self.assertFalse(process.is_alive())
                self.assertEqual(process.exitcode, 0)
            results = [result_queue.get(timeout=2) for _ in processes]
            self.assertEqual(results[0][0], results[1][0])
            self.assertEqual(sorted(item[1] for item in results), [False, True])
            self.assertEqual(len(list(Path(directory).rglob("*.npz"))), 1)

    def test_random_corruption_is_paired_across_algorithms(self):
        dataset = _small_corruption_dataset()
        with tempfile.TemporaryDirectory() as directory:
            outputs = []
            stats = []
            for algorithm in ("riql_naive", "wsrl"):
                config = ExperimentConfig(
                    algorithm,
                    "hopper-medium-replay-v2",
                    corruption="random",
                    corruption_target="rewards",
                    corruption_seed=47,
                )
                result, metadata = corrupt_offline_dataset(
                    dataset, config, None, Path(directory)
                )
                outputs.append(result["rewards"])
                stats.append(metadata)
            self.assertTrue(np.array_equal(outputs[0], outputs[1]))
            self.assertEqual(
                stats[0]["selected_transition_indices_sha256"],
                stats[1]["selected_transition_indices_sha256"],
            )
            self.assertEqual(
                stats[0]["corruption_value_sha256"],
                stats[1]["corruption_value_sha256"],
            )

    def test_200_updates_equal_100_checkpoint_plus_100(self):
        config = ExperimentConfig(
            "riql_naive",
            "hopper-medium-replay-v2",
            hidden_dim=4,
            hidden_layers=1,
            offline_steps=200,
            batch_size=2,
        )
        torch.manual_seed(41)
        uninterrupted = build_agent(config, 3, 2, 1.0, torch.device("cpu"))
        replay = ReplayBuffer(3, 2, 32, seed=43)
        for index in range(16):
            replay.add(
                np.asarray([index, index + 1, index + 2], dtype=np.float32) / 16,
                np.asarray([index, -index], dtype=np.float32) / 16,
                float(index) / 16,
                np.asarray([index + 1, index + 2, index + 3], dtype=np.float32) / 16,
                float(index % 7 == 0),
            )
        for _ in range(100):
            uninterrupted.update(replay.sample(2, torch.device("cpu")))
        agent_checkpoint = copy.deepcopy(uninterrupted.checkpoint_state())
        replay_checkpoint = copy.deepcopy(replay.state_dict())
        torch_checkpoint = torch.random.get_rng_state().clone()
        for _ in range(100):
            uninterrupted.update(replay.sample(2, torch.device("cpu")))

        torch.manual_seed(999)
        resumed = build_agent(config, 3, 2, 1.0, torch.device("cpu"))
        resumed.load_checkpoint_state(agent_checkpoint)
        resumed_replay = ReplayBuffer(3, 2, 32, seed=999)
        resumed_replay.load_state_dict(replay_checkpoint)
        torch.random.set_rng_state(torch_checkpoint)
        for _ in range(100):
            resumed.update(resumed_replay.sample(2, torch.device("cpu")))

        _assert_nested_equal(
            self,
            uninterrupted.checkpoint_state(),
            resumed.checkpoint_state(),
        )
        _assert_nested_equal(self, replay.state_dict(), resumed_replay.state_dict())
        self.assertEqual(uninterrupted.total_updates, 200)

    def test_run_path_contains_manifest_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            config = ExperimentConfig(
                "wsrl",
                "hopper-medium-replay-v2",
                output_dir=directory,
                comparison_name="manifest_path_test",
            )
            logger = RunLogger(config)
            resolved = config.to_dict()
            resolved.update(
                dataset_id=config.env_name,
                evaluation_env_id=config.env_name,
                online_env_id=config.env_name,
                dataset_sha256="dataset",
                normalizer_sha256="normalizer",
            )
            logger.write_config(resolved)
            self.assertTrue(any(part.startswith("manifest_") for part in logger.run_dir.parts))
            manifest = json.loads(
                (logger.run_dir / "experiment_manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertIn(
                f"manifest_{manifest['manifest_sha256'][:16]}",
                logger.run_dir.parts,
            )
            evaluation = {
                "return_mean": 1.0,
                "return_std": 0.0,
                "normalized_return_mean": 2.0,
                "normalized_return_std": 0.0,
                "evaluation_mode": "deterministic_diagnostic",
            }
            logger.log_evaluation("offline", 1, 0, 1, evaluation)
            for handler in logger.logger.handlers:
                handler.close()
            logger.logger.handlers = []

            resumed_config = ExperimentConfig(
                "wsrl",
                "hopper-medium-replay-v2",
                output_dir=directory,
                comparison_name="manifest_path_test",
                resume_run=str(logger.run_dir),
            )
            resumed_logger = RunLogger(resumed_config)
            resumed_resolved = resumed_config.to_dict()
            resumed_resolved.update(
                dataset_id=resumed_config.env_name,
                evaluation_env_id=resumed_config.env_name,
                online_env_id=resumed_config.env_name,
                dataset_sha256="dataset",
                normalizer_sha256="normalizer",
            )
            resumed_logger.write_config(resumed_resolved)
            resumed_logger.log_evaluation("offline", 2, 0, 2, evaluation)
            with resumed_logger.metrics_path.open(
                newline="", encoding="utf-8"
            ) as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 2)
            self.assertLess(
                float(rows[0]["elapsed_seconds"]),
                float(rows[1]["elapsed_seconds"]),
            )
            for handler in resumed_logger.logger.handlers:
                handler.close()
            resumed_logger.logger.handlers = []

    def test_manifest_hash_and_seed_only_aggregation(self):
        config = ExperimentConfig("wsrl", "hopper-medium-replay-v2").to_dict()
        config.update(
            algorithm="wsrl",
            environment_protocol="rpex_d4rl_v2_legacy",
            dataset_id="hopper-medium-replay-v2",
            evaluation_env_id="hopper-medium-replay-v2",
            online_env_id="hopper-medium-replay-v2",
            dataset_sha256="abc",
            normalizer_sha256="def",
        )
        first = build_experiment_manifest(config)
        second = json.loads(json.dumps(first))
        second["learner_seed"] = 99
        second["manifest_sha256"] = canonical_json_sha256(
            {key: value for key, value in second.items() if key != "manifest_sha256"}
        )
        self.assertEqual(aggregation_signature(first), aggregation_signature(second))
        second["selected_transition_count"] = 17
        second["selected_transition_hash"] = "seed-specific-mask"
        second["corruption_value_hash"] = "seed-specific-values"
        self.assertEqual(aggregation_signature(first), aggregation_signature(second))
        second["action_clipping"] = not second["action_clipping"]
        self.assertNotEqual(aggregation_signature(first), aggregation_signature(second))


if __name__ == "__main__":
    unittest.main()
