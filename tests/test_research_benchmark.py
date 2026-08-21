from __future__ import annotations

import unittest
from tempfile import TemporaryDirectory

from robust_o2o.config import ExperimentConfig
from robust_o2o.fidelity import (
    BASELINE_REPRODUCTION_REGISTRY,
    MAIN_BASELINES,
    OPTIONAL_ADAPTED_BASELINES,
    OPTIONAL_APPROXIMATION_BASELINES,
)
from robust_o2o.experiment import _runtime_update_metadata
from robust_o2o.paths import comparison_directory
from scripts.check_research_readiness import (
    build_parser as build_readiness_parser,
    run_checks,
)


class ResearchBenchmarkConfigTest(unittest.TestCase):
    def make_config(self, algorithm: str, **overrides) -> ExperimentConfig:
        values = {
            "algorithm": algorithm,
            "env_name": "hopper-medium-replay-v2",
            "run_purpose": "research_benchmark",
            "suite_profile": "research_benchmark",
            "implementation_profile": "research_benchmark",
        }
        values.update(overrides)
        return ExperimentConfig(**values)

    def test_main_and_optional_baseline_registry_is_explicit(self):
        self.assertEqual(MAIN_BASELINES, ("rpex", "riql_naive", "wsrl"))
        self.assertEqual(
            OPTIONAL_ADAPTED_BASELINES,
            ("cal_ql_locomotion_adaptation",),
        )
        self.assertEqual(
            OPTIONAL_APPROXIMATION_BASELINES,
            ("pqe_shared_actor_approx",),
        )
        for algorithm in MAIN_BASELINES:
            record = BASELINE_REPRODUCTION_REGISTRY[algorithm]
            self.assertEqual(record.benchmark_role, "main")
            self.assertTrue(record.main_table_eligible)
        for algorithm in (
            *OPTIONAL_ADAPTED_BASELINES,
            *OPTIONAL_APPROXIMATION_BASELINES,
        ):
            self.assertFalse(
                BASELINE_REPRODUCTION_REGISTRY[algorithm].main_table_eligible
            )

    def test_custom_budget_seed_and_common_reporting_are_not_overridden(self):
        for algorithm in MAIN_BASELINES:
            config = self.make_config(
                algorithm,
                seed=17,
                offline_steps=123,
                online_steps=456,
                eval_period=19,
                eval_episodes=7,
                final_window_size=2,
            )
            self.assertEqual(config.seed, 17)
            self.assertEqual(config.offline_steps, 123)
            self.assertEqual(config.online_steps, 456)
            self.assertEqual(config.eval_period, 19)
            self.assertEqual(config.eval_episodes, 7)
            self.assertEqual(config.final_window_size, 2)
            self.assertEqual(config.evaluation_mode, "deterministic_diagnostic")
            self.assertEqual(
                config.evaluation_policy_profile, "deterministic_diagnostic"
            )

    def test_retired_algorithm_names_are_never_silently_aliased(self):
        with self.assertRaisesRegex(ValueError, "Exact Pessimistic Q-Ensemble"):
            ExperimentConfig(
                "pessimistic_q_ensemble", "hopper-medium-replay-v2"
            )
        with self.assertRaisesRegex(ValueError, "task adaptation"):
            ExperimentConfig("cal_ql", "hopper-medium-replay-v2")

    def test_optional_results_have_non_main_metadata(self):
        adapted = self.make_config("cal_ql_locomotion_adaptation").to_dict()
        approximate = self.make_config("pqe_shared_actor_approx").to_dict()
        self.assertEqual(adapted["implementation_type"], "task_adaptation")
        self.assertEqual(adapted["benchmark_role"], "optional_adapted")
        self.assertFalse(adapted["main_table_eligible"])
        self.assertEqual(approximate["implementation_type"], "approximation")
        self.assertEqual(approximate["benchmark_role"], "optional_diagnostic")
        self.assertFalse(approximate["main_table_eligible"])

    def test_calql_oracle_corruption_labels_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "corruption masks/labels"):
            self.make_config(
                "cal_ql_locomotion_adaptation",
                calibration_mask_mode="oracle_exclude_corrupted",
            )

    def test_wsrl_core_research_protocol(self):
        config = self.make_config("wsrl")
        self.assertEqual(config.warmup_steps, 5_000)
        self.assertEqual(config.wsrl_num_critics, 10)
        self.assertEqual(config.wsrl_target_critic_subsample_size, 2)
        self.assertEqual(config.wsrl_utd_ratio, 4)
        self.assertEqual(config.effective_offline_ratio, 0.0)
        metadata = config.to_dict()
        self.assertEqual(metadata["offline_pretrainer"], "cql_redq")
        self.assertFalse(metadata["offline_data_retained_online"])

    def test_wsrl_nonzero_offline_replay_cannot_be_mislabeled_main(self):
        with self.assertRaisesRegex(ValueError, "separately named adaptation"):
            self.make_config("wsrl", offline_ratio=0.5)

    def test_adversarial_target_and_checkpoint_fail_fast(self):
        with self.assertRaisesRegex(ValueError, "mixed is unsupported"):
            self.make_config(
                "rpex",
                corruption="adversarial",
                corruption_target="mixed",
            )
        with self.assertRaisesRegex(FileNotFoundError, "checkpoint.*not found"):
            self.make_config(
                "rpex",
                corruption="adversarial",
                corruption_target="observations",
                attack_checkpoint="/tmp/research-benchmark-missing-oracle.pt",
                attack_checkpoint_sha256="0" * 64,
            )
        reward = self.make_config(
            "rpex",
            corruption="adversarial",
            corruption_target="rewards",
        )
        self.assertEqual(reward.corruption_target, "rewards")

    def test_research_metadata_records_contract_without_publication_claim(self):
        metadata = self.make_config("rpex").to_dict()
        self.assertEqual(
            metadata["corruption_application_contract"],
            "replay_transition_poisoning",
        )
        self.assertTrue(metadata["clean_evaluation"])
        self.assertFalse(metadata["uses_corruption_labels"])
        self.assertTrue(metadata["main_table_eligible"])
        self.assertFalse(metadata["publication_eligible"])

    def test_measured_update_counts_separate_compute_from_env_budget(self):
        config = self.make_config("wsrl")
        agent = type(
            "Counters",
            (),
            {
                "actor_updates": 113,
                "critic_updates": 152,
                "temperature_updates": 113,
                "total_updates": 152,
            },
        )()
        metadata = _runtime_update_metadata(
            config,
            agent,
            online_initial_update_counts={
                "actor": 100,
                "critic": 100,
                "temperature": 100,
            },
            online_environment_steps=20,
        )
        self.assertEqual(metadata["critic_gradient_updates"], 152)
        self.assertEqual(metadata["online_critic_gradient_updates"], 52)
        self.assertEqual(metadata["online_actor_gradient_updates"], 13)
        self.assertEqual(metadata["online_temperature_updates"], 13)
        self.assertEqual(metadata["configured_utd"], 4)
        self.assertAlmostEqual(metadata["actual_utd"], 2.6)


class ResearchReadinessTest(unittest.TestCase):
    def test_runtime_loader_can_satisfy_practical_readiness(self):
        args = build_readiness_parser().parse_args(
            [
                "--env-name",
                "hopper-medium-replay-v2",
                "--corruption-suite",
                "clean",
            ]
        )
        checks = run_checks(
            args,
            environment_loader=lambda *unused_args, **unused_kwargs: {
                "dataset": "loaded"
            },
        )
        failures = [check for check in checks if not check.ok]
        self.assertEqual(failures, [])

    def test_existing_explicit_comparison_path_is_a_collision(self):
        with TemporaryDirectory() as directory:
            path = comparison_directory(
                directory,
                "hopper-medium-replay-v2",
                "clean",
                "none",
                "existing",
                "rpex_d4rl_v2_legacy",
                "research_benchmark__research_benchmark__research_benchmark",
            )
            path.mkdir(parents=True)
            args = build_readiness_parser().parse_args(
                [
                    "--env-name",
                    "hopper-medium-replay-v2",
                    "--corruption-suite",
                    "clean",
                    "--output-root",
                    directory,
                    "--experiment-name",
                    "existing",
                ]
            )
            checks = run_checks(
                args,
                environment_loader=lambda *unused_args, **unused_kwargs: {},
            )
            output = next(check for check in checks if check.label == "OUTPUT PATH")
            self.assertFalse(output.ok)
            self.assertIn("already exists", output.detail)


if __name__ == "__main__":
    unittest.main()
