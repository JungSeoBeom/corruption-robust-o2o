from __future__ import annotations

import hashlib
import json
import os
import unittest
from pathlib import Path
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
    _discover_runtime_evidence,
    _latest_completed,
    _settings_for_suite,
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
        self.assertEqual(
            MAIN_BASELINES,
            (
                "rpex",
                "riql_naive",
                "wsrl",
                "cal_ql",
                "pessimistic_q_ensemble",
            ),
        )
        self.assertEqual(OPTIONAL_ADAPTED_BASELINES, ())
        self.assertEqual(OPTIONAL_APPROXIMATION_BASELINES, ())
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
            if algorithm == "rpex":
                self.assertEqual(config.evaluation_mode, "both")
                self.assertEqual(
                    config.evaluation_policy_profile,
                    "official_code_epsilon_switching",
                )
            else:
                self.assertEqual(
                    config.evaluation_mode, "deterministic_diagnostic"
                )
                self.assertEqual(
                    config.evaluation_policy_profile, "deterministic_diagnostic"
                )

    def test_canonical_and_alias_names_select_the_main_implementations(self):
        self.assertEqual(
            ExperimentConfig("calql", "hopper-medium-replay-v2").algorithm,
            "cal_ql",
        )
        self.assertEqual(
            ExperimentConfig("pqe", "hopper-medium-replay-v2").algorithm,
            "pessimistic_q_ensemble",
        )
        with self.assertRaisesRegex(ValueError, "historical result name"):
            ExperimentConfig(
                "cal_ql_locomotion_adaptation", "hopper-medium-replay-v2"
            )
        with self.assertRaisesRegex(ValueError, "retired for new runs"):
            ExperimentConfig(
                "pqe_shared_actor_approx", "hopper-medium-replay-v2"
            )

    def test_calql_and_pqe_have_honest_main_metadata(self):
        adapted = self.make_config("cal_ql").to_dict()
        port = self.make_config("pessimistic_q_ensemble").to_dict()
        self.assertEqual(
            adapted["implementation_type"],
            "source_aligned_locomotion_adaptation",
        )
        self.assertEqual(adapted["benchmark_role"], "main")
        self.assertTrue(adapted["main_table_eligible"])
        self.assertEqual(
            port["implementation_type"], "source_aligned_d4rl_v2_port"
        )
        self.assertEqual(port["benchmark_role"], "main")
        self.assertTrue(port["main_table_eligible"])
        self.assertEqual(port["upstream_task_version"], "v0")
        self.assertEqual(port["benchmark_task_version"], "v2")
        self.assertEqual(port["offline_compute_multiplier"], 5)
        self.assertFalse(port["shared_actor"])

    def test_calql_oracle_corruption_labels_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "corruption masks/labels"):
            self.make_config(
                "cal_ql",
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
    def test_all_suite_readiness_order_is_clean_adversarial_random(self):
        settings = _settings_for_suite("all")
        modes = [mode for mode, _target in settings]
        self.assertEqual(modes[0], "clean")
        first_random = modes.index("random")
        self.assertTrue(all(mode == "adversarial" for mode in modes[1:first_random]))
        self.assertTrue(all(mode == "random" for mode in modes[first_random:]))

    def test_latest_completed_uses_completion_manifest_mtime(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            completion_paths = {}
            for name in ("z_lexically_later_old", "a_lexically_earlier_new"):
                run_dir = root / name
                run_dir.mkdir()
                (run_dir / "config.json").write_text(
                    json.dumps(
                        {
                            "algorithm": "cal_ql",
                            "env_name": "hopper-medium-replay-v2",
                        }
                    ),
                    encoding="utf-8",
                )
                (run_dir / "summary.json").write_text(
                    json.dumps({"status": "completed"}), encoding="utf-8"
                )
                completion = run_dir / "completed_experiment_manifest.json"
                completion.write_text(
                    json.dumps({"actual_online_steps": 1}), encoding="utf-8"
                )
                completion_paths[name] = completion

            old_ns = 1_700_000_000_000_000_000
            new_ns = old_ns + 1_000_000_000
            os.utime(
                completion_paths["z_lexically_later_old"],
                ns=(old_ns, old_ns),
            )
            os.utime(
                completion_paths["a_lexically_earlier_new"],
                ns=(new_ns, new_ns),
            )

            records = _discover_runtime_evidence([directory])
            selected = _latest_completed(
                records, "cal_ql", "hopper-medium-replay-v2"
            )
            self.assertIsNotNone(selected)
            self.assertEqual(selected.run_dir.name, "a_lexically_earlier_new")

    def test_runtime_loader_leaves_post_training_evidence_pending(self):
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
        failures = [check for check in checks if check.ok is False]
        self.assertEqual(failures, [])
        pending = {check.label for check in checks if check.ok is None}
        self.assertEqual(
            pending,
            {
                "CAL-QL ONLINE MC EVIDENCE",
                "PQE MEMBER CHECKPOINT EVIDENCE",
            },
        )

    def test_static_only_never_claims_runtime_evidence(self):
        args = build_readiness_parser().parse_args(
            [
                "--env-name",
                "hopper-medium-replay-v2",
                "--corruption-suite",
                "clean",
                "--static-only",
            ]
        )
        checks = run_checks(args)
        environment = next(
            check for check in checks if check.label == "D4RL ENVIRONMENT"
        )
        self.assertIsNone(environment.ok)
        self.assertIn("PENDING", environment.detail)

    def test_completed_run_directories_validate_runtime_evidence(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)

            calql = root / "calql"
            calql.mkdir()
            (calql / "config.json").write_text(
                json.dumps(
                    {
                        "algorithm": "cal_ql",
                        "env_name": "hopper-medium-replay-v2",
                    }
                ),
                encoding="utf-8",
            )
            (calql / "summary.json").write_text(
                json.dumps({"status": "completed"}), encoding="utf-8"
            )
            (calql / "completed_experiment_manifest.json").write_text(
                json.dumps(
                    {
                        "completed_online_trajectories": 2,
                        "completed_online_transitions": 7,
                        "pending_episode_length": 0,
                        "effective_calql_training_transitions": 7,
                        "requested_online_steps": 5,
                        "actual_online_steps": 7,
                        "episode_boundary_overshoot": 2,
                        "online_budget_semantics": (
                            "calql_complete_current_episode_at_or_after_requested"
                        ),
                        "online_mc_return_valid_fraction": 1.0,
                        "online_critic_gradient_updates": 7,
                        "calql_online_cql_enabled": True,
                    }
                ),
                encoding="utf-8",
            )

            pqe = root / "pqe"
            members = pqe / "checkpoints" / "offline" / "members"
            members.mkdir(parents=True)
            member_hashes = []
            for index in range(5):
                payload = f"independent-{index}".encode()
                (members / f"member_{index}.pt").write_bytes(payload)
                member_hashes.append(hashlib.sha256(payload).hexdigest())
            (pqe / "config.json").write_text(
                json.dumps(
                    {
                        "algorithm": "pessimistic_q_ensemble",
                        "env_name": "hopper-medium-replay-v2",
                    }
                ),
                encoding="utf-8",
            )
            (pqe / "summary.json").write_text(
                json.dumps({"status": "completed"}), encoding="utf-8"
            )
            (pqe / "completed_experiment_manifest.json").write_text(
                json.dumps(
                    {
                        "pqe_member_checkpoint_hashes": member_hashes,
                        "shared_actor": False,
                        "actor_independence": True,
                        "critic_independence": True,
                        "pqe_replay_mode": "balanced_density",
                    }
                ),
                encoding="utf-8",
            )

            args = build_readiness_parser().parse_args(
                [
                    "--env-name",
                    "hopper-medium-replay-v2",
                    "--corruption-suite",
                    "clean",
                    "--run-dir",
                    directory,
                ]
            )
            checks = run_checks(
                args,
                environment_loader=lambda *unused_args, **unused_kwargs: {},
            )
            for label in (
                "CAL-QL ONLINE MC EVIDENCE",
                "PQE MEMBER CHECKPOINT EVIDENCE",
            ):
                check = next(item for item in checks if item.label == label)
                self.assertTrue(check.ok, check.detail)

    def test_calql_runtime_evidence_rejects_pending_or_unaccounted_steps(self):
        with TemporaryDirectory() as directory:
            run_dir = Path(directory) / "calql"
            run_dir.mkdir()
            (run_dir / "config.json").write_text(
                json.dumps(
                    {
                        "algorithm": "cal_ql",
                        "env_name": "hopper-medium-replay-v2",
                    }
                ),
                encoding="utf-8",
            )
            (run_dir / "summary.json").write_text(
                json.dumps({"status": "completed"}), encoding="utf-8"
            )
            (run_dir / "completed_experiment_manifest.json").write_text(
                json.dumps(
                    {
                        "completed_online_trajectories": 1,
                        "completed_online_transitions": 7,
                        "pending_episode_length": 1,
                        "effective_calql_training_transitions": 7,
                        "requested_online_steps": 5,
                        "actual_online_steps": 7,
                        "episode_boundary_overshoot": 2,
                        "online_budget_semantics": (
                            "calql_complete_current_episode_at_or_after_requested"
                        ),
                        "online_mc_return_valid_fraction": 1.0,
                        "online_critic_gradient_updates": 7,
                        "calql_online_cql_enabled": True,
                    }
                ),
                encoding="utf-8",
            )
            args = build_readiness_parser().parse_args(
                [
                    "--env-name",
                    "hopper-medium-replay-v2",
                    "--corruption-suite",
                    "clean",
                    "--run-dir",
                    directory,
                ]
            )
            checks = run_checks(
                args,
                environment_loader=lambda *unused_args, **unused_kwargs: {},
            )
            evidence = next(
                check
                for check in checks
                if check.label == "CAL-QL ONLINE MC EVIDENCE"
            )
            self.assertFalse(evidence.ok)
            self.assertRegex(
                evidence.detail,
                "pending_episode_length",
            )

    def test_supplied_incomplete_runtime_directory_fails_closed(self):
        with TemporaryDirectory() as directory:
            args = build_readiness_parser().parse_args(
                [
                    "--env-name",
                    "hopper-medium-replay-v2",
                    "--corruption-suite",
                    "clean",
                    "--run-dir",
                    directory,
                ]
            )
            checks = run_checks(
                args,
                environment_loader=lambda *unused_args, **unused_kwargs: {},
            )
            runtime = {
                check.label: check
                for check in checks
                if check.label
                in {
                    "CAL-QL ONLINE MC EVIDENCE",
                    "PQE MEMBER CHECKPOINT EVIDENCE",
                }
            }
            self.assertEqual(set(runtime), {
                "CAL-QL ONLINE MC EVIDENCE",
                "PQE MEMBER CHECKPOINT EVIDENCE",
            })
            self.assertTrue(all(check.ok is False for check in runtime.values()))

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
