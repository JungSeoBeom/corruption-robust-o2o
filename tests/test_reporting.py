from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from robust_o2o.fidelity import (
    BASELINE_REPRODUCTION_REGISTRY,
    MAIN_BASELINES,
    REPORTING_RULES,
)
from robust_o2o.reporting import (
    CALQL_ONLINE_BUDGET_SEMANTICS,
    D4RL_SCORE_SEMANTICS,
    LEGACY_RESEARCH_PROTOCOL,
    PER_SEED_COLUMNS,
    RESEARCH_SUMMARY_COLUMNS,
    SUMMARY_COLUMNS,
    ReportingValidationError,
    aggregate_seed_scores,
    classify_score_semantics,
    validate_calql_completion_accounting,
    write_reporting_outputs,
)


def evaluation_frame(
    *,
    seeds=(0, 1),
    online_scores=(1.0, 2.0, 3.0, 4.0),
    duplicate_last_step=False,
):
    rows = []
    for seed in seeds:
        run_dir = f"/tmp/rpex_seed_{seed}"
        rows.append(
            {
                "run_dir": run_dir,
                "algorithm": "rpex",
                "seed": seed,
                "env_name": "hopper-medium-replay-v2",
                "corruption": "random",
                "corruption_target": "observations",
                "phase": "offline",
                "step": 2_000_000,
                "env_steps": 0,
                "normalized_return_mean": 9999.0,
                "run_status": "completed",
                "eval_episodes": 10,
                "implementation_fidelity": "source_aligned_port",
                "upstream_commit": "35da71e",
                "publication_eligible": True,
                "paper_reproduction_eligible": False,
                "learner_parity_verified": True,
                "reporting_rule_verified": True,
                "condition_certificate_verified": True,
                "condition_status": "paper_reproduction_condition",
                "run_purpose": "final_benchmark",
                "protocol": LEGACY_RESEARCH_PROTOCOL,
                "score_semantics": D4RL_SCORE_SEMANTICS,
                "benchmark_eligible": True,
                "planned_online_steps": 40_000,
                "planned_offline_steps": 2_000_000,
                "actual_online_steps": 40_001,
                "episode_boundary_overshoot": 1,
                "eval_period": 10_000,
                "online_budget_semantics": (
                    "rpex_official_episode_boundary_strict_greater_than"
                ),
                "environment_horizon": 1_000,
            }
        )
        for index, score in enumerate(online_scores, start=1):
            rows.append(
                {
                    "run_dir": run_dir,
                    "algorithm": "rpex",
                    "seed": seed,
                    "env_name": "hopper-medium-replay-v2",
                    "corruption": "random",
                    "corruption_target": "observations",
                    "phase": "online",
                    "step": index * 10_000,
                    "env_steps": index * 10_000,
                    "normalized_return_mean": score + seed * 10.0,
                    "run_status": "completed",
                    "eval_episodes": 10,
                    "implementation_fidelity": "source_aligned_port",
                    "upstream_commit": "35da71e",
                    "publication_eligible": True,
                    "paper_reproduction_eligible": False,
                    "learner_parity_verified": True,
                    "reporting_rule_verified": True,
                    "condition_certificate_verified": True,
                    "condition_status": "paper_reproduction_condition",
                    "run_purpose": "final_benchmark",
                    "protocol": LEGACY_RESEARCH_PROTOCOL,
                    "score_semantics": D4RL_SCORE_SEMANTICS,
                    "benchmark_eligible": True,
                    "planned_online_steps": 40_000,
                    "planned_offline_steps": 2_000_000,
                    "actual_online_steps": 40_001,
                    "episode_boundary_overshoot": 1,
                    "eval_period": 10_000,
                    "online_budget_semantics": (
                        "rpex_official_episode_boundary_strict_greater_than"
                    ),
                    "environment_horizon": 1_000,
                }
            )
        if duplicate_last_step:
            rows.append(dict(rows[-1]))
    return pd.DataFrame(rows)


class ReportingRuleTest(unittest.TestCase):
    def test_score_semantics_classification_is_protocol_based_and_fail_closed(self):
        self.assertEqual(
            classify_score_semantics(
                {
                    "protocol": LEGACY_RESEARCH_PROTOCOL,
                    "score_semantics": D4RL_SCORE_SEMANTICS,
                    "benchmark_eligible": True,
                    "run_purpose": "research_benchmark",
                }
            ),
            (D4RL_SCORE_SEMANTICS, True),
        )
        diagnostic = classify_score_semantics(
            {
                "protocol": "local_gymnasium_v4_diagnostic",
                "score_semantics": D4RL_SCORE_SEMANTICS,
                "benchmark_eligible": True,
                "run_purpose": "research_benchmark",
            }
        )
        self.assertEqual(
            diagnostic,
            ("diagnostic_d4rl_reference_scaled_return", False),
        )
        self.assertEqual(
            classify_score_semantics(
                {
                    "protocol": LEGACY_RESEARCH_PROTOCOL,
                    "run_purpose": "research_benchmark",
                }
            ),
            ("unknown_legacy_score", False),
        )
        self.assertEqual(
            classify_score_semantics(
                {
                    "protocol": "unknown_legacy_protocol",
                    "score_semantics": D4RL_SCORE_SEMANTICS,
                    "benchmark_eligible": True,
                    "run_purpose": "research_benchmark",
                }
            ),
            ("unknown_legacy_score", False),
        )

    def test_calql_completion_accounting_contract(self):
        valid = {
            "requested_online_steps": 500,
            "actual_online_steps": 517,
            "episode_boundary_overshoot": 17,
            "completed_online_transitions": 517,
            "pending_episode_length": 0,
            "effective_calql_training_transitions": 517,
            "online_budget_semantics": CALQL_ONLINE_BUDGET_SEMANTICS,
        }
        self.assertEqual(
            validate_calql_completion_accounting(valid),
            {key: valid[key] for key in valid if key != "online_budget_semantics"},
        )
        invalid_cases = {
            "semantics": {"online_budget_semantics": "exact_environment_steps"},
            "overshoot": {"episode_boundary_overshoot": 16},
            "pending": {"pending_episode_length": 1},
            "completed": {"completed_online_transitions": 516},
            "effective": {"effective_calql_training_transitions": 516},
        }
        for label, mutation in invalid_cases.items():
            with self.subTest(label=label):
                with self.assertRaises(ReportingValidationError):
                    validate_calql_completion_accounting(
                        {**valid, **mutation}
                    )

    def test_empty_frame_is_typed_non_strict_and_fails_strict(self):
        per_seed, summary = aggregate_seed_scores(
            pd.DataFrame(),
            rule_by_algorithm=REPORTING_RULES,
            strict=False,
        )
        self.assertEqual(tuple(per_seed.columns), PER_SEED_COLUMNS)
        self.assertEqual(tuple(summary.columns), SUMMARY_COLUMNS)
        self.assertTrue(per_seed.empty)
        self.assertTrue(summary.empty)
        with self.assertRaisesRegex(ReportingValidationError, "no evaluation"):
            aggregate_seed_scores(
                pd.DataFrame(),
                rule_by_algorithm=REPORTING_RULES,
                strict=True,
            )

    def test_last_three_then_seed_population_mean_std_and_phase_isolation(self):
        per_seed, summary = aggregate_seed_scores(
            evaluation_frame(),
            rule_by_algorithm=REPORTING_RULES,
            strict=True,
            expected_seeds=[0, 1],
        )
        self.assertEqual(per_seed["seed_score"].tolist(), [3.0, 13.0])
        self.assertEqual(
            per_seed["selected_evaluation_steps"].tolist(),
            ["20000;30000;40000", "20000;30000;40000"],
        )
        self.assertEqual(float(summary.iloc[0]["mean"]), 8.0)
        self.assertEqual(float(summary.iloc[0]["std"]), 5.0)
        self.assertEqual(int(summary.iloc[0]["std_ddof"]), 0)

    def test_fewer_than_three_final_evaluations_fails_strict(self):
        with self.assertRaisesRegex(ReportingValidationError, "requires 3"):
            aggregate_seed_scores(
                evaluation_frame(online_scores=(1.0, 2.0)),
                rule_by_algorithm=REPORTING_RULES,
                strict=True,
                expected_seeds=[0, 1],
            )

    def test_non_strict_partial_window_is_not_labeled_as_full_last_three(self):
        per_seed, _ = aggregate_seed_scores(
            evaluation_frame(seeds=(0,), online_scores=(1.0, 2.0)),
            rule_by_algorithm=REPORTING_RULES,
            strict=False,
        )
        self.assertIn("partial_2_of_3", per_seed.iloc[0]["aggregation_rule"])

    def test_strict_rejects_stale_final_evaluation(self):
        frame = evaluation_frame(seeds=(0,))
        frame["planned_online_steps"] = 50_000
        frame["actual_online_steps"] = 50_001
        with self.assertRaisesRegex(ReportingValidationError, "stale"):
            aggregate_seed_scores(
                frame,
                rule_by_algorithm=REPORTING_RULES,
                strict=True,
                expected_seeds=[0],
            )

    def test_strict_rejects_noncontiguous_final_window(self):
        frame = evaluation_frame(seeds=(0,))
        online = frame["phase"] == "online"
        frame.loc[online & (frame["env_steps"] == 30_000), "env_steps"] = 10_000
        frame.loc[online & (frame["step"] == 30_000), "step"] = 10_000
        with self.assertRaisesRegex(ReportingValidationError, "duplicate online"):
            aggregate_seed_scores(
                frame,
                rule_by_algorithm=REPORTING_RULES,
                strict=True,
                expected_seeds=[0],
            )

        frame = evaluation_frame(seeds=(0,))
        frame.loc[
            (frame["phase"] == "online") & (frame["env_steps"] == 30_000),
            ["env_steps", "step"],
        ] = 25_000
        with self.assertRaisesRegex(ReportingValidationError, "grid|contiguous"):
            aggregate_seed_scores(
                frame,
                rule_by_algorithm=REPORTING_RULES,
                strict=True,
                expected_seeds=[0],
            )

    def test_strict_rejects_premature_or_excessively_overshot_budget(self):
        frame = evaluation_frame(seeds=(0,))
        frame["actual_online_steps"] = 39_999
        with self.assertRaisesRegex(ReportingValidationError, "budget semantics"):
            aggregate_seed_scores(
                frame,
                rule_by_algorithm=REPORTING_RULES,
                strict=True,
                expected_seeds=[0],
            )

        frame = evaluation_frame(seeds=(0,))
        frame["actual_online_steps"] = 42_000
        with self.assertRaisesRegex(ReportingValidationError, "budget semantics"):
            aggregate_seed_scores(
                frame,
                rule_by_algorithm=REPORTING_RULES,
                strict=True,
                expected_seeds=[0],
            )

    def test_strict_rejects_inconsistent_completion_overshoot(self):
        frame = evaluation_frame(seeds=(0,))
        frame["episode_boundary_overshoot"] = 2
        with self.assertRaisesRegex(
            ReportingValidationError, "completion overshoot mismatch"
        ):
            aggregate_seed_scores(
                frame,
                rule_by_algorithm=REPORTING_RULES,
                strict=True,
                expected_seeds=[0],
            )

    def test_missing_seed_fails_strict(self):
        with self.assertRaisesRegex(ReportingValidationError, "seed set mismatch"):
            aggregate_seed_scores(
                evaluation_frame(seeds=(0,)),
                rule_by_algorithm=REPORTING_RULES,
                strict=True,
                expected_seeds=[0, 1],
            )

    def test_duplicate_evaluation_step_fails(self):
        with self.assertRaisesRegex(ReportingValidationError, "duplicate online"):
            aggregate_seed_scores(
                evaluation_frame(duplicate_last_step=True),
                rule_by_algorithm=REPORTING_RULES,
                strict=False,
            )

    def test_multiple_runs_for_same_seed_fail(self):
        frame = evaluation_frame(seeds=(0,))
        duplicate = frame.copy()
        duplicate["run_dir"] = "/tmp/duplicate_rpex_seed_0"
        with self.assertRaisesRegex(ReportingValidationError, "same seed identity"):
            aggregate_seed_scores(
                pd.concat([frame, duplicate], ignore_index=True),
                rule_by_algorithm=REPORTING_RULES,
                strict=False,
            )

    def test_algorithm_specific_rules_do_not_inherit_rpex_window(self):
        rpex = evaluation_frame(seeds=(0,))
        wsrl = evaluation_frame(seeds=(0,), online_scores=(5.0, 9.0))
        wsrl["algorithm"] = "wsrl"
        wsrl["run_dir"] = "/tmp/wsrl_seed_0"
        wsrl["eval_episodes"] = 20
        per_seed, _ = aggregate_seed_scores(
            pd.concat([rpex, wsrl], ignore_index=True),
            rule_by_algorithm=REPORTING_RULES,
            strict=False,
        )
        scores = per_seed.set_index("algorithm")["seed_score"].to_dict()
        self.assertEqual(scores["rpex"], 3.0)
        self.assertEqual(scores["wsrl"], 9.0)
        windows = per_seed.set_index("algorithm")[
            "num_final_evaluations"
        ].to_dict()
        self.assertEqual(windows, {"rpex": 3, "wsrl": 1})

    def test_calql_common_score_excludes_post_requested_budget_evaluation(self):
        frame = evaluation_frame(
            seeds=(0,), online_scores=(48.0, 49.0, 50.0, 999.0)
        )
        frame["algorithm"] = "cal_ql"
        frame["run_dir"] = "/tmp/calql_budget_cap_seed_0"
        frame["run_purpose"] = "research_benchmark"
        frame["benchmark_role"] = "main"
        frame["implementation_type"] = (
            "source_aligned_locomotion_adaptation"
        )
        frame["evaluation_corruption"] = "clean"
        frame["uses_corruption_labels"] = False
        frame["final_window_size"] = 3
        frame["planned_online_steps"] = 500_000
        frame["requested_online_steps"] = 500_000
        frame["actual_online_steps"] = 500_137
        frame["episode_boundary_overshoot"] = 137
        frame["online_budget_semantics"] = CALQL_ONLINE_BUDGET_SEMANTICS
        frame["completed_online_transitions"] = 500_137
        frame["pending_episode_length"] = 0
        frame["effective_calql_training_transitions"] = 500_137
        online = frame["phase"] == "online"
        frame.loc[online, "env_steps"] = [480_000, 490_000, 500_000, 500_137]
        frame.loc[online, "step"] = [480_000, 490_000, 500_000, 500_137]

        with tempfile.TemporaryDirectory() as directory:
            outputs = write_reporting_outputs(
                frame,
                Path(directory),
                strict=False,
                expected_seeds=[0],
            )
            scores = pd.read_csv(
                outputs["research_per_seed_final_scores"]
            )
        self.assertEqual(
            scores.iloc[0]["selected_evaluation_steps"],
            "480000;490000;500000",
        )
        self.assertEqual(float(scores.iloc[0]["seed_score"]), 49.0)
        self.assertIn(500_137, frame.loc[online, "env_steps"].tolist())

    def test_research_and_diagnostic_score_semantics_are_separated(self):
        research = evaluation_frame(seeds=(0,))
        research["run_purpose"] = "research_benchmark"
        research["benchmark_role"] = "main"
        research["implementation_type"] = "source_aligned_port"
        research["evaluation_corruption"] = "clean"
        research["uses_corruption_labels"] = False
        research["final_window_size"] = 3

        diagnostic = research.copy()
        diagnostic["run_dir"] = "/tmp/rpex_diagnostic_seed_0"
        diagnostic["run_purpose"] = "diagnostic"
        diagnostic["benchmark_role"] = "diagnostic"
        diagnostic["protocol"] = "local_gymnasium_v4_diagnostic"
        diagnostic["score_semantics"] = (
            "diagnostic_d4rl_reference_scaled_return"
        )
        diagnostic["benchmark_eligible"] = False
        diagnostic[
            "diagnostic_d4rl_reference_scaled_return_mean"
        ] = diagnostic["normalized_return_mean"] + 100.0
        diagnostic["normalized_return_mean"] = float("nan")

        with tempfile.TemporaryDirectory() as directory:
            outputs = write_reporting_outputs(
                pd.concat([research, diagnostic], ignore_index=True),
                Path(directory),
                strict=False,
                expected_seeds=[0],
            )
            research_summary = pd.read_csv(outputs["research_summary"])
            diagnostic_summary = pd.read_csv(outputs["diagnostic_summary"])
            common_summary = pd.read_csv(
                outputs["common_benchmark_summary"]
            )

        self.assertEqual(research_summary["seed_score"].tolist(), [3.0])
        self.assertEqual(
            research_summary["score_semantics"].tolist(),
            [D4RL_SCORE_SEMANTICS],
        )
        self.assertEqual(diagnostic_summary["mean"].tolist(), [103.0])
        self.assertEqual(
            diagnostic_summary["score_semantics"].tolist(),
            ["diagnostic_d4rl_reference_scaled_return"],
        )
        self.assertEqual(len(common_summary), 2)
        self.assertEqual(
            set(common_summary["score_semantics"]),
            {
                D4RL_SCORE_SEMANTICS,
                "diagnostic_d4rl_reference_scaled_return",
            },
        )

    def test_missing_score_contract_is_fail_closed_from_research_summary(self):
        frame = evaluation_frame(seeds=(0,))
        frame["run_purpose"] = "research_benchmark"
        frame["benchmark_role"] = "main"
        frame["evaluation_corruption"] = "clean"
        frame = frame.drop(
            columns=["protocol", "score_semantics", "benchmark_eligible"]
        )
        with tempfile.TemporaryDirectory() as directory:
            outputs = write_reporting_outputs(
                frame,
                Path(directory),
                strict=False,
                expected_seeds=[0],
            )
            research = pd.read_csv(outputs["research_summary"])
            diagnostic = pd.read_csv(outputs["diagnostic_summary"])
        self.assertTrue(research.empty)
        self.assertEqual(len(diagnostic), 1)
        self.assertFalse(bool(diagnostic.iloc[0]["benchmark_eligible"]))

    def test_offline_outputs_skip_online_paper_rule_and_write_common_summary(self):
        frame = evaluation_frame(seeds=(0,))
        frame = frame[frame["phase"] == "offline"]
        with tempfile.TemporaryDirectory() as directory:
            outputs = write_reporting_outputs(
                frame,
                Path(directory),
                strict=False,
                expected_seeds=None,
                phase="offline",
            )
            paper = pd.read_csv(outputs["paper_reproduction_summary"])
            common = pd.read_csv(outputs["common_benchmark_summary"])
        self.assertTrue(paper.empty)
        self.assertEqual(len(common), 1)
        self.assertEqual(float(common.iloc[0]["mean"]), 9999.0)
        self.assertIn("last_3_offline", common.iloc[0]["aggregation_rule"])

    def test_non_strict_unregistered_source_algorithm_is_common_only(self):
        frame = evaluation_frame(seeds=(0,))
        frame["algorithm"] = "diagnostic_only"
        frame["run_dir"] = "/tmp/diagnostic_only_seed_0"
        with tempfile.TemporaryDirectory() as directory:
            outputs = write_reporting_outputs(
                frame,
                Path(directory),
                strict=False,
                expected_seeds=None,
            )
            paper = pd.read_csv(outputs["paper_reproduction_summary"])
            common = pd.read_csv(outputs["common_benchmark_summary"])
        self.assertTrue(paper.empty)
        self.assertEqual(common["algorithm"].tolist(), ["diagnostic_only"])
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(
                ReportingValidationError, "no upstream-verified reporting rule"
            ):
                write_reporting_outputs(
                    frame,
                    Path(directory),
                    strict=True,
                    expected_seeds=[0],
                )

    def test_diagnostic_source_rule_does_not_emit_paper_summary(self):
        frame = evaluation_frame(seeds=(0,))
        frame["publication_eligible"] = False
        frame["run_purpose"] = "diagnostic"
        with tempfile.TemporaryDirectory() as directory:
            outputs = write_reporting_outputs(
                frame,
                Path(directory),
                strict=False,
                expected_seeds=None,
            )
            per_seed = pd.read_csv(outputs["per_seed_final_scores"])
            paper = pd.read_csv(outputs["paper_reproduction_summary"])
            common = pd.read_csv(outputs["common_benchmark_summary"])
        self.assertTrue(per_seed.empty)
        self.assertTrue(paper.empty)
        self.assertEqual(len(common), 1)

    def test_source_aligned_run_cannot_enter_paper_reproduction_summary(self):
        frame = evaluation_frame(seeds=(0,))
        self.assertTrue(frame["publication_eligible"].all())
        self.assertFalse(frame["paper_reproduction_eligible"].any())
        with tempfile.TemporaryDirectory() as directory:
            outputs = write_reporting_outputs(
                frame,
                Path(directory),
                strict=False,
                expected_seeds=None,
            )
            per_seed = pd.read_csv(outputs["per_seed_final_scores"])
            paper = pd.read_csv(outputs["paper_reproduction_summary"])
            common = pd.read_csv(outputs["common_benchmark_summary"])
        self.assertEqual(per_seed["reproduction_status"].tolist(), [
            "source_aligned_port"
        ])
        self.assertEqual(float(per_seed.iloc[0]["seed_score"]), 3.0)
        self.assertTrue(paper.empty)
        self.assertEqual(common["reproduction_status"].tolist(), [
            "source_aligned_port"
        ])

    def test_paper_summary_requires_all_three_verification_flags(self):
        for column in (
            "learner_parity_verified",
            "reporting_rule_verified",
            "condition_certificate_verified",
        ):
            with self.subTest(column=column):
                frame = evaluation_frame(seeds=(0,))
                frame["paper_reproduction_eligible"] = True
                frame[column] = False
                with tempfile.TemporaryDirectory() as directory:
                    outputs = write_reporting_outputs(
                        frame,
                        Path(directory),
                        strict=False,
                        expected_seeds=None,
                    )
                    paper = pd.read_csv(
                        outputs["paper_reproduction_summary"]
                    )
                self.assertTrue(paper.empty)

    def test_upstream_verified_run_can_enter_paper_summary(self):
        frame = evaluation_frame(seeds=(0,))
        frame["paper_reproduction_eligible"] = True
        with tempfile.TemporaryDirectory() as directory:
            outputs = write_reporting_outputs(
                frame,
                Path(directory),
                strict=False,
                expected_seeds=None,
            )
            paper = pd.read_csv(outputs["paper_reproduction_summary"])
        self.assertEqual(len(paper), 1)
        self.assertTrue(bool(paper.iloc[0]["learner_parity_verified"]))
        self.assertTrue(bool(paper.iloc[0]["reporting_rule_verified"]))
        self.assertTrue(
            bool(paper.iloc[0]["condition_certificate_verified"])
        )

    def test_unverified_reporting_is_excluded_from_source_outputs(self):
        frame = evaluation_frame(seeds=(0,))
        frame["paper_reproduction_eligible"] = True
        frame["reporting_rule_verified"] = False
        with tempfile.TemporaryDirectory() as directory:
            outputs = write_reporting_outputs(
                frame,
                Path(directory),
                strict=False,
                expected_seeds=None,
            )
            source_seed = pd.read_csv(outputs["per_seed_final_scores"])
            paper = pd.read_csv(outputs["paper_reproduction_summary"])
            common = pd.read_csv(outputs["common_benchmark_summary"])
        self.assertTrue(source_seed.empty)
        self.assertTrue(paper.empty)
        self.assertEqual(len(common), 1)

    def test_failed_seed_is_retained_and_marks_summary_incomplete(self):
        frame = evaluation_frame()
        frame.loc[frame["seed"] == 1, "run_status"] = "failed"
        frame["error_message"] = ""
        frame.loc[frame["seed"] == 1, "error_message"] = "optimizer diverged"
        _, summary = aggregate_seed_scores(
            frame,
            rule_by_algorithm=REPORTING_RULES,
            strict=False,
            expected_seeds=[0, 1],
        )
        self.assertEqual(int(summary.iloc[0]["num_seeds"]), 1)
        self.assertEqual(summary.iloc[0]["completed_seeds"], "0")
        self.assertEqual(summary.iloc[0]["failed_seeds"], "1")
        self.assertEqual(summary.iloc[0]["summary_status"], "incomplete")
        self.assertFalse(bool(summary.iloc[0]["result_eligible"]))

        with tempfile.TemporaryDirectory() as directory:
            outputs = write_reporting_outputs(
                frame,
                Path(directory),
                strict=False,
                expected_seeds=[0, 1],
            )
            statuses = pd.read_csv(outputs["seed_run_status"])
        failed = statuses[statuses["run_status"] == "failed"].iloc[0]
        self.assertEqual(int(failed["seed"]), 1)
        self.assertEqual(failed["error_message"], "optimizer diverged")

    def test_historical_names_are_read_normalized_without_role_promotion(self):
        main = evaluation_frame(seeds=(0,))
        main["run_purpose"] = "research_benchmark"
        main["benchmark_role"] = "main"
        main["implementation_type"] = "source_aligned_port"
        main["uses_corruption_labels"] = False
        main["final_window_size"] = 3
        main["evaluation_corruption"] = "clean"

        adapted = main.copy()
        adapted["algorithm"] = "cal_ql_locomotion_adaptation"
        adapted["run_dir"] = "/tmp/calql_adapted_seed_0"
        adapted["benchmark_role"] = "optional_adapted"
        adapted["implementation_type"] = "task_adaptation"

        local_adapted = adapted.copy()
        local_adapted["run_dir"] = "/tmp/calql_local_adapted_seed_0"
        local_adapted["protocol"] = "local_gymnasium_v4_diagnostic"
        local_adapted["score_semantics"] = (
            "diagnostic_d4rl_reference_scaled_return"
        )
        local_adapted["benchmark_eligible"] = False

        diagnostic = main.copy()
        diagnostic["algorithm"] = "pqe_shared_actor_approx"
        diagnostic["run_dir"] = "/tmp/pqe_approx_seed_0"
        diagnostic["benchmark_role"] = "optional_diagnostic"
        diagnostic["implementation_type"] = "approximation"

        frame = pd.concat(
            [main, adapted, local_adapted, diagnostic], ignore_index=True
        )
        with tempfile.TemporaryDirectory() as directory:
            outputs = write_reporting_outputs(
                frame,
                Path(directory),
                strict=False,
                expected_seeds=[0],
            )
            research = pd.read_csv(outputs["research_summary"])
            adapted_result = pd.read_csv(
                outputs["adapted_baselines_summary"]
            )
            diagnostic_result = pd.read_csv(outputs["diagnostic_summary"])
        self.assertEqual(research["algorithm"].tolist(), ["rpex"])
        self.assertEqual(
            adapted_result["algorithm"].tolist(),
            ["cal_ql"],
        )
        self.assertEqual(
            set(diagnostic_result["algorithm"]),
            {"cal_ql", "pessimistic_q_ensemble"},
        )

    def test_research_summary_is_seed_explicit_five_baseline_main_table(self):
        frames = []
        for index, algorithm in enumerate(MAIN_BASELINES):
            item = evaluation_frame(seeds=(0,))
            item["algorithm"] = algorithm
            item["run_dir"] = f"/tmp/{algorithm}_seed_0"
            item["run_purpose"] = "research_benchmark"
            item["benchmark_role"] = "main"
            item["implementation_type"] = BASELINE_REPRODUCTION_REGISTRY[
                algorithm
            ].implementation_type
            item["uses_corruption_labels"] = False
            item["final_window_size"] = 3
            item["evaluation_corruption"] = "clean"
            item["requested_online_steps"] = 40_000
            if algorithm == "cal_ql":
                item["online_budget_semantics"] = (
                    CALQL_ONLINE_BUDGET_SEMANTICS
                )
                item["completed_online_transitions"] = 40_001
                item["pending_episode_length"] = 0
                item["effective_calql_training_transitions"] = 40_001
            online = item["phase"] == "online"
            item.loc[online, "normalized_return_mean"] += float(index)
            frames.append(item)

        with tempfile.TemporaryDirectory() as directory:
            outputs = write_reporting_outputs(
                pd.concat(frames, ignore_index=True),
                Path(directory),
                strict=False,
                expected_seeds=[0],
            )
            research = pd.read_csv(outputs["research_summary"])

        self.assertEqual(tuple(research.columns), RESEARCH_SUMMARY_COLUMNS)
        self.assertEqual(set(research["algorithm"]), set(MAIN_BASELINES))
        self.assertEqual(len(research), 5)
        self.assertEqual(set(research["status"]), {"completed"})
        calql = research[research["algorithm"] == "cal_ql"].iloc[0]
        self.assertEqual(int(calql["requested_online_steps"]), 40_000)
        self.assertEqual(int(calql["actual_online_steps"]), 40_001)
        self.assertEqual(int(calql["episode_boundary_overshoot"]), 1)
        self.assertEqual(int(calql["completed_online_transitions"]), 40_001)
        self.assertEqual(int(calql["pending_episode_length"]), 0)
        self.assertEqual(
            int(calql["effective_calql_training_transitions"]), 40_001
        )
        self.assertTrue(
            research.loc[
                research["algorithm"] != "cal_ql",
                "effective_calql_training_transitions",
            ].isna().all()
        )
        self.assertEqual(
            int(
                research.loc[
                    research["algorithm"] == "pessimistic_q_ensemble",
                    "ensemble_size",
                ].iloc[0]
            ),
            5,
        )
        self.assertEqual(
            float(
                research.loc[
                    research["algorithm"] == "pessimistic_q_ensemble",
                    "offline_compute_multiplier",
                ].iloc[0]
            ),
            5.0,
        )

    def test_research_summary_rejects_invalid_completed_calql_accounting(self):
        frame = evaluation_frame(seeds=(0,))
        frame["algorithm"] = "cal_ql"
        frame["run_dir"] = "/tmp/cal_ql_seed_0"
        frame["run_purpose"] = "research_benchmark"
        frame["benchmark_role"] = "main"
        frame["implementation_type"] = (
            "source_aligned_locomotion_adaptation"
        )
        frame["uses_corruption_labels"] = False
        frame["final_window_size"] = 3
        frame["evaluation_corruption"] = "clean"
        frame["requested_online_steps"] = 40_000
        frame["online_budget_semantics"] = CALQL_ONLINE_BUDGET_SEMANTICS
        frame["completed_online_transitions"] = 40_001
        frame["pending_episode_length"] = 1
        frame["effective_calql_training_transitions"] = 40_001
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(
                ReportingValidationError, "pending_episode_length"
            ):
                write_reporting_outputs(
                    frame,
                    Path(directory),
                    strict=False,
                    expected_seeds=[0],
                )

    def test_incomplete_research_cohort_cannot_publish_subset_mean(self):
        frame = evaluation_frame()
        frame["run_purpose"] = "research_benchmark"
        frame["benchmark_role"] = "main"
        frame["implementation_type"] = "source_aligned_port"
        frame["uses_corruption_labels"] = False
        frame["final_window_size"] = 3
        frame["evaluation_corruption"] = "clean"
        frame.loc[frame["seed"] == 1, "run_status"] = "failed"
        with tempfile.TemporaryDirectory() as directory:
            outputs = write_reporting_outputs(
                frame,
                Path(directory),
                strict=False,
                expected_seeds=[0, 1],
            )
            research = pd.read_csv(outputs["research_summary"])

        self.assertEqual(set(research["seed"]), {0})
        self.assertEqual(set(research["status"]), {"cohort_incomplete"})
        self.assertTrue(research["mean"].isna().all())
        self.assertTrue(research["std"].isna().all())

    def test_research_summary_requires_common_interval_and_clean_evaluation(self):
        frame = evaluation_frame(seeds=(0,))
        frame["run_purpose"] = "research_benchmark"
        frame["benchmark_role"] = "main"
        frame["evaluation_corruption"] = "clean"
        mixed = frame.copy()
        mixed["algorithm"] = "riql_naive"
        mixed["run_dir"] = "/tmp/riql_naive_seed_0"
        mixed["eval_period"] = 5_000
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(
                ReportingValidationError, "evaluation intervals"
            ):
                write_reporting_outputs(
                    pd.concat([frame, mixed], ignore_index=True),
                    Path(directory),
                    strict=False,
                    expected_seeds=[0],
                )

        frame["evaluation_corruption"] = "random"
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(
                ReportingValidationError, "require clean evaluation"
            ):
                write_reporting_outputs(
                    frame,
                    Path(directory),
                    strict=False,
                    expected_seeds=[0],
                )


if __name__ == "__main__":
    unittest.main()
