from __future__ import annotations

import unittest
from dataclasses import replace
from unittest.mock import patch

from robust_o2o.config import ExperimentConfig
from robust_o2o.fidelity import (
    BASELINE_REPRODUCTION_REGISTRY,
    PARITY_STATUSES,
    REPORTING_RULES,
    STRICT_ELIGIBLE_PARITY_STATUSES,
    STRICT_FINAL_SEEDS,
    BaselineReproductionRecord,
    FinalBenchmarkValidationError,
    baseline_record_is_strict_eligible,
    strict_final_algorithms,
)


class FailClosedEligibilityTest(unittest.TestCase):
    def test_parity_status_vocabulary_is_explicit(self):
        self.assertEqual(
            set(PARITY_STATUSES),
            {
                "unverified",
                "formula_only",
                "corruption_only",
                "fixed_batch_partial",
                "end_to_end_verified",
                "official_adapter_verified",
            },
        )
        self.assertEqual(
            STRICT_ELIGIBLE_PARITY_STATUSES,
            {"end_to_end_verified", "official_adapter_verified"},
        )
        with self.assertRaisesRegex(ValueError, "unknown parity_status"):
            BaselineReproductionRecord(
                paper_title="invalid fixture",
                upstream_repository="https://example.invalid/upstream",
                upstream_commit="0" * 40,
                official_task_support="fixture",
                implementation_type="fixture",
                reproduction_status="official_adapter_verified",
                parity_status="verified",
                strict_final_eligible=False,
                remaining_deviation="fixture",
            )

    def test_source_aligned_port_cannot_be_declared_strict(self):
        with self.assertRaisesRegex(ValueError, "strict_final_eligible requires"):
            replace(
                BASELINE_REPRODUCTION_REGISTRY["rpex"],
                parity_status="end_to_end_verified",
                strict_final_eligible=True,
            )

        for algorithm in ("rpex", "riql_naive"):
            with self.subTest(algorithm=algorithm):
                record = BASELINE_REPRODUCTION_REGISTRY[algorithm]
                self.assertEqual(record.reproduction_status, "source_aligned_port")
                self.assertEqual(record.parity_status, "fixed_batch_partial")
                self.assertFalse(record.strict_final_eligible)
                self.assertFalse(baseline_record_is_strict_eligible(record))
        self.assertEqual(strict_final_algorithms(), ())

    def test_wsrl_reporting_rule_is_not_self_labeled_verified(self):
        self.assertFalse(REPORTING_RULES["wsrl"].verified)

    def test_clean_condition_is_not_auto_certified(self):
        resolved = ExperimentConfig(
            "rpex",
            "hopper-medium-replay-v2",
            corruption="clean",
            corruption_target="none",
            implementation_profile="official_code_reference",
            run_purpose="diagnostic",
        ).to_dict()
        self.assertTrue(resolved["riql_config_extension"])
        self.assertFalse(resolved["corruption_fixture_verified"])
        self.assertFalse(resolved["condition_certificate_verified"])
        self.assertFalse(resolved["learner_parity_verified"])
        self.assertFalse(resolved["paper_reproduction_eligible"])
        self.assertFalse(resolved["common_benchmark_eligible"])
        self.assertFalse(resolved["publication_eligible"])

    def test_corruption_only_fixture_cannot_authorize_publication(self):
        resolved = ExperimentConfig(
            "rpex",
            "hopper-medium-replay-v2",
            corruption="random",
            corruption_target="observations",
            implementation_profile="official_code_reference",
            run_purpose="diagnostic",
        ).to_dict()
        self.assertTrue(resolved["corruption_fixture_verified"])
        self.assertFalse(resolved["condition_certificate_verified"])
        self.assertFalse(resolved["learner_parity_verified"])
        self.assertFalse(resolved["paper_reproduction_eligible"])
        self.assertFalse(resolved["common_benchmark_eligible"])
        self.assertFalse(resolved["publication_eligible"])

    def test_adversarial_core_fixture_is_diagnostic_not_end_to_end(self):
        resolved = ExperimentConfig(
            "rpex",
            "hopper-medium-replay-v2",
            corruption="adversarial",
            corruption_target="observations",
            implementation_profile="official_code_reference",
            run_purpose="diagnostic",
        ).to_dict()
        self.assertEqual(
            resolved["condition_status"],
            "adversarial_optimizer_core_diagnostic",
        )
        self.assertTrue(resolved["corruption_fixture_verified"])
        self.assertFalse(resolved["condition_certificate_verified"])
        self.assertTrue(resolved["config_extension_active"])
        self.assertFalse(resolved["publication_eligible"])

    def test_runtime_evidence_is_never_self_attested_by_config(self):
        verified_record = replace(
            BASELINE_REPRODUCTION_REGISTRY["rpex"],
            implementation_type="pinned upstream subprocess adapter",
            reproduction_status="official_adapter_verified",
            parity_status="official_adapter_verified",
            strict_final_eligible=True,
            remaining_deviation="none in fixture",
        )
        with patch.dict(
            BASELINE_REPRODUCTION_REGISTRY,
            {"rpex": verified_record},
        ):
            config = ExperimentConfig(
                "rpex",
                "hopper-medium-replay-v2",
                corruption="random",
                corruption_target="observations",
                suite_profile="primary_research_benchmark",
                run_purpose="final_benchmark",
                benchmark_seed_set=STRICT_FINAL_SEEDS,
            )
            config._controller_seed_cohort_attested = True
            resolved = config.to_dict()

        self.assertTrue(resolved["learner_parity_verified"])
        self.assertTrue(resolved["reporting_rule_verified"])
        for field in (
            "condition_certificate_verified",
            "strict_runtime_fixture_verified",
            "strict_environment_preflight_verified",
            "save_resume_certificate_verified",
            "audit_receipt_verified",
            "repository_worktree_clean",
            "dataset_hash_recorded",
            "required_checkpoint_hashes_verified",
        ):
            with self.subTest(field=field):
                self.assertFalse(resolved[field])
        self.assertFalse(resolved["paper_reproduction_eligible"])
        self.assertFalse(resolved["common_benchmark_eligible"])
        self.assertFalse(resolved["publication_eligible"])

    def test_strict_clean_cannot_fall_back_to_extension_row(self):
        verified_record = replace(
            BASELINE_REPRODUCTION_REGISTRY["rpex"],
            implementation_type="pinned upstream subprocess adapter",
            reproduction_status="official_adapter_verified",
            parity_status="official_adapter_verified",
            strict_final_eligible=True,
            remaining_deviation="none in fixture",
        )
        with patch.dict(
            BASELINE_REPRODUCTION_REGISTRY,
            {"rpex": verified_record},
        ):
            with self.assertRaisesRegex(
                FinalBenchmarkValidationError,
                "no official row",
            ):
                ExperimentConfig(
                    "rpex",
                    "hopper-medium-replay-v2",
                    corruption="clean",
                    corruption_target="none",
                    suite_profile="primary_research_benchmark",
                    run_purpose="final_benchmark",
                    benchmark_seed_set=STRICT_FINAL_SEEDS,
                )


if __name__ == "__main__":
    unittest.main()
