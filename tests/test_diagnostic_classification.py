from __future__ import annotations

import unittest

from scripts.diagnose_training import classify_diagnostic_summary


def evidence(**overrides):
    result = {
        "initial_deterministic_return": 10.0,
        "final_deterministic_return": 12.0,
        "return_delta": 2.0,
        "actor_parameter_delta": 0.1,
        "critic_parameter_delta": 0.2,
        "completed_actor_updates": 150,
        "completed_critic_updates": 150,
    }
    result.update(overrides)
    return result


class DiagnosticClassificationTest(unittest.TestCase):
    def classify(self, **overrides):
        return classify_diagnostic_summary(evidence(**overrides))[0]

    def test_no_optimizer_updates(self):
        self.assertEqual(
            self.classify(completed_actor_updates=0, completed_critic_updates=0),
            "FAIL_NO_PARAMETER_UPDATE",
        )

    def test_optimizer_calls_with_zero_parameter_change(self):
        self.assertEqual(
            self.classify(actor_parameter_delta=0.0, critic_parameter_delta=0.0),
            "FAIL_NO_PARAMETER_UPDATE",
        )

    def test_finite_updates_but_short_run_is_inconclusive(self):
        self.assertEqual(
            self.classify(completed_actor_updates=20, completed_critic_updates=20),
            "INCONCLUSIVE_SHORT_RUN",
        )

    def test_sufficient_updates_with_improved_return(self):
        self.assertEqual(self.classify(), "PASS_LEARNING_SIGNAL")

    def test_sufficient_updates_with_flat_return(self):
        self.assertEqual(
            self.classify(
                final_deterministic_return=10.0,
                return_delta=0.0,
            ),
            "FAIL_NO_RETURN_IMPROVEMENT",
        )

    def test_nonfinite_metric(self):
        self.assertEqual(
            self.classify(actor_parameter_delta=float("nan")),
            "FAIL_NUMERICAL",
        )

    def test_action_or_replay_invariant_violation(self):
        classification, _ = classify_diagnostic_summary(
            evidence(), invariant_violated=True
        )
        self.assertEqual(classification, "FAIL_CODE_INVARIANT")


if __name__ == "__main__":
    unittest.main()
