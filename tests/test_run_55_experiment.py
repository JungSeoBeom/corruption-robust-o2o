from __future__ import annotations

import unittest
from unittest.mock import patch

from robust_o2o.config import DEFAULT_PROTOCOL
from robust_o2o.fidelity import MAIN_BASELINES, OPTIONAL_BASELINES
from run_55_experiment import (
    ADVERSARIAL_SETTINGS,
    ALGORITHMS,
    DIAGNOSTIC_RANDOM_SETTINGS,
    ENV_NAME,
    SETTINGS,
    STRICT_ADVERSARIAL_SETTINGS,
    STRICT_RANDOM_SETTINGS,
    _validate_args,
    build_parser,
    commands,
    settings_for_suite,
)


class Run55ExperimentTest(unittest.TestCase):
    def test_default_matrix_and_step_schedule(self):
        parser = build_parser()
        args = parser.parse_args([])
        _validate_args(parser, args, ())
        generated = list(commands(args, (), "test_suite"))

        self.assertEqual(ALGORITHMS, MAIN_BASELINES)
        self.assertEqual(len(ALGORITHMS), 3)
        self.assertEqual(
            SETTINGS,
            (
                ("clean", "none"),
                ("random", "observations"),
                ("random", "actions"),
                ("random", "rewards"),
                ("random", "dynamics"),
            ),
        )
        self.assertEqual(len(generated), 5)
        self.assertEqual(args.offline_steps, 500_000)
        self.assertEqual(args.online_steps, 500_000)
        self.assertEqual(args.protocol, DEFAULT_PROTOCOL)
        self.assertEqual(args.run_purpose, "research_benchmark")
        self.assertEqual(args.suite_profile, "research_benchmark")
        self.assertEqual(args.implementation_profile, "research_benchmark")
        self.assertEqual(args.eval_period, 10_000)
        self.assertEqual(args.eval_episodes, 10)
        self.assertEqual(args.final_window_size, 3)

        for command, (corruption, target) in zip(generated, SETTINGS):
            self.assertEqual(command[command.index("--env-name") + 1], ENV_NAME)
            self.assertEqual(command[command.index("--corruption") + 1], corruption)
            self.assertEqual(command[command.index("--corruption-target") + 1], target)
            self.assertEqual(
                command[command.index("--algorithms") + 1], ",".join(ALGORITHMS)
            )
            self.assertEqual(command[command.index("--stage") + 1], "both")
            self.assertEqual(command[command.index("--offline-steps") + 1], "500000")
            self.assertEqual(command[command.index("--online-steps") + 1], "500000")
            self.assertEqual(command[command.index("--eval-period") + 1], "10000")
            self.assertEqual(command[command.index("--eval-episodes") + 1], "10")
            self.assertEqual(command[command.index("--final-window-size") + 1], "3")
            self.assertEqual(
                command[command.index("--comparison-name") + 1], "test_suite"
            )

    def test_halfcheetah_environment_override(self):
        parser = build_parser()
        args = parser.parse_args(
            ["--env-name", "halfcheetah-medium-replay-v2"]
        )
        _validate_args(parser, args, ())
        generated = list(commands(args, (), "halfcheetah_suite"))
        self.assertEqual(len(generated), 5)
        for command in generated:
            self.assertEqual(
                command[command.index("--env-name") + 1],
                "halfcheetah-medium-replay-v2",
            )

    def test_primary_suite_has_no_runs_when_registry_has_no_eligible_algorithm(self):
        parser = build_parser()
        args = parser.parse_args(
            [
                "--suite-profile",
                "primary_research_benchmark",
                "--seeds",
                "0,1,2,3,4",
                "--run-purpose",
                "final_benchmark",
            ]
        )
        with patch(
            "run_55_experiment.strict_final_algorithms", return_value=()
        ), self.assertRaises(SystemExit):
            _validate_args(parser, args, ())

    def test_hypothetical_certified_primary_suite_uses_only_strict_random(self):
        parser = build_parser()
        args = parser.parse_args(
            [
                "--suite-profile",
                "primary_research_benchmark",
                "--seeds",
                "0,1,2,3,4",
                "--run-purpose",
                "final_benchmark",
            ]
        )
        with patch(
            "run_55_experiment.strict_final_algorithms",
            return_value=("rpex", "riql_naive"),
        ), patch(
            "robust_o2o.final_gate.strict_final_algorithms",
            return_value=("rpex", "riql_naive"),
        ):
            _validate_args(parser, args, ())
            generated = list(commands(args, (), "primary_suite"))
        self.assertEqual(len(generated), len(STRICT_RANDOM_SETTINGS))
        for command in generated:
            algorithms = command[command.index("--algorithms") + 1].split(",")
            self.assertNotIn("pqe_shared_actor_approx", algorithms)
            self.assertEqual(algorithms, ["rpex", "riql_naive"])
            self.assertEqual(
                command[command.index("--online-corruption-scale-profile") + 1],
                "rpex_official_code",
            )
            self.assertNotIn("--offline-steps", command)
            self.assertNotIn("--online-steps", command)

        invalid = parser.parse_args(
            ["--run-purpose", "final_benchmark", "--seeds", "0"]
        )
        with patch(
            "run_55_experiment.strict_final_algorithms",
            return_value=("rpex", "riql_naive"),
        ), patch(
            "robust_o2o.final_gate.strict_final_algorithms",
            return_value=("rpex", "riql_naive"),
        ), self.assertRaises(SystemExit):
            _validate_args(parser, invalid, ())

    def test_adversarial_and_all_suites_are_explicit(self):
        parser = build_parser()
        adversarial = parser.parse_args(["--corruption-suite", "adversarial"])
        commands_only = list(commands(adversarial, (), "adv_suite"))
        self.assertEqual(len(commands_only), 4)
        self.assertEqual(settings_for_suite("adversarial"), ADVERSARIAL_SETTINGS)
        self.assertEqual(
            settings_for_suite("random"), DIAGNOSTIC_RANDOM_SETTINGS
        )
        self.assertEqual(len(settings_for_suite("all")), 9)
        self.assertEqual(
            settings_for_suite("adversarial", strict=True),
            STRICT_ADVERSARIAL_SETTINGS,
        )
        self.assertEqual(STRICT_ADVERSARIAL_SETTINGS, ())
        self.assertEqual(
            settings_for_suite("random", strict=True), STRICT_RANDOM_SETTINGS
        )
        self.assertNotIn(
            ("clean", "none"), settings_for_suite("random", strict=True)
        )
        self.assertEqual(settings_for_suite("clean", strict=True), ())
        self.assertEqual(
            settings_for_suite("all", strict=True), STRICT_RANDOM_SETTINGS
        )
        for command, (mode, target) in zip(commands_only, ADVERSARIAL_SETTINGS):
            self.assertEqual(command[command.index("--corruption") + 1], mode)
            self.assertEqual(
                command[command.index("--corruption-target") + 1], target
            )

    def test_optional_baselines_require_explicit_opt_in(self):
        parser = build_parser()
        args = parser.parse_args(
            [
                "--corruption-suite",
                "clean",
                "--optional-baselines",
                "cal_ql_locomotion_adaptation,pqe_shared_actor_approx",
            ]
        )
        _validate_args(parser, args, ())
        command = next(iter(commands(args, (), "optional_suite")))
        self.assertEqual(
            command[command.index("--algorithms") + 1].split(","),
            [*MAIN_BASELINES, *OPTIONAL_BASELINES],
        )

    def test_retired_official_names_fail_instead_of_aliasing(self):
        parser = build_parser()
        for name in ("pessimistic_q_ensemble", "cal_ql"):
            args = parser.parse_args(["--optional-baselines", name])
            with self.assertRaises(SystemExit):
                _validate_args(parser, args, ())

    def test_final_adversarial_suite_rejects_optimizer_core_fixture(self):
        parser = build_parser()
        args = parser.parse_args(
            [
                "--env-name",
                "hopper-medium-replay-v2",
                "--corruption-suite",
                "adversarial",
                "--suite-profile",
                "primary_research_benchmark",
                "--run-purpose",
                "final_benchmark",
                "--seeds",
                "0,1,2,3,4",
            ]
        )
        with self.assertRaises(SystemExit):
            _validate_args(parser, args, ())

    def test_final_all_suite_rejects_uncertified_adversarial_conditions(self):
        parser = build_parser()
        args = parser.parse_args(
            [
                "--corruption-suite",
                "all",
                "--suite-profile",
                "primary_research_benchmark",
                "--run-purpose",
                "final_benchmark",
                "--seeds",
                "0,1,2,3,4",
            ]
        )
        with self.assertRaises(SystemExit):
            _validate_args(parser, args, ())


if __name__ == "__main__":
    unittest.main()
