from __future__ import annotations

import unittest

from robust_o2o.config import DEFAULT_PROTOCOL
from run_55_experiment import (
    ADVERSARIAL_SETTINGS,
    ALGORITHMS,
    ENV_NAME,
    SETTINGS,
    STRICT_ADVERSARIAL_SETTINGS,
    _validate_args,
    build_parser,
    commands,
    settings_for_suite,
)


class Run55ExperimentTest(unittest.TestCase):
    def test_default_matrix_and_step_schedule(self):
        parser = build_parser()
        args = parser.parse_args([])
        generated = list(commands(args, (), "test_suite"))

        self.assertEqual(len(ALGORITHMS), 5)
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
            self.assertEqual(
                command[command.index("--comparison-name") + 1], "test_suite"
            )

    def test_halfcheetah_environment_override(self):
        parser = build_parser()
        args = parser.parse_args(
            ["--env-name", "halfcheetah-medium-replay-v2"]
        )
        generated = list(commands(args, (), "halfcheetah_suite"))
        self.assertEqual(len(generated), 5)
        for command in generated:
            self.assertEqual(
                command[command.index("--env-name") + 1],
                "halfcheetah-medium-replay-v2",
            )

    def test_primary_suite_excludes_pqe_and_final_requires_five_seeds(self):
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
        _validate_args(parser, args, ())
        generated = list(commands(args, (), "primary_suite"))
        for command in generated:
            algorithms = command[command.index("--algorithms") + 1].split(",")
            self.assertNotIn("pessimistic_q_ensemble", algorithms)
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
        with self.assertRaises(SystemExit):
            _validate_args(parser, invalid, ())

    def test_adversarial_and_all_suites_are_explicit(self):
        parser = build_parser()
        adversarial = parser.parse_args(["--corruption-suite", "adversarial"])
        commands_only = list(commands(adversarial, (), "adv_suite"))
        self.assertEqual(len(commands_only), 4)
        self.assertEqual(settings_for_suite("adversarial"), ADVERSARIAL_SETTINGS)
        self.assertEqual(len(settings_for_suite("all")), 9)
        self.assertEqual(
            settings_for_suite("adversarial", strict=True),
            STRICT_ADVERSARIAL_SETTINGS,
        )
        self.assertEqual(len(settings_for_suite("all", strict=True)), 6)
        for command, (mode, target) in zip(commands_only, ADVERSARIAL_SETTINGS):
            self.assertEqual(command[command.index("--corruption") + 1], mode)
            self.assertEqual(
                command[command.index("--corruption-target") + 1], target
            )

    def test_final_adversarial_suite_rejects_non_hopper_fixture_scope(self):
        parser = build_parser()
        args = parser.parse_args(
            [
                "--env-name",
                "halfcheetah-medium-replay-v2",
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


if __name__ == "__main__":
    unittest.main()
