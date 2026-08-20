from __future__ import annotations

import unittest

from robust_o2o.config import DEFAULT_PROTOCOL
from run_55_experiment import (
    ALGORITHMS,
    ENV_NAME,
    SETTINGS,
    _validate_args,
    build_parser,
    commands,
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
            self.assertEqual(len(algorithms), 4)
            self.assertEqual(
                command[command.index("--online-corruption-scale-profile") + 1],
                "rpex_official_code",
            )

        invalid = parser.parse_args(
            ["--run-purpose", "final_benchmark", "--seeds", "0"]
        )
        with self.assertRaises(SystemExit):
            _validate_args(parser, invalid, ())


if __name__ == "__main__":
    unittest.main()
