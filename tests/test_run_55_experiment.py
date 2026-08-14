from __future__ import annotations

import unittest

from robust_o2o.config import LOCAL_PROTOCOL
from run_55_experiment import ALGORITHMS, ENV_NAME, SETTINGS, build_parser, commands


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
        self.assertEqual(args.protocol, LOCAL_PROTOCOL)

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


if __name__ == "__main__":
    unittest.main()
