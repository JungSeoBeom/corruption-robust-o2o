from __future__ import annotations

import unittest

from run_matrix import build_parser, commands


class RunMatrixTest(unittest.TestCase):
    def test_reward_severity_sweep_propagates_each_range(self):
        parser = build_parser()
        args = parser.parse_args(
            [
                "--algorithms",
                "riql_naive",
                "--envs",
                "hopper-medium-replay-v2",
                "--corruptions",
                "random",
                "--targets",
                "rewards",
                "--seeds",
                "0",
                "--corruption-ranges",
                "0,0.5,1,2",
                "--suite-profile",
                "common_budget_diagnostic",
            ]
        )
        generated = list(commands(args, [], "severity"))
        self.assertEqual(len(generated), 4)
        self.assertEqual(
            [command[command.index("--corruption-range") + 1] for command in generated],
            ["0", "0.5", "1", "2"],
        )
        self.assertTrue(
            all(
                command[command.index("--online-corruption-scale-profile") + 1]
                == "dataset_std_scaled_extension"
                for command in generated
            )
        )


if __name__ == "__main__":
    unittest.main()
