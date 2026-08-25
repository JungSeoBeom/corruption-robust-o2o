from __future__ import annotations

import unittest

from run_matrix import _validate_args, build_parser, commands


class RunMatrixTest(unittest.TestCase):
    def test_legacy_protocol_controls_are_not_public_options(self):
        parser = build_parser()
        help_text = parser.format_help()
        self.assertNotIn("--protocol", help_text)
        self.assertNotIn("--run-purpose", help_text)
        self.assertNotIn("--allow-diagnostic-protocol", help_text)
        args, passthrough = parser.parse_known_args(["--run-purpose", "diagnostic"])
        with self.assertRaises(SystemExit):
            _validate_args(parser, args, passthrough)

    def test_non_clean_matrix_rejects_empty_corruption_ranges(self):
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
                "observations",
                "--corruption-ranges",
                ",",
                "--suite-profile",
                "common_budget_robustness",
            ]
        )
        with self.assertRaises(SystemExit):
            _validate_args(parser, args, [])

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
                "common_budget_robustness",
            ]
        )
        _validate_args(parser, args, [])
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
        self.assertTrue(all("--protocol" not in command for command in generated))
        self.assertTrue(all("--run-purpose" not in command for command in generated))


if __name__ == "__main__":
    unittest.main()
