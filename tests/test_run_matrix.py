from __future__ import annotations

import unittest
import io
from contextlib import redirect_stderr

from run_matrix import _validate_args, build_parser, commands


class RunMatrixTest(unittest.TestCase):
    def test_legacy_protocol_controls_are_not_public_options(self):
        parser = build_parser()
        help_text = parser.format_help()
        self.assertNotIn("--protocol", help_text)
        self.assertNotIn("--run-purpose", help_text)
        self.assertNotIn("--allow-diagnostic-protocol", help_text)
        self.assertNotIn("--evaluation-mode", help_text)
        args, passthrough = parser.parse_known_args(["--run-purpose", "diagnostic"])
        with self.assertRaises(SystemExit):
            _validate_args(parser, args, passthrough)

        args, passthrough = parser.parse_known_args(
            ["--evaluation-mode", "deterministic"]
        )
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
                == "rpex_official_code"
                for command in generated
            )
        )
        self.assertTrue(all("--protocol" not in command for command in generated))
        self.assertTrue(all("--run-purpose" not in command for command in generated))

    def test_default_source_matrix_uses_only_supported_targets(self):
        parser = build_parser()
        args = parser.parse_args(
            ["--algorithms", "riql_naive", "--envs", "hopper-medium-replay-v2"]
        )
        _validate_args(parser, args, [])
        generated = list(commands(args, [], "source"))
        self.assertEqual(len(generated), 9)
        self.assertNotIn("mixed", args.targets)
        self.assertTrue(
            all(
                command[command.index("--corruption-profile") + 1] == "riql_rpex_code"
                and command[command.index("--online-corruption-scale-profile") + 1]
                == "rpex_official_code"
                for command in generated
            )
        )

    def test_source_incompatibilities_fail_before_child_launch(self):
        parser = build_parser()
        for options in (
            ["--corruptions", "random", "--targets", "mixed"],
            ["--online-corruption-scale-profile", "dataset_std_scaled_extension"],
        ):
            with self.subTest(options=options):
                args = parser.parse_args(options)
                error = io.StringIO()
                with redirect_stderr(error), self.assertRaises(SystemExit):
                    _validate_args(parser, args, [])
                self.assertIn("--corruption-profile legacy_extension", error.getvalue())

    def test_explicit_legacy_mixed_preserves_extension_settings(self):
        parser = build_parser()
        args = parser.parse_args(
            [
                "--algorithms", "riql_naive", "--envs", "hopper-medium-replay-v2",
                "--corruptions", "random", "--targets", "mixed",
                "--corruption-profile", "legacy_extension",
            ]
        )
        _validate_args(parser, args, [])
        command = next(commands(args, [], "legacy"))
        self.assertEqual(command[command.index("--corruption-profile") + 1], "legacy_extension")
        self.assertEqual(
            command[command.index("--online-corruption-scale-profile") + 1],
            "dataset_std_scaled_extension",
        )

    def test_passthrough_cannot_override_corruption_profile(self):
        parser = build_parser()
        args = parser.parse_args([])
        with self.assertRaises(SystemExit):
            _validate_args(parser, args, ["--corruption-profile=legacy_extension"])


if __name__ == "__main__":
    unittest.main()
