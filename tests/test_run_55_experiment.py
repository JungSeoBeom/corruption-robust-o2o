from __future__ import annotations

import unittest
import importlib.util

if importlib.util.find_spec("run_55_experiment") is None:
    raise unittest.SkipTest("run_55_experiment.py is an optional gitignored local runner")

from robust_o2o.config import ALGORITHMS
from robust_o2o.fidelity import MAIN_BASELINES
from run_55_experiment import (
    ADVERSARIAL_SETTINGS,
    CLEAN_SETTINGS,
    ENV_NAME,
    FIXED_SETTINGS,
    RANDOM_SETTINGS,
    _default_experiment_name,
    _validate_args,
    build_parser,
    commands,
)
from run_matrix import (
    _validate_args as validate_matrix_args,
    build_parser as build_matrix_parser,
)


class Run55ExperimentTest(unittest.TestCase):
    def test_default_experiment_name_uses_compact_5x5_label(self):
        name = _default_experiment_name("halfcheetah-medium-replay-v2")
        self.assertRegex(name, r"^halfcheetah_5x5_\d{8}_\d{6}_[0-9a-f]{8}$")

    def test_default_matrix_and_step_schedule(self):
        parser = build_parser()
        args = parser.parse_args([])
        _validate_args(parser, args, ())
        generated = list(commands(args, (), "test_suite"))

        self.assertEqual(
            MAIN_BASELINES,
            (
                "rpex",
                "riql_naive",
                "wsrl",
                "cal_ql",
                "pessimistic_q_ensemble",
            ),
        )
        self.assertEqual(
            FIXED_SETTINGS,
            (
                ("adversarial", "observations"),
                ("adversarial", "actions"),
                ("adversarial", "rewards"),
                ("adversarial", "dynamics"),
                ("clean", "none"),
                ("random", "observations"),
                ("random", "actions"),
                ("random", "rewards"),
                ("random", "dynamics"),
            ),
        )
        self.assertEqual(len(generated), 9)
        self.assertEqual(args.offline_steps, 500_000)
        self.assertEqual(args.online_steps, 500_000)
        self.assertEqual(args.suite_profile, "research_benchmark")
        self.assertEqual(args.implementation_profile, "research_benchmark")
        self.assertEqual(args.eval_period, 10_000)
        self.assertEqual(args.eval_episodes, 10)
        self.assertEqual(args.final_window_size, 3)

        for command, (corruption, target) in zip(generated, FIXED_SETTINGS):
            self.assertEqual(command[command.index("--env-name") + 1], ENV_NAME)
            self.assertEqual(command[command.index("--corruption") + 1], corruption)
            self.assertEqual(command[command.index("--corruption-target") + 1], target)
            self.assertEqual(
                command[command.index("--corruption-profile") + 1], "riql_rpex_code"
            )
            self.assertEqual(
                command[command.index("--online-corruption-scale-profile") + 1],
                "rpex_official_code",
            )
            self.assertEqual(
                command[command.index("--algorithms") + 1],
                ",".join(MAIN_BASELINES),
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
            self.assertNotIn("--protocol", command)
            self.assertNotIn("--run-purpose", command)
            self.assertNotIn("--allow-diagnostic-protocol", command)
            self.assertNotIn("--pqe-member-offline-steps", command)
            self.assertNotIn("--pqe-member-checkpoints", command)
            self.assertNotIn("--cql-alpha-online", command)

    def test_legacy_protocol_controls_are_not_public_options(self):
        parser = build_parser()
        help_text = parser.format_help()
        self.assertNotIn("--protocol", help_text)
        self.assertNotIn("--run-purpose", help_text)
        self.assertNotIn("--allow-diagnostic-protocol", help_text)
        args, passthrough = parser.parse_known_args(
            ["--allow-diagnostic-protocol"]
        )
        with self.assertRaises(SystemExit):
            _validate_args(parser, args, passthrough)

    def test_environment_override(self):
        parser = build_parser()
        args = parser.parse_args(["--env-name", "hopper-medium-replay-v2"])
        _validate_args(parser, args, ())
        generated = list(commands(args, (), "hopper_suite"))
        self.assertEqual(len(generated), 9)
        self.assertTrue(
            all(
                command[command.index("--env-name") + 1]
                == "hopper-medium-replay-v2"
                for command in generated
            )
        )

    def test_fixed_settings_order_and_removed_suite_option(self):
        self.assertEqual(
            FIXED_SETTINGS,
            (*ADVERSARIAL_SETTINGS, *CLEAN_SETTINGS, *RANDOM_SETTINGS),
        )
        parser = build_parser()
        parsed, passthrough = parser.parse_known_args(["--corruption-suite", "random"])
        with self.assertRaises(SystemExit):
            _validate_args(parser, parsed, passthrough)

    def test_comma_space_and_alias_algorithm_inputs_are_canonical(self):
        parser = build_parser()
        args = parser.parse_args(
            [
                "--algorithms",
                "rpex,riql_naive",
                "wsrl",
                "calql",
                "pqe",
            ]
        )
        _validate_args(parser, args, ())
        command = next(iter(commands(args, (), "alias_suite")))
        self.assertEqual(
            command[command.index("--algorithms") + 1].split(","),
            list(MAIN_BASELINES),
        )

    def test_retired_result_only_names_cannot_launch(self):
        parser = build_parser()
        for name in ("cal_ql_locomotion_adaptation", "pqe_shared_actor_approx"):
            args = parser.parse_args(["--algorithms", name])
            with self.assertRaises(SystemExit):
                _validate_args(parser, args, ())

    def test_matrix_defaults_to_all_nine_and_normalizes_aliases(self):
        parser = build_matrix_parser()
        defaults = parser.parse_args([])
        validate_matrix_args(parser, defaults, [])
        self.assertEqual(tuple(defaults.algorithms), ALGORITHMS)

        aliases = parser.parse_args(
            ["--algorithms", "rpex", "calql,pessimistic-q-ensemble"]
        )
        validate_matrix_args(parser, aliases, [])
        self.assertEqual(
            aliases.algorithms,
            ["rpex", "cal_ql", "pessimistic_q_ensemble"],
        )


if __name__ == "__main__":
    unittest.main()
