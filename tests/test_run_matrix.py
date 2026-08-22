from __future__ import annotations

import io
import unittest
from contextlib import redirect_stderr

from robust_o2o.config import LOCAL_PROTOCOL
from run_matrix import _validate_args, build_parser, commands


class RunMatrixTest(unittest.TestCase):
    def test_research_matrix_rejects_local_protocol(self):
        parser = build_parser()
        args = parser.parse_args(
            [
                "--run-purpose",
                "research_benchmark",
                "--suite-profile",
                "research_benchmark",
                "--protocol",
                LOCAL_PROTOCOL,
                "--allow-diagnostic-protocol",
            ]
        )
        stderr = io.StringIO()
        with redirect_stderr(stderr), self.assertRaises(SystemExit) as raised:
            _validate_args(parser, args, [])
        self.assertEqual(raised.exception.code, 2)
        self.assertIn("ResearchBenchmarkProtocolError", stderr.getvalue())

    def test_diagnostic_matrix_allows_local_protocol(self):
        parser = build_parser()
        args = parser.parse_args(
            [
                "--run-purpose",
                "diagnostic",
                "--suite-profile",
                "common_budget_diagnostic",
                "--protocol",
                LOCAL_PROTOCOL,
                "--allow-diagnostic-protocol",
            ]
        )
        _validate_args(parser, args, [])
        self.assertEqual(args.protocol, LOCAL_PROTOCOL)

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
                "common_budget_diagnostic",
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
