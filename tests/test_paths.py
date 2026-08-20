from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from robust_o2o.paths import (
    comparison_directory,
    resolve_run_layout,
    results_root_from_output,
)


class ResultPathTest(unittest.TestCase):
    def test_new_comparisons_include_protocol_and_profile_namespaces(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            comparison = comparison_directory(
                str(root),
                "hopper-medium-replay-v2",
                "clean",
                "none",
                "group",
                "rpex_d4rl_v2_legacy",
                "reference",
            )
            self.assertEqual(
                comparison,
                root
                / "comparisons"
                / "rpex_d4rl_v2_legacy"
                / "reference"
                / "hopper-medium-replay-v2"
                / "clean"
                / "none"
                / "group",
            )

    def test_comparisons_do_not_include_protocol_namespace(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            comparison = comparison_directory(
                str(root),
                "hopper-medium-replay-v2",
                "clean",
                "none",
                "started_at",
            )
            self.assertEqual(
                comparison,
                root
                / "comparisons"
                / "hopper-medium-replay-v2"
                / "clean"
                / "none"
                / "started_at",
            )
            self.assertNotIn("local_gymnasium_v4", comparison.parts)

    def test_existing_comparison_runs_directory_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            runs = root / "comparisons/hopper/clean/none/group/runs"
            comparison, resolved_runs = resolve_run_layout(
                str(runs), "hopper", "clean", "none", "ignored"
            )
            self.assertEqual(comparison, runs.parent)
            self.assertEqual(resolved_runs, runs)
            self.assertEqual(results_root_from_output(str(runs)), root)


if __name__ == "__main__":
    unittest.main()
