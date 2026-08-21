from __future__ import annotations

import unittest
from pathlib import Path
from unittest import mock

from scripts import rpex_fixture_provenance as provenance


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class RuntimePreflightTest(unittest.TestCase):
    def test_exact_strict_stack_passes(self) -> None:
        with (
            mock.patch.object(provenance.platform, "system", return_value="Linux"),
            mock.patch.object(
                provenance.platform, "machine", return_value="x86_64"
            ),
            mock.patch.object(provenance.sys, "version_info", (3, 10, 14)),
            mock.patch.object(provenance.np, "__version__", "1.23.5"),
            mock.patch.object(
                provenance.torch, "__version__", "2.5.1+cu121"
            ),
        ):
            result = provenance.require_strict_runtime(
                allow_diagnostic_mismatch=False
            )
        self.assertTrue(result["passed"])

    def test_runtime_mismatch_is_rejected_by_default(self) -> None:
        with (
            mock.patch.object(provenance.platform, "system", return_value="Darwin"),
            mock.patch.object(provenance.platform, "machine", return_value="arm64"),
            mock.patch.object(provenance.sys, "version_info", (3, 11, 0)),
            mock.patch.object(provenance.np, "__version__", "2.0.0"),
            mock.patch.object(provenance.torch, "__version__", "2.6.0"),
        ):
            with self.assertRaisesRegex(
                RuntimeError, "strict RPEX fixture runtime mismatch"
            ):
                provenance.require_strict_runtime(
                    allow_diagnostic_mismatch=False
                )

    def test_runtime_override_remains_failed_and_auditable(self) -> None:
        with mock.patch.object(
            provenance.platform, "system", return_value="Darwin"
        ):
            result = provenance.require_strict_runtime(
                allow_diagnostic_mismatch=True
            )
        self.assertFalse(result["passed"])
        self.assertFalse(result["checks"]["platform"]["passed"])


class UpstreamPreflightTest(unittest.TestCase):
    def test_dirty_pinned_upstream_is_rejected(self) -> None:
        upstream = Path("/tmp/rpex-upstream").resolve()

        def fake_git(_upstream: Path, *args: str) -> str:
            if args == ("rev-parse", "--show-toplevel"):
                return f"{upstream}\n"
            if args == ("rev-parse", "HEAD"):
                return f"{provenance.PINNED_RPEX_COMMIT}\n"
            if args[0] == "status":
                return " M attack.py\n"
            raise AssertionError(args)

        with (
            mock.patch.object(Path, "is_dir", return_value=True),
            mock.patch.object(provenance, "_git", side_effect=fake_git),
        ):
            with self.assertRaisesRegex(RuntimeError, "checkout is dirty"):
                provenance.require_pinned_clean_upstream(upstream)


class GeneratorClaimBoundaryTest(unittest.TestCase):
    def test_adversarial_generator_is_diagnostic_only(self) -> None:
        source = (
            REPOSITORY_ROOT / "scripts/generate_rpex_adversarial_fixture.py"
        ).read_text(encoding="utf-8")
        self.assertIn(
            '"rpex_adversarial_optimizer_core_diagnostic_v2"', source
        )
        self.assertIn('"end_to_end_verified": False', source)
        self.assertNotIn('"rpex_adversarial_corruption_v2"', source)


if __name__ == "__main__":
    unittest.main()
