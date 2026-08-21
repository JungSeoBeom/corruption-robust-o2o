from __future__ import annotations

import hashlib
import json
import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import Mock, patch

from robust_o2o.certificates import (
    CERTIFICATE_INDEX_SCHEMA,
    CERTIFICATE_SCHEMA,
    CertificateContext,
    CertificateSpec,
    validate_certificate_receipt,
)
from robust_o2o.final_gate import (
    AUDIT_CONTEXT_ENV,
    AUDIT_RECEIPT_ENV,
    AUDIT_RECEIPT_SHA256_ENV,
    RECEIPT_SCHEMA,
    FinalAuditGateError,
    _audit_result_is_ready,
    _receipt_bytes,
    _validated_inherited_receipt,
    audit_context_token,
    extract_verified_audit_attestation,
    require_final_benchmark_audit,
)


class CertificateReceiptTest(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        fixture = self.root / "fixture.json"
        fixture.write_text('{"upstream": true}\n', encoding="utf-8")
        self.fixture_sha256 = hashlib.sha256(fixture.read_bytes()).hexdigest()
        self.spec = CertificateSpec(
            "example_parity",
            "example_parity.json",
            "https://example.invalid/upstream",
            "a" * 40,
            ("fixture.json",),
        )
        self.repository = {
            "root": "/repo",
            "commit": "b" * 40,
            "clean": True,
            "status_sha256": hashlib.sha256(b"").hexdigest(),
            "tree_sha256": "c" * 40,
            "source_tree_sha256": "d" * 64,
        }
        self.runtime = {
            "python_executable": "/python",
            "python_version": "3.10.0",
            "python_implementation": "CPython",
            "numpy_version": "1.23.5",
            "torch_version": "2.5.1",
            "gym_version": "0.23.1",
            "d4rl_version": "1.1",
            "mujoco_py_version": "2.1.2.14",
            "platform_system": "Linux",
            "platform_release": "test",
            "platform_machine": "x86_64",
        }

    def tearDown(self) -> None:
        self.directory.cleanup()

    def _context(self, **repository_updates: object) -> CertificateContext:
        return CertificateContext(
            repository={**self.repository, **repository_updates},
            runtime=self.runtime,
            fixture_hashes={"fixture.json": self.fixture_sha256},
        )

    def _receipt(self) -> dict[str, object]:
        completed = datetime.now(timezone.utc) - timedelta(minutes=1)
        return {
            "schema": CERTIFICATE_SCHEMA,
            "certificate_id": self.spec.certificate_id,
            "status": "PASS",
            "repository": self.repository,
            "upstream": {
                "repository": self.spec.upstream_repository,
                "commit": self.spec.upstream_commit,
            },
            "claims": {},
            "fixture_hashes": {"fixture.json": self.fixture_sha256},
            "runtime": self.runtime,
            "verification": {
                "command": ["python", "verify_upstream.py"],
                "returncode": 0,
                "started_at_utc": (completed - timedelta(minutes=1)).isoformat(),
                "completed_at_utc": completed.isoformat(),
                "stdout_sha256": hashlib.sha256(b"passed").hexdigest(),
                "stderr_sha256": hashlib.sha256(b"").hexdigest(),
            },
            "issued_at_utc": completed.isoformat(),
        }

    def _write(self, receipt: dict[str, object], *, digest: str | None = None) -> None:
        raw = (json.dumps(receipt, sort_keys=True) + "\n").encode("utf-8")
        (self.root / self.spec.filename).write_bytes(raw)
        index = {
            "schema": CERTIFICATE_INDEX_SCHEMA,
            "receipts": {
                self.spec.certificate_id: {
                    "filename": self.spec.filename,
                    "sha256": digest or hashlib.sha256(raw).hexdigest(),
                }
            },
        }
        (self.root / "index.json").write_text(
            json.dumps(index), encoding="utf-8"
        )

    def test_fully_bound_indexed_receipt_is_valid(self) -> None:
        self._write(self._receipt())
        result = validate_certificate_receipt(
            self.spec,
            certificate_dir=self.root,
            context=self._context(),
        )
        self.assertTrue(result.valid)
        self.assertEqual(result.status, "valid")

    def test_missing_receipt_and_registry_are_not_inferred_as_pass(self) -> None:
        result = validate_certificate_receipt(
            self.spec,
            certificate_dir=self.root,
            context=self._context(),
        )
        self.assertFalse(result.valid)
        self.assertEqual(result.status, "missing")

    def test_digest_tampering_is_invalid(self) -> None:
        self._write(self._receipt(), digest="0" * 64)
        result = validate_certificate_receipt(
            self.spec,
            certificate_dir=self.root,
            context=self._context(),
        )
        self.assertFalse(result.valid)
        self.assertEqual(result.status, "invalid")
        self.assertIn("digest registry", result.detail)

    def test_dirty_current_tree_is_stale_even_if_receipt_claims_clean(self) -> None:
        self._write(self._receipt())
        result = validate_certificate_receipt(
            self.spec,
            certificate_dir=self.root,
            context=self._context(clean=False),
        )
        self.assertFalse(result.valid)
        self.assertEqual(result.status, "stale")
        self.assertIn("dirty", result.detail)

    def test_commit_or_source_change_invalidates_receipt(self) -> None:
        self._write(self._receipt())
        for update in (
            {"commit": "e" * 40},
            {"source_tree_sha256": "f" * 64},
        ):
            with self.subTest(update=update):
                result = validate_certificate_receipt(
                    self.spec,
                    certificate_dir=self.root,
                    context=self._context(**update),
                )
                self.assertFalse(result.valid)
                self.assertEqual(result.status, "stale")
                self.assertIn("repository binding mismatch", result.detail)

    def test_fixture_or_runtime_change_invalidates_receipt(self) -> None:
        self._write(self._receipt())
        changed_fixture = CertificateContext(
            repository=self.repository,
            runtime=self.runtime,
            fixture_hashes={"fixture.json": "0" * 64},
        )
        fixture_result = validate_certificate_receipt(
            self.spec,
            certificate_dir=self.root,
            context=changed_fixture,
        )
        self.assertFalse(fixture_result.valid)
        self.assertEqual(fixture_result.status, "stale")
        self.assertIn("fixture hash", fixture_result.detail)

        changed_runtime = CertificateContext(
            repository=self.repository,
            runtime={**self.runtime, "torch_version": "different"},
            fixture_hashes={"fixture.json": self.fixture_sha256},
        )
        runtime_result = validate_certificate_receipt(
            self.spec,
            certificate_dir=self.root,
            context=changed_runtime,
        )
        self.assertFalse(runtime_result.valid)
        self.assertEqual(runtime_result.status, "stale")
        self.assertIn("runtime binding mismatch", runtime_result.detail)

    def test_nonzero_verification_command_is_invalid(self) -> None:
        receipt = self._receipt()
        receipt["verification"]["returncode"] = 1  # type: ignore[index]
        self._write(receipt)
        result = validate_certificate_receipt(
            self.spec,
            certificate_dir=self.root,
            context=self._context(),
        )
        self.assertFalse(result.valid)
        self.assertEqual(result.status, "invalid")
        self.assertIn("return zero", result.detail)

    def test_required_capability_claim_cannot_be_omitted(self) -> None:
        spec = CertificateSpec(
            self.spec.certificate_id,
            self.spec.filename,
            self.spec.upstream_repository,
            self.spec.upstream_commit,
            self.spec.required_fixture_paths,
            required_claims={"scope": "full_end_to_end"},
        )
        self._write(self._receipt())
        result = validate_certificate_receipt(
            spec,
            certificate_dir=self.root,
            context=self._context(),
        )
        self.assertFalse(result.valid)
        self.assertEqual(result.status, "invalid")
        self.assertIn("claims binding mismatch", result.detail)


class FinalGateStatusTest(unittest.TestCase):
    @staticmethod
    def _ready_result() -> dict[str, object]:
        capabilities = {
            "rpex_riql_learner": ["learner"],
            "five_baseline_learners": ["other-baselines"],
            "random_corruption": ["random"],
            "save_resume": ["resume"],
            "strict_environment": ["environment"],
        }
        verified_ids = [
            certificate_id
            for required_ids in capabilities.values()
            for certificate_id in required_ids
        ]
        return {
            "reproducibility_audit": "PASS",
            "rpex_riql_eligible_subset_status": "READY",
            "five_baseline_status": "READY",
            "random_corruption_status": "READY",
            "adversarial_corruption_status": "EXCLUDED",
            "save_resume_status": "READY",
            "strict_environment_status": "READY",
            "final_benchmark_status": "READY",
            "verified_certificate_ids": verified_ids,
            "certificate_statuses": {
                certificate_id: {
                    "certificate_id": certificate_id,
                    "status": "valid",
                    "valid": True,
                }
                for certificate_id in verified_ids
            },
            "certificate_capabilities": {
                capability: {
                    "required_certificate_ids": required_ids,
                    "verified": True,
                }
                for capability, required_ids in capabilities.items()
            },
        }

    def test_final_gate_requires_every_named_status(self) -> None:
        ready = self._ready_result()
        self.assertTrue(_audit_result_is_ready(ready))
        for field in tuple(ready):
            if field == "adversarial_corruption_status":
                continue
            with self.subTest(field=field):
                changed = dict(ready)
                changed[field] = "NOT READY"
                self.assertFalse(_audit_result_is_ready(changed))

    def test_final_gate_does_not_accept_missing_component_status(self) -> None:
        result = self._ready_result()
        del result["save_resume_status"]
        self.assertFalse(_audit_result_is_ready(result))

    def test_attestation_exports_only_cross_checked_capabilities(self) -> None:
        context = {"repository_clean": True}
        receipt = {
            "schema": RECEIPT_SCHEMA,
            "audit_returncode": 0,
            "context_token": audit_context_token(context),
            "context": context,
            "audit_result": self._ready_result(),
        }
        attestation = extract_verified_audit_attestation(
            receipt, current_context=context
        )
        self.assertTrue(attestation["audit_receipt_verified"])
        self.assertTrue(
            attestation["certificate_capabilities"]["rpex_riql_learner"][
                "verified"
            ]
        )

        receipt["audit_result"]["verified_certificate_ids"] = []  # type: ignore[index]
        with self.assertRaises(FinalAuditGateError):
            extract_verified_audit_attestation(receipt, current_context=context)

    def test_dirty_tree_is_rejected_before_audit_subprocess(self) -> None:
        completed = Mock()
        with (
            patch(
                "robust_o2o.final_gate.audit_context",
                return_value={"repository_clean": False},
            ),
            patch("robust_o2o.final_gate.subprocess.run", completed),
        ):
            with self.assertRaisesRegex(FinalAuditGateError, "clean repository"):
                require_final_benchmark_audit("final_benchmark")
        completed.assert_not_called()

    def test_invalid_or_stale_inherited_audit_receipt_is_rejected(self) -> None:
        context = {"repository_clean": True, "git_head": "a" * 40}
        token = audit_context_token(context)
        receipt = {
            "schema": RECEIPT_SCHEMA,
            "issued_at_utc": datetime.now(timezone.utc).isoformat(),
            "origin_pid": os.getpid(),
            "context_token": token,
            "context": context,
            "audit_command": ["python", "audit_reproducibility.py", "--json"],
            "audit_returncode": 0,
            "audit_stdout_sha256": "0" * 64,
            "audit_stderr_sha256": "0" * 64,
            "audit_result": self._ready_result(),
        }
        with tempfile.TemporaryDirectory(prefix="robust_o2o_final_audit_") as directory:
            os.chmod(directory, 0o700)
            path = Path(directory) / "receipt.json"
            content = _receipt_bytes(receipt)
            path.write_bytes(content)
            os.chmod(path, 0o600)
            environment = {
                AUDIT_CONTEXT_ENV: token,
                AUDIT_RECEIPT_ENV: str(path),
                AUDIT_RECEIPT_SHA256_ENV: hashlib.sha256(content).hexdigest(),
            }
            with patch.dict(os.environ, environment, clear=False):
                accepted, rejection = _validated_inherited_receipt(context, token)
                self.assertIsNotNone(accepted)
                self.assertIsNone(rejection)

                os.environ[AUDIT_RECEIPT_SHA256_ENV] = "f" * 64
                accepted, rejection = _validated_inherited_receipt(context, token)
                self.assertIsNone(accepted)
                self.assertIn("digest mismatch", rejection)

            stale_context = {**context, "git_head": "b" * 40}
            stale_token = audit_context_token(stale_context)
            stale_environment = {
                AUDIT_CONTEXT_ENV: stale_token,
                AUDIT_RECEIPT_ENV: str(path),
                AUDIT_RECEIPT_SHA256_ENV: hashlib.sha256(content).hexdigest(),
            }
            with patch.dict(os.environ, stale_environment, clear=False):
                accepted, rejection = _validated_inherited_receipt(
                    stale_context, stale_token
                )
                self.assertIsNone(accepted)
                self.assertIn("provenance", rejection)


if __name__ == "__main__":
    unittest.main()
