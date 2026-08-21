"""Validation for externally generated reproducibility certificates.

Certificates are evidence produced by executable comparisons against pinned
upstream implementations.  This module deliberately does not mint them.  A
receipt is accepted only when its digest is recorded in a separate index and
all repository, fixture, runtime, upstream, command, and timestamp bindings
match the current audit process.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import re
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping


CERTIFICATE_SCHEMA = "robust_o2o.reproducibility_certificate.v1"
CERTIFICATE_INDEX_SCHEMA = "robust_o2o.certificate_index.v1"
CERTIFICATE_DIRECTORY_ENV = "ROBUST_O2O_CERTIFICATE_DIR"
CERTIFICATE_INDEX_FILENAME = "index.json"
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_SOURCE_SUFFIXES = {
    ".cfg",
    ".ini",
    ".ipynb",
    ".json",
    ".md",
    ".py",
    ".sh",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}


class CertificateContextError(RuntimeError):
    """Raised when current repository/runtime identity cannot be established."""


@dataclass(frozen=True)
class CertificateSpec:
    certificate_id: str
    filename: str
    upstream_repository: str
    upstream_commit: str
    required_fixture_paths: tuple[str, ...] = ()
    required_claims: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CertificateContext:
    repository: Mapping[str, Any]
    runtime: Mapping[str, Any]
    fixture_hashes: Mapping[str, str]


@dataclass(frozen=True)
class CertificateValidation:
    certificate_id: str
    status: str
    valid: bool
    detail: str
    path: str | None = None
    receipt_sha256: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _git(root: Path, *arguments: str) -> bytes:
    completed = subprocess.run(
        ("git", "-C", str(root), *arguments),
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        raise CertificateContextError(
            "cannot establish certificate git provenance: "
            + (detail or f"git {' '.join(arguments)} exited {completed.returncode}")
        )
    return completed.stdout


def _tracked_source_sha256(root: Path) -> str:
    tracked = _git(root, "ls-files", "-z")
    digest = hashlib.sha256()
    for raw_relative in sorted(path for path in tracked.split(b"\0") if path):
        relative = raw_relative.decode("utf-8", errors="surrogateescape")
        path = root / relative
        if path.suffix.lower() not in _SOURCE_SUFFIXES or not path.is_file():
            continue
        payload = path.read_bytes()
        digest.update(len(raw_relative).to_bytes(8, "big"))
        digest.update(raw_relative)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def repository_binding(root: str | Path) -> dict[str, Any]:
    candidate = Path(root).resolve()
    status = _git(candidate, "status", "--porcelain=v1", "--untracked-files=all")
    tree_listing = _git(candidate, "ls-tree", "-r", "-z", "--full-tree", "HEAD")
    return {
        "root": str(candidate),
        "commit": _git(candidate, "rev-parse", "HEAD")
        .decode("ascii", errors="strict")
        .strip(),
        "clean": not bool(status),
        "status_sha256": _sha256_bytes(status),
        "tree_object_id": _git(candidate, "rev-parse", "HEAD^{tree}")
        .decode("ascii", errors="strict")
        .strip(),
        "tree_sha256": _sha256_bytes(tree_listing),
        "source_tree_sha256": _tracked_source_sha256(candidate),
    }


def _package_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def runtime_binding() -> dict[str, Any]:
    return {
        "python_executable": str(Path(sys.executable).resolve()),
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "numpy_version": _package_version("numpy"),
        "torch_version": _package_version("torch"),
        "gym_version": _package_version("gym"),
        "d4rl_version": _package_version("d4rl"),
        "mujoco_py_version": _package_version("mujoco-py"),
        "platform_system": platform.system(),
        "platform_release": platform.release(),
        "platform_machine": platform.machine(),
    }


def fixture_hashes(
    root: str | Path, relative_paths: Iterable[str]
) -> dict[str, str]:
    candidate = Path(root).resolve()
    result: dict[str, str] = {}
    for relative in sorted(set(relative_paths)):
        path = candidate / relative
        if path.is_file():
            result[relative] = _sha256_bytes(path.read_bytes())
    return result


def build_certificate_context(
    root: str | Path,
    *,
    fixture_paths: Iterable[str] = (),
) -> CertificateContext:
    return CertificateContext(
        repository=repository_binding(root),
        runtime=runtime_binding(),
        fixture_hashes=fixture_hashes(root, fixture_paths),
    )


def certificate_directory(root: str | Path) -> Path:
    configured = os.environ.get(CERTIFICATE_DIRECTORY_ENV)
    if configured:
        return Path(configured).expanduser().resolve()
    candidate = Path(root).resolve()
    # Receipts cannot be tracked in the tested commit: committing a receipt
    # necessarily changes the commit it attests.  Git's private metadata area
    # is repository-local but invisible to working-tree cleanliness checks.
    raw_git_path = _git(
        candidate, "rev-parse", "--git-path", "robust_o2o-certificates"
    ).decode("utf-8", errors="strict").strip()
    git_path = Path(raw_git_path)
    if not git_path.is_absolute():
        git_path = candidate / git_path
    return git_path.resolve()


def _invalid(
    spec: CertificateSpec,
    status: str,
    detail: str,
    *,
    path: Path | None = None,
    digest: str | None = None,
) -> CertificateValidation:
    return CertificateValidation(
        certificate_id=spec.certificate_id,
        status=status,
        valid=False,
        detail=detail,
        path=None if path is None else str(path),
        receipt_sha256=digest,
    )


def _load_json_object(path: Path, description: str) -> tuple[dict[str, Any] | None, str | None]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        return None, f"cannot read {description}: {exc}"
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        return None, f"invalid {description} JSON: {exc}"
    if not isinstance(payload, dict):
        return None, f"{description} root must be an object"
    return payload, None


def _parse_utc(value: object, field: str) -> tuple[datetime | None, str | None]:
    if not isinstance(value, str):
        return None, f"{field} must be an ISO-8601 timestamp"
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None, f"{field} is not a valid ISO-8601 timestamp"
    if parsed.tzinfo is None:
        return None, f"{field} must include a timezone"
    return parsed.astimezone(timezone.utc), None


def _validate_timestamps(receipt: Mapping[str, Any]) -> str | None:
    verification = receipt.get("verification")
    if not isinstance(verification, dict):
        return "verification must be an object"
    started, error = _parse_utc(
        verification.get("started_at_utc"), "verification.started_at_utc"
    )
    if error:
        return error
    completed, error = _parse_utc(
        verification.get("completed_at_utc"), "verification.completed_at_utc"
    )
    if error:
        return error
    issued, error = _parse_utc(receipt.get("issued_at_utc"), "issued_at_utc")
    if error:
        return error
    assert started is not None and completed is not None and issued is not None
    if completed < started:
        return "verification completed before it started"
    if issued < completed:
        return "certificate was issued before verification completed"
    if issued > datetime.now(timezone.utc) + timedelta(minutes=5):
        return "certificate issuance timestamp is in the future"
    return None


def _mismatch_detail(
    expected: Mapping[str, Any], actual: object, prefix: str
) -> str | None:
    if not isinstance(actual, dict):
        return f"{prefix} must be an object"
    mismatches = {
        key: {"expected": value, "actual": actual.get(key)}
        for key, value in expected.items()
        if actual.get(key) != value
    }
    if mismatches:
        return f"{prefix} binding mismatch: {mismatches}"
    return None


def validate_certificate_receipt(
    spec: CertificateSpec,
    *,
    certificate_dir: str | Path,
    context: CertificateContext,
) -> CertificateValidation:
    """Validate one indexed receipt without ever inferring a missing PASS."""

    directory = Path(certificate_dir).expanduser().resolve()
    index_path = directory / CERTIFICATE_INDEX_FILENAME
    receipt_path = directory / spec.filename
    if not index_path.is_file():
        return _invalid(
            spec,
            "missing",
            f"certificate digest registry is missing: {index_path}",
            path=receipt_path,
        )
    index, error = _load_json_object(index_path, "certificate digest registry")
    if error:
        return _invalid(spec, "invalid", error, path=receipt_path)
    assert index is not None
    if index.get("schema") != CERTIFICATE_INDEX_SCHEMA:
        return _invalid(
            spec, "invalid", "unsupported certificate digest registry schema", path=receipt_path
        )
    receipts = index.get("receipts")
    entry = receipts.get(spec.certificate_id) if isinstance(receipts, dict) else None
    if not isinstance(entry, dict):
        return _invalid(
            spec,
            "missing",
            "certificate digest is not recorded in the registry",
            path=receipt_path,
        )
    if entry.get("filename") != spec.filename:
        return _invalid(
            spec, "invalid", "certificate registry filename mismatch", path=receipt_path
        )
    expected_digest = entry.get("sha256")
    if not isinstance(expected_digest, str) or not _SHA256_PATTERN.fullmatch(
        expected_digest
    ):
        return _invalid(
            spec, "invalid", "certificate registry SHA256 is invalid", path=receipt_path
        )
    if not receipt_path.is_file() or receipt_path.is_symlink():
        return _invalid(
            spec,
            "missing",
            f"certificate receipt is missing or not a regular file: {receipt_path}",
            path=receipt_path,
        )
    try:
        raw = receipt_path.read_bytes()
    except OSError as exc:
        return _invalid(
            spec, "invalid", f"cannot read certificate receipt: {exc}", path=receipt_path
        )
    actual_digest = _sha256_bytes(raw)
    if actual_digest != expected_digest:
        return _invalid(
            spec,
            "invalid",
            "certificate receipt SHA256 does not match the digest registry",
            path=receipt_path,
            digest=actual_digest,
        )
    try:
        receipt = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        return _invalid(
            spec,
            "invalid",
            f"invalid certificate receipt JSON: {exc}",
            path=receipt_path,
            digest=actual_digest,
        )
    if not isinstance(receipt, dict):
        return _invalid(
            spec,
            "invalid",
            "certificate receipt root must be an object",
            path=receipt_path,
            digest=actual_digest,
        )
    if receipt.get("schema") != CERTIFICATE_SCHEMA:
        return _invalid(
            spec, "invalid", "unsupported certificate receipt schema", path=receipt_path, digest=actual_digest
        )
    if receipt.get("certificate_id") != spec.certificate_id:
        return _invalid(
            spec, "invalid", "certificate ID mismatch", path=receipt_path, digest=actual_digest
        )
    if receipt.get("status") != "PASS":
        return _invalid(
            spec,
            "invalid",
            "certificate receipt does not record PASS",
            path=receipt_path,
            digest=actual_digest,
        )

    if context.repository.get("clean") is not True:
        return _invalid(
            spec,
            "stale",
            "current repository working tree is dirty",
            path=receipt_path,
            digest=actual_digest,
        )
    error = _mismatch_detail(context.repository, receipt.get("repository"), "repository")
    if error:
        return _invalid(spec, "stale", error, path=receipt_path, digest=actual_digest)
    error = _mismatch_detail(context.runtime, receipt.get("runtime"), "runtime")
    if error:
        return _invalid(spec, "stale", error, path=receipt_path, digest=actual_digest)

    expected_upstream = {
        "repository": spec.upstream_repository,
        "commit": spec.upstream_commit,
    }
    error = _mismatch_detail(expected_upstream, receipt.get("upstream"), "upstream")
    if error:
        return _invalid(spec, "stale", error, path=receipt_path, digest=actual_digest)
    error = _mismatch_detail(spec.required_claims, receipt.get("claims"), "claims")
    if error:
        return _invalid(spec, "invalid", error, path=receipt_path, digest=actual_digest)
    for platform_field in ("platform_system", "platform_machine"):
        required_platform = spec.required_claims.get(platform_field)
        if (
            required_platform is not None
            and context.runtime.get(platform_field) != required_platform
        ):
            return _invalid(
                spec,
                "stale",
                f"current runtime does not satisfy claim {platform_field}="
                f"{required_platform!r}",
                path=receipt_path,
                digest=actual_digest,
            )

    missing_fixture_paths = sorted(
        set(spec.required_fixture_paths) - set(context.fixture_hashes)
    )
    if missing_fixture_paths:
        return _invalid(
            spec,
            "stale",
            f"required current fixtures are missing: {missing_fixture_paths}",
            path=receipt_path,
            digest=actual_digest,
        )
    expected_fixtures = {
        path: context.fixture_hashes[path] for path in spec.required_fixture_paths
    }
    if receipt.get("fixture_hashes") != expected_fixtures:
        return _invalid(
            spec,
            "stale",
            "fixture hash binding mismatch",
            path=receipt_path,
            digest=actual_digest,
        )

    verification = receipt.get("verification")
    if not isinstance(verification, dict):
        return _invalid(
            spec, "invalid", "verification must be an object", path=receipt_path, digest=actual_digest
        )
    command = verification.get("command")
    if (
        not isinstance(command, list)
        or not command
        or any(not isinstance(argument, str) or not argument for argument in command)
    ):
        return _invalid(
            spec,
            "invalid",
            "verification command must be a non-empty string array",
            path=receipt_path,
            digest=actual_digest,
        )
    if verification.get("returncode") != 0:
        return _invalid(
            spec,
            "invalid",
            "verification command did not return zero",
            path=receipt_path,
            digest=actual_digest,
        )
    for digest_field in ("stdout_sha256", "stderr_sha256"):
        value = verification.get(digest_field)
        if not isinstance(value, str) or not _SHA256_PATTERN.fullmatch(value):
            return _invalid(
                spec,
                "invalid",
                f"verification.{digest_field} is not a SHA256 digest",
                path=receipt_path,
                digest=actual_digest,
            )
    error = _validate_timestamps(receipt)
    if error:
        return _invalid(spec, "invalid", error, path=receipt_path, digest=actual_digest)

    return CertificateValidation(
        certificate_id=spec.certificate_id,
        status="valid",
        valid=True,
        detail=(
            f"receipt_sha256={actual_digest} commit={context.repository['commit']} "
            f"upstream={spec.upstream_commit}"
        ),
        path=str(receipt_path),
        receipt_sha256=actual_digest,
    )


def validate_required_certificates(
    specs: Iterable[CertificateSpec],
    *,
    root: str | Path,
    certificate_dir: str | Path | None = None,
) -> dict[str, CertificateValidation]:
    materialized = tuple(specs)
    all_fixture_paths = {
        path for spec in materialized for path in spec.required_fixture_paths
    }
    context = build_certificate_context(root, fixture_paths=all_fixture_paths)
    directory = (
        certificate_directory(root) if certificate_dir is None else Path(certificate_dir)
    )
    return {
        spec.certificate_id: validate_certificate_receipt(
            spec, certificate_dir=directory, context=context
        )
        for spec in materialized
    }
