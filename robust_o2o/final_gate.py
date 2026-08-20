"""Fail-closed launcher gate for publication-facing benchmark runs.

The strict audit is intentionally run by launchers rather than by the training
loop.  A successful launcher leaves a short-lived receipt in the system temp
directory and exports both its digest and a context token.  Descendant
launchers may reuse that receipt only while the repository/worktree,
interpreter, platform, and audit source are unchanged.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import stat
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .fidelity import strict_final_algorithms


ROOT = Path(__file__).resolve().parents[1]
AUDIT_SCRIPT = ROOT / "scripts" / "audit_reproducibility.py"
AUDIT_CONTEXT_ENV = "ROBUST_O2O_FINAL_AUDIT_CONTEXT"
AUDIT_RECEIPT_ENV = "ROBUST_O2O_FINAL_AUDIT_RECEIPT"
AUDIT_RECEIPT_SHA256_ENV = "ROBUST_O2O_FINAL_AUDIT_RECEIPT_SHA256"
RECEIPT_SCHEMA = "robust_o2o.final_audit_receipt.v1"
_RECEIPT_DIRECTORY_PREFIX = "robust_o2o_final_audit_"


class FinalAuditGateError(RuntimeError):
    """Raised before result directories exist when the strict audit is absent."""


class ResearchLabelContractError(ValueError):
    """Raised when a launcher could mislabel a diagnostic run as research."""


def _run_git(*arguments: str) -> bytes:
    result = subprocess.run(
        ("git", "-C", str(ROOT), *arguments),
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise FinalAuditGateError(
            "cannot establish final-audit git provenance: "
            + (detail or f"git {' '.join(arguments)} exited {result.returncode}")
        )
    return result.stdout


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _untracked_content_digest() -> str:
    paths = _run_git("ls-files", "--others", "--exclude-standard", "-z")
    digest = hashlib.sha256()
    for raw_path in paths.split(b"\0"):
        if not raw_path:
            continue
        relative = raw_path.decode("utf-8", errors="surrogateescape")
        path = ROOT / relative
        digest.update(len(raw_path).to_bytes(8, "big"))
        digest.update(raw_path)
        if path.is_symlink():
            payload = os.readlink(path).encode("utf-8", errors="surrogateescape")
            kind = b"symlink"
        elif path.is_file():
            payload = path.read_bytes()
            kind = b"file"
        else:
            payload = b"missing-or-special"
            kind = b"other"
        digest.update(kind)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def audit_context() -> dict[str, Any]:
    """Return the independently reproducible context covered by an audit."""

    if not AUDIT_SCRIPT.is_file():
        raise FinalAuditGateError(f"strict audit script is missing: {AUDIT_SCRIPT}")
    head = _run_git("rev-parse", "HEAD").decode("ascii", errors="strict").strip()
    diff = _run_git("diff", "--binary", "HEAD", "--", ".")
    status = _run_git(
        "status", "--porcelain=v1", "--untracked-files=all"
    )
    return {
        "repository_root": str(ROOT.resolve()),
        "git_head": head,
        "git_status_sha256": _sha256_bytes(status),
        "git_worktree_diff_sha256": _sha256_bytes(diff),
        "git_untracked_content_sha256": _untracked_content_digest(),
        "audit_script_sha256": _sha256_bytes(AUDIT_SCRIPT.read_bytes()),
        "python_executable": str(Path(sys.executable).resolve()),
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform_system": platform.system(),
        "platform_release": platform.release(),
        "platform_machine": platform.machine(),
    }


def audit_context_token(context: dict[str, Any] | None = None) -> str:
    payload = context if context is not None else audit_context()
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _receipt_bytes(receipt: dict[str, Any]) -> bytes:
    return (
        json.dumps(receipt, sort_keys=True, indent=2, ensure_ascii=False) + "\n"
    ).encode("utf-8")


def _origin_is_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _validated_inherited_receipt(
    context: dict[str, Any], token: str
) -> tuple[dict[str, Any] | None, str | None]:
    inherited_token = os.environ.get(AUDIT_CONTEXT_ENV)
    raw_path = os.environ.get(AUDIT_RECEIPT_ENV)
    inherited_sha256 = os.environ.get(AUDIT_RECEIPT_SHA256_ENV)
    if not any((inherited_token, raw_path, inherited_sha256)):
        return None, None
    if not all((inherited_token, raw_path, inherited_sha256)):
        return None, "incomplete inherited audit receipt environment"
    if inherited_token != token:
        return None, "audit context token does not match the current context"

    path = Path(raw_path).expanduser()
    try:
        resolved = path.resolve(strict=True)
        temp_root = Path(tempfile.gettempdir()).resolve(strict=True)
        if resolved.parent.parent != temp_root:
            return None, "audit receipt is outside the system temp directory"
        if not resolved.parent.name.startswith(_RECEIPT_DIRECTORY_PREFIX):
            return None, "audit receipt directory has an invalid name"
        directory_stat = resolved.parent.stat()
        receipt_stat = resolved.stat()
        if hasattr(os, "getuid"):
            uid = os.getuid()
            if directory_stat.st_uid != uid or receipt_stat.st_uid != uid:
                return None, "audit receipt is not owned by the current user"
        if stat.S_IMODE(directory_stat.st_mode) & 0o077:
            return None, "audit receipt directory permissions are too broad"
        if stat.S_IMODE(receipt_stat.st_mode) & 0o077:
            return None, "audit receipt permissions are too broad"
        content = resolved.read_bytes()
    except (OSError, ValueError) as exc:
        return None, f"cannot read inherited audit receipt: {exc}"

    if _sha256_bytes(content) != inherited_sha256:
        return None, "audit receipt content digest mismatch"
    try:
        receipt = json.loads(content)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        return None, f"invalid audit receipt JSON: {exc}"
    if receipt.get("schema") != RECEIPT_SCHEMA:
        return None, "unsupported audit receipt schema"
    if receipt.get("context_token") != token or receipt.get("context") != context:
        return None, "audit receipt provenance does not match the current context"
    if receipt.get("audit_returncode") != 0:
        return None, "audit receipt records a failed audit"
    result = receipt.get("audit_result")
    if not isinstance(result, dict) or (
        result.get("reproducibility_audit") != "PASS"
        or result.get("final_benchmark_status") != "READY"
    ):
        return None, "audit receipt does not record PASS/READY"
    try:
        origin_pid = int(receipt["origin_pid"])
    except (KeyError, TypeError, ValueError):
        return None, "audit receipt has no valid origin PID"
    if not _origin_is_alive(origin_pid):
        return None, "audit receipt origin process is no longer alive"
    return receipt, None


def _clear_inherited_receipt() -> None:
    for name in (
        AUDIT_CONTEXT_ENV,
        AUDIT_RECEIPT_ENV,
        AUDIT_RECEIPT_SHA256_ENV,
    ):
        os.environ.pop(name, None)


def _write_receipt(receipt: dict[str, Any], token: str) -> tuple[Path, str]:
    directory = Path(tempfile.mkdtemp(prefix=_RECEIPT_DIRECTORY_PREFIX))
    os.chmod(directory, 0o700)
    path = directory / "receipt.json"
    content = _receipt_bytes(receipt)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    digest = _sha256_bytes(content)
    os.environ[AUDIT_CONTEXT_ENV] = token
    os.environ[AUDIT_RECEIPT_ENV] = str(path)
    os.environ[AUDIT_RECEIPT_SHA256_ENV] = digest
    return path, digest


def require_final_benchmark_audit(
    run_purpose: str,
    *,
    dry_run: bool = False,
) -> dict[str, Any] | None:
    """Run or reuse the strict audit before any benchmark artifact is created."""

    if run_purpose != "final_benchmark":
        return None
    if dry_run:
        print(
            "FINAL_BENCHMARK_AUDIT_GATE: SKIPPED (dry-run; no training or "
            "publication artifact is authorized)",
            flush=True,
        )
        return None

    context = audit_context()
    token = audit_context_token(context)
    inherited, rejection = _validated_inherited_receipt(context, token)
    if inherited is not None:
        print("FINAL_BENCHMARK_AUDIT_GATE: PASS (inherited receipt)", flush=True)
        print(f"FINAL_BENCHMARK_AUDIT_CONTEXT: {token}", flush=True)
        print(
            f"FINAL_BENCHMARK_AUDIT_RECEIPT: {os.environ[AUDIT_RECEIPT_ENV]}",
            flush=True,
        )
        return inherited
    if rejection:
        print(
            f"FINAL_BENCHMARK_AUDIT_RECEIPT_REJECTED: {rejection}; rerunning audit",
            file=sys.stderr,
            flush=True,
        )
    _clear_inherited_receipt()

    command = [sys.executable, str(AUDIT_SCRIPT), "--json"]
    print(
        "FINAL_BENCHMARK_AUDIT_GATE: RUNNING " + " ".join(command),
        flush=True,
    )
    completed = subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    stdout = completed.stdout.strip()
    stderr = completed.stderr.strip()
    try:
        audit_result = json.loads(stdout)
    except json.JSONDecodeError as exc:
        audit_result = None
        parse_error = str(exc)
    else:
        parse_error = None
    passed = (
        completed.returncode == 0
        and isinstance(audit_result, dict)
        and audit_result.get("reproducibility_audit") == "PASS"
        and audit_result.get("final_benchmark_status") == "READY"
    )
    if not passed:
        details = [
            f"returncode={completed.returncode}",
            f"context={token}",
        ]
        if parse_error:
            details.append(f"invalid audit JSON: {parse_error}")
        if stdout:
            details.append(f"stdout:\n{stdout}")
        if stderr:
            details.append(f"stderr:\n{stderr}")
        raise FinalAuditGateError(
            "final benchmark audit failed before output creation; "
            + "\n".join(details)
        )

    receipt = {
        "schema": RECEIPT_SCHEMA,
        "issued_at_utc": datetime.now(timezone.utc).isoformat(),
        "origin_pid": os.getpid(),
        "context_token": token,
        "context": context,
        "audit_command": command,
        "audit_returncode": completed.returncode,
        "audit_stdout_sha256": _sha256_bytes(completed.stdout.encode("utf-8")),
        "audit_stderr_sha256": _sha256_bytes(completed.stderr.encode("utf-8")),
        "audit_result": audit_result,
    }
    path, digest = _write_receipt(receipt, token)
    print("FINAL_BENCHMARK_AUDIT_GATE: PASS (fresh audit)", flush=True)
    print(f"FINAL_BENCHMARK_AUDIT_CONTEXT: {token}", flush=True)
    print(f"FINAL_BENCHMARK_AUDIT_RECEIPT: {path}", flush=True)
    print(f"FINAL_BENCHMARK_AUDIT_RECEIPT_SHA256: {digest}", flush=True)
    return receipt


def write_final_audit_evidence(
    directory: str | Path,
    receipt: dict[str, Any] | None,
) -> Path | None:
    """Persist the consumed receipt beside strict results for later review."""

    if receipt is None:
        return None
    destination = Path(directory) / "final_audit_evidence.json"
    payload = {
        **receipt,
        "receipt_source": os.environ.get(AUDIT_RECEIPT_ENV),
        "receipt_sha256": os.environ.get(AUDIT_RECEIPT_SHA256_ENV),
    }
    content = (
        json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False) + "\n"
    )
    destination.write_text(content, encoding="utf-8")
    return destination


def validate_research_label_contract(
    run_purpose: str,
    suite_profile: str,
    algorithms: Iterable[str],
) -> None:
    """Reject launcher-level research labels that child arguments could hide."""

    requested = tuple(algorithms)
    if run_purpose == "paper_reproduction":
        raise ResearchLabelContractError(
            "run_purpose=paper_reproduction is not certified by this launcher. "
            "Use diagnostic for exploratory/common-budget runs or "
            "final_benchmark for the audited strict suite."
        )
    allowlist = set(strict_final_algorithms())
    if suite_profile == "primary_research_benchmark":
        forbidden = sorted(set(requested) - allowlist)
        if forbidden:
            raise ResearchLabelContractError(
                "primary_research_benchmark rejects non-allowlisted baselines: "
                + ", ".join(forbidden)
            )
    if run_purpose == "final_benchmark":
        if suite_profile != "primary_research_benchmark":
            raise ResearchLabelContractError(
                "final_benchmark requires suite_profile="
                "primary_research_benchmark"
            )
        forbidden = sorted(set(requested) - allowlist)
        if forbidden:
            raise ResearchLabelContractError(
                "final_benchmark rejects non-allowlisted baselines: "
                + ", ".join(forbidden)
            )
