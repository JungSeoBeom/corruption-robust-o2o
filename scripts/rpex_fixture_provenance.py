"""Fail-closed provenance helpers for RPEX fixture generators.

This module validates only the environment in which a fixture is generated.
It does not certify learner, evaluator, corruption-wrapper, or end-to-end
parity.  Those claims require separate executable receipts.
"""

from __future__ import annotations

import hashlib
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


PINNED_RPEX_COMMIT = "35da71ee5151b6179d21b9a2b4ce1b6408aedd04"
STRICT_PYTHON = (3, 10)
STRICT_NUMPY = "1.23.5"
STRICT_PYTORCH = "2.5.1"
STRICT_PLATFORM = "Linux"
STRICT_MACHINE = "x86_64"


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def _base_version(value: str) -> str:
    return value.split("+", 1)[0]


def runtime_preflight() -> dict[str, Any]:
    """Return exact strict-stack checks without mutating runtime state."""

    actual_python = tuple(sys.version_info[:2])
    actual_machine = platform.machine().lower()
    checks = {
        "platform": {
            "expected": STRICT_PLATFORM,
            "actual": platform.system(),
        },
        "machine": {
            "expected": STRICT_MACHINE,
            "actual": actual_machine,
        },
        "python": {
            "expected": ".".join(map(str, STRICT_PYTHON)),
            "actual": ".".join(map(str, actual_python)),
        },
        "numpy": {
            "expected": STRICT_NUMPY,
            "actual": np.__version__,
        },
        "pytorch": {
            "expected": STRICT_PYTORCH,
            "actual": _base_version(torch.__version__),
            "actual_full": torch.__version__,
        },
    }
    for item in checks.values():
        item["passed"] = item["actual"] == item["expected"]
    passed = all(bool(item["passed"]) for item in checks.values())
    return {"passed": passed, "checks": checks}


def require_strict_runtime(*, allow_diagnostic_mismatch: bool) -> dict[str, Any]:
    """Fail unless the exact fixture runtime is present.

    The override exists only to inspect diagnostic behavior.  Callers must
    give such output a diagnostic-only fixture identifier and must record the
    failed checks in the payload.
    """

    result = runtime_preflight()
    if result["passed"] or allow_diagnostic_mismatch:
        return result
    failures = [
        f"{name}: expected {item['expected']!r}, got {item['actual']!r}"
        for name, item in result["checks"].items()
        if not item["passed"]
    ]
    raise RuntimeError(
        "strict RPEX fixture runtime mismatch; generation refused: "
        + "; ".join(failures)
    )


def _git(upstream: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=upstream,
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def require_pinned_clean_upstream(upstream: Path) -> dict[str, Any]:
    """Validate the exact upstream commit and a completely clean worktree."""

    upstream = upstream.expanduser().resolve()
    if not upstream.is_dir():
        raise RuntimeError(f"upstream checkout does not exist: {upstream}")
    repository_root = Path(
        _git(upstream, "rev-parse", "--show-toplevel").strip()
    ).resolve()
    if repository_root != upstream:
        raise RuntimeError(
            f"--upstream-dir must be the repository root: {repository_root}"
        )
    commit = _git(upstream, "rev-parse", "HEAD").strip()
    if commit != PINNED_RPEX_COMMIT:
        raise RuntimeError(
            f"upstream commit {commit} != pinned {PINNED_RPEX_COMMIT}"
        )
    status = _git(
        upstream,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
    )
    if status:
        raise RuntimeError(
            "pinned RPEX checkout is dirty; fixture generation refused "
            f"(status_sha256={sha256_bytes(status.encode('utf-8'))})"
        )
    return {
        "repository_root": str(repository_root),
        "commit": commit,
        "clean": True,
        "status_sha256": sha256_bytes(status.encode("utf-8")),
    }


def platform_metadata(*, execution_device: str) -> dict[str, Any]:
    cuda_devices = []
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            cuda_devices.append(
                {
                    "index": index,
                    "name": torch.cuda.get_device_name(index),
                    "capability": list(torch.cuda.get_device_capability(index)),
                }
            )
    libc_name, libc_version = platform.libc_ver()
    return {
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "libc": {"name": libc_name, "version": libc_version},
        "numpy_version": np.__version__,
        "pytorch_version": torch.__version__,
        "execution_device": execution_device,
        "cuda_available": torch.cuda.is_available(),
        "cuda_runtime_version": torch.version.cuda,
        "cudnn_version": (
            torch.backends.cudnn.version() if torch.cuda.is_available() else None
        ),
        "cuda_devices": cuda_devices,
    }


def numpy_rng_state_hash(state: tuple[Any, ...]) -> str:
    """Hash a NumPy RandomState/global MT19937 state deterministically."""

    algorithm, keys, position, has_gauss, cached_gaussian = state
    digest = hashlib.sha256()
    digest.update(str(algorithm).encode("ascii"))
    digest.update(np.ascontiguousarray(keys, dtype=np.uint32).tobytes())
    digest.update(int(position).to_bytes(8, "little", signed=False))
    digest.update(int(has_gauss).to_bytes(1, "little", signed=False))
    digest.update(np.asarray([cached_gaussian], dtype=np.float64).tobytes())
    return digest.hexdigest()


def torch_rng_state_hash(generator: torch.Generator | None = None) -> str:
    state = (
        generator.get_state()
        if generator is not None
        else torch.random.get_rng_state()
    )
    return sha256_bytes(state.detach().cpu().numpy().tobytes())

