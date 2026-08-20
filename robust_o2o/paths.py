from __future__ import annotations

from pathlib import Path
from typing import Tuple


def comparison_directory(
    output_root: str,
    env_name: str,
    corruption: str,
    target: str,
    comparison_id: str,
    protocol: str | None = None,
    algorithm_profile: str | None = None,
) -> Path:
    """Return the canonical protocol/profile-aware comparison directory."""
    base = Path(output_root).expanduser().resolve() / "comparisons"
    if protocol is not None:
        base = base / protocol
    if algorithm_profile is not None:
        base = base / algorithm_profile
    return (
        base / env_name
        / corruption
        / target
        / comparison_id
    )


def resolve_run_layout(
    output_dir: str,
    env_name: str,
    corruption: str,
    target: str,
    comparison_id: str,
    protocol: str | None = None,
    algorithm_profile: str | None = None,
) -> Tuple[Path, Path]:
    """Resolve a comparison directory and its runs directory for every CLI."""
    root = Path(output_dir).expanduser().resolve()
    if root.name == "runs" and root.parent.parent.name == target:
        return root.parent, root
    comparison_dir = comparison_directory(
        str(root), env_name, corruption, target, comparison_id,
        protocol, algorithm_profile,
    )
    return comparison_dir, comparison_dir / "runs"


def results_root_from_output(output_dir: str) -> Path:
    """Recover the output root even when a comparison's runs path is supplied."""
    path = Path(output_dir).expanduser().resolve()
    for candidate in (path, *path.parents):
        if candidate.name == "comparisons":
            return candidate.parent
    return path
