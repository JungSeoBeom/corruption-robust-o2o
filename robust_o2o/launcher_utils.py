from __future__ import annotations

import re
from collections.abc import Iterable, Mapping


CHILD_IDENTITY_OPTIONS = frozenset(
    {
        "--algorithm",
        "--benchmark-seed-set",
        "--comparison-name",
        "--corruption",
        "--corruption-target",
        "--env-name",
        "--implementation-profile",
        "--algorithm-profile",
        "--online-corruption-scale-profile",
        "--output-dir",
        "--protocol",
        "--run-purpose",
        "--seed",
        "--stage",
        "--suite-profile",
    }
)

RUN55_FIXED_OPTIONS = CHILD_IDENTITY_OPTIONS | {
    "--algorithms",
    "--optional-baselines",
    "--corruption-suite",
    "--evaluation-interval",
    "--eval-period",
    "--evaluation-episodes",
    "--eval-episodes",
    "--final-window-size",
    "--offline-steps",
    "--online-steps",
    "--seeds",
}


def flatten_cli_values(values: Iterable[str]) -> list[str]:
    """Normalize comma-separated, space-separated, or mixed CLI lists."""

    return [
        item.strip()
        for value in values
        for item in value.split(",")
        if item.strip()
    ]


def canonical_algorithms(
    values: Iterable[str], aliases: Mapping[str, str]
) -> list[str]:
    requested = (value.lower() for value in flatten_cli_values(values))
    return [aliases.get(value, value) for value in requested]


def passthrough_conflicts(
    values: Iterable[str], reserved: frozenset[str]
) -> list[str]:
    return sorted(
        option
        for option in values
        if option.split("=", 1)[0] in reserved
    )


def valid_comparison_name(value: str) -> bool:
    return bool(
        value not in (".", "..")
        and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", value)
    )
