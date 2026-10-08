from __future__ import annotations

import re
import argparse
from collections.abc import Iterable, Mapping


CORRUPTION_PROFILES = ("riql_rpex_code", "legacy_extension")

REMOVED_LAUNCH_OPTIONS = frozenset(
    {
        "--allow-diagnostic-protocol",
        "--benchmark-seed-set",
        "--evaluation-mode",
        "--protocol",
        "--run-purpose",
    }
)

CHILD_IDENTITY_OPTIONS = frozenset(
    {
        "--algorithm",
        "--comparison-name",
        "--corruption",
        "--corruption-profile",
        "--corruption-target",
        "--env-name",
        "--implementation-profile",
        "--algorithm-profile",
        "--online-corruption-scale-profile",
        "--output-dir",
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


def resolved_corruption_scale_profile(args: argparse.Namespace) -> str:
    """Resolve the shared attack scale independently of the learner profile."""

    return args.online_corruption_scale_profile or (
        "rpex_official_code"
        if args.corruption_profile == "riql_rpex_code"
        or args.suite_profile in (
            "research_benchmark",
            "method_fidelity",
            "primary_research_benchmark",
        )
        else "dataset_std_scaled_extension"
    )


def validate_corruption_launch_settings(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    *,
    nonclean_targets: Iterable[str],
) -> None:
    args.online_corruption_scale_profile = resolved_corruption_scale_profile(args)
    if args.corruption_profile != "riql_rpex_code":
        return
    if "mixed" in nonclean_targets:
        parser.error(
            "the pinned RIQL/RPEX source code has no mixed corruption contract; "
            "--corruption-target mixed / --targets mixed requires "
            "--corruption-profile legacy_extension"
        )
    if args.online_corruption_scale_profile != "rpex_official_code":
        parser.error(
            "--corruption-profile riql_rpex_code requires "
            "--online-corruption-scale-profile rpex_official_code; "
            "dataset_std_scaled_extension requires "
            "--corruption-profile legacy_extension"
        )


def valid_comparison_name(value: str) -> bool:
    return bool(
        value not in (".", "..")
        and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", value)
    )
