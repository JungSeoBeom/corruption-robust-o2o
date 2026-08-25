#!/usr/bin/env python3
"""Launch the custom-budget research benchmark across corruption settings."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Iterable

from robust_o2o.config import (
    ALGORITHM_ALIASES,
    BENCHMARK_ENVS,
)
from robust_o2o.corruption import SUPPORTED_ADVERSARIAL_TARGETS
from robust_o2o.fidelity import (
    IMPLEMENTATION_PROFILES,
    MAIN_BASELINES,
    ONLINE_CORRUPTION_SCALE_PROFILES,
    SUITE_PROFILES,
)
from robust_o2o.launcher_utils import (
    REMOVED_LAUNCH_OPTIONS,
    RUN55_FIXED_OPTIONS,
    canonical_algorithms,
    flatten_cli_values,
    passthrough_conflicts,
    valid_comparison_name,
)
ENV_NAME = "halfcheetah-medium-replay-v2"
CLEAN_SETTINGS = (("clean", "none"),)
RANDOM_SETTINGS = (
    ("random", "observations"),
    ("random", "actions"),
    ("random", "rewards"),
    ("random", "dynamics"),
)
ADVERSARIAL_SETTINGS = tuple(
    ("adversarial", target)
    for target in SUPPORTED_ADVERSARIAL_TARGETS
)
FIXED_SETTINGS = (*ADVERSARIAL_SETTINGS, *CLEAN_SETTINGS, *RANDOM_SETTINGS)


def _default_experiment_name(env_name: str) -> str:
    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    domain = env_name.split("-", 1)[0]
    return f"{domain}_5x5_{stamp}_{str(uuid.uuid4())[:8]}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the five main baselines on the fixed nine-condition "
            "adversarial, clean, and random corruption benchmark."
        )
    )
    parser.add_argument("--env-name", choices=BENCHMARK_ENVS, default=ENV_NAME)
    parser.add_argument(
        "--algorithms",
        nargs="+",
        default=list(MAIN_BASELINES),
        help=(
            "main baselines as comma-separated and/or space-separated names; "
            "Cal-QL/PQE aliases are normalized to canonical names"
        ),
    )
    parser.add_argument("--seeds", nargs="+", default=["0"])
    parser.add_argument("--offline-steps", type=int, default=500_000)
    parser.add_argument("--online-steps", type=int, default=500_000)
    parser.add_argument(
        "--evaluation-interval",
        "--eval-period",
        dest="eval_period",
        type=int,
        default=10_000,
    )
    parser.add_argument(
        "--evaluation-episodes",
        "--eval-episodes",
        dest="eval_episodes",
        type=int,
        default=10,
    )
    parser.add_argument("--final-window-size", type=int, default=3)
    parser.add_argument(
        "--implementation-profile", choices=IMPLEMENTATION_PROFILES
    )
    parser.add_argument(
        "--suite-profile", choices=SUITE_PROFILES,
        default="research_benchmark",
    )
    parser.add_argument(
        "--online-corruption-scale-profile",
        choices=ONLINE_CORRUPTION_SCALE_PROFILES,
    )
    parser.add_argument("--output-root", default="results")
    parser.add_argument("--dataset-dir")
    parser.add_argument(
        "--experiment-name",
        help="shared comparison ID used under each of the nine setting directories",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="continue with remaining algorithms/settings after a failed run",
    )
    return parser


def _validate_args(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    passthrough: Iterable[str],
) -> None:
    args.algorithms = canonical_algorithms(args.algorithms, ALGORITHM_ALIASES)
    args.seeds = flatten_cli_values(args.seeds)
    if not args.seeds:
        parser.error("--seeds cannot be empty")
    for seed in args.seeds:
        try:
            int(seed)
        except ValueError:
            parser.error(f"invalid seed: {seed!r}")
    requested = tuple(args.algorithms)
    unknown_main = sorted(set(args.algorithms) - set(MAIN_BASELINES))
    if unknown_main:
        parser.error(
            "--algorithms accepts only main baselines "
            f"{','.join(MAIN_BASELINES)}; invalid: {','.join(unknown_main)}"
        )
    if not args.algorithms:
        parser.error("--algorithms cannot be empty for a research benchmark")
    if len(requested) != len(set(requested)):
        parser.error("algorithm selections cannot contain duplicates")
    if args.suite_profile == "research_benchmark":
        if args.implementation_profile is None:
            args.implementation_profile = "research_benchmark"
        elif args.implementation_profile != "research_benchmark":
            parser.error(
                "research_benchmark requires "
                "--implementation-profile research_benchmark"
            )
    if args.offline_steps < 0 or args.online_steps < 0:
        parser.error("--offline-steps and --online-steps cannot be negative")
    for name in ("eval_period", "eval_episodes", "final_window_size"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.suite_profile == "method_fidelity":
        parser.error(
            "method_fidelity is unavailable for the fixed suite: canonical "
            "Cal-QL is a frozen "
            "locomotion adaptation and canonical PQE is a D4RL-v2 port. Use "
            "--suite-profile common_budget_robustness."
        )
    if args.online_corruption_scale_profile is None:
        args.online_corruption_scale_profile = (
            "rpex_official_code"
            if args.suite_profile == "research_benchmark"
            else "dataset_std_scaled_extension"
        )
    if args.experiment_name and not valid_comparison_name(args.experiment_name):
        parser.error(
            "--experiment-name may contain only letters, digits, '.', '_', and '-', "
            "and must start with a letter or digit"
        )
    conflicts = passthrough_conflicts(passthrough, RUN55_FIXED_OPTIONS)
    if conflicts:
        parser.error(
            "these options are fixed by the nine-condition suite and cannot "
            "be overridden: "
            + ", ".join(conflicts)
        )
    removed = passthrough_conflicts(passthrough, REMOVED_LAUNCH_OPTIONS)
    if removed:
        parser.error("these options were removed: " + ", ".join(removed))


def commands(
    args: argparse.Namespace,
    passthrough: Iterable[str],
    experiment_name: str,
) -> Iterable[list[str]]:
    runner = Path(__file__).resolve().parent / "run_all_algorithms.py"
    scale_profile = args.online_corruption_scale_profile or (
        "rpex_official_code"
        if args.suite_profile == "research_benchmark"
        else "dataset_std_scaled_extension"
    )
    algorithm_csv = ",".join(args.algorithms)
    seed_csv = ",".join(args.seeds)
    for corruption, target in FIXED_SETTINGS:
        command = [
            sys.executable,
            str(runner),
            "--env-name",
            args.env_name,
            "--corruption",
            corruption,
            "--corruption-target",
            target,
            "--algorithms",
            algorithm_csv,
            "--seeds",
            seed_csv,
            "--stage",
            "both",
            "--suite-profile",
            args.suite_profile,
            "--online-corruption-scale-profile",
            scale_profile,
            "--output-root",
            args.output_root,
            "--comparison-name",
            experiment_name,
        ]
        command.extend(
            (
                "--offline-steps",
                str(args.offline_steps),
                "--online-steps",
                str(args.online_steps),
                "--eval-period",
                str(args.eval_period),
                "--eval-episodes",
                str(args.eval_episodes),
                "--final-window-size",
                str(args.final_window_size),
            )
        )
        if args.implementation_profile:
            command.extend(("--implementation-profile", args.implementation_profile))
        if args.dataset_dir:
            command.extend(("--dataset-dir", args.dataset_dir))
        if args.keep_going:
            command.append("--keep-going")
        if args.dry_run:
            command.append("--dry-run")
        command.extend(passthrough)
        yield command


def main() -> int:
    parser = build_parser()
    args, passthrough = parser.parse_known_args()
    _validate_args(parser, args, passthrough)
    experiment_name = args.experiment_name or _default_experiment_name(args.env_name)
    generated_commands = list(commands(args, passthrough, experiment_name))
    selected_settings = FIXED_SETTINGS
    total_runs = len(args.algorithms) * len(selected_settings) * len(args.seeds)

    print(f"EXPERIMENT_NAME: {experiment_name}", flush=True)
    print(f"ENVIRONMENT: {args.env_name}", flush=True)
    print(f"ALGORITHMS: {', '.join(args.algorithms)}", flush=True)
    print(
        "SETTINGS: "
        + ", ".join(f"{mode}/{target}" for mode, target in selected_settings),
        flush=True,
    )
    print(
        f"SCHEDULE: offline={args.offline_steps:,}, online={args.online_steps:,}, "
        f"eval_interval={args.eval_period:,}, eval_episodes={args.eval_episodes}, "
        f"final_window={args.final_window_size}",
        flush=True,
    )
    print(f"SUITE_PROFILE: {args.suite_profile}", flush=True)
    print(
        f"ONLINE_CORRUPTION_SCALE_PROFILE: {args.online_corruption_scale_profile}",
        flush=True,
    )
    print(f"TOTAL_RUNS: {total_runs} ({len(args.seeds)} seed(s))", flush=True)

    failures = 0
    for index, command in enumerate(generated_commands, start=1):
        corruption, target = selected_settings[index - 1]
        print(
            f"SETTING [{index}/{len(selected_settings)}]: {corruption}/{target}",
            flush=True,
        )
        print(shlex.join(command), flush=True)
        result = subprocess.run(command, check=False)
        if result.returncode != 0:
            failures += 1
            print(
                f"SETTING_FAILED: {corruption}/{target} returncode={result.returncode}",
                file=sys.stderr,
                flush=True,
            )
            if not args.keep_going:
                return result.returncode

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
