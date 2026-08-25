#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import shlex
import subprocess
import sys
import uuid
from datetime import datetime
from pathlib import Path

from robust_o2o.config import (
    ALGORITHM_ALIASES,
    ALGORITHMS,
    BENCHMARK_ENVS,
    CORRUPTION_MODES,
    CORRUPTION_TARGETS,
    normalize_env_name,
)
from robust_o2o.fidelity import (
    IMPLEMENTATION_PROFILES,
    MAIN_BASELINES,
    ONLINE_CORRUPTION_SCALE_PROFILES,
    SUITE_PROFILES,
)
from robust_o2o.launcher_utils import (
    CHILD_IDENTITY_OPTIONS,
    REMOVED_LAUNCH_OPTIONS,
    canonical_algorithms,
    passthrough_conflicts,
)


def _csv(value: str):
    return [item.strip() for item in value.split(",") if item.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run or print an RPEX-benchmark algorithm/corruption matrix"
    )
    parser.add_argument(
        "--algorithms",
        nargs="+",
        help="algorithm names separated by commas, spaces, or both",
    )
    parser.add_argument(
        "--envs",
        type=_csv,
        default=[
            "halfcheetah-medium-replay-v2",
            "hopper-medium-replay-v2",
            "walker2d-medium-replay-v2",
        ],
    )
    parser.add_argument("--corruptions", type=_csv, default=list(CORRUPTION_MODES))
    parser.add_argument(
        "--targets",
        type=_csv,
        default=["observations", "actions", "rewards", "dynamics", "mixed"],
    )
    parser.add_argument("--seeds", type=_csv, default=["0"])
    parser.add_argument("--corruption-ranges", type=_csv, default=["1.0"])
    parser.add_argument("--stage", choices=("offline", "online", "both"), default="both")
    parser.add_argument("--implementation-profile", choices=IMPLEMENTATION_PROFILES)
    parser.add_argument(
        "--suite-profile", choices=SUITE_PROFILES,
        default="common_budget_robustness",
    )
    parser.add_argument(
        "--online-corruption-scale-profile",
        choices=ONLINE_CORRUPTION_SCALE_PROFILES,
    )
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--comparison-name")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--keep-going", action="store_true")
    return parser


def commands(
    args: argparse.Namespace,
    passthrough: list[str],
    comparison_name: str,
):
    script = Path(__file__).resolve().parent / "run_experiment.py"
    scale_profile = args.online_corruption_scale_profile or (
        "rpex_official_code"
        if args.suite_profile in (
            "method_fidelity",
            "primary_research_benchmark",
        )
        else "dataset_std_scaled_extension"
    )
    for algorithm, env_name, corruption, seed in itertools.product(
        args.algorithms, args.envs, args.corruptions, args.seeds
    ):
        targets = ["none"] if corruption == "clean" else args.targets
        ranges = ["1.0"] if corruption == "clean" else args.corruption_ranges
        for target, corruption_range in itertools.product(targets, ranges):
            command = [
                sys.executable,
                str(script),
                "--algorithm",
                algorithm,
                "--env-name",
                env_name,
                "--corruption",
                corruption,
                "--corruption-target",
                target,
                "--seed",
                seed,
                "--stage",
                args.stage,
                "--suite-profile",
                args.suite_profile,
                "--online-corruption-scale-profile",
                scale_profile,
                "--corruption-range",
                corruption_range,
                "--output-dir",
                args.output_dir,
                "--comparison-name",
                comparison_name,
                *passthrough,
            ]
            if args.implementation_profile:
                command.extend(("--implementation-profile", args.implementation_profile))
            yield command


def _validate_args(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    passthrough: list[str],
) -> None:
    if args.algorithms is None:
        args.algorithms = list(
            MAIN_BASELINES
            if args.suite_profile == "research_benchmark"
            else ALGORITHMS
        )
    args.algorithms = canonical_algorithms(args.algorithms, ALGORITHM_ALIASES)

    unknown_algorithms = sorted(set(args.algorithms) - set(ALGORITHMS))
    if unknown_algorithms:
        parser.error(f"unknown algorithms: {', '.join(unknown_algorithms)}")
    if not args.algorithms:
        parser.error("--algorithms cannot be empty")
    if len(args.algorithms) != len(set(args.algorithms)):
        parser.error(
            "algorithm selections cannot contain duplicates after alias normalization"
        )
    args.envs = [normalize_env_name(env_name) for env_name in args.envs]
    unknown_envs = sorted(set(args.envs) - set(BENCHMARK_ENVS))
    if unknown_envs:
        parser.error(f"unknown benchmark environments: {', '.join(unknown_envs)}")
    if not args.envs:
        parser.error("--envs cannot be empty")
    unknown_corruptions = sorted(set(args.corruptions) - set(CORRUPTION_MODES))
    if unknown_corruptions:
        parser.error(f"unknown corruptions: {', '.join(unknown_corruptions)}")
    unknown_targets = sorted(set(args.targets) - set(CORRUPTION_TARGETS))
    if unknown_targets:
        parser.error(f"unknown corruption targets: {', '.join(unknown_targets)}")
    if not args.corruptions:
        parser.error("--corruptions cannot be empty")
    if not args.targets and any(mode != "clean" for mode in args.corruptions):
        parser.error("non-clean corruptions require at least one --targets value")
    if not args.corruption_ranges and any(
        mode != "clean" for mode in args.corruptions
    ):
        parser.error(
            "non-clean corruptions require at least one --corruption-ranges value"
        )

    if not args.seeds:
        parser.error("--seeds cannot be empty")
    for seed in args.seeds:
        try:
            int(seed)
        except ValueError:
            parser.error(f"invalid seed: {seed!r}")

    conflicts = passthrough_conflicts(passthrough, CHILD_IDENTITY_OPTIONS)
    if conflicts:
        parser.error(
            "these child identity/provenance options cannot be overridden: "
            + ", ".join(conflicts)
        )
    removed = passthrough_conflicts(passthrough, REMOVED_LAUNCH_OPTIONS)
    if removed:
        parser.error("these options were removed: " + ", ".join(removed))
    for value in args.corruption_ranges:
        try:
            parsed = float(value)
        except ValueError:
            parser.error(f"invalid corruption range: {value!r}")
        if parsed < 0.0:
            parser.error("--corruption-ranges cannot contain negative values")



def main() -> int:
    parser = build_parser()
    args, passthrough = parser.parse_known_args()
    _validate_args(parser, args, passthrough)
    comparison_name = args.comparison_name
    if not comparison_name:
        stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
        comparison_name = f"matrix_{stamp}_{str(uuid.uuid4())[:8]}"
    generated_commands = list(commands(args, passthrough, comparison_name))
    if not generated_commands:
        parser.error("the resolved matrix contains no runs")
    failures = 0
    for index, command in enumerate(generated_commands, start=1):
        print(f"[{index}] {shlex.join(command)}", flush=True)
        if args.dry_run:
            continue
        result = subprocess.run(command, check=False)
        if result.returncode != 0:
            failures += 1
            if not args.keep_going:
                return result.returncode
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
