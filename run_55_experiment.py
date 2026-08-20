#!/usr/bin/env python3
"""Run one environment's 5-algorithm by 5-condition experiment suite."""

from __future__ import annotations

import argparse
import re
import shlex
import subprocess
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Iterable

from robust_o2o.config import (
    BENCHMARK_ENVS,
    DEFAULT_PROTOCOL,
    LEGACY_LOCAL_PROTOCOL_ALIAS,
    LOCAL_PROTOCOL,
    PROTOCOLS,
)
from robust_o2o.fidelity import IMPLEMENTATION_PROFILES, SUITE_PROFILES


ENV_NAME = "hopper-medium-replay-v2"
ALGORITHMS = (
    "rpex",
    "riql_naive",
    "wsrl",
    "cal_ql",
    "pessimistic_q_ensemble",
)
SETTINGS = (
    ("clean", "none"),
    ("random", "observations"),
    ("random", "actions"),
    ("random", "rewards"),
    ("random", "dynamics"),
)
RESERVED_PASSTHROUGH_OPTIONS = {
    "--algorithms",
    "--comparison-name",
    "--corruption",
    "--corruption-target",
    "--env-name",
    "--stage",
    "--suite-profile",
    "--implementation-profile",
    "--algorithm-profile",
}


def _csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _default_experiment_name(env_name: str) -> str:
    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    domain = env_name.split("-", 1)[0]
    return f"{domain}_5x5_{stamp}_{str(uuid.uuid4())[:8]}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run RPEX, RIQL naive, WSRL, Cal-QL, and Pessimistic Q-Ensemble "
            "on one environment's clean and four random-corruption targets "
            "(25 runs per seed)"
        )
    )
    parser.add_argument("--env-name", choices=BENCHMARK_ENVS, default=ENV_NAME)
    parser.add_argument("--seeds", type=_csv, default=["0"])
    parser.add_argument("--offline-steps", type=int, default=500_000)
    parser.add_argument("--online-steps", type=int, default=500_000)
    parser.add_argument("--protocol", choices=PROTOCOLS, default=DEFAULT_PROTOCOL)
    parser.add_argument(
        "--implementation-profile", choices=IMPLEMENTATION_PROFILES
    )
    parser.add_argument(
        "--suite-profile", choices=SUITE_PROFILES,
        default="common_budget_robustness",
    )
    parser.add_argument("--allow-diagnostic-protocol", action="store_true")
    parser.add_argument("--output-root", default="results")
    parser.add_argument("--dataset-dir")
    parser.add_argument(
        "--experiment-name",
        help="shared comparison ID used under each of the five setting directories",
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
    if args.protocol == LEGACY_LOCAL_PROTOCOL_ALIAS:
        args.protocol = LOCAL_PROTOCOL
    if not args.seeds:
        parser.error("--seeds cannot be empty")
    for seed in args.seeds:
        try:
            int(seed)
        except ValueError:
            parser.error(f"invalid seed: {seed!r}")
    if args.offline_steps < 0 or args.online_steps < 0:
        parser.error("--offline-steps and --online-steps cannot be negative")
    if args.suite_profile == "method_fidelity":
        parser.error(
            "method_fidelity 5x5 is unavailable: Cal-QL locomotion is a task "
            "port and the local PQE is pqe_shared_actor_approx. Use "
            "--suite-profile common_budget_robustness; no run will be mislabeled "
            "as paper reproduction."
        )
    if args.protocol in (LOCAL_PROTOCOL, "local_gymnasium_v4") and not args.allow_diagnostic_protocol:
        parser.error(
            "the local Gymnasium protocol is diagnostic-only; pass "
            "--allow-diagnostic-protocol to acknowledge this"
        )
    if args.experiment_name and (
        not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", args.experiment_name)
        or args.experiment_name in (".", "..")
    ):
        parser.error(
            "--experiment-name may contain only letters, digits, '.', '_', and '-', "
            "and must start with a letter or digit"
        )
    conflicts = sorted(
        option for option in passthrough if option.split("=", 1)[0] in RESERVED_PASSTHROUGH_OPTIONS
    )
    if conflicts:
        parser.error(
            "these options are fixed by the 5x5 suite and cannot be overridden: "
            + ", ".join(conflicts)
        )


def commands(
    args: argparse.Namespace,
    passthrough: Iterable[str],
    experiment_name: str,
) -> Iterable[list[str]]:
    runner = Path(__file__).resolve().parent / "run_all_algorithms.py"
    algorithm_csv = ",".join(ALGORITHMS)
    seed_csv = ",".join(args.seeds)
    for corruption, target in SETTINGS:
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
            "--protocol",
            args.protocol,
            "--suite-profile",
            args.suite_profile,
            "--output-root",
            args.output_root,
            "--comparison-name",
            experiment_name,
            "--offline-steps",
            str(args.offline_steps),
            "--online-steps",
            str(args.online_steps),
        ]
        if args.implementation_profile:
            command.extend(("--implementation-profile", args.implementation_profile))
        if args.allow_diagnostic_protocol:
            command.append("--allow-diagnostic-protocol")
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
    total_runs = len(ALGORITHMS) * len(SETTINGS) * len(args.seeds)

    print(f"EXPERIMENT_NAME: {experiment_name}", flush=True)
    print(f"ENVIRONMENT: {args.env_name}", flush=True)
    print(f"ALGORITHMS: {', '.join(ALGORITHMS)}", flush=True)
    print("SETTINGS: clean/none, random/{observations,actions,rewards,dynamics}", flush=True)
    print(
        f"SCHEDULE: offline={args.offline_steps:,}, online={args.online_steps:,}",
        flush=True,
    )
    print(f"SUITE_PROFILE: {args.suite_profile}", flush=True)
    print(f"TOTAL_RUNS: {total_runs} ({len(args.seeds)} seed(s))", flush=True)

    failures = 0
    for index, command in enumerate(generated_commands, start=1):
        corruption, target = SETTINGS[index - 1]
        print(
            f"SETTING [{index}/{len(SETTINGS)}]: {corruption}/{target}",
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
