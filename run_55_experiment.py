#!/usr/bin/env python3
"""Run the declared five-baseline diagnostic or strict eligible subset suite."""

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
from robust_o2o.fidelity import (
    IMPLEMENTATION_PROFILES,
    ONLINE_CORRUPTION_SCALE_PROFILES,
    RUN_PURPOSES,
    STRICT_FINAL_SEEDS,
    STRICT_FINAL_TASKS,
    SUITE_PROFILES,
    strict_final_algorithms,
)
from robust_o2o.final_gate import (
    FinalAuditGateError,
    ResearchLabelContractError,
    require_final_benchmark_audit,
    validate_research_label_contract,
)


ENV_NAME = "hopper-medium-replay-v2"
ALGORITHMS = (
    "rpex",
    "riql_naive",
    "wsrl",
    "cal_ql",
    "pessimistic_q_ensemble",
)
CORRUPTION_SUITES = ("clean", "random", "adversarial", "all")
CLEAN_SETTINGS = (("clean", "none"),)
RANDOM_SETTINGS = (
    *CLEAN_SETTINGS,
    ("random", "observations"),
    ("random", "actions"),
    ("random", "rewards"),
    ("random", "dynamics"),
)
ADVERSARIAL_SETTINGS = (
    ("adversarial", "observations"),
    ("adversarial", "actions"),
    ("adversarial", "rewards"),
    ("adversarial", "dynamics"),
)
# The pinned upstream-derived adversarial certificate currently covers only
# the observation optimizer core.  Other official RPEX targets remain useful
# diagnostic conditions, but cannot enter a strict result until target-specific
# fixtures are added.
STRICT_ADVERSARIAL_SETTINGS = (("adversarial", "observations"),)
# Backward-compatible import used by older callers; the default suite remains
# clean plus the four random targets.
SETTINGS = RANDOM_SETTINGS
RESERVED_PASSTHROUGH_OPTIONS = {
    "--algorithm",
    "--algorithms",
    "--benchmark-seed-set",
    "--comparison-name",
    "--corruption",
    "--corruption-target",
    "--corruption-suite",
    "--env-name",
    "--stage",
    "--suite-profile",
    "--run-purpose",
    "--online-corruption-scale-profile",
    "--implementation-profile",
    "--algorithm-profile",
    "--output-dir",
    "--protocol",
    "--seed",
    "--seeds",
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
            "Run the five diagnostic baselines, or the registry-approved "
            "strict subset, on clean/random/adversarial RPEX conditions"
        )
    )
    parser.add_argument("--env-name", choices=BENCHMARK_ENVS, default=ENV_NAME)
    parser.add_argument(
        "--corruption-suite",
        choices=CORRUPTION_SUITES,
        default="random",
        help=(
            "clean; clean+four official random targets; four official "
            "adversarial targets; or their union"
        ),
    )
    parser.add_argument("--seeds", type=_csv, default=["0"])
    parser.add_argument("--offline-steps", type=int, default=500_000)
    parser.add_argument("--online-steps", type=int, default=500_000)
    parser.add_argument("--protocol", choices=PROTOCOLS, default=DEFAULT_PROTOCOL)
    parser.add_argument(
        "--implementation-profile", choices=IMPLEMENTATION_PROFILES
    )
    parser.add_argument(
        "--suite-profile", choices=SUITE_PROFILES,
        default="common_budget_diagnostic",
    )
    parser.add_argument("--run-purpose", choices=RUN_PURPOSES, default="diagnostic")
    parser.add_argument(
        "--online-corruption-scale-profile",
        choices=ONLINE_CORRUPTION_SCALE_PROFILES,
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
    selected_algorithms = (
        tuple(
            algorithm
            for algorithm in ALGORITHMS
            if algorithm in strict_final_algorithms()
        )
        if args.suite_profile == "primary_research_benchmark"
        else ALGORITHMS
    )
    try:
        validate_research_label_contract(
            args.run_purpose,
            args.suite_profile,
            selected_algorithms,
        )
    except ResearchLabelContractError as exc:
        parser.error(str(exc))
    if args.offline_steps < 0 or args.online_steps < 0:
        parser.error("--offline-steps and --online-steps cannot be negative")
    if args.suite_profile == "method_fidelity":
        parser.error(
            "method_fidelity 5x5 is unavailable: Cal-QL locomotion is a task "
            "port and the local PQE is pqe_shared_actor_approx. Use "
            "--suite-profile common_budget_diagnostic; no run will be mislabeled "
            "as paper reproduction."
        )
    if args.run_purpose == "final_benchmark":
        required = {str(seed) for seed in STRICT_FINAL_SEEDS}
        if set(args.seeds) != required or len(args.seeds) != len(required):
            parser.error(
                "final_benchmark requires exactly seeds 0,1,2,3,4; "
                "single-seed runs are smoke/diagnostic only"
            )
        if args.suite_profile != "primary_research_benchmark":
            parser.error(
                "final_benchmark requires "
                "--suite-profile primary_research_benchmark"
            )
        if args.protocol != DEFAULT_PROTOCOL:
            parser.error(
                "final_benchmark requires the strict rpex_d4rl_v2_legacy protocol"
            )
        if args.env_name not in STRICT_FINAL_TASKS:
            parser.error(
                "final_benchmark permits only the three medium-replay-v2 tasks"
            )
        if (
            args.corruption_suite in ("adversarial", "all")
            and args.env_name != "hopper-medium-replay-v2"
        ):
            parser.error(
                "final_benchmark adversarial observations currently require "
                "hopper-medium-replay-v2: the registered optimizer-core "
                "fixture is bound to the Hopper EDAC checkpoint"
            )
    if args.online_corruption_scale_profile is None:
        args.online_corruption_scale_profile = (
            "rpex_official_code"
            if args.suite_profile == "primary_research_benchmark"
            else "dataset_std_scaled_extension"
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
    scale_profile = args.online_corruption_scale_profile or (
        "rpex_official_code"
        if args.suite_profile == "primary_research_benchmark"
        else "dataset_std_scaled_extension"
    )
    selected_algorithms = (
        tuple(
            algorithm
            for algorithm in ALGORITHMS
            if algorithm in strict_final_algorithms()
        )
        if args.suite_profile == "primary_research_benchmark"
        else ALGORITHMS
    )
    algorithm_csv = ",".join(selected_algorithms)
    seed_csv = ",".join(args.seeds)
    for corruption, target in settings_for_suite(
        args.corruption_suite,
        strict=args.suite_profile == "primary_research_benchmark",
    ):
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
            "--run-purpose",
            args.run_purpose,
            "--online-corruption-scale-profile",
            scale_profile,
            "--output-root",
            args.output_root,
            "--comparison-name",
            experiment_name,
        ]
        if args.suite_profile != "primary_research_benchmark":
            command.extend(
                (
                    "--offline-steps",
                    str(args.offline_steps),
                    "--online-steps",
                    str(args.online_steps),
                )
            )
        if args.implementation_profile:
            command.extend(("--implementation-profile", args.implementation_profile))
        if args.allow_diagnostic_protocol:
            command.append("--allow-diagnostic-protocol")
        if args.dataset_dir:
            command.extend(("--dataset-dir", args.dataset_dir))
        if args.run_purpose == "final_benchmark":
            command.extend(
                (
                    "--benchmark-seed-set",
                    *(str(seed) for seed in STRICT_FINAL_SEEDS),
                )
            )
        if args.keep_going:
            command.append("--keep-going")
        if args.dry_run:
            command.append("--dry-run")
        command.extend(passthrough)
        yield command


def settings_for_suite(
    suite: str,
    *,
    strict: bool = False,
) -> tuple[tuple[str, str], ...]:
    adversarial = STRICT_ADVERSARIAL_SETTINGS if strict else ADVERSARIAL_SETTINGS
    if suite == "clean":
        return CLEAN_SETTINGS
    if suite == "random":
        return RANDOM_SETTINGS
    if suite == "adversarial":
        return adversarial
    if suite == "all":
        return (*RANDOM_SETTINGS, *adversarial)
    raise ValueError(f"unknown corruption suite {suite!r}")


def main() -> int:
    parser = build_parser()
    args, passthrough = parser.parse_known_args()
    _validate_args(parser, args, passthrough)
    try:
        require_final_benchmark_audit(
            args.run_purpose,
            dry_run=args.dry_run,
        )
    except FinalAuditGateError as exc:
        print(f"FINAL_BENCHMARK_AUDIT_GATE_FAILED: {exc}", file=sys.stderr)
        return 2
    experiment_name = args.experiment_name or _default_experiment_name(args.env_name)
    generated_commands = list(commands(args, passthrough, experiment_name))
    selected_algorithms = (
        tuple(
            algorithm
            for algorithm in ALGORITHMS
            if algorithm in strict_final_algorithms()
        )
        if args.suite_profile == "primary_research_benchmark"
        else ALGORITHMS
    )
    selected_settings = settings_for_suite(
        args.corruption_suite,
        strict=args.suite_profile == "primary_research_benchmark",
    )
    total_runs = len(selected_algorithms) * len(selected_settings) * len(args.seeds)

    print(f"EXPERIMENT_NAME: {experiment_name}", flush=True)
    print(f"ENVIRONMENT: {args.env_name}", flush=True)
    print(f"ALGORITHMS: {', '.join(selected_algorithms)}", flush=True)
    print(f"CORRUPTION_SUITE: {args.corruption_suite}", flush=True)
    print(
        "SETTINGS: "
        + ", ".join(f"{mode}/{target}" for mode, target in selected_settings),
        flush=True,
    )
    if args.suite_profile == "primary_research_benchmark":
        if args.corruption_suite in ("adversarial", "all"):
            print(
                "EXCLUDED_UNCERTIFIED_ADVERSARIAL_TARGETS: "
                "actions,rewards,dynamics (diagnostic-only until target-specific "
                "upstream fixtures exist)",
                flush=True,
            )
        print(
            "SCHEDULE: method-specific upstream budgets; RPEX/RIQL "
            "offline=2,000,001 updates, online=1,000,001 nominal steps",
            flush=True,
        )
    else:
        print(
            f"SCHEDULE: offline={args.offline_steps:,}, online={args.online_steps:,}",
            flush=True,
        )
    print(f"SUITE_PROFILE: {args.suite_profile}", flush=True)
    print(f"RUN_PURPOSE: {args.run_purpose}", flush=True)
    if args.run_purpose in ("smoke", "diagnostic"):
        print("NOT A PAPER REPRODUCTION RUN", flush=True)
        print("NOT PUBLICATION-ELIGIBLE", flush=True)
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
