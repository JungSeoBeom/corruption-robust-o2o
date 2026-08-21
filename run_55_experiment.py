#!/usr/bin/env python3
"""Launch the custom-budget research benchmark across corruption settings."""

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
from robust_o2o.corruption import SUPPORTED_ADVERSARIAL_TARGETS
from robust_o2o.fidelity import (
    IMPLEMENTATION_PROFILES,
    MAIN_BASELINES,
    ONLINE_CORRUPTION_SCALE_PROFILES,
    OPTIONAL_BASELINES,
    RUN_PURPOSES,
    STRICT_FINAL_SEEDS,
    STRICT_FINAL_TASKS,
    SUITE_PROFILES,
    strict_final_algorithms,
)
ENV_NAME = "hopper-medium-replay-v2"
# Backward-compatible import name. The default suite is now intentionally
# three main baselines; adaptations/approximations require an explicit option.
ALGORITHMS = MAIN_BASELINES
CORRUPTION_SUITES = ("clean", "random", "adversarial", "all")
CLEAN_SETTINGS = (("clean", "none"),)
DIAGNOSTIC_RANDOM_SETTINGS = (
    *CLEAN_SETTINGS,
    ("random", "observations"),
    ("random", "actions"),
    ("random", "rewards"),
    ("random", "dynamics"),
)
STRICT_RANDOM_SETTINGS = (
    ("random", "observations"),
    ("random", "actions"),
    ("random", "rewards"),
    ("random", "dynamics"),
)
# Backward-compatible name for callers that use the historical diagnostic
# clean-plus-four matrix.
RANDOM_SETTINGS = DIAGNOSTIC_RANDOM_SETTINGS
ADVERSARIAL_SETTINGS = tuple(
    ("adversarial", target)
    for target in SUPPORTED_ADVERSARIAL_TARGETS
)
# The pinned upstream-derived adversarial fixture covers only the optimizer
# core.  It is not an end-to-end condition certificate and therefore
# authorizes no strict adversarial setting.
STRICT_ADVERSARIAL_SETTINGS: tuple[tuple[str, str], ...] = ()
# Backward-compatible import used by older callers; the default suite remains
# clean plus the four random targets.
SETTINGS = RANDOM_SETTINGS
RESERVED_PASSTHROUGH_OPTIONS = {
    "--algorithm",
    "--algorithms",
    "--optional-baselines",
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
    "--evaluation-interval",
    "--eval-period",
    "--evaluation-episodes",
    "--eval-episodes",
    "--final-window-size",
}


def _csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _default_experiment_name(env_name: str) -> str:
    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    domain = env_name.split("-", 1)[0]
    return f"{domain}_research_benchmark_{stamp}_{str(uuid.uuid4())[:8]}"


def _selected_algorithms(args: argparse.Namespace) -> tuple[str, ...]:
    if args.suite_profile == "primary_research_benchmark":
        return tuple(
            algorithm
            for algorithm in args.algorithms
            if algorithm in strict_final_algorithms()
        )
    return (*tuple(args.algorithms), *tuple(args.optional_baselines))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the RPEX, RIQL-naive, and WSRL main benchmark. Cal-QL "
            "locomotion and shared-actor PQE are explicit optional results."
        )
    )
    parser.add_argument("--env-name", choices=BENCHMARK_ENVS, default=ENV_NAME)
    parser.add_argument(
        "--corruption-suite",
        choices=CORRUPTION_SUITES,
        default="random",
        help=(
            "random means clean plus the four replay-transition poisoning "
            "targets; adversarial includes only targets declared supported"
        ),
    )
    parser.add_argument(
        "--algorithms",
        type=_csv,
        default=list(MAIN_BASELINES),
        help="comma-separated subset of main baselines",
    )
    parser.add_argument(
        "--optional-baselines",
        type=_csv,
        default=[],
        help=(
            "explicit opt-in list: cal_ql_locomotion_adaptation and/or "
            "pqe_shared_actor_approx"
        ),
    )
    parser.add_argument("--seeds", type=_csv, default=["0"])
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
    parser.add_argument("--protocol", choices=PROTOCOLS, default=DEFAULT_PROTOCOL)
    parser.add_argument(
        "--implementation-profile", choices=IMPLEMENTATION_PROFILES
    )
    parser.add_argument(
        "--suite-profile", choices=SUITE_PROFILES,
        default="research_benchmark",
    )
    parser.add_argument(
        "--run-purpose", choices=RUN_PURPOSES, default="research_benchmark"
    )
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
    retired = {
        "pessimistic_q_ensemble",
        "pessimistic-q-ensemble",
        "pqe",
    }
    requested = (*tuple(args.algorithms), *tuple(args.optional_baselines))
    retired_requested = sorted(set(requested) & retired)
    if retired_requested:
        parser.error(
            "exact Pessimistic Q-Ensemble is not implemented; the shared-actor "
            "approximation must be explicitly selected as "
            "--optional-baselines pqe_shared_actor_approx"
        )
    retired_calql = sorted(set(requested) & {"cal_ql", "cal-ql", "calql"})
    if retired_calql:
        parser.error(
            "Cal-QL locomotion is a task adaptation; select it explicitly as "
            "--optional-baselines cal_ql_locomotion_adaptation"
        )
    unknown_main = sorted(set(args.algorithms) - set(MAIN_BASELINES))
    if unknown_main:
        parser.error(
            "--algorithms accepts only main baselines "
            f"{','.join(MAIN_BASELINES)}; invalid: {','.join(unknown_main)}"
        )
    unknown_optional = sorted(set(args.optional_baselines) - set(OPTIONAL_BASELINES))
    if unknown_optional:
        parser.error(
            "--optional-baselines accepts only "
            f"{','.join(OPTIONAL_BASELINES)}; invalid: {','.join(unknown_optional)}"
        )
    if not args.algorithms:
        parser.error("--algorithms cannot be empty for a research benchmark")
    if len(requested) != len(set(requested)):
        parser.error("algorithm selections cannot contain duplicates")
    if args.run_purpose == "research_benchmark" and args.suite_profile != "research_benchmark":
        parser.error(
            "--run-purpose research_benchmark requires "
            "--suite-profile research_benchmark"
        )
    if args.suite_profile == "research_benchmark" and args.run_purpose != "research_benchmark":
        parser.error(
            "--suite-profile research_benchmark requires "
            "--run-purpose research_benchmark"
        )
    if args.run_purpose == "research_benchmark":
        if args.implementation_profile is None:
            args.implementation_profile = "research_benchmark"
        elif args.implementation_profile != "research_benchmark":
            parser.error(
                "research_benchmark requires "
                "--implementation-profile research_benchmark"
            )
    selected_algorithms = _selected_algorithms(args)
    if (
        args.run_purpose in ("paper_reproduction", "final_benchmark")
        or args.suite_profile == "primary_research_benchmark"
    ):
        # Strict publication infrastructure stays entirely outside the normal
        # research path; importing it must never become a research prerequisite.
        from robust_o2o.final_gate import (
            ResearchLabelContractError,
            validate_research_label_contract,
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
    for name in ("eval_period", "eval_episodes", "final_window_size"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.suite_profile == "method_fidelity":
        parser.error(
            "method_fidelity 5x5 is unavailable: Cal-QL locomotion is a task "
            "port and the local PQE is pqe_shared_actor_approx. Use "
            "--suite-profile common_budget_diagnostic; no run will be mislabeled "
            "as paper reproduction."
        )
    if args.suite_profile == "primary_research_benchmark":
        if args.corruption_suite == "clean":
            parser.error(
                "primary_research_benchmark excludes clean: pinned RPEX has "
                "no official clean config row or condition certificate"
            )
        if args.corruption_suite in ("adversarial", "all"):
            parser.error(
                "primary_research_benchmark adversarial is unavailable: the "
                "registered fixture verifies only the optimizer core, not the "
                "end-to-end corruption condition"
            )
        if not selected_algorithms:
            parser.error(
                "primary_research_benchmark has no strict-eligible algorithms; "
                "complete an official-adapter or end-to-end parity certificate "
                "before launching this suite"
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
    if args.online_corruption_scale_profile is None:
        args.online_corruption_scale_profile = (
            "rpex_official_code"
            if args.suite_profile
            in ("research_benchmark", "primary_research_benchmark")
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
        if args.suite_profile
        in ("research_benchmark", "primary_research_benchmark")
        else "dataset_std_scaled_extension"
    )
    selected_algorithms = _selected_algorithms(args)
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
    random = STRICT_RANDOM_SETTINGS if strict else DIAGNOSTIC_RANDOM_SETTINGS
    adversarial = STRICT_ADVERSARIAL_SETTINGS if strict else ADVERSARIAL_SETTINGS
    if suite == "clean":
        return () if strict else CLEAN_SETTINGS
    if suite == "random":
        return random
    if suite == "adversarial":
        return adversarial
    if suite == "all":
        return (*random, *adversarial)
    raise ValueError(f"unknown corruption suite {suite!r}")


def main() -> int:
    parser = build_parser()
    args, passthrough = parser.parse_known_args()
    _validate_args(parser, args, passthrough)
    if args.run_purpose == "final_benchmark":
        from robust_o2o.final_gate import (
            FinalAuditGateError,
            require_final_benchmark_audit,
        )

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
    selected_algorithms = _selected_algorithms(args)
    selected_settings = settings_for_suite(
        args.corruption_suite,
        strict=args.suite_profile == "primary_research_benchmark",
    )
    total_runs = len(selected_algorithms) * len(selected_settings) * len(args.seeds)

    print(f"EXPERIMENT_NAME: {experiment_name}", flush=True)
    print(f"ENVIRONMENT: {args.env_name}", flush=True)
    print(f"ALGORITHMS: {', '.join(selected_algorithms)}", flush=True)
    print(f"MAIN_BASELINES: {', '.join(args.algorithms)}", flush=True)
    print(
        "OPTIONAL_BASELINES: "
        + (", ".join(args.optional_baselines) if args.optional_baselines else "none"),
        flush=True,
    )
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
            f"SCHEDULE: offline={args.offline_steps:,}, online={args.online_steps:,}, "
            f"eval_interval={args.eval_period:,}, eval_episodes={args.eval_episodes}, "
            f"final_window={args.final_window_size}",
            flush=True,
        )
    print(f"SUITE_PROFILE: {args.suite_profile}", flush=True)
    print(f"RUN_PURPOSE: {args.run_purpose}", flush=True)
    if args.run_purpose in ("smoke", "diagnostic"):
        print("NOT A PAPER REPRODUCTION RUN", flush=True)
        print("NOT PUBLICATION-ELIGIBLE", flush=True)
    elif args.run_purpose == "research_benchmark":
        print("CUSTOM RESEARCH BENCHMARK (NOT OFFICIAL PAPER REPRODUCTION)", flush=True)
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
