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
    ALGORITHMS,
    BENCHMARK_ENVS,
    CORRUPTION_MODES,
    CORRUPTION_TARGETS,
    DEFAULT_PROTOCOL,
    LOCAL_PROTOCOL,
    LEGACY_LOCAL_PROTOCOL_ALIAS,
    PROTOCOLS,
    normalize_env_name,
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


RESERVED_PASSTHROUGH_OPTIONS = {
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


def _csv(value: str):
    return [item.strip() for item in value.split(",") if item.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run or print an RPEX-benchmark algorithm/corruption matrix"
    )
    parser.add_argument("--algorithms", type=_csv, default=list(ALGORITHMS))
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
        default=["observations", "actions", "rewards", "dynamics"],
    )
    parser.add_argument("--seeds", type=_csv, default=["0"])
    parser.add_argument("--corruption-ranges", type=_csv, default=["1.0"])
    parser.add_argument("--stage", choices=("offline", "online", "both"), default="both")
    parser.add_argument("--protocol", choices=PROTOCOLS, default=DEFAULT_PROTOCOL)
    parser.add_argument("--implementation-profile", choices=IMPLEMENTATION_PROFILES)
    parser.add_argument(
        "--suite-profile", choices=SUITE_PROFILES,
        default="common_budget_robustness",
    )
    parser.add_argument("--run-purpose", choices=RUN_PURPOSES, default="diagnostic")
    parser.add_argument(
        "--online-corruption-scale-profile",
        choices=ONLINE_CORRUPTION_SCALE_PROFILES,
    )
    parser.add_argument("--allow-diagnostic-protocol", action="store_true")
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
        if args.suite_profile in ("method_fidelity", "primary_research_benchmark")
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
                "--protocol",
                args.protocol,
                "--suite-profile",
                args.suite_profile,
                "--run-purpose",
                args.run_purpose,
                "--online-corruption-scale-profile",
                scale_profile,
                "--corruption-range",
                corruption_range,
                "--output-dir",
                args.output_dir,
                "--comparison-name",
                comparison_name,
                *(["--allow-diagnostic-protocol"] if args.allow_diagnostic_protocol else []),
                *passthrough,
            ]
            if args.implementation_profile:
                command.extend(("--implementation-profile", args.implementation_profile))
            if args.run_purpose == "final_benchmark":
                command.extend(
                    (
                        "--benchmark-seed-set",
                        *(str(seed) for seed in STRICT_FINAL_SEEDS),
                    )
                )
            yield command


def _validate_args(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    passthrough: list[str],
) -> None:
    if args.protocol == LEGACY_LOCAL_PROTOCOL_ALIAS:
        args.protocol = LOCAL_PROTOCOL

    unknown_algorithms = sorted(set(args.algorithms) - set(ALGORITHMS))
    if unknown_algorithms:
        parser.error(f"unknown algorithms: {', '.join(unknown_algorithms)}")
    if not args.algorithms:
        parser.error("--algorithms cannot be empty")
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

    parsed_seeds: list[int] = []
    if not args.seeds:
        parser.error("--seeds cannot be empty")
    for seed in args.seeds:
        try:
            parsed_seeds.append(int(seed))
        except ValueError:
            parser.error(f"invalid seed: {seed!r}")

    conflicts = sorted(
        option
        for option in passthrough
        if option.split("=", 1)[0] in RESERVED_PASSTHROUGH_OPTIONS
    )
    if conflicts:
        parser.error(
            "these child identity/provenance options cannot be overridden: "
            + ", ".join(conflicts)
        )
    try:
        validate_research_label_contract(
            args.run_purpose,
            args.suite_profile,
            args.algorithms,
        )
    except ResearchLabelContractError as exc:
        parser.error(str(exc))

    if args.protocol in (LOCAL_PROTOCOL, "local_gymnasium_v4") and not args.allow_diagnostic_protocol:
        parser.error(
            "the local Gymnasium protocol is diagnostic-only; pass "
            "--allow-diagnostic-protocol to acknowledge this"
        )
    for value in args.corruption_ranges:
        try:
            parsed = float(value)
        except ValueError:
            parser.error(f"invalid corruption range: {value!r}")
        if parsed < 0.0:
            parser.error("--corruption-ranges cannot contain negative values")

    if args.run_purpose == "final_benchmark":
        if tuple(parsed_seeds) != STRICT_FINAL_SEEDS:
            parser.error("final_benchmark requires exactly ordered seeds 0,1,2,3,4")
        if args.suite_profile != "primary_research_benchmark":
            parser.error(
                "final_benchmark requires --suite-profile "
                "primary_research_benchmark"
            )
        if args.protocol != DEFAULT_PROTOCOL:
            parser.error(
                "final_benchmark requires rpex_d4rl_v2_legacy; no local fallback"
            )
        if args.stage != "both":
            parser.error("final_benchmark requires --stage both")
        if args.implementation_profile not in (None, "official_code_reference"):
            parser.error(
                "final_benchmark requires --implementation-profile "
                "official_code_reference"
            )
        if args.online_corruption_scale_profile not in (
            None,
            "rpex_official_code",
        ):
            parser.error(
                "final_benchmark requires --online-corruption-scale-profile "
                "rpex_official_code"
            )
        if any(float(value) != 1.0 for value in args.corruption_ranges):
            parser.error("final_benchmark requires exactly --corruption-ranges 1.0")
        unsupported_tasks = sorted(set(args.envs) - set(STRICT_FINAL_TASKS))
        if unsupported_tasks:
            parser.error(
                "final_benchmark permits only medium-replay-v2 tasks: "
                + ", ".join(unsupported_tasks)
            )
        forbidden = sorted(
            set(args.algorithms) - set(strict_final_algorithms())
        )
        if forbidden:
            parser.error(
                "final_benchmark rejects non-allowlisted baselines: "
                + ", ".join(forbidden)
            )
        if (
            "adversarial" in args.corruptions
            and set(args.targets) - {"observations"}
        ):
            parser.error(
                "final_benchmark adversarial corruption is certified only for "
                "the observations target"
            )
        if (
            "random" in args.corruptions
            and set(args.targets) - {"observations", "actions", "rewards", "dynamics"}
        ):
            parser.error(
                "final_benchmark random corruption requires certified individual "
                "targets (observations/actions/rewards/dynamics)"
            )


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
    try:
        require_final_benchmark_audit(
            args.run_purpose,
            dry_run=args.dry_run,
        )
    except FinalAuditGateError as exc:
        print(f"FINAL_BENCHMARK_AUDIT_GATE_FAILED: {exc}", file=sys.stderr)
        return 2
    if args.run_purpose in ("smoke", "diagnostic"):
        print("NOT A PAPER REPRODUCTION RUN", flush=True)
        print("NOT PUBLICATION-ELIGIBLE", flush=True)
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
