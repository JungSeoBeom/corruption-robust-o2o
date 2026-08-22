#!/usr/bin/env python3
"""Run one environment/corruption configuration across every algorithm."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shlex
import subprocess
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Iterable

from plot_results import (
    update_comparison_plots,
    write_final_score_summary,
    write_reproduction_summaries,
)
from robust_o2o.config import (
    ALGORITHM_ALIASES,
    ALGORITHMS,
    BENCHMARK_ENVS,
    CORRUPTION_MODES,
    CORRUPTION_TARGETS,
    DEFAULT_PROTOCOL,
    LOCAL_PROTOCOL,
    LEGACY_LOCAL_PROTOCOL_ALIAS,
    PROTOCOLS,
    RESEARCH_BENCHMARK_PROTOCOL_ERROR,
    is_inflight_pre_gate_run55_descendant,
    normalize_env_name,
)
from robust_o2o.fidelity import (
    IMPLEMENTATION_PROFILES,
    MAIN_BASELINES,
    ONLINE_CORRUPTION_SCALE_PROFILES,
    RUN_PURPOSES,
    STRICT_FINAL_SEEDS,
    SUITE_PROFILES,
    STRICT_FINAL_TASKS,
    strict_final_algorithms,
)
from robust_o2o.environment import preflight_runtime
from robust_o2o.logging_utils import format_duration, format_timestamp
from robust_o2o.paths import comparison_directory


ALGORITHM_DISPLAY_NAMES = {
    "rpex": "RPEX",
    "riql_pex": "RIQL+PEX",
    "riql_naive": "RIQL naive",
    "uwmsg": "UWMSG",
    "pex": "PEX",
    "cal_ql": "Cal-QL locomotion adaptation",
    "wsrl": "WSRL",
    "ro2o": "RO2O",
    "pessimistic_q_ensemble": "Pessimistic Q-Ensemble (D4RL-v2 port)",
}

TIMING_FIELDS = (
    "algorithm",
    "algorithm_name",
    "seed",
    "status",
    "start_time",
    "end_time",
    "elapsed_hms",
    "elapsed_seconds",
    "returncode",
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

def _flatten_cli_values(values: Iterable[str]) -> list[str]:
    """Accept comma-separated, space-separated, or mixed CLI lists."""

    return [
        item.strip()
        for value in values
        for item in value.split(",")
        if item.strip()
    ]


def _canonical_algorithms(values: Iterable[str]) -> list[str]:
    return [
        ALGORITHM_ALIASES.get(value.strip().lower(), value.strip().lower())
        for value in _flatten_cli_values(values)
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run all algorithm classes for one environment/corruption setting "
            "and aggregate their CSV/plot results"
        )
    )
    parser.add_argument("--env-name", required=True)
    parser.add_argument("--corruption", choices=CORRUPTION_MODES, required=True)
    parser.add_argument(
        "--corruption-target", choices=CORRUPTION_TARGETS, default="none"
    )
    parser.add_argument(
        "--mixed-ratios",
        type=float,
        nargs=4,
        metavar=("OBS", "ACT", "REW", "DYN"),
        default=(0.25, 0.25, 0.25, 0.25),
    )
    parser.add_argument(
        "--algorithms",
        nargs="+",
        help="algorithm names separated by commas, spaces, or both",
    )
    parser.add_argument("--seeds", nargs="+", default=["0"])
    parser.add_argument(
        "--benchmark-seed-set",
        type=int,
        nargs="+",
        help="controller-declared strict cohort propagated to child runs",
    )
    parser.add_argument("--stage", choices=("offline", "both"), default="both")
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
    parser.add_argument("--output-root", default="results")
    parser.add_argument("--dataset-dir")
    parser.add_argument("--comparison-name")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--keep-going", action="store_true")
    return parser


def _validate_args(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    passthrough: Iterable[str] = (),
) -> None:
    if args.algorithms is None:
        args.algorithms = list(MAIN_BASELINES)
    else:
        args.algorithms = _canonical_algorithms(args.algorithms)
    args.seeds = _flatten_cli_values(args.seeds)
    if args.protocol == LEGACY_LOCAL_PROTOCOL_ALIAS:
        args.protocol = LOCAL_PROTOCOL
    original_env_name = args.env_name
    args.env_name = normalize_env_name(args.env_name)
    if args.env_name != original_env_name:
        print(
            f"NORMALIZED_ENV_NAME: {original_env_name} -> {args.env_name}",
            flush=True,
        )
    if args.env_name not in BENCHMARK_ENVS:
        parser.error(
            f"unknown benchmark environment {args.env_name!r}; "
            f"choose from {', '.join(BENCHMARK_ENVS)}"
        )
    unknown_algorithms = sorted(set(args.algorithms) - set(ALGORITHMS))
    if unknown_algorithms:
        parser.error(f"unknown algorithms: {', '.join(unknown_algorithms)}")
    if not args.algorithms:
        parser.error("--algorithms cannot be empty")
    if len(args.algorithms) != len(set(args.algorithms)):
        parser.error(
            "algorithm selections cannot contain duplicates after alias normalization"
        )
    if args.run_purpose == "research_benchmark":
        if args.suite_profile != "research_benchmark":
            parser.error(
                "--run-purpose research_benchmark requires "
                "--suite-profile research_benchmark"
            )
        if args.implementation_profile is None:
            args.implementation_profile = "research_benchmark"
        elif args.implementation_profile != "research_benchmark":
            parser.error(
                "research_benchmark requires "
                "--implementation-profile research_benchmark"
            )
        unsupported_research = sorted(
            set(args.algorithms) - set(MAIN_BASELINES)
        )
        if unsupported_research:
            parser.error(
                "research_benchmark supports only the five main baselines "
                f"{','.join(MAIN_BASELINES)}; invalid: "
                + ",".join(unsupported_research)
            )
    if not args.seeds:
        parser.error("--seeds cannot be empty")
    for seed in args.seeds:
        try:
            int(seed)
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
    if (
        args.run_purpose == "final_benchmark"
        or args.suite_profile == "primary_research_benchmark"
    ):
        from robust_o2o.final_gate import (
            ResearchLabelContractError,
            validate_research_label_contract,
        )

        try:
            validate_research_label_contract(
                args.run_purpose,
                args.suite_profile,
                args.algorithms,
            )
        except ResearchLabelContractError as exc:
            parser.error(str(exc))
    if args.run_purpose == "final_benchmark":
        required = {str(seed) for seed in STRICT_FINAL_SEEDS}
        if set(args.seeds) != required or len(args.seeds) != len(required):
            parser.error(
                "final_benchmark requires exactly seeds 0,1,2,3,4; "
                "single-seed runs are smoke/diagnostic only"
            )
        declared_seed_set = tuple(
            args.benchmark_seed_set
            if args.benchmark_seed_set is not None
            else (int(seed) for seed in args.seeds)
        )
        if declared_seed_set != STRICT_FINAL_SEEDS:
            parser.error(
                "final_benchmark requires --benchmark-seed-set 0 1 2 3 4 "
                "(it is inferred from --seeds when omitted)"
            )
        args.benchmark_seed_set = list(STRICT_FINAL_SEEDS)
        if args.suite_profile != "primary_research_benchmark":
            parser.error(
                "final_benchmark requires "
                "--suite-profile primary_research_benchmark"
            )
        if args.protocol != DEFAULT_PROTOCOL:
            parser.error(
                "final_benchmark requires rpex_d4rl_v2_legacy; no local fallback"
            )
        if args.stage != "both":
            parser.error("final_benchmark requires --stage both")
        if args.env_name not in STRICT_FINAL_TASKS:
            parser.error(
                "final_benchmark permits only hopper/halfcheetah/walker2d "
                "medium-replay-v2 tasks"
            )
        forbidden = sorted(
            set(args.algorithms) - set(strict_final_algorithms())
        )
        if forbidden:
            parser.error(
                "final_benchmark rejects non-exact/non-allowlisted baselines: "
                + ", ".join(forbidden)
            )
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
        if args.corruption not in ("clean", "random", "adversarial"):
            parser.error(
                "final_benchmark permits only clean, random, or certified "
                "adversarial corruption"
            )
        if (
            args.corruption == "adversarial"
            and args.corruption_target != "observations"
        ):
            parser.error(
                "final_benchmark adversarial corruption is certified only for "
                "the observations target"
            )
        if (
            args.corruption == "adversarial"
            and args.env_name != "hopper-medium-replay-v2"
        ):
            parser.error(
                "final_benchmark adversarial observations currently require "
                "hopper-medium-replay-v2: the registered optimizer-core "
                "fixture is bound to the Hopper EDAC checkpoint"
            )
    if args.suite_profile == "primary_research_benchmark":
        forbidden = sorted(
            set(args.algorithms) - set(strict_final_algorithms())
        )
        if forbidden:
            parser.error(
                "primary_research_benchmark excludes non-allowlisted ports: "
                + ", ".join(forbidden)
                + "; use common_budget_diagnostic for these algorithms"
            )
    if args.online_corruption_scale_profile is None:
        args.online_corruption_scale_profile = (
            "rpex_official_code"
            if args.suite_profile in (
                "research_benchmark",
                "method_fidelity",
                "primary_research_benchmark",
            )
            else "dataset_std_scaled_extension"
        )
    if args.corruption == "clean":
        args.corruption_target = "none"
    elif args.corruption_target == "none":
        parser.error("random/adversarial corruption requires --corruption-target")
    if any(ratio < 0.0 for ratio in args.mixed_ratios):
        parser.error("--mixed-ratios cannot contain negative values")
    if abs(sum(args.mixed_ratios) - 1.0) > 1e-6:
        parser.error("--mixed-ratios must sum to 1.0")
    if args.comparison_name:
        if (
            not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", args.comparison_name)
            or args.comparison_name in (".", "..")
        ):
            parser.error(
                "--comparison-name may contain only letters, digits, '.', '_', "
                "and '-', and must start with a letter or digit"
            )
    if (
        args.run_purpose == "research_benchmark"
        and args.protocol == LOCAL_PROTOCOL
        and not is_inflight_pre_gate_run55_descendant()
    ):
        parser.error(RESEARCH_BENCHMARK_PROTOCOL_ERROR)
    if args.protocol in (LOCAL_PROTOCOL, "local_gymnasium_v4") and not args.allow_diagnostic_protocol:
        parser.error(
            "the local Gymnasium protocol is diagnostic-only; pass "
            "--allow-diagnostic-protocol to acknowledge this"
        )


def _comparison_directory(args: argparse.Namespace) -> Path:
    name = args.comparison_name
    if not name:
        stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
        name = f"{stamp}_{str(uuid.uuid4())[:8]}"
    return comparison_directory(
        args.output_root,
        args.env_name,
        args.corruption,
        args.corruption_target,
        name,
        args.protocol,
        (
            f"{args.run_purpose}__{args.suite_profile}__"
            f"{args.implementation_profile or 'auto'}"
        ),
    )


def commands(
    args: argparse.Namespace,
    passthrough: Iterable[str],
    runs_dir: Path,
) -> Iterable[list[str]]:
    script = Path(__file__).resolve().parent / "run_experiment.py"
    scale_profile = args.online_corruption_scale_profile or (
        "rpex_official_code"
        if args.suite_profile
        in ("research_benchmark", "method_fidelity", "primary_research_benchmark")
        else "dataset_std_scaled_extension"
    )
    for algorithm in args.algorithms:
        for seed in args.seeds:
            command = [
                sys.executable,
                str(script),
                "--algorithm",
                algorithm,
                "--env-name",
                args.env_name,
                "--corruption",
                args.corruption,
                "--corruption-target",
                args.corruption_target,
                "--mixed-ratios",
                *(f"{ratio:g}" for ratio in args.mixed_ratios),
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
                "--output-dir",
                str(runs_dir),
            ]
            if args.implementation_profile:
                command.extend(
                    ("--implementation-profile", args.implementation_profile)
                )
            if args.allow_diagnostic_protocol:
                command.append("--allow-diagnostic-protocol")
            if args.dataset_dir:
                command.extend(("--dataset-dir", args.dataset_dir))
            if args.run_purpose == "final_benchmark":
                declared_seed_set = (
                    args.benchmark_seed_set or list(STRICT_FINAL_SEEDS)
                )
                command.extend(
                    (
                        "--benchmark-seed-set",
                        *(str(seed) for seed in declared_seed_set),
                    )
                )
            command.extend(passthrough)
            yield command


def summarize_algorithm_timings(
    records: list[dict],
    algorithm_order: Iterable[str],
) -> list[dict]:
    summaries = []
    for algorithm in algorithm_order:
        rows = [record for record in records if record["algorithm"] == algorithm]
        if not rows:
            continue
        elapsed = sum(float(record["elapsed_seconds"]) for record in rows)
        failed_runs = sum(record["returncode"] != 0 for record in rows)
        summaries.append(
            {
                "algorithm": algorithm,
                "algorithm_name": ALGORITHM_DISPLAY_NAMES[algorithm],
                "start_time": rows[0]["start_time"],
                "end_time": rows[-1]["end_time"],
                "elapsed_seconds": elapsed,
                "elapsed_hms": format_duration(elapsed),
                "runs": len(rows),
                "completed_runs": len(rows) - failed_runs,
                "failed_runs": failed_runs,
                "status": "completed" if failed_runs == 0 else "failed",
            }
        )
    return summaries


def write_timing_csv(path: Path, records: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=TIMING_FIELDS)
        writer.writeheader()
        for record in records:
            writer.writerow({field: record[field] for field in TIMING_FIELDS})
    return path


def _print_algorithm_summary(summary: dict, prefix: str) -> None:
    print(
        f"{prefix}: {summary['algorithm_name']} ({summary['algorithm']}) | "
        f"runs={summary['completed_runs']}/{summary['runs']} completed | "
        f"START={summary['start_time']} | END={summary['end_time']} | "
        f"ELAPSED={summary['elapsed_hms']} "
        f"({summary['elapsed_seconds']:.3f} seconds)",
        flush=True,
    )


def _remove_invalid_canonical_artifacts(comparison_dir: Path) -> None:
    """Remove canonical-looking outputs after any incomplete aggregation."""

    names = {
        "final_scores.csv",
        "per_seed_final_scores.csv",
        "paper_reproduction_summary.csv",
        "common_per_seed_final_scores.csv",
        "common_benchmark_summary.csv",
        "seed_run_status.csv",
        "research_per_seed_final_scores.csv",
        "research_summary.csv",
        "adapted_baselines_per_seed_final_scores.csv",
        "adapted_baselines_summary.csv",
        "diagnostic_per_seed_final_scores.csv",
        "diagnostic_summary.csv",
    }
    for phase in ("offline_online", "offline", "online"):
        names.add(f"comparison_{phase}.png")
        names.add(f"comparison_{phase}.csv")
    for name in names:
        (comparison_dir / name).unlink(missing_ok=True)


def main() -> int:
    parser = build_parser()
    args, passthrough = parser.parse_known_args()
    _validate_args(parser, args, passthrough)
    audit_receipt = None
    if args.run_purpose == "final_benchmark":
        from robust_o2o.final_gate import (
            FinalAuditGateError,
            require_final_benchmark_audit,
        )

        try:
            audit_receipt = require_final_benchmark_audit(
                args.run_purpose,
                dry_run=args.dry_run,
            )
        except FinalAuditGateError as exc:
            print(f"FINAL_BENCHMARK_AUDIT_GATE_FAILED: {exc}", file=sys.stderr)
            return 2
    comparison_dir = _comparison_directory(args)
    runs_dir = comparison_dir / "runs"
    generated_commands = list(commands(args, passthrough, runs_dir))

    if not args.dry_run:
        try:
            preflight_metadata = preflight_runtime(
                args.env_name, args.dataset_dir, args.protocol
            )
        except Exception as exc:
            print(
                f"PREFLIGHT_FAILED: {type(exc).__name__}: {exc}",
                file=sys.stderr,
                flush=True,
            )
            if args.protocol == LOCAL_PROTOCOL:
                guidance = (
                    "Activate the `corruption` Conda environment and verify the "
                    "D4RL-v2 HDF5 dataset under ~/.d4rl/datasets."
                )
            else:
                guidance = (
                    "Create the pinned environment with `conda env create -f "
                    "environment-rpex-v2.yml` and retry."
                )
            print(guidance, file=sys.stderr, flush=True)
            return 2
        print(
            f"PROTOCOL: {preflight_metadata['protocol']}",
            flush=True,
        )
        print(
            f"D4RL_ENV_ID: {preflight_metadata['d4rl_env_id']}",
            flush=True,
        )
        print(
            f"DATASET_PATH: {preflight_metadata.get('dataset_path') or 'unavailable'}",
            flush=True,
        )

    print(f"COMPARISON_DIR: {comparison_dir}", flush=True)
    if args.run_purpose in ("smoke", "diagnostic"):
        print("NOT A PAPER REPRODUCTION RUN", flush=True)
        print("NOT PUBLICATION-ELIGIBLE", flush=True)
    elif args.run_purpose == "research_benchmark":
        print(
            "CUSTOM RESEARCH BENCHMARK (NOT OFFICIAL PAPER REPRODUCTION)",
            flush=True,
        )
    for index, command in enumerate(generated_commands, start=1):
        print(f"[{index}/{len(generated_commands)}] {shlex.join(command)}", flush=True)
    if args.dry_run:
        return 0

    comparison_dir.mkdir(parents=True, exist_ok=False)
    audit_evidence_path = None
    if audit_receipt is not None:
        from robust_o2o.final_gate import write_final_audit_evidence

        audit_evidence_path = write_final_audit_evidence(
            comparison_dir, audit_receipt
        )
    start_wall = datetime.now().astimezone()
    start_monotonic = time.perf_counter()
    run_records = []
    for index, command in enumerate(generated_commands, start=1):
        algorithm_index = (index - 1) // len(args.seeds)
        seed_index = (index - 1) % len(args.seeds)
        algorithm = args.algorithms[algorithm_index]
        seed = int(args.seeds[seed_index])
        print(f"RUNNING [{index}/{len(generated_commands)}]", flush=True)
        run_start_wall = datetime.now().astimezone()
        run_start_monotonic = time.perf_counter()
        result = subprocess.run(command, check=False)
        run_end_wall = datetime.now().astimezone()
        run_elapsed = time.perf_counter() - run_start_monotonic
        run_status = "completed" if result.returncode == 0 else "failed"
        run_records.append(
            {
                "algorithm": algorithm,
                "algorithm_name": ALGORITHM_DISPLAY_NAMES[algorithm],
                "seed": seed,
                "status": run_status,
                "start_time": format_timestamp(run_start_wall),
                "end_time": format_timestamp(run_end_wall),
                "elapsed_seconds": run_elapsed,
                "elapsed_hms": format_duration(run_elapsed),
                "command": command,
                "returncode": result.returncode,
            }
        )
        print(
            f"RUN_FINISHED: {ALGORITHM_DISPLAY_NAMES[algorithm]} "
            f"({algorithm}) seed={seed} status={run_status} | "
            f"ELAPSED={format_duration(run_elapsed)} "
            f"({run_elapsed:.3f} seconds)",
            flush=True,
        )
        algorithm_finished = seed_index == len(args.seeds) - 1
        stopping_after_failure = result.returncode != 0 and not args.keep_going
        if algorithm_finished or stopping_after_failure:
            current_summary = summarize_algorithm_timings(
                run_records, (algorithm,)
            )[0]
            _print_algorithm_summary(current_summary, "ALGORITHM_FINISHED")
        if result.returncode != 0 and not args.keep_going:
            break

    algorithm_timings = summarize_algorithm_timings(run_records, args.algorithms)
    timing_path = write_timing_csv(comparison_dir / "timing.csv", run_records)
    phase = "online" if args.stage == "both" else "offline"
    artifacts = {"timing_csv": str(timing_path)}
    aggregation_error = None
    failures = sum(record["returncode"] != 0 for record in run_records)
    incomplete_runs = len(run_records) != len(generated_commands)
    if failures or incomplete_runs:
        aggregation_error = (
            f"suite is incomplete: completed controller records="
            f"{len(run_records)}/{len(generated_commands)}, failed={failures}; "
            "canonical result artifacts were not published"
        )
    else:
        try:
            # Validate reporting first. In strict mode this catches missing or
            # duplicate seeds/evaluations before canonical plots or the common
            # final_scores alias are published.
            reporting_paths = write_reproduction_summaries(
                runs_dir,
                comparison_dir,
                args.env_name,
                args.corruption,
                args.corruption_target,
                strict=args.run_purpose == "final_benchmark",
                expected_seeds=(
                    [int(seed) for seed in args.seeds]
                    if args.run_purpose == "final_benchmark"
                    else None
                ),
                phase=phase,
            )
            plot_paths = update_comparison_plots(
                comparison_dir,
                args.env_name,
                args.corruption,
                args.corruption_target,
            )
            final_scores_path = write_final_score_summary(
                runs_dir,
                comparison_dir / "final_scores.csv",
                args.env_name,
                args.corruption,
                args.corruption_target,
                phase,
            )
            artifacts = {
                **artifacts,
                "plots": {name: str(path) for name, path in plot_paths.items()},
                "curve_csvs": {
                    name: str(path.with_suffix(".csv"))
                    for name, path in plot_paths.items()
                },
                "final_scores_csv": str(final_scores_path),
                "reporting_csvs": {
                    name: str(path) for name, path in reporting_paths.items()
                },
            }
        except Exception as exc:
            aggregation_error = f"{type(exc).__name__}: {exc}"
    if aggregation_error:
        print(f"Aggregation skipped: {aggregation_error}", file=sys.stderr)
        _remove_invalid_canonical_artifacts(comparison_dir)

    end_wall = datetime.now().astimezone()
    elapsed = time.perf_counter() - start_monotonic
    manifest = {
        "protocol": args.protocol,
        "implementation_profile": args.implementation_profile or "auto",
        "suite_profile": args.suite_profile,
        "run_purpose": args.run_purpose,
        "final_audit": None,
        "online_corruption_scale_profile": args.online_corruption_scale_profile,
        "environment": args.env_name,
        "corruption": args.corruption,
        "corruption_target": args.corruption_target,
        "mixed_ratios": list(args.mixed_ratios),
        "algorithms": args.algorithms,
        "seeds": [int(seed) for seed in args.seeds],
        "stage": args.stage,
        "preflight": preflight_metadata,
        "environment_backend": preflight_metadata["environment_backend"],
        "dataset_backend": preflight_metadata["dataset_backend"],
        "dataset_path": preflight_metadata.get("dataset_path"),
        "start_time": format_timestamp(start_wall),
        "end_time": format_timestamp(end_wall),
        "timezone": str(start_wall.tzinfo),
        "elapsed_seconds": elapsed,
        "elapsed_hms": format_duration(elapsed),
        "runs": run_records,
        "algorithm_timings": algorithm_timings,
        "artifacts": artifacts,
        "aggregation_error": aggregation_error,
        "benchmark_valid": not failures and aggregation_error is None,
    }
    if audit_receipt is not None:
        from robust_o2o.final_gate import (
            AUDIT_RECEIPT_ENV,
            AUDIT_RECEIPT_SHA256_ENV,
        )

        manifest["final_audit"] = {
            "context_token": audit_receipt["context_token"],
            "issued_at_utc": audit_receipt["issued_at_utc"],
            "receipt_source": os.environ.get(AUDIT_RECEIPT_ENV),
            "receipt_sha256": os.environ.get(AUDIT_RECEIPT_SHA256_ENV),
            "evidence_path": str(audit_evidence_path),
            "audit_result": audit_receipt["audit_result"],
        }
    with (comparison_dir / "manifest.json").open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, ensure_ascii=False)

    print(f"COMPARISON_DIR: {comparison_dir}", flush=True)
    print("ALGORITHM_TIMING_SUMMARY:", flush=True)
    for algorithm_summary in algorithm_timings:
        _print_algorithm_summary(algorithm_summary, "  ALGORITHM")
    # Keep these as the final three lines of the all-algorithms command.
    print(f"START_TIME: {format_timestamp(start_wall)}", flush=True)
    print(f"END_TIME: {format_timestamp(end_wall)}", flush=True)
    print(
        f"ELAPSED: {format_duration(elapsed)} ({elapsed:.3f} seconds)",
        flush=True,
    )
    return 1 if failures or aggregation_error else 0


if __name__ == "__main__":
    sys.exit(main())
