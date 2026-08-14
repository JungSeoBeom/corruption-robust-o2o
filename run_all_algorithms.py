#!/usr/bin/env python3
"""Run one environment/corruption configuration across every algorithm."""

from __future__ import annotations

import argparse
import csv
import json
import re
import shlex
import subprocess
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Iterable

from plot_results import update_comparison_plots, write_final_score_summary
from robust_o2o.config import (
    ALGORITHMS,
    BENCHMARK_ENVS,
    CORRUPTION_MODES,
    CORRUPTION_TARGETS,
    DEFAULT_PROTOCOL,
    LOCAL_PROTOCOL,
    PROTOCOLS,
    normalize_env_name,
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
    "cal_ql": "Cal-QL",
    "wsrl": "WSRL",
    "ro2o": "RO2O",
    "pessimistic_q_ensemble": "Pessimistic Q-Ensemble",
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


def _csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


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
    parser.add_argument("--algorithms", type=_csv, default=list(ALGORITHMS))
    parser.add_argument("--seeds", type=_csv, default=["0"])
    parser.add_argument("--stage", choices=("offline", "both"), default="both")
    parser.add_argument("--protocol", choices=PROTOCOLS, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output-root", default="results")
    parser.add_argument("--dataset-dir")
    parser.add_argument("--comparison-name")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--keep-going", action="store_true")
    return parser


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
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
    if not args.seeds:
        parser.error("--seeds cannot be empty")
    for seed in args.seeds:
        try:
            int(seed)
        except ValueError:
            parser.error(f"invalid seed: {seed!r}")
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
    )


def commands(
    args: argparse.Namespace,
    passthrough: Iterable[str],
    runs_dir: Path,
) -> Iterable[list[str]]:
    script = Path(__file__).resolve().parent / "run_experiment.py"
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
                "--output-dir",
                str(runs_dir),
            ]
            if args.dataset_dir:
                command.extend(("--dataset-dir", args.dataset_dir))
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


def main() -> int:
    parser = build_parser()
    args, passthrough = parser.parse_known_args()
    _validate_args(parser, args)
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
    for index, command in enumerate(generated_commands, start=1):
        print(f"[{index}/{len(generated_commands)}] {shlex.join(command)}", flush=True)
    if args.dry_run:
        return 0

    comparison_dir.mkdir(parents=True, exist_ok=False)
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
    try:
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
        }
    except Exception as exc:
        aggregation_error = f"{type(exc).__name__}: {exc}"
        print(f"Aggregation skipped: {aggregation_error}", file=sys.stderr)

    end_wall = datetime.now().astimezone()
    elapsed = time.perf_counter() - start_monotonic
    manifest = {
        "protocol": args.protocol,
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
    }
    with (comparison_dir / "manifest.json").open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, ensure_ascii=False)

    failures = sum(record["returncode"] != 0 for record in run_records)
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
