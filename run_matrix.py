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
    CORRUPTION_MODES,
    DEFAULT_PROTOCOL,
    LOCAL_PROTOCOL,
    LEGACY_LOCAL_PROTOCOL_ALIAS,
    PROTOCOLS,
)
from robust_o2o.fidelity import IMPLEMENTATION_PROFILES, SUITE_PROFILES


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
    parser.add_argument("--stage", choices=("offline", "online", "both"), default="both")
    parser.add_argument("--protocol", choices=PROTOCOLS, default=DEFAULT_PROTOCOL)
    parser.add_argument("--implementation-profile", choices=IMPLEMENTATION_PROFILES)
    parser.add_argument(
        "--suite-profile", choices=SUITE_PROFILES,
        default="common_budget_robustness",
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
    for algorithm, env_name, corruption, seed in itertools.product(
        args.algorithms, args.envs, args.corruptions, args.seeds
    ):
        targets = ["none"] if corruption == "clean" else args.targets
        for target in targets:
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
                "--output-dir",
                args.output_dir,
                "--comparison-name",
                comparison_name,
                *(["--allow-diagnostic-protocol"] if args.allow_diagnostic_protocol else []),
                *passthrough,
            ]
            if args.implementation_profile:
                command.extend(("--implementation-profile", args.implementation_profile))
            yield command


def main() -> int:
    parser = build_parser()
    args, passthrough = parser.parse_known_args()
    if args.protocol == LEGACY_LOCAL_PROTOCOL_ALIAS:
        args.protocol = LOCAL_PROTOCOL
    if args.protocol in (LOCAL_PROTOCOL, "local_gymnasium_v4") and not args.allow_diagnostic_protocol:
        parser.error(
            "the local Gymnasium protocol is diagnostic-only; pass "
            "--allow-diagnostic-protocol to acknowledge this"
        )
    comparison_name = args.comparison_name
    if not comparison_name:
        stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
        comparison_name = f"matrix_{stamp}_{str(uuid.uuid4())[:8]}"
    failures = 0
    for index, command in enumerate(
        commands(args, passthrough, comparison_name), start=1
    ):
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
