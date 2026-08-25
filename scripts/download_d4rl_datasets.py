#!/usr/bin/env python3
"""Download the official D4RL-v2 locomotion HDF5 datasets."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from robust_o2o.config import BENCHMARK_ENVS  # noqa: E402
from robust_o2o.environment import download_d4rl_dataset  # noqa: E402


DEFAULT_ENV_NAMES = tuple(
    env_name for env_name in BENCHMARK_ENVS if "-medium-replay-v2" in env_name
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--env-name",
        nargs="+",
        default=list(DEFAULT_ENV_NAMES),
        help=(
            "one or more D4RL-v2 environment IDs "
            "(default: the three medium-replay-v2 tasks)"
        ),
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        help="cache directory (default: ~/.d4rl/datasets)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="download again even when the destination file already exists",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    seen: set[str] = set()
    for env_name in args.env_name:
        if env_name in seen:
            continue
        seen.add(env_name)
        path = download_d4rl_dataset(
            env_name,
            dataset_dir=args.dataset_dir,
            force=args.force,
        )
        print(f"DATASET_READY: {env_name} -> {path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
