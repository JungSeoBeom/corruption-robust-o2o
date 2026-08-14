#!/usr/bin/env python3
"""Integration smoke test for the pinned RPEX/D4RL-v2 legacy protocol."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from robust_o2o.config import BENCHMARK_ENVS  # noqa: E402
from robust_o2o.device import seed_everything  # noqa: E402
from robust_o2o.environment import (  # noqa: E402
    environment_metadata,
    load_d4rl_dataset,
    make_env,
    normalized_d4rl_scores,
    reset_env,
    runtime_package_versions,
    step_env,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-name", choices=BENCHMARK_ENVS, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dataset-dir")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    env = make_env(args.env_name)
    try:
        seed_everything(args.seed, env)
        observation = reset_env(env, seed=args.seed)
        action = env.action_space.sample()
        next_observation, reward, terminated, truncated, _ = step_env(env, action)
        dataset = load_d4rl_dataset(env, args.dataset_dir)
        metadata = environment_metadata(env, args.env_name, dataset, args.seed)
        normalized_zero = normalized_d4rl_scores(
            args.env_name, np.asarray([0.0], dtype=np.float64)
        )
        if observation.shape != env.observation_space.shape:
            raise RuntimeError("reset observation shape does not match observation_space")
        if next_observation.shape != env.observation_space.shape:
            raise RuntimeError("step observation shape does not match observation_space")
        report = {
            **metadata,
            "package_versions": runtime_package_versions(),
            "sample_reward": reward,
            "sample_terminated": terminated,
            "sample_truncated": truncated,
            "normalized_score_for_return_0": float(normalized_zero[0]),
        }
        print(json.dumps(report, indent=2, ensure_ascii=False))
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
