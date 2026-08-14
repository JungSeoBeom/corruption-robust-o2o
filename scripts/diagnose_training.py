#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import traceback
from datetime import datetime
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from robust_o2o.config import ALGORITHMS, ExperimentConfig, LOCAL_PROTOCOL  # noqa: E402
from robust_o2o.experiment import run_experiment  # noqa: E402
from robust_o2o.logging_utils import RunLogger  # noqa: E402


STATUS_CODES = (
    "PASS",
    "FAIL_CODE_INVARIANT",
    "FAIL_NUMERICAL",
    "FAIL_NO_PARAMETER_UPDATE",
    "FAIL_NO_RETURN_IMPROVEMENT",
    "LIKELY_HYPERPARAMETER_ISSUE",
    "PROTOCOL_MISMATCH",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Quick training-health diagnostic")
    parser.add_argument("--algorithm", choices=ALGORITHMS, default="riql_naive")
    parser.add_argument("--env", default="hopper-medium-replay-v2")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--offline-steps", type=int, default=20)
    parser.add_argument("--online-steps", type=int, default=20)
    parser.add_argument("--corruption-rate", type=float, default=0.0)
    parser.add_argument(
        "--corruption-target",
        choices=("none", "observations", "actions", "rewards", "dynamics"),
        default="none",
    )
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--output-dir", default="results")
    return parser


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _read_evaluations(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _classify(
    run_dir: Path, error: BaseException | None, require_replay_match: bool
) -> dict:
    if error is not None:
        message = f"{type(error).__name__}: {error}"
        code = "PROTOCOL_MISMATCH" if "protocol" in message.lower() else "FAIL_CODE_INVARIANT"
        return {"status": code, "reason": message}

    train_rows = _read_jsonl(run_dir / "train_metrics.jsonl")
    eval_rows = _read_evaluations(run_dir / "metrics.csv")
    numeric_values = []
    for row in train_rows:
        numeric_values.extend(
            float(value)
            for key, value in row.items()
            if key not in ("timestamp", "phase") and isinstance(value, (int, float))
        )
    if any(not math.isfinite(value) for value in numeric_values):
        return {"status": "FAIL_NUMERICAL", "reason": "non-finite train metric"}
    updates = max((int(row.get("updates", 0)) for row in train_rows), default=0)
    if updates == 0:
        return {
            "status": "FAIL_NO_PARAMETER_UPDATE",
            "reason": "no optimizer update was recorded",
        }
    returns = [
        float(row["normalized_return_mean"])
        for row in eval_rows
        if row.get("normalized_return_mean") not in (None, "")
    ]
    # A tiny one-episode diagnostic is a correctness gate, not a statistically
    # powered benchmark. Only classify a material collapse here.
    material_return_collapse = (
        len(returns) >= 2 and returns[-1] < returns[0] - 10.0
    )
    action_oob = max(
        (float(row.get("executed_action_oob_fraction", 0.0)) for row in train_rows),
        default=0.0,
    )
    clean_mismatch = max(
        (
            float(row.get("replay_env_action_mismatch_fraction", 0.0))
            for row in train_rows
        ),
        default=0.0,
    )
    if action_oob > 0.0 or (require_replay_match and clean_mismatch > 0.0):
        return {
            "status": "FAIL_CODE_INVARIANT",
            "reason": "action bound or clean replay-action invariant failed",
        }
    if material_return_collapse:
        return {
            "status": "LIKELY_HYPERPARAMETER_ISSUE",
            "reason": "updates are finite but normalized return collapsed by >10",
        }
    return {
        "status": "PASS",
        "reason": "finite updates and invariants passed; quick return is diagnostic only",
    }


def _write_summary(run_dir: Path, summary: dict) -> None:
    with (run_dir / "diagnostics_summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, ensure_ascii=False)
    with (run_dir / "diagnostics_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary))
        writer.writeheader()
        writer.writerow(summary)


def main() -> int:
    args = build_parser().parse_args()
    corruption = "clean" if args.corruption_target == "none" else "random"
    stamp = datetime.now().astimezone().strftime("diagnostic_%Y%m%d_%H%M%S")
    config = ExperimentConfig(
        algorithm=args.algorithm,
        env_name=args.env,
        protocol=LOCAL_PROTOCOL,
        corruption=corruption,
        corruption_target=args.corruption_target,
        stage="offline" if args.online_steps == 0 else "both",
        seed=args.seed,
        output_dir=args.output_dir,
        comparison_name=stamp,
        device="cpu",
        offline_steps=args.offline_steps,
        online_steps=args.online_steps,
        offline_corruption_rate=args.corruption_rate,
        online_corruption_rate=args.corruption_rate,
        initial_collection_steps=min(4, args.online_steps),
        warmup_steps=min(4, args.online_steps),
        batch_size=16,
        eval_period=max(1, min(10, max(args.offline_steps, args.online_steps) // 2)),
        eval_episodes=1,
        train_log_period=1,
        checkpoint_period=0,
        max_episode_steps=100 if args.quick else 1_000,
        hidden_dim=32 if args.quick else 256,
        cql_n_actions=2 if args.quick else 10,
        sac_num_critics=2 if args.quick else 10,
        num_critics=3 if args.quick else 5,
    )
    logger = RunLogger(config)
    error = None
    try:
        run_dir = run_experiment(config, logger)
    except BaseException as exc:
        error = exc
        run_dir = logger.run_dir
        traceback.print_exc()
    diagnosis = _classify(run_dir, error, config.corruption == "clean")
    summary = {
        "status": diagnosis["status"],
        "reason": diagnosis["reason"],
        "algorithm": config.algorithm,
        "env": config.env_name,
        "seed": config.seed,
        "offline_steps": config.offline_steps,
        "online_steps": config.online_steps,
        "corruption_rate": args.corruption_rate,
        "corruption_target": config.corruption_target,
        "run_dir": str(run_dir),
    }
    _write_summary(run_dir, summary)
    logger.finish("completed" if error is None else "failed", str(error) if error else None)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if diagnosis["status"] in ("PASS", "LIKELY_HYPERPARAMETER_ISSUE") else 1


if __name__ == "__main__":
    sys.exit(main())
