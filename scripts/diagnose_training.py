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
    "PASS_CODE_INVARIANTS",
    "PASS_LEARNING_SIGNAL",
    "FAIL_CODE_INVARIANT",
    "FAIL_NUMERICAL",
    "FAIL_NO_PARAMETER_UPDATE",
    "FAIL_NO_RETURN_IMPROVEMENT",
    "INCONCLUSIVE_SHORT_RUN",
)

PARAMETER_DELTA_TOLERANCE = 1e-8
MIN_LEARNING_UPDATES = 100
MIN_RETURN_IMPROVEMENT = 1.0


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


def classify_diagnostic_summary(
    evidence: dict,
    *,
    invariant_violated: bool = False,
    numerical_failure: bool = False,
    min_learning_updates: int = MIN_LEARNING_UPDATES,
    min_return_improvement: float = MIN_RETURN_IMPROVEMENT,
) -> tuple[str, str]:
    """Classify code health separately from statistically credible learning."""
    numeric_keys = (
        "initial_deterministic_return",
        "final_deterministic_return",
        "return_delta",
        "actor_parameter_delta",
        "critic_parameter_delta",
    )
    numeric_values = [evidence.get(key) for key in numeric_keys]
    if numerical_failure or any(
        value is None or not math.isfinite(float(value)) for value in numeric_values
    ):
        return "FAIL_NUMERICAL", "non-finite diagnostic metric"
    if invariant_violated:
        return "FAIL_CODE_INVARIANT", "action or replay invariant failed"

    actor_updates = int(evidence.get("completed_actor_updates", 0))
    critic_updates = int(evidence.get("completed_critic_updates", 0))
    if actor_updates <= 0 and critic_updates <= 0:
        return "FAIL_NO_PARAMETER_UPDATE", "no optimizer updates completed"
    actor_delta = float(evidence["actor_parameter_delta"])
    critic_delta = float(evidence["critic_parameter_delta"])
    if (
        actor_delta <= PARAMETER_DELTA_TOLERANCE
        and critic_delta <= PARAMETER_DELTA_TOLERANCE
    ):
        return (
            "FAIL_NO_PARAMETER_UPDATE",
            "optimizer updates completed but actor and critic parameters did not change",
        )

    completed_updates = min(actor_updates, critic_updates)
    if completed_updates < min_learning_updates:
        return (
            "INCONCLUSIVE_SHORT_RUN",
            "code invariants passed and parameters changed, but the run is too short "
            "to judge return improvement",
        )
    if float(evidence["return_delta"]) >= min_return_improvement:
        return "PASS_LEARNING_SIGNAL", "deterministic return improved meaningfully"
    return (
        "FAIL_NO_RETURN_IMPROVEMENT",
        "parameters changed but deterministic return did not improve meaningfully",
    )


def _empty_evidence() -> dict:
    return {
        "initial_deterministic_return": None,
        "final_deterministic_return": None,
        "return_delta": None,
        "actor_parameter_delta": None,
        "critic_parameter_delta": None,
        "completed_actor_updates": 0,
        "completed_critic_updates": 0,
    }


def _classify(
    run_dir: Path, error: BaseException | None, require_replay_match: bool
) -> dict:
    if error is not None:
        message = f"{type(error).__name__}: {error}"
        return {
            **_empty_evidence(),
            "classification": "FAIL_CODE_INVARIANT",
            "classification_reason": message,
        }

    train_rows = _read_jsonl(run_dir / "train_metrics.jsonl")
    numeric_values = [
        float(value)
        for row in train_rows
        for key, value in row.items()
        if key not in ("timestamp", "phase") and isinstance(value, (int, float))
    ]
    numerical_failure = any(not math.isfinite(value) for value in numeric_values)
    numerical_failure = numerical_failure or any(
        float(row.get("NaN_or_inf_count", 0.0)) > 0.0 for row in train_rows
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

    evidence_path = run_dir / "diagnostic_evidence.json"
    if not evidence_path.exists():
        return {
            **_empty_evidence(),
            "classification": "FAIL_CODE_INVARIANT",
            "classification_reason": "diagnostic parameter evidence was not written",
        }
    with evidence_path.open(encoding="utf-8") as stream:
        evidence = json.load(stream)
    classification, reason = classify_diagnostic_summary(
        evidence,
        invariant_violated=action_oob > 0.0
        or (require_replay_match and clean_mismatch > 0.0),
        numerical_failure=numerical_failure,
    )
    return {
        **evidence,
        "classification": classification,
        "classification_reason": reason,
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
        diagnostic_mode=True,
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
        **diagnosis,
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
    healthy = diagnosis["classification"] in (
        "PASS_CODE_INVARIANTS",
        "PASS_LEARNING_SIGNAL",
        "INCONCLUSIVE_SHORT_RUN",
    )
    return 0 if healthy else 1


if __name__ == "__main__":
    sys.exit(main())
