#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import platform
import shutil
import sys
import tempfile
from importlib import metadata
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from robust_o2o.config import LEGACY_PROTOCOL, ExperimentConfig  # noqa: E402
from robust_o2o.environment import preflight_runtime  # noqa: E402
from robust_o2o.experiment import _torch_load, run_experiment  # noqa: E402
from robust_o2o.fidelity import STRICT_FINAL_TASKS  # noqa: E402
from robust_o2o.logging_utils import RunLogger  # noqa: E402


PINNED = {
    "numpy": "1.23.5",
    "gym": "0.23.1",
    "torch": "2.5.1",
    "h5py": "3.8.0",
    "mujoco-py": "2.1.2.14",
}


def _require_strict_host() -> dict[str, str]:
    if platform.system() != "Linux" or platform.machine() not in (
        "x86_64",
        "AMD64",
    ):
        raise RuntimeError(
            "Strict D4RL-v2 preflight is Linux x86_64 only; local Gymnasium is "
            "not a substitute and will not be reported as success"
        )
    versions = {name: metadata.version(name) for name in PINNED}
    mismatches = {
        name: {"required": PINNED[name], "actual": actual}
        for name, actual in versions.items()
        if actual != PINNED[name]
    }
    if mismatches:
        raise RuntimeError(f"Pinned package mismatch: {mismatches}")
    return versions


def _validate_task_metadata(task: str, result: Mapping[str, Any]) -> None:
    required = {
        "dataset_sha256",
        "environment_fingerprint",
        "dataset_size",
        "observation_dim",
        "action_dim",
        "mujoco_py_version",
        "mujoco_runtime_version_code",
    }
    missing = sorted(key for key in required if not result.get(key))
    if missing:
        raise RuntimeError(f"{task} strict metadata is incomplete: {missing}")
    if result.get("d4rl_env_id") != task or result.get("env_spec_id") != task:
        raise RuntimeError(
            f"{task} was silently remapped: "
            f"dataset={result.get('d4rl_env_id')!r}, "
            f"env={result.get('env_spec_id')!r}"
        )
    if result.get("dataset_backend") != (
        "d4rl.qlearning_dataset(terminate_on_end=False)"
    ):
        raise RuntimeError(
            f"{task} did not use the pinned qlearning_dataset conversion: "
            f"{result.get('dataset_backend')!r}"
        )
    if result.get("mujoco_py_version") != PINNED["mujoco-py"]:
        raise RuntimeError(
            f"{task} used mujoco-py={result.get('mujoco_py_version')!r}; "
            f"required {PINNED['mujoco-py']}"
        )
    if result.get("mujoco_runtime_version_code") != 210:
        raise RuntimeError(
            f"{task} linked MuJoCo runtime code "
            f"{result.get('mujoco_runtime_version_code')!r}; required 210 (2.1.0)"
        )
    dataset_sha = str(result["dataset_sha256"])
    if len(dataset_sha) != 64 or any(
        character not in "0123456789abcdef" for character in dataset_sha.lower()
    ):
        raise RuntimeError(f"{task} dataset SHA-256 is malformed: {dataset_sha}")


def _read_config(run_dir: Path) -> dict[str, Any]:
    return json.loads((run_dir / "config.json").read_text(encoding="utf-8"))


def _final_checkpoint(run_dir: Path) -> tuple[Path, dict[str, Any]]:
    candidates = sorted((run_dir / "checkpoints").rglob("final_*.pt"))
    if not candidates:
        raise RuntimeError(f"checkpoint save smoke produced no final checkpoint: {run_dir}")
    path = candidates[-1]
    payload = _torch_load(path, torch.device("cpu"))
    required = {"agent", "normalizer", "config", "environment_fingerprint"}
    missing = sorted(required - set(payload))
    if missing:
        raise RuntimeError(f"checkpoint reload found an incomplete payload: {missing}")
    return path, payload


def _assert_nested_equal(expected: Any, actual: Any, path: str = "root") -> None:
    if torch.is_tensor(expected):
        if not torch.is_tensor(actual) or not torch.equal(expected, actual):
            raise RuntimeError(f"resume mismatch at {path}")
        return
    if isinstance(expected, np.ndarray):
        if not isinstance(actual, np.ndarray) or not np.array_equal(expected, actual):
            raise RuntimeError(f"resume mismatch at {path}")
        return
    if isinstance(expected, Mapping):
        if not isinstance(actual, Mapping) or set(expected) != set(actual):
            raise RuntimeError(f"resume mapping mismatch at {path}")
        for key in expected:
            _assert_nested_equal(expected[key], actual[key], f"{path}.{key}")
        return
    if isinstance(expected, (list, tuple)):
        if not isinstance(actual, type(expected)) or len(expected) != len(actual):
            raise RuntimeError(f"resume sequence mismatch at {path}")
        for index, (left, right) in enumerate(zip(expected, actual)):
            _assert_nested_equal(left, right, f"{path}[{index}]")
        return
    if isinstance(expected, float) and np.isnan(expected):
        if not isinstance(actual, float) or not np.isnan(actual):
            raise RuntimeError(f"resume NaN mismatch at {path}")
        return
    if expected != actual:
        raise RuntimeError(
            f"resume scalar mismatch at {path}: {expected!r} != {actual!r}"
        )


def _evaluation_signature(run_dir: Path) -> list[tuple[str, ...]]:
    compared = (
        "phase",
        "step",
        "env_steps",
        "updates",
        "return_mean",
        "return_std",
        "normalized_return_mean",
        "normalized_return_std",
        "evaluation_mode",
    )
    with (run_dir / "metrics.csv").open(newline="", encoding="utf-8") as stream:
        return [tuple(row.get(key, "") for key in compared) for row in csv.DictReader(stream)]


def _truncate_to_checkpoint(run_dir: Path, checkpoint: Path) -> None:
    payload = _torch_load(checkpoint, torch.device("cpu"))
    resume_state = payload.get("resume_state")
    if not payload.get("exact_resume_available") or not resume_state:
        raise RuntimeError("offline split checkpoint is not exact-resume capable")
    positions = resume_state.get("writer_append_position", {})
    files = {
        "metrics_csv": run_dir / "metrics.csv",
        "train_metrics_jsonl": run_dir / "train_metrics.jsonl",
    }
    for key, path in files.items():
        if key not in positions:
            raise RuntimeError(f"resume checkpoint omitted writer position {key}")
        with path.open("r+b") as stream:
            stream.truncate(int(positions[key]))
    for candidate in (run_dir / "checkpoints").rglob("*.pt"):
        if candidate.resolve() != checkpoint.resolve():
            candidate.unlink()
    for filename in (
        "summary.json",
        "online_corruption_manifest.json",
        "completed_experiment_manifest.json",
        "diagnostic_evidence.json",
    ):
        candidate = run_dir / filename
        if candidate.exists():
            candidate.unlink()


def _smoke_config(output_root: str, comparison_name: str) -> ExperimentConfig:
    return ExperimentConfig(
        algorithm="riql_naive",
        env_name="hopper-medium-replay-v2",
        corruption="random",
        corruption_target="observations",
        suite_profile="common_budget_diagnostic",
        run_purpose="smoke",
        protocol=LEGACY_PROTOCOL,
        output_dir=output_root,
        comparison_name=comparison_name,
        device="cpu",
        offline_steps=10,
        online_steps=20,
        initial_collection_steps=5,
        batch_size=4,
        eval_period=10,
        eval_episodes=1,
        checkpoint_period=5,
        keep_last_checkpoints=0,
        train_log_period=5,
    )


def _run(logger: RunLogger, config: ExperimentConfig) -> Path:
    run_dir = run_experiment(config, logger)
    logger.finish("completed")
    return run_dir


def _executable_smoke(output_root: str) -> dict[str, Any]:
    uninterrupted_config = _smoke_config(output_root, "strict_preflight_a")
    uninterrupted_dir = _run(RunLogger(uninterrupted_config), uninterrupted_config)
    first_config = _read_config(uninterrupted_dir)
    first_corruption = first_config["offline_corruption"]
    if not first_corruption.get("cache_miss"):
        raise RuntimeError("first corruption materialization was not a cache miss")

    # A second independent run must consume the same immutable cache artifact.
    cache_hit_config = _smoke_config(output_root, "strict_preflight_cache_hit")
    cache_hit_dir = _run(RunLogger(cache_hit_config), cache_hit_config)
    cache_hit_corruption = _read_config(cache_hit_dir)["offline_corruption"]
    if not cache_hit_corruption.get("cache_hit"):
        raise RuntimeError("second corruption materialization was not a cache hit")
    if (
        first_corruption.get("final_artifact_sha256")
        != cache_hit_corruption.get("final_artifact_sha256")
    ):
        raise RuntimeError("cache miss/hit produced different corruption artifacts")

    _, uninterrupted_final = _final_checkpoint(uninterrupted_dir)
    split_candidates = sorted(
        (uninterrupted_dir / "checkpoints" / "offline").glob("step_*.pt")
    )
    if not split_candidates:
        raise RuntimeError("save/resume smoke produced no periodic offline checkpoint")
    split_checkpoint = split_candidates[0]

    resumed_dir = uninterrupted_dir.parent / "resume_fork"
    shutil.copytree(uninterrupted_dir, resumed_dir)
    fork_checkpoint = resumed_dir / split_checkpoint.relative_to(uninterrupted_dir)
    _truncate_to_checkpoint(resumed_dir, fork_checkpoint)
    resumed_config = _smoke_config(output_root, "strict_preflight_a")
    resumed_config.resume_run = str(resumed_dir)
    resumed_dir = _run(RunLogger(resumed_config), resumed_config)
    _, resumed_final = _final_checkpoint(resumed_dir)

    _assert_nested_equal(
        uninterrupted_final["agent"], resumed_final["agent"], "agent"
    )
    _assert_nested_equal(
        uninterrupted_final["normalizer"],
        resumed_final["normalizer"],
        "normalizer",
    )
    if _evaluation_signature(uninterrupted_dir) != _evaluation_signature(resumed_dir):
        raise RuntimeError("checkpoint resume changed evaluation metrics")
    first_online = json.loads(
        (uninterrupted_dir / "online_corruption_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    resumed_online = json.loads(
        (resumed_dir / "online_corruption_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    _assert_nested_equal(first_online, resumed_online, "online_corruption")

    phases = {row[0] for row in _evaluation_signature(uninterrupted_dir)}
    if not {"offline", "online"}.issubset(phases):
        raise RuntimeError(f"smoke did not evaluate both phases: {sorted(phases)}")
    return {
        "coverage_cell": {
            "algorithm": "riql_naive",
            "corruption": "random",
            "target": "observations",
            "implementation_profile": "common_budget_robustness",
        },
        "offline_updates": 10,
        "online_environment_steps": 20,
        "eval_episodes": 1,
        "checkpoint_save_load": True,
        "resume_checkpoint_phase": "offline",
        "offline_checkpoint_deterministic_resume": True,
        "online_checkpoint_resume_exercised": False,
        "full_resume_state_compared": False,
        "compared_after_resume": [
            "agent",
            "normalizer",
            "evaluation_metrics",
            "online_corruption_audit",
        ],
        "cache_miss_then_hit": True,
        "corruption_artifact_sha256": first_corruption[
            "final_artifact_sha256"
        ],
    }


def main() -> int:
    versions = _require_strict_host()
    environments: dict[str, dict[str, Any]] = {}
    for task in STRICT_FINAL_TASKS:
        result = preflight_runtime(task, protocol=LEGACY_PROTOCOL)
        _validate_task_metadata(task, result)
        environments[task] = result

    with tempfile.TemporaryDirectory(prefix="o2o-strict-preflight-") as directory:
        smoke = _executable_smoke(directory)
    report = {
        "status": "passed",
        "platform": platform.platform(),
        "versions": versions,
        "environments": environments,
        "qlearning_dataset_backend_and_metadata_validated": True,
        "qlearning_dataset_numerical_parity_fixture": False,
        "smoke": smoke,
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
