#!/usr/bin/env python3
from __future__ import annotations

import json
import platform
import tempfile
from importlib import metadata
import torch

from robust_o2o.config import LEGACY_PROTOCOL, ExperimentConfig
from robust_o2o.environment import preflight_runtime
from robust_o2o.experiment import _torch_load, run_experiment
from robust_o2o.logging_utils import RunLogger


PINNED = {
    "numpy": "1.23.5",
    "gym": "0.23.1",
    "torch": "2.5.1",
    "h5py": "3.8.0",
}


def main() -> int:
    if platform.system() != "Linux" or platform.machine() not in ("x86_64", "AMD64"):
        raise RuntimeError(
            "Strict D4RL-v2 preflight is Linux x86_64 only; local Gymnasium is "
            "not a substitute and will not be reported as success"
        )
    versions = {name: metadata.version(name) for name in PINNED}
    mismatches = {
        name: (PINNED[name], actual)
        for name, actual in versions.items()
        if actual != PINNED[name]
    }
    if mismatches:
        raise RuntimeError(f"Pinned package mismatch: {mismatches}")

    environment = preflight_runtime(
        "hopper-medium-replay-v2", protocol=LEGACY_PROTOCOL
    )
    with tempfile.TemporaryDirectory(prefix="o2o-strict-preflight-") as directory:
        config = ExperimentConfig(
            algorithm="riql_naive",
            env_name="hopper-medium-replay-v2",
            corruption="clean",
            suite_profile="common_budget_robustness",
            protocol=LEGACY_PROTOCOL,
            output_dir=directory,
            offline_steps=10,
            online_steps=20,
            initial_collection_steps=5,
            batch_size=4,
            eval_period=100,
            eval_episodes=2,
            checkpoint_period=0,
            train_log_period=10,
        )
        logger = RunLogger(config)
        run_dir = run_experiment(config, logger)
        logger.finish("completed")
        checkpoints = sorted((run_dir / "checkpoints").rglob("*.pt"))
        if not checkpoints:
            raise RuntimeError("checkpoint save smoke produced no checkpoint")
        payload = _torch_load(checkpoints[-1], torch.device("cpu"))
        if "agent" not in payload or "normalizer" not in payload:
            raise RuntimeError("checkpoint load smoke found an incomplete payload")
        report = {
            "status": "passed",
            "platform": platform.platform(),
            "versions": versions,
            "environment": environment,
            "smoke": {
                "offline_updates": 10,
                "online_environment_steps": 20,
                "eval_episodes": 2,
                "checkpoint_save_load": True,
            },
        }
        print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
