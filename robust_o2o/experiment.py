from __future__ import annotations

import json
import hashlib
import random
import warnings
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

from .agents import build_agent
from .config import LEGACY_PROTOCOL, ExperimentConfig
from .calql_online import CalQLTrajectoryAccumulator, dynamic_batch_counts
from .corruption import (
    AttackOracle,
    OnlineCorruptionAudit,
    corrupt_pre_action_value,
    corrupt_offline_dataset,
    corrupt_online_transition,
    make_numpy_corruption_rng,
    make_attack_oracle,
    numpy_rng_state,
    restore_numpy_rng_state,
    sample_online_corruption_target,
)
from .device import resolve_device, seed_env_only, seed_everything
from .environment import (
    StateNormalizer,
    EXPECTED_LOCOMOTION_DIMS,
    apply_normalizer,
    environment_metadata,
    evaluate_agent,
    expected_env_spec_id,
    load_d4rl_dataset,
    make_env,
    preserve_training_rng_state,
    reset_env,
    step_env,
)
from .logging_utils import RunLogger, resolve_resume_run_directory
from .manifest import verify_experiment_manifest
from .paths import results_root_from_output
from .replay import (
    NUMPY_REPLAY_SAMPLING,
    RPEX_OFFICIAL_REPLAY_SAMPLING,
    OfflineDataset,
    ReplayBuffer,
    concatenate_batches,
    mixed_batch,
    sample_pqe_update_batches,
    update_sample_priorities,
)


class MetricAccumulator:
    def __init__(self) -> None:
        self.values: Dict[str, list[float]] = {}

    def add(self, metrics: Dict[str, float]) -> None:
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                self.values.setdefault(key, []).append(float(value))

    def means(self, reset: bool = True) -> Dict[str, float]:
        result = {
            key: float(np.mean(values)) for key, values in self.values.items()
        }
        result["NaN_or_inf_count"] = float(
            sum(not np.isfinite(value) for values in self.values.values() for value in values)
        )
        if reset:
            self.values.clear()
        return result


_WSRL_UPDATE_COUNT_KEYS = (
    "number_of_actor_updates",
    "number_of_critic_updates",
    "number_of_temperature_updates",
)
_WSRL_TOTAL_UPDATE_KEYS = (
    "total_actor_updates",
    "total_critic_updates",
    "total_temperature_updates",
)


def _agent_update_count(agent: object, role: str) -> int:
    """Read split optimizer counters with legacy/mock-agent compatibility."""

    return int(
        getattr(
            agent,
            f"{role}_updates",
            getattr(agent, "total_updates", 0),
        )
    )


def _runtime_update_metadata(
    config: ExperimentConfig,
    agent: object,
    *,
    online_initial_update_counts: Dict[str, int] | None = None,
    online_environment_steps: int = 0,
) -> Dict[str, Any]:
    """Return measured optimizer counts without conflating compute and data budgets."""

    totals = {
        role: _agent_update_count(agent, role)
        for role in ("actor", "critic", "temperature")
    }
    initial = online_initial_update_counts or totals
    online = {
        role: totals[role] - int(initial[role])
        for role in totals
    }
    configured_utd = int(
        config.wsrl_utd_ratio
        if config.algorithm == "wsrl"
        else config.updates_per_step
    )
    actual_utd = (
        float(online["critic"] / online_environment_steps)
        if online_environment_steps > 0
        else 0.0
    )
    return {
        "critic_gradient_updates": totals["critic"],
        "actor_gradient_updates": totals["actor"],
        "temperature_updates": totals["temperature"],
        "online_critic_gradient_updates": online["critic"],
        "online_actor_gradient_updates": online["actor"],
        "online_temperature_updates": online["temperature"],
        "configured_utd": configured_utd,
        "actual_utd": actual_utd,
    }


def _split_wsrl_high_utd_batch(
    batch: Dict[str, torch.Tensor], utd_ratio: int
) -> list[Dict[str, torch.Tensor]]:
    """Match upstream ``update_high_utd``'s reshape into contiguous minibatches."""

    if utd_ratio <= 0:
        raise ValueError("WSRL UTD ratio must be positive")
    if not batch:
        raise ValueError("WSRL high-UTD update requires a non-empty batch")
    batch_sizes = {int(value.shape[0]) for value in batch.values()}
    if len(batch_sizes) != 1:
        raise ValueError("WSRL batch fields must have one shared leading dimension")
    total_batch_size = batch_sizes.pop()
    if total_batch_size % utd_ratio != 0:
        raise ValueError(
            "WSRL total batch size must be divisible by the critic UTD ratio"
        )
    per_critic_batch_size = total_batch_size // utd_ratio
    return [
        {
            key: value[
                index * per_critic_batch_size : (index + 1)
                * per_critic_batch_size
            ]
            for key, value in batch.items()
        }
        for index in range(utd_ratio)
    ]


def _aggregate_wsrl_high_utd_metrics(
    critic_metrics: list[Dict[str, float]],
    actor_temperature_metrics: Dict[str, float],
) -> Dict[str, float]:
    """Mirror upstream: mean critic info, then one actor/temperature info."""

    if not critic_metrics:
        raise ValueError("WSRL high-UTD update requires critic metrics")
    critic_keys = set.intersection(*(set(metrics) for metrics in critic_metrics))
    result = {
        key: float(np.mean([metrics[key] for metrics in critic_metrics]))
        for key in critic_keys
        if isinstance(critic_metrics[0][key], (int, float))
    }
    result.update(actor_temperature_metrics)
    for key in _WSRL_UPDATE_COUNT_KEYS:
        result[key] = float(
            sum(float(metrics.get(key, 0.0)) for metrics in critic_metrics)
            + float(actor_temperature_metrics.get(key, 0.0))
        )
    for key in _WSRL_TOTAL_UPDATE_KEYS:
        if key in actor_temperature_metrics:
            result[key] = float(actor_temperature_metrics[key])
    return result


def _wsrl_first_update_env_step(
    *, warmup_steps: int, total_batch_size: int
) -> int:
    """Return the first local env-step that updates under pinned WSRL timing.

    Pinned ``finetune.py`` performs the transition while the zero-based online
    offset is less than or equal to ``max(warmup_steps, min_steps_to_update)``.
    With this runner's one-based env-step counter that makes the first update
    occur at threshold + 2.
    """

    return max(int(warmup_steps), int(total_batch_size)) + 2


def _run_wsrl_high_utd_update(
    agent: object,
    offline: OfflineDataset,
    replay: ReplayBuffer,
    config: ExperimentConfig,
    device: torch.device,
) -> Dict[str, float]:
    """Execute one pinned WSRL online REDQ update cycle.

    Upstream samples one ``batch_size=1024`` replay batch, reshapes it into four
    contiguous 256-sample critic minibatches, then updates actor/temperature
    once on the original 1024-sample batch. Sampling four independent batches
    changes both replay membership and RNG consumption.
    """

    offline_ratio = float(config.effective_offline_ratio)
    if not np.isclose(offline_ratio, 0.0):
        raise ValueError(
            "WSRL online fine-tuning retains no offline data; "
            f"effective_offline_ratio must be 0.0, got {offline_ratio}"
        )
    total_batch_size = (
        config.wsrl_utd_ratio * config.wsrl_per_critic_batch_size
    )
    full_batch = mixed_batch(
        offline,
        replay,
        total_batch_size,
        offline_ratio,
        device,
        online_replace=True,
    )
    critic_metrics = [
        agent.update(
            minibatch,
            update_actor_temperature=False,
            update_critic=True,
        )
        for minibatch in _split_wsrl_high_utd_batch(
            full_batch, config.wsrl_utd_ratio
        )
    ]
    actor_temperature_metrics = agent.update(
        full_batch,
        update_actor_temperature=True,
        update_critic=False,
    )
    return _aggregate_wsrl_high_utd_metrics(
        critic_metrics, actor_temperature_metrics
    )


def _wsrl_runtime_metadata(
    config: ExperimentConfig,
    agent: object,
    *,
    online_initial_update_counts: Dict[str, int] | None = None,
) -> Dict[str, Any]:
    """Describe the executed WSRL protocol and measured optimizer counts."""

    if config.algorithm != "wsrl":
        return {}
    total_batch_size = (
        config.wsrl_utd_ratio * config.wsrl_per_critic_batch_size
    )
    first_update_env_step = _wsrl_first_update_env_step(
        warmup_steps=max(
            config.initial_collection_steps, config.warmup_steps
        ),
        total_batch_size=total_batch_size,
    )
    current_counts = {
        "actor": _agent_update_count(agent, "actor"),
        "critic": _agent_update_count(agent, "critic"),
        "temperature": _agent_update_count(agent, "temperature"),
    }
    metadata: Dict[str, Any] = {
        "wsrl_online_replay_profile_executed": "online_only",
        "wsrl_online_offline_ratio_executed": 0.0,
        "wsrl_online_cql_enabled": False,
        "wsrl_configured_warmup_steps": int(config.warmup_steps),
        "wsrl_first_update_env_step": first_update_env_step,
        "wsrl_total_sampled_batch_size": total_batch_size,
        "wsrl_per_critic_batch_size": int(
            config.wsrl_per_critic_batch_size
        ),
        "wsrl_critic_updates_per_update_step": int(config.wsrl_utd_ratio),
        "wsrl_actor_updates_per_update_step": 1,
        "wsrl_temperature_updates_per_update_step": 1,
        "wsrl_batch_reuse_profile": (
            "one_online_batch_contiguously_split_for_critic_then_reused_for_actor_temperature"
        ),
        "wsrl_total_actor_updates": current_counts["actor"],
        "wsrl_total_critic_updates": current_counts["critic"],
        "wsrl_total_temperature_updates": current_counts["temperature"],
    }
    if online_initial_update_counts is not None:
        metadata.update(
            {
                "wsrl_online_actor_updates": current_counts["actor"]
                - int(online_initial_update_counts["actor"]),
                "wsrl_online_critic_updates": current_counts["critic"]
                - int(online_initial_update_counts["critic"]),
                "wsrl_online_temperature_updates": current_counts[
                    "temperature"
                ]
                - int(online_initial_update_counts["temperature"]),
            }
        )
    return metadata


def _restore_pqe_block_schedule(
    config: ExperimentConfig,
    resume_state: Optional[Dict[str, Any]],
    start_env_step: int,
) -> tuple[list[int], bool]:
    """Restore the completed PQE update blocks without replaying a block.

    New checkpoints persist every block's optimizer-update count.  A legacy
    checkpoint is migratable only before any update or exactly after the first
    1,000-step/5,000-update block.  Later legacy checkpoints contain
    step-interleaved updates that cannot be reconstructed as official blocks,
    so they fail closed instead of receiving an invented ledger.
    """

    if config.algorithm != "pessimistic_q_ensemble" or resume_state is None:
        return [], False

    raw_counts = resume_state.get("pqe_block_update_counts")
    if raw_counts is not None:
        if not isinstance(raw_counts, (list, tuple)):
            raise TypeError("PQE block update counts must be a sequence")
        counts = [int(value) for value in raw_counts]
        if any(value < 0 for value in counts):
            raise ValueError("PQE block update counts cannot be negative")
        saved_count = int(
            resume_state.get("pqe_completed_block_count", len(counts))
        )
        saved_next = int(resume_state.get("pqe_next_block_index", len(counts)))
        if saved_count != len(counts) or saved_next != len(counts):
            raise ValueError(
                "PQE block schedule checkpoint is internally inconsistent"
            )
        block_size = int(config.pqe_first_online_block_steps)
        expected_completed_blocks = start_env_step // block_size
        if len(counts) != expected_completed_blocks:
            raise ValueError(
                "PQE checkpoint block ledger does not match the number of "
                "fully collected blocks"
            )
        normal_count = int(block_size * config.updates_per_step)
        expected_counts = (
            [normal_count * config.pqe_first_epoch_multiplier]
            + [normal_count] * (expected_completed_blocks - 1)
            if expected_completed_blocks
            else []
        )
        if counts != expected_counts:
            raise ValueError(
                "PQE checkpoint block update counts do not match the configured "
                "epoch schedule"
            )
        saved_index = resume_state.get("pqe_block_index")
        saved_steps = resume_state.get("pqe_steps_in_current_block")
        if saved_index is not None and int(saved_index) != start_env_step // block_size:
            raise ValueError("PQE checkpoint block index does not match phase_step")
        if saved_steps is not None and int(saved_steps) != start_env_step % block_size:
            raise ValueError(
                "PQE checkpoint in-block step count does not match phase_step"
            )
        return counts, bool(
            resume_state.get("pqe_block_schedule_inferred_from_legacy", False)
        )

    block_size = int(config.pqe_first_online_block_steps)
    normal_count = int(
        block_size * config.updates_per_step
    )
    expected_first_count = int(
        normal_count * config.pqe_first_epoch_multiplier
    )
    first_applied = bool(
        resume_state.get("pqe_first_block_updates_applied", False)
    )
    first_count = int(resume_state.get("pqe_first_block_update_count", 0))
    explicit_count = int(resume_state.get("pqe_completed_block_count", 0))

    # The retired controller applied one update after every environment step
    # beyond step 1000. Once it crossed that boundary, a checkpoint no longer
    # contains enough information to reconstruct an official block ledger.
    # Inventing [5000, 1000, ...] would silently discard unknown interleaved
    # updates, so fail closed instead of claiming a source-aligned resume.
    if start_env_step > block_size:
        raise ValueError(
            "legacy PQE checkpoint after the first 1000-step block cannot be "
            "safely resumed: old step-interleaved updates have no exact block ledger"
        )
    if start_env_step < block_size:
        if first_applied or first_count != 0 or explicit_count != 0:
            raise ValueError(
                "legacy PQE checkpoint claims updates before its first full block"
            )
        return [], True

    if (
        not first_applied
        or first_count != expected_first_count
        or explicit_count not in (0, 1)
    ):
        raise ValueError(
            "legacy PQE checkpoint at step 1000 must record exactly 5000 "
            "first-block updates"
        )
    return [expected_first_count], True


def _pqe_block_schedule_metadata(
    config: ExperimentConfig,
    block_update_counts: list[int],
    *,
    env_step: int,
    inferred_from_legacy: bool,
) -> Dict[str, Any]:
    """Serialize both the complete block ledger and legacy first-block keys."""

    completed = len(block_update_counts)
    first_count = int(block_update_counts[0]) if completed else 0
    block_size = int(config.pqe_first_online_block_steps)
    return {
        "pqe_steps_in_current_block": int(env_step % block_size),
        "pqe_block_index": int(env_step // block_size),
        "pqe_completed_block_count": completed,
        "pqe_last_completed_block_index": completed - 1,
        "pqe_next_block_index": completed,
        "pqe_block_update_counts": [int(value) for value in block_update_counts],
        "pqe_block_schedule_inferred_from_legacy": bool(inferred_from_legacy),
        # Retained so existing readers and checkpoints continue to work.
        "pqe_first_block_updates_applied": bool(completed),
        "pqe_first_block_update_count": first_count,
    }


def bounded_executed_action(
    raw_policy_action: np.ndarray,
    action_low: np.ndarray,
    action_high: np.ndarray,
) -> np.ndarray:
    return np.clip(raw_policy_action, action_low, action_high).astype(np.float32)


def _replay_transition_coordinates(
    stored_state: np.ndarray,
    stored_next_state: np.ndarray,
    normalizer: StateNormalizer,
    already_normalized: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Return one replay transition in exactly one normalized coordinate pass."""

    if already_normalized:
        return stored_state, stored_next_state
    return (
        normalizer.transform(stored_state),
        normalizer.transform(stored_next_state),
    )


def _poison_replay_in_learner_coordinates(config: ExperimentConfig) -> bool:
    """Use the normalized replay coordinates expected by pinned RPEX attacks."""

    return (
        config.implementation_profile
        in ("official_code_reference", "research_benchmark")
        and config.attack_timing
        == "official_code_post_transition_replay_poisoning"
    )


def normalizer_sha256(normalizer: StateNormalizer) -> str:
    digest = hashlib.sha256(normalizer.mode.encode("utf-8"))
    for value in (normalizer.mean, normalizer.std):
        array = np.ascontiguousarray(value)
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def _make_evaluation_env(
    env_name: str, protocol: str, seed: int
) -> object:
    """Construct and seed evaluation state without advancing training RNGs."""
    with preserve_training_rng_state():
        env = make_env(env_name, protocol)
        seed_env_only(env, seed)
    return env


def _parameter_snapshot(agent: object) -> Dict[str, list[torch.Tensor]]:
    groups: Dict[str, list[torch.Tensor]] = {"actor": [], "critic": []}
    actor = getattr(agent, "actor", None)
    if actor is not None:
        groups["actor"] = [parameter.detach().clone() for parameter in actor.parameters()]
    else:
        actors = getattr(agent, "actors", None)
        if actors is not None:
            groups["actor"] = [
                parameter.detach().clone()
                for module in actors
                for parameter in module.parameters()
            ]
    critic_modules = []
    for name in ("critic", "critic2", "q1", "q2", "value"):
        module = getattr(agent, name, None)
        if module is not None:
            critic_modules.append(module)
    for name in (
        "q1_members",
        "q2_members",
    ):
        modules = getattr(agent, name, None)
        if modules is not None:
            critic_modules.extend(list(modules))
    groups["critic"] = [
        parameter.detach().clone()
        for module in critic_modules
        for parameter in module.parameters()
    ]
    return groups


def _parameter_deltas(
    agent: object, previous: Dict[str, list[torch.Tensor]]
) -> tuple[Dict[str, float], Dict[str, list[torch.Tensor]]]:
    current = _parameter_snapshot(agent)
    deltas = {}
    for group in ("actor", "critic"):
        squared = sum(
            float((now - before).square().sum().item())
            for now, before in zip(current[group], previous[group])
        )
        deltas[f"{group}_parameter_delta"] = float(np.sqrt(squared))
    return deltas, current


def _policy_log_std_metrics(agent: object) -> Dict[str, float]:
    actor = getattr(agent, "actor", None)
    log_std = getattr(actor, "log_std", None)
    if log_std is None or not torch.is_tensor(log_std):
        return {}
    values = log_std.detach()
    return {
        "policy_log_std_mean": float(values.mean().item()),
        "policy_log_std_min": float(values.min().item()),
        "policy_log_std_max": float(values.max().item()),
    }


def _torch_load(path: Path, device: torch.device) -> Dict[str, Any]:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def resolve_resume_checkpoint(path_value: str, device: torch.device) -> tuple[Path, Dict[str, Any]]:
    path = Path(path_value).expanduser().resolve()
    run_dir = resolve_resume_run_directory(path_value)
    manifest_path = run_dir / "experiment_manifest.json"
    if not manifest_path.exists():
        raise ValueError("exact resume source has no canonical launch manifest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    launch_manifest_sha256 = verify_experiment_manifest(manifest)
    candidates = (
        [path]
        if path.is_file()
        else list(run_dir.glob("checkpoints/*/*.pt"))
    )
    exact = []
    for candidate in candidates:
        payload = _torch_load(candidate, device)
        if payload.get("exact_resume_available"):
            exact.append((int(payload.get("env_steps", 0)), int(payload.get("step", 0)), candidate, payload))
    if not exact:
        raise ValueError(
            "--resume-run found no exact episode-boundary resume checkpoint; "
            "initialization checkpoints must use --initialize-from-checkpoint"
        )
    _, _, candidate, payload = max(exact, key=lambda item: (item[0], item[1]))
    checkpoint_manifest_sha256 = payload.get("manifest_sha256")
    if checkpoint_manifest_sha256 is None:
        raise ValueError(
            "exact resume checkpoint has no launch manifest SHA256; legacy "
            "checkpoints may only initialize a new run"
        )
    if checkpoint_manifest_sha256 != launch_manifest_sha256:
        raise ValueError(
            "exact resume checkpoint belongs to a different run manifest: "
            f"checkpoint={checkpoint_manifest_sha256}, "
            f"target={launch_manifest_sha256}"
        )
    return candidate, payload


def capture_global_rng_state(
    corruption_rng: Optional[
        np.random.RandomState | np.random.Generator
    ] = None,
    oracle: Optional[AttackOracle] = None,
) -> Dict[str, Any]:
    state: Dict[str, Any] = {
        "python": random.getstate(),
        "numpy_global": np.random.get_state(),
        "torch_cpu": torch.random.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "corruption_rng": (
            numpy_rng_state(corruption_rng)
            if corruption_rng is not None
            else None
        ),
        "attack_rng": (
            oracle.generator.get_state() if oracle is not None else None
        ),
    }
    if hasattr(torch, "mps") and hasattr(torch.mps, "get_rng_state"):
        try:
            state["torch_mps"] = torch.mps.get_rng_state()
        except RuntimeError:
            state["torch_mps"] = None
    return state


def restore_global_rng_state(
    state: Dict[str, Any],
    corruption_rng: Optional[
        np.random.RandomState | np.random.Generator
    ] = None,
    oracle: Optional[AttackOracle] = None,
) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy_global"])
    torch.random.set_rng_state(state["torch_cpu"])
    if state.get("torch_cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])
    if state.get("torch_mps") is not None and hasattr(torch, "mps"):
        torch.mps.set_rng_state(state["torch_mps"])
    if corruption_rng is not None and state.get("corruption_rng") is not None:
        restore_numpy_rng_state(corruption_rng, state["corruption_rng"])
    if oracle is not None and state.get("attack_rng") is not None:
        oracle.generator.set_state(state["attack_rng"])


def capture_environment_rng_state(env: object) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for label, owner in (
        ("environment", getattr(env, "unwrapped", env)),
        ("action_space", getattr(env, "action_space", None)),
    ):
        rng = getattr(owner, "np_random", None)
        if rng is None:
            continue
        if hasattr(rng, "bit_generator"):
            result[label] = ("generator", rng.bit_generator.state)
        elif hasattr(rng, "get_state"):
            result[label] = ("random_state", rng.get_state())
    return result


def restore_environment_rng_state(env: object, state: Dict[str, Any]) -> None:
    for label, owner in (
        ("environment", getattr(env, "unwrapped", env)),
        ("action_space", getattr(env, "action_space", None)),
    ):
        if label not in state:
            continue
        rng = getattr(owner, "np_random", None)
        kind, value = state[label]
        if kind == "generator":
            rng.bit_generator.state = value
        else:
            rng.set_state(value)


def save_checkpoint(
    path: Path,
    agent: object,
    config: ExperimentConfig,
    normalizer: StateNormalizer,
    phase: str,
    step: int,
    env_steps: int,
    state_dim: int,
    action_dim: int,
    resume_state: Optional[Dict[str, Any]] = None,
) -> None:
    payload = {
        "format_version": 4,
        "algorithm": config.algorithm,
        "env_name": config.env_name,
        "protocol": config.protocol,
        "algorithm_profile": config.algorithm_profile,
        "resolved_algorithm_profile": config.resolved_algorithm_profile,
        "implementation_profile": config.implementation_profile,
        "implementation_fidelity": config.implementation_fidelity,
        "suite_profile": config.suite_profile,
        "manifest_sha256": getattr(config, "_manifest_sha256", None),
        "config": config.to_dict(),
        "normalizer": normalizer.state_dict(),
        "agent": agent.checkpoint_state(),
        "phase": phase,
        "step": step,
        "env_steps": env_steps,
        "state_dim": state_dim,
        "action_dim": action_dim,
        "environment_fingerprint": getattr(
            config, "_environment_fingerprint", None
        ),
        "environment_fingerprint_payload": getattr(
            config, "_environment_fingerprint_payload", None
        ),
        "resume_state": resume_state,
        "exact_resume_available": bool(
            resume_state is not None and resume_state.get("episode_boundary", False)
        ),
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _checkpoint_directory(logger: RunLogger, phase: str) -> Path:
    directory = logger.run_dir / "checkpoints" / phase
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _writer_positions(logger: RunLogger) -> Dict[str, int]:
    return {
        "metrics_csv": (
            logger.metrics_path.stat().st_size if logger.metrics_path.exists() else 0
        ),
        "train_metrics_jsonl": (
            logger.train_metrics_path.stat().st_size
            if logger.train_metrics_path.exists()
            else 0
        ),
    }


def _validate_writer_positions(
    logger: RunLogger, resume_state: Optional[Dict[str, Any]]
) -> None:
    if resume_state is None:
        return
    expected = resume_state.get("writer_append_position")
    if expected is None:
        raise ValueError("exact resume checkpoint has no writer append position")
    if isinstance(expected, int):
        expected = {"train_metrics_jsonl": expected}
    actual = _writer_positions(logger)
    mismatches = {
        key: (value, actual.get(key))
        for key, value in expected.items()
        if actual.get(key) != value
    }
    if mismatches:
        raise ValueError(
            "resume log position mismatch; refusing to duplicate or overwrite "
            f"metrics: {mismatches}"
        )


def _validate_resume_precommit(
    config: ExperimentConfig,
    logger: RunLogger,
    resume_state: Optional[Dict[str, Any]],
) -> None:
    """Reject unsafe exact-resume state before changing original run files."""

    if resume_state is None:
        return
    phase = resume_state.get("phase")
    if phase not in ("offline", "online"):
        raise ValueError(f"exact resume checkpoint has invalid phase {phase!r}")
    if config.stage == "offline" and phase != "offline":
        raise ValueError("offline-only run cannot resume an online-phase checkpoint")
    if phase == "online" and not resume_state.get("episode_boundary"):
        raise ValueError("exact online resume is only supported at episode boundaries")
    _validate_writer_positions(logger, resume_state)


def _prune_periodic_checkpoints(directory: Path, keep_last: int) -> None:
    """Retain only the newest periodic checkpoints; zero means keep all."""
    if keep_last == 0:
        return
    checkpoints = sorted(directory.glob("step_*.pt"))
    for checkpoint in checkpoints[:-keep_last]:
        checkpoint.unlink()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _save_pqe_member_checkpoints(
    directory: Path,
    agent: object,
    config: ExperimentConfig,
    normalizer: StateNormalizer,
    state_dim: int,
    action_dim: int,
    manifest_tag: str,
) -> list[Path]:
    """Write five independently loadable CQL members before online PQE."""

    states = agent.member_checkpoint_states()
    if len(states) != 5:
        raise RuntimeError("PQE offline checkpoint must contain five members")
    member_directory = directory / "members"
    member_directory.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    hashes: list[str] = []
    for index, member_state in enumerate(states):
        path = member_directory / (
            f"member_{index}_seed_{member_state['member_seed']}_manifest_"
            f"{manifest_tag}.pt"
        )
        payload = {
            "format_version": 1,
            "format": "pqe_independent_member_checkpoint_v1",
            "algorithm": "pessimistic_q_ensemble",
            "env_name": config.env_name,
            "protocol": config.protocol,
            "state_dim": state_dim,
            "action_dim": action_dim,
            "environment_fingerprint": getattr(
                config, "_environment_fingerprint", None
            ),
            "normalizer": normalizer.state_dict(),
            "normalizer_sha256": normalizer_sha256(normalizer),
            "member": member_state,
            "offline_artifact_identity": getattr(
                agent, "offline_artifact_identity", None
            ),
        }
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(payload, temporary)
        temporary.replace(path)
        digest = _file_sha256(path)
        agent.record_member_checkpoint_hash(index, digest)
        paths.append(path)
        hashes.append(digest)
    if len(set(hashes)) != 5:
        raise RuntimeError("PQE member checkpoints are not content-distinct")
    return paths


def _load_pqe_member_checkpoints(
    paths: tuple[str, ...],
    agent: object,
    config: ExperimentConfig,
    normalizer: StateNormalizer,
    state_dim: int,
    action_dim: int,
    device: torch.device,
) -> None:
    if len(paths) != 5 or len(set(paths)) != 5:
        raise ValueError("PQE online stage requires five unique member paths")
    payloads: list[Dict[str, Any]] = []
    hashes: list[str] = []
    for index, value in enumerate(paths):
        path = Path(value).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"PQE member checkpoint does not exist: {path}")
        payload = _torch_load(path, device)
        expected = {
            "format": "pqe_independent_member_checkpoint_v1",
            "algorithm": "pessimistic_q_ensemble",
            "env_name": config.env_name,
            "protocol": config.protocol,
            "state_dim": state_dim,
            "action_dim": action_dim,
            "environment_fingerprint": getattr(
                config, "_environment_fingerprint", None
            ),
            "normalizer_sha256": normalizer_sha256(normalizer),
        }
        mismatches = {
            key: {"actual": payload.get(key), "expected": expected_value}
            for key, expected_value in expected.items()
            if payload.get(key) != expected_value
        }
        if mismatches:
            raise ValueError(
                f"PQE member checkpoint {index} provenance mismatch: {mismatches}"
            )
        if "member" not in payload:
            raise ValueError(f"PQE member checkpoint {index} has no member state")
        payloads.append(payload)
        hashes.append(_file_sha256(path))
    if len(set(hashes)) != 5:
        raise ValueError("duplicate PQE member checkpoint file content is forbidden")
    artifact_identities = {
        tuple(payload["offline_artifact_identity"])
        for payload in payloads
        if payload.get("offline_artifact_identity") is not None
    }
    if len(artifact_identities) != 1:
        raise ValueError("PQE members are not bound to one offline artifact")
    loaded_artifact_identity = next(iter(artifact_identities))
    current_artifact_identity = getattr(agent, "offline_artifact_identity", None)
    if (
        current_artifact_identity is not None
        and tuple(current_artifact_identity) != loaded_artifact_identity
    ):
        raise ValueError(
            "PQE member checkpoints were trained on a different corrupted artifact"
        )
    agent.load_member_checkpoint_states(
        [payload["member"] for payload in payloads], hashes
    )
    agent.offline_artifact_identity = loaded_artifact_identity


def _save_phase_checkpoint(
    logger: RunLogger,
    agent: object,
    config: ExperimentConfig,
    normalizer: StateNormalizer,
    phase: str,
    step: int,
    env_steps: int,
    state_dim: int,
    action_dim: int,
    *,
    final: bool,
    resume_state: Optional[Dict[str, Any]] = None,
) -> Path:
    directory = _checkpoint_directory(logger, phase)
    manifest_tag = str(getattr(config, "_manifest_sha256", "unresolved"))[:16]
    path = directory / (
        f"final_manifest_{manifest_tag}.pt"
        if final
        else f"step_{step:09d}_manifest_{manifest_tag}.pt"
    )
    if (
        config.algorithm == "pessimistic_q_ensemble"
        and phase == "offline"
        and final
    ):
        member_paths = _save_pqe_member_checkpoints(
            directory,
            agent,
            config,
            normalizer,
            state_dim,
            action_dim,
            manifest_tag,
        )
        logger.logger.info(
            "PQE member checkpoints saved: %s",
            ", ".join(str(member_path) for member_path in member_paths),
        )
    save_checkpoint(
        path,
        agent,
        config,
        normalizer,
        phase,
        step,
        env_steps,
        state_dim,
        action_dim,
        resume_state,
    )
    if not final:
        _prune_periodic_checkpoints(directory, config.keep_last_checkpoints)
    logger.logger.info("checkpoint saved: %s", path)
    return path


def _validate_checkpoint(
    payload: Dict[str, Any],
    config: ExperimentConfig,
    state_dim: int,
    action_dim: int,
) -> None:
    if payload.get("algorithm") != config.algorithm:
        raise ValueError(
            f"Checkpoint algorithm={payload.get('algorithm')!r}, "
            f"requested={config.algorithm!r}"
        )
    if payload.get("env_name") != config.env_name:
        raise ValueError(
            f"Checkpoint env={payload.get('env_name')!r}, requested={config.env_name!r}"
        )
    if payload.get("protocol") != config.protocol:
        raise ValueError(
            f"Checkpoint protocol={payload.get('protocol')!r}, "
            f"requested={config.protocol!r}; checkpoints from different "
            "environment backends cannot be mixed"
        )
    saved_profile = payload.get(
        "implementation_profile", payload.get("algorithm_profile")
    )
    if saved_profile is not None and saved_profile != config.implementation_profile:
        raise ValueError(
            f"Checkpoint implementation_profile={saved_profile!r}, "
            f"requested={config.implementation_profile!r}"
        )
    if payload.get("state_dim") != state_dim or payload.get("action_dim") != action_dim:
        raise ValueError("Checkpoint observation/action dimensions do not match the env")
    saved_fingerprint = payload.get("environment_fingerprint")
    current_fingerprint = getattr(config, "_environment_fingerprint", None)
    if saved_fingerprint is None:
        if not config.allow_legacy_checkpoint_without_fingerprint:
            raise ValueError(
                "Legacy checkpoint has no environment fingerprint. Pass "
                "--allow-legacy-checkpoint-without-fingerprint only after "
                "manually verifying the dataset and environment backend."
            )
        warnings.warn(
            "Loading a legacy checkpoint without an environment fingerprint; "
            "this run is not provenance-verified.",
            RuntimeWarning,
            stacklevel=2,
        )
        config._legacy_checkpoint_without_fingerprint_loaded = True
    elif saved_fingerprint != current_fingerprint:
        raise ValueError(
            "Checkpoint environment fingerprint mismatch: "
            f"saved={saved_fingerprint}, current={current_fingerprint}; "
            f"saved_payload={payload.get('environment_fingerprint_payload')}, "
            f"current_payload={getattr(config, '_environment_fingerprint_payload', None)}"
        )
    saved_config = payload.get("config", {})
    if config.algorithm in ("rpex", "riql_pex", "pex"):
        saved_distribution = saved_config.get("action_distribution")
        if saved_distribution is None:
            raise ValueError(
                "Legacy expansion checkpoint lacks action_distribution metadata; "
                "load it only with a converted checkpoint or retrain explicitly"
            )
        if saved_distribution != config.action_distribution:
            raise ValueError(
                "Checkpoint action_distribution="
                f"{saved_distribution!r}, requested={config.action_distribution!r}"
            )


def _restore_agent_config(config: ExperimentConfig, payload: Dict[str, Any]) -> None:
    """Restore architecture/objective fields while keeping new run controls."""
    saved = payload.get("config", {})
    fields = (
        "hidden_dim",
        "hidden_layers",
        "learning_rate",
        "actor_learning_rate",
        "critic_learning_rate",
        "temperature_learning_rate",
        "max_grad_norm",
        "discount",
        "target_update_rate",
        "deterministic_policy",
        "expectile",
        "beta",
        "riql_sigma",
        "riql_quantile",
        "num_critics",
        "inv_temperature",
        "kappa",
        "sac_num_critics",
        "lcb_ratio",
        "uncertainty_ratio",
        "uncertainty_basic",
        "uncertainty_min",
        "uncertainty_max",
        "entropy_lr",
        "cql_alpha",
        "cql_alpha_online",
        "cql_n_actions",
        "cql_temperature",
        "bc_steps",
        "calql_bc_warmup_steps",
        "backup_entropy",
        "cql_max_target_backup",
        "calibration_mask_mode",
        "enable_calql",
        "cql_importance_sample",
        "orthogonal_initialization",
        "policy_log_std_multiplier",
        "policy_log_std_offset",
        "wsrl_num_critics",
        "wsrl_target_critic_subsample_size",
        "wsrl_layer_norm",
        "wsrl_utd_ratio",
        "wsrl_per_critic_batch_size",
        "mc_return_source",
        "pqe_ensemble_size",
        "pqe_member_offline_steps",
        "pqe_init_online_fraction",
        "pqe_first_epoch_multiplier",
        "pqe_first_online_block_steps",
        "pqe_online_buffer_size",
        "pqe_weight_batch_size",
        "pqe_priority_temperature",
        "pqe_target_update_period",
        "priority_floor",
        "priority_ceiling",
        "action_distribution",
        "ro2o_beta_policy",
        "ro2o_beta_ood",
        "ro2o_q_smooth_eps",
        "ro2o_policy_smooth_eps",
        "ro2o_ood_smooth_eps",
        "ro2o_sample_size",
        "ro2o_uncertainty",
        "ro2o_uncertainty_min",
        "ro2o_uncertainty_decay",
    )
    for name in fields:
        if name in saved:
            setattr(config, name, saved[name])


def _evaluate(
    logger: RunLogger,
    env: object,
    config: ExperimentConfig,
    agent: object,
    normalizer: StateNormalizer,
    device: torch.device,
    phase: str,
    step: int,
    env_steps: int,
) -> None:
    modes = (
        ("method_faithful", "deterministic_diagnostic")
        if config.evaluation_mode == "both"
        else (config.evaluation_mode,)
    )
    evaluations = {
        mode: evaluate_agent(
            env,
            config.env_name,
            agent,
            normalizer,
            device,
            config.eval_episodes,
            config.max_episode_steps,
            config.eval_seed,
            config.protocol,
            mode,
            config.action_execution_profile,
        )
        for mode in modes
    }
    primary_mode = (
        "method_faithful"
        if config.algorithm == "rpex" and "method_faithful" in evaluations
        else "deterministic_diagnostic"
        if "deterministic_diagnostic" in evaluations
        else modes[0]
    )
    metrics = dict(evaluations[primary_mode])
    for mode, values in evaluations.items():
        suffix = "deterministic" if mode == "deterministic_diagnostic" else "method_faithful"
        metrics[f"return_{suffix}"] = values["return_mean"]
        metrics[f"normalized_return_{suffix}"] = values[
            "normalized_return_mean"
        ]
    metrics["evaluation_mode"] = primary_mode
    logger.log_evaluation(
        phase, step, env_steps, agent.total_updates, metrics
    )


def run_experiment(
    config: ExperimentConfig,
    logger: RunLogger,
    *,
    final_audit_receipt: Optional[Dict[str, Any]] = None,
) -> Path:
    device = resolve_device(config.device, config.cuda_device)
    seed_everything(config.learner_seed)
    dataset_env = make_env(config.env_name, config.protocol)
    try:
        seed_env_only(dataset_env, config.train_env_seed)
        raw_dataset = load_d4rl_dataset(
            dataset_env,
            config.dataset_dir,
            config.discount,
            config.protocol,
            config.env_name,
        )
        protocol_metadata = environment_metadata(
            dataset_env,
            config.env_name,
            raw_dataset,
            config.train_env_seed,
            config.protocol,
            config.dataset_dir,
        )
    finally:
        dataset_env.close()

    state_dim = raw_dataset["observations"].shape[1]
    action_dim = raw_dataset["actions"].shape[1]
    if config.protocol == LEGACY_PROTOCOL:
        expected_dims = EXPECTED_LOCOMOTION_DIMS[
            config.env_name.split("-", 1)[0]
        ]
        if (state_dim, action_dim) != expected_dims:
            raise RuntimeError(
                "Strict D4RL-v2 observation/action dimensions mismatch: "
                f"expected={expected_dims}, actual={(state_dim, action_dim)}"
            )

    with ExitStack() as stack:
        env = make_env(config.env_name, config.protocol)
        stack.callback(env.close)
        eval_env = _make_evaluation_env(
            config.env_name, config.protocol, config.eval_seed
        )
        stack.callback(eval_env.close)
        seed_env_only(env, config.train_env_seed)
        for role, instance in (("online", env), ("evaluation", eval_env)):
            spec_id = getattr(getattr(instance, "spec", None), "id", None)
            expected_spec_id = expected_env_spec_id(config.env_name, config.protocol)
            if spec_id != expected_spec_id:
                raise RuntimeError(
                    f"{role} environment ID mismatch: {spec_id!r} != "
                    f"{expected_spec_id!r}"
                )
            if tuple(instance.observation_space.shape) != (state_dim,):
                raise RuntimeError(f"{role} observation dimension mismatch")
            if tuple(instance.action_space.shape) != (action_dim,):
                raise RuntimeError(f"{role} action dimension mismatch")
            if not np.array_equal(
                np.asarray(instance.action_space.low, dtype=np.float64),
                np.asarray(protocol_metadata["action_low"], dtype=np.float64),
            ) or not np.array_equal(
                np.asarray(instance.action_space.high, dtype=np.float64),
                np.asarray(protocol_metadata["action_high"], dtype=np.float64),
            ):
                raise RuntimeError(f"{role} action bounds mismatch")
            instance_horizon = getattr(instance, "_max_episode_steps", None)
            if instance_horizon is None:
                instance_horizon = getattr(
                    getattr(instance, "spec", None), "max_episode_steps", None
                )
            if int(instance_horizon) != int(
                protocol_metadata["environment_max_episode_steps"]
            ):
                raise RuntimeError(f"{role} environment horizon mismatch")
        max_action = float(np.max(np.abs(env.action_space.high)))
        environment_horizon = protocol_metadata["environment_max_episode_steps"]
        if (
            config.protocol == LEGACY_PROTOCOL
            and config.max_episode_steps != environment_horizon
        ):
            raise RuntimeError(
                "Strict protocol horizon mismatch: "
                f"--max-episode-steps={config.max_episode_steps}, "
                f"environment spec={environment_horizon}"
            )
        config._environment_fingerprint = protocol_metadata[
            "environment_fingerprint"
        ]
        config._environment_fingerprint_payload = protocol_metadata[
            "environment_fingerprint_payload"
        ]

        checkpoint_payload: Optional[Dict[str, Any]] = None
        resume_payload: Optional[Dict[str, Any]] = None
        if config.resume_run:
            _, checkpoint_payload = resolve_resume_checkpoint(config.resume_run, device)
            _validate_checkpoint(checkpoint_payload, config, state_dim, action_dim)
            _restore_agent_config(config, checkpoint_payload)
            if config.run_purpose == "final_benchmark":
                config._validate_final_benchmark()
            resume_payload = checkpoint_payload["resume_state"]
        if config.initialize_from_checkpoint:
            checkpoint_path = Path(
                config.initialize_from_checkpoint
            ).expanduser().resolve()
            checkpoint_payload = _torch_load(checkpoint_path, device)
            _validate_checkpoint(checkpoint_payload, config, state_dim, action_dim)
            _restore_agent_config(config, checkpoint_payload)
            if config.run_purpose == "final_benchmark":
                config._validate_final_benchmark()

        oracle: Optional[AttackOracle] = make_attack_oracle(
            config, state_dim, action_dim, max_action, device
        )
        if oracle is not None:
            stack.callback(oracle.close)
        cache_root = (
            results_root_from_output(config.output_dir)
            / "attack_cache"
            / config.protocol
        )
        corrupted_dataset, corruption_stats = corrupt_offline_dataset(
            raw_dataset, config, oracle, cache_root
        )

        if checkpoint_payload is not None:
            normalizer = StateNormalizer.from_state_dict(
                checkpoint_payload["normalizer"]
            )
        else:
            normalizer = StateNormalizer.fit(
                corrupted_dataset,
                enabled=config.normalize_states,
                mode=config.state_normalization,
                additive_epsilon=(
                    config.implementation_profile == "official_code_reference"
                ),
            )
        normalized_dataset = apply_normalizer(corrupted_dataset, normalizer)
        replay_sampling_profile = (
            RPEX_OFFICIAL_REPLAY_SAMPLING
            if config.implementation_profile == "official_code_reference"
            and config.algorithm in ("rpex", "riql_naive", "riql_pex")
            else NUMPY_REPLAY_SAMPLING
        )
        offline = OfflineDataset(
            normalized_dataset,
            config.replay_seed,
            sampling_profile=replay_sampling_profile,
        )
        if resume_payload is not None and resume_payload.get("offline_dataset"):
            offline.load_state_dict(resume_payload["offline_dataset"])

        # Decouple learner initialization and subsequent training randomness
        # from preprocessing, adversarial attacks, and cache hit/miss state.
        seed_everything(config.learner_seed)
        agent = build_agent(config, state_dim, action_dim, max_action, device)
        if config.algorithm == "pessimistic_q_ensemble":
            artifact_identity = (
                str(corruption_stats.get("cache_key", "clean")),
                str(corruption_stats["final_artifact_sha256"]),
            )
            agent.bind_offline_artifact(*artifact_identity)
        if checkpoint_payload is not None:
            agent.load_checkpoint_state(checkpoint_payload["agent"])
            if (
                config.algorithm == "pessimistic_q_ensemble"
                and tuple(agent.offline_artifact_identity or ())
                != artifact_identity
            ):
                raise ValueError(
                    "PQE checkpoint offline artifact differs from this run's "
                    "corrupted D4RL-v2 artifact"
                )
        elif (
            config.algorithm == "pessimistic_q_ensemble"
            and config.stage == "online"
        ):
            _load_pqe_member_checkpoints(
                config.pqe_member_checkpoints,
                agent,
                config,
                normalizer,
                state_dim,
                action_dim,
                device,
            )
        if resume_payload is not None:
            restore_global_rng_state(resume_payload["global_rng"], oracle=oracle)

        # RunLogger.write_config intentionally commits a resume by superseding
        # any completion marker and transitioning summary.json to `running`.
        # Validate append safety first so a rejected checkpoint is read-only.
        _validate_resume_precommit(config, logger, resume_payload)

        diagnostic_snapshot = (
            _parameter_snapshot(agent) if config.diagnostic_mode else None
        )
        diagnostic_initial_updates = agent.total_updates
        diagnostic_initial_update_counts = {
            "actor": _agent_update_count(agent, "actor"),
            "critic": _agent_update_count(agent, "critic"),
            "temperature": _agent_update_count(agent, "temperature"),
        }
        diagnostic_initial_return: float | None = None

        logger.write_config(
            {
                **protocol_metadata,
                **config.to_dict(),
                "resolved_device": str(device),
                "run_dir": str(logger.run_dir),
                "state_dim": state_dim,
                "action_dim": action_dim,
                "offline_corruption": corruption_stats,
                "normalizer": normalizer.diagnostics(corrupted_dataset),
                "normalizer_sha256": normalizer_sha256(normalizer),
                **(
                    agent.algorithm_metadata()
                    if hasattr(agent, "algorithm_metadata")
                    else {}
                ),
                "legacy_checkpoint_without_fingerprint_loaded": bool(
                    getattr(
                        config,
                        "_legacy_checkpoint_without_fingerprint_loaded",
                        False,
                    )
                ),
            }
        )
        if final_audit_receipt is not None:
            from .final_gate import write_final_audit_evidence

            evidence_dir = logger.run_dir
            if config.resume_run:
                receipt_sha256 = str(
                    getattr(config, "_final_audit_receipt_sha256", "unknown")
                )
                evidence_dir = (
                    logger.run_dir / "resume_audit_evidence" / receipt_sha256
                )
                evidence_dir.mkdir(parents=True, exist_ok=True)
            write_final_audit_evidence(evidence_dir, final_audit_receipt)
        logger.logger.info(
            "protocol=%s algorithm=%s env=%s corruption=%s/%s device=%s",
            config.protocol,
            config.algorithm,
            config.env_name,
            config.corruption,
            config.corruption_target,
            device,
        )
        if config.algorithm == "pessimistic_q_ensemble":
            logger.logger.info(
                "PQE uses five independent actor/twin-critic CQL members; "
                "offline compute multiplier=%d",
                config.pqe_ensemble_size,
            )
        logger.logger.info(
            "dataset=%d offline_steps=%d online_steps=%d offline_ratio=%.2f",
            offline.size,
            config.offline_steps,
            config.online_steps,
            config.effective_offline_ratio,
        )

        if config.diagnostic_mode:
            initial_metrics = evaluate_agent(
                eval_env,
                config.env_name,
                agent,
                normalizer,
                device,
                config.eval_episodes,
                config.max_episode_steps,
                config.eval_seed,
                config.protocol,
                "deterministic_diagnostic",
                config.action_execution_profile,
            )
            diagnostic_initial_return = initial_metrics["return_mean"]
            logger.log_evaluation(
                "diagnostic_initial",
                0,
                0,
                agent.total_updates,
                {**initial_metrics, "evaluation_mode": "deterministic_diagnostic"},
            )

        if config.stage in ("offline", "both"):
            if resume_payload is not None and resume_payload.get("phase") != "offline":
                pass
            else:
                _run_offline(
                    eval_env,
                    config,
                    agent,
                    offline,
                    normalizer,
                    device,
                    logger,
                    state_dim,
                    action_dim,
                    resume_state=resume_payload,
                )
        if config.stage in ("online", "both"):
            if not agent.online_phase:
                agent.begin_online()
            _run_online(
                env,
                eval_env,
                raw_dataset,
                config,
                agent,
                offline,
                normalizer,
                oracle,
                device,
                logger,
                state_dim,
                action_dim,
                resume_state=resume_payload,
            )
        else:
            logger.write_completion_manifest(
                {
                    "online_budget_semantics": "online_phase_not_run",
                    "requested_online_steps": config.online_steps,
                    "actual_online_steps": 0,
                    "episode_boundary_overshoot": 0,
                    **_runtime_update_metadata(config, agent),
                    **(
                        agent.algorithm_metadata()
                        if hasattr(agent, "algorithm_metadata")
                        else {}
                    ),
                }
            )
        if diagnostic_snapshot is not None:
            parameter_deltas, _ = _parameter_deltas(agent, diagnostic_snapshot)
            final_metrics = evaluate_agent(
                eval_env,
                config.env_name,
                agent,
                normalizer,
                device,
                config.eval_episodes,
                config.max_episode_steps,
                config.eval_seed,
                config.protocol,
                "deterministic_diagnostic",
                config.action_execution_profile,
            )
            final_return = final_metrics["return_mean"]
            evidence = {
                "initial_deterministic_return": diagnostic_initial_return,
                "final_deterministic_return": final_return,
                "return_delta": final_return - float(diagnostic_initial_return),
                **parameter_deltas,
                "completed_updates": agent.total_updates
                - diagnostic_initial_updates,
                "completed_actor_updates": _agent_update_count(agent, "actor")
                - diagnostic_initial_update_counts["actor"],
                "completed_critic_updates": _agent_update_count(agent, "critic")
                - diagnostic_initial_update_counts["critic"],
                "completed_temperature_updates": _agent_update_count(
                    agent, "temperature"
                )
                - diagnostic_initial_update_counts["temperature"],
            }
            with (logger.run_dir / "diagnostic_evidence.json").open(
                "w", encoding="utf-8"
            ) as stream:
                json.dump(evidence, stream, indent=2, ensure_ascii=False)
    return logger.run_dir


def _run_offline(
    env: object,
    config: ExperimentConfig,
    agent: object,
    offline: OfflineDataset,
    normalizer: StateNormalizer,
    device: torch.device,
    logger: RunLogger,
    state_dim: int,
    action_dim: int,
    resume_state: Optional[Dict[str, Any]] = None,
) -> None:
    logger.logger.info("offline pre-training started")
    is_pqe = config.algorithm == "pessimistic_q_ensemble"
    offline_budget = int(
        config.pqe_member_offline_steps if is_pqe else config.offline_steps
    )
    pqe_member_datasets = (
        [
            OfflineDataset(
                offline.dataset,
                seed=int(member_seed),
                sampling_profile=NUMPY_REPLAY_SAMPLING,
            )
            for member_seed in agent.member_seeds
        ]
        if is_pqe
        else []
    )
    last_metrics: Dict[str, float] = {}
    accumulator = MetricAccumulator()
    start_step = 0
    if resume_state is not None and resume_state.get("phase") == "offline":
        _validate_writer_positions(logger, resume_state)
        start_step = int(resume_state["phase_step"])
        if start_step > offline_budget:
            raise ValueError(
                "offline resume checkpoint exceeds the configured update budget: "
                f"checkpoint={start_step}, configured={offline_budget}"
            )
        if is_pqe:
            sampler_states = resume_state.get("pqe_member_offline_datasets")
            if not isinstance(sampler_states, list) or len(sampler_states) != 5:
                raise ValueError(
                    "PQE offline resume requires five member sampler states"
                )
            for sampler, sampler_state in zip(
                pqe_member_datasets, sampler_states
            ):
                sampler.load_state_dict(sampler_state)
        if start_step == offline_budget:
            logger.logger.info(
                "offline resume is already complete at update=%d; no new "
                "evaluation, optimizer update, or checkpoint write was executed",
                start_step,
            )
            return
        accumulator.values = {
            key: list(values)
            for key, values in resume_state.get("metric_accumulator", {}).items()
        }
    parameter_snapshot = _parameter_snapshot(agent)
    for step in range(start_step + 1, offline_budget + 1):
        if is_pqe:
            last_metrics = agent.update(
                member_batches=[
                    sampler.sample(config.batch_size, device)
                    for sampler in pqe_member_datasets
                ]
            )
        else:
            batch = offline.sample(config.batch_size, device)
            last_metrics = agent.update(batch)
        accumulator.add(last_metrics)
        if step % config.train_log_period == 0:
            parameter_deltas, parameter_snapshot = _parameter_deltas(
                agent, parameter_snapshot
            )
            logger.log_train(
                "offline",
                step,
                0,
                agent.total_updates,
                {
                    **accumulator.means(),
                    **parameter_deltas,
                    **_policy_log_std_metrics(agent),
                    "replay_size_offline": float(offline.size),
                    "replay_size_online": 0.0,
                    "offline_batch_fraction": 1.0,
                    "online_batch_fraction": 0.0,
                    "offline_compute_multiplier": float(
                        config.pqe_ensemble_size if is_pqe else 1
                    ),
                },
            )
        if step % config.eval_period == 0:
            _evaluate(
                logger,
                env,
                config,
                agent,
                normalizer,
                device,
                "offline",
                step,
                0,
            )
        period = config.effective_offline_checkpoint_period
        if (
            period > 0
            and step % period == 0
            and step != offline_budget
        ):
            _save_phase_checkpoint(
                logger,
                agent,
                config,
                normalizer,
                "offline",
                step,
                0,
                state_dim,
                action_dim,
                final=False,
                resume_state={
                    "phase": "offline",
                    "phase_step": step,
                    "episode_boundary": True,
                    "offline_dataset": offline.state_dict(),
                    "pqe_member_offline_datasets": [
                        sampler.state_dict() for sampler in pqe_member_datasets
                    ],
                    "global_rng": capture_global_rng_state(),
                    "metric_accumulator": accumulator.values,
                    "writer_append_position": _writer_positions(logger),
                },
            )
    if (
        config.implementation_profile != "official_code_reference"
        and (
            offline_budget == 0
            or offline_budget % config.eval_period != 0
        )
    ):
        _evaluate(
            logger,
            env,
            config,
            agent,
            normalizer,
            device,
            "offline",
            offline_budget,
            0,
        )
    _save_phase_checkpoint(
        logger,
        agent,
        config,
        normalizer,
        "offline",
        offline_budget,
        0,
        state_dim,
        action_dim,
        final=True,
        resume_state={
            "phase": "offline",
            "phase_step": offline_budget,
            "episode_boundary": True,
            "offline_dataset": offline.state_dict(),
            "pqe_member_offline_datasets": [
                sampler.state_dict() for sampler in pqe_member_datasets
            ],
            "global_rng": capture_global_rng_state(),
            "metric_accumulator": accumulator.values,
            "writer_append_position": _writer_positions(logger),
        },
    )
    logger.logger.info("offline pre-training completed")


def _run_online(
    env: object,
    eval_env: object,
    raw_dataset: Dict[str, np.ndarray],
    config: ExperimentConfig,
    agent: object,
    offline: OfflineDataset,
    normalizer: StateNormalizer,
    oracle: Optional[AttackOracle],
    device: torch.device,
    logger: RunLogger,
    state_dim: int,
    action_dim: int,
    resume_state: Optional[Dict[str, Any]] = None,
) -> None:
    logger.logger.info("online fine-tuning started")
    is_pqe = config.algorithm == "pessimistic_q_ensemble"
    is_calql = config.algorithm == "cal_ql"
    is_wsrl = config.algorithm == "wsrl"
    replay = ReplayBuffer(
        state_dim,
        action_dim,
        config.pqe_online_buffer_size if is_pqe else config.replay_size,
        config.replay_seed,
        sampling_profile=offline.sampling_profile,
    )
    calql_trajectory = (
        CalQLTrajectoryAccumulator(config.discount) if is_calql else None
    )
    rng = make_numpy_corruption_rng(config)
    state_std = raw_dataset["observations"].std(axis=0).astype(np.float32)
    action_std = raw_dataset["actions"].std(axis=0).astype(np.float32)
    online_resume = resume_state is not None and resume_state.get("phase") == "online"
    start_env_step = int(resume_state["phase_step"]) if online_resume else 0
    online_initial_update_counts = {
        "actor": int(
            resume_state.get(
                "online_initial_actor_updates",
                _agent_update_count(agent, "actor"),
            )
            if online_resume
            else _agent_update_count(agent, "actor")
        ),
        "critic": int(
            resume_state.get(
                "online_initial_critic_updates",
                _agent_update_count(agent, "critic"),
            )
            if online_resume
            else _agent_update_count(agent, "critic")
        ),
        "temperature": int(
            resume_state.get(
                "online_initial_temperature_updates",
                _agent_update_count(agent, "temperature"),
            )
            if online_resume
            else _agent_update_count(agent, "temperature")
        ),
    }
    rpex_episode_boundary_budget = (
        config.implementation_profile == "official_code_reference"
        and config.algorithm in ("rpex", "riql_naive", "riql_pex")
    )
    calql_episode_boundary_budget = bool(is_calql and config.online_steps > 0)
    pqe_block_update_counts, pqe_schedule_inferred_from_legacy = (
        _restore_pqe_block_schedule(
            config,
            resume_state if online_resume else None,
            start_env_step,
        )
    )
    if online_resume:
        _validate_writer_positions(logger, resume_state)
        if not resume_state.get("episode_boundary"):
            raise ValueError("exact online resume is only supported at episode boundaries")
        already_complete = (
            start_env_step > config.online_steps
            if rpex_episode_boundary_budget
            else start_env_step >= config.online_steps
        )
        if already_complete:
            if is_calql:
                trajectory_state = resume_state.get("calql_trajectory")
                if trajectory_state is None:
                    raise ValueError(
                        "Cal-QL online resume is missing trajectory-return state"
                    )
                calql_trajectory.load_state_dict(trajectory_state)
                if calql_trajectory.pending_episode_length:
                    raise ValueError(
                        "completed Cal-QL resume contains a pending trajectory"
                    )
            corruption_audit = OnlineCorruptionAudit(
                resume_state.get("online_corruption_audit")
            )
            online_corruption_metadata = corruption_audit.metadata(config)
            online_corruption_metadata.update(
                {
                    "online_budget_semantics": (
                        "rpex_official_episode_boundary_strict_greater_than"
                        if rpex_episode_boundary_budget
                        else "calql_complete_current_episode_at_or_after_requested"
                        if is_calql
                        else "exact_environment_steps"
                    ),
                    "requested_online_steps": config.online_steps,
                    "actual_online_steps": start_env_step,
                    "episode_boundary_overshoot": max(
                        start_env_step - config.online_steps, 0
                    ),
                    "resume_noop_already_complete": True,
                    "raw_action_oob_fraction": (
                        float(resume_state.get("raw_oob", 0) / start_env_step)
                        if start_env_step > 0
                        else 0.0
                    ),
                    "raw_action_abs_max": float(
                        resume_state.get("raw_action_abs_max", 0.0)
                    ),
                    "executed_action_abs_max": float(
                        resume_state.get("executed_action_abs_max", 0.0)
                    ),
                }
            )
            online_corruption_metadata.update(
                _wsrl_runtime_metadata(
                    config,
                    agent,
                    online_initial_update_counts=online_initial_update_counts,
                )
            )
            online_corruption_metadata.update(
                _runtime_update_metadata(
                    config,
                    agent,
                    online_initial_update_counts=online_initial_update_counts,
                    online_environment_steps=start_env_step,
                )
            )
            if is_calql:
                online_corruption_metadata.update(calql_trajectory.metadata())
                online_corruption_metadata[
                    "effective_calql_training_transitions"
                ] = int(
                    calql_trajectory.completed_transitions
                )
                online_corruption_metadata["calql_online_cql_enabled"] = True
                online_corruption_metadata[
                    "calql_dynamic_offline_ratio"
                ] = float(
                    resume_state.get(
                        "calql_dynamic_offline_ratio",
                        offline.size
                        / max(
                            offline.size
                            + calql_trajectory.completed_transitions,
                            1,
                        ),
                    )
                )
                online_corruption_metadata[
                    "online_calibration_bound_rate"
                ] = float(
                    resume_state.get(
                        "calql_last_online_calibration_bound_rate", 0.0
                    )
                )
            if is_pqe:
                if hasattr(agent, "algorithm_metadata"):
                    online_corruption_metadata.update(agent.algorithm_metadata())
                online_corruption_metadata.update(
                    _pqe_block_schedule_metadata(
                        config,
                        pqe_block_update_counts,
                        env_step=start_env_step,
                        inferred_from_legacy=pqe_schedule_inferred_from_legacy,
                    )
                )
            with (logger.run_dir / "online_corruption_manifest.json").open(
                "w", encoding="utf-8"
            ) as stream:
                json.dump(
                    online_corruption_metadata,
                    stream,
                    indent=2,
                    ensure_ascii=False,
                )
            logger.write_completion_manifest(online_corruption_metadata)
            logger.logger.info(
                "online resume is already complete at env_step=%d; no new "
                "environment transition or optimizer update was executed",
                start_env_step,
            )
            return
        replay.load_state_dict(resume_state["online_replay"])
        if is_calql:
            trajectory_state = resume_state.get("calql_trajectory")
            if trajectory_state is None:
                raise ValueError(
                    "Cal-QL online resume is missing trajectory-return state"
                )
            calql_trajectory.load_state_dict(trajectory_state)
            if calql_trajectory.pending_episode_length:
                raise ValueError(
                    "Cal-QL exact resume requires an empty pending trajectory"
                )
        restore_global_rng_state(resume_state["global_rng"], rng, oracle)
        restore_environment_rng_state(env, resume_state["environment_rng"])
        raw_state = None
    else:
        # A valid custom research budget may contain no online interaction.
        # Keep that state at an episode boundary so the final exact-resume
        # snapshot is well-defined instead of resetting an episode that is
        # never stepped.
        raw_state = (
            reset_env(env, seed=config.train_env_seed, protocol=config.protocol)
            if config.online_steps > 0
            else None
        )
    episode_steps = 0
    episode_return = 0.0
    corrupted_online = int(resume_state.get("corrupted_online", 0)) if online_resume else 0
    corruption_audit = OnlineCorruptionAudit(
        resume_state.get("online_corruption_audit") if online_resume else None
    )
    last_metrics: Dict[str, float] = {}
    accumulator = MetricAccumulator()
    if online_resume:
        accumulator.values = {
            key: list(values)
            for key, values in resume_state.get("metric_accumulator", {}).items()
        }
    parameter_snapshot = _parameter_snapshot(agent)
    offline_ratio = config.effective_offline_ratio
    current_offline_ratio = float(offline_ratio)
    if is_wsrl and not np.isclose(offline_ratio, 0.0):
        raise ValueError(
            "WSRL online fine-tuning is online-replay-only; "
            f"effective_offline_ratio must be 0.0, got {offline_ratio}"
        )
    warmup = (
        max(config.initial_collection_steps, config.warmup_steps)
        if is_wsrl
        else config.pqe_first_online_block_steps
        if is_pqe
        else 0
        if is_calql
        else config.initial_collection_steps
    )
    action_low = np.asarray(env.action_space.low, dtype=np.float32)
    action_high = np.asarray(env.action_space.high, dtype=np.float32)
    raw_action_abs_max = float(resume_state.get("raw_action_abs_max", 0.0)) if online_resume else 0.0
    executed_action_abs_max = float(resume_state.get("executed_action_abs_max", 0.0)) if online_resume else 0.0
    raw_oob = int(resume_state.get("raw_oob", 0)) if online_resume else 0
    executed_oob = int(resume_state.get("executed_oob", 0)) if online_resume else 0
    replay_mismatch = int(resume_state.get("replay_mismatch", 0)) if online_resume else 0
    priority_metrics: Dict[str, float] = {}
    initial_online_priority = (
        agent.initial_online_priority(
            offline.size, config.pqe_first_online_block_steps
        )
        if is_pqe
        else 1.0
    )
    pqe_first_block_updates_applied = bool(pqe_block_update_counts)
    pqe_first_block_update_count = (
        int(pqe_block_update_counts[0]) if pqe_block_update_counts else 0
    )

    def online_resume_snapshot(step: int) -> Dict[str, Any]:
        if not episode_boundary or raw_state is not None:
            raise RuntimeError("online exact-resume snapshots require an episode boundary")
        return {
            "phase": "online",
            "phase_step": step,
            "episode_boundary": True,
            "current_observation": None,
            "episode_step": 0,
            "episode_return": 0.0,
            "online_replay": replay.state_dict(),
            "offline_dataset": offline.state_dict(),
            "global_rng": capture_global_rng_state(rng, oracle),
            "environment_rng": capture_environment_rng_state(env),
            "corrupted_online": corrupted_online,
            "online_corruption_audit": corruption_audit.state_dict(),
            "raw_action_abs_max": raw_action_abs_max,
            "executed_action_abs_max": executed_action_abs_max,
            "raw_oob": raw_oob,
            "executed_oob": executed_oob,
            "replay_mismatch": replay_mismatch,
            "calql_trajectory": (
                calql_trajectory.state_dict() if is_calql else None
            ),
            "calql_dynamic_offline_ratio": (
                float(current_offline_ratio) if is_calql else None
            ),
            "calql_last_online_calibration_bound_rate": (
                float(last_metrics.get("online_calibration_bound_rate", 0.0))
                if is_calql
                else None
            ),
            "pqe_first_block_updates_applied": pqe_first_block_updates_applied,
            "pqe_first_block_update_count": pqe_first_block_update_count,
            **(
                _pqe_block_schedule_metadata(
                    config,
                    pqe_block_update_counts,
                    env_step=step,
                    inferred_from_legacy=pqe_schedule_inferred_from_legacy,
                )
                if is_pqe
                else {}
            ),
            "metric_accumulator": accumulator.values,
            "writer_append_position": _writer_positions(logger),
            "online_initial_actor_updates": online_initial_update_counts[
                "actor"
            ],
            "online_initial_critic_updates": online_initial_update_counts[
                "critic"
            ],
            "online_initial_temperature_updates": online_initial_update_counts[
                "temperature"
            ],
        }

    pending_checkpoint = False
    episode_boundary = True
    if rpex_episode_boundary_budget or calql_episode_boundary_budget:
        import itertools

        online_step_iterator = itertools.count(start_env_step + 1)
    else:
        online_step_iterator = range(
            start_env_step + 1, config.online_steps + 1
        )

    # RPEX/RIQL source ordering is independent of whether the controller uses
    # the official episode-overshoot budget or the exact custom budget.
    official_pre_transition_updates = (
        config.algorithm in ("rpex", "riql_naive", "riql_pex")
        and config.implementation_profile
        in ("official_code_reference", "research_benchmark")
    )
    update_batch_size = (
        config.wsrl_per_critic_batch_size
        if is_wsrl
        else config.batch_size
    )
    wsrl_total_batch_size = (
        config.wsrl_utd_ratio * config.wsrl_per_critic_batch_size
        if is_wsrl
        else 0
    )
    required_online_samples = (
        config.pqe_weight_batch_size
        if is_pqe
        else 1
        if is_calql
        else wsrl_total_batch_size
        if is_wsrl
        else max(
            update_batch_size - int(round(update_batch_size * offline_ratio)),
            1,
        )
    )
    wsrl_first_update_step = (
        _wsrl_first_update_env_step(
            warmup_steps=warmup,
            total_batch_size=wsrl_total_batch_size,
        )
        if is_wsrl
        else 0
    )
    last_count_log_env_step = start_env_step
    last_logged_actor_updates = _agent_update_count(agent, "actor")
    last_logged_critic_updates = _agent_update_count(agent, "critic")
    last_logged_temperature_updates = _agent_update_count(
        agent, "temperature"
    )

    def perform_online_updates(env_step: int, *, before_transition: bool) -> None:
        nonlocal last_metrics, priority_metrics
        nonlocal pqe_first_block_updates_applied, pqe_first_block_update_count

        if is_calql:
            # Cal-QL is updated only after a complete trajectory has exact RTG.
            return
        if is_wsrl:
            can_update = (
                env_step >= wsrl_first_update_step
                and replay.size >= required_online_samples
            )
        elif is_pqe:
            full_block_boundary = (
                env_step % config.pqe_first_online_block_steps == 0
            )
            can_update = (
                not before_transition
                and full_block_boundary
                and env_step // config.pqe_first_online_block_steps
                > len(pqe_block_update_counts)
                and replay.size >= required_online_samples
            )
        elif before_transition:
            can_update = (
                replay.size > warmup
                and replay.size >= required_online_samples
            )
        else:
            can_update = (
                env_step > warmup
                and replay.size >= required_online_samples
            )
        if not can_update:
            return

        if is_wsrl:
            last_metrics = _run_wsrl_high_utd_update(
                agent,
                offline,
                replay,
                config,
                device,
            )
            accumulator.add(last_metrics)
            return

        if is_pqe:
            completed_blocks_at_boundary = int(
                env_step // config.pqe_first_online_block_steps
            )
            normal_update_count = int(
                config.pqe_first_online_block_steps * config.updates_per_step
            )
            # The frozen research profile has exactly one block due here.  The
            # range keeps explicitly non-reference miniature configurations
            # well-defined if their density batch is larger than one block.
            for block_index in range(
                len(pqe_block_update_counts), completed_blocks_at_boundary
            ):
                update_repeats = int(
                    agent.online_update_count_for_block(
                        block_index, normal_update_count
                    )
                )
                if update_repeats < 0:
                    raise RuntimeError("PQE block update count cannot be negative")
                for _ in range(update_repeats):
                    (
                        batch,
                        density_offline_batch,
                        density_online_batch,
                    ) = sample_pqe_update_batches(
                        offline,
                        replay,
                        config.batch_size,
                        offline_ratio,
                        device,
                        prioritized_rl=(
                            config.pqe_replay_mode == "balanced_density"
                        ),
                        density_batch_size=config.pqe_weight_batch_size,
                    )
                    last_metrics = agent.update(
                        rl_batch=batch,
                        density_offline_batch=density_offline_batch,
                        density_online_batch=density_online_batch,
                        rl_batch_prioritized=(
                            config.pqe_replay_mode == "balanced_density"
                        ),
                    )
                    accumulator.add(last_metrics)
                    priorities = agent.consume_priority_values()
                    if priorities is None:
                        raise RuntimeError(
                            "Pessimistic Q-Ensemble skipped a required priority update"
                        )
                    priority_metrics = update_sample_priorities(
                        offline, replay, batch, priorities
                    )
                pqe_block_update_counts.append(update_repeats)
                pqe_first_block_updates_applied = True
                pqe_first_block_update_count = int(pqe_block_update_counts[0])
            return

        for _ in range(config.updates_per_step):
            batch = mixed_batch(
                offline,
                replay,
                update_batch_size,
                offline_ratio,
                device,
            )
            last_metrics = agent.update(batch)
            accumulator.add(last_metrics)

    def perform_calql_trajectory_updates(completed: object) -> None:
        nonlocal last_metrics, current_offline_ratio

        if not is_calql:
            raise RuntimeError("Cal-QL trajectory update called for another algorithm")
        update_count = completed.update_count(config.updates_per_step)
        for _ in range(update_count):
            offline_count, online_count, ratio = dynamic_batch_counts(
                config.batch_size, offline.size, replay.size
            )
            if online_count <= 0:
                raise RuntimeError(
                    "Cal-QL dynamic mixture produced no completed online sample"
                )
            batch = concatenate_batches(
                offline.sample(offline_count, device),
                replay.sample(
                    online_count,
                    device,
                    replace=True,
                ),
            )
            last_metrics = agent.update(batch)
            last_metrics["calql_dynamic_offline_ratio"] = float(ratio)
            last_metrics["calql_trajectory_update_count"] = float(update_count)
            accumulator.add(last_metrics)
            current_offline_ratio = float(ratio)
    actual_online_steps = start_env_step
    for env_step in online_step_iterator:
        actual_online_steps = env_step
        if raw_state is None:
            raw_state = reset_env(env, protocol=config.protocol)
        episode_boundary = False
        pre_action = (
            config.random_attack_semantics
            == "pre_action_sensor_actuator_corruption"
        )
        selected_target = (
            sample_online_corruption_target(config, rng) if pre_action else None
        )
        policy_state = raw_state
        if pre_action and selected_target == "observations":
            policy_state = corrupt_pre_action_value(
                raw_state,
                "observations",
                raw_state,
                None,
                config,
                oracle,
                rng,
                state_std,
                action_std,
            )
        state_tensor = torch.as_tensor(
            normalizer.transform(policy_state), dtype=torch.float32, device=device
        )
        with torch.no_grad():
            raw_policy_action = agent.select_action(state_tensor, evaluate=False)
        raw_action_np = raw_policy_action.detach().cpu().numpy().astype(np.float32)
        if pre_action and selected_target == "actions":
            raw_action_np = corrupt_pre_action_value(
                raw_action_np,
                "actions",
                raw_state,
                raw_action_np,
                config,
                oracle,
                rng,
                state_std,
                action_std,
            )
        executed_action = (
            bounded_executed_action(raw_action_np, action_low, action_high)
            if config.action_execution_profile == "clip_to_action_space"
            else raw_action_np.copy()
        )
        raw_action_abs_max = max(raw_action_abs_max, float(np.abs(raw_action_np).max()))
        raw_oob += int(
            np.any(raw_action_np < action_low) or np.any(raw_action_np > action_high)
        )
        executed_action_abs_max = max(
            executed_action_abs_max, float(np.abs(executed_action).max())
        )
        executed_oob += int(
            np.any(executed_action < action_low) or np.any(executed_action > action_high)
        )
        if official_pre_transition_updates:
            perform_online_updates(env_step, before_transition=True)
        raw_next_state, reward, terminated, truncated, _ = step_env(
            env, executed_action, protocol=config.protocol
        )
        episode_steps += 1
        episode_return += reward
        episode_timeout = bool(
            truncated or episode_steps >= config.max_episode_steps
        )
        episode_finished = bool(terminated or episode_timeout)

        if not pre_action:
            selected_target = sample_online_corruption_target(config, rng)

        if pre_action and selected_target in ("observations", "actions"):
            normalized_replay_poisoning = False
            stored_state = policy_state.copy()
            stored_action = executed_action.copy()
            stored_reward = float(reward)
            stored_next_state = raw_next_state.copy()
            was_corrupted = True
        else:
            normalized_replay_poisoning = _poison_replay_in_learner_coordinates(
                config
            )
            corruption_state = (
                normalizer.transform(raw_state)
                if normalized_replay_poisoning
                else raw_state
            )
            corruption_next_state = (
                normalizer.transform(raw_next_state)
                if normalized_replay_poisoning
                else raw_next_state
            )
            (
                stored_state,
                stored_action,
                stored_reward,
                stored_next_state,
                was_corrupted,
            ) = corrupt_online_transition(
                corruption_state,
                executed_action,
                reward,
                corruption_next_state,
                config,
                oracle,
                rng,
                state_std,
                action_std,
                selected_target=selected_target,
                selection_already_sampled=True,
            )
        corrupted_online += int(was_corrupted)
        if was_corrupted and selected_target is not None:
            corruption_audit.update(
                env_step,
                selected_target,
                stored_state,
                stored_action,
                stored_reward,
                stored_next_state,
            )
        replay_mismatch += int(
            not np.allclose(stored_action, executed_action, rtol=1e-6, atol=1e-6)
        )
        replay_state, replay_next_state = _replay_transition_coordinates(
            stored_state,
            stored_next_state,
            normalizer,
            normalized_replay_poisoning,
        )
        if is_calql:
            completed = calql_trajectory.append(
                observation=replay_state,
                action=stored_action,
                reward=stored_reward,
                next_observation=replay_next_state,
                terminal=bool(terminated),
                timeout=episode_timeout,
            )
            if completed is not None:
                replay.add_batch(
                    completed.batch["observations"],
                    completed.batch["actions"],
                    completed.batch["rewards"],
                    completed.batch["next_observations"],
                    completed.batch["terminals"],
                    mc_returns=completed.batch["mc_returns"],
                )
                perform_calql_trajectory_updates(completed)
        else:
            replay.add(
                replay_state,
                stored_action,
                stored_reward,
                replay_next_state,
                float(terminated),
                priority=initial_online_priority,
            )
        raw_state = raw_next_state
        if episode_finished:
            raw_state = None
            episode_steps = 0
            episode_return = 0.0
            episode_boundary = True

        if not official_pre_transition_updates:
            perform_online_updates(env_step, before_transition=False)
        if env_step % config.train_log_period == 0:
            parameter_deltas, parameter_snapshot = _parameter_deltas(
                agent, parameter_snapshot
            )
            wsrl_update_metrics: Dict[str, float] = {}
            if is_wsrl:
                count_window_env_steps = max(
                    env_step - last_count_log_env_step, 1
                )
                critic_update_delta = (
                    _agent_update_count(agent, "critic")
                    - last_logged_critic_updates
                )
                actor_update_delta = (
                    _agent_update_count(agent, "actor")
                    - last_logged_actor_updates
                )
                temperature_update_delta = (
                    _agent_update_count(agent, "temperature")
                    - last_logged_temperature_updates
                )
                wsrl_update_metrics = {
                    "wsrl_warmup_active": float(
                        env_step < wsrl_first_update_step
                    ),
                    "wsrl_first_update_env_step": float(
                        wsrl_first_update_step
                    ),
                    "wsrl_configured_critic_updates_per_update_step": float(
                        config.wsrl_utd_ratio
                    ),
                    "wsrl_configured_actor_updates_per_update_step": 1.0,
                    "wsrl_configured_temperature_updates_per_update_step": 1.0,
                    "wsrl_critic_updates_since_last_log": float(
                        critic_update_delta
                    ),
                    "wsrl_actor_updates_since_last_log": float(
                        actor_update_delta
                    ),
                    "wsrl_temperature_updates_since_last_log": float(
                        temperature_update_delta
                    ),
                    "wsrl_critic_updates_per_env_step": float(
                        critic_update_delta / count_window_env_steps
                    ),
                    "wsrl_actor_updates_per_env_step": float(
                        actor_update_delta / count_window_env_steps
                    ),
                    "wsrl_temperature_updates_per_env_step": float(
                        temperature_update_delta / count_window_env_steps
                    ),
                    "wsrl_total_actor_updates": float(
                        _agent_update_count(agent, "actor")
                    ),
                    "wsrl_total_critic_updates": float(
                        _agent_update_count(agent, "critic")
                    ),
                    "wsrl_total_temperature_updates": float(
                        _agent_update_count(agent, "temperature")
                    ),
                }
            logger.log_train(
                "online",
                env_step,
                env_step,
                agent.total_updates,
                {
                    **accumulator.means(),
                    **parameter_deltas,
                    **_policy_log_std_metrics(agent),
                    "replay_size": float(replay.size),
                    "replay_size_offline": float(offline.size),
                    "replay_size_online": float(replay.size),
                    "offline_batch_fraction": float(current_offline_ratio),
                    "online_batch_fraction": float(1.0 - current_offline_ratio),
                    "parameters_frozen_during_warmup": float(
                        config.algorithm == "wsrl"
                    ),
                    "offline_data_retained_online": float(
                        current_offline_ratio > 0.0
                    ),
                    **wsrl_update_metrics,
                    "online_corruption_fraction": corrupted_online / env_step,
                    "raw_action_abs_max": raw_action_abs_max,
                    "executed_action_abs_max": executed_action_abs_max,
                    "raw_action_oob_fraction": raw_oob / env_step,
                    "executed_action_oob_fraction": executed_oob / env_step,
                    "replay_env_action_mismatch_fraction": replay_mismatch / env_step,
                    **(
                        {
                            key: float(value)
                            for key, value in calql_trajectory.metadata().items()
                        }
                        if is_calql
                        else {}
                    ),
                    **(
                        {
                            "pqe_initial_online_priority": float(
                                initial_online_priority
                            ),
                            "pqe_member_count": float(config.pqe_ensemble_size),
                            **_pqe_block_schedule_metadata(
                                config,
                                pqe_block_update_counts,
                                env_step=env_step,
                                inferred_from_legacy=(
                                    pqe_schedule_inferred_from_legacy
                                ),
                            ),
                        }
                        if is_pqe
                        else {}
                    ),
                    **priority_metrics,
                },
            )
            if is_wsrl:
                last_count_log_env_step = env_step
                last_logged_actor_updates = _agent_update_count(
                    agent, "actor"
                )
                last_logged_critic_updates = _agent_update_count(
                    agent, "critic"
                )
                last_logged_temperature_updates = _agent_update_count(
                    agent, "temperature"
                )
        if env_step % config.eval_period == 0:
            _evaluate(
                logger,
                eval_env,
                config,
                agent,
                normalizer,
                device,
                "online",
                env_step,
                env_step,
            )
        period = config.effective_online_checkpoint_period
        if period > 0 and env_step % period == 0 and env_step != config.online_steps:
            pending_checkpoint = True
        if pending_checkpoint and episode_boundary and env_step != config.online_steps:
            _save_phase_checkpoint(
                logger,
                agent,
                config,
                normalizer,
                "online",
                env_step,
                env_step,
                state_dim,
                action_dim,
                final=False,
                resume_state=online_resume_snapshot(env_step),
            )
            pending_checkpoint = False

        if (
            rpex_episode_boundary_budget
            and episode_boundary
            and env_step > config.online_steps
        ):
            break
        if (
            calql_episode_boundary_budget
            and episode_boundary
            and env_step >= config.online_steps
        ):
            break

    if is_calql and calql_trajectory.pending_episode_length:
        raise RuntimeError(
            "Cal-QL online phase ended before its pending trajectory completed"
        )

    if (
        config.implementation_profile != "official_code_reference"
        and (
            actual_online_steps == 0
            or actual_online_steps % config.eval_period != 0
        )
    ):
        _evaluate(
            logger,
            eval_env,
            config,
            agent,
            normalizer,
            device,
            "online",
            actual_online_steps,
            actual_online_steps,
        )
    _save_phase_checkpoint(
        logger,
        agent,
        config,
        normalizer,
        "online",
        actual_online_steps,
        actual_online_steps,
        state_dim,
        action_dim,
        final=True,
        resume_state=(
            online_resume_snapshot(actual_online_steps) if episode_boundary else None
        ),
    )
    logger.logger.info(
        "online fine-tuning completed; corrupted=%d/%d",
        corrupted_online,
        actual_online_steps,
    )
    online_corruption_metadata = corruption_audit.metadata(config)
    online_corruption_metadata.update(
        {
            "online_budget_semantics": (
                "rpex_official_episode_boundary_strict_greater_than"
                if rpex_episode_boundary_budget
                else "calql_complete_current_episode_at_or_after_requested"
                if is_calql
                else "exact_environment_steps"
            ),
            "requested_online_steps": config.online_steps,
            "actual_online_steps": actual_online_steps,
            "episode_boundary_overshoot": max(
                actual_online_steps - config.online_steps, 0
            ),
        }
    )
    online_corruption_metadata.update(
        _wsrl_runtime_metadata(
            config,
            agent,
            online_initial_update_counts=online_initial_update_counts,
        )
    )
    online_corruption_metadata.update(
        _runtime_update_metadata(
            config,
            agent,
            online_initial_update_counts=online_initial_update_counts,
            online_environment_steps=actual_online_steps,
        )
    )
    online_corruption_metadata.update(
        {
            "raw_action_oob_fraction": (
                float(raw_oob / actual_online_steps)
                if actual_online_steps > 0
                else 0.0
            ),
            "raw_action_abs_max": raw_action_abs_max,
            "executed_action_abs_max": executed_action_abs_max,
            "executed_action_oob_fraction": (
                float(executed_oob / actual_online_steps)
                if actual_online_steps > 0
                else 0.0
            ),
        }
    )
    if is_calql:
        online_corruption_metadata.update(calql_trajectory.metadata())
        online_corruption_metadata["effective_calql_training_transitions"] = int(
            calql_trajectory.completed_transitions
        )
        online_corruption_metadata["online_calibration_bound_rate"] = float(
            last_metrics.get("online_calibration_bound_rate", 0.0)
        )
        online_corruption_metadata["calql_online_cql_enabled"] = True
        online_corruption_metadata["calql_dynamic_offline_ratio"] = float(
            current_offline_ratio
        )
    if is_pqe:
        online_corruption_metadata.update(agent.algorithm_metadata())
        online_corruption_metadata.update(
            {
                "pqe_initial_online_priority": float(initial_online_priority),
                "pqe_first_block_updates_applied": bool(
                    pqe_first_block_updates_applied
                ),
                "pqe_first_block_update_count": int(
                    pqe_first_block_update_count
                ),
                **_pqe_block_schedule_metadata(
                    config,
                    pqe_block_update_counts,
                    env_step=actual_online_steps,
                    inferred_from_legacy=pqe_schedule_inferred_from_legacy,
                ),
                "priority_replay_offline": offline.priority_statistics(),
                "priority_replay_online": replay.priority_statistics(),
            }
        )
    with (logger.run_dir / "online_corruption_manifest.json").open(
        "w", encoding="utf-8"
    ) as stream:
        json.dump(online_corruption_metadata, stream, indent=2, ensure_ascii=False)
    logger.write_completion_manifest(online_corruption_metadata)
