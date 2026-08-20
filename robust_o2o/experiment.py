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
from .corruption import (
    AttackOracle,
    OnlineCorruptionAudit,
    corrupt_pre_action_value,
    corrupt_offline_dataset,
    corrupt_online_transition,
    make_attack_oracle,
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
from .logging_utils import RunLogger
from .paths import results_root_from_output
from .replay import (
    OfflineDataset,
    ReplayBuffer,
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


def bounded_executed_action(
    raw_policy_action: np.ndarray,
    action_low: np.ndarray,
    action_high: np.ndarray,
) -> np.ndarray:
    return np.clip(raw_policy_action, action_low, action_high).astype(np.float32)


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
    critic_modules = []
    for name in ("critic", "critic2", "q1", "q2", "value"):
        module = getattr(agent, name, None)
        if module is not None:
            critic_modules.append(module)
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
    candidates = [path] if path.is_file() else list(path.glob("checkpoints/*/*.pt"))
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
    return candidate, payload


def capture_global_rng_state(
    corruption_rng: Optional[np.random.Generator] = None,
    oracle: Optional[AttackOracle] = None,
) -> Dict[str, Any]:
    state: Dict[str, Any] = {
        "python": random.getstate(),
        "numpy_global": np.random.get_state(),
        "torch_cpu": torch.random.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "corruption_rng": (
            corruption_rng.bit_generator.state if corruption_rng is not None else None
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
    corruption_rng: Optional[np.random.Generator] = None,
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
        corruption_rng.bit_generator.state = state["corruption_rng"]
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


def _prune_periodic_checkpoints(directory: Path, keep_last: int) -> None:
    """Retain only the newest periodic checkpoints; zero means keep all."""
    if keep_last == 0:
        return
    checkpoints = sorted(directory.glob("step_*.pt"))
    for checkpoint in checkpoints[:-keep_last]:
        checkpoint.unlink()


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
        "wsrl_num_critics",
        "wsrl_target_critic_subsample_size",
        "wsrl_layer_norm",
        "wsrl_utd_ratio",
        "wsrl_per_critic_batch_size",
        "mc_return_source",
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
        ("deterministic_diagnostic", "method_faithful")
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
        "deterministic_diagnostic"
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


def run_experiment(config: ExperimentConfig, logger: RunLogger) -> Path:
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
            resume_payload = checkpoint_payload["resume_state"]
        if config.initialize_from_checkpoint:
            checkpoint_path = Path(
                config.initialize_from_checkpoint
            ).expanduser().resolve()
            checkpoint_payload = _torch_load(checkpoint_path, device)
            _validate_checkpoint(checkpoint_payload, config, state_dim, action_dim)
            _restore_agent_config(config, checkpoint_payload)

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
            )
        normalized_dataset = apply_normalizer(corrupted_dataset, normalizer)
        offline = OfflineDataset(normalized_dataset, config.replay_seed)
        if resume_payload is not None and resume_payload.get("offline_dataset"):
            offline.load_state_dict(resume_payload["offline_dataset"])

        # Decouple learner initialization and subsequent training randomness
        # from preprocessing, adversarial attacks, and cache hit/miss state.
        seed_everything(config.learner_seed)
        agent = build_agent(config, state_dim, action_dim, max_action, device)
        if checkpoint_payload is not None:
            agent.load_checkpoint_state(checkpoint_payload["agent"])
        if resume_payload is not None:
            restore_global_rng_state(resume_payload["global_rng"], oracle=oracle)

        diagnostic_snapshot = (
            _parameter_snapshot(agent) if config.diagnostic_mode else None
        )
        diagnostic_initial_updates = agent.total_updates
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
                "legacy_checkpoint_without_fingerprint_loaded": bool(
                    getattr(
                        config,
                        "_legacy_checkpoint_without_fingerprint_loaded",
                        False,
                    )
                ),
            }
        )
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
            logger.logger.warning(
                "Pessimistic Q-Ensemble uses implementation_variant=%s: "
                "one shared actor with critic ensembles, not the exact official "
                "Off2OnRL independent actor/critic ensemble.",
                config.implementation_variant,
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
                "completed_actor_updates": agent.total_updates
                - diagnostic_initial_updates,
                "completed_critic_updates": agent.total_updates
                - diagnostic_initial_updates,
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
    last_metrics: Dict[str, float] = {}
    accumulator = MetricAccumulator()
    start_step = 0
    if resume_state is not None and resume_state.get("phase") == "offline":
        _validate_writer_positions(logger, resume_state)
        start_step = int(resume_state["phase_step"])
        accumulator.values = {
            key: list(values)
            for key, values in resume_state.get("metric_accumulator", {}).items()
        }
    parameter_snapshot = _parameter_snapshot(agent)
    for step in range(start_step + 1, config.offline_steps + 1):
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
            and step != config.offline_steps
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
                    "global_rng": capture_global_rng_state(),
                    "metric_accumulator": accumulator.values,
                    "writer_append_position": _writer_positions(logger),
                },
            )
    if config.offline_steps == 0 or config.offline_steps % config.eval_period != 0:
        _evaluate(
            logger,
            env,
            config,
            agent,
            normalizer,
            device,
            "offline",
            config.offline_steps,
            0,
        )
    _save_phase_checkpoint(
        logger,
        agent,
        config,
        normalizer,
        "offline",
        config.offline_steps,
        0,
        state_dim,
        action_dim,
        final=True,
        resume_state={
            "phase": "offline",
            "phase_step": config.offline_steps,
            "episode_boundary": True,
            "offline_dataset": offline.state_dict(),
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
    replay = ReplayBuffer(
        state_dim, action_dim, config.replay_size, config.replay_seed
    )
    rng = np.random.default_rng(config.corruption_seed)
    state_std = raw_dataset["observations"].std(axis=0).astype(np.float32) + 1e-6
    action_std = raw_dataset["actions"].std(axis=0).astype(np.float32) + 1e-6
    online_resume = resume_state is not None and resume_state.get("phase") == "online"
    start_env_step = int(resume_state["phase_step"]) if online_resume else 0
    if online_resume:
        _validate_writer_positions(logger, resume_state)
        if not resume_state.get("episode_boundary"):
            raise ValueError("exact online resume is only supported at episode boundaries")
        replay.load_state_dict(resume_state["online_replay"])
        restore_global_rng_state(resume_state["global_rng"], rng, oracle)
        restore_environment_rng_state(env, resume_state["environment_rng"])
        raw_state = None
    else:
        raw_state = reset_env(env, seed=config.train_env_seed, protocol=config.protocol)
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
    warmup = (
        max(config.initial_collection_steps, config.warmup_steps)
        if config.algorithm == "wsrl"
        else config.initial_collection_steps
    )
    action_low = np.asarray(env.action_space.low, dtype=np.float32)
    action_high = np.asarray(env.action_space.high, dtype=np.float32)
    raw_action_abs_max = float(resume_state.get("raw_action_abs_max", 0.0)) if online_resume else 0.0
    executed_action_abs_max = float(resume_state.get("executed_action_abs_max", 0.0)) if online_resume else 0.0
    executed_oob = int(resume_state.get("executed_oob", 0)) if online_resume else 0
    replay_mismatch = int(resume_state.get("replay_mismatch", 0)) if online_resume else 0
    priority_metrics: Dict[str, float] = {}
    desired_online_fraction = 1.0 - offline_ratio
    initial_online_priority = (
        offline.size
        * desired_online_fraction
        / max(offline_ratio * max(warmup, 1), 1e-12)
        if config.algorithm == "pessimistic_q_ensemble"
        and config.pqe_replay_mode == "balanced_density"
        and offline_ratio > 0.0
        else 1.0
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
            "executed_oob": executed_oob,
            "replay_mismatch": replay_mismatch,
            "metric_accumulator": accumulator.values,
            "writer_append_position": _writer_positions(logger),
        }

    pending_checkpoint = False
    episode_boundary = True
    for env_step in range(start_env_step + 1, config.online_steps + 1):
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
        executed_action_abs_max = max(
            executed_action_abs_max, float(np.abs(executed_action).max())
        )
        executed_oob += int(
            np.any(executed_action < action_low) or np.any(executed_action > action_high)
        )
        raw_next_state, reward, terminated, truncated, _ = step_env(
            env, executed_action, protocol=config.protocol
        )
        episode_steps += 1
        episode_return += reward

        if not pre_action:
            selected_target = sample_online_corruption_target(config, rng)

        if pre_action and selected_target in ("observations", "actions"):
            stored_state = policy_state.copy()
            stored_action = executed_action.copy()
            stored_reward = float(reward)
            stored_next_state = raw_next_state.copy()
            was_corrupted = True
        else:
            (
                stored_state,
                stored_action,
                stored_reward,
                stored_next_state,
                was_corrupted,
            ) = corrupt_online_transition(
                raw_state,
                executed_action,
                reward,
                raw_next_state,
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
        replay.add(
            normalizer.transform(stored_state),
            stored_action,
            stored_reward,
            normalizer.transform(stored_next_state),
            float(terminated),
            priority=initial_online_priority,
        )
        raw_state = raw_next_state
        if terminated or truncated or episode_steps >= config.max_episode_steps:
            raw_state = None
            episode_steps = 0
            episode_return = 0.0
            episode_boundary = True

        is_pqe = config.algorithm == "pessimistic_q_ensemble"
        update_batch_size = (
            config.wsrl_per_critic_batch_size
            if config.algorithm == "wsrl"
            else config.batch_size
        )
        required_online_samples = (
            config.batch_size if is_pqe else max(
                update_batch_size - int(round(update_batch_size * offline_ratio)),
                1,
            )
        )
        can_update = env_step > warmup and replay.size >= required_online_samples
        if can_update:
            update_repeats = (
                config.wsrl_utd_ratio
                if config.algorithm == "wsrl"
                else config.updates_per_step
            )
            wsrl_batches: list[Dict[str, torch.Tensor]] = []
            for _ in range(update_repeats):
                prioritized = (
                    is_pqe
                    and config.pqe_replay_mode == "balanced_density"
                )
                if is_pqe:
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
                        prioritized_rl=prioritized,
                    )
                    last_metrics = agent.update(
                        rl_batch=batch,
                        density_offline_batch=density_offline_batch,
                        density_online_batch=density_online_batch,
                        rl_batch_prioritized=prioritized,
                    )
                else:
                    batch = mixed_batch(
                        offline,
                        replay,
                        update_batch_size,
                        offline_ratio,
                        device,
                    )
                    if config.algorithm == "wsrl":
                        wsrl_batches.append(batch)
                        last_metrics = agent.update(
                            batch,
                            update_actor_temperature=False,
                            update_critic=True,
                        )
                    else:
                        last_metrics = agent.update(batch)
                accumulator.add(last_metrics)
                if is_pqe:
                    priorities = agent.consume_priority_values()
                    if priorities is None:
                        raise RuntimeError(
                            "Pessimistic Q-Ensemble skipped a required priority update"
                        )
                    priority_metrics = update_sample_priorities(
                        offline, replay, batch, priorities
                    )
            if config.algorithm == "wsrl":
                actor_batch = {
                    key: torch.cat([batch[key] for batch in wsrl_batches], dim=0)
                    for key in wsrl_batches[0]
                }
                last_metrics = agent.update(
                    actor_batch,
                    update_actor_temperature=True,
                    update_critic=False,
                )
                accumulator.add(last_metrics)
        if env_step % config.train_log_period == 0:
            parameter_deltas, parameter_snapshot = _parameter_deltas(
                agent, parameter_snapshot
            )
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
                    "offline_batch_fraction": float(offline_ratio),
                    "online_batch_fraction": float(1.0 - offline_ratio),
                    "parameters_frozen_during_warmup": float(
                        config.algorithm == "wsrl"
                    ),
                    "offline_data_retained_online": float(offline_ratio > 0.0),
                    "wsrl_critic_updates_per_env_step": float(
                        config.wsrl_utd_ratio if config.algorithm == "wsrl" else 0
                    ),
                    "wsrl_actor_updates_per_env_step": float(
                        1 if config.algorithm == "wsrl" else 0
                    ),
                    "online_corruption_fraction": corrupted_online / env_step,
                    "raw_action_abs_max": raw_action_abs_max,
                    "executed_action_abs_max": executed_action_abs_max,
                    "executed_action_oob_fraction": executed_oob / env_step,
                    "replay_env_action_mismatch_fraction": replay_mismatch / env_step,
                    **priority_metrics,
                },
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

    if config.online_steps == 0 or config.online_steps % config.eval_period != 0:
        _evaluate(
            logger,
            eval_env,
            config,
            agent,
            normalizer,
            device,
            "online",
            config.online_steps,
            config.online_steps,
        )
    _save_phase_checkpoint(
        logger,
        agent,
        config,
        normalizer,
        "online",
        config.online_steps,
        config.online_steps,
        state_dim,
        action_dim,
        final=True,
        resume_state=(
            online_resume_snapshot(config.online_steps) if episode_boundary else None
        ),
    )
    logger.logger.info(
        "online fine-tuning completed; corrupted=%d/%d",
        corrupted_online,
        config.online_steps,
    )
    online_corruption_metadata = corruption_audit.metadata(config)
    with (logger.run_dir / "online_corruption_manifest.json").open(
        "w", encoding="utf-8"
    ) as stream:
        json.dump(online_corruption_metadata, stream, indent=2, ensure_ascii=False)
