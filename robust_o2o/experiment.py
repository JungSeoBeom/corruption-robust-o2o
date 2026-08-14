from __future__ import annotations

from contextlib import ExitStack
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

from .agents import build_agent
from .config import ExperimentConfig
from .corruption import (
    AttackOracle,
    corrupt_offline_dataset,
    corrupt_online_transition,
    make_attack_oracle,
)
from .device import resolve_device, seed_everything
from .environment import (
    StateNormalizer,
    apply_normalizer,
    environment_metadata,
    evaluate_agent,
    expected_env_spec_id,
    load_d4rl_dataset,
    make_env,
    reset_env,
    step_env,
)
from .logging_utils import RunLogger
from .paths import results_root_from_output
from .replay import OfflineDataset, ReplayBuffer, mixed_batch


def _torch_load(path: Path, device: torch.device) -> Dict[str, Any]:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


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
) -> None:
    payload = {
        "format_version": 1,
        "algorithm": config.algorithm,
        "env_name": config.env_name,
        "protocol": config.protocol,
        "config": config.to_dict(),
        "normalizer": normalizer.state_dict(),
        "agent": agent.checkpoint_state(),
        "phase": phase,
        "step": step,
        "env_steps": env_steps,
        "state_dim": state_dim,
        "action_dim": action_dim,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _checkpoint_directory(logger: RunLogger, phase: str) -> Path:
    directory = logger.run_dir / "checkpoints" / phase
    directory.mkdir(parents=True, exist_ok=True)
    return directory


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
) -> Path:
    directory = _checkpoint_directory(logger, phase)
    path = directory / ("final.pt" if final else f"step_{step:09d}.pt")
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
    if payload.get("state_dim") != state_dim or payload.get("action_dim") != action_dim:
        raise ValueError("Checkpoint observation/action dimensions do not match the env")


def _restore_agent_config(config: ExperimentConfig, payload: Dict[str, Any]) -> None:
    """Restore architecture/objective fields while keeping new run controls."""
    saved = payload.get("config", {})
    fields = (
        "hidden_dim",
        "hidden_layers",
        "learning_rate",
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
        "bc_steps",
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
    metrics = evaluate_agent(
        env,
        config.env_name,
        agent,
        normalizer,
        device,
        config.eval_episodes,
        config.max_episode_steps,
        config.seed,
        config.protocol,
    )
    logger.log_evaluation(
        phase, step, env_steps, agent.total_updates, metrics
    )


def run_experiment(config: ExperimentConfig, logger: RunLogger) -> Path:
    device = resolve_device(config.device, config.cuda_device)
    dataset_env = make_env(config.env_name, config.protocol)
    try:
        seed_everything(config.seed, dataset_env)
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
            config.seed,
            config.protocol,
            config.dataset_dir,
        )
    finally:
        dataset_env.close()

    state_dim = raw_dataset["observations"].shape[1]
    action_dim = raw_dataset["actions"].shape[1]

    with ExitStack() as stack:
        env = make_env(config.env_name, config.protocol)
        stack.callback(env.close)
        eval_env = make_env(config.env_name, config.protocol)
        stack.callback(eval_env.close)
        seed_everything(config.seed, env)
        seed_everything(config.seed + 10_000, eval_env)
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
        max_action = float(np.max(np.abs(env.action_space.high)))

        checkpoint_payload: Optional[Dict[str, Any]] = None
        if config.checkpoint:
            checkpoint_path = Path(config.checkpoint).expanduser().resolve()
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

        if checkpoint_payload is not None and config.stage == "online":
            normalizer = StateNormalizer.from_state_dict(
                checkpoint_payload["normalizer"]
            )
        else:
            normalizer = StateNormalizer.fit(
                corrupted_dataset, enabled=config.normalize_states
            )
        normalized_dataset = apply_normalizer(corrupted_dataset, normalizer)
        offline = OfflineDataset(normalized_dataset, config.seed)

        agent = build_agent(config, state_dim, action_dim, max_action, device)
        if checkpoint_payload is not None:
            agent.load_checkpoint_state(checkpoint_payload["agent"])

        logger.write_config(
            {
                **config.to_dict(),
                **protocol_metadata,
                "resolved_device": str(device),
                "run_dir": str(logger.run_dir),
                "state_dim": state_dim,
                "action_dim": action_dim,
                "offline_corruption": corruption_stats,
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
        logger.logger.info(
            "dataset=%d offline_steps=%d online_steps=%d offline_ratio=%.2f",
            offline.size,
            config.offline_steps,
            config.online_steps,
            config.effective_offline_ratio,
        )

        if config.stage in ("offline", "both"):
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
            )
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
) -> None:
    logger.logger.info("offline pre-training started")
    last_metrics: Dict[str, float] = {}
    for step in range(1, config.offline_steps + 1):
        batch = offline.sample(config.batch_size, device)
        last_metrics = agent.update(batch)
        if step % config.train_log_period == 0:
            logger.log_train("offline", step, 0, agent.total_updates, last_metrics)
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
) -> None:
    logger.logger.info("online fine-tuning started")
    replay = ReplayBuffer(
        state_dim, action_dim, config.replay_size, config.seed + 1
    )
    rng = np.random.default_rng(config.seed + 2)
    state_std = raw_dataset["observations"].std(axis=0).astype(np.float32) + 1e-6
    action_std = raw_dataset["actions"].std(axis=0).astype(np.float32) + 1e-6
    raw_state = reset_env(env, seed=config.seed, protocol=config.protocol)
    episode_steps = 0
    corrupted_online = 0
    last_metrics: Dict[str, float] = {}
    offline_ratio = config.effective_offline_ratio
    online_batch_count = config.batch_size - int(
        round(config.batch_size * offline_ratio)
    )
    warmup = (
        max(config.initial_collection_steps, config.warmup_steps)
        if config.algorithm == "wsrl"
        else config.initial_collection_steps
    )

    for env_step in range(1, config.online_steps + 1):
        state_tensor = torch.as_tensor(
            normalizer.transform(raw_state), dtype=torch.float32, device=device
        )
        with torch.no_grad():
            action = agent.select_action(state_tensor, evaluate=False)
        action_np = action.detach().cpu().numpy().astype(np.float32)
        raw_next_state, reward, terminated, truncated, _ = step_env(
            env, action_np, protocol=config.protocol
        )
        episode_steps += 1

        (
            stored_state,
            stored_action,
            stored_reward,
            stored_next_state,
            was_corrupted,
        ) = corrupt_online_transition(
            raw_state,
            action_np,
            reward,
            raw_next_state,
            config,
            oracle,
            rng,
            state_std,
            action_std,
        )
        corrupted_online += int(was_corrupted)
        replay.add(
            normalizer.transform(stored_state),
            stored_action,
            stored_reward,
            normalizer.transform(stored_next_state),
            float(terminated),
        )
        raw_state = raw_next_state
        if terminated or truncated or episode_steps >= config.max_episode_steps:
            raw_state = reset_env(env, protocol=config.protocol)
            episode_steps = 0

        can_update = env_step > warmup and replay.size >= max(online_batch_count, 1)
        if can_update:
            for _ in range(config.updates_per_step):
                batch = mixed_batch(
                    offline,
                    replay,
                    config.batch_size,
                    offline_ratio,
                    device,
                    prioritized_online=(
                        config.algorithm == "pessimistic_q_ensemble"
                    ),
                )
                last_metrics = agent.update(batch)
        if env_step % config.train_log_period == 0:
            logger.log_train(
                "online",
                env_step,
                env_step,
                agent.total_updates,
                {
                    **last_metrics,
                    "replay_size": float(replay.size),
                    "online_corruption_fraction": corrupted_online / env_step,
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
        if (
            period > 0
            and env_step % period == 0
            and env_step != config.online_steps
        ):
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
            )

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
    )
    logger.logger.info(
        "online fine-tuning completed; corrupted=%d/%d",
        corrupted_online,
        config.online_steps,
    )
