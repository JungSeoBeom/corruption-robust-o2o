from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal

from .config import (
    INDIVIDUAL_CORRUPTION_TARGETS,
    ExperimentConfig,
    default_attack_checkpoint,
)
from .device import clear_accelerator_cache
from .environment import Dataset
from .networks import VectorizedLinear


class EDACActor(nn.Module):
    """Architecture-compatible loader for the supplied RPEX/RIQL attack oracle."""

    def __init__(self, state_dim: int, action_dim: int, max_action: float):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        self.mu = nn.Linear(256, action_dim)
        self.log_sigma = nn.Linear(256, action_dim)
        self.max_action = max_action

    def forward(self, states: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        hidden = self.trunk(states)
        mean = self.mu(hidden)
        if deterministic:
            raw = mean
        else:
            std = self.log_sigma(hidden).clamp(-5, 2).exp()
            raw = Normal(mean, std).rsample()
        return torch.tanh(raw) * self.max_action


class EDACCritic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, num_critics: int = 10):
        super().__init__()
        self.critic = nn.Sequential(
            VectorizedLinear(state_dim + action_dim, 256, num_critics),
            nn.ReLU(),
            VectorizedLinear(256, 256, num_critics),
            nn.ReLU(),
            VectorizedLinear(256, 256, num_critics),
            nn.ReLU(),
            VectorizedLinear(256, 1, num_critics),
        )
        self.num_critics = num_critics

    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        inputs = torch.cat((states, actions), dim=-1)
        inputs = inputs.unsqueeze(0).expand(self.num_critics, -1, -1)
        return self.critic(inputs).squeeze(-1)


class AttackOracle:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        max_action: float,
        checkpoint: Path,
        device: torch.device,
    ):
        if not checkpoint.exists():
            raise FileNotFoundError(f"Adversarial attack checkpoint not found: {checkpoint}")
        self.device = device
        self.actor = EDACActor(state_dim, action_dim, max_action).to(device).eval()
        self.critic = EDACCritic(state_dim, action_dim).to(device).eval()
        state = torch.load(checkpoint, map_location=device)
        self.actor.load_state_dict(state["actor"])
        self.critic.load_state_dict(state["critic"])
        for parameter in self.actor.parameters():
            parameter.requires_grad_(False)
        for parameter in self.critic.parameters():
            parameter.requires_grad_(False)

    def attack(
        self,
        original: np.ndarray,
        std: np.ndarray,
        observations: np.ndarray,
        actions: np.ndarray,
        target: str,
        scale: float,
        steps: int,
        step_size: float,
    ) -> np.ndarray:
        original_tensor = torch.as_tensor(original, dtype=torch.float32, device=self.device)
        observations_tensor = torch.as_tensor(
            observations, dtype=torch.float32, device=self.device
        )
        actions_tensor = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
        std_tensor = torch.as_tensor(std, dtype=torch.float32, device=self.device)
        noise = (
            torch.empty_like(original_tensor).uniform_(-scale, scale) * std_tensor
        )
        for _ in range(steps):
            noise.requires_grad_(True)
            attacked = original_tensor + noise
            if target == "observations":
                loss = self.critic(attacked, actions_tensor).mean()
            elif target == "actions":
                loss = self.critic(observations_tensor, attacked).mean()
            elif target == "dynamics":
                attacked_actions = self.actor(attacked, deterministic=True)
                loss = self.critic(attacked, attacked_actions).mean()
            else:
                raise ValueError(f"Gradient attack is unsupported for {target}")
            gradient = torch.autograd.grad(loss, noise)[0]
            noise = (noise - step_size * scale * gradient.sign()).detach()
            bound = scale * std_tensor
            noise = torch.maximum(torch.minimum(noise, bound), -bound)
        return (original_tensor + noise).detach().cpu().numpy().astype(np.float32)

    def close(self) -> None:
        self.actor.to("cpu")
        self.critic.to("cpu")
        clear_accelerator_cache(self.device)


def resolve_attack_checkpoint(config: ExperimentConfig) -> Path:
    if config.attack_checkpoint:
        return Path(config.attack_checkpoint).expanduser().resolve()
    candidate = default_attack_checkpoint(config.env_name)
    if candidate is None:
        raise FileNotFoundError(
            "Adversarial corruption needs an EDAC checkpoint. Pass "
            "--attack-checkpoint. The supplied checkpoints cover the three "
            "*-medium-replay-v2 environments."
        )
    return candidate


def make_attack_oracle(
    config: ExperimentConfig,
    state_dim: int,
    action_dim: int,
    max_action: float,
    device: torch.device,
) -> Optional[AttackOracle]:
    if config.corruption != "adversarial":
        return None
    if config.corruption_target == "rewards":
        return None
    if config.corruption_target == "mixed":
        gradient_ratios = (
            ratio
            for target, ratio in zip(
                INDIVIDUAL_CORRUPTION_TARGETS, config.mixed_ratios
            )
            if target != "rewards"
        )
        if not any(ratio > 0.0 for ratio in gradient_ratios):
            return None
    if config.corruption_target == "none":
        return None
    return AttackOracle(
        state_dim,
        action_dim,
        max_action,
        resolve_attack_checkpoint(config),
        device,
    )


def _cache_file(config: ExperimentConfig, cache_root: Path) -> Path:
    mixed_suffix = ""
    if config.corruption_target == "mixed":
        ratios = "-".join(f"{ratio:g}" for ratio in config.mixed_ratios)
        mixed_suffix = f"_mix{ratios}"
    name = (
        f"{config.corruption}_{config.corruption_target}"
        f"{mixed_suffix}"
        f"_range{config.corruption_range:g}"
        f"_rate{config.offline_corruption_rate:g}"
        f"_seed{config.seed}.npz"
    )
    return cache_root / config.env_name / name


def _target_dataset_key(target: str) -> str:
    return {
        "observations": "observations",
        "actions": "actions",
        "rewards": "rewards",
        "dynamics": "next_observations",
    }[target]


def _corruption_stats(
    dataset_size: int,
    target_indices: Dict[str, np.ndarray],
    config: ExperimentConfig,
    loaded_from_cache: bool,
) -> Dict[str, float]:
    total = sum(len(indices) for indices in target_indices.values())
    stats = {
        "corrupted_count": int(total),
        "corrupted_fraction": float(total / dataset_size),
        "loaded_from_cache": float(loaded_from_cache),
    }
    for target, ratio in zip(
        INDIVIDUAL_CORRUPTION_TARGETS, config.mixed_ratios
    ):
        count = len(target_indices.get(target, ()))
        stats[f"{target}_corrupted_count"] = int(count)
        stats[f"{target}_corrupted_fraction"] = float(count / dataset_size)
        if config.corruption_target == "mixed":
            stats[f"{target}_allocation_ratio"] = float(ratio)
    return stats


def _load_cached_corruption(
    dataset: Dataset,
    config: ExperimentConfig,
    cache_file: Path,
) -> Tuple[Dataset, Dict[str, float]]:
    result = {key: value.copy() for key, value in dataset.items()}
    target_indices: Dict[str, np.ndarray] = {}
    with np.load(cache_file, allow_pickle=False) as cached:
        if "format_version" not in cached.files:
            # Compatibility with single-target cache files created by the
            # original unified implementation.
            indices = cached["indices"]
            dataset_key = cached["dataset_key"].item()
            result[dataset_key][indices] = cached["values"]
            target_indices[config.corruption_target] = indices
        else:
            targets = [str(value) for value in cached["targets"].tolist()]
            for target in targets:
                indices = cached[f"indices_{target}"]
                result[_target_dataset_key(target)][indices] = cached[
                    f"values_{target}"
                ]
                target_indices[target] = indices
    return result, _corruption_stats(
        len(dataset["rewards"]), target_indices, config, loaded_from_cache=True
    )


def _corrupt_target_values(
    dataset: Dataset,
    target: str,
    indices: np.ndarray,
    config: ExperimentConfig,
    oracle: Optional[AttackOracle],
    rng: np.random.Generator,
) -> np.ndarray:
    key = _target_dataset_key(target)
    original = dataset[key][indices].copy()
    if len(indices) == 0:
        return original

    if target == "rewards":
        if config.corruption == "random":
            return (
                rng.uniform(-1.0, 1.0, size=original.shape).astype(np.float32)
                * 30.0
            )
        return (-config.corruption_range * original).astype(np.float32)

    std = dataset[key].std(axis=0, keepdims=True).astype(np.float32)
    if config.corruption == "random":
        noise = rng.uniform(
            -config.corruption_range,
            config.corruption_range,
            size=original.shape,
        ).astype(np.float32)
        return (original + noise * std).astype(np.float32)

    if oracle is None:
        raise RuntimeError("An AttackOracle is required for adversarial corruption")
    observations = dataset["observations"][indices]
    actions = dataset["actions"][indices]
    chunks = []
    # Bound attack memory on CPU/MPS as well as CUDA.
    for start in range(0, len(indices), 16_384):
        stop = min(start + 16_384, len(indices))
        chunks.append(
            oracle.attack(
                original[start:stop],
                std,
                observations[start:stop],
                actions[start:stop],
                target,
                config.corruption_range,
                config.offline_attack_steps,
                config.attack_step_size,
            )
        )
    return np.concatenate(chunks, axis=0) if chunks else original


def corrupt_offline_dataset(
    dataset: Dataset,
    config: ExperimentConfig,
    oracle: Optional[AttackOracle],
    cache_root: Path,
) -> Tuple[Dataset, Dict[str, float]]:
    if config.corruption == "clean":
        return {key: value.copy() for key, value in dataset.items()}, {
            "corrupted_count": 0,
            "corrupted_fraction": 0.0,
            "loaded_from_cache": 0.0,
        }

    cache_file = _cache_file(config, cache_root)
    if cache_file.exists() and not config.force_regenerate_attack:
        return _load_cached_corruption(dataset, config, cache_file)

    rng = np.random.default_rng(config.seed)
    selected = rng.random(len(dataset["rewards"])) < config.offline_corruption_rate
    indices = np.flatnonzero(selected)
    result = {key: value.copy() for key, value in dataset.items()}
    if config.corruption_target == "mixed":
        assignments = rng.choice(
            len(INDIVIDUAL_CORRUPTION_TARGETS),
            size=len(indices),
            p=np.asarray(config.mixed_ratios, dtype=np.float64),
        )
        target_indices = {
            target: indices[assignments == target_index]
            for target_index, target in enumerate(INDIVIDUAL_CORRUPTION_TARGETS)
        }
    else:
        target_indices = {config.corruption_target: indices}

    cache_payload = {
        "format_version": np.asarray(2, dtype=np.int64),
        "targets": np.asarray(list(target_indices), dtype=np.str_),
    }
    for target, target_rows in target_indices.items():
        values = _corrupt_target_values(
            dataset, target, target_rows, config, oracle, rng
        )
        result[_target_dataset_key(target)][target_rows] = values
        cache_payload[f"indices_{target}"] = target_rows
        cache_payload[f"values_{target}"] = values

    cache_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_file, **cache_payload)
    return result, _corruption_stats(
        len(dataset["rewards"]), target_indices, config, loaded_from_cache=False
    )


def corrupt_online_transition(
    raw_state: np.ndarray,
    action: np.ndarray,
    reward: float,
    raw_next_state: np.ndarray,
    config: ExperimentConfig,
    oracle: Optional[AttackOracle],
    rng: np.random.Generator,
    state_std: np.ndarray,
    action_std: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray, bool]:
    if config.corruption == "clean" or rng.random() >= config.online_corruption_rate:
        return raw_state, action, reward, raw_next_state, False

    state = raw_state.copy()
    stored_action = action.copy()
    stored_reward = float(reward)
    next_state = raw_next_state.copy()
    if config.corruption_target == "mixed":
        target = str(
            rng.choice(
                INDIVIDUAL_CORRUPTION_TARGETS,
                p=np.asarray(config.mixed_ratios, dtype=np.float64),
            )
        )
    else:
        target = config.corruption_target
    if target == "rewards":
        if config.corruption == "random":
            stored_reward = float(rng.uniform(-1.0, 1.0) * 30.0)
        else:
            stored_reward = float(-config.corruption_range * reward)
        return state, stored_action, stored_reward, next_state, True

    original = {
        "observations": state,
        "actions": stored_action,
        "dynamics": next_state,
    }[target]
    std = action_std if target == "actions" else state_std
    if config.corruption == "random":
        attacked = original + rng.uniform(
            -config.corruption_range,
            config.corruption_range,
            size=original.shape,
        ) * std
    else:
        if oracle is None:
            raise RuntimeError("An AttackOracle is required for adversarial corruption")
        attacked = oracle.attack(
            original[None, :],
            std[None, :],
            state[None, :],
            stored_action[None, :],
            target,
            config.corruption_range,
            config.online_attack_steps,
            max(config.attack_step_size, 0.1),
        )[0]
    if target == "observations":
        state = attacked.astype(np.float32)
    elif target == "actions":
        stored_action = attacked.astype(np.float32)
    else:
        next_state = attacked.astype(np.float32)
    return state, stored_action, stored_reward, next_state, True
