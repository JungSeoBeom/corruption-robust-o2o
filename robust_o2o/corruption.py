from __future__ import annotations

import hashlib
import json
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

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


ATTACK_IMPLEMENTATION_VERSION = "corruption_v4_profiled_rng_timing_atomic_cache"
ATTACK_OBJECTIVE = "minimize_edac_ensemble_mean_q"


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
        seed: int = 0,
        implementation_profile: str = "experimental_sign_pgd",
    ):
        if not checkpoint.exists():
            raise FileNotFoundError(f"Adversarial attack checkpoint not found: {checkpoint}")
        self.device = device
        self.checkpoint = checkpoint.resolve()
        self.implementation_profile = implementation_profile
        self.generator = torch.Generator(device="cpu")
        self.generator.manual_seed(int(seed))
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
        initial = torch.empty(original_tensor.shape, dtype=torch.float32, device="cpu")
        initial.uniform_(-scale, scale, generator=self.generator)
        if self.implementation_profile == "rpex_official_adam":
            # Upstream Attack.sample_para includes std here and optimize_para
            # multiplies by std again when constructing the attacked input.
            para = initial.to(self.device) * std_tensor
            for _ in range(steps):
                para = torch.nn.Parameter(para.detach().clone(), requires_grad=True)
                optimizer = torch.optim.Adam([para], lr=step_size * scale)
                attacked = original_tensor + para * std_tensor
                if target == "observations":
                    loss = self.critic(attacked, actions_tensor).mean()
                elif target == "actions":
                    loss = self.critic(observations_tensor, attacked).mean()
                elif target == "dynamics":
                    attacked_actions = self.actor(attacked, deterministic=True)
                    loss = self.critic(attacked, attacked_actions).mean()
                else:
                    raise ValueError(f"Gradient attack is unsupported for {target}")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                para = para.clamp(-scale, scale).detach()
            return (original_tensor + para * std_tensor).detach().cpu().numpy().astype(
                np.float32
            )

        noise = initial.to(self.device) * std_tensor
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
        config.corruption_seed,
        config.adversarial_attack_profile,
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dataset_fingerprint(dataset: Dataset) -> str:
    digest = hashlib.sha256()
    for key in sorted(dataset):
        array = np.ascontiguousarray(dataset[key])
        digest.update(key.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def corruption_cache_fingerprint(
    dataset: Dataset,
    config: ExperimentConfig,
    oracle: Optional[AttackOracle],
) -> Tuple[str, Dict[str, Any]]:
    dataset_hash = dataset_fingerprint(dataset)
    checkpoint_hash = (
        _sha256_file(oracle.checkpoint) if oracle is not None else "none"
    )
    states = np.concatenate(
        (dataset["observations"], dataset["next_observations"]), axis=0
    )
    if config.state_normalization == "standard":
        location = states.mean(axis=0)
        scale_values = np.maximum(states.std(axis=0), 1e-3)
    elif config.state_normalization == "robust_median_mad":
        location = np.median(states, axis=0)
        scale_values = np.maximum(
            1.4826 * np.median(np.abs(states - location), axis=0), 1e-3
        )
    else:
        location = np.zeros(states.shape[1], dtype=np.float32)
        scale_values = np.ones(states.shape[1], dtype=np.float32)
    normalizer_digest = hashlib.sha256(config.state_normalization.encode("utf-8"))
    normalizer_digest.update(np.ascontiguousarray(location).tobytes())
    normalizer_digest.update(np.ascontiguousarray(scale_values).tobytes())
    metadata = {
        "dataset_fingerprint": dataset_hash,
        "attack_checkpoint_fingerprint": checkpoint_hash,
        "corruption": config.corruption,
        "corruption_target": config.corruption_target,
        "corruption_range": config.corruption_range,
        "offline_corruption_rate": config.offline_corruption_rate,
        "seed": config.corruption_seed,
        "attack_seed": config.corruption_seed,
        "offline_attack_steps": config.offline_attack_steps,
        "online_attack_steps": config.online_attack_steps,
        "attack_step_size": config.attack_step_size,
        "online_attack_step_size": config.online_attack_step_size,
        "attack_min_step_size": config.attack_min_step_size,
        "attack_norm": config.attack_norm,
        "mixed_ratios": list(config.mixed_ratios),
        "attack_implementation_version": ATTACK_IMPLEMENTATION_VERSION,
        "attack_implementation_profile": config.adversarial_attack_profile,
        "attack_objective": ATTACK_OBJECTIVE,
        "attack_optimizer": (
            "adam_reinitialized_each_step"
            if config.adversarial_attack_profile == "rpex_official_adam"
            else "sign_gradient_descent"
        ),
        "attack_timing": config.attack_timing,
        "random_attack_semantics": config.random_attack_semantics,
        "mixed_corruption_profile": config.mixed_corruption_profile,
        "device": str(getattr(oracle, "device", "unknown")) if oracle is not None else "numpy",
        "source_commit": "35da71ee5151b6179d21b9a2b4ce1b6408aedd04",
        "normalizer_sha256": normalizer_digest.hexdigest(),
        "mc_return_source": config.mc_return_source,
        "discount": config.discount,
    }
    encoded = json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest(), metadata


def _cache_file(
    dataset: Dataset,
    config: ExperimentConfig,
    oracle: Optional[AttackOracle],
    cache_root: Path,
) -> Tuple[Path, str, Dict[str, Any]]:
    cache_key, metadata = corruption_cache_fingerprint(dataset, config, oracle)
    name = f"{config.corruption}_{config.corruption_target}_{cache_key}.npz"
    return cache_root / config.env_name / name, cache_key, metadata


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
) -> Dict[str, Any]:
    total = sum(len(indices) for indices in target_indices.values())
    stats = {
        "corrupted_count": int(total),
        "selected_transition_count": int(total),
        "corrupted_fraction": float(total / dataset_size),
        "loaded_from_cache": float(loaded_from_cache),
        "attack_semantics": config.random_attack_semantics,
        "attack_timing": config.attack_timing,
        "corruption_seed": config.corruption_seed,
        "corruption_rate": config.offline_corruption_rate,
        "corruption_range": config.corruption_range,
    }
    for target, ratio in zip(
        INDIVIDUAL_CORRUPTION_TARGETS, config.mixed_ratios
    ):
        count = len(target_indices.get(target, ()))
        stats[f"{target}_corrupted_count"] = int(count)
        stats[f"{target}_corrupted_fraction"] = float(count / dataset_size)
        if config.corruption_target == "mixed":
            stats[f"{target}_allocation_ratio"] = float(ratio)
        indices_array = np.asarray(target_indices.get(target, ()), dtype=np.int64)
        stats[f"{target}_selected_indices_sha256"] = hashlib.sha256(
            indices_array.tobytes()
        ).hexdigest()
    combined = np.concatenate(
        [np.asarray(value, dtype=np.int64) for value in target_indices.values()]
    ) if target_indices else np.empty(0, dtype=np.int64)
    stats["selected_transition_indices_sha256"] = hashlib.sha256(
        np.sort(combined).tobytes()
    ).hexdigest()
    return stats


def _corruption_value_sha256(
    dataset: Dataset, target_indices: Dict[str, np.ndarray]
) -> str:
    digest = hashlib.sha256()
    for target in sorted(target_indices):
        indices = np.asarray(target_indices[target], dtype=np.int64)
        values = np.ascontiguousarray(dataset[_target_dataset_key(target)][indices])
        digest.update(target.encode("utf-8"))
        digest.update(indices.tobytes())
        digest.update(str(values.dtype).encode("ascii"))
        digest.update(str(values.shape).encode("ascii"))
        digest.update(memoryview(values).cast("B"))
    return digest.hexdigest()


def recompute_mc_returns(dataset: Dataset, discount: float) -> np.ndarray:
    if "episode_id" not in dataset:
        raise RuntimeError(
            "post-corruption MC returns require episode_id trajectory metadata"
        )
    rewards = np.asarray(dataset["rewards"], dtype=np.float64).reshape(-1)
    episode_ids = np.asarray(dataset["episode_id"], dtype=np.int64).reshape(-1)
    result = np.zeros(len(rewards), dtype=np.float32)
    running = 0.0
    previous_episode: Optional[int] = None
    for index in range(len(rewards) - 1, -1, -1):
        episode = int(episode_ids[index])
        if previous_episode is None or episode != previous_episode:
            running = 0.0
        running = float(rewards[index]) + discount * running
        result[index] = running
        previous_episode = episode
    return result


def mc_returns_from_reward_deltas(
    clean_rewards: np.ndarray,
    corrupted_rewards: np.ndarray,
    clean_mc_returns: np.ndarray,
    episode_ids: np.ndarray,
    discount: float,
) -> np.ndarray:
    """Adjust clean returns while retaining rewards from filtered timeout tails."""
    clean_rewards = np.array(clean_rewards, dtype=np.float64, copy=True).reshape(-1)
    corrupted_rewards = np.array(
        corrupted_rewards, dtype=np.float64, copy=True
    ).reshape(-1)
    clean_mc_returns = np.array(
        clean_mc_returns, dtype=np.float64, copy=True
    ).reshape(-1)
    episode_ids = np.array(episode_ids, dtype=np.int64, copy=True).reshape(-1)
    lengths = {
        len(clean_rewards),
        len(corrupted_rewards),
        len(clean_mc_returns),
        len(episode_ids),
    }
    if len(lengths) != 1:
        raise ValueError("reward-delta MC return inputs must have equal lengths")

    reward_delta = corrupted_rewards - clean_rewards
    delta_returns = np.zeros(len(reward_delta), dtype=np.float64)
    running = 0.0
    next_episode: Optional[int] = None
    for index in range(len(reward_delta) - 1, -1, -1):
        episode = int(episode_ids[index])
        if next_episode is None or episode != next_episode:
            running = 0.0
        running = float(reward_delta[index]) + discount * running
        delta_returns[index] = running
        next_episode = episode
    return (clean_mc_returns + delta_returns).astype(np.float32)


def _apply_mc_return_semantics(
    clean_dataset: Dataset,
    result: Dataset,
    target_indices: Dict[str, np.ndarray],
    config: ExperimentConfig,
) -> None:
    if "mc_returns" not in result:
        # Generic corruption callers that are not Cal-QL datasets do not need
        # trajectory returns. Real D4RL loads always include this metadata.
        return
    if config.mc_return_source == "legacy_pre_corruption":
        result["mc_calibration_valid"] = np.ones(
            len(result["rewards"]), dtype=np.float32
        )
        return
    reward_rows = target_indices.get("rewards", np.empty(0, dtype=np.int64))
    if len(reward_rows):
        if "episode_id" not in clean_dataset:
            raise RuntimeError(
                "post-corruption MC returns require episode_id trajectory metadata"
            )
        result["mc_returns"] = mc_returns_from_reward_deltas(
            clean_rewards=clean_dataset["rewards"],
            corrupted_rewards=result["rewards"],
            clean_mc_returns=clean_dataset["mc_returns"],
            episode_ids=clean_dataset["episode_id"],
            discount=config.discount,
        )
    valid = np.ones(len(result["rewards"]), dtype=np.float32)
    for target in ("observations", "actions", "dynamics"):
        valid[target_indices.get(target, np.empty(0, dtype=np.int64))] = 0.0
    result["mc_calibration_valid"] = valid


def _reward_diagnostics(clean: Dataset, corrupted: Dataset) -> Dict[str, float]:
    clean_rewards = np.asarray(clean["rewards"])
    corrupted_rewards = np.asarray(corrupted["rewards"])
    return {
        "clean_reward_mean": float(clean_rewards.mean()),
        "clean_reward_std": float(clean_rewards.std()),
        "clean_reward_min": float(clean_rewards.min()),
        "clean_reward_max": float(clean_rewards.max()),
        "corrupted_reward_mean": float(corrupted_rewards.mean()),
        "corrupted_reward_std": float(corrupted_rewards.std()),
        "corrupted_reward_min": float(corrupted_rewards.min()),
        "corrupted_reward_max": float(corrupted_rewards.max()),
    }


def _load_cached_corruption(
    dataset: Dataset,
    config: ExperimentConfig,
    cache_file: Path,
    cache_key: str,
    cache_metadata: Dict[str, Any],
) -> Tuple[Dataset, Dict[str, Any]]:
    checksum_path = cache_file.with_suffix(cache_file.suffix + ".sha256")
    if not checksum_path.exists():
        raise RuntimeError(f"corruption cache checksum missing: {checksum_path}")
    expected_checksum = checksum_path.read_text(encoding="ascii").strip()
    actual_checksum = _sha256_file(cache_file)
    if expected_checksum != actual_checksum:
        raise RuntimeError(
            f"corruption cache checksum mismatch: {cache_file}"
        )
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
            format_version = int(cached["format_version"].item())
            if format_version < 3:
                raise RuntimeError(
                    "legacy corruption cache lacks embedded identity/shape "
                    f"validation: {cache_file}; regenerate explicitly"
                )
            if str(cached["cache_key"].item()) != cache_key:
                raise RuntimeError("corruption cache embedded key mismatch")
            embedded_metadata = json.loads(str(cached["metadata_json"].item()))
            if embedded_metadata != cache_metadata:
                raise RuntimeError("corruption cache embedded metadata mismatch")
            if int(cached["dataset_size"].item()) != len(dataset["rewards"]):
                raise RuntimeError("corruption cache dataset-size mismatch")
            targets = [str(value) for value in cached["targets"].tolist()]
            if not targets or any(
                target not in INDIVIDUAL_CORRUPTION_TARGETS for target in targets
            ):
                raise RuntimeError("corruption cache contains invalid targets")
            for target in targets:
                indices = cached[f"indices_{target}"]
                values = cached[f"values_{target}"]
                source = dataset[_target_dataset_key(target)]
                expected_shape = (len(indices), *source.shape[1:])
                if indices.dtype.kind not in "iu" or indices.ndim != 1:
                    raise RuntimeError("corruption cache indices have invalid dtype/shape")
                if len(indices) and (
                    int(indices.min()) < 0 or int(indices.max()) >= len(source)
                ):
                    raise RuntimeError("corruption cache indices are out of bounds")
                if len(np.unique(indices)) != len(indices):
                    raise RuntimeError("corruption cache indices contain duplicates")
                if values.shape != expected_shape or values.dtype != source.dtype:
                    raise RuntimeError(
                        "corruption cache values have invalid dtype/shape"
                    )
                result[_target_dataset_key(target)][indices] = values
                target_indices[target] = indices
    _apply_mc_return_semantics(dataset, result, target_indices, config)
    stats = _corruption_stats(
        len(dataset["rewards"]), target_indices, config, loaded_from_cache=True
    )
    stats.update(
        cache_hit=True,
        cache_miss=False,
        cache_key=cache_key,
        **cache_metadata,
    )
    stats["corruption_value_sha256"] = _corruption_value_sha256(
        result, target_indices
    )
    stats.update(_reward_diagnostics(dataset, result))
    return result, stats


@contextmanager
def _exclusive_cache_lock(path: Path):
    import fcntl

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _atomic_write_cache(path: Path, payload: Dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".npz"
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        np.savez_compressed(temporary, **payload)
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        checksum = _sha256_file(temporary)
        os.replace(temporary, path)
        checksum_path = path.with_suffix(path.suffix + ".sha256")
        checksum_tmp = checksum_path.with_suffix(checksum_path.suffix + ".tmp")
        with checksum_tmp.open("w", encoding="ascii") as stream:
            stream.write(checksum + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(checksum_tmp, checksum_path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


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
) -> Tuple[Dataset, Dict[str, Any]]:
    if config.corruption == "clean":
        result = {key: value.copy() for key, value in dataset.items()}
        result["mc_calibration_valid"] = np.ones(
            len(result["rewards"]), dtype=np.float32
        )
        return result, {
            "corrupted_count": 0,
            "selected_transition_count": 0,
            "corrupted_fraction": 0.0,
            "loaded_from_cache": 0.0,
            "cache_hit": False,
            "cache_miss": False,
            "attack_semantics": config.random_attack_semantics,
            "attack_timing": config.attack_timing,
            "corruption_seed": config.corruption_seed,
            "corruption_rate": config.offline_corruption_rate,
            "corruption_range": config.corruption_range,
            "selected_transition_indices_sha256": hashlib.sha256(b"").hexdigest(),
            "corruption_value_sha256": hashlib.sha256(b"").hexdigest(),
        }

    cache_file, cache_key, cache_metadata = _cache_file(
        dataset, config, oracle, cache_root
    )
    lock_file = cache_file.with_suffix(cache_file.suffix + ".lock")
    with _exclusive_cache_lock(lock_file):
        if cache_file.exists() and not config.force_regenerate_attack:
            return _load_cached_corruption(
                dataset, config, cache_file, cache_key, cache_metadata
            )
        return _generate_and_cache_corruption(
            dataset,
            config,
            oracle,
            cache_file,
            cache_key,
            cache_metadata,
        )


def _generate_and_cache_corruption(
    dataset: Dataset,
    config: ExperimentConfig,
    oracle: Optional[AttackOracle],
    cache_file: Path,
    cache_key: str,
    cache_metadata: Dict[str, Any],
) -> Tuple[Dataset, Dict[str, Any]]:

    rng = np.random.default_rng(config.corruption_seed)
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
        "format_version": np.asarray(3, dtype=np.int64),
        "cache_key": np.asarray(cache_key, dtype=np.str_),
        "metadata_json": np.asarray(
            json.dumps(cache_metadata, sort_keys=True, separators=(",", ":")),
            dtype=np.str_,
        ),
        "dataset_size": np.asarray(len(dataset["rewards"]), dtype=np.int64),
        "targets": np.asarray(list(target_indices), dtype=np.str_),
    }
    for target, target_rows in target_indices.items():
        values = _corrupt_target_values(
            dataset, target, target_rows, config, oracle, rng
        )
        result[_target_dataset_key(target)][target_rows] = values
        cache_payload[f"indices_{target}"] = target_rows
        cache_payload[f"values_{target}"] = values

    _apply_mc_return_semantics(dataset, result, target_indices, config)

    _atomic_write_cache(cache_file, cache_payload)
    stats = _corruption_stats(
        len(dataset["rewards"]), target_indices, config, loaded_from_cache=False
    )
    stats.update(
        cache_hit=False,
        cache_miss=True,
        cache_key=cache_key,
        **cache_metadata,
    )
    stats["corruption_value_sha256"] = _corruption_value_sha256(
        result, target_indices
    )
    stats.update(_reward_diagnostics(dataset, result))
    return result, stats


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
    selected_target: Optional[str] = None,
    selection_already_sampled: bool = False,
) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray, bool]:
    if selected_target is None and not selection_already_sampled:
        selected_target = sample_online_corruption_target(config, rng)
    if selected_target is None:
        return raw_state, action, reward, raw_next_state, False

    state = raw_state.copy()
    stored_action = action.copy()
    stored_reward = float(reward)
    next_state = raw_next_state.copy()
    target = selected_target
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
            max(config.online_attack_step_size, config.attack_min_step_size),
        )[0]
    if target == "observations":
        state = attacked.astype(np.float32)
    elif target == "actions":
        stored_action = attacked.astype(np.float32)
    else:
        next_state = attacked.astype(np.float32)
    return state, stored_action, stored_reward, next_state, True


def sample_online_corruption_target(
    config: ExperimentConfig, rng: np.random.Generator
) -> Optional[str]:
    if config.corruption == "clean" or rng.random() >= config.online_corruption_rate:
        return None
    if config.corruption_target == "mixed":
        return str(
            rng.choice(
                INDIVIDUAL_CORRUPTION_TARGETS,
                p=np.asarray(config.mixed_ratios, dtype=np.float64),
            )
        )
    return config.corruption_target


def corrupt_pre_action_value(
    original: np.ndarray,
    target: str,
    raw_state: np.ndarray,
    action: Optional[np.ndarray],
    config: ExperimentConfig,
    oracle: Optional[AttackOracle],
    rng: np.random.Generator,
    state_std: np.ndarray,
    action_std: np.ndarray,
) -> np.ndarray:
    if target not in ("observations", "actions"):
        return original.copy()
    std = state_std if target == "observations" else action_std
    if config.corruption == "random":
        return (
            original
            + rng.uniform(-config.corruption_range, config.corruption_range, original.shape)
            * std
        ).astype(np.float32)
    if oracle is None:
        raise RuntimeError("An AttackOracle is required for adversarial corruption")
    if action is None:
        with torch.no_grad():
            action = (
                oracle.actor(
                    torch.as_tensor(
                        raw_state[None, :], dtype=torch.float32, device=oracle.device
                    ),
                    deterministic=True,
                )[0]
                .cpu()
                .numpy()
            )
    return oracle.attack(
        original[None, :],
        std[None, :],
        raw_state[None, :],
        action[None, :],
        target,
        config.corruption_range,
        config.online_attack_steps,
        max(config.online_attack_step_size, config.attack_min_step_size),
    )[0]
