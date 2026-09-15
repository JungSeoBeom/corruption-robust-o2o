from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import subprocess
import tempfile
import urllib.request
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple

import numpy as np
import torch

from .config import BENCHMARK_ENVS, DEFAULT_PROTOCOL, normalize_env_name


Dataset = Dict[str, np.ndarray]
MODERN_PROTOCOL = DEFAULT_PROTOCOL
STANDARD_DATASET_KEYS = (
    "observations",
    "actions",
    "next_observations",
    "rewards",
    "terminals",
)
GYMNASIUM_ENV_IDS = {
    "halfcheetah": "HalfCheetah-v4",
    "hopper": "Hopper-v4",
    "walker2d": "Walker2d-v4",
}
EXPECTED_LOCOMOTION_DIMS = {
    "halfcheetah": (17, 6),
    "hopper": (11, 3),
    "walker2d": (17, 6),
}
# Reference returns copied from D4RL's v2 score table. Keeping these constants
# preserves the original normalized-return definition without importing the
# unmaintained D4RL package at runtime.
D4RL_REFERENCE_SCORES = {
    "halfcheetah": (-280.178953, 12_135.0),
    "hopper": (-20.272305, 3_234.3),
    "walker2d": (1.629008, 4_592.3),
}
D4RL_DATASET_BASE_URL = (
    "https://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco_v2"
)


class EnvironmentSetupError(RuntimeError):
    """Raised when the Gymnasium/MuJoCo environment cannot be constructed."""


def require_finite_action(action: np.ndarray, *, label: str) -> np.ndarray:
    """Validate an action without silently repairing NaN or infinity."""

    array = np.asarray(action)
    if not np.all(np.isfinite(array)):
        raise FloatingPointError(f"{label} contains NaN or infinity: {array!r}")
    return array


def clip_action_to_space(
    proposal: np.ndarray,
    action_low: np.ndarray,
    action_high: np.ndarray,
) -> np.ndarray:
    """Map a finite proposal to the finite action-space box."""

    proposal_array = require_finite_action(proposal, label="action proposal")
    low = require_finite_action(action_low, label="action-space lower bound")
    high = require_finite_action(action_high, label="action-space upper bound")
    if proposal_array.shape != low.shape or low.shape != high.shape:
        raise ValueError(
            "action proposal and bounds must have identical shapes; got "
            f"proposal={proposal_array.shape}, low={low.shape}, high={high.shape}"
        )
    if np.any(low > high):
        raise ValueError("action-space lower bound exceeds upper bound")
    executed = np.clip(proposal_array, low, high).astype(np.float32)
    require_finite_action(executed, label="executed action")
    return executed


def _validate_protocol(protocol: str) -> None:
    if protocol != MODERN_PROTOCOL:
        raise ValueError(
            f"Unknown environment protocol {protocol!r}; "
            f"the only supported protocol is {MODERN_PROTOCOL!r}"
        )


def runtime_package_versions() -> Dict[str, str]:
    versions = {"Python": platform.python_version()}
    for label, package in (
        ("numpy", "numpy"),
        ("torch", "torch"),
        ("gymnasium", "gymnasium"),
        ("mujoco", "mujoco"),
        ("h5py", "h5py"),
    ):
        try:
            versions[label] = version(package)
        except PackageNotFoundError:
            versions[label] = "not-installed"
    return versions


def repository_state_metadata() -> Dict[str, Any]:
    """Return cheap repository provenance for a run manifest."""

    repository_root = Path(__file__).resolve().parents[1]

    def git(*arguments: str) -> bytes:
        completed = subprocess.run(
            ("git", *arguments),
            cwd=repository_root,
            check=True,
            capture_output=True,
        )
        return completed.stdout

    commit = git("rev-parse", "HEAD").decode("ascii").strip()
    status = git(
        "status", "--porcelain=v1", "--untracked-files=normal", "-z"
    )
    return {
        "git_commit": commit,
        "repository_commit": commit,
        "repository_dirty": bool(status),
        "repository_status_sha256": hashlib.sha256(status).hexdigest(),
    }


def _benchmark_env_name(env_name: str) -> str:
    normalized = normalize_env_name(env_name)
    if normalized not in BENCHMARK_ENVS:
        raise ValueError(
            f"Unsupported benchmark environment {normalized!r}; "
            f"choose from {BENCHMARK_ENVS}"
        )
    return normalized


def gymnasium_env_id(env_name: str) -> str:
    full_env_name = _benchmark_env_name(env_name)
    domain = full_env_name.split("-", 1)[0]
    return GYMNASIUM_ENV_IDS[domain]


def expected_env_spec_id(
    env_name: str,
    protocol: str = DEFAULT_PROTOCOL,
) -> str:
    _validate_protocol(protocol)
    return gymnasium_env_id(env_name)


def make_env(
    env_name: str,
    protocol: str = DEFAULT_PROTOCOL,
) -> object:
    _validate_protocol(protocol)
    full_env_name = _benchmark_env_name(env_name)
    try:
        import gymnasium as gym
    except ImportError as exc:
        raise EnvironmentSetupError(
            "Gymnasium is required. Create the Conda environment from "
            "environment.yml."
        ) from exc
    gymnasium_id = gymnasium_env_id(full_env_name)
    try:
        # Gymnasium recommends v5 globally, but this project intentionally
        # retains v4 because v5 changes the benchmark MDP. Avoid repeating the
        # same migration warning for every train/evaluation environment.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r".*environment .* is out of date.*",
                category=DeprecationWarning,
            )
            env = gym.make(gymnasium_id)
    except Exception as exc:
        raise EnvironmentSetupError(
            f"Failed to create Gymnasium environment {gymnasium_id!r}. "
            "Install gymnasium[mujoco] and mujoco from requirements.txt."
        ) from exc
    spec_id = getattr(getattr(env, "spec", None), "id", None)
    if spec_id != gymnasium_id:
        try:
            env.close()
        finally:
            raise EnvironmentSetupError(
                f"Environment registration mismatch: requested {gymnasium_id!r}, "
                f"but env.spec.id is {spec_id!r}."
            )
    setattr(env, "_robust_o2o_dataset_id", full_env_name)
    return env


def _max_episode_steps(env: object) -> int:
    value = getattr(env, "_max_episode_steps", None)
    if value is None:
        value = getattr(getattr(env, "spec", None), "max_episode_steps", None)
    if value is None:
        raise ValueError("Environment does not expose its maximum episode length")
    return int(value)


def _validate_raw_dataset(raw: Mapping[str, np.ndarray]) -> int:
    required = ("observations", "actions", "rewards", "terminals")
    missing = [key for key in required if key not in raw]
    if missing:
        raise KeyError(f"Raw D4RL dataset is missing keys: {missing}")
    size = len(raw["rewards"])
    if size < 2:
        raise ValueError("Raw D4RL dataset must contain at least two rows")
    checked = required + (("timeouts",) if "timeouts" in raw else ())
    if any(len(raw[key]) != size for key in checked):
        raise ValueError("Raw D4RL dataset arrays have inconsistent lengths")
    return size


def qlearning_valid_indices(
    raw: Mapping[str, np.ndarray],
    max_episode_steps: int,
) -> np.ndarray:
    """Reproduce pinned D4RL qlearning_dataset(..., terminate_on_end=False)."""
    size = _validate_raw_dataset(raw)
    terminals = np.asarray(raw["terminals"], dtype=bool)
    timeouts = (
        np.asarray(raw["timeouts"], dtype=bool) if "timeouts" in raw else None
    )
    indices = []
    episode_step = 0
    for index in range(size - 1):
        final_timestep = (
            bool(timeouts[index])
            if timeouts is not None
            else episode_step == max_episode_steps - 1
        )
        if final_timestep:
            episode_step = 0
            continue
        # Preserve D4RL's original qlearning_dataset ordering exactly: a
        # terminal resets the counter before the retained transition and the
        # unconditional increment happens afterwards.
        if bool(terminals[index]):
            episode_step = 0
        indices.append(index)
        episode_step += 1
    return np.asarray(indices, dtype=np.int64)


def _index_aware_qlearning_dataset(
    raw: Mapping[str, np.ndarray],
    valid_indices: np.ndarray,
) -> Dataset:
    return {
        "observations": np.asarray(
            raw["observations"][valid_indices], dtype=np.float32
        ).copy(),
        "actions": np.asarray(
            raw["actions"][valid_indices], dtype=np.float32
        ).copy(),
        "next_observations": np.asarray(
            raw["observations"][valid_indices + 1], dtype=np.float32
        ).copy(),
        "rewards": np.asarray(
            raw["rewards"][valid_indices], dtype=np.float32
        ).reshape(-1).copy(),
        "terminals": np.asarray(
            raw["terminals"][valid_indices], dtype=np.float32
        ).reshape(-1).copy(),
    }


def raw_monte_carlo_returns(
    raw: Mapping[str, np.ndarray],
    discount: float,
    max_episode_steps: int,
) -> np.ndarray:
    """Compute return-to-go before D4RL removes timeout transitions."""
    size = _validate_raw_dataset(raw)
    rewards = np.asarray(raw["rewards"], dtype=np.float64).reshape(-1)
    terminals = np.asarray(raw["terminals"], dtype=bool).reshape(-1)
    timeouts = (
        np.asarray(raw["timeouts"], dtype=bool).reshape(-1)
        if "timeouts" in raw
        else None
    )
    returns = np.zeros(size, dtype=np.float32)
    episode_start = 0
    episode_step = 0

    def fill_episode(end: int) -> None:
        running = 0.0
        for cursor in range(end, episode_start - 1, -1):
            running = float(rewards[cursor]) + discount * running
            returns[cursor] = running

    for index in range(size):
        final_timestep = (
            bool(timeouts[index])
            if timeouts is not None
            else episode_step == max_episode_steps - 1
        )
        if bool(terminals[index]) or final_timestep:
            fill_episode(index)
            episode_start = index + 1
            episode_step = 0
        else:
            episode_step += 1
    if episode_start < size:
        fill_episode(size - 1)
    return returns


def converted_monte_carlo_returns(
    dataset: Mapping[str, np.ndarray], discount: float
) -> np.ndarray:
    """Return-to-go over the transitions retained by qlearning_dataset.

    With ``terminate_on_end=False`` D4RL drops the timeout row before forming
    the replay dataset. Cal-QL therefore must not leak the dropped row's reward
    into the final retained transition's calibration target.
    """

    if "episode_id" not in dataset:
        raise KeyError("converted MC returns require episode_id")
    rewards = np.asarray(dataset["rewards"], dtype=np.float64).reshape(-1)
    episode_ids = np.asarray(dataset["episode_id"], dtype=np.int64).reshape(-1)
    if len(rewards) != len(episode_ids):
        raise ValueError("rewards and episode_id have different lengths")
    returns = np.empty(len(rewards), dtype=np.float32)
    running = 0.0
    next_episode: int | None = None
    for index in range(len(rewards) - 1, -1, -1):
        episode = int(episode_ids[index])
        if next_episode is None or episode != next_episode:
            running = 0.0
        running = float(rewards[index]) + discount * running
        returns[index] = running
        next_episode = episode
    return returns


def raw_episode_ids(
    raw: Mapping[str, np.ndarray], max_episode_steps: int
) -> np.ndarray:
    """Return stable trajectory IDs while respecting terminals and timeouts."""
    size = _validate_raw_dataset(raw)
    terminals = np.asarray(raw["terminals"], dtype=bool).reshape(-1)
    timeouts = (
        np.asarray(raw["timeouts"], dtype=bool).reshape(-1)
        if "timeouts" in raw
        else None
    )
    ids = np.empty(size, dtype=np.int64)
    episode_id = 0
    episode_step = 0
    for index in range(size):
        ids[index] = episode_id
        final_timestep = (
            bool(timeouts[index])
            if timeouts is not None
            else episode_step == max_episode_steps - 1
        )
        if bool(terminals[index]) or final_timestep:
            episode_id += 1
            episode_step = 0
        else:
            episode_step += 1
    return ids


def validate_dataset(dataset: Mapping[str, np.ndarray], env: object) -> Dataset:
    missing = [key for key in STANDARD_DATASET_KEYS if key not in dataset]
    if missing:
        raise KeyError(f"D4RL q-learning dataset is missing keys: {missing}")
    arrays = {
        key: np.asarray(value) for key, value in dataset.items()
    }
    size = len(arrays["rewards"])
    if size == 0:
        raise ValueError("D4RL q-learning dataset is empty")
    if any(len(arrays[key]) != size for key in arrays):
        raise ValueError("D4RL q-learning dataset arrays have inconsistent lengths")
    if arrays["observations"].ndim != 2 or arrays["next_observations"].ndim != 2:
        raise ValueError("observations and next_observations must be rank-2")
    if arrays["actions"].ndim != 2:
        raise ValueError("actions must be rank-2")
    if arrays["rewards"].ndim != 1 or arrays["terminals"].ndim != 1:
        raise ValueError("rewards and terminals must be rank-1")
    if "mc_returns" in arrays and arrays["mc_returns"].ndim != 1:
        raise ValueError("mc_returns must be rank-1")

    observation_shape = tuple(getattr(env.observation_space, "shape", ()))
    action_shape = tuple(getattr(env.action_space, "shape", ()))
    if len(observation_shape) != 1 or arrays["observations"].shape[1:] != observation_shape:
        raise ValueError(
            "Dataset observation dimension does not match the environment: "
            f"{arrays['observations'].shape[1:]} vs {observation_shape}"
        )
    if arrays["next_observations"].shape[1:] != observation_shape:
        raise ValueError("next_observations dimension does not match the environment")
    if len(action_shape) != 1 or arrays["actions"].shape[1:] != action_shape:
        raise ValueError(
            "Dataset action dimension does not match the environment: "
            f"{arrays['actions'].shape[1:]} vs {action_shape}"
        )
    for key, value in arrays.items():
        if not np.issubdtype(value.dtype, np.number) or not np.all(np.isfinite(value)):
            raise ValueError(f"Dataset array {key!r} contains non-finite values")
    return {
        key: np.asarray(value, dtype=np.float32).copy()
        for key, value in arrays.items()
    }


def local_dataset_path(
    env_name: str,
    dataset_dir: str | None = None,
) -> Path:
    full_env_name = _benchmark_env_name(env_name)
    root = (
        Path(dataset_dir).expanduser().resolve()
        if dataset_dir
        else Path.home() / ".d4rl" / "datasets"
    )
    if root.is_file() or root.suffix.lower() in {".h5", ".hdf5"}:
        return root
    domain, dataset_and_version = full_env_name.split("-", 1)
    dataset, version = dataset_and_version.rsplit("-", 1)
    filename = f"{domain}_{dataset.replace('-', '_')}-{version}.hdf5"
    return root / filename


def d4rl_dataset_url(env_name: str) -> str:
    """Return the official D4RL-v2 HDF5 URL for a benchmark task."""

    return f"{D4RL_DATASET_BASE_URL}/{local_dataset_path(env_name).name}"


def download_d4rl_dataset(
    env_name: str,
    dataset_dir: str | None = None,
    force: bool = False,
) -> Path:
    """Download one official D4RL-v2 HDF5 file atomically.

    A unique temporary file in the destination directory prevents interrupted
    downloads from being mistaken for complete datasets. The destination is
    replaced only after the HDF5 structure has been validated.
    """

    target = local_dataset_path(env_name, dataset_dir)
    if target.is_file() and not force:
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(
        d4rl_dataset_url(env_name),
        headers={"User-Agent": "corruption-robust-o2o/1"},
    )
    descriptor, temporary_name = tempfile.mkstemp(
        dir=target.parent,
        prefix=f".{target.name}.",
        suffix=".part",
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as output, urllib.request.urlopen(
            request, timeout=60
        ) as response:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
            output.flush()
            os.fsync(output.fileno())
        _load_local_raw_dataset(temporary_path)
        os.replace(temporary_path, target)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
    return target


def _load_local_raw_dataset(path: Path) -> Dataset:
    try:
        import h5py
    except ImportError as exc:
        raise EnvironmentSetupError(
            "Reading D4RL datasets requires h5py from requirements.txt."
        ) from exc
    if not path.is_file():
        raise FileNotFoundError(
            f"D4RL-v2 dataset not found: {path}. Run "
            "'python scripts/download_d4rl_datasets.py' first, or pass "
            "--dataset-dir."
        )
    with h5py.File(path, "r") as stream:
        required = ("observations", "actions", "rewards", "terminals")
        missing = [key for key in required if key not in stream]
        if missing:
            raise KeyError(f"D4RL HDF5 dataset is missing keys: {missing}")
        raw = {key: np.asarray(stream[key]) for key in required}
        if "timeouts" in stream:
            raw["timeouts"] = np.asarray(stream["timeouts"])
    return raw


def load_d4rl_dataset(
    env: object,
    dataset_dir: str | None = None,
    discount: float = 0.99,
    protocol: str = DEFAULT_PROTOCOL,
    env_name: str | None = None,
) -> Dataset:
    _validate_protocol(protocol)
    if env_name is None:
        env_name = getattr(env, "_robust_o2o_dataset_id", None)
    if env_name is None:
        raise ValueError(
            "env_name is required when the environment was not created by make_env()"
        )
    raw = _load_local_raw_dataset(local_dataset_path(env_name, dataset_dir))
    max_episode_steps = _max_episode_steps(env)
    valid_indices = qlearning_valid_indices(raw, max_episode_steps)
    dataset = _index_aware_qlearning_dataset(raw, valid_indices)
    dataset["episode_id"] = raw_episode_ids(raw, max_episode_steps)[
        valid_indices
    ].astype(np.float32, copy=True)
    dataset["mc_returns"] = converted_monte_carlo_returns(dataset, discount)
    dataset["mc_calibration_valid"] = np.ones(
        len(valid_indices), dtype=np.float32
    )
    return validate_dataset(dataset, env)


def make_env_and_dataset(
    env_name: str,
    dataset_dir: str | None = None,
    discount: float = 0.99,
    protocol: str = DEFAULT_PROTOCOL,
) -> Tuple[object, Dataset]:
    env = make_env(env_name, protocol)
    try:
        dataset = load_d4rl_dataset(
            env,
            dataset_dir,
            discount,
            protocol,
            env_name,
        )
    except BaseException:
        env.close()
        raise
    return env, dataset


def normalized_d4rl_scores(
    env_name: str,
    returns: np.ndarray,
    protocol: str = DEFAULT_PROTOCOL,
) -> np.ndarray:
    _validate_protocol(protocol)
    full_env_name = _benchmark_env_name(env_name)
    domain = full_env_name.split("-", 1)[0]
    random_return, expert_return = D4RL_REFERENCE_SCORES[domain]
    values = np.asarray(returns, dtype=np.float64)
    return (values - random_return) / (expert_return - random_return) * 100.0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def environment_metadata(
    env: object,
    env_name: str,
    dataset: Mapping[str, np.ndarray],
    seed: int,
    protocol: str = DEFAULT_PROTOCOL,
    dataset_dir: str | None = None,
) -> Dict[str, Any]:
    _validate_protocol(protocol)
    full_env_name = _benchmark_env_name(env_name)
    unwrapped = getattr(env, "unwrapped", env)
    source_path = local_dataset_path(full_env_name, dataset_dir)
    environment_backend = "gymnasium-v4+native-mujoco"
    dataset_backend = "d4rl-v2-hdf5+index-aware-qlearning-conversion"
    versions = runtime_package_versions()
    try:
        repository_metadata = repository_state_metadata()
    except (OSError, RuntimeError, subprocess.SubprocessError):
        repository_metadata = {
            "git_commit": "unknown",
            "repository_commit": "unknown",
            "repository_dirty": None,
            "repository_status_sha256": None,
        }
    action_space = env.action_space
    action_low = np.asarray(
        getattr(action_space, "low", np.full(action_space.shape, np.nan)),
        dtype=np.float64,
    ).tolist()
    action_high = np.asarray(
        getattr(action_space, "high", np.full(action_space.shape, np.nan)),
        dtype=np.float64,
    ).tolist()
    evaluation_env_id = getattr(getattr(env, "spec", None), "id", None)
    metadata = {
        "protocol": protocol,
        "environment_protocol": protocol,
        "environment_id": full_env_name,
        "d4rl_env_id": full_env_name,
        "dataset_id": full_env_name,
        "env_spec_id": evaluation_env_id,
        "evaluation_env_id": evaluation_env_id,
        "online_env_id": evaluation_env_id,
        "unwrapped_environment_class": type(unwrapped).__name__,
        "unwrapped_environment_module": type(unwrapped).__module__,
        "environment_backend": environment_backend,
        "dataset_backend": dataset_backend,
        "runtime_package_versions": versions,
        "python_version": versions["Python"],
        "gymnasium_version": versions["gymnasium"],
        "numpy_version": versions["numpy"],
        "torch_version": versions["torch"],
        "h5py_version": versions["h5py"],
        "mujoco_backend": "native_mujoco",
        "mujoco_runtime_version": versions["mujoco"],
        **repository_metadata,
        "dataset_url": d4rl_dataset_url(full_env_name),
        "dataset_path": str(source_path),
        "dataset_sha256": (
            _sha256(source_path)
            if source_path.is_file()
            else None
        ),
        "observation_dim": int(dataset["observations"].shape[1]),
        "action_dim": int(dataset["actions"].shape[1]),
        "dataset_size": int(len(dataset["rewards"])),
        "action_low": action_low,
        "action_high": action_high,
        "environment_max_episode_steps": _max_episode_steps(env),
        "environment_seed": int(seed),
        # Backward-compatible metadata aliases. Run configuration values are
        # merged afterwards so these cannot overwrite resolved CLI values.
        "max_episode_steps": _max_episode_steps(env),
        "seed": int(seed),
    }
    fingerprint_payload = {
        key: metadata[key]
        for key in (
            "protocol",
            "d4rl_env_id",
            "env_spec_id",
            "environment_backend",
            "dataset_backend",
            "dataset_sha256",
            "observation_dim",
            "action_dim",
            "dataset_size",
            "environment_max_episode_steps",
            "action_low",
            "action_high",
            "python_version",
            "gymnasium_version",
            "numpy_version",
            "torch_version",
            "h5py_version",
            "mujoco_backend",
            "mujoco_runtime_version",
        )
    }
    serialized = json.dumps(
        fingerprint_payload, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    metadata["environment_fingerprint_payload"] = fingerprint_payload
    metadata["environment_fingerprint"] = hashlib.sha256(serialized).hexdigest()
    return metadata


def preflight_runtime(
    env_name: str,
    dataset_dir: str | None = None,
    protocol: str = DEFAULT_PROTOCOL,
) -> Dict[str, Any]:
    """Validate the modern backend and load its D4RL-v2 dataset."""
    _validate_protocol(protocol)
    env, dataset = make_env_and_dataset(
        env_name,
        dataset_dir,
        protocol=protocol,
    )
    try:
        reset_env(env, seed=0, protocol=protocol)
        domain = normalize_env_name(env_name).split("-", 1)[0]
        expected_state_dim, expected_action_dim = EXPECTED_LOCOMOTION_DIMS[domain]
        actual = (
            int(dataset["observations"].shape[1]),
            int(dataset["actions"].shape[1]),
        )
        if actual != (expected_state_dim, expected_action_dim):
            raise EnvironmentSetupError(
                "D4RL-v2 observation/action dimensions mismatch: "
                f"expected={(expected_state_dim, expected_action_dim)}, actual={actual}"
            )
        return environment_metadata(
            env,
            env_name,
            dataset,
            seed=0,
            protocol=protocol,
            dataset_dir=dataset_dir,
        )
    finally:
        env.close()


@dataclass
class StateNormalizer:
    mean: np.ndarray
    std: np.ndarray
    mode: str = "standard"

    @classmethod
    def fit(
        cls,
        dataset: Dataset,
        enabled: bool = True,
        mode: str = "standard",
        additive_epsilon: bool = False,
    ) -> "StateNormalizer":
        state_dim = dataset["observations"].shape[-1]
        if not enabled or mode == "none":
            return cls(
                mean=np.zeros(state_dim, dtype=np.float32),
                std=np.ones(state_dim, dtype=np.float32),
                mode="none",
            )
        states = np.concatenate(
            (dataset["observations"], dataset["next_observations"]), axis=0
        )
        if mode == "standard":
            location = states.mean(axis=0)
            scale = states.std(axis=0)
        elif mode == "robust_median_mad":
            location = np.median(states, axis=0)
            scale = 1.4826 * np.median(np.abs(states - location), axis=0)
        else:
            raise ValueError(f"Unknown normalization mode {mode!r}")
        return cls(
            mean=np.asarray(location, dtype=np.float32),
            std=(
                np.asarray(scale, dtype=np.float32) + np.float32(1e-3)
                if additive_epsilon and mode == "standard"
                else np.maximum(np.asarray(scale, dtype=np.float32), 1e-3)
            ),
            mode=mode,
        )

    def transform(self, states: np.ndarray) -> np.ndarray:
        return ((states - self.mean) / self.std).astype(np.float32)

    def state_dict(self) -> Dict[str, np.ndarray]:
        return {"mean": self.mean, "std": self.std, "mode": self.mode}

    def diagnostics(self, dataset: Dataset) -> Dict[str, float | str]:
        transformed = self.transform(dataset["observations"])
        return {
            "normalizer_mode": self.mode,
            "mean_or_median_min": float(self.mean.min()),
            "mean_or_median_max": float(self.mean.max()),
            "scale_min": float(self.std.min()),
            "scale_max": float(self.std.max()),
            "fraction_of_near_constant_dimensions": float(
                np.mean(self.std <= 1.001e-3)
            ),
            "maximum_normalized_absolute_value": float(
                np.max(np.abs(transformed))
            ),
        }

    @classmethod
    def from_state_dict(cls, state: Dict[str, np.ndarray]) -> "StateNormalizer":
        return cls(
            np.asarray(state["mean"]),
            np.asarray(state["std"]),
            str(state.get("mode", "standard")),
        )


def apply_normalizer(dataset: Dataset, normalizer: StateNormalizer) -> Dataset:
    result = {key: value.copy() for key, value in dataset.items()}
    result["observations"] = normalizer.transform(result["observations"])
    result["next_observations"] = normalizer.transform(result["next_observations"])
    return result


def reset_env(
    env: object,
    seed: int | None = None,
    protocol: str = DEFAULT_PROTOCOL,
) -> np.ndarray:
    _validate_protocol(protocol)
    result = env.reset(seed=seed) if seed is not None else env.reset()
    if not isinstance(result, tuple) or len(result) != 2:
        raise EnvironmentSetupError(
            "Gymnasium env.reset() must return (observation, info)"
        )
    observation, info = result
    if not isinstance(info, dict):
        raise EnvironmentSetupError("Gymnasium env.reset() info must be a dict")
    return np.asarray(observation, dtype=np.float32)


def step_env(
    env: object,
    action: np.ndarray,
    protocol: str = DEFAULT_PROTOCOL,
) -> Tuple[np.ndarray, float, bool, bool, dict]:
    _validate_protocol(protocol)
    result = env.step(action)
    if not isinstance(result, tuple) or len(result) != 5:
        count = len(result) if isinstance(result, tuple) else "non-tuple"
        raise EnvironmentSetupError(
            f"Gymnasium env.step() must return five values, got {count}"
        )
    observation, reward, terminated, truncated, info = result
    if not isinstance(info, dict):
        raise EnvironmentSetupError("Gymnasium env.step() info must be a dict")
    return (
        np.asarray(observation, dtype=np.float32),
        float(reward),
        bool(terminated),
        bool(truncated),
        info,
    )


@contextmanager
def preserve_training_rng_state():
    """Keep evaluation from consuming any training RNG stream."""
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    mps_available = (
        hasattr(torch, "mps")
        and hasattr(torch.mps, "get_rng_state")
        and hasattr(torch.mps, "set_rng_state")
        and hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    )
    mps_state = torch.mps.get_rng_state().clone() if mps_available else None
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)
        if mps_state is not None:
            torch.mps.set_rng_state(mps_state)


@torch.no_grad()
def evaluate_agent(
    env: object,
    env_name: str,
    agent: object,
    normalizer: StateNormalizer,
    device: torch.device,
    episodes: int,
    max_episode_steps: int,
    seed: int,
    protocol: str = DEFAULT_PROTOCOL,
    evaluation_mode: str = "deterministic",
    action_execution_profile: str = "clip_to_action_space",
) -> Dict[str, float]:
    returns = []
    with preserve_training_rng_state():
        for episode in range(episodes):
            raw_state = reset_env(
                env,
                seed=seed + 10_000 + episode,
                protocol=protocol,
            )
            episode_return = 0.0
            for _ in range(max_episode_steps):
                state = torch.as_tensor(
                    normalizer.transform(raw_state),
                    dtype=torch.float32,
                    device=device,
                )
                action = agent.select_action(
                    state,
                    evaluate=True,
                    evaluation_mode=evaluation_mode,
                )
                action_np = action.detach().cpu().numpy()
                if action_execution_profile == "clip_to_action_space":
                    action_np = clip_action_to_space(
                        action_np,
                        env.action_space.low,
                        env.action_space.high,
                    )
                elif action_execution_profile != "official_algorithm_behavior":
                    raise ValueError(
                        f"Unknown action_execution_profile {action_execution_profile!r}"
                    )
                else:
                    action_np = require_finite_action(
                        action_np, label="unbounded diagnostic action"
                    ).astype(np.float32)
                raw_state, reward, terminated, truncated, _ = step_env(
                    env,
                    action_np,
                    protocol=protocol,
                )
                episode_return += reward
                if terminated or truncated:
                    break
            returns.append(episode_return)

    returns_np = np.asarray(returns, dtype=np.float64)
    normalized = normalized_d4rl_scores(env_name, returns_np, protocol)
    result = {
        "return_mean": float(returns_np.mean()),
        "return_std": float(returns_np.std()),
        "normalized_return_mean": float(np.nanmean(normalized)),
        "normalized_return_std": float(np.nanstd(normalized)),
    }
    return result
