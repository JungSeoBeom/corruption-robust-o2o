from __future__ import annotations

import hashlib
import json
import platform
import random
import subprocess
from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, distribution, version
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple

import numpy as np
import torch

from .config import (
    BENCHMARK_ENVS,
    DEFAULT_PROTOCOL,
    LEGACY_PROTOCOL,
    LOCAL_PROTOCOL,
    normalize_env_name,
)


Dataset = Dict[str, np.ndarray]
EXPECTED_D4RL_COMMIT = "d842aa194b416e564e54b0730d9f934e3e32f854"
EXPECTED_GYM_VERSION = "0.23.1"
EXPECTED_NUMPY_VERSION = "1.23.5"
EXPECTED_PYTHON_SERIES = "3.10"
EXPECTED_MUJOCO_PY_VERSION = "2.1.2.14"
# ``mujoco_py`` reports the linked MuJoCo C library as an integer.  The
# official 2.1 binary is 210; checking the binding package alone cannot detect
# a mismatched simulator installation.
EXPECTED_MUJOCO_RUNTIME_VERSION_CODE = 210
STANDARD_DATASET_KEYS = (
    "observations",
    "actions",
    "next_observations",
    "rewards",
    "terminals",
)
LOCAL_GYMNASIUM_ENV_IDS = {
    "halfcheetah": "HalfCheetah-v4",
    "hopper": "Hopper-v4",
    "walker2d": "Walker2d-v4",
}
EXPECTED_LOCOMOTION_DIMS = {
    "halfcheetah": (17, 6),
    "hopper": (11, 3),
    "walker2d": (17, 6),
}
# D4RL v2 reference returns used by d4rl.get_normalized_score().
LOCAL_D4RL_REFERENCE_SCORES = {
    "halfcheetah": (-280.178953, 12_135.0),
    "hopper": (-20.272305, 3_234.3),
    "walker2d": (1.629008, 4_592.3),
}


class RPEXProtocolError(RuntimeError):
    """Raised when the strict RPEX/D4RL-v2 runtime cannot be reproduced."""


def runtime_package_versions() -> Dict[str, str]:
    versions = {"Python": platform.python_version()}
    for label, package in (
        ("numpy", "numpy"),
        ("torch", "torch"),
        ("gym", "gym"),
        ("d4rl", "d4rl"),
        ("mujoco-py", "mujoco-py"),
        ("h5py", "h5py"),
        ("gymnasium", "gymnasium"),
        ("mujoco", "mujoco"),
    ):
        try:
            versions[label] = version(package)
        except PackageNotFoundError:
            versions[label] = "not-installed"
    return versions


def installed_d4rl_commit() -> str | None:
    """Read the immutable VCS commit recorded by pip's direct_url metadata."""
    try:
        metadata = distribution("d4rl").read_text("direct_url.json")
    except PackageNotFoundError:
        return None
    if not metadata:
        return None
    try:
        payload = json.loads(metadata)
    except json.JSONDecodeError:
        return None
    vcs = payload.get("vcs_info", {})
    return vcs.get("commit_id") or vcs.get("requested_revision")


def mujoco_runtime_identity() -> Dict[str, Any]:
    """Return both Python-binding and linked native MuJoCo identities."""

    identity: Dict[str, Any] = {
        "mujoco_py_version": runtime_package_versions()["mujoco-py"],
        "mujoco_runtime_version_code": None,
        "mujoco_runtime_version": None,
        "mujoco_runtime_path": None,
        "mujoco_runtime_error": None,
    }
    try:
        import mujoco_py

        version_code = int(mujoco_py.functions.mj_version())
        identity["mujoco_runtime_version_code"] = version_code
        identity["mujoco_runtime_version"] = (
            f"{version_code // 100}.{(version_code // 10) % 10}."
            f"{version_code % 10}"
        )
        discover = getattr(
            getattr(mujoco_py, "utils", None), "discover_mujoco", None
        )
        if discover is not None:
            identity["mujoco_runtime_path"] = str(Path(discover()).resolve())
    except BaseException as exc:
        identity["mujoco_runtime_error"] = f"{type(exc).__name__}: {exc}"
    return identity


@lru_cache(maxsize=1)
def verify_rpex_runtime() -> None:
    versions = runtime_package_versions()
    errors = []
    if not versions["Python"].startswith(f"{EXPECTED_PYTHON_SERIES}."):
        errors.append(
            f"Python {EXPECTED_PYTHON_SERIES} is required, found {versions['Python']}"
        )
    if versions["gym"] != EXPECTED_GYM_VERSION:
        errors.append(
            f"gym=={EXPECTED_GYM_VERSION} is required, found {versions['gym']}"
        )
    if versions["numpy"] != EXPECTED_NUMPY_VERSION:
        errors.append(
            f"numpy=={EXPECTED_NUMPY_VERSION} is required, found {versions['numpy']}"
        )
    if versions["mujoco-py"] != EXPECTED_MUJOCO_PY_VERSION:
        errors.append(
            f"mujoco-py=={EXPECTED_MUJOCO_PY_VERSION} is required, "
            f"found {versions['mujoco-py']}"
        )
    mujoco_identity = mujoco_runtime_identity()
    if (
        mujoco_identity["mujoco_runtime_version_code"]
        != EXPECTED_MUJOCO_RUNTIME_VERSION_CODE
    ):
        found = (
            mujoco_identity["mujoco_runtime_version_code"]
            if mujoco_identity["mujoco_runtime_version_code"] is not None
            else mujoco_identity["mujoco_runtime_error"] or "unavailable"
        )
        errors.append(
            "linked MuJoCo runtime version code "
            f"{EXPECTED_MUJOCO_RUNTIME_VERSION_CODE} (2.1.0) is required, "
            f"found {found}"
        )
    commit = installed_d4rl_commit()
    if commit != EXPECTED_D4RL_COMMIT:
        found = commit or "unavailable (direct_url.json missing)"
        errors.append(
            "D4RL must be installed from commit "
            f"{EXPECTED_D4RL_COMMIT}, found {found}"
        )
    if errors:
        raise RPEXProtocolError(
            f"{DEFAULT_PROTOCOL} reproducibility check failed:\n- "
            + "\n- ".join(errors)
            + "\nCreate the pinned environment from environment-rpex-v2.yml. "
            "Do not substitute Gymnasium v4/v5."
        )


def _import_legacy_backend() -> Tuple[object, object]:
    try:
        import gym
    except ImportError as exc:
        raise RPEXProtocolError(
            "The strict RPEX protocol requires legacy gym==0.23.1. "
            "Create the environment from environment-rpex-v2.yml."
        ) from exc
    try:
        import d4rl
    except BaseException as exc:
        raise RPEXProtocolError(
            "D4RL or its mujoco_py backend could not be imported. Exact RPEX "
            "reproduction generally requires Linux x86_64, MuJoCo 2.1, and "
            "the dependencies in requirements-rpex-v2.txt; no Gymnasium "
            "fallback will be used."
        ) from exc
    verify_rpex_runtime()
    return gym, d4rl


def _strict_env_name(env_name: str) -> str:
    normalized = normalize_env_name(env_name)
    if normalized not in BENCHMARK_ENVS:
        if normalized.endswith(("-v4", "-v5")):
            raise RPEXProtocolError(
                f"{env_name!r} is a Gymnasium ID. {DEFAULT_PROTOCOL} requires "
                "a complete D4RL-v2 ID such as 'walker2d-medium-replay-v2'."
            )
        raise ValueError(
            f"Unsupported RPEX benchmark environment {normalized!r}; "
            f"choose from {BENCHMARK_ENVS}"
        )
    if not normalized.endswith("-v2"):
        raise RPEXProtocolError(
            f"Strict RPEX environments must end in '-v2', got {normalized!r}"
        )
    return normalized


def _set_dataset_path(d4rl: object, dataset_dir: str | None) -> None:
    if dataset_dir is None:
        return
    setter = getattr(d4rl, "set_dataset_path", None)
    if setter is None:
        raise RPEXProtocolError(
            "Pinned D4RL does not expose d4rl.set_dataset_path(); the installed "
            "package does not match the required commit."
        )
    setter(str(Path(dataset_dir).expanduser().resolve()))


def local_gymnasium_env_id(env_name: str) -> str:
    full_env_name = _strict_env_name(env_name)
    domain = full_env_name.split("-", 1)[0]
    return LOCAL_GYMNASIUM_ENV_IDS[domain]


def expected_env_spec_id(
    env_name: str,
    protocol: str = DEFAULT_PROTOCOL,
) -> str:
    if protocol == LOCAL_PROTOCOL:
        return local_gymnasium_env_id(env_name)
    return _strict_env_name(env_name)


def make_env(
    env_name: str,
    protocol: str = DEFAULT_PROTOCOL,
) -> object:
    full_env_name = _strict_env_name(env_name)
    if protocol == LOCAL_PROTOCOL:
        try:
            import gymnasium as gym
        except ImportError as exc:
            raise RPEXProtocolError(
                "The local protocol requires Gymnasium. Activate the "
                "'corruption' Conda environment."
            ) from exc
        gymnasium_id = local_gymnasium_env_id(full_env_name)
        try:
            return gym.make(gymnasium_id)
        except BaseException as exc:
            raise RPEXProtocolError(
                f"Failed to create local Gymnasium environment {gymnasium_id!r}."
            ) from exc
    if protocol != LEGACY_PROTOCOL:
        raise ValueError(f"Unknown environment protocol {protocol!r}")
    gym, _ = _import_legacy_backend()
    try:
        env = gym.make(full_env_name)
    except BaseException as exc:
        raise RPEXProtocolError(
            f"Failed to create registered D4RL environment {full_env_name!r}. "
            "Verify the pinned D4RL commit, mujoco_py, and MuJoCo 2.1. "
            "The ID will not be translated to a Gymnasium v4/v5 task."
        ) from exc
    spec_id = getattr(getattr(env, "spec", None), "id", None)
    if spec_id != full_env_name:
        try:
            env.close()
        finally:
            raise RPEXProtocolError(
                f"Environment registration mismatch: requested {full_env_name!r}, "
                f"but env.spec.id is {spec_id!r}."
            )
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


def _assert_official_dataset_matches(
    official: Mapping[str, np.ndarray],
    converted: Mapping[str, np.ndarray],
) -> None:
    missing = [key for key in STANDARD_DATASET_KEYS if key not in official]
    if missing:
        raise AssertionError(f"d4rl.qlearning_dataset omitted keys: {missing}")
    for key in STANDARD_DATASET_KEYS:
        actual = np.asarray(official[key])
        expected = np.asarray(converted[key])
        if actual.shape != expected.shape or not np.allclose(
            actual, expected, rtol=1e-6, atol=1e-6
        ):
            raise AssertionError(
                "Pinned d4rl.qlearning_dataset disagrees with the index-aware "
                f"conversion for {key}: {actual.shape} vs {expected.shape}"
            )


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
    full_env_name = _strict_env_name(env_name)
    root = (
        Path(dataset_dir).expanduser().resolve()
        if dataset_dir
        else Path.home() / ".d4rl" / "datasets"
    )
    if root.is_file():
        return root
    domain, dataset_and_version = full_env_name.split("-", 1)
    dataset, version = dataset_and_version.rsplit("-", 1)
    filename = f"{domain}_{dataset.replace('-', '_')}-{version}.hdf5"
    return root / filename


def _load_local_raw_dataset(path: Path) -> Dataset:
    try:
        import h5py
    except ImportError as exc:
        raise RPEXProtocolError(
            "The local protocol requires h5py in the 'corruption' environment."
        ) from exc
    if not path.is_file():
        raise FileNotFoundError(
            f"Local D4RL dataset not found: {path}. Place the v2 HDF5 file in "
            "~/.d4rl/datasets or pass --dataset-dir."
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
    if protocol == LOCAL_PROTOCOL:
        if env_name is None:
            raise ValueError("env_name is required when loading a local D4RL dataset")
        raw = _load_local_raw_dataset(local_dataset_path(env_name, dataset_dir))
        max_episode_steps = _max_episode_steps(env)
        valid_indices = qlearning_valid_indices(raw, max_episode_steps)
        dataset = _index_aware_qlearning_dataset(raw, valid_indices)
        raw_returns = raw_monte_carlo_returns(raw, discount, max_episode_steps)
        dataset["mc_returns"] = raw_returns[valid_indices].astype(
            np.float32, copy=True
        )
        dataset["episode_id"] = raw_episode_ids(raw, max_episode_steps)[
            valid_indices
        ].astype(np.float32, copy=True)
        dataset["mc_calibration_valid"] = np.ones(
            len(valid_indices), dtype=np.float32
        )
        return validate_dataset(dataset, env)
    if protocol != LEGACY_PROTOCOL:
        raise ValueError(f"Unknown dataset protocol {protocol!r}")
    _, d4rl = _import_legacy_backend()
    _set_dataset_path(d4rl, dataset_dir)
    raw = env.get_dataset()
    max_episode_steps = _max_episode_steps(env)
    valid_indices = qlearning_valid_indices(raw, max_episode_steps)
    official = d4rl.qlearning_dataset(
        env,
        dataset=raw,
        terminate_on_end=False,
    )
    converted = _index_aware_qlearning_dataset(raw, valid_indices)
    _assert_official_dataset_matches(official, converted)
    dataset = {
        key: np.asarray(official[key], dtype=np.float32).copy()
        for key in STANDARD_DATASET_KEYS
    }
    raw_returns = raw_monte_carlo_returns(raw, discount, max_episode_steps)
    dataset["mc_returns"] = raw_returns[valid_indices].astype(np.float32, copy=True)
    dataset["episode_id"] = raw_episode_ids(raw, max_episode_steps)[
        valid_indices
    ].astype(np.float32, copy=True)
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
    full_env_name = _strict_env_name(env_name)
    if protocol == LOCAL_PROTOCOL:
        domain = full_env_name.split("-", 1)[0]
        random_return, expert_return = LOCAL_D4RL_REFERENCE_SCORES[domain]
        values = np.asarray(returns, dtype=np.float64)
        return (values - random_return) / (expert_return - random_return) * 100.0
    if protocol != LEGACY_PROTOCOL:
        raise ValueError(f"Unknown score protocol {protocol!r}")
    _, d4rl = _import_legacy_backend()
    values = np.asarray(returns, dtype=np.float64)
    return np.asarray(
        d4rl.get_normalized_score(full_env_name, values), dtype=np.float64
    ) * 100.0


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
    full_env_name = _strict_env_name(env_name)
    unwrapped = getattr(env, "unwrapped", env)
    if protocol == LOCAL_PROTOCOL:
        dataset_url = None
        dataset_path = local_dataset_path(full_env_name, dataset_dir)
        environment_backend = "gymnasium-v4+native-mujoco"
        dataset_backend = "local-d4rl-v2-hdf5+index-aware-qlearning-conversion"
        expected_commit = None
        installed_commit = None
    elif protocol == LEGACY_PROTOCOL:
        dataset_url = getattr(unwrapped, "dataset_url", None)
        dataset_path_value = getattr(unwrapped, "dataset_filepath", None)
        dataset_path = (
            Path(dataset_path_value).expanduser().resolve()
            if dataset_path_value
            else None
        )
        environment_backend = "gym-0.23.1+d4rl-v2+mujoco_py"
        dataset_backend = "d4rl.qlearning_dataset(terminate_on_end=False)"
        expected_commit = EXPECTED_D4RL_COMMIT
        installed_commit = installed_d4rl_commit()
    else:
        raise ValueError(f"Unknown metadata protocol {protocol!r}")
    versions = runtime_package_versions()
    mujoco_identity = (
        mujoco_runtime_identity()
        if protocol == LEGACY_PROTOCOL
        else {
            "mujoco_py_version": versions["mujoco-py"],
            "mujoco_runtime_version_code": None,
            "mujoco_runtime_version": versions["mujoco"],
            "mujoco_runtime_path": None,
            "mujoco_runtime_error": None,
        }
    )
    repository_worktree_sha256 = None
    try:
        repository_root = Path(__file__).resolve().parents[1]
        git_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        git_dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=repository_root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        from .final_gate import audit_context

        repository_context = audit_context()
        worktree_payload = {
            key: repository_context[key]
            for key in (
                "git_head",
                "git_status_sha256",
                "git_worktree_diff_sha256",
                "git_untracked_content_sha256",
            )
        }
        repository_worktree_sha256 = hashlib.sha256(
            json.dumps(
                worktree_payload,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
    except (OSError, RuntimeError, subprocess.SubprocessError):
        git_commit = "unknown"
        git_dirty = None
        repository_worktree_sha256 = None
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
        "gym_version": versions["gym"],
        "gymnasium_version": versions["gymnasium"],
        "numpy_version": versions["numpy"],
        "mujoco_backend": (
            "mujoco_py" if protocol == LEGACY_PROTOCOL else "native_mujoco"
        ),
        **mujoco_identity,
        "d4rl_version_or_commit": installed_commit or versions["d4rl"],
        "git_commit": git_commit,
        "repository_commit": git_commit,
        "repository_dirty": git_dirty,
        "repository_worktree_sha256": repository_worktree_sha256,
        "benchmark_comparable": protocol == LEGACY_PROTOCOL,
        "diagnostic_reason": (
            None
            if protocol == LEGACY_PROTOCOL
            else "D4RL-v2 dataset evaluated on Gymnasium-v4 simulator"
        ),
        "expected_d4rl_commit": expected_commit,
        "installed_d4rl_commit": installed_commit,
        "dataset_url": dataset_url,
        "dataset_path": str(dataset_path) if dataset_path else None,
        "dataset_sha256": (
            _sha256(dataset_path)
            if dataset_path is not None and dataset_path.is_file()
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
            "gym_version",
            "gymnasium_version",
            "numpy_version",
            "mujoco_backend",
            "mujoco_py_version",
            "mujoco_runtime_version_code",
            "mujoco_runtime_version",
            "d4rl_version_or_commit",
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
    """Validate the selected backend and materialize its D4RL dataset."""
    env, dataset = make_env_and_dataset(
        env_name,
        dataset_dir,
        protocol=protocol,
    )
    try:
        reset_env(env, seed=0, protocol=protocol)
        if protocol == LEGACY_PROTOCOL:
            domain = normalize_env_name(env_name).split("-", 1)[0]
            expected_state_dim, expected_action_dim = EXPECTED_LOCOMOTION_DIMS[domain]
            actual = (
                int(dataset["observations"].shape[1]),
                int(dataset["actions"].shape[1]),
            )
            if actual != (expected_state_dim, expected_action_dim):
                raise RPEXProtocolError(
                    "Strict D4RL-v2 observation/action dimensions mismatch: "
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
    if protocol == LOCAL_PROTOCOL:
        result = env.reset(seed=seed) if seed is not None else env.reset()
        observation = result[0] if isinstance(result, tuple) else result
        return np.asarray(observation, dtype=np.float32)
    if protocol != LEGACY_PROTOCOL:
        raise ValueError(f"Unknown reset protocol {protocol!r}")
    if seed is not None:
        seed_method = getattr(env, "seed", None)
        if seed_method is None:
            raise RPEXProtocolError(
                "Legacy RPEX environment does not expose env.seed(); the wrong "
                "environment backend may be active."
            )
        seed_method(seed)
    result = env.reset()
    if isinstance(result, tuple):
        raise RPEXProtocolError(
            "env.reset() returned a tuple. The strict RPEX protocol expects the "
            "legacy Gym API and will not accept a Gymnasium reset result."
        )
    return np.asarray(result, dtype=np.float32)


def step_env(
    env: object,
    action: np.ndarray,
    protocol: str = DEFAULT_PROTOCOL,
) -> Tuple[np.ndarray, float, bool, bool, dict]:
    result = env.step(action)
    if not isinstance(result, tuple):
        raise RPEXProtocolError("env.step() must return a tuple")
    if protocol == LOCAL_PROTOCOL:
        if len(result) != 5:
            raise RPEXProtocolError(
                f"Local Gymnasium env.step() must return five values, got {len(result)}"
            )
        observation, reward, terminated, truncated, info = result
        if not isinstance(info, dict):
            raise RPEXProtocolError("Gymnasium env.step() info value must be a dict")
        return (
            np.asarray(observation, dtype=np.float32),
            float(reward),
            bool(terminated),
            bool(truncated),
            info,
        )
    if protocol != LEGACY_PROTOCOL:
        raise ValueError(f"Unknown step protocol {protocol!r}")
    if len(result) == 5:
        raise RPEXProtocolError(
            "env.step() returned five values. A Gymnasium backend was loaded, "
            "but rpex_d4rl_v2_legacy requires Gym's four-value step API."
        )
    if len(result) != 4:
        raise RPEXProtocolError(
            f"Legacy env.step() must return four values, got {len(result)}"
        )
    observation, reward, done, info = result
    if not isinstance(info, dict):
        raise RPEXProtocolError("Legacy env.step() info value must be a dict")
    truncated = bool(info.get("TimeLimit.truncated", False))
    terminated = bool(done and not truncated)
    return (
        np.asarray(observation, dtype=np.float32),
        float(reward),
        terminated,
        truncated,
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
    evaluation_mode: str = "deterministic_diagnostic",
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
                    action_np = np.clip(
                        action_np, env.action_space.low, env.action_space.high
                    )
                elif action_execution_profile != "official_algorithm_behavior":
                    raise ValueError(
                        f"Unknown action_execution_profile {action_execution_profile!r}"
                    )
                action_np = action_np.astype(np.float32)
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
    if protocol == LOCAL_PROTOCOL:
        result["diagnostic_d4rl_reference_scaled_return_mean"] = result[
            "normalized_return_mean"
        ]
        result["diagnostic_d4rl_reference_scaled_return_std"] = result[
            "normalized_return_std"
        ]
    return result
