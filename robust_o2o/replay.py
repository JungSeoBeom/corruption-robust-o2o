from __future__ import annotations

import random
from typing import Dict, Optional

import numpy as np
import torch

from .dataset import assert_no_corruption_labels, learner_dataset_view
from .environment import Dataset


TensorBatch = Dict[str, torch.Tensor]
NUMPY_REPLAY_SAMPLING = "private_numpy_default_rng"
RPEX_OFFICIAL_REPLAY_SAMPLING = "rpex_official_global_rng"


def _tensor_batch(dataset: Dataset, indices: np.ndarray, device: torch.device) -> TensorBatch:
    return {
        key: torch.as_tensor(value[indices], dtype=torch.float32, device=device)
        for key, value in dataset.items()
    }


class OfflineDataset:
    def __init__(
        self,
        dataset: Dataset,
        seed: int,
        sampling_profile: str = NUMPY_REPLAY_SAMPLING,
    ):
        # Preprocessing keeps corruption diagnostics next to the artifact, but
        # the learner receives only ordinary transition fields.  In
        # particular, Cal-QL cannot use ``mc_calibration_valid`` as an oracle
        # corruption mask in a benchmark run.
        self.dataset = learner_dataset_view(dataset)
        self.size = len(dataset["rewards"])
        self.sampling_profile = sampling_profile
        if sampling_profile not in (
            NUMPY_REPLAY_SAMPLING,
            RPEX_OFFICIAL_REPLAY_SAMPLING,
        ):
            raise ValueError(f"Unknown offline replay sampling profile {sampling_profile!r}")
        self.rng = np.random.default_rng(seed)
        self.priorities = np.ones(self.size, dtype=np.float64)
        self.priority_updates = 0

    def sample(
        self,
        batch_size: int,
        device: torch.device,
        probabilities: Optional[np.ndarray] = None,
        prioritized: bool = False,
    ) -> TensorBatch:
        if batch_size <= 0:
            return {}
        if prioritized and probabilities is None:
            probabilities = _safe_probabilities(self.priorities)
        if self.sampling_profile == RPEX_OFFICIAL_REPLAY_SAMPLING:
            if prioritized or probabilities is not None:
                raise ValueError(
                    "official RPEX offline sampling does not support priorities"
                )
            # felix-thu/RPEX pex/utils/util.py::sample_batch uses the global
            # Torch stream and samples with replacement on the training device.
            torch_indices = torch.randint(
                low=0,
                high=self.size,
                size=(batch_size,),
                device=device,
            )
            indices = torch_indices.detach().cpu().numpy()
        else:
            indices = self.rng.choice(
                self.size, size=batch_size, replace=True, p=probabilities
            )
        result = _tensor_batch(self.dataset, indices, device)
        result["_indices"] = torch.as_tensor(indices, dtype=torch.long, device=device)
        result["_source"] = torch.zeros(batch_size, dtype=torch.long, device=device)
        assert_no_corruption_labels(result)
        return result

    def update_priorities(self, indices: torch.Tensor, priorities: torch.Tensor) -> None:
        idx = indices.detach().cpu().numpy().astype(np.int64, copy=False)
        values = priorities.detach().cpu().numpy().astype(np.float64, copy=False)
        self.priorities[idx] = np.maximum(values, 1e-12)
        self.priority_updates += int(len(idx))

    def priority_statistics(self) -> Dict[str, float]:
        return _priority_statistics(self.priorities, self.priority_updates)

    def state_dict(self) -> Dict[str, object]:
        return {
            "sampling_profile": self.sampling_profile,
            "rng_state": (
                self.rng.bit_generator.state
                if self.sampling_profile == NUMPY_REPLAY_SAMPLING
                else None
            ),
            "priorities": self.priorities.copy(),
            "priority_updates": self.priority_updates,
        }

    def load_state_dict(self, state: Dict[str, object]) -> None:
        saved_profile = state.get("sampling_profile", NUMPY_REPLAY_SAMPLING)
        if saved_profile != self.sampling_profile:
            raise ValueError("offline replay sampling profile changed across resume")
        if state.get("rng_state") is not None:
            self.rng.bit_generator.state = state["rng_state"]
        self.priorities[...] = np.asarray(state["priorities"], dtype=np.float64)
        self.priority_updates = int(state.get("priority_updates", 0))


class ReplayBuffer:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        capacity: int,
        seed: int,
        sampling_profile: str = NUMPY_REPLAY_SAMPLING,
    ):
        self.capacity = int(capacity)
        self.states = np.empty((capacity, state_dim), dtype=np.float32)
        self.actions = np.empty((capacity, action_dim), dtype=np.float32)
        self.rewards = np.empty(capacity, dtype=np.float32)
        self.next_states = np.empty((capacity, state_dim), dtype=np.float32)
        self.terminals = np.empty(capacity, dtype=np.float32)
        self.mc_returns = np.zeros(capacity, dtype=np.float32)
        self.priorities = np.ones(capacity, dtype=np.float64)
        self.position = 0
        self.size = 0
        self.sampling_profile = sampling_profile
        if sampling_profile not in (
            NUMPY_REPLAY_SAMPLING,
            RPEX_OFFICIAL_REPLAY_SAMPLING,
        ):
            raise ValueError(f"Unknown online replay sampling profile {sampling_profile!r}")
        self.rng = np.random.default_rng(seed)
        if sampling_profile == RPEX_OFFICIAL_REPLAY_SAMPLING:
            # Upstream ReplayMemory.__init__ resets the process-global Python
            # stream to args.seed immediately before online interaction.
            random.seed(seed)
        self.priority_updates = 0

    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        terminal: float,
        priority: float = 1.0,
    ) -> None:
        index = self.position
        self.states[index] = state
        self.actions[index] = action
        self.rewards[index] = reward
        self.next_states[index] = next_state
        self.terminals[index] = terminal
        self.priorities[index] = max(float(priority), 1e-6)
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(
        self,
        batch_size: int,
        device: torch.device,
        prioritized: bool = False,
        replace: bool = False,
    ) -> TensorBatch:
        if batch_size <= 0:
            return {}
        if self.size <= 0:
            raise ValueError("Replay contains no samples")
        if not replace and self.size < batch_size:
            raise ValueError(f"Replay contains {self.size} samples, need {batch_size}")
        probabilities = None
        if prioritized:
            probabilities = _safe_probabilities(self.priorities[: self.size])
        if self.sampling_profile == RPEX_OFFICIAL_REPLAY_SAMPLING:
            if prioritized:
                raise ValueError(
                    "official RPEX online replay sampling does not support priorities"
                )
            if replace:
                raise ValueError(
                    "official RPEX replay sampling is without replacement"
                )
            # ReplayMemory.sample uses random.sample(self.buffer, batch_size).
            indices = np.asarray(
                random.sample(range(self.size), batch_size), dtype=np.int64
            )
        else:
            indices = self.rng.choice(
                self.size, size=batch_size, replace=replace, p=probabilities
            )
        batch = {
            "observations": self.states[indices],
            "actions": self.actions[indices],
            "rewards": self.rewards[indices],
            "next_observations": self.next_states[indices],
            "terminals": self.terminals[indices],
            "mc_returns": self.mc_returns[indices],
        }
        result = _tensor_batch(batch, np.arange(batch_size), device)
        result["_indices"] = torch.as_tensor(indices, dtype=torch.long, device=device)
        result["_source"] = torch.ones(batch_size, dtype=torch.long, device=device)
        assert_no_corruption_labels(result)
        return result

    def update_priorities(self, indices: torch.Tensor, priorities: torch.Tensor) -> None:
        idx = indices.detach().cpu().numpy()
        values = priorities.detach().cpu().numpy()
        self.priorities[idx] = np.maximum(values, 1e-12)
        self.priority_updates += int(len(idx))

    def priority_statistics(self) -> Dict[str, float]:
        return _priority_statistics(
            self.priorities[: self.size], self.priority_updates
        )

    def state_dict(self) -> Dict[str, object]:
        size = self.size
        return {
            "capacity": self.capacity,
            "states": self.states[:size].copy(),
            "actions": self.actions[:size].copy(),
            "rewards": self.rewards[:size].copy(),
            "next_states": self.next_states[:size].copy(),
            "terminals": self.terminals[:size].copy(),
            "mc_returns": self.mc_returns[:size].copy(),
            "priorities": self.priorities[:size].copy(),
            "position": self.position,
            "size": size,
            "sampling_profile": self.sampling_profile,
            "rng_state": (
                self.rng.bit_generator.state
                if self.sampling_profile == NUMPY_REPLAY_SAMPLING
                else None
            ),
            "priority_updates": self.priority_updates,
        }

    def load_state_dict(self, state: Dict[str, object]) -> None:
        if int(state["capacity"]) != self.capacity:
            raise ValueError("resume replay capacity does not match resolved config")
        size = int(state["size"])
        for name in (
            "states",
            "actions",
            "rewards",
            "next_states",
            "terminals",
            "mc_returns",
            "priorities",
        ):
            getattr(self, name)[:size] = np.asarray(state[name])
        self.position = int(state["position"])
        self.size = size
        saved_profile = state.get("sampling_profile", NUMPY_REPLAY_SAMPLING)
        if saved_profile != self.sampling_profile:
            raise ValueError("online replay sampling profile changed across resume")
        if state.get("rng_state") is not None:
            self.rng.bit_generator.state = state["rng_state"]
        self.priority_updates = int(state.get("priority_updates", 0))


def _safe_probabilities(priorities: np.ndarray) -> np.ndarray:
    values = np.asarray(priorities, dtype=np.float64)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    values = np.maximum(values, 1e-12)
    total = float(values.sum())
    if not np.isfinite(total) or total <= 0.0:
        return np.full(len(values), 1.0 / len(values), dtype=np.float64)
    return values / total


def _priority_statistics(
    priorities: np.ndarray, number_of_updates: int
) -> Dict[str, float]:
    if len(priorities) == 0:
        return {
            "priority_min": 0.0,
            "priority_max": 0.0,
            "priority_mean": 0.0,
            "priority_std": 0.0,
            "priority_entropy": 0.0,
            "effective_sample_size": 0.0,
            "number_of_priority_updates": float(number_of_updates),
        }
    probabilities = _safe_probabilities(priorities)
    entropy = -np.sum(probabilities * np.log(probabilities + 1e-300))
    return {
        "priority_min": float(np.min(priorities)),
        "priority_max": float(np.max(priorities)),
        "priority_mean": float(np.mean(priorities)),
        "priority_std": float(np.std(priorities)),
        "priority_entropy": float(entropy),
        "effective_sample_size": float(1.0 / np.sum(np.square(probabilities))),
        "number_of_priority_updates": float(number_of_updates),
    }


def concatenate_batches(*batches: TensorBatch) -> TensorBatch:
    batches = tuple(batch for batch in batches if batch)
    if not batches:
        raise ValueError("At least one non-empty batch is required")
    keys = set.intersection(*(set(batch) for batch in batches))
    return {key: torch.cat([batch[key] for batch in batches], dim=0) for key in keys}


def mixed_batch(
    offline: OfflineDataset,
    online: ReplayBuffer,
    batch_size: int,
    offline_ratio: float,
    device: torch.device,
    prioritized_online: bool = False,
    prioritized_offline: bool = False,
    online_replace: bool = False,
) -> TensorBatch:
    offline_count = int(round(batch_size * offline_ratio))
    online_count = batch_size - offline_count
    if online_count > 0 and online.size <= 0:
        raise ValueError("Online replay contains no transitions")
    if not online_replace and online.size < online_count:
        raise ValueError(
            f"Online replay has {online.size} transitions; {online_count} are required"
        )
    offline_batch = offline.sample(
        offline_count, device, prioritized=prioritized_offline
    )
    online_batch = online.sample(
        online_count,
        device,
        prioritized=prioritized_online,
        replace=online_replace,
    )
    return concatenate_batches(offline_batch, online_batch)


def balanced_priority_batch(
    offline: OfflineDataset,
    online: ReplayBuffer,
    batch_size: int,
    device: torch.device,
) -> TensorBatch:
    """Sample the official Off2OnRL-style combined priority mass."""
    if online.size == 0:
        raise ValueError("Balanced replay requires at least one online transition")
    offline_mass = float(np.maximum(offline.priorities, 1e-12).sum())
    online_mass = float(
        np.maximum(online.priorities[: online.size], 1e-12).sum()
    )
    online_fraction = online_mass / (offline_mass + online_mass)
    online_count = int(round(batch_size * online_fraction))
    online_count = min(max(online_count, 1), batch_size - 1, online.size)
    offline_count = batch_size - online_count
    return concatenate_batches(
        offline.sample(offline_count, device, prioritized=True),
        online.sample(online_count, device, prioritized=True),
    )


def sample_pqe_update_batches(
    offline: OfflineDataset,
    online: ReplayBuffer,
    batch_size: int,
    offline_ratio: float,
    device: torch.device,
    prioritized_rl: bool,
) -> tuple[TensorBatch, TensorBatch, TensorBatch]:
    """Route uniform density data separately from the PQE RL replay batch."""
    if online.size < batch_size:
        raise ValueError(
            f"PQE density replay contains {online.size} transitions; "
            f"{batch_size} are required"
        )
    density_offline_batch = offline.sample(
        batch_size, device, prioritized=False
    )
    density_online_batch = online.sample(
        batch_size, device, prioritized=False
    )
    if prioritized_rl:
        rl_batch = balanced_priority_batch(offline, online, batch_size, device)
    else:
        rl_batch = mixed_batch(
            offline,
            online,
            batch_size,
            offline_ratio,
            device,
            prioritized_online=False,
            prioritized_offline=False,
        )
    return rl_batch, density_offline_batch, density_online_batch


def update_sample_priorities(
    offline: OfflineDataset,
    online: ReplayBuffer,
    batch: TensorBatch,
    priorities: torch.Tensor,
) -> Dict[str, float]:
    """Apply aligned density-ratio priorities to their original source rows."""
    if "_indices" not in batch or "_source" not in batch:
        raise RuntimeError("priority update requires replay indices and source metadata")
    indices = batch["_indices"].reshape(-1)
    source = batch["_source"].reshape(-1)
    values = priorities.reshape(-1)
    if not (len(indices) == len(source) == len(values)):
        raise RuntimeError("priority metadata is not aligned with the sampled batch")
    offline_mask = source == 0
    online_mask = source == 1
    if offline_mask.any():
        offline.update_priorities(indices[offline_mask], values[offline_mask])
    if online_mask.any():
        online.update_priorities(indices[online_mask], values[online_mask])
    offline_stats = offline.priority_statistics()
    online_stats = online.priority_statistics()
    return {
        **offline_stats,
        "effective_offline_sample_size": offline_stats["effective_sample_size"],
        "online_priority_std": online_stats["priority_std"],
        "number_of_priority_updates": offline_stats[
            "number_of_priority_updates"
        ]
        + online_stats["number_of_priority_updates"],
        "offline_sampling_fraction": float(offline_mask.float().mean().item()),
        "online_sampling_fraction": float(online_mask.float().mean().item()),
    }
