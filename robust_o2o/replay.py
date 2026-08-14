from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch

from .environment import Dataset


TensorBatch = Dict[str, torch.Tensor]


def _tensor_batch(dataset: Dataset, indices: np.ndarray, device: torch.device) -> TensorBatch:
    return {
        key: torch.as_tensor(value[indices], dtype=torch.float32, device=device)
        for key, value in dataset.items()
    }


class OfflineDataset:
    def __init__(self, dataset: Dataset, seed: int):
        self.dataset = dataset
        self.size = len(dataset["rewards"])
        self.rng = np.random.default_rng(seed)

    def sample(
        self,
        batch_size: int,
        device: torch.device,
        probabilities: Optional[np.ndarray] = None,
    ) -> TensorBatch:
        if batch_size <= 0:
            return {}
        indices = self.rng.choice(
            self.size, size=batch_size, replace=True, p=probabilities
        )
        return _tensor_batch(self.dataset, indices, device)


class ReplayBuffer:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        capacity: int,
        seed: int,
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
        self.rng = np.random.default_rng(seed)

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
        self, batch_size: int, device: torch.device, prioritized: bool = False
    ) -> TensorBatch:
        if batch_size <= 0:
            return {}
        if self.size < batch_size:
            raise ValueError(f"Replay contains {self.size} samples, need {batch_size}")
        probabilities = None
        if prioritized:
            probabilities = self.priorities[: self.size]
            probabilities = probabilities / probabilities.sum()
        indices = self.rng.choice(
            self.size, size=batch_size, replace=False, p=probabilities
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
        return result

    def update_priorities(self, indices: torch.Tensor, priorities: torch.Tensor) -> None:
        idx = indices.detach().cpu().numpy()
        values = priorities.detach().cpu().numpy()
        self.priorities[idx] = np.maximum(values, 1e-6)


def concatenate_batches(*batches: TensorBatch) -> TensorBatch:
    batches = tuple(batch for batch in batches if batch)
    if not batches:
        raise ValueError("At least one non-empty batch is required")
    keys = set.intersection(*(set(batch) for batch in batches))
    keys.discard("_indices")
    return {key: torch.cat([batch[key] for batch in batches], dim=0) for key in keys}


def mixed_batch(
    offline: OfflineDataset,
    online: ReplayBuffer,
    batch_size: int,
    offline_ratio: float,
    device: torch.device,
    prioritized_online: bool = False,
) -> TensorBatch:
    offline_count = int(round(batch_size * offline_ratio))
    online_count = batch_size - offline_count
    if online.size < online_count:
        raise ValueError(
            f"Online replay has {online.size} transitions; {online_count} are required"
        )
    offline_batch = offline.sample(offline_count, device)
    online_batch = online.sample(online_count, device, prioritized=prioritized_online)
    if offline_batch:
        offline_batch["_source"] = torch.zeros(offline_count, device=device)
    if online_batch:
        online_batch["_source"] = torch.ones(online_count, device=device)
    return concatenate_batches(offline_batch, online_batch)
