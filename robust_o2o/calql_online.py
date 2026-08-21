from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional

import numpy as np


CALQL_TRANSITION_KEYS = (
    "observations",
    "actions",
    "rewards",
    "next_observations",
    "terminals",
    "mc_returns",
)


def discounted_return_to_go(
    rewards: np.ndarray | list[float],
    discount: float,
) -> np.ndarray:
    """Return-to-go for one completed trajectory.

    The caller must pass the rewards actually stored by the learner.  In the
    corruption benchmark that means post-corruption rewards, not the clean
    environment rewards retained for diagnostics.
    """

    if not 0.0 <= float(discount) <= 1.0:
        raise ValueError("discount must be in [0, 1]")
    values = np.asarray(rewards, dtype=np.float64).reshape(-1)
    if not np.all(np.isfinite(values)):
        raise ValueError("trajectory rewards must be finite")
    returns = np.empty(len(values), dtype=np.float32)
    running = 0.0
    for index in range(len(values) - 1, -1, -1):
        running = float(values[index]) + float(discount) * running
        returns[index] = running
    return returns


def episodic_return_to_go(
    rewards: np.ndarray,
    terminals: np.ndarray,
    timeouts: np.ndarray,
    discount: float,
) -> np.ndarray:
    """Compute returns without leaking across terminal or timeout boundaries."""

    if not 0.0 <= float(discount) <= 1.0:
        raise ValueError("discount must be in [0, 1]")
    rewards = np.asarray(rewards, dtype=np.float64).reshape(-1)
    terminals = np.asarray(terminals, dtype=bool).reshape(-1)
    timeouts = np.asarray(timeouts, dtype=bool).reshape(-1)
    if not (len(rewards) == len(terminals) == len(timeouts)):
        raise ValueError("rewards, terminals, and timeouts must have equal length")
    if not np.all(np.isfinite(rewards)):
        raise ValueError("rewards must be finite")
    result = np.empty(len(rewards), dtype=np.float32)
    running = 0.0
    for index in range(len(rewards) - 1, -1, -1):
        if index == len(rewards) - 1 or terminals[index] or timeouts[index]:
            running = 0.0
        running = float(rewards[index]) + float(discount) * running
        result[index] = running
    return result


def dynamic_offline_ratio(offline_size: int, online_size: int) -> float:
    """Source-derived Cal-QL offline fraction for a completed online replay."""

    if offline_size < 0 or online_size < 0:
        raise ValueError("replay sizes cannot be negative")
    total = int(offline_size) + int(online_size)
    if total <= 0:
        raise ValueError("at least one replay must contain data")
    return float(offline_size) / float(total)


def dynamic_batch_counts(
    batch_size: int,
    offline_size: int,
    online_size: int,
) -> tuple[int, int, float]:
    """Return the exact floor-based split used by the pinned Cal-QL loop."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    ratio = dynamic_offline_ratio(offline_size, online_size)
    offline_count = int(batch_size * ratio)
    online_count = batch_size - offline_count
    return offline_count, online_count, ratio


@dataclass(frozen=True)
class CompletedCalQLTrajectory:
    """A complete, calibration-safe trajectory ready for online replay."""

    batch: Dict[str, np.ndarray]

    @property
    def length(self) -> int:
        return int(len(self.batch["rewards"]))

    def update_count(self, utd_ratio: int) -> int:
        if utd_ratio < 0:
            raise ValueError("utd_ratio cannot be negative")
        return self.length * int(utd_ratio)


class CalQLTrajectoryAccumulator:
    """Hold online transitions until their exact MC returns are available.

    Transitions must be appended *after* replay-only corruption.  Nothing is
    emitted before a terminal or timeout, preventing fabricated zero returns
    from entering a Cal-QL update.
    """

    def __init__(self, discount: float):
        if not 0.0 <= float(discount) <= 1.0:
            raise ValueError("discount must be in [0, 1]")
        self.discount = float(discount)
        self._pending: list[Dict[str, object]] = []
        self.completed_trajectories = 0
        self.completed_transitions = 0

    @property
    def pending_episode_length(self) -> int:
        return len(self._pending)

    @property
    def online_mc_return_valid_fraction(self) -> float:
        return 1.0 if self.completed_transitions > 0 else 0.0

    def append(
        self,
        *,
        observation: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_observation: np.ndarray,
        terminal: bool,
        timeout: bool = False,
    ) -> Optional[CompletedCalQLTrajectory]:
        """Append one post-corruption transition and emit only at a boundary."""

        transition = {
            "observations": np.asarray(observation, dtype=np.float32).copy(),
            "actions": np.asarray(action, dtype=np.float32).copy(),
            "rewards": float(reward),
            "next_observations": np.asarray(next_observation, dtype=np.float32).copy(),
            # Time-limit truncation is an MC boundary but not an MDP terminal
            # for the Bellman backup, matching D4RL qlearning semantics.
            "terminals": float(bool(terminal)),
        }
        if not all(
            np.all(np.isfinite(value))
            for value in (
                transition["observations"],
                transition["actions"],
                transition["next_observations"],
            )
        ) or not np.isfinite(transition["rewards"]):
            raise ValueError("online Cal-QL transition contains non-finite values")
        self._pending.append(transition)
        if not (terminal or timeout):
            return None
        return self._complete_pending()

    def _complete_pending(self) -> CompletedCalQLTrajectory:
        if not self._pending:
            raise RuntimeError("cannot complete an empty trajectory")
        rewards = np.asarray(
            [transition["rewards"] for transition in self._pending],
            dtype=np.float32,
        )
        batch: Dict[str, np.ndarray] = {
            "observations": np.stack(
                [transition["observations"] for transition in self._pending]
            ).astype(np.float32, copy=False),
            "actions": np.stack(
                [transition["actions"] for transition in self._pending]
            ).astype(np.float32, copy=False),
            "rewards": rewards,
            "next_observations": np.stack(
                [transition["next_observations"] for transition in self._pending]
            ).astype(np.float32, copy=False),
            "terminals": np.asarray(
                [transition["terminals"] for transition in self._pending],
                dtype=np.float32,
            ),
            "mc_returns": discounted_return_to_go(rewards, self.discount),
        }
        completed = CompletedCalQLTrajectory(batch=batch)
        self.completed_trajectories += 1
        self.completed_transitions += completed.length
        self._pending.clear()
        return completed

    def metadata(self) -> Dict[str, float | int]:
        return {
            "completed_online_trajectories": self.completed_trajectories,
            "completed_online_transitions": self.completed_transitions,
            "pending_episode_length": self.pending_episode_length,
            "online_mc_return_valid_fraction": (self.online_mc_return_valid_fraction),
        }

    def state_dict(self) -> Dict[str, object]:
        return {
            "discount": self.discount,
            "pending": [
                {
                    key: value.copy() if isinstance(value, np.ndarray) else value
                    for key, value in transition.items()
                }
                for transition in self._pending
            ],
            "completed_trajectories": self.completed_trajectories,
            "completed_transitions": self.completed_transitions,
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        if not np.isclose(float(state["discount"]), self.discount):
            raise ValueError("Cal-QL trajectory discount changed across resume")
        pending = state.get("pending", [])
        if not isinstance(pending, list):
            raise TypeError("pending Cal-QL trajectory state must be a list")
        self._pending = []
        for transition in pending:
            if not isinstance(transition, Mapping):
                raise TypeError("pending Cal-QL transition must be a mapping")
            self._pending.append(
                {
                    "observations": np.asarray(
                        transition["observations"], dtype=np.float32
                    ).copy(),
                    "actions": np.asarray(
                        transition["actions"], dtype=np.float32
                    ).copy(),
                    "rewards": float(transition["rewards"]),
                    "next_observations": np.asarray(
                        transition["next_observations"], dtype=np.float32
                    ).copy(),
                    "terminals": float(transition["terminals"]),
                }
            )
        self.completed_trajectories = int(state.get("completed_trajectories", 0))
        self.completed_transitions = int(state.get("completed_transitions", 0))
