from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn

from ..replay import TensorBatch


def soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    with torch.no_grad():
        for target_parameter, parameter in zip(target.parameters(), source.parameters()):
            target_parameter.data.mul_(1.0 - tau).add_(parameter.data, alpha=tau)


class BaseAgent(nn.Module):
    def __init__(self, device: torch.device):
        super().__init__()
        self.device = device
        self.online_phase = False
        self.total_updates = 0

    def update(self, batch: TensorBatch) -> Dict[str, float]:
        raise NotImplementedError

    def select_action(
        self,
        state: torch.Tensor,
        evaluate: bool = False,
        evaluation_mode: str = "deterministic_diagnostic",
    ) -> torch.Tensor:
        raise NotImplementedError

    def begin_online(self) -> None:
        self.online_phase = True

    def checkpoint_state(self) -> Dict[str, Any]:
        return {
            "model": self.state_dict(),
            "optimizers": self.optimizer_state(),
            "online_phase": self.online_phase,
            "total_updates": self.total_updates,
        }

    def load_checkpoint_state(self, state: Dict[str, Any]) -> None:
        self.load_state_dict(state["model"], strict=False)
        self.load_optimizer_state(state.get("optimizers", {}))
        self.online_phase = bool(state.get("online_phase", False))
        self.total_updates = int(state.get("total_updates", 0))

    def optimizer_state(self) -> Dict[str, Any]:
        return {}

    def load_optimizer_state(self, state: Dict[str, Any]) -> None:
        del state
