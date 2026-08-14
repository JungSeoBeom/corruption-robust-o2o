from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.distributions import Independent, Normal


LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0


def mlp(input_dim: int, hidden_dim: int, hidden_layers: int, output_dim: int) -> nn.Sequential:
    dims = [input_dim, *([hidden_dim] * hidden_layers), output_dim]
    layers = []
    for index in range(len(dims) - 2):
        layers.extend((nn.Linear(dims[index], dims[index + 1]), nn.ReLU()))
    layers.append(nn.Linear(dims[-2], dims[-1]))
    return nn.Sequential(*layers)


class TanhGaussianPolicy(nn.Module):
    """Reparameterized, action-bounded Gaussian policy shared by all agents."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        hidden_layers: int = 2,
        max_action: float = 1.0,
        deterministic: bool = False,
    ):
        super().__init__()
        self.trunk = mlp(state_dim, hidden_dim, hidden_layers, hidden_dim)
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)
        self.action_dim = action_dim
        self.max_action = float(max_action)
        self.deterministic_policy = deterministic

        nn.init.uniform_(self.mean.weight, -1e-3, 1e-3)
        nn.init.uniform_(self.mean.bias, -1e-3, 1e-3)
        nn.init.uniform_(self.log_std.weight, -1e-3, 1e-3)
        nn.init.uniform_(self.log_std.bias, -1e-3, 1e-3)

    def distribution(self, states: torch.Tensor) -> Normal:
        hidden = self.trunk(states)
        mean = self.mean(hidden)
        log_std = self.log_std(hidden).clamp(LOG_STD_MIN, LOG_STD_MAX)
        return Normal(mean, log_std.exp())

    def forward(
        self,
        states: torch.Tensor,
        deterministic: bool = False,
        need_log_prob: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, torch.Tensor]:
        dist = self.distribution(states)
        raw_action = dist.mean if (deterministic or self.deterministic_policy) else dist.rsample()
        squashed = torch.tanh(raw_action)
        log_prob = None
        if need_log_prob:
            log_prob = dist.log_prob(raw_action).sum(dim=-1)
            log_prob -= torch.log(1.0 - squashed.pow(2) + 1e-6).sum(dim=-1)
        action = squashed * self.max_action
        return action, log_prob, dist.mean, dist.stddev

    def log_prob(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        scaled = (actions / self.max_action).clamp(-0.999999, 0.999999)
        raw = torch.atanh(scaled)
        dist = self.distribution(states)
        result = dist.log_prob(raw).sum(dim=-1)
        return result - torch.log(1.0 - scaled.pow(2) + 1e-6).sum(dim=-1)

    @torch.no_grad()
    def act(self, states: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        return self(states, deterministic=deterministic)[0]


class ExpansionGaussianPolicy(nn.Module):
    """Gaussian used by the official PEX/RPEX online expansion policy."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        hidden_layers: int = 2,
        max_action: float = 1.0,
    ):
        super().__init__()
        self.trunk = mlp(state_dim, hidden_dim, hidden_layers, hidden_dim)
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        self.max_action = float(max_action)
        self.apply(self._initialize_linear)

    @staticmethod
    def _initialize_linear(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)

    def distribution(self, states: torch.Tensor) -> Independent:
        hidden = self.trunk(states)
        mean = torch.tanh(self.mean(hidden)) * self.max_action
        std = self.log_std.clamp(LOG_STD_MIN, LOG_STD_MAX).exp()
        return Independent(Normal(mean, std), 1)

    def log_prob(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return self.distribution(states).log_prob(actions)

    @torch.no_grad()
    def act(self, states: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        distribution = self.distribution(states)
        return distribution.mean if deterministic else distribution.sample()


class DeterministicPolicy(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        hidden_layers: int = 2,
        max_action: float = 1.0,
    ):
        super().__init__()
        self.net = mlp(state_dim, hidden_dim, hidden_layers, action_dim)
        self.action_dim = action_dim
        self.max_action = float(max_action)
        self.deterministic_policy = True

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.net(states)) * self.max_action

    @torch.no_grad()
    def act(self, states: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        return self(states)

    def log_prob(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        del states, actions
        raise RuntimeError("A deterministic policy does not define log_prob")


class ValueNetwork(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int = 256, hidden_layers: int = 2):
        super().__init__()
        self.net = mlp(state_dim, hidden_dim, hidden_layers, 1)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.net(states).squeeze(-1)


class QNetwork(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        hidden_layers: int = 2,
    ):
        super().__init__()
        self.net = mlp(state_dim + action_dim, hidden_dim, hidden_layers, 1)

    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        multiple = actions.ndim == 3 and states.ndim == 2
        if multiple:
            batch, count, action_dim = actions.shape
            states = states[:, None, :].expand(-1, count, -1).reshape(batch * count, -1)
            actions = actions.reshape(batch * count, action_dim)
        values = self.net(torch.cat((states, actions), dim=-1)).squeeze(-1)
        return values.reshape(batch, count) if multiple else values


class DoubleQNetwork(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        hidden_layers: int = 2,
    ):
        super().__init__()
        self.q1 = QNetwork(state_dim, action_dim, hidden_dim, hidden_layers)
        self.q2 = QNetwork(state_dim, action_dim, hidden_dim, hidden_layers)

    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return torch.stack((self.q1(states, actions), self.q2(states, actions)), dim=0)


class VectorizedLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, ensemble_size: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(ensemble_size, in_features, out_features))
        self.bias = nn.Parameter(torch.empty(ensemble_size, 1, out_features))
        self.ensemble_size = ensemble_size
        for index in range(ensemble_size):
            nn.init.kaiming_uniform_(self.weight[index], a=math.sqrt(5))
        bound = 1.0 / math.sqrt(in_features)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs @ self.weight + self.bias


class EnsembleQNetwork(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        hidden_layers: int = 2,
        num_critics: int = 5,
    ):
        super().__init__()
        modules = []
        input_dim = state_dim + action_dim
        for _ in range(hidden_layers):
            modules.extend(
                (VectorizedLinear(input_dim, hidden_dim, num_critics), nn.ReLU())
            )
            input_dim = hidden_dim
        modules.append(VectorizedLinear(input_dim, 1, num_critics))
        self.net = nn.Sequential(*modules)
        self.num_critics = num_critics
        last = self.net[-1]
        nn.init.uniform_(last.weight, -3e-3, 3e-3)
        nn.init.uniform_(last.bias, -3e-3, 3e-3)

    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        state_action = torch.cat((states, actions), dim=-1)
        repeated = state_action.unsqueeze(0).expand(self.num_critics, -1, -1)
        return self.net(repeated).squeeze(-1)


class DensityRatioNetwork(nn.Module):
    """Small positive network used by balanced replay in Off2OnRL."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        hidden_layers: int = 2,
    ):
        super().__init__()
        self.net = mlp(state_dim + action_dim, hidden_dim, hidden_layers, 1)

    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.softplus(
            self.net(torch.cat((states, actions), dim=-1)).squeeze(-1)
        )
