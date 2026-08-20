from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.distributions import Independent, Normal


LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0


def mlp(
    input_dim: int,
    hidden_dim: int,
    hidden_layers: int,
    output_dim: int,
    layer_norm: bool = False,
) -> nn.Sequential:
    dims = [input_dim, *([hidden_dim] * hidden_layers), output_dim]
    layers = []
    for index in range(len(dims) - 2):
        layers.append(nn.Linear(dims[index], dims[index + 1]))
        if layer_norm:
            layers.append(nn.LayerNorm(dims[index + 1]))
        layers.append(nn.ReLU())
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
        action_low: Optional[torch.Tensor] = None,
        action_high: Optional[torch.Tensor] = None,
        layer_norm: bool = False,
        wsrl_profile: bool = False,
    ):
        super().__init__()
        self.wsrl_profile = bool(wsrl_profile)
        if self.wsrl_profile:
            # WSRL's Flax MLP treats ``hidden_dims=[256, 256]`` as exactly two
            # Dense-LayerNorm-ReLU blocks.  The generic helper has a separate
            # output layer and would silently create a third affine transform.
            modules: list[nn.Module] = []
            input_dim = state_dim
            hidden_linears: list[nn.Linear] = []
            for _ in range(hidden_layers):
                linear = nn.Linear(input_dim, hidden_dim)
                hidden_linears.append(linear)
                modules.append(linear)
                if layer_norm:
                    modules.append(nn.LayerNorm(hidden_dim))
                modules.append(nn.ReLU())
                input_dim = hidden_dim
            self.trunk = nn.Sequential(*modules)
        else:
            self.trunk = mlp(
                state_dim,
                hidden_dim,
                hidden_layers,
                hidden_dim,
                layer_norm=layer_norm,
            )
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)
        self.action_dim = action_dim
        self.max_action = float(max_action)
        self.deterministic_policy = deterministic
        low = torch.full((action_dim,), -self.max_action) if action_low is None else torch.as_tensor(action_low, dtype=torch.float32)
        high = torch.full((action_dim,), self.max_action) if action_high is None else torch.as_tensor(action_high, dtype=torch.float32)
        if low.shape != (action_dim,) or high.shape != (action_dim,) or not torch.all(high > low):
            raise ValueError("action bounds must be finite vectors with high > low")
        self.register_buffer("action_scale", (high - low) / 2.0)
        self.register_buffer("action_bias", (high + low) / 2.0)

        if self.wsrl_profile:
            # locomotion_wsrl sets kernel_scale_final=1e-2 on the final hidden
            # Dense. All other Dense kernels use orthogonal(sqrt(2)).
            for index, linear in enumerate(hidden_linears):
                scale = 1e-2 if index == len(hidden_linears) - 1 else math.sqrt(2.0)
                nn.init.orthogonal_(linear.weight, gain=scale)
                nn.init.zeros_(linear.bias)
            for head in (self.mean, self.log_std):
                nn.init.orthogonal_(head.weight, gain=math.sqrt(2.0))
                nn.init.zeros_(head.bias)
        else:
            nn.init.uniform_(self.mean.weight, -1e-3, 1e-3)
            nn.init.uniform_(self.mean.bias, -1e-3, 1e-3)
            nn.init.uniform_(self.log_std.weight, -1e-3, 1e-3)
            nn.init.uniform_(self.log_std.bias, -1e-3, 1e-3)

    def distribution(self, states: torch.Tensor) -> Normal:
        hidden = self.trunk(states)
        mean = self.mean(hidden)
        log_std = self.log_std(hidden)
        if self.wsrl_profile:
            return Normal(mean, log_std.exp().clamp(1e-5, 10.0))
        return Normal(mean, log_std.clamp(LOG_STD_MIN, LOG_STD_MAX).exp())

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
            jacobian = self.action_scale * (1.0 - squashed.pow(2))
            log_prob -= torch.log(jacobian + 1e-6).sum(dim=-1)
        action = self.action_bias + self.action_scale * squashed
        return action, log_prob, dist.mean, dist.stddev

    def log_prob(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        scaled = ((actions - self.action_bias) / self.action_scale).clamp(
            -0.999999, 0.999999
        )
        raw = torch.atanh(scaled)
        dist = self.distribution(states)
        result = dist.log_prob(raw).sum(dim=-1)
        jacobian = self.action_scale * (1.0 - scaled.pow(2))
        return result - torch.log(jacobian + 1e-6).sum(dim=-1)

    @torch.no_grad()
    def act(self, states: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        return self(states, deterministic=deterministic)[0]


class ExpansionGaussianPolicy(nn.Module):
    """Safe bounded PEX/RPEX Gaussian with a corrected transformed density."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        hidden_layers: int = 2,
        max_action: float = 1.0,
        action_low: Optional[torch.Tensor] = None,
        action_high: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.trunk = mlp(state_dim, hidden_dim, hidden_layers, hidden_dim)
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        self.action_dim = action_dim
        self.max_action = float(max_action)
        low = torch.full((action_dim,), -self.max_action) if action_low is None else torch.as_tensor(action_low, dtype=torch.float32)
        high = torch.full((action_dim,), self.max_action) if action_high is None else torch.as_tensor(action_high, dtype=torch.float32)
        if low.shape != (action_dim,) or high.shape != (action_dim,) or not torch.all(high > low):
            raise ValueError("action bounds must be finite vectors with high > low")
        self.register_buffer("action_scale", (high - low) / 2.0)
        self.register_buffer("action_bias", (high + low) / 2.0)
        self.apply(self._initialize_linear)

    @staticmethod
    def _initialize_linear(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)

    def distribution(self, states: torch.Tensor) -> Independent:
        hidden = self.trunk(states)
        mean = self.mean(hidden)
        std = self.log_std.clamp(LOG_STD_MIN, LOG_STD_MAX).exp()
        return Independent(Normal(mean, std), 1)

    def log_prob(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        normalized = ((actions - self.action_bias) / self.action_scale).clamp(
            -0.999999, 0.999999
        )
        raw = torch.atanh(normalized)
        jacobian = self.action_scale * (1.0 - normalized.square())
        return self.distribution(states).log_prob(raw) - torch.log(
            jacobian + 1e-6
        ).sum(dim=-1)

    def forward(
        self,
        states: torch.Tensor,
        deterministic: bool = False,
        need_log_prob: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, torch.Tensor]:
        distribution = self.distribution(states)
        raw = distribution.mean if deterministic else distribution.rsample()
        normalized = torch.tanh(raw)
        action = self.action_bias + self.action_scale * normalized
        log_prob = None
        if need_log_prob:
            jacobian = self.action_scale * (1.0 - normalized.square())
            log_prob = distribution.log_prob(raw) - torch.log(
                jacobian + 1e-6
            ).sum(dim=-1)
        return action, log_prob, distribution.mean, distribution.stddev

    @torch.no_grad()
    def act(self, states: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        return self(states, deterministic=deterministic)[0]


class OfficialRPEXGaussianPolicy(nn.Module):
    """felix-thu/RPEX GaussianPolicy with ``scale_distribution=False``.

    The mean is bounded before constructing an ordinary Normal distribution;
    samples are deliberately not squashed and log-probability has no Jacobian.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        hidden_layers: int = 2,
        max_action: float = 1.0,
    ):
        super().__init__()
        modules: list[nn.Module] = []
        input_dim = state_dim
        for _ in range(hidden_layers):
            modules.extend((nn.Linear(input_dim, hidden_dim), nn.ReLU()))
            input_dim = hidden_dim
        self.trunk = nn.Sequential(*modules)
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        self.action_dim = action_dim
        self.max_action = float(max_action)
        self.apply(ExpansionGaussianPolicy._initialize_linear)

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


# Checkpoint/source compatibility name for results produced before fidelity
# profiles were introduced.  New code always records OfficialRPEXGaussianPolicy.
LegacyExpansionGaussianPolicy = OfficialRPEXGaussianPolicy


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


class EnsembleLayerNorm(nn.Module):
    def __init__(self, hidden_dim: int, ensemble_size: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ensemble_size, 1, hidden_dim))
        self.bias = nn.Parameter(torch.zeros(ensemble_size, 1, hidden_dim))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        mean = inputs.mean(dim=-1, keepdim=True)
        variance = inputs.var(dim=-1, keepdim=True, unbiased=False)
        return (inputs - mean) * torch.rsqrt(variance + 1e-5) * self.weight + self.bias


class EnsembleQNetwork(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        hidden_layers: int = 2,
        num_critics: int = 5,
        layer_norm: bool = False,
        wsrl_profile: bool = False,
        rpex_profile: bool = False,
    ):
        super().__init__()
        modules = []
        input_dim = state_dim + action_dim
        hidden_linears = []
        for _ in range(hidden_layers):
            linear = VectorizedLinear(input_dim, hidden_dim, num_critics)
            hidden_linears.append(linear)
            modules.append(linear)
            if layer_norm:
                modules.append(EnsembleLayerNorm(hidden_dim, num_critics))
            modules.append(nn.ReLU())
            input_dim = hidden_dim
        modules.append(VectorizedLinear(input_dim, 1, num_critics))
        self.net = nn.Sequential(*modules)
        self.num_critics = num_critics
        last = self.net[-1]
        if wsrl_profile:
            for layer_index, linear in enumerate(hidden_linears):
                scale = (
                    1e-2
                    if layer_index == len(hidden_linears) - 1
                    else math.sqrt(2.0)
                )
                for critic_index in range(num_critics):
                    nn.init.orthogonal_(linear.weight[critic_index], gain=scale)
                nn.init.zeros_(linear.bias)
            for critic_index in range(num_critics):
                nn.init.orthogonal_(last.weight[critic_index], gain=math.sqrt(2.0))
            nn.init.zeros_(last.bias)
        else:
            if rpex_profile:
                for linear in hidden_linears:
                    nn.init.constant_(linear.bias, 0.1)
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
