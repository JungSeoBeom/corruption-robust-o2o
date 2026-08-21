from __future__ import annotations

import copy
import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from ..config import ExperimentConfig
from ..replay import TensorBatch
from .base import BaseAgent, gradient_norm, soft_update


class CalQLTanhGaussianPolicy(nn.Module):
    """Pinned Cal-QL policy: two orthogonal hidden layers and one joint head."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int,
        hidden_layers: int,
        max_action: float,
    ) -> None:
        super().__init__()
        if hidden_layers != 2:
            raise ValueError("Cal-QL requires exactly two hidden layers")
        modules: list[nn.Module] = []
        input_dim = state_dim
        for _ in range(hidden_layers):
            linear = nn.Linear(input_dim, hidden_dim)
            nn.init.orthogonal_(linear.weight, gain=math.sqrt(2.0))
            nn.init.zeros_(linear.bias)
            modules.extend((linear, nn.ReLU()))
            input_dim = hidden_dim
        self.trunk = nn.Sequential(*modules)
        self.output = nn.Linear(hidden_dim, 2 * action_dim)
        nn.init.orthogonal_(self.output.weight, gain=1e-2)
        nn.init.zeros_(self.output.bias)
        # The pinned Flax policy represents these two values as trainable
        # Scalar modules rather than folding them into the output bias.
        self.log_std_multiplier = nn.Parameter(torch.ones(()))
        self.log_std_offset = nn.Parameter(torch.full((), -1.0))
        self.action_dim = action_dim
        self.max_action = float(max_action)
        self.register_buffer("action_scale", torch.full((action_dim,), max_action))
        self.register_buffer("action_bias", torch.zeros(action_dim))

    def distribution(self, states: torch.Tensor) -> Normal:
        mean, raw_log_std = self.output(self.trunk(states)).chunk(2, dim=-1)
        log_std = (self.log_std_multiplier * raw_log_std + self.log_std_offset).clamp(
            -20.0, 2.0
        )
        return Normal(mean, log_std.exp())

    def forward(
        self,
        states: torch.Tensor,
        deterministic: bool = False,
        need_log_prob: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor]:
        distribution = self.distribution(states)
        raw_action = distribution.mean if deterministic else distribution.rsample()
        normalized = torch.tanh(raw_action)
        action = self.action_bias + self.action_scale * normalized
        log_prob = None
        if need_log_prob:
            log_prob = distribution.log_prob(raw_action).sum(dim=-1)
            jacobian = self.action_scale * (1.0 - normalized.square())
            log_prob = log_prob - torch.log(jacobian + 1e-6).sum(dim=-1)
        return action, log_prob, distribution.mean, distribution.stddev

    def log_prob(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        normalized = ((actions - self.action_bias) / self.action_scale).clamp(
            -0.999999, 0.999999
        )
        raw_action = torch.atanh(normalized)
        distribution = self.distribution(states)
        jacobian = self.action_scale * (1.0 - normalized.square())
        return distribution.log_prob(raw_action).sum(dim=-1) - torch.log(
            jacobian + 1e-6
        ).sum(dim=-1)

    @torch.no_grad()
    def act(self, states: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        return self(states, deterministic=deterministic)[0]


class CalQLQNetwork(nn.Module):
    """Pinned Cal-QL two-hidden-layer orthogonal Q function."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int,
        hidden_layers: int,
    ) -> None:
        super().__init__()
        if hidden_layers != 2:
            raise ValueError("Cal-QL requires exactly two hidden layers")
        modules: list[nn.Module] = []
        input_dim = state_dim + action_dim
        for _ in range(hidden_layers):
            linear = nn.Linear(input_dim, hidden_dim)
            nn.init.orthogonal_(linear.weight, gain=math.sqrt(2.0))
            nn.init.zeros_(linear.bias)
            modules.extend((linear, nn.ReLU()))
            input_dim = hidden_dim
        output = nn.Linear(hidden_dim, 1)
        nn.init.orthogonal_(output.weight, gain=1e-2)
        nn.init.zeros_(output.bias)
        modules.append(output)
        self.net = nn.Sequential(*modules)

    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        multiple_actions = actions.ndim == 3 and states.ndim == 2
        if multiple_actions:
            batch_size, action_count, action_dim = actions.shape
            states = (
                states[:, None, :]
                .expand(-1, action_count, -1)
                .reshape(batch_size * action_count, -1)
            )
            actions = actions.reshape(batch_size * action_count, action_dim)
        values = self.net(torch.cat((states, actions), dim=-1)).squeeze(-1)
        if multiple_actions:
            return values.reshape(batch_size, action_count)
        return values


def calql_td_target(
    rewards: torch.Tensor,
    terminals: torch.Tensor,
    next_q: torch.Tensor,
    next_log_prob: torch.Tensor,
    discount: float,
    alpha: torch.Tensor,
    backup_entropy: bool,
) -> torch.Tensor:
    backed_up = next_q
    if backup_entropy:
        backed_up = backed_up - alpha.detach() * next_log_prob
    return rewards + (1.0 - terminals) * discount * backed_up


def calql_max_target_backup(
    q1_candidates: torch.Tensor,
    q2_candidates: torch.Tensor,
    log_prob_candidates: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Select the action maximizing the clipped-double-Q target."""
    candidates = torch.minimum(q1_candidates, q2_candidates)
    best = candidates.argmax(dim=1, keepdim=True)
    return (
        candidates.gather(1, best).squeeze(1),
        log_prob_candidates.gather(1, best).squeeze(1),
    )


class CalQLAgent(BaseAgent):
    """Source-aligned Cal-QL locomotion adaptation.

    Calibration is deliberately phase-independent: completed online samples
    carry exact trajectory returns and receive the same current/next proposal
    lower bound as offline samples.
    """

    def __init__(
        self,
        config: ExperimentConfig,
        state_dim: int,
        action_dim: int,
        max_action: float,
        device: torch.device,
    ):
        super().__init__(device)
        self.config = config
        self.actor = CalQLTanhGaussianPolicy(
            state_dim,
            action_dim,
            config.hidden_dim,
            2,
            max_action,
        )
        self.q1 = CalQLQNetwork(state_dim, action_dim, config.hidden_dim, 2)
        self.q2 = CalQLQNetwork(state_dim, action_dim, config.hidden_dim, 2)
        self.target_q1 = copy.deepcopy(self.q1).requires_grad_(False)
        self.target_q2 = copy.deepcopy(self.q2).requires_grad_(False)
        self.log_alpha = torch.nn.Parameter(torch.zeros(1))
        self.to(device)
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=config.actor_learning_rate
        )
        self.q1_optimizer = torch.optim.Adam(
            self.q1.parameters(), lr=config.critic_learning_rate
        )
        self.q2_optimizer = torch.optim.Adam(
            self.q2.parameters(), lr=config.critic_learning_rate
        )
        self.alpha_optimizer = torch.optim.Adam(
            [self.log_alpha], lr=config.temperature_learning_rate
        )

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    def select_action(
        self,
        state: torch.Tensor,
        evaluate: bool = False,
        evaluation_mode: str = "deterministic_diagnostic",
    ) -> torch.Tensor:
        del evaluation_mode
        single = state.ndim == 1
        states = state.unsqueeze(0) if single else state
        action = self.actor.act(states, deterministic=evaluate)
        return action.squeeze(0) if single else action

    def _sample_many(
        self, states: torch.Tensor, count: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = states.shape[0]
        expanded = (
            states[:, None, :].expand(-1, count, -1).reshape(batch_size * count, -1)
        )
        actions, log_prob, _, _ = self.actor(expanded, need_log_prob=True)
        return (
            actions.reshape(batch_size, count, -1),
            log_prob.reshape(batch_size, count),
        )

    def _cql_loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        next_states: torch.Tensor,
        mc_returns: torch.Tensor,
        mc_valid: torch.Tensor,
        q1_data: torch.Tensor,
        q2_data: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        batch_size, action_dim = actions.shape
        action_count = self.config.cql_n_actions
        random_actions = torch.empty(
            batch_size, action_count, action_dim, device=states.device
        ).uniform_(-self.actor.max_action, self.actor.max_action)
        current_actions, current_log_prob = self._sample_many(states, action_count)
        next_actions, next_log_prob = self._sample_many(next_states, action_count)

        q1_random = self.q1(states, random_actions)
        q2_random = self.q2(states, random_actions)
        q1_current = self.q1(states, current_actions.detach())
        q2_current = self.q2(states, current_actions.detach())
        # Pinned Cal-QL generates proposals at s' but evaluates them at s.
        q1_next = self.q1(states, next_actions.detach())
        q2_next = self.q2(states, next_actions.detach())

        lower = mc_returns.reshape(-1, 1)
        valid = mc_valid.bool().reshape(-1, 1)
        current_rates = (
            torch.stack(
                tuple(
                    values[valid.expand_as(values)]
                    .lt(lower.expand_as(values)[valid.expand_as(values)])
                    .float()
                    .mean()
                    for values in (q1_current, q2_current)
                )
            )
            if valid.any()
            else states.new_zeros(2)
        )
        next_rates = (
            torch.stack(
                tuple(
                    values[valid.expand_as(values)]
                    .lt(lower.expand_as(values)[valid.expand_as(values)])
                    .float()
                    .mean()
                    for values in (q1_next, q2_next)
                )
            )
            if valid.any()
            else states.new_zeros(2)
        )

        calibrate = bool(getattr(self.config, "enable_calql", True)) and (
            self.config.calibration_mask_mode != "disabled"
        )
        if calibrate:
            q1_current = torch.where(
                valid, torch.maximum(q1_current, lower), q1_current
            )
            q2_current = torch.where(
                valid, torch.maximum(q2_current, lower), q2_current
            )
            q1_next = torch.where(valid, torch.maximum(q1_next, lower), q1_next)
            q2_next = torch.where(valid, torch.maximum(q2_next, lower), q2_next)

        random_density = math.log((0.5 / self.actor.max_action) ** action_dim)
        temperature = self.config.cql_temperature

        def conservative_difference(
            q_random: torch.Tensor,
            q_next: torch.Tensor,
            q_current: torch.Tensor,
            q_data: torch.Tensor,
        ) -> torch.Tensor:
            candidates = torch.cat(
                (
                    q_random - random_density,
                    q_next - next_log_prob.detach(),
                    q_current - current_log_prob.detach(),
                ),
                dim=1,
            )
            return (
                temperature * torch.logsumexp(candidates / temperature, dim=1) - q_data
            )

        cql_q1 = conservative_difference(q1_random, q1_next, q1_current, q1_data)
        cql_q2 = conservative_difference(q2_random, q2_next, q2_current, q2_data)
        weight = (
            self.config.cql_alpha_online if self.online_phase else self.config.cql_alpha
        )
        loss = weight * (cql_q1.mean() + cql_q2.mean())
        current_rate = current_rates.mean()
        next_rate = next_rates.mean()
        bound_rate = torch.stack((current_rate, next_rate)).mean()
        return loss, {
            "calibration_bound_rate": float(bound_rate.item()),
            "calql_current_calibration_bound_rate": float(current_rate.item()),
            "calql_next_calibration_bound_rate": float(next_rate.item()),
            "calql_calibration_enabled": float(calibrate),
            "online_calibration_bound_rate": float(
                bound_rate.item() if self.online_phase and calibrate else 0.0
            ),
            "cql_q1_diff": float(cql_q1.mean().item()),
            "cql_q2_diff": float(cql_q2.mean().item()),
            "mc_calibration_valid_fraction": float(mc_valid.float().mean().item()),
            "cql_weight": float(weight),
        }

    def update(self, batch: TensorBatch) -> Dict[str, float]:
        states = batch["observations"]
        actions = batch["actions"]
        rewards = batch["rewards"].reshape(-1)
        next_states = batch["next_observations"]
        terminals = batch["terminals"].reshape(-1)
        if "mc_returns" not in batch:
            raise ValueError(
                "Cal-QL requires exact mc_returns; incomplete online episodes "
                "must not be used for updates"
            )
        mc_returns = batch["mc_returns"].reshape(-1)
        if mc_returns.shape != rewards.shape or not torch.isfinite(mc_returns).all():
            raise ValueError("Cal-QL mc_returns must be finite and match rewards")
        if self.config.calibration_mask_mode == "all":
            mc_valid = torch.ones_like(mc_returns)
        elif self.config.calibration_mask_mode == "oracle_exclude_corrupted":
            if "mc_calibration_valid" not in batch:
                raise ValueError(
                    "oracle calibration mode requires mc_calibration_valid"
                )
            mc_valid = batch["mc_calibration_valid"].reshape(-1)
        else:
            mc_valid = torch.zeros_like(mc_returns)

        new_actions, log_prob, _, policy_std = self.actor(states, need_log_prob=True)
        alpha_for_losses = self.alpha.detach()
        alpha_loss = -(
            self.log_alpha * (log_prob + float(self.config.target_entropy)).detach()
        ).mean()

        # The pinned JAX implementation always uses the SAC policy objective;
        # BC warmup belongs only to a historical ablation.
        q_new = torch.minimum(
            self.q1(states, new_actions), self.q2(states, new_actions)
        )
        actor_loss = (alpha_for_losses * log_prob - q_new).mean()

        # Build every loss from the same pre-update parameter snapshot, as in
        # the pinned JAX multi-gradient step. Optimizers are applied only after
        # these forward passes have completed.
        with torch.no_grad():
            if self.config.cql_max_target_backup:
                next_actions, next_log_probs = self._sample_many(
                    next_states, self.config.cql_n_actions
                )
                next_q, next_log_prob = calql_max_target_backup(
                    self.target_q1(next_states, next_actions),
                    self.target_q2(next_states, next_actions),
                    next_log_probs,
                )
            else:
                next_actions, next_log_prob, _, _ = self.actor(
                    next_states, need_log_prob=True
                )
                next_q = torch.minimum(
                    self.target_q1(next_states, next_actions),
                    self.target_q2(next_states, next_actions),
                )
            td_target = calql_td_target(
                rewards,
                terminals,
                next_q,
                next_log_prob,
                self.config.discount,
                alpha_for_losses,
                self.config.backup_entropy,
            )
        q1_data = self.q1(states, actions)
        q2_data = self.q2(states, actions)
        td_loss = F.mse_loss(q1_data, td_target) + F.mse_loss(q2_data, td_target)
        cql_loss, cql_metrics = self._cql_loss(
            states,
            actions,
            next_states,
            mc_returns,
            mc_valid,
            q1_data,
            q2_data,
        )
        critic_loss = td_loss + cql_loss

        self.alpha_optimizer.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.alpha_optimizer.step()

        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        actor_grad_norm = (
            gradient_norm(self.actor.parameters())
            if self.config.max_grad_norm is None
            else torch.nn.utils.clip_grad_norm_(
                self.actor.parameters(), self.config.max_grad_norm
            )
        )
        actor_grad_norm_after = gradient_norm(self.actor.parameters())
        self.actor_optimizer.step()

        # actor_loss also evaluates Q(s, pi(s)); discard those policy-objective
        # Q gradients before applying the dedicated Bellman+CQL critic loss.
        self.q1_optimizer.zero_grad(set_to_none=True)
        self.q2_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        critic_parameters = list(self.q1.parameters()) + list(self.q2.parameters())
        critic_grad_norm = (
            gradient_norm(critic_parameters)
            if self.config.max_grad_norm is None
            else torch.nn.utils.clip_grad_norm_(
                critic_parameters, self.config.max_grad_norm
            )
        )
        critic_grad_norm_after = gradient_norm(critic_parameters)
        self.q1_optimizer.step()
        self.q2_optimizer.step()
        soft_update(self.target_q1, self.q1, self.config.target_update_rate)
        soft_update(self.target_q2, self.q2, self.config.target_update_rate)
        self.total_updates += 1
        self.actor_updates += 1
        self.critic_updates += 1
        self.temperature_updates += 1
        return {
            "critic_loss": float(critic_loss.item()),
            "td_loss": float(td_loss.item()),
            "cql_loss": float(cql_loss.item()),
            "actor_loss": float(actor_loss.item()),
            "alpha_loss": float(alpha_loss.item()),
            "alpha": float(self.alpha.item()),
            "bellman_loss": float(td_loss.item()),
            "cql_penalty": float(cql_loss.item()),
            "cql_to_bellman_ratio": float(
                cql_loss.detach().abs().div(td_loss.detach().abs() + 1e-8).item()
            ),
            "temperature_alpha": float(self.alpha.item()),
            "entropy": float((-log_prob.detach()).mean().item()),
            "gradient_norm_actor": float(actor_grad_norm.item()),
            "gradient_norm_critic": float(critic_grad_norm.item()),
            "actor_grad_norm": float(actor_grad_norm.item()),
            "critic_grad_norm": float(critic_grad_norm.item()),
            "actor_grad_norm_before_clip": float(actor_grad_norm.item()),
            "actor_grad_norm_after_clip": float(actor_grad_norm_after.item()),
            "critic_grad_norm_before_clip": float(critic_grad_norm.item()),
            "critic_grad_norm_after_clip": float(critic_grad_norm_after.item()),
            "actor_update_mode_bc_warmup": float(False),
            "number_of_actor_updates": 1.0,
            "number_of_critic_updates": 1.0,
            "total_actor_updates": float(self.actor_updates),
            "total_critic_updates": float(self.critic_updates),
            "total_temperature_updates": float(self.temperature_updates),
            "policy_log_std_mean": float(policy_std.log().mean().item()),
            "policy_log_std_min": float(policy_std.log().min().item()),
            "policy_log_std_max": float(policy_std.log().max().item()),
            **cql_metrics,
        }

    def optimizer_state(self) -> Dict[str, object]:
        return {
            "actor": self.actor_optimizer.state_dict(),
            "q1": self.q1_optimizer.state_dict(),
            "q2": self.q2_optimizer.state_dict(),
            "alpha": self.alpha_optimizer.state_dict(),
        }

    def load_optimizer_state(self, state: Dict[str, object]) -> None:
        if "actor" in state:
            self.actor_optimizer.load_state_dict(state["actor"])
        if "q1" in state:
            self.q1_optimizer.load_state_dict(state["q1"])
        if "q2" in state:
            self.q2_optimizer.load_state_dict(state["q2"])
        if "alpha" in state:
            self.alpha_optimizer.load_state_dict(state["alpha"])
