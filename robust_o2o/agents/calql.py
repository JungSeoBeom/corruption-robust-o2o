from __future__ import annotations

import copy
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from ..config import ExperimentConfig
from ..networks import QNetwork, TanhGaussianPolicy
from ..replay import TensorBatch
from .base import BaseAgent, soft_update


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


class CalQLAgent(BaseAgent):
    """Cal-QL with MC-return calibration during offline pre-training."""

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
        self.actor = TanhGaussianPolicy(
            state_dim,
            action_dim,
            config.hidden_dim,
            max(config.hidden_layers, 3),
            max_action,
        )
        self.q1 = QNetwork(
            state_dim, action_dim, config.hidden_dim, max(config.hidden_layers, 3)
        )
        self.q2 = QNetwork(
            state_dim, action_dim, config.hidden_dim, max(config.hidden_layers, 3)
        )
        self.target_q1 = copy.deepcopy(self.q1).requires_grad_(False)
        self.target_q2 = copy.deepcopy(self.q2).requires_grad_(False)
        self.log_alpha = torch.nn.Parameter(torch.zeros(1))
        self.to(device)
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=config.learning_rate
        )
        self.q1_optimizer = torch.optim.Adam(
            self.q1.parameters(), lr=config.learning_rate
        )
        self.q2_optimizer = torch.optim.Adam(
            self.q2.parameters(), lr=config.learning_rate
        )
        self.alpha_optimizer = torch.optim.Adam(
            [self.log_alpha], lr=config.entropy_lr
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
        expanded = states[:, None, :].expand(-1, count, -1).reshape(
            batch_size * count, -1
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
        count = self.config.cql_n_actions
        random_actions = torch.empty(
            batch_size, count, action_dim, device=self.device
        ).uniform_(-self.actor.max_action, self.actor.max_action)
        current_actions, current_log_prob = self._sample_many(states, count)
        next_actions, next_log_prob = self._sample_many(next_states, count)

        q1_random = self.q1(states, random_actions)
        q2_random = self.q2(states, random_actions)
        q1_current = self.q1(states, current_actions.detach())
        q2_current = self.q2(states, current_actions.detach())
        # Cal-QL evaluates next-policy actions at the current state.
        q1_next = self.q1(states, next_actions.detach())
        q2_next = self.q2(states, next_actions.detach())

        lower_bound = mc_returns.reshape(-1, 1).expand(-1, count)
        valid = mc_valid.reshape(-1, 1).bool().expand(-1, count)
        if valid.any():
            bound_rate = 0.25 * (
                (q1_current[valid] < lower_bound[valid]).float().mean()
                + (q2_current[valid] < lower_bound[valid]).float().mean()
                + (q1_next[valid] < lower_bound[valid]).float().mean()
                + (q2_next[valid] < lower_bound[valid]).float().mean()
            )
        else:
            bound_rate = states.new_tensor(0.0)
        if not self.online_phase:
            q1_current = torch.where(valid, torch.maximum(q1_current, lower_bound), q1_current)
            q2_current = torch.where(valid, torch.maximum(q2_current, lower_bound), q2_current)
            q1_next = torch.where(valid, torch.maximum(q1_next, lower_bound), q1_next)
            q2_next = torch.where(valid, torch.maximum(q2_next, lower_bound), q2_next)

        random_density = np.log((0.5 / self.actor.max_action) ** action_dim)
        cat_q1 = torch.cat(
            (
                q1_random - random_density,
                q1_next - next_log_prob.detach(),
                q1_current - current_log_prob.detach(),
            ),
            dim=1,
        )
        cat_q2 = torch.cat(
            (
                q2_random - random_density,
                q2_next - next_log_prob.detach(),
                q2_current - current_log_prob.detach(),
            ),
            dim=1,
        )
        cql_q1 = torch.logsumexp(cat_q1, dim=1) - q1_data
        cql_q2 = torch.logsumexp(cat_q2, dim=1) - q2_data
        weight = (
            self.config.cql_alpha_online
            if self.online_phase
            else self.config.cql_alpha
        )
        loss = weight * (cql_q1.mean() + cql_q2.mean())
        return loss, {
            "calibration_bound_rate": float(bound_rate.item()),
            "cql_q1_diff": float(cql_q1.mean().item()),
            "cql_q2_diff": float(cql_q2.mean().item()),
            "mc_calibration_valid_fraction": float(mc_valid.float().mean().item()),
        }

    def update(self, batch: TensorBatch) -> Dict[str, float]:
        states = batch["observations"]
        actions = batch["actions"]
        rewards = batch["rewards"].reshape(-1)
        next_states = batch["next_observations"]
        terminals = batch["terminals"].reshape(-1)
        mc_returns = batch.get("mc_returns", torch.zeros_like(rewards)).reshape(-1)
        mc_valid = batch.get(
            "mc_calibration_valid", torch.ones_like(rewards)
        ).reshape(-1)

        new_actions, log_prob, _, policy_std = self.actor(
            states, need_log_prob=True
        )
        alpha_loss = -(
            self.log_alpha * (log_prob - self.actor.action_dim).detach()
        ).mean()
        self.alpha_optimizer.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.alpha_optimizer.step()

        if not self.online_phase and self.total_updates <= self.config.bc_steps:
            actor_loss = (
                self.alpha.detach() * log_prob
                - self.actor.log_prob(states, actions)
            ).mean()
        else:
            q_new = torch.minimum(
                self.q1(states, new_actions), self.q2(states, new_actions)
            )
            actor_loss = (self.alpha.detach() * log_prob - q_new).mean()
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        actor_grad_norm = torch.nn.utils.clip_grad_norm_(
            self.actor.parameters(), float("inf")
        )
        self.actor_optimizer.step()

        with torch.no_grad():
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
                self.alpha,
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
        self.q1_optimizer.zero_grad(set_to_none=True)
        self.q2_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        critic_grad_norm = torch.nn.utils.clip_grad_norm_(
            list(self.q1.parameters()) + list(self.q2.parameters()), float("inf")
        )
        self.q1_optimizer.step()
        self.q2_optimizer.step()
        soft_update(
            self.target_q1, self.q1, self.config.target_update_rate
        )
        soft_update(
            self.target_q2, self.q2, self.config.target_update_rate
        )
        self.total_updates += 1
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
            "number_of_actor_updates": 1.0,
            "number_of_critic_updates": 1.0,
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
