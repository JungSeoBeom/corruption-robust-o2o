from __future__ import annotations

import copy
from typing import Dict, Tuple

import torch
import torch.nn.functional as F

from ..cql import importance_sampled_cql
from ..config import ExperimentConfig
from ..networks import QNetwork, TanhGaussianPolicy
from ..replay import TensorBatch
from .base import BaseAgent, gradient_norm, soft_update


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
        calibrate = (
            not self.online_phase
            and self.config.calibration_mask_mode != "disabled"
        )
        result = importance_sampled_cql(
            policy=self.actor,
            evaluators=(self.q1, self.q2),
            states=states,
            next_states=next_states,
            data_actions=actions,
            data_values=(q1_data, q2_data),
            num_actions=self.config.cql_n_actions,
            temperature=self.config.cql_temperature,
            calibration_lower_bound=mc_returns if calibrate else None,
            calibration_valid=mc_valid if calibrate else None,
        )
        cql_q1, cql_q2 = result.differences
        weight = (
            self.config.cql_alpha_online
            if self.online_phase
            else self.config.cql_alpha
        )
        loss = weight * result.loss
        return loss, {
            "calibration_bound_rate": float(result.calibration_bound_rate.item()),
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
        oracle_valid = batch.get(
            "mc_calibration_valid", torch.ones_like(rewards)
        ).reshape(-1)
        if self.config.calibration_mask_mode == "all":
            mc_valid = torch.ones_like(oracle_valid)
        elif self.config.calibration_mask_mode == "oracle_exclude_corrupted":
            mc_valid = oracle_valid
        else:
            mc_valid = torch.zeros_like(oracle_valid)

        new_actions, log_prob, _, policy_std = self.actor(
            states, need_log_prob=True
        )
        alpha_loss = -(
            self.log_alpha * (log_prob - self.actor.action_dim).detach()
        ).mean()
        self.alpha_optimizer.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.alpha_optimizer.step()

        if (
            not self.online_phase
            and self.total_updates < self.config.calql_bc_warmup_steps
        ):
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
        actor_grad_norm = (
            gradient_norm(self.actor.parameters())
            if self.config.max_grad_norm is None
            else torch.nn.utils.clip_grad_norm_(
                self.actor.parameters(), self.config.max_grad_norm
            )
        )
        actor_grad_norm_after = gradient_norm(self.actor.parameters())
        self.actor_optimizer.step()

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
        critic_parameters = list(self.q1.parameters()) + list(self.q2.parameters())
        critic_grad_norm = (
            gradient_norm(critic_parameters)
            if self.config.max_grad_norm is None
            else torch.nn.utils.clip_grad_norm_(
                critic_parameters, self.config.max_grad_norm
            )
        )
        critic_grad_norm_after = gradient_norm(
            critic_parameters
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
            "actor_update_mode_bc_warmup": float(
                not self.online_phase
                and self.total_updates <= self.config.calql_bc_warmup_steps
            ),
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
