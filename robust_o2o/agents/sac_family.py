from __future__ import annotations

import copy
import itertools
from typing import Dict

import torch
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence

from ..config import ExperimentConfig
from ..networks import DensityRatioNetwork, EnsembleQNetwork, TanhGaussianPolicy
from ..replay import TensorBatch
from .base import BaseAgent, soft_update


class SACEnsembleAgent(BaseAgent):
    """Shared implementation for UWMSG, WSRL, RO2O, and Pessimistic Q-Ensemble."""

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
        self.variant = config.algorithm
        self.actor = TanhGaussianPolicy(
            state_dim,
            action_dim,
            config.hidden_dim,
            max(config.hidden_layers, 3),
            max_action,
        )
        self.critic = EnsembleQNetwork(
            state_dim,
            action_dim,
            config.hidden_dim,
            max(config.hidden_layers, 3),
            config.sac_num_critics,
        )
        self.target_critic = copy.deepcopy(self.critic).requires_grad_(False)
        self.critic2 = (
            EnsembleQNetwork(
                state_dim,
                action_dim,
                config.hidden_dim,
                max(config.hidden_layers, 3),
                config.sac_num_critics,
            )
            if self.variant == "pessimistic_q_ensemble"
            else None
        )
        self.target_critic2 = (
            copy.deepcopy(self.critic2).requires_grad_(False)
            if self.critic2 is not None
            else None
        )
        self.density_ratio = (
            DensityRatioNetwork(
                state_dim, action_dim, config.hidden_dim, config.hidden_layers
            )
            if self.variant == "pessimistic_q_ensemble"
            else None
        )
        self.log_alpha = torch.nn.Parameter(torch.zeros(1))
        self.ro2o_uncertainty = config.ro2o_uncertainty
        self.last_priority_values: torch.Tensor | None = None
        self.to(device)
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=config.learning_rate
        )
        critic_parameters = self.critic.parameters()
        if self.critic2 is not None:
            critic_parameters = itertools.chain(
                critic_parameters, self.critic2.parameters()
            )
        self.critic_optimizer = torch.optim.Adam(
            critic_parameters, lr=config.learning_rate
        )
        self.alpha_optimizer = torch.optim.Adam(
            [self.log_alpha], lr=config.entropy_lr
        )
        self.density_optimizer = (
            torch.optim.Adam(self.density_ratio.parameters(), lr=config.learning_rate)
            if self.density_ratio is not None
            else None
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

    def _q_for_policy(self, values: torch.Tensor) -> torch.Tensor:
        if self.variant == "uwmsg":
            return values.mean(dim=0) - self.config.lcb_ratio * values.std(dim=0)
        return values.min(dim=0).values

    def _critic_values_for_actions(
        self,
        critic: EnsembleQNetwork,
        states: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        # actions [B, K, A] -> [N, B, K]
        batch_size, count, action_dim = actions.shape
        repeated_states = states[:, None, :].expand(-1, count, -1).reshape(
            batch_size * count, -1
        )
        flat_actions = actions.reshape(batch_size * count, action_dim)
        return critic(repeated_states, flat_actions).reshape(
            critic.num_critics, batch_size, count
        )

    def _cql_penalty(self, states: torch.Tensor, data_actions: torch.Tensor) -> torch.Tensor:
        batch_size, action_dim = data_actions.shape
        count = self.config.cql_n_actions
        random_actions = torch.empty(
            batch_size, count, action_dim, device=self.device
        ).uniform_(-self.actor.max_action, self.actor.max_action)
        expanded_states = states[:, None, :].expand(-1, count, -1).reshape(
            batch_size * count, -1
        )
        with torch.no_grad():
            policy_actions = self.actor.act(expanded_states).reshape(
                batch_size, count, action_dim
            )
        q_random = self._critic_values_for_actions(
            self.critic, states, random_actions
        )
        q_policy = self._critic_values_for_actions(
            self.critic, states, policy_actions
        )
        q_data = self.critic(states, data_actions)
        ood = torch.logsumexp(torch.cat((q_random, q_policy), dim=-1), dim=-1)
        penalty = (ood - q_data).mean()
        if self.critic2 is not None:
            q2_random = self._critic_values_for_actions(
                self.critic2, states, random_actions
            )
            q2_policy = self._critic_values_for_actions(
                self.critic2, states, policy_actions
            )
            q2_data = self.critic2(states, data_actions)
            q2_ood = torch.logsumexp(
                torch.cat((q2_random, q2_policy), dim=-1), dim=-1
            )
            penalty = penalty + (q2_ood - q2_data).mean()
        return penalty

    def _density_loss(
        self,
        offline_batch: TensorBatch,
        online_batch: TensorBatch,
    ) -> torch.Tensor:
        if self.density_ratio is None:
            return self.log_alpha.new_tensor(0.0)
        offline_weights = self.density_ratio(
            offline_batch["observations"], offline_batch["actions"]
        )
        online_weights = self.density_ratio(
            online_batch["observations"], online_batch["actions"]
        )
        offline_f_star = -torch.log(2.0 / (offline_weights + 1.0) + 1e-10)
        online_f_prime = torch.log(
            2.0 * online_weights / (online_weights + 1.0) + 1e-10
        )
        return offline_f_star.mean() - online_f_prime.mean()

    def _density_priorities(
        self,
        rl_batch: TensorBatch,
        density_offline_batch: TensorBatch,
    ) -> torch.Tensor:
        if self.density_ratio is None:
            return torch.ones(
                len(rl_batch["observations"]), device=self.device
            )
        weights = self.density_ratio(
            rl_batch["observations"], rl_batch["actions"]
        )
        offline_weights = self.density_ratio(
            density_offline_batch["observations"],
            density_offline_batch["actions"],
        )
        temperature = self.config.balanced_replay_temperature
        denominator = offline_weights.pow(1.0 / temperature).mean()
        normalized = weights.pow(1.0 / temperature)
        normalized = normalized / (denominator.detach() + 1e-10)
        return normalized.detach().clamp(self.config.priority_floor, 1_000.0)

    def consume_priority_values(self) -> torch.Tensor | None:
        result = self.last_priority_values
        self.last_priority_values = None
        return result

    def _ro2o_noised_states(
        self, states: torch.Tensor, epsilon: float
    ) -> tuple[torch.Tensor, int]:
        count = self.config.ro2o_sample_size
        noise = torch.empty(
            states.shape[0], count, states.shape[1], device=self.device
        ).uniform_(-epsilon, epsilon)
        noised = (states[:, None, :] + noise).reshape(-1, states.shape[1])
        return noised, count

    def _ro2o_q_smoothness(
        self, states: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        noised_states, count = self._ro2o_noised_states(
            states, self.config.ro2o_q_smooth_eps
        )
        repeated_actions = actions[:, None, :].expand(-1, count, -1).reshape(
            -1, actions.shape[1]
        )
        noised_q = self.critic(noised_states, repeated_actions).reshape(
            self.critic.num_critics, states.shape[0], count
        )
        base_q = self.critic(states, actions).unsqueeze(-1)
        difference = noised_q - base_q
        asymmetric = torch.where(
            difference > 0, 0.8 * difference.square(), 0.2 * difference.square()
        ).mean(dim=0)
        return 1e-4 * asymmetric.max(dim=-1).values.mean()

    def _ro2o_policy_smoothness(self, states: torch.Tensor) -> torch.Tensor:
        noised_states, count = self._ro2o_noised_states(
            states, self.config.ro2o_policy_smooth_eps
        )
        base = self.actor.distribution(states)
        noised = self.actor.distribution(noised_states)
        repeated = Normal(
            base.mean[:, None, :].expand(-1, count, -1).reshape(
                -1, self.actor.action_dim
            ),
            base.stddev[:, None, :].expand(-1, count, -1).reshape(
                -1, self.actor.action_dim
            ),
        )
        symmetric_kl = (
            kl_divergence(repeated, noised).sum(dim=-1)
            + kl_divergence(noised, repeated).sum(dim=-1)
        ).reshape(states.shape[0], count)
        return (
            self.config.ro2o_beta_policy
            * symmetric_kl.max(dim=-1).values.mean()
        )

    def _ro2o_ood_penalty(self, states: torch.Tensor) -> torch.Tensor:
        noised_states, _ = self._ro2o_noised_states(
            states, self.config.ro2o_ood_smooth_eps
        )
        actions = self.actor.act(noised_states)
        q_values = self.critic(noised_states, actions)
        target = q_values - self.ro2o_uncertainty * q_values.std(
            dim=0, keepdim=True
        )
        penalty = self.config.ro2o_beta_ood * F.mse_loss(q_values, target.detach())
        self.ro2o_uncertainty = max(
            self.ro2o_uncertainty - self.config.ro2o_uncertainty_decay,
            self.config.ro2o_uncertainty_min,
        )
        return penalty

    def update(
        self,
        batch: TensorBatch | None = None,
        *,
        rl_batch: TensorBatch | None = None,
        density_offline_batch: TensorBatch | None = None,
        density_online_batch: TensorBatch | None = None,
        rl_batch_prioritized: bool = False,
    ) -> Dict[str, float]:
        if rl_batch is not None:
            if batch is not None:
                raise ValueError("pass either batch or rl_batch, not both")
            batch = rl_batch
        if batch is None:
            raise ValueError("an RL batch is required")
        self.last_priority_values = None
        states = batch["observations"]
        actions = batch["actions"]
        rewards = batch["rewards"].reshape(-1)
        next_states = batch["next_observations"]
        terminals = batch["terminals"].reshape(-1)

        sampled_actions, log_prob, _, policy_std = self.actor(
            states, need_log_prob=True
        )
        alpha_loss = -(
            self.log_alpha * (log_prob + float(-self.actor.action_dim)).detach()
        ).mean()
        self.alpha_optimizer.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.alpha_optimizer.step()

        policy_values = self.critic(states, sampled_actions)
        if self.critic2 is not None:
            policy_values = torch.minimum(
                policy_values, self.critic2(states, sampled_actions)
            )
            policy_q = policy_values.mean(dim=0)
        else:
            policy_q = self._q_for_policy(policy_values)
        actor_loss = (self.alpha.detach() * log_prob - policy_q).mean()
        policy_smoothness = states.new_tensor(0.0)
        if self.variant == "ro2o" and not self.online_phase:
            policy_smoothness = self._ro2o_policy_smoothness(states)
            actor_loss = actor_loss + policy_smoothness
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
            next_q_all = self.target_critic(next_states, next_actions)
            if self.variant == "uwmsg":
                target = rewards.unsqueeze(0) + (
                    (1.0 - terminals) * self.config.discount
                ).unsqueeze(0) * (
                    next_q_all - self.alpha.detach() * next_log_prob.unsqueeze(0)
                )
            elif self.target_critic2 is not None:
                next_q2_all = self.target_critic2(next_states, next_actions)
                next_q_all = torch.minimum(next_q_all, next_q2_all)
                target = rewards.unsqueeze(0) + (
                    (1.0 - terminals) * self.config.discount
                ).unsqueeze(0) * (
                    next_q_all - self.alpha.detach() * next_log_prob.unsqueeze(0)
                )
            else:
                next_q = next_q_all.min(dim=0).values
                next_q = next_q - self.alpha.detach() * next_log_prob
                target_scalar = rewards + (
                    1.0 - terminals
                ) * self.config.discount * next_q
                target = target_scalar.unsqueeze(0).expand_as(next_q_all)

        predicted = self.critic(states, actions)
        predicted2 = (
            self.critic2(states, actions) if self.critic2 is not None else None
        )
        td_error = (predicted - target).square()
        td_error2 = (
            (predicted2 - target).square() if predicted2 is not None else None
        )
        uncertainty_mean = states.new_tensor(0.0)
        if self.variant == "uwmsg":
            uncertainty = (
                self.config.uncertainty_basic
                + self.config.uncertainty_ratio
                * predicted.std(dim=0, keepdim=True).detach()
            ).clamp(self.config.uncertainty_min, self.config.uncertainty_max)
            critic_loss = (td_error / uncertainty).mean(dim=1).sum()
            uncertainty_mean = uncertainty.mean()
        else:
            per_sample_td = td_error.mean(dim=0)
            if td_error2 is not None:
                per_sample_td = per_sample_td + td_error2.mean(dim=0)
            critic_loss = per_sample_td.mean()

        cql_penalty = states.new_tensor(0.0)
        if self.variant in ("wsrl", "pessimistic_q_ensemble") and not self.online_phase:
            cql_penalty = self.config.cql_alpha * self._cql_penalty(states, actions)
            critic_loss = critic_loss + cql_penalty

        q_smoothness = states.new_tensor(0.0)
        ood_penalty = states.new_tensor(0.0)
        if self.variant == "ro2o" and not self.online_phase:
            q_smoothness = self._ro2o_q_smoothness(states, actions)
            ood_penalty = self._ro2o_ood_penalty(states)
            critic_loss = critic_loss + q_smoothness + ood_penalty

        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        critic_parameters = list(self.critic.parameters())
        if self.critic2 is not None:
            critic_parameters += list(self.critic2.parameters())
        critic_grad_norm = torch.nn.utils.clip_grad_norm_(
            critic_parameters, float("inf")
        )
        self.critic_optimizer.step()
        soft_update(
            self.target_critic, self.critic, self.config.target_update_rate
        )
        if self.target_critic2 is not None and self.critic2 is not None:
            soft_update(
                self.target_critic2,
                self.critic2,
                self.config.target_update_rate,
            )

        density_value = states.new_tensor(0.0)
        density_batches_available = (
            density_offline_batch is not None
            and density_online_batch is not None
        )
        if self.density_optimizer is not None and density_batches_available:
            density_value = self._density_loss(
                density_offline_batch, density_online_batch
            )
            self.density_optimizer.zero_grad(set_to_none=True)
            density_value.backward()
            self.density_optimizer.step()
            with torch.no_grad():
                self.last_priority_values = self._density_priorities(
                    batch, density_offline_batch
                )

        source = batch.get("_source")
        if source is None:
            rl_offline_count = len(states)
            rl_online_count = 0
        else:
            rl_offline_count = int((source.reshape(-1) == 0).sum().item())
            rl_online_count = int((source.reshape(-1) == 1).sum().item())
        density_offline_count = (
            len(density_offline_batch["observations"])
            if density_offline_batch is not None
            else 0
        )
        density_online_count = (
            len(density_online_batch["observations"])
            if density_online_batch is not None
            else 0
        )

        self.total_updates += 1
        ensemble_mean_per_sample = predicted.detach().mean(dim=0)
        ensemble_std_per_sample = predicted.detach().std(dim=0, unbiased=False)
        lcb_penalty = (
            self.config.lcb_ratio * ensemble_std_per_sample
            if self.variant == "uwmsg"
            else torch.zeros_like(ensemble_std_per_sample)
        )
        return {
            "critic_loss": float(critic_loss.item()),
            "actor_loss": float(actor_loss.item()),
            "alpha_loss": float(alpha_loss.item()),
            "alpha": float(self.alpha.item()),
            "cql_penalty": float(cql_penalty.item()),
            "uncertainty_mean": float(uncertainty_mean.item()),
            "q_smoothness": float(q_smoothness.item()),
            "policy_smoothness": float(policy_smoothness.item()),
            "ood_penalty": float(ood_penalty.item()),
            "density_loss": float(density_value.item()),
            "density_offline_count": float(density_offline_count),
            "density_online_count": float(density_online_count),
            "rl_offline_count": float(rl_offline_count),
            "rl_online_count": float(rl_online_count),
            "density_batches_prioritized": 0.0,
            "rl_batch_prioritized": float(rl_batch_prioritized),
            "bellman_loss": float(
                (td_error.mean() + (td_error2.mean() if td_error2 is not None else 0.0)).item()
            ),
            "cql_to_bellman_ratio": float(
                cql_penalty.abs().div(td_error.mean().detach().abs() + 1e-8).item()
            ),
            "temperature_alpha": float(self.alpha.item()),
            "entropy": float((-log_prob.detach()).mean().item()),
            "gradient_norm_actor": float(actor_grad_norm.item()),
            "gradient_norm_critic": float(critic_grad_norm.item()),
            "number_of_actor_updates": 1.0,
            "number_of_critic_updates": 1.0,
            "ensemble_q_mean": float(predicted.detach().mean().item()),
            "ensemble_q_std": float(predicted.detach().std(unbiased=False).item()),
            "lcb_penalty_mean": float(lcb_penalty.mean().item()),
            "policy_lcb_value": float(policy_q.detach().mean().item()),
            "fraction_lcb_penalty_larger_than_q_mean": float(
                (lcb_penalty > ensemble_mean_per_sample.abs()).float().mean().item()
            ),
            "policy_log_std_mean": float(policy_std.log().mean().item()),
            "policy_log_std_min": float(policy_std.log().min().item()),
            "policy_log_std_max": float(policy_std.log().max().item()),
            "q_mean": float(
                (
                    torch.minimum(predicted, predicted2).mean()
                    if predicted2 is not None
                    else predicted.mean()
                ).item()
            ),
        }

    def optimizer_state(self) -> Dict[str, object]:
        state = {
            "actor": self.actor_optimizer.state_dict(),
            "critic": self.critic_optimizer.state_dict(),
            "alpha": self.alpha_optimizer.state_dict(),
        }
        if self.density_optimizer is not None:
            state["density"] = self.density_optimizer.state_dict()
        return state

    def load_optimizer_state(self, state: Dict[str, object]) -> None:
        if "actor" in state:
            self.actor_optimizer.load_state_dict(state["actor"])
        if "critic" in state:
            self.critic_optimizer.load_state_dict(state["critic"])
        if "alpha" in state:
            self.alpha_optimizer.load_state_dict(state["alpha"])
        if "density" in state and self.density_optimizer is not None:
            self.density_optimizer.load_state_dict(state["density"])
