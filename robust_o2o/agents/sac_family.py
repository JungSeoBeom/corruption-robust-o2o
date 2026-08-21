from __future__ import annotations

import copy
import math
from typing import Dict

import torch
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence

from ..cql import importance_sampled_cql
from ..config import ExperimentConfig
from ..networks import EnsembleQNetwork, TanhGaussianPolicy
from ..replay import TensorBatch
from .base import BaseAgent, gradient_norm, soft_update


class SACEnsembleAgent(BaseAgent):
    """Shared implementation for UWMSG, WSRL, and RO2O.

    The canonical Pessimistic Q-Ensemble has a dedicated agent with five
    independent policies and twin critics; it must never enter this
    shared-actor implementation.
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
        self.variant = config.algorithm
        network_hidden_layers = (
            2
            if self.variant == "wsrl"
            and config.implementation_profile != "legacy_current"
            else max(config.hidden_layers, 3)
        )
        wsrl_profile = (
            self.variant == "wsrl"
            and config.implementation_profile != "legacy_current"
        )
        self.actor = TanhGaussianPolicy(
            state_dim,
            action_dim,
            config.hidden_dim,
            network_hidden_layers,
            max_action,
            layer_norm=(self.variant == "wsrl" and config.wsrl_layer_norm),
            wsrl_profile=wsrl_profile,
        )
        self.critic = EnsembleQNetwork(
            state_dim,
            action_dim,
            config.hidden_dim,
            network_hidden_layers,
            config.sac_num_critics,
            layer_norm=(self.variant == "wsrl" and config.wsrl_layer_norm),
            wsrl_profile=wsrl_profile,
        )
        self.target_critic = copy.deepcopy(self.critic).requires_grad_(False)
        # Historical shared-actor PQE execution was removed.  These optional
        # attributes stay ``None`` only to keep old generic optimizer payloads
        # readable; no canonical algorithm activates them.
        self.critic2 = None
        self.target_critic2 = None
        self.density_ratio = None
        alpha_parameter_init = (
            math.log(math.expm1(1.0))
            if self.variant == "wsrl"
            and config.wsrl_entropy_profile == "official_negative_action_dim"
            else 0.0
        )
        self.log_alpha = torch.nn.Parameter(
            torch.full((1,), alpha_parameter_init)
        )
        self.ro2o_uncertainty = config.ro2o_uncertainty
        self.last_priority_values: torch.Tensor | None = None
        self.to(device)
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=config.actor_learning_rate
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=config.critic_learning_rate
        )
        self.alpha_optimizer = torch.optim.Adam(
            [self.log_alpha], lr=config.temperature_learning_rate
        )
        self.density_optimizer = (
            torch.optim.Adam(self.density_ratio.parameters(), lr=config.learning_rate)
            if self.density_ratio is not None
            else None
        )

    @property
    def alpha(self) -> torch.Tensor:
        if (
            self.variant == "wsrl"
            and self.config.wsrl_entropy_profile
            == "official_negative_action_dim"
        ):
            # zhouzypaul/wsrl uses a softplus-parameterized Geq multiplier.
            return F.softplus(self.log_alpha)
        return self.log_alpha.exp()

    def _temperature_loss(self, log_prob: torch.Tensor) -> torch.Tensor:
        target_entropy = float(self.config.target_entropy)
        if (
            self.variant == "wsrl"
            and self.config.wsrl_entropy_profile
            == "official_negative_action_dim"
        ):
            entropy = -log_prob.detach().mean()
            return self.alpha * (entropy - target_entropy)
        return -(
            self.log_alpha * (log_prob + target_entropy).detach()
        ).mean()

    def _cql_loss_enabled(self) -> bool:
        """CQL is an offline pretrainer only for WSRL in this agent family."""

        return self.variant == "wsrl" and not self.online_phase

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

    def _sample_target_critic_indices(self, critic_count: int) -> torch.Tensor:
        """Seeded through the learner torch RNG; sampling is with replacement."""
        return torch.randint(
            critic_count,
            (self.config.wsrl_target_critic_subsample_size,),
            device=self.device,
        )

    @staticmethod
    def _wsrl_subsampled_min(
        q_values: torch.Tensor, indices: torch.Tensor
    ) -> torch.Tensor:
        """REDQ minimum for an explicit with-replacement critic sample."""
        return q_values.index_select(0, indices).min(dim=0).values

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

    def _cql_penalty(
        self,
        states: torch.Tensor,
        next_states: torch.Tensor,
        data_actions: torch.Tensor,
    ) -> torch.Tensor:
        if self.config.implementation_profile == "legacy_current":
            return self._legacy_cql_penalty(states, data_actions)
        critics = [self.critic]
        if self.critic2 is not None:
            critics.append(self.critic2)
        wsrl_indices = (
            self._sample_target_critic_indices(self.critic.num_critics)
            if self.variant == "wsrl"
            else None
        )

        def evaluate(critic, sample_states, sample_actions):
            values = self._critic_values_for_actions(
                critic, sample_states, sample_actions
            )
            return (
                values.index_select(0, wsrl_indices)
                if wsrl_indices is not None
                else values
            )

        data_values = []
        for critic in critics:
            values = critic(states, data_actions)
            if wsrl_indices is not None:
                values = values.index_select(0, wsrl_indices)
            data_values.append(values)
        result = importance_sampled_cql(
            policy=self.actor,
            evaluators=tuple(
                lambda sample_states, sample_actions, critic=critic: evaluate(
                    critic, sample_states, sample_actions
                )
                for critic in critics
            ),
            states=states,
            next_states=next_states,
            data_actions=data_actions,
            data_values=tuple(data_values),
            num_actions=self.config.cql_n_actions,
            temperature=self.config.cql_temperature,
        )
        return result.loss

    def _legacy_cql_penalty(
        self, states: torch.Tensor, data_actions: torch.Tensor
    ) -> torch.Tensor:
        """Pre-profile repository objective, retained only for reproduction."""
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
        critics = [self.critic]
        if self.critic2 is not None:
            critics.append(self.critic2)
        penalty = states.new_zeros(())
        for critic in critics:
            q_random = self._critic_values_for_actions(
                critic, states, random_actions
            )
            q_policy = self._critic_values_for_actions(
                critic, states, policy_actions
            )
            q_data = critic(states, data_actions)
            penalty = penalty + (
                torch.logsumexp(
                    torch.cat((q_random, q_policy), dim=-1), dim=-1
                )
                - q_data
            ).mean()
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
        update_actor_temperature: bool = True,
        update_critic: bool = True,
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

        # A high-UTD critic-only step in upstream WSRL does not execute the
        # actor/temperature loss functions.  Besides wasting work, sampling an
        # unused policy action here would advance PyTorch's learner RNG before
        # target-action and REDQ-subset sampling.
        log_prob = states.new_zeros(len(states))
        policy_std = states.new_ones((len(states), actions.shape[-1]))
        policy_q = states.new_zeros(len(states))
        actor_loss = states.new_zeros(())
        alpha_loss = states.new_zeros(())
        policy_smoothness = states.new_tensor(0.0)
        actor_grad_norm = states.new_zeros(())
        actor_grad_norm_after = states.new_zeros(())
        defer_actor_temperature_step = (
            self.variant == "wsrl"
            and update_actor_temperature
            and update_critic
        )
        if update_actor_temperature:
            actor_alpha = self.alpha.detach().clone()
            sampled_actions, log_prob, _, policy_std = self.actor(
                states, need_log_prob=True
            )
            temperature_log_prob = log_prob
            if (
                self.variant == "wsrl"
                and self.config.wsrl_entropy_profile
                == "official_negative_action_dim"
            ):
                # Upstream temperature_loss_fn samples from next observations.
                _, temperature_log_prob, _, _ = self.actor(
                    next_states, need_log_prob=True
                )
            alpha_loss = self._temperature_loss(temperature_log_prob)
            self.alpha_optimizer.zero_grad(set_to_none=True)
            alpha_loss.backward()
            if not defer_actor_temperature_step:
                self.alpha_optimizer.step()

            policy_values = self.critic(states, sampled_actions)
            if self.critic2 is not None:
                policy_values = torch.minimum(
                    policy_values, self.critic2(states, sampled_actions)
                )
                policy_q = policy_values.mean(dim=0)
            else:
                policy_q = self._q_for_policy(policy_values)
            # Flax computes actor and temperature gradients from one old
            # parameter tree.  Preserve that temperature value for WSRL even
            # though these disjoint PyTorch optimizers are stepped serially.
            entropy_coefficient = (
                actor_alpha
                if self.variant == "wsrl"
                else self.alpha.detach()
            )
            actor_loss = (entropy_coefficient * log_prob - policy_q).mean()
            if self.variant == "ro2o" and not self.online_phase:
                policy_smoothness = self._ro2o_policy_smoothness(states)
                actor_loss = actor_loss + policy_smoothness
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
            if not defer_actor_temperature_step:
                self.actor_optimizer.step()

        if not update_critic:
            if not update_actor_temperature:
                raise ValueError("an update must select critics and/or actor-temperature")
            self.actor_updates += 1
            self.temperature_updates += 1
            return {
                "actor_loss": float(actor_loss.item()),
                "alpha_loss": float(alpha_loss.item()),
                "alpha": float(self.alpha.item()),
                "temperature_alpha": float(self.alpha.item()),
                "entropy": float((-log_prob.detach()).mean().item()),
                "actor_grad_norm": float(actor_grad_norm.item()),
                "actor_grad_norm_before_clip": float(actor_grad_norm.item()),
                "actor_grad_norm_after_clip": float(actor_grad_norm_after.item()),
                "number_of_actor_updates": 1.0,
                "number_of_critic_updates": 0.0,
                "number_of_temperature_updates": 1.0,
                "total_actor_updates": float(self.actor_updates),
                "total_critic_updates": float(self.critic_updates),
                "total_temperature_updates": float(self.temperature_updates),
                "policy_lcb_value": float(policy_q.detach().mean().item()),
                "policy_log_std_mean": float(policy_std.log().mean().item()),
                "policy_log_std_min": float(policy_std.log().min().item()),
                "policy_log_std_max": float(policy_std.log().max().item()),
                "cql_loss_enabled": float(self._cql_loss_enabled()),
                "wsrl_online_cql_disabled": float(
                    self.variant == "wsrl" and self.online_phase
                ),
            }

        with torch.no_grad():
            wsrl_max_backup = (
                self.variant == "wsrl"
                and not self.online_phase
                and self.config.cql_max_target_backup
            )
            if wsrl_max_backup:
                batch_size = len(next_states)
                count = self.config.cql_n_actions
                expanded_next_states = next_states[:, None, :].expand(
                    -1, count, -1
                ).reshape(batch_size * count, -1)
                next_actions, next_log_prob, _, _ = self.actor(
                    expanded_next_states, need_log_prob=True
                )
                next_actions = next_actions.reshape(batch_size, count, -1)
                next_log_prob = next_log_prob.reshape(batch_size, count)
                next_q_candidates = self._critic_values_for_actions(
                    self.target_critic, next_states, next_actions
                )
                indices = self._sample_target_critic_indices(
                    next_q_candidates.shape[0]
                )
                clipped_candidates = self._wsrl_subsampled_min(
                    next_q_candidates, indices
                )
                best = clipped_candidates.argmax(dim=1, keepdim=True)
                next_q = clipped_candidates.gather(1, best).squeeze(1)
                selected_log_prob = next_log_prob.gather(1, best).squeeze(1)
                if self.config.backup_entropy:
                    next_q = next_q - self.alpha.detach() * selected_log_prob
                target_scalar = rewards + (
                    1.0 - terminals
                ) * self.config.discount * next_q
                target = target_scalar.unsqueeze(0).expand(
                    self.critic.num_critics, -1
                )
            else:
                next_actions, next_log_prob, _, _ = self.actor(
                    next_states, need_log_prob=True
                )
                next_q_all = self.target_critic(next_states, next_actions)
            if wsrl_max_backup:
                pass
            elif self.variant == "uwmsg":
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
                target_values = next_q_all
                if self.variant == "wsrl":
                    # Official WSRL uses randint: REDQ heads are sampled with
                    # replacement, so duplicate indices are intentionally valid.
                    indices = self._sample_target_critic_indices(
                        next_q_all.shape[0]
                    )
                    next_q = self._wsrl_subsampled_min(next_q_all, indices)
                else:
                    next_q = target_values.min(dim=0).values
                if self.variant != "wsrl" or self.config.backup_entropy:
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
        cql_loss_enabled = self._cql_loss_enabled()
        if cql_loss_enabled:
            cql_penalty = self.config.cql_alpha * self._cql_penalty(
                states, next_states, actions
            )
            critic_loss = critic_loss + cql_penalty

        q_smoothness = states.new_tensor(0.0)
        ood_penalty = states.new_tensor(0.0)
        if self.variant == "ro2o" and not self.online_phase:
            q_smoothness = self._ro2o_q_smoothness(states, actions)
            ood_penalty = self._ro2o_ood_penalty(states)
            critic_loss = critic_loss + q_smoothness + ood_penalty

        critic_parameters = list(self.critic.parameters())
        if self.critic2 is not None:
            critic_parameters += list(self.critic2.parameters())
        critic_grad_norm = states.new_zeros(())
        critic_grad_norm_after = states.new_zeros(())
        if update_critic:
            self.critic_optimizer.zero_grad(set_to_none=True)
            critic_loss.backward()
            critic_grad_norm = (
                gradient_norm(critic_parameters)
                if self.config.max_grad_norm is None
                else torch.nn.utils.clip_grad_norm_(
                    critic_parameters, self.config.max_grad_norm
                )
            )
            critic_grad_norm_after = gradient_norm(critic_parameters)
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
        if defer_actor_temperature_step:
            # Upstream applies critic, actor, and temperature gradients from
            # one immutable Flax parameter tree.  Delay the disjoint PyTorch
            # optimizer steps until every WSRL loss has been evaluated from
            # the same pre-update parameters.
            self.actor_optimizer.step()
            self.alpha_optimizer.step()

        density_value = states.new_tensor(0.0)
        density_batches_available = (
            density_offline_batch is not None
            and density_online_batch is not None
        )
        if (
            update_critic
            and self.density_optimizer is not None
            and density_batches_available
        ):
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

        if update_critic:
            self.total_updates += 1
            self.critic_updates += 1
        if update_actor_temperature:
            self.actor_updates += 1
            self.temperature_updates += 1
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
            "cql_loss_enabled": float(cql_loss_enabled),
            "wsrl_online_cql_disabled": float(
                self.variant == "wsrl" and self.online_phase
            ),
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
            "actor_grad_norm": float(actor_grad_norm.item()),
            "critic_grad_norm": float(critic_grad_norm.item()),
            "actor_grad_norm_before_clip": float(actor_grad_norm.item()),
            "actor_grad_norm_after_clip": float(actor_grad_norm_after.item()),
            "critic_grad_norm_before_clip": float(critic_grad_norm.item()),
            "critic_grad_norm_after_clip": float(critic_grad_norm_after.item()),
            "number_of_actor_updates": float(update_actor_temperature),
            "number_of_critic_updates": float(update_critic),
            "number_of_temperature_updates": float(update_actor_temperature),
            "total_actor_updates": float(self.actor_updates),
            "total_critic_updates": float(self.critic_updates),
            "total_temperature_updates": float(self.temperature_updates),
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
