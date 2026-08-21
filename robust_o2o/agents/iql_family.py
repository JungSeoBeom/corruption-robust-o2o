from __future__ import annotations

import copy
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import ExperimentConfig
from ..networks import (
    DeterministicPolicy,
    DoubleQNetwork,
    EnsembleQNetwork,
    ExpansionGaussianPolicy,
    LegacyExpansionGaussianPolicy,
    OfficialRPEXGaussianPolicy,
    TanhGaussianPolicy,
    ValueNetwork,
)
from ..replay import TensorBatch
from .base import BaseAgent, soft_update


def expectile_loss(diff: torch.Tensor, expectile: float) -> torch.Tensor:
    weight = torch.where(diff > 0, expectile, 1.0 - expectile)
    return (weight * diff.square()).mean()


def robust_huber(diff: torch.Tensor, sigma: float) -> torch.Tensor:
    beta = 1.0 / (sigma**2)
    absolute = diff.abs()
    return torch.where(absolute < beta, 0.5 * absolute.square() / beta, absolute - 0.5 * beta)


def official_epsilon_greedy_sample(
    distribution_or_action: torch.distributions.Distribution | torch.Tensor,
    epsilon: float,
) -> torch.Tensor:
    """Literal port of RPEX ``epsilon_greedy_sample`` RNG semantics."""

    if torch.is_tensor(distribution_or_action):
        greedy_action = distribution_or_action
    elif isinstance(distribution_or_action, torch.distributions.Categorical):
        greedy_action = distribution_or_action.probs.argmax(dim=-1)
    else:
        greedy_action = distribution_or_action.mean
    if epsilon >= 1.0:
        if torch.is_tensor(distribution_or_action):
            return distribution_or_action + 0.01 * torch.randn_like(
                distribution_or_action
            )
        return distribution_or_action.sample()
    if epsilon == 0.0:
        return greedy_action
    sample_action = (
        distribution_or_action
        if torch.is_tensor(distribution_or_action)
        else distribution_or_action.sample()
    )
    # Pinned upstream draws this mask on the default CPU device after the
    # distribution sample, then uses it to index the model-device tensor.
    greedy_mask = torch.rand(sample_action.shape[0]) > epsilon
    if greedy_mask.device != sample_action.device:
        greedy_mask = greedy_mask.to(sample_action.device)
    result = sample_action.clone()
    result[greedy_mask] = greedy_action[greedy_mask]
    return result


class IQLFamilyAgent(BaseAgent):
    """IQL/RIQL pre-training plus the three RPEX/PEX online adapters."""

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
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.robust = config.algorithm in ("rpex", "riql_pex", "riql_naive")
        self.expansion = config.algorithm in ("rpex", "riql_pex", "pex")
        self.use_ipw = config.algorithm == "rpex"

        if self.robust:
            self.critic = EnsembleQNetwork(
                state_dim,
                action_dim,
                config.hidden_dim,
                config.hidden_layers,
                config.num_critics,
                rpex_profile=config.implementation_profile != "legacy_current",
            )
        else:
            self.critic = DoubleQNetwork(
                state_dim, action_dim, config.hidden_dim, config.hidden_layers
            )
        self.target_critic = copy.deepcopy(self.critic).requires_grad_(False)
        self.value = ValueNetwork(
            state_dim, config.hidden_dim, config.hidden_layers
        )
        self.actor = self._make_offline_actor()
        # Registered only after begin_online; None keeps offline checkpoints compact.
        self.offline_actor: nn.Module | None = None
        self.to(device)
        self._make_optimizers()

    def _make_offline_actor(self) -> nn.Module:
        if self.config.deterministic_policy:
            return DeterministicPolicy(
                self.state_dim,
                self.action_dim,
                self.config.hidden_dim,
                self.config.hidden_layers,
                self.max_action,
            )
        if self.config.action_distribution == "official_unsquashed_gaussian":
            return OfficialRPEXGaussianPolicy(
                self.state_dim,
                self.action_dim,
                self.config.hidden_dim,
                self.config.hidden_layers,
                self.max_action,
            )
        if self.config.algorithm == "rpex":
            policy_class = (
                ExpansionGaussianPolicy
                if self.config.action_distribution == "tanh_gaussian"
                else LegacyExpansionGaussianPolicy
            )
            return policy_class(
                self.state_dim,
                self.action_dim,
                self.config.hidden_dim,
                self.config.hidden_layers,
                self.max_action,
            )
        return TanhGaussianPolicy(
            self.state_dim,
            self.action_dim,
            self.config.hidden_dim,
            self.config.hidden_layers,
            self.max_action,
        )

    def _make_online_actor(self) -> nn.Module:
        policy_class = {
            "tanh_gaussian": ExpansionGaussianPolicy,
            "legacy_gaussian": LegacyExpansionGaussianPolicy,
            "official_unsquashed_gaussian": OfficialRPEXGaussianPolicy,
        }[self.config.action_distribution]
        return policy_class(
            self.state_dim,
            self.action_dim,
            self.config.hidden_dim,
            self.config.hidden_layers,
            self.max_action,
        ).to(self.device)

    def _make_optimizers(self) -> None:
        self.q_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=self.config.critic_learning_rate
        )
        self.value_optimizer = torch.optim.Adam(
            self.value.parameters(), lr=self.config.critic_learning_rate
        )
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=self.config.actor_learning_rate
        )
        self.actor_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.actor_optimizer, T_max=max(self.config.offline_steps, 1)
        )

    def begin_online(self) -> None:
        if self.online_phase:
            return
        fresh_online_optimizers = self.config.algorithm in (
            "rpex",
            "riql_naive",
            "riql_pex",
        )
        official_phase_transition = (
            self.config.implementation_profile == "official_code_reference"
            and fresh_online_optimizers
        )
        if self.expansion:
            self.offline_actor = copy.deepcopy(self.actor).eval().requires_grad_(False)
            if official_phase_transition:
                # attack_online.py starts a fresh process from args.seed before
                # constructing RPEX's new online policy. At minimum the active
                # policy initialization must not inherit the exhausted offline
                # training RNG stream.
                torch.manual_seed(self.config.learner_seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(self.config.learner_seed)
            self.actor = self._make_online_actor()
        if fresh_online_optimizers:
            # The pinned online constructors create fresh Adam instances and
            # load module weights only; no offline optimizer/scheduler state is
            # restored. Apply the same phase boundary to the custom-budget
            # research benchmark: otherwise RIQL-naive retains the exhausted
            # offline cosine schedule (and its Adam moments) while claiming to
            # perform online fine-tuning. Module weights remain untouched for
            # RIQL-naive; RPEX/RIQL+PEX have already installed their new online
            # policy above.
            self.q_optimizer = torch.optim.Adam(
                self.critic.parameters(), lr=self.config.critic_learning_rate
            )
            self.value_optimizer = torch.optim.Adam(
                self.value.parameters(), lr=self.config.critic_learning_rate
            )
            self.actor_optimizer = torch.optim.Adam(
                self.actor.parameters(), lr=self.config.actor_learning_rate
            )
            self.actor_scheduler = None
        elif self.expansion:
            self.actor_optimizer = torch.optim.Adam(
                self.actor.parameters(), lr=self.config.actor_learning_rate
            )
            self.actor_scheduler = None
        self.online_phase = True

    def _aggregate_q(self, critic: nn.Module, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        values = critic(states, actions)
        if self.robust:
            return torch.quantile(values, self.config.riql_quantile, dim=0)
        return values.min(dim=0).values

    def _actor_action(
        self, actor: nn.Module, states: torch.Tensor, deterministic: bool
    ) -> torch.Tensor:
        if isinstance(actor, DeterministicPolicy):
            return actor(states)
        return actor.act(states, deterministic=deterministic)

    def _expansion_action(
        self,
        states: torch.Tensor,
        evaluate: bool,
        return_components: bool = False,
        evaluation_mode: str = "deterministic_diagnostic",
    ):
        if self.offline_actor is None:
            raise RuntimeError("begin_online() must be called before policy expansion")
        offline_action = self._actor_action(self.offline_actor, states, True)
        evaluation_profile = self.config.evaluation_policy_profile
        method_faithful = evaluate and (
            evaluation_profile == "official_code_epsilon_switching"
            and evaluation_mode != "deterministic_diagnostic"
        )
        if method_faithful:
            if isinstance(self.actor, DeterministicPolicy):
                online_distribution = self.actor(states)
            else:
                online_distribution = self.actor.distribution(states)
            online_action = official_epsilon_greedy_sample(
                online_distribution, 0.1
            )
        elif evaluate:
            online_action = self._actor_action(self.actor, states, True)
        else:
            online_action = self._actor_action(self.actor, states, False)
        q_offline = self._aggregate_q(self.critic, states, offline_action)
        q_online = self._aggregate_q(self.critic, states, online_action)

        if self.use_ipw:
            value = self.value(states)
            online_log_prob = self.actor.log_prob(states, online_action.detach())
            online_ipw = ((q_online - value) / (online_log_prob.exp() + 1e-6)).clamp(
                -10_000.0, 100.0
            )
            if isinstance(self.offline_actor, DeterministicPolicy):
                offline_ipw = online_ipw
            else:
                offline_log_prob = self.offline_actor.log_prob(
                    states, offline_action.detach()
                )
                offline_ipw = (
                    (q_offline - value) / (offline_log_prob.exp() + 1e-6)
                ).clamp(-10_000.0, 100.0)
            coefficient = self.config.kappa / self.config.inv_temperature
            q_offline = q_offline + coefficient * offline_ipw
            q_online = q_online + coefficient * online_ipw

        logits = torch.stack((q_offline, q_online), dim=-1) * self.config.inv_temperature
        choice_distribution = torch.distributions.Categorical(logits=logits)
        if method_faithful:
            choice = official_epsilon_greedy_sample(
                choice_distribution, 0.1
            )
        elif evaluate:
            choice = logits.argmax(dim=-1)
        else:
            choice = choice_distribution.sample()
        result = torch.where(choice.unsqueeze(-1).bool(), online_action, offline_action)
        if return_components:
            return result, offline_action, online_action, choice.float().mean()
        return result

    def select_action(
        self,
        state: torch.Tensor,
        evaluate: bool = False,
        evaluation_mode: str = "deterministic_diagnostic",
    ) -> torch.Tensor:
        single = state.ndim == 1
        states = state.unsqueeze(0) if single else state
        if self.online_phase and self.expansion:
            action = self._expansion_action(
                states, evaluate, evaluation_mode=evaluation_mode
            )
        else:
            action = self._actor_action(self.actor, states, evaluate)
        return action.squeeze(0) if single else action

    def update(self, batch: TensorBatch) -> Dict[str, float]:
        states = batch["observations"]
        actions = batch["actions"]
        rewards = batch["rewards"].reshape(-1)
        next_states = batch["next_observations"]
        terminals = batch["terminals"].reshape(-1)

        with torch.no_grad():
            target_values_all = self.target_critic(states, actions)
            if self.robust:
                target_q = torch.quantile(
                    target_values_all, self.config.riql_quantile, dim=0
                )
            else:
                target_q = target_values_all.min(dim=0).values
            next_value = self.value(next_states)

        value = self.value(states)
        advantage = target_q - value
        value_loss = expectile_loss(advantage, self.config.expectile)
        self.value_optimizer.zero_grad(set_to_none=True)
        value_loss.backward()
        value_grad_norm = torch.nn.utils.clip_grad_norm_(
            self.value.parameters(), float("inf")
        )
        self.value_optimizer.step()

        targets = rewards + (1.0 - terminals) * self.config.discount * next_value
        predicted = self.critic(states, actions)
        if self.robust:
            targets = targets.clamp(-100.0, 1_000.0)
            q_loss = robust_huber(
                targets.unsqueeze(0) - predicted, self.config.riql_sigma
            ).mean()
        else:
            q_loss = F.mse_loss(predicted, targets.unsqueeze(0).expand_as(predicted))
        self.q_optimizer.zero_grad(set_to_none=True)
        q_loss.backward()
        critic_grad_norm = torch.nn.utils.clip_grad_norm_(
            self.critic.parameters(), float("inf")
        )
        self.q_optimizer.step()
        soft_update(
            self.target_critic, self.critic, self.config.target_update_rate
        )

        policy_actions = actions
        expansion_online_fraction = states.new_tensor(0.0)
        if self.online_phase and self.expansion:
            with torch.no_grad():
                (
                    policy_actions,
                    _,
                    _,
                    expansion_online_fraction,
                ) = self._expansion_action(states, evaluate=False, return_components=True)
                target_for_actor = self._aggregate_q(
                    self.target_critic, states, policy_actions
                )
                advantage = target_for_actor - self.value(states)

        if self.config.policy_extraction == "align_iql":
            weights = torch.exp(-self.config.beta * advantage.detach().square())
        else:
            weights = torch.exp(self.config.beta * advantage.detach()).clamp(max=100.0)
        if isinstance(self.actor, DeterministicPolicy):
            bc_loss = (self.actor(states) - policy_actions.detach()).square().sum(dim=-1)
        else:
            bc_loss = -self.actor.log_prob(states, policy_actions.detach())
        actor_loss = (weights * bc_loss).mean()
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        actor_grad_norm = torch.nn.utils.clip_grad_norm_(
            self.actor.parameters(), float("inf")
        )
        self.actor_optimizer.step()
        if self.actor_scheduler is not None and not self.online_phase:
            self.actor_scheduler.step()
        self.total_updates += 1
        self.actor_updates += 1
        self.critic_updates += 1
        absolute_td = (targets.unsqueeze(0) - predicted).detach().abs().reshape(-1)
        threshold = 1.0 / (self.config.riql_sigma**2)
        saturated = weights.detach() >= 100.0 - 1e-6
        policy_log_std = None
        if not isinstance(self.actor, DeterministicPolicy):
            policy_log_std = self.actor.distribution(states).stddev.log()
        metrics = {
            "q_loss": float(q_loss.item()),
            "critic_loss": float(q_loss.item()),
            "value_loss": float(value_loss.item()),
            "actor_loss": float(actor_loss.item()),
            "advantage_mean": float(advantage.mean().item()),
            "advantage_std": float(advantage.std(unbiased=False).item()),
            "awr_weight_p50": float(torch.quantile(weights.detach(), 0.5).item()),
            "awr_weight_p95": float(torch.quantile(weights.detach(), 0.95).item()),
            "awr_weight_max": float(weights.detach().max().item()),
            "awr_weight_saturation_fraction": float(saturated.float().mean().item()),
            "gradient_norm_actor": float(actor_grad_norm.item()),
            "gradient_norm_critic": float(critic_grad_norm.item()),
            "gradient_norm_value": float(value_grad_norm.item()),
            "number_of_actor_updates": 1.0,
            "number_of_critic_updates": 1.0,
            "robust_loss_threshold": float(threshold),
            "absolute_td_error_mean": float(absolute_td.mean().item()),
            "absolute_td_error_p95": float(torch.quantile(absolute_td, 0.95).item()),
            "linear_regime_fraction": float((absolute_td >= threshold).float().mean().item()),
            "quadratic_regime_fraction": float((absolute_td < threshold).float().mean().item()),
            "expansion_online_fraction": float(expansion_online_fraction.item()),
        }
        if policy_log_std is not None:
            metrics.update(
                policy_log_std_mean=float(policy_log_std.mean().item()),
                policy_log_std_min=float(policy_log_std.min().item()),
                policy_log_std_max=float(policy_log_std.max().item()),
            )
        return metrics

    def optimizer_state(self) -> Dict[str, object]:
        return {
            "q": self.q_optimizer.state_dict(),
            "value": self.value_optimizer.state_dict(),
            "actor": self.actor_optimizer.state_dict(),
            "actor_scheduler": (
                self.actor_scheduler.state_dict()
                if self.actor_scheduler is not None
                else None
            ),
        }

    def load_checkpoint_state(self, state: Dict[str, object]) -> None:
        saved_online = bool(state.get("online_phase", False))
        if saved_online and self.expansion and self.offline_actor is None:
            self.offline_actor = copy.deepcopy(self.actor).eval().requires_grad_(False)
            self.actor = self._make_online_actor()
            self.actor_optimizer = torch.optim.Adam(
                self.actor.parameters(), lr=self.config.actor_learning_rate
            )
            self.actor_scheduler = None
        super().load_checkpoint_state(state)

    def load_optimizer_state(self, state: Dict[str, object]) -> None:
        if "q" in state:
            self.q_optimizer.load_state_dict(state["q"])
        if "value" in state:
            self.value_optimizer.load_state_dict(state["value"])
        if "actor" in state:
            self.actor_optimizer.load_state_dict(state["actor"])
        if self.actor_scheduler is not None and state.get("actor_scheduler") is not None:
            self.actor_scheduler.load_state_dict(state["actor_scheduler"])
