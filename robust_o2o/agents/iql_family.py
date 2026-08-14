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
        if self.config.algorithm == "rpex":
            return ExpansionGaussianPolicy(
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

    def _make_online_actor(self) -> ExpansionGaussianPolicy:
        return ExpansionGaussianPolicy(
            self.state_dim,
            self.action_dim,
            self.config.hidden_dim,
            self.config.hidden_layers,
            self.max_action,
        ).to(self.device)

    def _make_optimizers(self) -> None:
        learning_rate = self.config.learning_rate
        self.q_optimizer = torch.optim.Adam(self.critic.parameters(), lr=learning_rate)
        self.value_optimizer = torch.optim.Adam(self.value.parameters(), lr=learning_rate)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=learning_rate)

    def begin_online(self) -> None:
        if self.online_phase:
            return
        if self.expansion:
            self.offline_actor = copy.deepcopy(self.actor).eval().requires_grad_(False)
            self.actor = self._make_online_actor()
            self.actor_optimizer = torch.optim.Adam(
                self.actor.parameters(), lr=self.config.learning_rate
            )
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
        self, states: torch.Tensor, evaluate: bool, return_components: bool = False
    ):
        if self.offline_actor is None:
            raise RuntimeError("begin_online() must be called before policy expansion")
        offline_action = self._actor_action(self.offline_actor, states, True)
        if evaluate:
            online_action = self._actor_action(self.actor, states, True)
            sampled_online_action = self._actor_action(self.actor, states, False)
            sample_mask = (
                torch.rand(states.shape[0], device=states.device) < 0.1
            ).unsqueeze(-1)
            online_action = torch.where(
                sample_mask, sampled_online_action, online_action
            )
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
        if evaluate:
            choice = logits.argmax(dim=-1)
            # RPEX evaluates the policy-expansion categorical with eps=0.1.
            random_mask = torch.rand(choice.shape, device=choice.device) < 0.1
            random_choice = torch.randint(0, 2, choice.shape, device=choice.device)
            choice = torch.where(random_mask, random_choice, choice)
        else:
            choice = torch.distributions.Categorical(logits=logits).sample()
        result = torch.where(choice.unsqueeze(-1).bool(), online_action, offline_action)
        if return_components:
            return result, offline_action, online_action, choice.float().mean()
        return result

    def select_action(self, state: torch.Tensor, evaluate: bool = False) -> torch.Tensor:
        single = state.ndim == 1
        states = state.unsqueeze(0) if single else state
        if self.online_phase and self.expansion:
            action = self._expansion_action(states, evaluate)
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

        weights = torch.exp(self.config.beta * advantage.detach()).clamp(max=100.0)
        if isinstance(self.actor, DeterministicPolicy):
            bc_loss = (self.actor(states) - policy_actions.detach()).square().sum(dim=-1)
        else:
            bc_loss = -self.actor.log_prob(states, policy_actions.detach())
        actor_loss = (weights * bc_loss).mean()
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_optimizer.step()
        self.total_updates += 1
        return {
            "q_loss": float(q_loss.item()),
            "value_loss": float(value_loss.item()),
            "actor_loss": float(actor_loss.item()),
            "advantage_mean": float(advantage.mean().item()),
            "expansion_online_fraction": float(expansion_online_fraction.item()),
        }

    def optimizer_state(self) -> Dict[str, object]:
        return {
            "q": self.q_optimizer.state_dict(),
            "value": self.value_optimizer.state_dict(),
            "actor": self.actor_optimizer.state_dict(),
        }

    def load_checkpoint_state(self, state: Dict[str, object]) -> None:
        saved_online = bool(state.get("online_phase", False))
        if saved_online and self.expansion and self.offline_actor is None:
            self.offline_actor = copy.deepcopy(self.actor).eval().requires_grad_(False)
            self.actor = self._make_online_actor()
            self.actor_optimizer = torch.optim.Adam(
                self.actor.parameters(), lr=self.config.learning_rate
            )
        super().load_checkpoint_state(state)

    def load_optimizer_state(self, state: Dict[str, object]) -> None:
        if "q" in state:
            self.q_optimizer.load_state_dict(state["q"])
        if "value" in state:
            self.value_optimizer.load_state_dict(state["value"])
        if "actor" in state:
            self.actor_optimizer.load_state_dict(state["actor"])
