"""Independent-policy Pessimistic Q-Ensemble for the D4RL-v2 benchmark.

This is a source-aligned port of shlee94/Off2OnRL at commit
``6f298fa9ef040d725067d0f2775022bd2900d635``.  The upstream online method
loads five independently trained CQL actors and twin critics, moment-matches
their pre-tanh Gaussian policies, and trains from a density-ratio-prioritized
union of offline and online transitions.

The class intentionally does not inherit from :class:`SACEnsembleAgent`:
there is no shared actor, critic trunk, encoder, or parameter storage.
"""

from __future__ import annotations

import copy
import hashlib
import math
from collections.abc import Mapping, Sequence
from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from ..cql import importance_sampled_cql
from ..config import ExperimentConfig
from ..dataset import assert_no_corruption_labels
from ..networks import QNetwork, TanhGaussianPolicy, mlp
from ..replay import TensorBatch
from .base import BaseAgent, gradient_norm, soft_update


PQE_ENSEMBLE_SIZE = 5
PQE_MEMBER_SEED_STRIDE = 4
PQE_MOMENT_LOG_STD_MIN = -20.0
PQE_MOMENT_LOG_STD_MAX = 2.0
PQE_CHECKPOINT_FORMAT = "pqe_independent_member_v1"


def _config_value(config: Any, names: Sequence[str], default: Any) -> Any:
    for name in names:
        value = getattr(config, name, None)
        if value is not None:
            return value
    return default


def _clone_state_dict(module: nn.Module) -> Dict[str, torch.Tensor]:
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in module.state_dict().items()
    }


def _update_fingerprint(digest: Any, value: Any, prefix: str = "") -> None:
    """Hash a nested checkpoint without depending on ``torch.save`` bytes."""

    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        digest.update(prefix.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
        return
    if isinstance(value, Mapping):
        for key in sorted(value, key=str):
            _update_fingerprint(digest, value[key], f"{prefix}/{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _update_fingerprint(digest, item, f"{prefix}/{index}")
        return
    digest.update(prefix.encode("utf-8"))
    digest.update(repr(value).encode("utf-8"))


def member_checkpoint_fingerprint(checkpoint: Mapping[str, Any]) -> str:
    """Return a stable content digest used to reject duplicated members."""

    digest = hashlib.sha256()
    _update_fingerprint(digest, checkpoint)
    return digest.hexdigest()


def _validated_sha256(value: str, label: str) -> str:
    normalized = str(value).lower()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ValueError(f"{label} must be a 64-character SHA-256 digest")
    return normalized


class NonnegativeDensityRatioNetwork(nn.Module):
    """Off2OnRL weight network with the public code's ReLU output."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        hidden_layers: int = 2,
    ) -> None:
        super().__init__()
        self.net = mlp(
            state_dim + action_dim,
            hidden_dim,
            hidden_layers,
            1,
        )
        final = next(
            module
            for module in reversed(self.net)
            if isinstance(module, nn.Linear)
        )
        # examples/ours.py initializes the final bias to one before training.
        nn.init.ones_(final.bias)

    def forward(
        self, states: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        inputs = torch.cat((states, actions), dim=-1)
        return F.relu(self.net(inputs).squeeze(-1))


class PessimisticQEnsembleAgent(BaseAgent):
    """Five independent CQL members followed by Off2OnRL online SAC.

    Offline training requires one independently sampled batch per member.  The
    experiment controller must construct those samplers from the *same*
    corrupted artifact using ``member_seeds``.  Online training requires a
    proportional priority batch plus separate offline and online batches for
    the density-ratio objective; passing a uniform RL batch is rejected.
    """

    def __init__(
        self,
        config: ExperimentConfig,
        state_dim: int,
        action_dim: int,
        max_action: float,
        device: torch.device,
    ) -> None:
        super().__init__(device)
        self.config = config
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.max_action = float(max_action)
        self.ensemble_size = int(
            _config_value(config, ("pqe_ensemble_size",), PQE_ENSEMBLE_SIZE)
        )
        if self.ensemble_size != PQE_ENSEMBLE_SIZE:
            raise ValueError(
                "Pessimistic Q-Ensemble main requires exactly five members; "
                "other sizes need a separately named ablation"
            )

        replay_mode = str(
            _config_value(
                config,
                ("pqe_replay_mode",),
                "balanced_density",
            )
        )
        if replay_mode != "balanced_density":
            raise ValueError(
                "Pessimistic Q-Ensemble main requires balanced density-ratio "
                "priority replay; uniform replay is an ablation"
            )
        self.replay_mode = replay_mode

        self.hidden_dim = int(
            _config_value(config, ("pqe_hidden_dim", "hidden_dim"), 256)
        )
        self.hidden_layers = int(
            _config_value(config, ("pqe_hidden_layers",), 2)
        )
        if self.hidden_layers != 2:
            raise ValueError(
                "The source-aligned Pessimistic Q-Ensemble uses two hidden layers"
            )

        self.base_seed = int(getattr(config, "seed", 0))
        self.member_seeds = tuple(
            self.base_seed + PQE_MEMBER_SEED_STRIDE * index
            for index in range(self.ensemble_size)
        )
        actors: list[TanhGaussianPolicy] = []
        q1_members: list[QNetwork] = []
        q2_members: list[QNetwork] = []
        target_q1_members: list[QNetwork] = []
        target_q2_members: list[QNetwork] = []
        for member_seed in self.member_seeds:
            # Construct on CPU under isolated RNG streams, then move the full
            # module once.  This avoids perturbing the benchmark learner RNG.
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(member_seed)
                actor = TanhGaussianPolicy(
                    state_dim,
                    action_dim,
                    self.hidden_dim,
                    self.hidden_layers,
                    max_action,
                )
                q1 = QNetwork(
                    state_dim,
                    action_dim,
                    self.hidden_dim,
                    self.hidden_layers,
                )
                q2 = QNetwork(
                    state_dim,
                    action_dim,
                    self.hidden_dim,
                    self.hidden_layers,
                )
            actors.append(actor)
            q1_members.append(q1)
            q2_members.append(q2)
            target_q1_members.append(
                copy.deepcopy(q1).requires_grad_(False)
            )
            target_q2_members.append(
                copy.deepcopy(q2).requires_grad_(False)
            )

        self.actors = nn.ModuleList(actors)
        self.q1_members = nn.ModuleList(q1_members)
        self.q2_members = nn.ModuleList(q2_members)
        self.target_q1_members = nn.ModuleList(target_q1_members)
        self.target_q2_members = nn.ModuleList(target_q2_members)
        self.offline_log_alphas = nn.ParameterList(
            nn.Parameter(torch.zeros(1))
            for _ in range(self.ensemble_size)
        )
        # Upstream online fine-tuning starts a single ensemble-policy alpha at 1.
        self.log_alpha = nn.Parameter(torch.zeros(1))
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.base_seed + 100_003)
            self.density_ratio = NonnegativeDensityRatioNetwork(
                state_dim,
                action_dim,
                self.hidden_dim,
                self.hidden_layers,
            )

        self.to(device)
        self.actor_learning_rate = float(
            _config_value(
                config,
                ("actor_learning_rate", "learning_rate"),
                3e-4,
            )
        )
        self.critic_learning_rate = float(
            _config_value(
                config,
                ("critic_learning_rate", "learning_rate"),
                3e-4,
            )
        )
        self.temperature_learning_rate = float(
            _config_value(
                config,
                ("temperature_learning_rate", "learning_rate"),
                3e-4,
            )
        )
        self.density_learning_rate = float(
            _config_value(
                config,
                ("pqe_weight_learning_rate", "learning_rate"),
                3e-4,
            )
        )
        self.actor_optimizers = [
            torch.optim.Adam(actor.parameters(), lr=self.actor_learning_rate)
            for actor in self.actors
        ]
        self.q1_optimizers = [
            torch.optim.Adam(q1.parameters(), lr=self.critic_learning_rate)
            for q1 in self.q1_members
        ]
        self.q2_optimizers = [
            torch.optim.Adam(q2.parameters(), lr=self.critic_learning_rate)
            for q2 in self.q2_members
        ]
        self.offline_alpha_optimizers = [
            torch.optim.Adam(
                [log_alpha], lr=self.temperature_learning_rate
            )
            for log_alpha in self.offline_log_alphas
        ]
        self.alpha_optimizer = torch.optim.Adam(
            [self.log_alpha], lr=self.temperature_learning_rate
        )
        self.density_optimizer = torch.optim.Adam(
            self.density_ratio.parameters(), lr=self.density_learning_rate
        )

        self.discount = float(getattr(config, "discount", 0.99))
        self.target_tau = float(
            _config_value(config, ("target_update_rate",), 0.005)
        )
        self.target_entropy = float(
            _config_value(config, ("target_entropy",), -float(action_dim))
        )
        self.cql_alpha = float(getattr(config, "cql_alpha", 5.0))
        self.cql_n_actions = int(getattr(config, "cql_n_actions", 10))
        self.cql_temperature = float(
            getattr(config, "cql_temperature", 1.0)
        )
        self.backup_entropy = bool(getattr(config, "backup_entropy", False))
        self.max_grad_norm = getattr(config, "max_grad_norm", None)

        self.priority_temperature = float(
            _config_value(
                config,
                ("pqe_priority_temperature", "balanced_replay_temperature"),
                5.0,
            )
        )
        self.priority_floor = float(getattr(config, "priority_floor", 1e-3))
        self.priority_ceiling = float(
            _config_value(
                config,
                ("pqe_priority_ceiling", "priority_ceiling"),
                1e3,
            )
        )
        self.init_online_fraction = float(
            _config_value(config, ("pqe_init_online_fraction",), 0.75)
        )
        self.first_epoch_multiplier = int(
            _config_value(config, ("pqe_first_epoch_multiplier",), 5)
        )
        self.first_online_block_steps = int(
            _config_value(config, ("pqe_first_online_block_steps",), 1_000)
        )
        self.online_buffer_size = int(
            _config_value(config, ("pqe_online_buffer_size",), 250_000)
        )
        self.weight_batch_size = int(
            _config_value(config, ("pqe_weight_batch_size",), 256)
        )
        if not 0.0 < self.init_online_fraction < 1.0:
            raise ValueError("pqe_init_online_fraction must lie strictly in (0, 1)")
        if self.priority_temperature <= 0.0:
            raise ValueError("pqe priority temperature must be positive")
        if self.priority_floor <= 0.0 or self.priority_ceiling < self.priority_floor:
            raise ValueError("invalid PQE priority clipping bounds")
        if self.first_epoch_multiplier < 1 or self.first_online_block_steps < 1:
            raise ValueError("invalid PQE first-online-block schedule")

        self.offline_updates_per_member = [0] * self.ensemble_size
        self.member_checkpoint_hashes: list[str | None] = [
            None
        ] * self.ensemble_size
        self.last_priority_values: torch.Tensor | None = None
        self.offline_artifact_identity: tuple[str, str] | None = None
        self.assert_independent_parameter_storage()

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    @property
    def total_offline_gradient_updates(self) -> int:
        return int(sum(self.offline_updates_per_member))

    def offline_alpha(self, member_index: int) -> torch.Tensor:
        return self.offline_log_alphas[member_index].exp()

    def begin_online(self) -> None:
        """Start Off2OnRL with fresh SAC optimizers and exact Q targets.

        The public implementation creates the online trainer only after loading
        the five CQL checkpoints.  Consequently, its actor/Q Adam moments are
        not inherited from CQL and its target critics start as hard copies of
        the loaded critics.  Preserve that boundary for ``stage=both`` while
        keeping the method idempotent for exact online resume.
        """

        if self.online_phase:
            return
        for index in range(self.ensemble_size):
            self.target_q1_members[index].load_state_dict(
                self.q1_members[index].state_dict(), strict=True
            )
            self.target_q2_members[index].load_state_dict(
                self.q2_members[index].state_dict(), strict=True
            )
        self.actor_optimizers = [
            torch.optim.Adam(
                actor.parameters(), lr=self.actor_learning_rate
            )
            for actor in self.actors
        ]
        self.q1_optimizers = [
            torch.optim.Adam(
                q1.parameters(), lr=self.critic_learning_rate
            )
            for q1 in self.q1_members
        ]
        self.q2_optimizers = [
            torch.optim.Adam(
                q2.parameters(), lr=self.critic_learning_rate
            )
            for q2 in self.q2_members
        ]
        with torch.no_grad():
            self.log_alpha.zero_()
        self.alpha_optimizer = torch.optim.Adam(
            [self.log_alpha], lr=self.temperature_learning_rate
        )
        self.density_optimizer = torch.optim.Adam(
            self.density_ratio.parameters(), lr=self.density_learning_rate
        )
        super().begin_online()

    def assert_independent_parameter_storage(self) -> None:
        """Fail if any actor/critic/target parameter shares tensor storage."""

        modules: list[tuple[str, nn.Module]] = []
        for index in range(self.ensemble_size):
            modules.extend(
                (
                    (f"actor_{index}", self.actors[index]),
                    (f"q1_{index}", self.q1_members[index]),
                    (f"q2_{index}", self.q2_members[index]),
                    (f"target_q1_{index}", self.target_q1_members[index]),
                    (f"target_q2_{index}", self.target_q2_members[index]),
                )
            )
        owners: dict[int, str] = {}
        for module_name, module in modules:
            for parameter_name, parameter in module.named_parameters():
                pointer = parameter.untyped_storage().data_ptr()
                owner = f"{module_name}.{parameter_name}"
                if pointer in owners:
                    raise RuntimeError(
                        "Pessimistic Q-Ensemble parameter storage is shared: "
                        f"{owners[pointer]} and {owner}"
                    )
                owners[pointer] = owner

    def bind_offline_artifact(self, cache_key: str, sha256: str) -> None:
        """Bind all member pretraining to one immutable corruption artifact."""

        identity = (str(cache_key), str(sha256))
        if self.offline_artifact_identity not in (None, identity):
            raise ValueError(
                "PQE members cannot switch corrupted offline artifacts"
            )
        self.offline_artifact_identity = identity

    def moment_parameters(
        self, states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Moment-match the five pre-tanh Gaussian member policies."""

        distributions = [actor.distribution(states) for actor in self.actors]
        means = torch.stack([distribution.mean for distribution in distributions])
        stds = torch.stack([distribution.stddev for distribution in distributions])
        average_mean = means.mean(dim=0)
        average_variance = (
            (stds.square() + means.square()).mean(dim=0)
            - average_mean.square()
        )
        average_std = average_variance.clamp_min(0.0).sqrt().clamp(
            math.exp(PQE_MOMENT_LOG_STD_MIN),
            math.exp(PQE_MOMENT_LOG_STD_MAX),
        )
        return average_mean, average_std

    def _moment_policy(
        self,
        states: torch.Tensor,
        *,
        deterministic: bool = False,
        need_log_prob: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor]:
        mean, std = self.moment_parameters(states)
        distribution = Normal(mean, std)
        raw_action = mean if deterministic else distribution.rsample()
        normalized_action = torch.tanh(raw_action)
        action_scale = self.actors[0].action_scale
        action_bias = self.actors[0].action_bias
        action = action_bias + action_scale * normalized_action
        log_prob = None
        if need_log_prob:
            log_prob = distribution.log_prob(raw_action).sum(dim=-1)
            jacobian = action_scale * (1.0 - normalized_action.square())
            log_prob = log_prob - torch.log(jacobian + 1e-6).sum(dim=-1)
        return action, log_prob, mean, std

    def select_action(
        self,
        state: torch.Tensor,
        evaluate: bool = False,
        evaluation_mode: str = "deterministic_diagnostic",
    ) -> torch.Tensor:
        del evaluation_mode
        single = state.ndim == 1
        states = state.unsqueeze(0) if single else state
        with torch.no_grad():
            action = self._moment_policy(
                states,
                deterministic=evaluate,
            )[0]
        return action.squeeze(0) if single else action

    def _stack_q(
        self,
        members: Sequence[nn.Module],
        states: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        return torch.stack(
            [member(states, actions) for member in members], dim=0
        )

    def ensemble_clipped_q(
        self, states: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        """Mean across member-wise clipped double-Q values."""

        q1 = self._stack_q(self.q1_members, states, actions)
        q2 = self._stack_q(self.q2_members, states, actions)
        return torch.minimum(q1, q2).mean(dim=0)

    @staticmethod
    def density_ratio_objective_from_weights(
        offline_weights: torch.Tensor,
        online_weights: torch.Tensor,
        epsilon: float = 1e-10,
    ) -> torch.Tensor:
        offline_term = -torch.log(
            2.0 / (offline_weights + 1.0) + epsilon
        )
        online_term = torch.log(
            2.0 * online_weights / (online_weights + 1.0) + epsilon
        )
        return offline_term.mean() - online_term.mean()

    def density_ratio_loss(
        self,
        offline_batch: TensorBatch,
        online_batch: TensorBatch,
    ) -> torch.Tensor:
        assert_no_corruption_labels(offline_batch)
        assert_no_corruption_labels(online_batch)
        offline_weights = self.density_ratio(
            offline_batch["observations"], offline_batch["actions"]
        )
        online_weights = self.density_ratio(
            online_batch["observations"], online_batch["actions"]
        )
        return self.density_ratio_objective_from_weights(
            offline_weights, online_weights
        )

    @staticmethod
    def priority_values_from_weights(
        weights: torch.Tensor,
        offline_weights: torch.Tensor,
        *,
        temperature: float = 5.0,
        floor: float = 1e-3,
        ceiling: float = 1e3,
        epsilon: float = 1e-10,
    ) -> torch.Tensor:
        numerator = weights.pow(1.0 / temperature)
        denominator = offline_weights.pow(1.0 / temperature).mean()
        return (numerator / (denominator + epsilon)).clamp(floor, ceiling)

    @torch.no_grad()
    def density_priorities(
        self,
        rl_batch: TensorBatch,
        offline_batch: TensorBatch,
    ) -> torch.Tensor:
        assert_no_corruption_labels(rl_batch)
        assert_no_corruption_labels(offline_batch)
        weights = self.density_ratio(
            rl_batch["observations"], rl_batch["actions"]
        )
        offline_weights = self.density_ratio(
            offline_batch["observations"], offline_batch["actions"]
        )
        return self.priority_values_from_weights(
            weights,
            offline_weights,
            temperature=self.priority_temperature,
            floor=self.priority_floor,
            ceiling=self.priority_ceiling,
        )

    def consume_priority_values(self) -> torch.Tensor | None:
        values = self.last_priority_values
        self.last_priority_values = None
        return values

    def initial_online_priority(
        self,
        offline_size: int,
        first_block_steps: int | None = None,
    ) -> float:
        """Priority yielding the requested initial online sampling fraction."""

        online_count = (
            self.first_online_block_steps
            if first_block_steps is None
            else int(first_block_steps)
        )
        if offline_size <= 0 or online_count <= 0:
            raise ValueError("offline and first-block sizes must be positive")
        fraction = self.init_online_fraction
        return float(offline_size * fraction / ((1.0 - fraction) * online_count))

    def online_update_count_for_block(
        self,
        block_index: int,
        normal_update_count: int,
    ) -> int:
        """Map the first collected block to the source demo's 5x updates."""

        if block_index < 0 or normal_update_count < 0:
            raise ValueError("block index and update count must be nonnegative")
        multiplier = self.first_epoch_multiplier if block_index == 0 else 1
        return int(normal_update_count * multiplier)

    def _offline_member_update(
        self,
        member_index: int,
        batch: TensorBatch,
    ) -> Dict[str, float]:
        if not 0 <= member_index < self.ensemble_size:
            raise IndexError("PQE member index out of range")
        assert_no_corruption_labels(batch)
        actor = self.actors[member_index]
        q1 = self.q1_members[member_index]
        q2 = self.q2_members[member_index]
        target_q1 = self.target_q1_members[member_index]
        target_q2 = self.target_q2_members[member_index]
        log_alpha = self.offline_log_alphas[member_index]
        alpha = log_alpha.exp()
        states = batch["observations"]
        actions = batch["actions"]
        rewards = batch["rewards"].reshape(-1)
        next_states = batch["next_observations"]
        terminals = batch["terminals"].reshape(-1)

        sampled_actions, log_prob, _, policy_std = actor(
            states, need_log_prob=True
        )
        alpha_loss = -(
            log_alpha * (log_prob + self.target_entropy).detach()
        ).mean()
        policy_q = torch.minimum(
            q1(states, sampled_actions), q2(states, sampled_actions)
        )
        actor_loss = (alpha.detach() * log_prob - policy_q).mean()

        with torch.no_grad():
            next_actions, next_log_prob, _, _ = actor(
                next_states, need_log_prob=True
            )
            next_q = torch.minimum(
                target_q1(next_states, next_actions),
                target_q2(next_states, next_actions),
            )
            if self.backup_entropy:
                next_q = next_q - alpha.detach() * next_log_prob
            target = rewards + (
                1.0 - terminals
            ) * self.discount * next_q

        q1_data = q1(states, actions)
        q2_data = q2(states, actions)
        td_loss = F.mse_loss(q1_data, target) + F.mse_loss(q2_data, target)
        cql_result = importance_sampled_cql(
            policy=actor,
            evaluators=(q1, q2),
            states=states,
            next_states=next_states,
            data_actions=actions,
            data_values=(q1_data, q2_data),
            num_actions=self.cql_n_actions,
            temperature=self.cql_temperature,
        )
        cql_loss = self.cql_alpha * cql_result.loss
        critic_loss = td_loss + cql_loss

        # Match the public trainer's compute-all-losses/then-step boundary so
        # targets and CQL proposals come from the pre-update CQL member.
        self.offline_alpha_optimizers[member_index].zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.offline_alpha_optimizers[member_index].step()
        self.actor_optimizers[member_index].zero_grad(set_to_none=True)
        actor_loss.backward()
        actor_grad_norm = self._clip_or_measure(actor.parameters())
        self.actor_optimizers[member_index].step()
        self.q1_optimizers[member_index].zero_grad(set_to_none=True)
        self.q2_optimizers[member_index].zero_grad(set_to_none=True)
        critic_loss.backward()
        critic_parameters = list(q1.parameters()) + list(q2.parameters())
        critic_grad_norm = self._clip_or_measure(critic_parameters)
        self.q1_optimizers[member_index].step()
        self.q2_optimizers[member_index].step()
        soft_update(target_q1, q1, self.target_tau)
        soft_update(target_q2, q2, self.target_tau)

        self.offline_updates_per_member[member_index] += 1
        self.total_updates += 1
        self.actor_updates += 1
        self.critic_updates += 1
        self.temperature_updates += 1
        return {
            "member_index": float(member_index),
            "member_seed": float(self.member_seeds[member_index]),
            "actor_loss": float(actor_loss.item()),
            "critic_loss": float(critic_loss.item()),
            "td_loss": float(td_loss.item()),
            "cql_loss": float(cql_loss.item()),
            "cql_loss_enabled": 1.0,
            "alpha_loss": float(alpha_loss.item()),
            "alpha": float(alpha.detach().item()),
            "actor_grad_norm": float(actor_grad_norm.item()),
            "critic_grad_norm": float(critic_grad_norm.item()),
            "policy_log_std_mean": float(policy_std.log().mean().item()),
            "offline_updates_for_member": float(
                self.offline_updates_per_member[member_index]
            ),
            "total_offline_gradient_updates": float(
                self.total_offline_gradient_updates
            ),
            "number_of_actor_updates": 1.0,
            "number_of_critic_updates": 1.0,
            "number_of_temperature_updates": 1.0,
        }

    def _clip_or_measure(self, parameters: Any) -> torch.Tensor:
        parameters = list(parameters)
        if self.max_grad_norm is None:
            return gradient_norm(parameters)
        return torch.nn.utils.clip_grad_norm_(
            parameters, float(self.max_grad_norm)
        )

    def update_offline_member(
        self,
        member_index: int,
        batch: TensorBatch,
    ) -> Dict[str, float]:
        if self.online_phase:
            raise RuntimeError("offline CQL member update requested online")
        return self._offline_member_update(member_index, batch)

    def update_offline_members(
        self, member_batches: Sequence[TensorBatch]
    ) -> Dict[str, float]:
        if len(member_batches) != self.ensemble_size:
            raise ValueError(
                "offline PQE update requires five independently sampled member batches"
            )
        results = [
            self.update_offline_member(index, batch)
            for index, batch in enumerate(member_batches)
        ]
        metrics: Dict[str, float] = {
            "number_of_actor_updates": float(self.ensemble_size),
            "number_of_critic_updates": float(self.ensemble_size),
            "number_of_temperature_updates": float(self.ensemble_size),
            "total_actor_updates": float(self.actor_updates),
            "total_critic_updates": float(self.critic_updates),
            "total_temperature_updates": float(self.temperature_updates),
            "total_offline_gradient_updates": float(
                self.total_offline_gradient_updates
            ),
            "pqe_member_count": float(self.ensemble_size),
            "cql_loss_enabled": 1.0,
        }
        for name in (
            "actor_loss",
            "critic_loss",
            "td_loss",
            "cql_loss",
            "alpha_loss",
            "alpha",
            "actor_grad_norm",
            "critic_grad_norm",
            "policy_log_std_mean",
        ):
            metrics[name] = float(
                sum(result[name] for result in results) / len(results)
            )
        return metrics

    def _online_update(
        self,
        rl_batch: TensorBatch,
        density_offline_batch: TensorBatch,
        density_online_batch: TensorBatch,
        *,
        rl_batch_prioritized: bool,
    ) -> Dict[str, float]:
        if not self.online_phase:
            raise RuntimeError("PQE online SAC update requires begin_online()")
        if not rl_batch_prioritized:
            raise ValueError(
                "PQE main online updates require proportional priority replay"
            )
        for batch in (
            rl_batch,
            density_offline_batch,
            density_online_batch,
        ):
            assert_no_corruption_labels(batch)

        states = rl_batch["observations"]
        actions = rl_batch["actions"]
        rewards = rl_batch["rewards"].reshape(-1)
        next_states = rl_batch["next_observations"]
        terminals = rl_batch["terminals"].reshape(-1)

        new_actions, log_prob, _, policy_std = self._moment_policy(
            states, need_log_prob=True
        )
        alpha_for_losses = self.alpha.detach().clone()
        alpha_loss = -(
            self.log_alpha * (log_prob + self.target_entropy).detach()
        ).mean()
        policy_q = self.ensemble_clipped_q(states, new_actions)
        actor_loss = (alpha_for_losses * log_prob - policy_q).mean()

        with torch.no_grad():
            next_actions, next_log_prob, _, _ = self._moment_policy(
                next_states, need_log_prob=True
            )
            target_q1 = self._stack_q(
                self.target_q1_members, next_states, next_actions
            )
            target_q2 = self._stack_q(
                self.target_q2_members, next_states, next_actions
            )
            target_values = torch.minimum(target_q1, target_q2)
            target_values = target_values - alpha_for_losses * next_log_prob
            q_target = rewards.unsqueeze(0) + (
                (1.0 - terminals) * self.discount
            ).unsqueeze(0) * target_values

        q1_pred = self._stack_q(self.q1_members, states, actions)
        q2_pred = self._stack_q(self.q2_members, states, actions)
        # Upstream sums member losses for each sample, then averages the batch.
        q1_loss = (q1_pred - q_target).square().sum(dim=0).mean()
        q2_loss = (q2_pred - q_target).square().sum(dim=0).mean()
        critic_loss = q1_loss + q2_loss
        density_loss = self.density_ratio_loss(
            density_offline_batch, density_online_batch
        )
        # The public trainer computes and applies replay priorities from the
        # same pre-update weight network used to form ``weight_loss``.
        self.last_priority_values = self.density_priorities(
            rl_batch, density_offline_batch
        )

        # OursTrainer.compute_loss builds every loss from one pre-update model
        # snapshot, then steps alpha, policy, Q1, Q2, and the weight network.
        self.alpha_optimizer.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.alpha_optimizer.step()
        for optimizer in self.actor_optimizers:
            optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        actor_parameters = [
            parameter
            for actor in self.actors
            for parameter in actor.parameters()
        ]
        actor_grad_norm = self._clip_or_measure(actor_parameters)
        for optimizer in self.actor_optimizers:
            optimizer.step()
        for optimizer in (*self.q1_optimizers, *self.q2_optimizers):
            optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        critic_parameters = [
            parameter
            for members in (self.q1_members, self.q2_members)
            for member in members
            for parameter in member.parameters()
        ]
        critic_grad_norm = self._clip_or_measure(critic_parameters)
        for optimizer in (*self.q1_optimizers, *self.q2_optimizers):
            optimizer.step()
        for index in range(self.ensemble_size):
            soft_update(
                self.target_q1_members[index],
                self.q1_members[index],
                self.target_tau,
            )
            soft_update(
                self.target_q2_members[index],
                self.q2_members[index],
                self.target_tau,
            )
        self.density_optimizer.zero_grad(set_to_none=True)
        density_loss.backward()
        self.density_optimizer.step()

        source = rl_batch.get("_source")
        offline_count = (
            int((source.reshape(-1) == 0).sum().item())
            if source is not None
            else 0
        )
        online_count = (
            int((source.reshape(-1) == 1).sum().item())
            if source is not None
            else 0
        )
        self.total_updates += 1
        self.actor_updates += 1
        self.critic_updates += 1
        self.temperature_updates += 1
        return {
            "actor_loss": float(actor_loss.item()),
            "critic_loss": float(critic_loss.item()),
            "q1_loss": float(q1_loss.item()),
            "q2_loss": float(q2_loss.item()),
            "alpha_loss": float(alpha_loss.item()),
            "alpha": float(self.alpha.item()),
            "density_loss": float(density_loss.item()),
            "actor_grad_norm": float(actor_grad_norm.item()),
            "critic_grad_norm": float(critic_grad_norm.item()),
            "ensemble_q_mean": float(
                torch.minimum(q1_pred, q2_pred).detach().mean().item()
            ),
            "policy_log_std_mean": float(policy_std.log().mean().item()),
            "priority_min": float(self.last_priority_values.min().item()),
            "priority_max": float(self.last_priority_values.max().item()),
            "priority_mean": float(self.last_priority_values.mean().item()),
            "rl_offline_count": float(offline_count),
            "rl_online_count": float(online_count),
            "density_offline_count": float(
                len(density_offline_batch["observations"])
            ),
            "density_online_count": float(
                len(density_online_batch["observations"])
            ),
            "rl_batch_prioritized": 1.0,
            "cql_loss_enabled": 0.0,
            "number_of_actor_updates": 1.0,
            "number_of_critic_updates": 1.0,
            "number_of_temperature_updates": 1.0,
            "total_actor_updates": float(self.actor_updates),
            "total_critic_updates": float(self.critic_updates),
            "total_temperature_updates": float(self.temperature_updates),
        }

    def update(
        self,
        batch: TensorBatch | None = None,
        *,
        member_batches: Sequence[TensorBatch] | None = None,
        rl_batch: TensorBatch | None = None,
        density_offline_batch: TensorBatch | None = None,
        density_online_batch: TensorBatch | None = None,
        rl_batch_prioritized: bool = False,
        **unsupported: Any,
    ) -> Dict[str, float]:
        if unsupported:
            names = ", ".join(sorted(unsupported))
            raise TypeError(f"unsupported PQE update arguments: {names}")
        if self.online_phase:
            if batch is not None:
                if rl_batch is not None:
                    raise ValueError("pass either batch or rl_batch, not both")
                rl_batch = batch
            if (
                rl_batch is None
                or density_offline_batch is None
                or density_online_batch is None
            ):
                raise ValueError(
                    "online PQE requires priority RL, offline-density, and "
                    "online-density batches"
                )
            if member_batches is not None:
                raise ValueError("member_batches are offline-only")
            return self._online_update(
                rl_batch,
                density_offline_batch,
                density_online_batch,
                rl_batch_prioritized=rl_batch_prioritized,
            )
        if batch is not None:
            raise ValueError(
                "offline PQE requires five independent member_batches; a "
                "single shared minibatch is not accepted"
            )
        if rl_batch is not None:
            raise ValueError("rl_batch is online-only")
        if member_batches is None:
            raise ValueError("offline PQE requires member_batches")
        return self.update_offline_members(member_batches)

    def member_checkpoint_state(self, member_index: int) -> Dict[str, Any]:
        if not 0 <= member_index < self.ensemble_size:
            raise IndexError("PQE member index out of range")
        return {
            "format": PQE_CHECKPOINT_FORMAT,
            "member_index": member_index,
            "member_seed": self.member_seeds[member_index],
            "offline_updates": self.offline_updates_per_member[member_index],
            "actor": _clone_state_dict(self.actors[member_index]),
            "q1": _clone_state_dict(self.q1_members[member_index]),
            "q2": _clone_state_dict(self.q2_members[member_index]),
            "target_q1": _clone_state_dict(
                self.target_q1_members[member_index]
            ),
            "target_q2": _clone_state_dict(
                self.target_q2_members[member_index]
            ),
        }

    def member_checkpoint_states(self) -> list[Dict[str, Any]]:
        return [
            self.member_checkpoint_state(index)
            for index in range(self.ensemble_size)
        ]

    @staticmethod
    def _validate_module_state(
        module: nn.Module,
        state: Mapping[str, torch.Tensor],
        label: str,
    ) -> None:
        expected = module.state_dict()
        missing = sorted(set(expected).difference(state))
        unexpected = sorted(set(state).difference(expected))
        if missing or unexpected:
            raise ValueError(
                f"{label} state keys mismatch; missing={missing}, "
                f"unexpected={unexpected}"
            )
        for name, expected_tensor in expected.items():
            loaded = state[name]
            if not isinstance(loaded, torch.Tensor):
                raise TypeError(f"{label}.{name} is not a tensor")
            if loaded.shape != expected_tensor.shape:
                raise ValueError(
                    f"{label}.{name} shape mismatch: expected "
                    f"{tuple(expected_tensor.shape)}, got {tuple(loaded.shape)}"
                )

    def load_member_checkpoint_states(
        self,
        checkpoints: Sequence[Mapping[str, Any]],
        checkpoint_hashes: Sequence[str] | None = None,
    ) -> None:
        """Strictly load five unique, independently seeded CQL members."""

        if len(checkpoints) != self.ensemble_size:
            raise ValueError("exactly five PQE member checkpoints are required")
        content_hashes = [
            member_checkpoint_fingerprint(checkpoint)
            for checkpoint in checkpoints
        ]
        if len(set(content_hashes)) != self.ensemble_size:
            raise ValueError(
                "duplicate PQE member checkpoint content is forbidden"
            )
        if checkpoint_hashes is not None:
            if len(checkpoint_hashes) != self.ensemble_size:
                raise ValueError("exactly five PQE checkpoint hashes are required")
            recorded_hashes = [
                _validated_sha256(value, f"member checkpoint hash {index}")
                for index, value in enumerate(checkpoint_hashes)
            ]
            if len(set(recorded_hashes)) != self.ensemble_size:
                raise ValueError("duplicate PQE member checkpoint hashes are forbidden")
        else:
            recorded_hashes = content_hashes

        module_names = (
            "actor",
            "q1",
            "q2",
            "target_q1",
            "target_q2",
        )
        module_lists: tuple[Sequence[nn.Module], ...] = (
            self.actors,
            self.q1_members,
            self.q2_members,
            self.target_q1_members,
            self.target_q2_members,
        )
        updates: list[int] = []
        for index, checkpoint in enumerate(checkpoints):
            if checkpoint.get("format") != PQE_CHECKPOINT_FORMAT:
                raise ValueError(f"member {index} has an unsupported checkpoint format")
            if int(checkpoint.get("member_index", -1)) != index:
                raise ValueError(f"member checkpoint {index} has the wrong index")
            if int(checkpoint.get("member_seed", -1)) != self.member_seeds[index]:
                raise ValueError(f"member checkpoint {index} has the wrong seed")
            for name, modules in zip(module_names, module_lists):
                if name not in checkpoint:
                    raise ValueError(f"member checkpoint {index} is missing {name}")
                self._validate_module_state(
                    modules[index],
                    checkpoint[name],
                    f"member_{index}.{name}",
                )
            updates.append(int(checkpoint.get("offline_updates", 0)))

        # Mutate only after every member has passed structural validation.
        for index, checkpoint in enumerate(checkpoints):
            for name, modules in zip(module_names, module_lists):
                modules[index].load_state_dict(checkpoint[name], strict=True)
        self.offline_updates_per_member = updates
        self.member_checkpoint_hashes = recorded_hashes
        self.assert_independent_parameter_storage()

    def record_member_checkpoint_hash(
        self, member_index: int, checkpoint_sha256: str
    ) -> None:
        if not 0 <= member_index < self.ensemble_size:
            raise IndexError("PQE member index out of range")
        value = _validated_sha256(
            checkpoint_sha256, f"member checkpoint hash {member_index}"
        )
        for index, existing in enumerate(self.member_checkpoint_hashes):
            if index != member_index and existing == value:
                raise ValueError("duplicate PQE member checkpoint hashes are forbidden")
        self.member_checkpoint_hashes[member_index] = value

    def checkpoint_state(self) -> Dict[str, Any]:
        state = super().checkpoint_state()
        state["pqe"] = {
            "format": "pqe_independent_ensemble_v1",
            "ensemble_size": self.ensemble_size,
            "base_seed": self.base_seed,
            "member_seeds": self.member_seeds,
            "offline_updates_per_member": tuple(
                self.offline_updates_per_member
            ),
            "member_checkpoint_hashes": tuple(
                self.member_checkpoint_hashes
            ),
            "offline_artifact_identity": self.offline_artifact_identity,
        }
        return state

    def load_checkpoint_state(self, state: Dict[str, Any]) -> None:
        metadata = state.get("pqe")
        if not isinstance(metadata, Mapping):
            raise ValueError("checkpoint is missing PQE ensemble metadata")
        if metadata.get("format") != "pqe_independent_ensemble_v1":
            raise ValueError("unsupported PQE ensemble checkpoint format")
        if int(metadata.get("ensemble_size", -1)) != self.ensemble_size:
            raise ValueError("PQE checkpoint ensemble size mismatch")
        if tuple(metadata.get("member_seeds", ())) != self.member_seeds:
            raise ValueError("PQE checkpoint member seeds mismatch")
        hashes = list(metadata.get("member_checkpoint_hashes", ()))
        nonempty_hashes = [value for value in hashes if value is not None]
        if len(hashes) != self.ensemble_size:
            raise ValueError("PQE checkpoint hash metadata is incomplete")
        if len(nonempty_hashes) != len(set(nonempty_hashes)):
            raise ValueError("PQE checkpoint contains duplicate member hashes")

        self.load_state_dict(state["model"], strict=True)
        self.load_optimizer_state(state.get("optimizers", {}))
        self.online_phase = bool(state.get("online_phase", False))
        self.total_updates = int(state.get("total_updates", 0))
        self.actor_updates = int(state.get("actor_updates", 0))
        self.critic_updates = int(state.get("critic_updates", 0))
        self.temperature_updates = int(state.get("temperature_updates", 0))
        updates = list(metadata.get("offline_updates_per_member", ()))
        if len(updates) != self.ensemble_size:
            raise ValueError("PQE offline member update metadata is incomplete")
        self.offline_updates_per_member = [int(value) for value in updates]
        self.member_checkpoint_hashes = hashes
        artifact = metadata.get("offline_artifact_identity")
        self.offline_artifact_identity = (
            tuple(artifact) if artifact is not None else None
        )
        self.assert_independent_parameter_storage()

    def optimizer_state(self) -> Dict[str, Any]:
        return {
            "actors": [
                optimizer.state_dict() for optimizer in self.actor_optimizers
            ],
            "q1": [optimizer.state_dict() for optimizer in self.q1_optimizers],
            "q2": [optimizer.state_dict() for optimizer in self.q2_optimizers],
            "offline_alpha": [
                optimizer.state_dict()
                for optimizer in self.offline_alpha_optimizers
            ],
            "online_alpha": self.alpha_optimizer.state_dict(),
            "density_ratio": self.density_optimizer.state_dict(),
        }

    def load_optimizer_state(self, state: Dict[str, Any]) -> None:
        if not state:
            return
        groups = (
            ("actors", self.actor_optimizers),
            ("q1", self.q1_optimizers),
            ("q2", self.q2_optimizers),
            ("offline_alpha", self.offline_alpha_optimizers),
        )
        for name, optimizers in groups:
            saved = state.get(name)
            if not isinstance(saved, Sequence) or len(saved) != self.ensemble_size:
                raise ValueError(f"PQE optimizer state {name} must contain five members")
            for optimizer, optimizer_state in zip(optimizers, saved):
                optimizer.load_state_dict(optimizer_state)
        self.alpha_optimizer.load_state_dict(state["online_alpha"])
        self.density_optimizer.load_state_dict(state["density_ratio"])

    def algorithm_metadata(self) -> Dict[str, Any]:
        """Agent-level fields consumed by run manifests and readiness checks."""

        return {
            "pqe_member_count": self.ensemble_size,
            "ensemble_size": self.ensemble_size,
            "pqe_member_seeds": list(self.member_seeds),
            "pqe_member_checkpoint_hashes": list(
                self.member_checkpoint_hashes
            ),
            "offline_updates_per_member": list(
                self.offline_updates_per_member
            ),
            "total_offline_gradient_updates": self.total_offline_gradient_updates,
            "offline_compute_multiplier": self.ensemble_size,
            "evaluation_policy": "tanh_of_moment_matched_pre_tanh_mean",
            "actor_independence": True,
            "critic_independence": True,
            "shared_actor": False,
            "shared_critic": False,
            "pqe_replay_mode": self.replay_mode,
            "priority_temperature": self.priority_temperature,
            "priority_floor": self.priority_floor,
            "priority_ceiling": self.priority_ceiling,
            "init_online_fraction": self.init_online_fraction,
            "first_epoch_multiplier": self.first_epoch_multiplier,
            "first_online_block_steps": self.first_online_block_steps,
            "online_buffer_size": self.online_buffer_size,
            "weight_network_batch_size": self.weight_batch_size,
            "task_scope": "d4rl_v2_port",
            "upstream_task_version": "v0",
            "benchmark_task_version": "v2",
        }
