from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
import torch


QEvaluator = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


@dataclass
class CQLResult:
    """Importance-sampled CQL(H) differences before the external weight."""

    differences: tuple[torch.Tensor, ...]
    calibration_bound_rate: torch.Tensor

    @property
    def loss(self) -> torch.Tensor:
        return sum((value.mean() for value in self.differences), start=self.differences[0].new_zeros(()))


def _sample_policy_actions(policy, states: torch.Tensor, count: int):
    batch_size = states.shape[0]
    expanded = states[:, None, :].expand(-1, count, -1).reshape(batch_size * count, -1)
    actions, log_prob, _, _ = policy(expanded, need_log_prob=True)
    return actions.reshape(batch_size, count, -1), log_prob.reshape(batch_size, count)


def importance_sampled_cql(
    *,
    policy,
    evaluators: Sequence[QEvaluator],
    states: torch.Tensor,
    next_states: torch.Tensor,
    data_actions: torch.Tensor,
    data_values: Sequence[torch.Tensor],
    num_actions: int,
    temperature: float = 1.0,
    calibration_lower_bound: torch.Tensor | None = None,
    calibration_valid: torch.Tensor | None = None,
) -> CQLResult:
    """Compute the common CQL random/current/next-policy log-sum-exp term.

    Evaluators may return ``[B, K]`` (one critic) or ``[N, B, K]`` (an
    ensemble). Policy samples are detached so this loss updates critics only.
    """

    batch_size, action_dim = data_actions.shape
    random_actions = torch.empty(
        batch_size, num_actions, action_dim, device=states.device
    ).uniform_(-policy.max_action, policy.max_action)
    current_actions, current_log_prob = _sample_policy_actions(policy, states, num_actions)
    next_actions, next_log_prob = _sample_policy_actions(policy, next_states, num_actions)
    random_density = np.log((0.5 / policy.max_action) ** action_dim)

    differences = []
    rates = []
    for evaluate, data_q in zip(evaluators, data_values):
        q_random = evaluate(states, random_actions)
        q_current = evaluate(states, current_actions.detach())
        # CQL evaluates next-state policy proposals against the current states.
        q_next = evaluate(states, next_actions.detach())
        proposal_axis = q_current.ndim - 1
        expand_prefix = [1] * (q_current.ndim - 2)
        current_lp = current_log_prob.reshape(*expand_prefix, batch_size, num_actions)
        next_lp = next_log_prob.reshape(*expand_prefix, batch_size, num_actions)

        if calibration_lower_bound is not None and calibration_valid is not None:
            lower = calibration_lower_bound.reshape(*expand_prefix, batch_size, 1)
            valid = calibration_valid.bool().reshape(*expand_prefix, batch_size, 1)
            expanded_valid = valid.expand_as(q_current)
            if expanded_valid.any():
                rates.extend(
                    ((values[expanded_valid] < lower.expand_as(values)[expanded_valid]).float().mean()
                     for values in (q_current, q_next))
                )
            q_current = torch.where(valid, torch.maximum(q_current, lower), q_current)
            q_next = torch.where(valid, torch.maximum(q_next, lower), q_next)

        candidates = torch.cat(
            (q_random - random_density, q_next - next_lp.detach(), q_current - current_lp.detach()),
            dim=proposal_axis,
        )
        differences.append(
            temperature
            * torch.logsumexp(candidates / temperature, dim=proposal_axis)
            - data_q
        )

    rate = torch.stack(rates).mean() if rates else states.new_zeros(())
    return CQLResult(tuple(differences), rate)
