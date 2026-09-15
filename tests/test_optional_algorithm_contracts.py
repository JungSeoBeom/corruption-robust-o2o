"""Execution/gradient checks, not certificates of upstream equivalence.

See docs/optional-algorithm-audit.md for the unresolved source differences.
"""

import math
from unittest.mock import patch

import pytest
import torch

from robust_o2o.agents import build_agent
from robust_o2o.config import ExperimentConfig


def config(algorithm):
    return ExperimentConfig(
        algorithm,
        "hopper-medium-replay-v2",
        hidden_dim=16,
        batch_size=8,
        offline_steps=2,
        online_steps=2,
        ro2o_sample_size=3,
    )


def batch():
    return {
        "observations": torch.randn(8, 5),
        "actions": torch.tanh(torch.randn(8, 2)),
        "rewards": torch.randn(8),
        "next_observations": torch.randn(8, 5),
        "terminals": torch.tensor([0., 1., 0., 1., 0., 1., 0., 1.]),
    }


def snapshot(module):
    return {name: value.detach().clone() for name, value in module.named_parameters()}


def changed(before, module):
    return any(
        not torch.equal(before[name], parameter)
        for name, parameter in module.named_parameters()
    )


@pytest.mark.parametrize("algorithm", ("pex", "riql_pex", "uwmsg", "ro2o"))
def test_optional_agents_update_intended_modules_in_both_phases(algorithm):
    torch.manual_seed(71)
    agent = build_agent(config(algorithm), 5, 2, 1., torch.device("cpu"))
    data = batch()
    for phase in ("offline", "online"):
        frozen = None
        if phase == "online":
            agent.begin_online()
            if algorithm in ("pex", "riql_pex"):
                frozen = snapshot(agent.offline_actor)
                assert not any(p.requires_grad for p in agent.offline_actor.parameters())
                assert {id(p) for p in agent.actor.parameters()}.isdisjoint(
                    id(p) for p in agent.offline_actor.parameters()
                )
        modules = {"actor": agent.actor, "critic": agent.critic}
        if hasattr(agent, "value"):
            modules["value"] = agent.value
        before = {name: snapshot(module) for name, module in modules.items()}
        target_before = snapshot(agent.target_critic)
        for _ in range(2):
            metrics = agent.update(data)
            assert metrics
            assert all(math.isfinite(float(value)) for value in metrics.values())
        for name, module in modules.items():
            assert changed(before[name], module), (algorithm, phase, name)
            assert all(torch.isfinite(p).all() for p in module.parameters())
        assert changed(target_before, agent.target_critic)
        assert all(p.grad is None for p in agent.target_critic.parameters())
        if frozen is not None:
            assert not changed(frozen, agent.offline_actor)
            assert all(p.grad is None for p in agent.offline_actor.parameters())
        rng = torch.random.get_rng_state().clone()
        action = agent.select_action(data["observations"][0], evaluate=True)
        assert torch.equal(rng, torch.random.get_rng_state())
        assert action.shape == (2,)
        assert torch.isfinite(action).all()
        assert torch.all(action.abs() <= 1.)


@pytest.mark.parametrize("algorithm", ("pex", "riql_pex"))
def test_pex_gate_is_q_only_and_deterministic_ties_choose_offline(algorithm):
    torch.manual_seed(17)
    agent = build_agent(config(algorithm), 5, 2, 1., torch.device("cpu"))
    agent.begin_online()
    assert not agent.use_ipw
    states = torch.randn(3, 5)
    offline = torch.full((3, 2), -.25)
    online = torch.full((3, 2), .75)
    for q_online, expected in ((1., online), (0., offline), (-1., offline)):
        with patch.object(agent, "_actor_action", side_effect=[offline, online]), \
             patch.object(agent, "_aggregate_q", side_effect=[
                 torch.zeros(3), torch.full((3,), q_online)
             ]), \
             patch.object(agent.value, "forward", side_effect=AssertionError("IPW used")):
            selected = agent._expansion_action(states, evaluate=True)
        torch.testing.assert_close(selected, expected, rtol=0, atol=0)


def test_uwmsg_lcb_value_and_gradient():
    agent = build_agent(config("uwmsg"), 5, 2, 1., torch.device("cpu"))
    values = torch.tensor([[1., 4.], [3., 2.], [5., 6.]], requires_grad=True)
    actual = agent._q_for_policy(values)
    expected = values.mean(0) - agent.config.lcb_ratio * values.std(0)
    torch.testing.assert_close(actual, expected)
    actual.sum().backward()
    assert values.grad is not None
    assert torch.isfinite(values.grad).all()
    assert torch.count_nonzero(values.grad) > 0


def test_ro2o_offline_smoothness_has_parameter_gradients():
    torch.manual_seed(41)
    cfg = config("ro2o")
    cfg.ro2o_q_smooth_eps = .2
    cfg.ro2o_policy_smooth_eps = .2
    agent = build_agent(cfg, 5, 2, 1., torch.device("cpu"))
    data = batch()
    for loss_fn, module in (
        (lambda: agent._ro2o_q_smoothness(data["observations"], data["actions"]), agent.critic),
        (lambda: agent._ro2o_policy_smoothness(data["observations"]), agent.actor),
        (lambda: agent._ro2o_ood_penalty(data["observations"]), agent.critic),
    ):
        agent.zero_grad(set_to_none=True)
        loss = loss_fn()
        assert torch.isfinite(loss) and loss > 0
        loss.backward()
        gradients = [p.grad for p in module.parameters() if p.grad is not None]
        assert gradients and all(torch.isfinite(g).all() for g in gradients)
        assert any(torch.count_nonzero(g) > 0 for g in gradients)
