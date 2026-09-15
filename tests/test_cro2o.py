import copy
from unittest.mock import patch

import numpy as np
import pytest
import torch

from robust_o2o.agents import build_agent
from robust_o2o.agents.cro2o import (
    SourceAudit, center_uncertainty, trust_score, retention_target,
    fixed_mask, weighted_mean, register_offline_blocks,
)
from robust_o2o.config import ExperimentConfig, CANDIDATE_ALGORITHMS
from robust_o2o.cro2o_training import prepare_candidate, candidate_update
from robust_o2o.environment import StateNormalizer, apply_normalizer, evaluate_agent
from robust_o2o.experiment import _run_online, capture_global_rng_state, restore_global_rng_state
from robust_o2o.replay import OfflineDataset, ReplayBuffer
from test_main_controller import _GymnasiumEnv, _logger
from test_reliability_regressions import assert_nested_equal


def fixture(algorithm="care_o2o", **overrides):
    values = dict(hidden_dim=8, hidden_layers=1, num_critics=5, batch_size=4,
                  offline_steps=2, online_steps=6, initial_collection_steps=2, warmup_steps=2,
                  candidate_audit_steps=2, candidate_generator_steps=2,
                  candidate_recalibration_steps=2, candidate_candidates=3,
                  candidate_diffusion_steps=3, candidate_retention_period=2,
                  replay_size=32, eval_period=10000, train_log_period=10000,
                  checkpoint_period=10000)
    values.update(overrides)
    config = ExperimentConfig(algorithm, "hopper-medium-replay-v2", **values)
    rng = np.random.default_rng(17)
    data = dict(observations=rng.normal(size=(16, 2)).astype(np.float32),
                actions=rng.uniform(-.8, .8, (16, 1)).astype(np.float32),
                next_observations=rng.normal(size=(16, 2)).astype(np.float32),
                rewards=rng.normal(size=16).astype(np.float32),
                terminals=np.zeros(16, np.float32), episode_id=np.repeat(np.arange(4), 4))
    normalizer = StateNormalizer.fit(data, enabled=config.normalize_states)
    offline = OfflineDataset(apply_normalizer(data, normalizer), 3)
    torch.manual_seed(11)
    agent = build_agent(config, 2, 1, 1., torch.device("cpu"))
    register_offline_blocks(agent, data)
    return config, data, normalizer, offline, agent


def prepared(algorithm="care_o2o", **overrides):
    config, data, norm, offline, agent = fixture(algorithm, **overrides)
    for _ in range(2):
        agent.update(offline.sample(4, torch.device("cpu")))
    prepare_candidate(agent, data, offline)
    agent.begin_online()
    return config, data, norm, offline, agent


def fill_online(agent):
    replay = ReplayBuffer(2, 1, 32, 4)
    for i in range(4):
        state = np.array([i * .1, 0.], np.float32)
        action = np.array([.2], np.float32)
        agent.observe_transition(state, action, .3, state + .1, 0., replay.position, i % 2 == 1)
        replay.add(state, action, .3, state + .1, 0.)
    return replay


def test_robust_center_and_trust_formula():
    config, *_ = fixture()
    q = torch.tensor([[1., 2.], [2., 3.], [3., 4.], [4., 5.], [100., 99.]])
    center, u = center_uncertainty(q, 1.)
    torch.testing.assert_close(center, torch.tensor([3., 4.]))
    torch.testing.assert_close(u, torch.ones(2))
    c, w = trust_score(torch.tensor([8., 8.]), torch.tensor([0., 5.]), 2., config)
    torch.testing.assert_close(c, torch.tensor([2., 0.]))
    assert w[1] == 1 and 0 < w[0] < 1
    assert not c.requires_grad and not w.requires_grad


def test_bootstrap_is_fixed_by_source_block_not_replay_draw():
    a = fixed_mask(42, 0, 8, 100, .5)
    np.testing.assert_array_equal(a, fixed_mask(42, 0, 8, 100, .5))
    assert not np.array_equal(a, fixed_mask(42, 1, 8, 100, .5))
    assert not np.array_equal(a, fixed_mask(42, 0, 9, 100, .5))


def test_source_rule_and_distinct_block_counts():
    config, *_ = fixture()
    audit = SourceAudit(config)
    for _ in range(10):
        audit.record(0, 1., 3., .1)
    assert audit.diagnostics()["count"] == audit.diagnostics()["effective_count"] == 1
    clean = dict(count=10, suspicion=0., effective_count=10., scale=1.)
    scarce = dict(count=1, suspicion=0., effective_count=1., scale=1.)
    assert retention_target(clean, scarce, 1.) == pytest.approx(1 / 1.1)
    dirty = {**clean, "suspicion": 1.}
    assert retention_target(dirty, scarce, 10.) < retention_target(clean, scarce, 10.)
    assert retention_target(clean, {**scarce, "count": 0}, 1.) == 1


def test_batch_weight_normalization_does_not_reduce_uniformly_low_source():
    values = torch.tensor([1., 9.])
    assert weighted_mean(values, torch.ones(2)) == pytest.approx(5.)
    assert weighted_mean(values, torch.full((2,), .05)) == pytest.approx(5.)


def test_matched_initializers_and_independent_critics():
    agents = [fixture(name)[-1] for name in CANDIDATE_ALGORITHMS]
    for other in agents[1:]:
        assert_nested_equal(agents[0].critics.state_dict(), other.critics.state_dict())
        assert_nested_equal(agents[0].actor.state_dict(), other.actor.state_dict())
        assert_nested_equal(agents[0].value.state_dict(), other.value.state_dict())
    weights = [next(head.parameters()) for head in agents[0].critics]
    assert len({w.data_ptr() for w in weights}) == 5
    assert not torch.equal(weights[0], weights[1])


def test_crossfit_models_fit_only_disjoint_blocks_and_preserve_main_rng():
    config, data, norm, offline, agent = fixture()
    trained_blocks = []
    original_fit = StateNormalizer.fit

    def record(training, **kwargs):
        trained_blocks.append(set(training["episode_id"].tolist()))
        return original_fit(training, **kwargs)

    before_rng = capture_global_rng_state()
    before = copy.deepcopy(agent.state_dict())
    with patch("robust_o2o.cro2o_training.StateNormalizer.fit", side_effect=record):
        prepare_candidate(agent, data, offline)
    assert trained_blocks == [{1, 3}, {0, 2}]
    assert not trained_blocks[0].intersection(trained_blocks[1])
    assert_nested_equal(before, agent.state_dict())
    assert_nested_equal(before_rng, capture_global_rng_state())
    assert agent.audit_updates == 4
    assert agent.audit[0].diagnostics()["count"] == 4
    assert np.isfinite(agent.offline_weights).all()


@pytest.mark.parametrize("algorithm", CANDIDATE_ALGORITHMS)
def test_prequential_score_and_recalibration_leave_actor_unchanged(algorithm):
    config, data, norm, offline, agent = prepared(algorithm)
    before = copy.deepcopy(agent.state_dict())
    replay = fill_online(agent)
    assert_nested_equal(before, agent.state_dict())
    weights = copy.deepcopy(agent.online_metadata)
    actor = copy.deepcopy((agent.proposal if agent.generative else agent.actor).state_dict())
    q_before = copy.deepcopy(agent.critics.state_dict())
    for _ in range(2):
        candidate_update(agent, offline, replay, config, critic_only=True)
    assert_nested_equal(actor, (agent.proposal if agent.generative else agent.actor).state_dict())
    assert any(not torch.equal(v, agent.critics.state_dict()[k]) for k, v in q_before.items())
    assert weights == agent.online_metadata
    assert agent.recalibration_updates == 2
    assert agent.audit[1].diagnostics()["count"] == 2


@pytest.mark.parametrize("algorithm", CANDIDATE_ALGORITHMS)
def test_actual_online_backward_and_exact_checkpoint_resume(algorithm, tmp_path):
    config, data, norm, offline, agent = prepared(algorithm)
    replay = fill_online(agent)
    actor_before = copy.deepcopy((agent.proposal if agent.generative else agent.actor).state_dict())
    candidate_update(agent, offline, replay, config)
    assert any(not torch.equal(v, (agent.proposal if agent.generative else agent.actor).state_dict()[k])
               for k, v in actor_before.items())
    saved = dict(agent=agent.checkpoint_state(), offline=offline.state_dict(), online=replay.state_dict(),
                 rng=capture_global_rng_state())
    path = tmp_path / "resume.pt"
    torch.save(saved, path)
    for _ in range(2):
        candidate_update(agent, offline, replay, config)
    expected_rng = capture_global_rng_state()
    config2, data2, norm2, offline2, resumed = fixture(algorithm)
    payload = torch.load(path, weights_only=False)
    resumed.load_checkpoint_state(payload["agent"])
    offline2.load_state_dict(payload["offline"])
    replay2 = ReplayBuffer(2, 1, 32, 99)
    replay2.load_state_dict(payload["online"])
    restore_global_rng_state(payload["rng"])
    for _ in range(2):
        candidate_update(resumed, offline2, replay2, config2)
    assert_nested_equal(agent.checkpoint_state(), resumed.checkpoint_state())
    assert_nested_equal(offline.state_dict(), offline2.state_dict())
    assert_nested_equal(replay.state_dict(), replay2.state_dict())
    assert_nested_equal(expected_rng, capture_global_rng_state())
    assert all(torch.isfinite(p).all() for p in agent.parameters())


def test_rg_has_no_entropy_or_density_term_and_keeps_training_proposal():
    config, data, norm, offline, agent = prepared("rg_o2o")
    replay = fill_online(agent)
    alpha = agent.log_alpha.detach().clone()
    with patch.object(agent.actor, "forward", side_effect=AssertionError("RG cannot call Gaussian actor")):
        candidate_update(agent, offline, replay, config)
        action = agent.select_action(torch.zeros(2), evaluate=True)
    assert torch.isfinite(action).all() and action.abs().max() <= 1
    assert torch.equal(alpha, agent.log_alpha)
    assert agent.generator_updates == 3


@pytest.mark.parametrize("algorithm", CANDIDATE_ALGORITHMS)
def test_controller_collects_audits_and_performs_online_updates(algorithm, tmp_path):
    config, data, norm, offline, agent = prepared(algorithm)
    logger = _logger(str(tmp_path), algorithm)
    snapshots = []
    original = agent.observe_transition

    def observe(*args):
        snapshots.append((agent.collection_steps, agent.actor_updates, agent.critic_updates))
        original(*args)

    with patch.object(agent, "observe_transition", side_effect=observe), \
         patch("robust_o2o.experiment._evaluate"), patch("robust_o2o.experiment._save_phase_checkpoint"):
        _run_online(_GymnasiumEnv(terminal_at=2), None, data, config, agent, offline,
                    norm, None, torch.device("cpu"), logger, 2, 1)
    assert agent.collection_steps == 6
    assert len(agent.online_metadata) == 6
    assert agent.recalibration_updates == 2
    assert snapshots[0][1] == snapshots[1][1] == snapshots[2][1]
    assert snapshots[0][2] == snapshots[1][2]
    assert snapshots[2][2] == snapshots[1][2] + config.candidate_recalibration_steps
    assert agent.actor_updates > snapshots[0][1]
    assert agent.critic_updates > snapshots[0][2]
    assert agent.audit[1].diagnostics()["count"] == 3


@pytest.mark.parametrize("algorithm", CANDIDATE_ALGORITHMS)
def test_evaluation_preserves_rng_models_optimizers_and_audit(algorithm):
    config, data, norm, offline, agent = prepared(algorithm)
    before = copy.deepcopy(agent.checkpoint_state())
    rng = capture_global_rng_state()
    evaluate_agent(_GymnasiumEnv(terminal_at=2), config.env_name, agent, norm,
                   torch.device("cpu"), 2, 10, 13)
    assert_nested_equal(before, agent.checkpoint_state())
    assert_nested_equal(rng, capture_global_rng_state())


def test_actor_does_not_use_transition_trust_and_arw_normalizes_sources_separately():
    config, data, norm, offline, agent = prepared("arw_o2o")
    replay = fill_online(agent)
    off, on = offline.sample(4, agent.device), replay.sample(4, agent.device)
    agent.omega = .25
    losses = iter([torch.tensor(4., requires_grad=True), torch.tensor(8., requires_grad=True)])
    with patch.object(agent, "_critic_loss", side_effect=lambda batch: next(losses)), \
         patch.object(agent, "_step") as step:
        agent.update_sources(off, on, critic_only=True)
    assert step.call_args.args[0].item() == 7.


def test_candidates_do_not_enter_verified_main_baselines():
    from robust_o2o.fidelity import MAIN_BASELINES
    for name in CANDIDATE_ALGORITHMS:
        config, *_ = fixture(name)
        assert name not in MAIN_BASELINES
        assert not config.to_dict()["main_table_eligible"]


def test_actor_loss_does_not_inherit_transition_weights():
    config, data, norm, offline, agent = prepared()
    batch = offline.sample(4, agent.device)
    rng = capture_global_rng_state()
    with patch.object(agent, "_step"):
        first = agent.update_sources(batch, None)
        agent.offline_weights[:] = .05
        restore_global_rng_state(rng)
        second = agent.update_sources(batch, None)
    assert first["actor_loss"] == second["actor_loss"]


@pytest.mark.parametrize("mode,expected", [("fixed", .5), ("none", 0.)])
def test_arw_retention_controls_keep_identical_source_learner(mode, expected):
    config, data, norm, offline, agent = prepared("arw_o2o", candidate_retention_mode=mode)
    replay = fill_online(agent)
    assert agent.omega == expected
    candidate_update(agent, offline, replay, config)
    assert agent.omega == expected


@pytest.mark.parametrize("algorithm", CANDIDATE_ALGORITHMS)
def test_candidate_corruption_keeps_clean_env_and_poisoned_replay_separate(algorithm, tmp_path):
    config, data, norm, offline, agent = prepared(algorithm, corruption="random", corruption_target="actions",
                                                online_corruption_rate=1., corruption_range=100.)
    env = _GymnasiumEnv(terminal_at=2)
    observed = []
    original = agent.observe_transition

    def observe(*args):
        observed.append(np.asarray(args[1]).copy())
        return original(*args)

    with patch.object(agent, "observe_transition", side_effect=observe), \
         patch("robust_o2o.experiment._evaluate"), patch("robust_o2o.experiment._save_phase_checkpoint"):
        _run_online(env, None, data, config, agent, offline, norm, None,
                    torch.device("cpu"), _logger(str(tmp_path), algorithm), 2, 1)
    assert all(np.abs(action).max() <= 1 for action in env.actions)
    assert any(np.abs(action).max() > 1 for action in observed)
    assert len(agent.online_metadata) == 6


def test_three_riql_initialization_trajectories_match():
    agents = []
    for name in CANDIDATE_ALGORITHMS:
        config, data, norm, offline, agent = fixture(name)
        for _ in range(3):
            agent.update(offline.sample(4, agent.device))
        agents.append(agent)
    for other in agents[1:]:
        assert_nested_equal(agents[0].critics.state_dict(), other.critics.state_dict())
        assert_nested_equal(agents[0].value.state_dict(), other.value.state_dict())
        assert_nested_equal(agents[0].actor.state_dict(), other.actor.state_dict())


@pytest.mark.parametrize("name", ["CARE-O2O", "ARW-O2O", "RG-O2O"])
def test_pdf_names_are_cli_aliases(name):
    from robust_o2o.config import build_parser, config_from_args
    args = build_parser().parse_args(["--algorithm", name, "--env-name", "hopper-medium-replay-v2"])
    config = config_from_args(args)
    assert config.algorithm == name.lower().replace("-", "_")
    assert config.to_dict()["display_name"] == name
