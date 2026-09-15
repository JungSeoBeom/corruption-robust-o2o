"""Synthetic EDAC, actual PQE backward, and priority failure regressions."""
import importlib
import json
import math
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch
from torch import nn
from torch.distributions import Normal

from robust_o2o.config import ExperimentConfig, PQE_NUMERICS_VERSION
from robust_o2o.corruption import (
    ATTACK_RNG_SCHEMA, AttackOracle, EDACActor, EDACCritic,
    corrupt_offline_dataset, corrupt_online_transition,
)
from robust_o2o.experiment import (
    _validate_checkpoint, _validate_reliability_resume,
    capture_global_rng_state, restore_global_rng_state,
)
from robust_o2o.logging_utils import RunLogger
from robust_o2o.manifest import build_experiment_manifest, aggregation_signature
from robust_o2o.replay import (
    OfflineDataset, ReplayBuffer, _safe_probabilities,
    balanced_priority_batch, update_sample_priorities,
)
from robust_o2o.agents.pessimistic_q_ensemble import PessimisticQEnsembleAgent
from test_pqe_main import _agent, _batch
from test_research_corruption_contract import synthetic_dataset


@pytest.fixture
def oracle_factory(tmp_path):
    checkpoint = tmp_path / "synthetic_edac.pt"
    torch.save({"actor": EDACActor(4, 2, 1.0).state_dict(),
                "critic": EDACCritic(4, 2).state_dict()}, checkpoint)
    oracles = []

    def make():
        oracle = AttackOracle(4, 2, 1.0, checkpoint, torch.device("cpu"), seed=23,
                              implementation_profile="rpex_official_adam",
                              benchmark_profile="research_benchmark")
        oracles.append(oracle)
        return oracle

    yield make
    for oracle in oracles:
        oracle.close()


def attack_config(target):
    config = ExperimentConfig("rpex", "hopper-medium-replay-v2",
                              corruption="adversarial", corruption_target=target,
                              offline_corruption_rate=1.0, online_corruption_rate=1.0)
    config.implementation_profile = "research_benchmark"
    config.offline_attack_steps = config.online_attack_steps = 2
    config.corruption_range = 0.2
    return config


def online_sequence(oracle, config, count=8):
    dataset = synthetic_dataset(16)
    rng = np.random.default_rng(41)
    return [corrupt_online_transition(
        dataset["observations"][i], dataset["actions"][i], float(dataset["rewards"][i]),
        dataset["next_observations"][i], config, oracle, rng,
        np.ones(4, np.float32), np.ones(2, np.float32),
    ) for i in range(count)]


def assert_sequences_equal(left, right):
    for a, b in zip(left, right, strict=True):
        for x, y in zip(a, b, strict=True):
            np.testing.assert_array_equal(x, y)


@pytest.mark.parametrize("target", ["observations", "actions", "dynamics", "mixed"])
def test_cache_lifecycles_do_not_change_online_attacks(tmp_path, oracle_factory, target):
    config = attack_config(target)
    runs, states, artifacts = [], [], []
    for mode in ("miss", "hit", "regenerate"):
        oracle = oracle_factory()
        config.force_regenerate_attack = mode == "regenerate"
        with patch.object(oracle, "attack", wraps=oracle.attack) as attack:
            data, stats = corrupt_offline_dataset(synthetic_dataset(16), config, oracle, tmp_path / "cache")
            if target == "mixed" and mode != "hit":
                assert len({call.args[4] for call in attack.call_args_list}) >= 2
        assert bool(stats["cache_hit"]) == (mode == "hit")
        before = torch.random.get_rng_state().clone()
        initial_online = oracle.online_generator.get_state().clone()
        runs.append(online_sequence(oracle, config))
        states.append(oracle.online_generator.get_state())
        artifacts.append(data)
        assert torch.equal(before, torch.random.get_rng_state())
        assert not torch.equal(initial_online, states[-1])
    for i in (1, 2):
        assert_sequences_equal(runs[0], runs[i])
        assert torch.equal(states[0], states[i])
        for key in artifacts[0]:
            np.testing.assert_array_equal(artifacts[0][key], artifacts[i][key])


@pytest.mark.parametrize("target", ["observations", "actions", "dynamics", "mixed"])
def test_offline_draw_counts_and_offline_checkpoint_boundary(tmp_path, oracle_factory, target):
    config = attack_config(target)
    fresh, consumed = oracle_factory(), oracle_factory()
    for n, rate in ((16, 1.0), (8, 0.5)):
        config.offline_corruption_rate = rate
        corrupt_offline_dataset(synthetic_dataset(n), config, consumed, tmp_path / str(n))
    assert not torch.equal(fresh.generator.get_state(), consumed.generator.get_state())
    assert torch.equal(fresh.online_generator.get_state(), consumed.online_generator.get_state())
    # Production offline checkpoints omit oracle state; the phase seed alone
    # reconstructs the unused online stream, even when the cache lifecycle differs.
    offline_checkpoint_rng = capture_global_rng_state()
    resumed = oracle_factory()
    restore_global_rng_state(offline_checkpoint_rng, oracle=resumed)
    assert_sequences_equal(online_sequence(consumed, config), online_sequence(resumed, config))
    assert torch.equal(consumed.online_generator.get_state(), resumed.online_generator.get_state())


@pytest.mark.parametrize("target", ["observations", "actions", "dynamics", "mixed"])
def test_online_attack_checkpoint_restores_sequence(tmp_path, oracle_factory, target):
    config = attack_config(target)
    oracle = oracle_factory()
    online_sequence(oracle, config, 3)
    before = oracle.online_generator.get_state().clone()
    learner_rng = torch.random.get_rng_state().clone()
    snapshot = capture_global_rng_state(oracle=oracle)
    path = tmp_path / "rng.pt"
    torch.save(snapshot, path)
    assert torch.equal(before, oracle.online_generator.get_state())
    assert torch.equal(learner_rng, torch.random.get_rng_state())
    expected = online_sequence(oracle, config)
    resumed = oracle_factory()
    restore_global_rng_state(torch.load(path, weights_only=False), oracle=resumed)
    assert_sequences_equal(expected, online_sequence(resumed, config))
    assert torch.equal(oracle.online_generator.get_state(), resumed.online_generator.get_state())
    with pytest.raises(ValueError, match="legacy single-stream"):
        resumed.load_rng_state_dict(torch.Generator().get_state())


def test_reward_only_does_not_require_oracle(tmp_path):
    config = attack_config("rewards")
    _, stats = corrupt_offline_dataset(synthetic_dataset(16), config, None, tmp_path)
    assert stats["attack_rng_semantics"] == "none_reward_rule"
    assert_sequences_equal(online_sequence(None, config), online_sequence(None, config))


class MomentMember(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.mean = nn.Parameter(torch.tensor(float(mean)))
        self.log_std = nn.Parameter(torch.tensor(math.log(std)))

    def distribution(self, states):
        return Normal(self.mean.expand(len(states), 2), self.log_std.exp().expand(len(states), 2))


@pytest.mark.parametrize("means,stds", [
    ([0., .2, -.4, .7, 1.], [.2, .3, .4, .5, .6]),
    ([1.] * 5, [1e-4] * 5),
    ([10.] * 5, [1e-3] * 5),
    ([1. + i * 1e-6 for i in range(5)], [1e-5] * 5),
    ([1.] * 5, [math.exp(-20)] * 5),
    ([-2., -.5, 0., 1., 3.], [.05, .1, .2, .3, .4]),
])
def test_centered_moments_forward_backward(means, stds):
    members = nn.ModuleList([MomentMember(m, s) for m, s in zip(means, stds)])
    mean, std = PessimisticQEnsembleAgent.moment_parameters(SimpleNamespace(actors=members), torch.zeros(3, 4))
    # Reference computed in double from the actual float32 member parameters.
    m = torch.stack([a.mean.double() for a in members])
    s = torch.stack([a.log_std.exp().double() for a in members])
    reference = (s.square().mean() + (m - m.mean()).square().mean()).clamp_min(math.exp(-40)).sqrt().clamp_max(math.exp(2))
    if min(stds) >= .05:
        original_formula = ((s.square() + m.square()).mean() - m.mean().square()).sqrt()
        torch.testing.assert_close(reference, original_formula)
    assert torch.isfinite(mean).all() and torch.isfinite(std).all()
    torch.testing.assert_close(std.double(), reference.expand_as(std), rtol=2e-5, atol=1e-12)
    (mean.sum() + std.log().sum()).backward()
    for member in members:
        assert torch.isfinite(member.mean.grad) and member.mean.grad != 0
        assert torch.isfinite(member.log_std.grad)
        if min(stds) > math.exp(-20):
            assert member.log_std.grad != 0


def test_small_std_actual_pqe_online_optimizer_steps():
    agent = _agent()
    with torch.no_grad():
        for actor in agent.actors:
            actor.mean.weight.zero_()
            actor.mean.bias.fill_(1.)
            actor.log_std.weight.zero_()
            actor.log_std.bias.fill_(math.log(1e-4))
    agent.begin_online()
    snapshots = [[p.clone() for p in a.parameters()] for a in agent.actors]
    batch = _batch(10)
    batch["_source"] = torch.tensor([0, 0, 0, 1, 1, 1])
    for _ in range(3):
        actions, logp, _, std = agent._moment_policy(batch["observations"], need_log_prob=True)
        assert torch.isfinite(actions).all() and torch.isfinite(logp).all()
        assert torch.isfinite(std).all()
        agent.update(rl_batch=batch, density_offline_batch=_batch(11),
                     density_online_batch=_batch(12), rl_batch_prioritized=True)
        for actor in agent.actors:
            assert all(torch.isfinite(p).all() for p in actor.parameters())
        assert torch.isfinite(agent.consume_priority_values()).all()
    for before, actor in zip(snapshots, agent.actors):
        assert any(not torch.equal(p, q) for p, q in zip(before, actor.parameters()))


@pytest.mark.parametrize("values", [[np.nan, 1], [np.inf, 1], [-np.inf, 1], [-1, 1], [], [[1, 2]], 1.])
def test_invalid_probabilities_raise(values):
    with pytest.raises(ValueError, match="_safe_probabilities"):
        _safe_probabilities(np.asarray(values))


@pytest.mark.parametrize("values", [[1., 2., 3.], [0., 0., 1.], [0., 0., 0.], [1e308, 1e308]])
def test_valid_probability_ratios(values):
    values = np.maximum(np.asarray(values), 1e-12)
    scaled = values / values.max()
    np.testing.assert_allclose(_safe_probabilities(values), scaled / scaled.sum())


def replay_pair():
    offline = OfflineDataset(synthetic_dataset(8), seed=3)
    online = ReplayBuffer(4, 2, 8, seed=4)
    online.add(np.zeros(4), np.zeros(2), 0., np.zeros(4), 0.)
    return offline, online


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf, -1.])
def test_priority_paths_validate_before_mutation(bad):
    offline, online = replay_pair()
    prior_off, prior_on = offline.priorities.copy(), online.priorities.copy()
    batch = {"_indices": torch.tensor([0, 0]), "_source": torch.tensor([0, 1])}
    with pytest.raises(ValueError, match="update_sample_priorities"):
        update_sample_priorities(offline, online, batch, torch.tensor([2., bad]))
    for replay in (offline, online):
        with pytest.raises(ValueError, match="update_priorities"):
            replay.update_priorities(torch.tensor([0]), torch.tensor([bad]))
    with pytest.raises(ValueError, match="ReplayBuffer.add"):
        online.add(np.ones(4), np.ones(2), 1., np.ones(4), 1., priority=bad)
    with pytest.raises(ValueError, match="ReplayBuffer.add_batch"):
        online.add_batch(np.zeros((2, 4)), np.zeros((2, 2)), np.zeros(2), np.zeros((2, 4)),
                         np.zeros(2), priorities=np.array([2., bad]))
    np.testing.assert_array_equal(prior_off, offline.priorities)
    np.testing.assert_array_equal(prior_on, online.priorities)
    assert online.size == online.position == 1
    assert offline.priority_updates == online.priority_updates == 0
    online.priorities[0] = bad
    with pytest.raises(ValueError, match="_safe_probabilities"):
        balanced_priority_batch(offline, online, 4, torch.device("cpu"))


def test_cross_replay_bad_index_is_atomic_and_valid_update_keeps_mc_sentinel():
    offline, online = replay_pair()
    batch = {"_indices": torch.tensor([0, 99]), "_source": torch.tensor([0, 1])}
    with pytest.raises(ValueError, match="out of bounds"):
        update_sample_priorities(offline, online, batch, torch.tensor([2., 3.]))
    assert offline.priorities[0] == 1 and offline.priority_updates == 0
    batch["_indices"][1] = 0
    update_sample_priorities(offline, online, batch, torch.tensor([2., 3.]))
    assert offline.priorities[0] == 2 and online.priorities[0] == 3
    assert np.isneginf(online.mc_returns[0])
    assert balanced_priority_batch(offline, online, 4, torch.device("cpu"))["rewards"].numel() == 4


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf"), -1.])
def test_density_invalid_weights_cannot_be_clamped(bad):
    fn = PessimisticQEnsembleAgent.priority_values_from_weights
    for weights, offline in ((torch.tensor([bad]), torch.ones(2)), (torch.ones(2), torch.tensor([bad]))):
        with pytest.raises(ValueError, match="PQE priority"):
            fn(weights, offline)


def test_overflowing_density_denominator_is_not_hidden_by_floor():
    with pytest.raises(ValueError, match="denominator"):
        PessimisticQEnsembleAgent.priority_values_from_weights(
            torch.ones(2), torch.full((8,), 3e38), temperature=1.0)


@pytest.mark.parametrize("indices,values", [
    (torch.tensor([0]), torch.tensor([])),
    (torch.tensor([0]), torch.tensor([[1.]])),
    (torch.tensor([0., 1.]), torch.tensor([1., 2.])),
    (torch.tensor([0]), torch.tensor([1., 2.])),
])
def test_bad_priority_update_shape_is_atomic(indices, values):
    offline, online = replay_pair()
    for replay in (offline, online):
        before = replay.priorities.copy()
        with pytest.raises(ValueError):
            replay.update_priorities(indices, values)
        np.testing.assert_array_equal(before, replay.priorities)


def test_legacy_rng_resume_rejection_preserves_completed_run(tmp_path):
    cli = importlib.import_module("run_experiment")
    config = attack_config("observations")
    config.output_dir = str(tmp_path)
    logger = RunLogger(config)
    logger.write_config(config.to_dict())
    logger.write_completion_manifest({"actual_online_steps": 100})
    logger.finish("completed")
    before = {p.relative_to(logger.run_dir): p.read_bytes()
              for p in logger.run_dir.rglob("*") if p.is_file()}
    config.resume_run = str(logger.run_dir)

    def reject_before_write(config, resumed_logger):
        _validate_checkpoint({}, config, 4, 2)
        resumed_logger.write_config(config.to_dict())

    with patch.object(cli, "config_from_args", return_value=config), patch.object(cli, "build_parser"), patch.object(cli, "run_experiment", side_effect=reject_before_write):
        assert cli.main() == 1
    after = {p.relative_to(logger.run_dir): p.read_bytes()
             for p in logger.run_dir.rglob("*") if p.is_file()}
    assert before == after


def assert_nested_equal(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, np.ndarray):
        np.testing.assert_array_equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_nested_equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert_nested_equal(a, b)
    else:
        assert left == right


def test_pqe_replay_optimizer_exact_resume(tmp_path):
    from robust_o2o.replay import sample_pqe_update_batches
    agent = _agent()
    agent.begin_online()
    data = {k: v.numpy() for k, v in _batch(121, 16).items()}
    offline = OfflineDataset(data, seed=44)
    online = ReplayBuffer(3, 2, 32, seed=45)
    online.add_batch(data["observations"], data["actions"], data["rewards"],
                     data["next_observations"], data["terminals"])

    def step(agent, offline, online):
        rl, density_off, density_on = sample_pqe_update_batches(
            offline, online, 6, .5, torch.device("cpu"), True, 6)
        agent.update(rl_batch=rl, density_offline_batch=density_off,
                     density_online_batch=density_on, rl_batch_prioritized=True)
        update_sample_priorities(offline, online, rl, agent.consume_priority_values())

    step(agent, offline, online)
    payload = {"agent": agent.checkpoint_state(), "offline": offline.state_dict(),
               "online": online.state_dict(), "rng": capture_global_rng_state()}
    path = tmp_path / "pqe_resume.pt"
    torch.save(payload, path)
    for _ in range(3):
        step(agent, offline, online)
    expected_rng = capture_global_rng_state()
    saved = torch.load(path, weights_only=False)
    resumed = _agent()
    resumed.load_checkpoint_state(saved["agent"])
    resumed_off = OfflineDataset(data, seed=999)
    resumed_on = ReplayBuffer(3, 2, 32, seed=998)
    resumed_off.load_state_dict(saved["offline"])
    resumed_on.load_state_dict(saved["online"])
    restore_global_rng_state(saved["rng"])
    for _ in range(3):
        step(resumed, resumed_off, resumed_on)
    assert_nested_equal(agent.checkpoint_state(), resumed.checkpoint_state())
    assert_nested_equal(offline.state_dict(), resumed_off.state_dict())
    assert_nested_equal(online.state_dict(), resumed_on.state_dict())
    assert_nested_equal(expected_rng, capture_global_rng_state())


def test_old_resume_rejected_but_offline_phase_has_no_online_state_requirement():
    config = attack_config("observations")
    with pytest.raises(ValueError, match="phase-private"):
        _validate_reliability_resume({}, config)
    _validate_reliability_resume({"attack_rng_schema": ATTACK_RNG_SCHEMA,
                                  "resume_state": {"phase": "offline"}}, config)
    with pytest.raises(ValueError, match="missing"):
        _validate_reliability_resume({"attack_rng_schema": ATTACK_RNG_SCHEMA,
                                      "resume_state": {"phase": "online"}}, config)
    config.algorithm = "pessimistic_q_ensemble"
    with pytest.raises(ValueError, match="PQE exact resume"):
        _validate_reliability_resume({}, config)


def test_numerics_and_rng_versions_split_manifest_groups():
    config = ExperimentConfig("pessimistic_q_ensemble", "hopper-medium-replay-v2")
    resolved = config.to_dict()
    assert resolved["pqe_numerics_version"] == PQE_NUMERICS_VERSION
    current = build_experiment_manifest(resolved)
    old = build_experiment_manifest({k: v for k, v in resolved.items() if k != "pqe_numerics_version"})
    assert "pqe_numerics_version" not in old
    assert aggregation_signature(current) != aggregation_signature(old)
    for version in ("persistent_private_torch_generator", ATTACK_RNG_SCHEMA):
        resolved["offline_corruption"] = {"attack_rng_semantics": version}
        manifest = build_experiment_manifest(resolved)
        if version == ATTACK_RNG_SCHEMA:
            assert aggregation_signature(manifest) != aggregation_signature(previous)
        previous = manifest


def test_priority_failure_is_recorded_as_failed_run(tmp_path):
    cli = importlib.import_module("run_experiment")
    config = ExperimentConfig("pessimistic_q_ensemble", "hopper-medium-replay-v2", output_dir=str(tmp_path))
    loggers = []

    def run(config, logger):
        loggers.append(logger)
        logger.write_config(config.to_dict())
        offline, online = replay_pair()
        online.priorities[0] = np.nan
        balanced_priority_batch(offline, online, 4, torch.device("cpu"))

    with patch.object(cli, "config_from_args", return_value=config), patch.object(cli, "build_parser") as parser, patch.object(cli, "run_experiment", side_effect=run):
        parser.return_value.parse_args.return_value = None
        assert cli.main() == 1
    summary = json.loads((loggers[0].run_dir / "summary.json").read_text())
    assert summary["status"] == "failed"
    assert "_safe_probabilities" in summary["error"]
    assert not (loggers[0].run_dir / "completion_manifest.json").exists()
