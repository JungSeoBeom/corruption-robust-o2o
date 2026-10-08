"""Behavioral checks for the shared, versioned RIQL/RPEX corruption setting."""

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch

from robust_o2o.config import ExperimentConfig
from robust_o2o.corruption import (
    AttackOracle,
    EDACActor,
    EDACCritic,
    corrupt_offline_dataset,
    corrupt_offline_reward_values,
    corrupt_online_reward_value,
    corrupt_online_transition,
    make_numpy_corruption_rng,
    reward_corruption_metadata,
    sample_online_corruption_target,
)
from robust_o2o.environment import StateNormalizer
from robust_o2o.experiment import _run_online
from robust_o2o.fidelity import MAIN_BASELINES
from robust_o2o.manifest import build_experiment_manifest, comparison_condition
from robust_o2o.replay import OfflineDataset, ReplayBuffer
from test_main_controller import _GymnasiumEnv, _dataset, _logger, _normalizer


def config(target="observations", **kwargs):
    return ExperimentConfig(
        "rpex", "hopper-medium-replay-v2",
        corruption="random", corruption_target=target, **kwargs,
    )


@pytest.mark.parametrize("algorithm", MAIN_BASELINES)
def test_source_corruption_default_is_shared_across_learners(algorithm):
    resolved = ExperimentConfig(
        algorithm, "hopper-medium-replay-v2",
        corruption="random", corruption_target="observations",
    )
    assert resolved.corruption_profile == "riql_rpex_code"
    assert resolved.uses_source_corruption
    assert resolved.online_corruption_scale_profile == "rpex_official_code"
    assert resolved.attack_timing == "official_code_post_transition_replay_poisoning"
    assert resolved.random_attack_semantics == "post_transition_replay_poisoning"
    assert resolved.action_execution_profile == "official_algorithm_behavior"
    assert resolved.offline_corruption_rate == 0.3
    assert resolved.online_corruption_rate == 0.5
    assert resolved.corruption_range == 1.0
    assert isinstance(make_numpy_corruption_rng(resolved), np.random.RandomState)


def test_legacy_extension_retains_explicit_clipped_partitioned_behavior():
    resolved = config(
        "mixed", corruption_profile="legacy_extension",
        implementation_profile="research_benchmark",
    )
    assert not resolved.uses_source_corruption
    assert resolved.action_execution_profile == "clip_to_action_space"
    assert isinstance(make_numpy_corruption_rng(resolved), np.random.Generator)
    rng = make_numpy_corruption_rng(resolved)
    resolved.online_corruption_rate = 1.0
    observed = {sample_online_corruption_target(resolved, rng) for _ in range(200)}
    assert observed == {"observations", "actions", "rewards", "dynamics"}


def test_source_mixed_does_not_claim_parity_with_unpublished_upstream_setting():
    with pytest.raises(ValueError, match="mixed"):
        config("mixed")


def test_source_requires_upstream_mean_std_normalization_and_legacy_allows_mad():
    with pytest.raises(ValueError, match="normalization|robust_median_mad"):
        config(state_normalization="robust_median_mad")
    legacy = config(
        corruption_profile="legacy_extension", state_normalization="robust_median_mad",
    )
    assert legacy.state_normalization == "robust_median_mad"
    assert legacy.normalize_states


def test_offline_source_mask_and_noise_match_seeded_upstream_randomstate(tmp_path):
    dataset = _dataset(size=29)
    dataset["episode_id"] = np.zeros(29, dtype=np.float32)
    resolved = config("actions", seed=17, corruption_range=1.7)
    reference = np.random.RandomState(resolved.corruption_seed)
    indices = np.flatnonzero(reference.random(29) < resolved.offline_corruption_rate)
    expected = dataset["actions"].copy()
    expected[indices] += reference.uniform(-1.7, 1.7, (len(indices), 1)) * (
        dataset["actions"].std(axis=0, keepdims=True)
    )
    poisoned, stats = corrupt_offline_dataset(dataset, resolved, None, tmp_path)
    np.testing.assert_array_equal(poisoned["actions"], expected)
    assert stats["selected_transition_count"] == len(indices)
    for field in ("observations", "next_observations", "rewards", "terminals"):
        np.testing.assert_array_equal(poisoned[field], dataset[field])


@pytest.mark.parametrize("target", ["observations", "actions", "rewards", "dynamics"])
@pytest.mark.parametrize("selection_sampled_by_caller", [True, False])
def test_source_consumes_candidate_draws_for_unselected_online_transitions(target, selection_sampled_by_caller):
    resolved = config(target, online_corruption_rate=0.0)
    rng = make_numpy_corruption_rng(resolved)
    reference = np.random.RandomState(resolved.corruption_seed)
    state = np.asarray([2.0, 3.0], dtype=np.float32)
    action = np.asarray([0.2], dtype=np.float32)
    following = state + 1.0
    for _ in range(4):
        selected = sample_online_corruption_target(resolved, rng) if selection_sampled_by_caller else None
        assert selected is None
        reference.uniform(0.0, 1.0)
        reference.uniform(-1.0, 1.0, size=() if target == "rewards" else (1,) if target == "actions" else (2,))
        result = corrupt_online_transition(
            state, action, 4.0, following, resolved, None, rng,
            np.asarray([4.0, 5.0], np.float32), np.asarray([0.7], np.float32),
            selected_target=selected, selection_already_sampled=selection_sampled_by_caller,
            normalizer_std=np.asarray([2.0, 3.0], np.float32),
        )
        assert result[-1] is False
        for actual, clean in zip(result[:4], (state, action, 4.0, following), strict=True):
            np.testing.assert_array_equal(actual, clean)
    np.testing.assert_array_equal(rng.random(6), reference.random(6))


@pytest.mark.parametrize("algorithm", MAIN_BASELINES)
@pytest.mark.parametrize("epsilon", [0.0, 0.25, 2.0])
def test_source_online_reward_support_is_fixed_independent_of_learner(algorithm, epsilon):
    resolved = ExperimentConfig(
        algorithm, "hopper-medium-replay-v2",
        corruption="random", corruption_target="rewards", corruption_range=epsilon,
    )
    expected = np.random.RandomState(13).uniform(-1.0, 1.0) * 30.0
    assert corrupt_online_reward_value(400.0, resolved, np.random.RandomState(13)) == expected
    np.testing.assert_allclose(
        corrupt_offline_reward_values(np.asarray([400.0]), resolved, np.random.RandomState(13)),
        [expected * epsilon],
    )
    metadata = reward_corruption_metadata(resolved, "online")
    assert metadata["reward_corruption_low"] == -30.0
    assert metadata["reward_corruption_high"] == 30.0


def test_legacy_online_reward_retains_epsilon_scaled_support():
    resolved = config("rewards", corruption_profile="legacy_extension", corruption_range=2.0)
    expected = np.random.RandomState(13).uniform(-1.0, 1.0) * 60.0
    assert corrupt_online_reward_value(400.0, resolved, np.random.RandomState(13)) == expected


class RecordingOracle:
    def __init__(self):
        self.calls = []

    def attack(self, original, std, observations, actions, target, scale, steps, step_size, *, online=False):
        self.calls.append(SimpleNamespace(
            original=original.copy(), std=std.copy(), observations=observations.copy(),
            actions=actions.copy(), target=target, online=online,
        ))
        return original + np.float32(0.25) * std


@pytest.mark.parametrize("target,field", [("observations", 0), ("actions", 1), ("dynamics", 3)])
def test_source_adversarial_oracle_uses_wrapped_state_coordinates(target, field):
    resolved = config(target)
    resolved.corruption = "adversarial"
    oracle = RecordingOracle()
    state = np.asarray([5.0, -3.0], np.float32)
    following = np.asarray([9.0, 1.0], np.float32)
    action = np.asarray([2.4], np.float32)
    mean = np.asarray([1.0, -7.0], np.float32)
    std = np.asarray([2.0, 8.0], np.float32)
    action_std = np.asarray([0.4], np.float32)
    result = corrupt_online_transition(
        state, action, 3.0, following, resolved, oracle, np.random.RandomState(7),
        np.asarray([17.0, 19.0], np.float32), action_std,
        selected_target=target, selection_already_sampled=True,
        normalizer_mean=mean, normalizer_std=std,
    )
    call = oracle.calls[0]
    assert call.online
    np.testing.assert_array_equal(call.observations, ((state - mean) / std)[None, :])
    np.testing.assert_array_equal(call.actions, action[None, :])
    clean = (state, action, 3.0, following)
    if target == "actions":
        np.testing.assert_array_equal(call.original, action[None, :])
        np.testing.assert_array_equal(call.std, action_std[None, :])
        np.testing.assert_allclose(result[field], action + 0.25 * action_std)
    else:
        original = state if target == "observations" else following
        np.testing.assert_array_equal(call.original, ((original - mean) / std)[None, :])
        np.testing.assert_array_equal(call.std, np.ones((1, 2), np.float32))
        np.testing.assert_array_equal(result[field], original + 0.25 * std)
    for index, (actual, expected) in enumerate(zip(result[:4], clean, strict=True)):
        if index != field:
            np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(state, [5.0, -3.0])
    np.testing.assert_array_equal(following, [9.0, 1.0])


@pytest.mark.parametrize("target", ["observations", "dynamics"])
def test_source_without_state_normalization_uses_unit_online_state_bounds(target):
    resolved = config(target, normalize_states=False)
    resolved.corruption = "adversarial"
    oracle = RecordingOracle()
    state = np.asarray([4.0, 8.0], np.float32)
    following = state + 1.0
    corrupt_online_transition(
        state, np.asarray([0.2], np.float32), 3.0, following,
        resolved, oracle, np.random.RandomState(7),
        np.asarray([17.0, 19.0], np.float32), np.asarray([0.4], np.float32),
        selected_target=target, selection_already_sampled=True,
    )
    np.testing.assert_array_equal(oracle.calls[0].std, np.ones((1, 2), np.float32))
    np.testing.assert_array_equal(oracle.calls[0].observations, state[None, :])


def test_source_normalizer_matches_additive_upstream_epsilon_after_poisoning(tmp_path):
    dataset = _dataset(size=29)
    dataset["episode_id"] = np.zeros(29, dtype=np.float32)
    resolved = config("dynamics", seed=17)
    poisoned, _ = corrupt_offline_dataset(dataset, resolved, None, tmp_path)
    normalizer = StateNormalizer.fit(poisoned, additive_epsilon=resolved.uses_source_corruption)
    states = np.concatenate((poisoned["observations"], poisoned["next_observations"]))
    np.testing.assert_array_equal(normalizer.mean, states.mean(axis=0))
    np.testing.assert_array_equal(normalizer.std, states.std(axis=0) + np.float32(1e-3))
    assert np.any(normalizer.std != np.maximum(states.std(axis=0), 1e-3))


@pytest.mark.parametrize("target", ["observations", "actions"])
def test_source_oracle_initialization_preserves_double_std_and_fresh_online_generator(tmp_path, target):
    checkpoint = tmp_path / "edac.pt"
    torch.save({"actor": EDACActor(2, 1, 1.0).state_dict(), "critic": EDACCritic(2, 1).state_dict()}, checkpoint)
    oracle = AttackOracle(
        2, 1, 1.0, checkpoint, torch.device("cpu"), seed=23,
        implementation_profile="rpex_official_adam", benchmark_profile="research_benchmark",
        corruption_profile="riql_rpex_code", record_trace=True,
    )
    original = np.asarray([[4.0, 8.0]] if target == "observations" else [[0.2]], np.float32)
    scale_std = np.asarray([[2.0, 7.0]] if target == "observations" else [[0.4]], np.float32)
    try:
        results = [oracle.attack(original, scale_std, np.asarray([[4.0, 8.0]], np.float32),
                                np.asarray([[0.2]], np.float32), target, 1.5, 0, 0.1,
                                online=True) for _ in range(2)]
        reference = 3.0 * (torch.rand(original.shape, generator=torch.Generator()).numpy() - 0.5)
        np.testing.assert_array_equal(results[0], original + reference * scale_std * scale_std)
        np.testing.assert_array_equal(results[1], results[0])
        np.testing.assert_array_equal(oracle.attack_traces[0]["initial_parameter"], reference * scale_std)
    finally:
        oracle.close()


def test_source_dynamics_actor_rng_is_private_and_exactly_restorable(tmp_path):
    checkpoint = tmp_path / "edac.pt"
    torch.save({"actor": EDACActor(2, 1, 1.0).state_dict(), "critic": EDACCritic(2, 1).state_dict()}, checkpoint)
    oracle = AttackOracle(
        2, 1, 1.0, checkpoint, torch.device("cpu"), seed=23,
        implementation_profile="rpex_official_adam", benchmark_profile="research_benchmark",
        corruption_profile="riql_rpex_code",
    )

    def attack():
        return oracle.attack(
            np.asarray([[4.0, 8.0]], np.float32), np.ones((1, 2), np.float32),
            np.asarray([[3.0, 5.0]], np.float32), np.asarray([[0.2]], np.float32),
            "dynamics", 0.3, 2, 0.1, online=True,
        )

    try:
        before = torch.random.get_rng_state().clone()
        attack()
        state = oracle.rng_state_dict()
        expected = attack()
        oracle.load_rng_state_dict(state)
        np.testing.assert_array_equal(attack(), expected)
        assert torch.equal(before, torch.random.get_rng_state())
    finally:
        oracle.close()


def test_repaired_source_offline_dynamics_retains_stochastic_actor_objective(tmp_path):
    checkpoint = tmp_path / "edac.pt"
    torch.save({"actor": EDACActor(2, 1, 1.0).state_dict(), "critic": EDACCritic(2, 1).state_dict()}, checkpoint)
    oracle = AttackOracle(
        2, 1, 1.0, checkpoint, torch.device("cpu"), seed=23,
        implementation_profile="rpex_official_adam", benchmark_profile="research_benchmark",
        corruption_profile="riql_rpex_code",
    )
    flags = []

    class RecordingActor(torch.nn.Module):
        def forward(self, states, deterministic=True, *, generator=None):
            flags.append((deterministic, generator))
            return states[:, :1].tanh()

    oracle.actor = RecordingActor()
    try:
        result = oracle.attack(
            np.asarray([[4.0, 8.0]], np.float32), np.ones((1, 2), np.float32),
            np.asarray([[3.0, 5.0]], np.float32), np.asarray([[0.2]], np.float32),
            "dynamics", 0.3, 2, 0.01, online=False,
        )
        assert np.isfinite(result).all()
        assert len(flags) == 2
        assert all(not deterministic and generator is not None for deterministic, generator in flags)
    finally:
        oracle.close()


def test_source_offline_actor_sampling_does_not_shift_next_chunk_initialization(tmp_path):
    checkpoint = tmp_path / "edac.pt"
    torch.save({"actor": EDACActor(2, 1, 1.0).state_dict(), "critic": EDACCritic(2, 1).state_dict()}, checkpoint)
    oracle = AttackOracle(
        2, 1, 1.0, checkpoint, torch.device("cpu"), seed=23,
        implementation_profile="rpex_official_adam", benchmark_profile="research_benchmark",
        corruption_profile="riql_rpex_code",
    )
    initialization_rng = torch.Generator()
    initialization_rng.set_state(oracle.generator.get_state())
    torch.rand((3, 2), generator=initialization_rng)
    try:
        oracle.attack(
            np.ones((3, 2), np.float32), np.ones((1, 2), np.float32),
            np.zeros((3, 2), np.float32), np.zeros((3, 1), np.float32),
            "dynamics", 0.3, 2, 0.01, online=False,
        )
        assert torch.equal(oracle.generator.get_state(), initialization_rng.get_state())
    finally:
        oracle.close()


def test_source_checkpoint_restores_both_offline_initialization_and_actor_streams(tmp_path):
    checkpoint = tmp_path / "edac.pt"
    torch.save({"actor": EDACActor(2, 1, 1.0).state_dict(), "critic": EDACCritic(2, 1).state_dict()}, checkpoint)
    oracle = AttackOracle(
        2, 1, 1.0, checkpoint, torch.device("cpu"), seed=23,
        implementation_profile="rpex_official_adam", benchmark_profile="research_benchmark",
        corruption_profile="riql_rpex_code",
    )

    def attack():
        return oracle.attack(
            np.ones((3, 2), np.float32), np.ones((1, 2), np.float32),
            np.zeros((3, 2), np.float32), np.zeros((3, 1), np.float32),
            "dynamics", 0.3, 2, 0.01, online=False,
        )

    try:
        attack()
        snapshot = oracle.rng_state_dict()
        assert "offline_actor" in snapshot
        expected = attack()
        after = oracle.rng_state_dict()
        assert not torch.equal(snapshot["offline"], after["offline"])
        assert not torch.equal(snapshot["offline_actor"], after["offline_actor"])
        assert torch.equal(snapshot["online"], after["online"])
        oracle.load_rng_state_dict(snapshot)
        assert torch.equal(oracle.offline_actor_generator.get_state(), snapshot["offline_actor"])
        np.testing.assert_array_equal(attack(), expected)
        restored_after = oracle.rng_state_dict()
        for name in ("offline", "offline_actor", "online"):
            assert torch.equal(restored_after[name], after[name])
    finally:
        oracle.close()


def test_source_controller_executes_and_poisons_unclipped_policy_proposal(tmp_path):
    replays = []

    class CapturingReplay(ReplayBuffer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            replays.append(self)

    class Agent:
        total_updates = 0

        def select_action(self, state, evaluate=False):
            return torch.asarray([2.5], dtype=torch.float32)

    resolved = config("actions", online_steps=1, online_corruption_rate=1.0,
                      corruption_range=0.8, eval_period=100_000,
                      train_log_period=100_000, checkpoint_period=100_000)
    resolved.initial_collection_steps = 10
    env = _GymnasiumEnv()
    with (patch("robust_o2o.experiment.ReplayBuffer", CapturingReplay),
          patch("robust_o2o.experiment._evaluate"),
          patch("robust_o2o.experiment._save_phase_checkpoint")):
        _run_online(env, object(), _dataset(), resolved, Agent(),
                    OfflineDataset(_dataset(), seed=0), _normalizer(), None,
                    torch.device("cpu"), _logger(str(tmp_path), "source-action"), 2, 1)
    np.testing.assert_array_equal(env.actions[0], [2.5])
    reference = np.random.RandomState(resolved.corruption_seed)
    reference.random()
    expected = np.asarray([2.5], np.float32) + reference.uniform(-0.8, 0.8, 1) * _dataset()["actions"].std(axis=0)
    np.testing.assert_array_equal(replays[0].actions[0], expected.astype(np.float32))
    assert replays[0].actions[0, 0] > 1.0


def test_comparison_contract_distinguishes_source_and_legacy_profiles():
    conditions = []
    for profile in ("riql_rpex_code", "legacy_extension"):
        resolved = config("rewards", corruption_profile=profile)
        values = resolved.to_dict()
        conditions.append(comparison_condition(build_experiment_manifest(values), values))
    assert conditions[0] != conditions[1]
    assert conditions[0]["corruption_profile"] == "riql_rpex_code"
    assert conditions[1]["corruption_profile"] == "legacy_extension"
