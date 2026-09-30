"""Regressions for the result/code audit: coordinates, comparisons and rewards."""

import json
import subprocess

import numpy as np
import pytest
import torch

from plot_results import _validate_score_contract, select_latest_comparable_records
from robust_o2o.agents.iql_family import IQLFamilyAgent
from robust_o2o.calql_online import CalQLTrajectoryAccumulator
from robust_o2o.config import ExperimentConfig
from robust_o2o.corruption import (
    corrupt_offline_dataset,
    corrupt_online_transition,
    corrupt_offline_reward_values,
    corrupt_online_reward_value,
    online_corruption_scale,
    reward_corruption_metadata,
)
from robust_o2o.environment import StateNormalizer, repository_state_metadata
from robust_o2o.experiment import _validate_reliability_resume
from robust_o2o.fidelity import canonical_json_sha256
from robust_o2o.manifest import build_experiment_manifest, comparison_condition


def config(target="observations", **kwargs):
    return ExperimentConfig(
        "rpex",
        "halfcheetah-medium-replay-v2",
        corruption="random",
        corruption_target=target,
        **kwargs,
    )


def test_iql_family_honors_gradient_cap_for_all_three_optimizers():
    torch.manual_seed(38)
    batch = {
        "observations": torch.randn(16, 3),
        "next_observations": torch.randn(16, 3),
        "actions": torch.randn(16, 2),
        "rewards": torch.randn(16),
        "terminals": torch.zeros(16),
    }
    agents = []
    for cap in (None, 1e-8):
        torch.manual_seed(17)
        agent = IQLFamilyAgent(
            config("dynamics", hidden_dim=8, max_grad_norm=cap),
            3,
            2,
            1.0,
            torch.device("cpu"),
        )
        agent.update(batch)
        agents.append(agent)
    for name in ("actor", "critic", "value"):
        for agent, bounded in zip(agents, (False, True)):
            norm = (
                torch.stack([p.grad.norm() for p in getattr(agent, name).parameters()])
                .norm()
                .item()
            )
            assert norm <= 1.01e-8 if bounded else norm > 1e-5
        assert any(
            not torch.equal(a, b)
            for a, b in zip(
                getattr(agents[0], name).parameters(),
                getattr(agents[1], name).parameters(),
            )
        )


@pytest.mark.parametrize("profile", ["official_code_reference", "research_benchmark"])
@pytest.mark.parametrize("target,field", [("observations", 0), ("dynamics", 3)])
def test_unit_noise_is_added_in_normalized_coordinates(profile, target, field):
    c = config(
        target,
        implementation_profile=profile,
        suite_profile="research_benchmark"
        if profile == "research_benchmark"
        else "common_budget_robustness",
    )
    norm = StateNormalizer(
        np.array([3.0, -7.0, 4.0], np.float32), np.array([0.1, 2.0, 9.0], np.float32)
    )
    state = np.array([1.0, 2.0, 3.0], np.float32)
    following = np.array([4.0, 5.0, 6.0], np.float32)
    original = state if field == 0 else following
    result = corrupt_online_transition(
        state,
        np.zeros(2, np.float32),
        5.0,
        following,
        c,
        None,
        np.random.default_rng(41),
        np.ones(3, np.float32),
        np.ones(2, np.float32),
        selected_target=target,
        selection_already_sampled=True,
        normalizer_std=norm.std,
    )
    expected_noise = np.random.default_rng(41).uniform(-1, 1, size=3)
    np.testing.assert_allclose(
        norm.transform(result[field]),
        norm.transform(original) + expected_noise,
        atol=3e-6,
    )
    np.testing.assert_array_equal(
        result[3 if field == 0 else 0], following if field == 0 else state
    )
    np.testing.assert_array_equal(state, [1.0, 2.0, 3.0])
    assert result[2] == 5.0


def test_normalized_noise_requires_explicit_fitted_scale_and_preserves_other_profiles():
    c = config(implementation_profile="official_code_reference")
    args = dict(state_std=np.array([2.0, 4.0]), action_std=np.ones(1))
    with pytest.raises(ValueError, match="requires normalizer_std"):
        online_corruption_scale("observations", c, **args)
    c.normalize_states = False
    np.testing.assert_array_equal(
        online_corruption_scale("observations", c, **args), [1, 1]
    )
    c.normalize_states = True
    c.corruption = "adversarial"
    np.testing.assert_array_equal(
        online_corruption_scale("observations", c, **args), [1, 1]
    )
    c.corruption = "random"
    c.online_corruption_scale_profile = "dataset_std_scaled_extension"
    np.testing.assert_array_equal(
        online_corruption_scale("observations", c, **args), [2, 4]
    )


def test_changed_online_coordinates_cannot_exact_resume_old_weights():
    c = config(implementation_profile="official_code_reference")
    with pytest.raises(ValueError, match="coordinates changed"):
        _validate_reliability_resume({"config": {}}, c)
    _validate_reliability_resume({"config": c.to_dict()}, c)
    _validate_reliability_resume({"config": {}}, config("rewards"))


def condition(c):
    resolved = c.to_dict()
    return comparison_condition(build_experiment_manifest(resolved), resolved)


def test_comparison_detects_state_noise_units_but_ignores_inactive_reward_knobs():
    left = config(online_corruption_scale_profile="rpex_official_code")
    right = config(online_corruption_scale_profile="dataset_std_scaled_extension")
    assert condition(left) != condition(right)
    for c in (left, right):
        c.corruption_target = "rewards"
    assert condition(left) == condition(right)
    right.online_corruption_rate = 0.2
    assert condition(left) != condition(right)


def test_comparison_includes_online_reward_support_at_nonunit_epsilon():
    official = config(
        "rewards",
        implementation_profile="official_code_reference",
        corruption_range=2.0,
    )
    custom = config("rewards", corruption_range=2.0)
    assert condition(official)["reward_replacement_bounds"] == {
        "offline": 60.0,
        "online": 30.0,
    }
    assert condition(custom)["reward_replacement_bounds"] == {
        "offline": 60.0,
        "online": 60.0,
    }


def test_cross_algorithm_plot_rejects_different_conditions():
    import pandas as pd

    conditions = [
        condition(config(online_corruption_scale_profile=scale))
        for scale in ("rpex_official_code", "dataset_std_scaled_extension")
    ]
    frame = pd.DataFrame(
        {
            "algorithm": ["rpex", "wsrl"],
            "env_name": ["halfcheetah"] * 2,
            "corruption": ["random"] * 2,
            "corruption_target": ["observations"] * 2,
            "protocol": ["test"] * 2,
            "score_semantics": ["test"] * 2,
            "benchmark_eligible": [True] * 2,
            "comparison_condition_signature": [
                canonical_json_sha256(c) for c in conditions
            ],
            "comparison_condition_json": [json.dumps(c) for c in conditions],
        }
    )
    with pytest.raises(
        RuntimeError, match="Mixed experimental conditions.*online_corruption"
    ):
        _validate_score_contract(frame, "test")


def test_latest_selection_keeps_one_shared_condition_and_reports_exclusions():
    def row(algorithm, date, shared, seed=0, variant="v1"):
        return dict(
            algorithm=algorithm,
            started_at=date,
            comparison_condition_signature=shared,
            aggregation_signature=variant,
            learner_seed=seed,
            corruption_seed=100 + seed,
            run_dir=f"{algorithm}/{date}/{seed}",
        )

    runs = [
        row("rpex", 2, "correct"),
        row("rpex", 3, "correct", 1),
        row("wsrl", 1, "correct"),
        row("wsrl", 4, "different"),
        row("rpex", 0, "old"),
    ]
    selected, excluded = select_latest_comparable_records(runs)
    assert selected == runs[:3]
    assert excluded == runs[3:]


def test_provenance_changes_when_dirty_content_changes_without_status_change(tmp_path):
    def git(*args):
        subprocess.run(["git", *args], cwd=tmp_path, check=True, capture_output=True)

    git("init")
    source = tmp_path / "learner.py"
    source.write_text("x = 0\n")
    git("add", "learner.py")
    git(
        "-c",
        "user.name=Audit Test",
        "-c",
        "user.email=audit@example.invalid",
        "commit",
        "-m",
        "initial",
    )
    source.write_text("x = 1\n")
    first = repository_state_metadata(tmp_path)
    source.write_text("x = 2\n")
    second = repository_state_metadata(tmp_path)
    assert first["repository_status_sha256"] == second["repository_status_sha256"]
    assert first["repository_code_sha256"] != second["repository_code_sha256"]
    assert first["repository_diff_sha256"] != second["repository_diff_sha256"]
    extra = tmp_path / "untracked.py"
    extra.write_text("x = 1\n")
    third = repository_state_metadata(tmp_path)
    extra.write_text("x = 2\n")
    fourth = repository_state_metadata(tmp_path)
    assert third["repository_status_sha256"] == fourth["repository_status_sha256"]
    assert third["repository_code_sha256"] != fourth["repository_code_sha256"]


@pytest.mark.parametrize(
    "profile", ["common_budget_robustness", "official_code_reference"]
)
@pytest.mark.parametrize("epsilon", [0.0, 1.0, 2.0])
def test_random_reward_replacement_matches_declared_support(profile, epsilon):
    c = config("rewards", implementation_profile=profile, corruption_range=epsilon)
    original = np.full(8, 500.0, np.float32)
    offline = corrupt_offline_reward_values(original, c, np.random.default_rng(7))
    expected = np.random.default_rng(7).uniform(-1, 1, size=8) * 30 * epsilon
    np.testing.assert_allclose(offline, expected, atol=3e-6)
    scale = 1.0 if profile == "official_code_reference" else epsilon
    online = corrupt_online_reward_value(500.0, c, np.random.default_rng(8))
    assert online == np.random.default_rng(8).uniform(-1, 1) * 30 * scale
    assert (
        reward_corruption_metadata(c, "online")["reward_corruption_high"] == 30 * scale
    )
    np.testing.assert_array_equal(original, np.full(8, 500.0))


def test_reward_mc_returns_use_poisoned_rewards_and_stop_at_boundaries(tmp_path):
    c = config("rewards", offline_corruption_rate=1.0, online_corruption_rate=1.0)
    data = {
        "observations": np.zeros((6, 3), np.float32),
        "next_observations": np.ones((6, 3), np.float32),
        "actions": np.zeros((6, 2), np.float32),
        "rewards": np.full(6, 500.0, np.float32),
        "terminals": np.zeros(6, np.float32),
        "episode_id": np.array([0, 0, 0, 1, 1, 1]),
        "mc_returns": np.full(6, 123456.0, np.float32),
    }
    poisoned, _ = corrupt_offline_dataset(data, c, None, tmp_path)
    r = poisoned["rewards"].astype(float)
    g = c.discount
    expected = [
        r[0] + g * r[1] + g * g * r[2],
        r[1] + g * r[2],
        r[2],
        r[3] + g * r[4] + g * g * r[5],
        r[4] + g * r[5],
        r[5],
    ]
    np.testing.assert_allclose(poisoned["mc_returns"], expected, atol=5e-6)
    cached, stats = corrupt_offline_dataset(data, c, None, tmp_path)
    assert stats["loaded_from_cache"]
    for key in poisoned:
        np.testing.assert_array_equal(poisoned[key], cached[key])
    accumulator = CalQLTrajectoryAccumulator(g)
    stored_rewards = []
    for i in range(2):
        transition = corrupt_online_transition(
            data["observations"][i],
            data["actions"][i],
            500.0,
            data["next_observations"][i],
            c,
            None,
            np.random.default_rng(i),
            np.ones(3),
            np.ones(2),
            selected_target="rewards",
            selection_already_sampled=True,
        )
        stored_rewards.append(transition[2])
        completed = accumulator.append(
            observation=transition[0],
            action=transition[1],
            reward=transition[2],
            next_observation=transition[3],
            terminal=False,
            timeout=i == 1,
        )
    np.testing.assert_allclose(
        completed.batch["mc_returns"],
        [stored_rewards[0] + g * stored_rewards[1], stored_rewards[1]],
        atol=5e-6,
    )
    np.testing.assert_array_equal(data["rewards"], np.full(6, 500.0))
