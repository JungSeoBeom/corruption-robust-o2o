from __future__ import annotations

import json
from typing import Any, Mapping

from .fidelity import canonical_json_sha256


UPSTREAM_REPOSITORIES = {
    "rpex": "https://github.com/felix-thu/RPEX",
    "riql_naive": "https://github.com/felix-thu/RPEX",
    "riql_pex": "https://github.com/felix-thu/RPEX",
    "wsrl": "https://github.com/zhouzypaul/wsrl",
    "cal_ql": "https://github.com/nakamotoo/Cal-QL",
    "pessimistic_q_ensemble": "https://github.com/shlee94/Off2OnRL",
}


def build_experiment_manifest(resolved: Mapping[str, Any]) -> dict[str, Any]:
    algorithm = str(resolved["algorithm"])
    policy_distribution = str(resolved.get("action_distribution"))
    hyperparameter_fields = (
        "stage",
        "batch_size",
        "replay_size",
        "replay_sampling_profile",
        "offline_replay_sampler",
        "online_replay_sampler",
        "replay_seed_mapping",
        "replay_rng_parity_verified",
        "online_optimizer_transition",
        "online_actor_initialization",
        "online_phase_rng_parity_verified",
        "evaluation_action_sampling",
        "evaluation_env_strategy",
        "evaluation_seed_schedule",
        "evaluation_protocol_parity_verified",
        "max_episode_steps",
        "initial_collection_steps",
        "warmup_steps",
        "updates_per_step",
        "hidden_dim",
        "hidden_layers",
        "learning_rate",
        "actor_learning_rate",
        "critic_learning_rate",
        "temperature_learning_rate",
        "max_grad_norm",
        "discount",
        "target_update_rate",
        "normalize_states",
        "state_normalization",
        "deterministic_policy",
        "action_distribution",
        "evaluation_mode",
        "online_replay_profile",
        "evaluation_policy_profile",
        "attack_timing",
        "random_attack_semantics",
        "mixed_corruption_profile",
        "action_execution_profile",
        "policy_extraction",
        "task_profile",
        "adversarial_attack_profile",
        "allow_experimental_adversarial_attack",
        "online_corruption_scale_profile",
        "offline_adversarial_reward_rule",
        "online_adversarial_reward_rule",
        "expectile",
        "beta",
        "riql_sigma",
        "riql_quantile",
        "num_critics",
        "inv_temperature",
        "kappa",
        "riql_config_row",
        "riql_config_extension",
        "sac_num_critics",
        "lcb_ratio",
        "uncertainty_ratio",
        "uncertainty_basic",
        "uncertainty_min",
        "uncertainty_max",
        "entropy_lr",
        "cql_alpha",
        "cql_alpha_online",
        "cql_n_actions",
        "cql_temperature",
        "bc_steps",
        "calql_bc_warmup_steps",
        "backup_entropy",
        "cql_max_target_backup",
        "calibration_mask_mode",
        "wsrl_num_critics",
        "wsrl_target_critic_subsample_size",
        "wsrl_per_critic_batch_size",
        "wsrl_utd_ratio",
        "wsrl_layer_norm",
        "wsrl_entropy_profile",
        "target_entropy",
        "pqe_replay_mode",
        "balanced_replay_temperature",
        "priority_floor",
        "implementation_variant",
        "pqe_member_checkpoints",
        "ro2o_beta_policy",
        "ro2o_beta_ood",
        "ro2o_q_smooth_eps",
        "ro2o_policy_smooth_eps",
        "ro2o_ood_smooth_eps",
        "ro2o_sample_size",
        "ro2o_uncertainty",
        "ro2o_uncertainty_min",
        "ro2o_uncertainty_decay",
        "effective_offline_ratio",
        "offline_ratio",
        "offline_corruption_rate",
        "online_corruption_rate",
        "corruption_range",
        "offline_attack_steps",
        "online_attack_steps",
        "attack_step_size",
        "online_attack_step_size",
        "attack_min_step_size",
        "attack_norm",
        "mixed_ratios",
        "mc_return_source",
    )
    manifest = {
        "manifest_schema_version": 2,
        "repository_commit": resolved.get(
            "repository_commit", resolved.get("git_commit")
        ),
        "repository_dirty": resolved.get("repository_dirty"),
        "repository_worktree_sha256": resolved.get(
            "repository_worktree_sha256"
        ),
        "benchmark_seed_set": resolved.get("benchmark_seed_set"),
        "controller_seed_cohort_attested": resolved.get(
            "controller_seed_cohort_attested", False
        ),
        "final_audit_context_token": resolved.get("final_audit_context_token"),
        "final_audit_receipt_sha256": resolved.get(
            "final_audit_receipt_sha256"
        ),
        "publication_scope": (
            "individual_seed_member_requires_complete_cohort_aggregation"
            if resolved.get("run_purpose") == "final_benchmark"
            else "non_publication_run"
        ),
        "algorithm": algorithm,
        "paper_title": resolved.get("paper_title"),
        "implementation_profile": resolved.get("implementation_profile"),
        "algorithm_profile": resolved.get("resolved_algorithm_profile"),
        "implementation_fidelity": resolved.get("implementation_fidelity"),
        "reproduction_status": resolved.get("implementation_fidelity"),
        "condition_status": resolved.get("condition_status"),
        "corruption_protocol_source": resolved.get(
            "corruption_protocol_source"
        ),
        "corruption_fixture_id": resolved.get("corruption_fixture_id"),
        "corruption_fixture_verified": resolved.get(
            "corruption_fixture_verified"
        ),
        "upstream_repository": UPSTREAM_REPOSITORIES.get(algorithm),
        "upstream_commit": resolved.get("upstream_commit"),
        "suite_profile": resolved.get("suite_profile"),
        "run_purpose": resolved.get("run_purpose"),
        "budget_profile": resolved.get("budget_profile"),
        "not_paper_reproduction": resolved.get("not_paper_reproduction"),
        "environment_protocol": resolved.get("environment_protocol"),
        "score_semantics": resolved.get("score_semantics"),
        "dataset_id": resolved.get("dataset_id"),
        "dataset_name": resolved.get("dataset_id"),
        "evaluation_env_id": resolved.get("evaluation_env_id"),
        "online_env_id": resolved.get("online_env_id"),
        "environment_fingerprint": resolved.get("environment_fingerprint"),
        "environment_horizon": resolved.get("environment_max_episode_steps"),
        "dataset_sha256": resolved.get("dataset_sha256"),
        "environment_versions": resolved.get("runtime_package_versions"),
        "mujoco_py_version": resolved.get("mujoco_py_version"),
        "mujoco_runtime_version_code": resolved.get(
            "mujoco_runtime_version_code"
        ),
        "mujoco_runtime_version": resolved.get("mujoco_runtime_version"),
        "requested_device": resolved.get("device"),
        "resolved_device": resolved.get("resolved_device"),
        "normalizer_sha256": resolved.get("normalizer_sha256"),
        "normalization_enabled": resolved.get("normalize_states"),
        "normalization_source": resolved.get("state_normalization"),
        "normalization_before_or_after_corruption": "after_corruption",
        "task_profile": resolved.get("task_profile"),
        "policy_distribution": policy_distribution,
        "state_dependent_std": policy_distribution == "tanh_gaussian",
        "action_squashing": policy_distribution == "tanh_gaussian",
        "action_clipping": resolved.get("action_execution_profile")
        == "clip_to_action_space",
        "network_architecture": {
            "hidden_dim": resolved.get("hidden_dim"),
            "hidden_layers": resolved.get("hidden_layers"),
            "critic_count": resolved.get("num_critics")
            if algorithm in ("rpex", "riql_naive", "riql_pex")
            else resolved.get("sac_num_critics"),
            "wsrl_layer_norm": resolved.get("wsrl_layer_norm"),
            "wsrl_final_hidden_kernel_scale": (
                1e-2 if algorithm == "wsrl" else None
            ),
        },
        "evaluation_policy_profile": resolved.get("evaluation_policy_profile"),
        "target_entropy": resolved.get("target_entropy"),
        "wsrl_entropy_profile": resolved.get("wsrl_entropy_profile"),
        "evaluation_mode": resolved.get("evaluation_mode"),
        "online_replay_profile": resolved.get("online_replay_profile"),
        "attack_semantics": resolved.get("random_attack_semantics"),
        "attack_timing": resolved.get("attack_timing"),
        "attack_implementation": resolved.get("adversarial_attack_profile"),
        "adversarial_attack_profile": resolved.get("adversarial_attack_profile"),
        "online_corruption_scale_profile": resolved.get(
            "online_corruption_scale_profile"
        ),
        "offline_adversarial_reward_rule": resolved.get(
            "offline_adversarial_reward_rule"
        ),
        "online_adversarial_reward_rule": resolved.get(
            "online_adversarial_reward_rule"
        ),
        "attack_implementation_version": resolved.get(
            "offline_corruption", {}
        ).get("attack_implementation_version"),
        "attacker_checkpoint_sha256": resolved.get(
            "offline_corruption", {}
        ).get("attack_checkpoint_fingerprint"),
        "attacker_checkpoint_source": resolved.get("attack_checkpoint_source"),
        "attacker_checkpoint_expected_sha256": resolved.get(
            "attack_checkpoint_expected_sha256"
        ),
        "corruption": resolved.get("corruption"),
        "attack_mode": resolved.get("corruption"),
        "corruption_target": resolved.get("corruption_target"),
        "corruption_rate": {
            "offline": resolved.get("offline_corruption_rate"),
            "online": resolved.get("online_corruption_rate"),
        },
        "corruption_range": resolved.get("corruption_range"),
        "reward_corruption_support": {
            "distribution": resolved.get("offline_corruption", {}).get(
                "reward_corruption_distribution"
            ),
            "low": resolved.get("offline_corruption", {}).get(
                "reward_corruption_low"
            ),
            "high": resolved.get("offline_corruption", {}).get(
                "reward_corruption_high"
            ),
        },
        "selected_transition_count": resolved.get("offline_corruption", {}).get(
            "selected_transition_count"
        ),
        "selected_transition_hash": resolved.get("offline_corruption", {}).get(
            "selected_transition_indices_sha256"
        ),
        "corruption_value_hash": resolved.get("offline_corruption", {}).get(
            "corruption_value_sha256"
        ),
        "corruption_artifact_hash": resolved.get("offline_corruption", {}).get(
            "final_artifact_sha256"
        ),
        "corruption_rng_implementation": resolved.get(
            "offline_corruption", {}
        ).get("rng_implementation"),
        "base_seed": resolved.get("seed"),
        "learner_seed": resolved.get("learner_seed"),
        "corruption_seed": resolved.get("corruption_seed"),
        "attack_seed": resolved.get("corruption_seed"),
        "replay_seed": resolved.get("replay_seed"),
        "replay_sampling_profile": resolved.get("replay_sampling_profile"),
        "offline_replay_sampler": resolved.get("offline_replay_sampler"),
        "online_replay_sampler": resolved.get("online_replay_sampler"),
        "replay_seed_mapping": resolved.get("replay_seed_mapping"),
        "replay_rng_parity_verified": resolved.get(
            "replay_rng_parity_verified"
        ),
        "online_optimizer_transition": resolved.get(
            "online_optimizer_transition"
        ),
        "online_actor_initialization": resolved.get(
            "online_actor_initialization"
        ),
        "online_phase_rng_parity_verified": resolved.get(
            "online_phase_rng_parity_verified"
        ),
        "evaluation_action_sampling": resolved.get(
            "evaluation_action_sampling"
        ),
        "evaluation_env_strategy": resolved.get("evaluation_env_strategy"),
        "evaluation_seed_schedule": resolved.get("evaluation_seed_schedule"),
        "evaluation_protocol_parity_verified": resolved.get(
            "evaluation_protocol_parity_verified"
        ),
        "train_env_seed": resolved.get("train_env_seed"),
        "eval_seed": resolved.get("eval_seed"),
        "offline_updates": resolved.get("offline_update_budget"),
        "online_environment_steps": resolved.get(
            "online_environment_step_budget"
        ),
        "requested_online_steps": resolved.get(
            "online_environment_step_budget"
        ),
        "actual_online_steps": resolved.get("actual_online_steps"),
        "online_budget_semantics": resolved.get(
            "online_budget_semantics",
            (
                "rpex_official_episode_boundary_strict_greater_than"
                if resolved.get("implementation_profile")
                == "official_code_reference"
                and algorithm in ("rpex", "riql_naive", "riql_pex")
                else "exact_environment_steps"
            ),
        ),
        "episode_boundary_overshoot": resolved.get(
            "episode_boundary_overshoot"
        ),
        "evaluation_interval": resolved.get("eval_period"),
        "evaluation_episodes": resolved.get("eval_episodes"),
        "reporting_rule": resolved.get("reporting_rule"),
        "reporting_rule_verified": resolved.get("reporting_rule_verified"),
        "publication_eligible": resolved.get("publication_eligible", False),
        "paper_reproduction_eligible": resolved.get(
            "paper_reproduction_eligible", False
        ),
        "utd_ratio": resolved.get("utd_ratio"),
        "per_condition_hyperparameter_row": resolved.get("riql_config_row"),
        "per_condition_hyperparameter_extension": resolved.get(
            "riql_config_extension"
        ),
        "calibration_mask_mode": resolved.get("calibration_mask_mode"),
        "oracle_information": resolved.get("oracle_information"),
        "pqe_implementation_variant": resolved.get("implementation_variant"),
        "mixed_corruption_profile": resolved.get("mixed_corruption_profile"),
        "resolved_hyperparameters": {
            field: resolved.get(field) for field in hyperparameter_fields
        },
    }
    manifest["manifest_sha256"] = canonical_json_sha256(manifest)
    return manifest


def verify_experiment_manifest(manifest: Mapping[str, Any]) -> str:
    """Verify and return the full immutable launch-artifact digest.

    The launch digest deliberately covers provenance such as the exact audit
    receipt consumed at launch.  That receipt is immutable evidence, but its
    issuance timestamp/PID make it unsuitable as a behavioral resume key.
    """

    recorded = manifest.get("manifest_sha256")
    if not isinstance(recorded, str) or not recorded:
        raise ValueError("experiment manifest has no SHA256 digest")
    payload = {
        key: value for key, value in manifest.items() if key != "manifest_sha256"
    }
    actual = canonical_json_sha256(payload)
    if actual != recorded:
        raise ValueError(
            "experiment manifest SHA256 mismatch: "
            f"recorded={recorded}, actual={actual}"
        )
    return recorded


_RESUME_TRANSIENT_PROVENANCE_FIELDS = {
    "manifest_sha256",
    # A fresh PASS/READY audit receipt is issued when a stopped controller is
    # restarted.  Its digest changes because the receipt records issuance time
    # and origin PID; the covered context token remains the stable provenance
    # identity and is intentionally *not* excluded here.
    "final_audit_receipt_sha256",
}


def resume_identity_signature(manifest: Mapping[str, Any]) -> str:
    """Return the behavior/provenance identity that an exact resume must match."""

    comparable = {
        key: value
        for key, value in manifest.items()
        if key not in _RESUME_TRANSIENT_PROVENANCE_FIELDS
    }
    return canonical_json_sha256(comparable)


SEED_FIELDS = {
    "base_seed",
    "learner_seed",
    "corruption_seed",
    "attack_seed",
    "replay_seed",
    "train_env_seed",
    "eval_seed",
    "manifest_sha256",
    "launch_manifest_sha256",
    "completion_manifest_sha256",
    "selected_transition_count",
    "selected_transition_hash",
    "corruption_value_hash",
    "corruption_artifact_hash",
    # The normalizer is fitted after corruption. Observation/dynamics
    # corruption therefore makes this a seed-derived artifact even when the
    # normalization rule itself is identical across runs.
    "normalizer_sha256",
    # Official RPEX-style episode-boundary stopping can overshoot by a
    # seed-dependent number of environment steps.  These are run outcomes,
    # not experimental settings, so they must not split otherwise comparable
    # seeds into different aggregation groups.
    "actual_online_steps",
    "episode_boundary_overshoot",
    # Audit receipts are per-controller issuance evidence, not an algorithmic
    # setting.  The stable audit context token remains in the aggregation key.
    "final_audit_receipt_sha256",
}


def aggregation_signature(manifest: Mapping[str, Any]) -> str:
    comparable = {key: value for key, value in manifest.items() if key not in SEED_FIELDS}
    return canonical_json_sha256(comparable)


def canonical_manifest_json(manifest: Mapping[str, Any]) -> str:
    return json.dumps(
        manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
