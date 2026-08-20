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
        "batch_size",
        "initial_collection_steps",
        "warmup_steps",
        "updates_per_step",
        "learning_rate",
        "actor_learning_rate",
        "critic_learning_rate",
        "temperature_learning_rate",
        "discount",
        "target_update_rate",
        "expectile",
        "beta",
        "riql_sigma",
        "riql_quantile",
        "num_critics",
        "inv_temperature",
        "kappa",
        "cql_alpha",
        "cql_alpha_online",
        "cql_n_actions",
        "cql_temperature",
        "calql_bc_warmup_steps",
        "backup_entropy",
        "cql_max_target_backup",
        "wsrl_num_critics",
        "wsrl_target_critic_subsample_size",
        "wsrl_per_critic_batch_size",
        "wsrl_utd_ratio",
        "pqe_replay_mode",
        "balanced_replay_temperature",
        "priority_floor",
        "effective_offline_ratio",
        "offline_corruption_rate",
        "online_corruption_rate",
        "corruption_range",
        "offline_attack_steps",
        "online_attack_steps",
        "attack_step_size",
        "online_attack_step_size",
        "mixed_ratios",
        "mc_return_source",
    )
    manifest = {
        "manifest_schema_version": 1,
        "algorithm": algorithm,
        "paper_title": resolved.get("paper_title"),
        "implementation_profile": resolved.get("implementation_profile"),
        "algorithm_profile": resolved.get("resolved_algorithm_profile"),
        "implementation_fidelity": resolved.get("implementation_fidelity"),
        "upstream_repository": UPSTREAM_REPOSITORIES.get(algorithm),
        "upstream_commit": resolved.get("upstream_commit"),
        "suite_profile": resolved.get("suite_profile"),
        "run_purpose": resolved.get("run_purpose"),
        "budget_profile": resolved.get("budget_profile"),
        "not_paper_reproduction": resolved.get("not_paper_reproduction"),
        "environment_protocol": resolved.get("environment_protocol"),
        "dataset_id": resolved.get("dataset_id"),
        "evaluation_env_id": resolved.get("evaluation_env_id"),
        "online_env_id": resolved.get("online_env_id"),
        "environment_fingerprint": resolved.get("environment_fingerprint"),
        "environment_horizon": resolved.get("environment_max_episode_steps"),
        "dataset_sha256": resolved.get("dataset_sha256"),
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
        "learner_seed": resolved.get("learner_seed"),
        "corruption_seed": resolved.get("corruption_seed"),
        "attack_seed": resolved.get("corruption_seed"),
        "replay_seed": resolved.get("replay_seed"),
        "train_env_seed": resolved.get("train_env_seed"),
        "eval_seed": resolved.get("eval_seed"),
        "offline_updates": resolved.get("offline_update_budget"),
        "online_environment_steps": resolved.get(
            "online_environment_step_budget"
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


SEED_FIELDS = {
    "learner_seed",
    "corruption_seed",
    "attack_seed",
    "replay_seed",
    "train_env_seed",
    "eval_seed",
    "manifest_sha256",
    "selected_transition_count",
    "selected_transition_hash",
    "corruption_value_hash",
}


def aggregation_signature(manifest: Mapping[str, Any]) -> str:
    comparable = {key: value for key, value in manifest.items() if key not in SEED_FIELDS}
    return canonical_json_sha256(comparable)


def canonical_manifest_json(manifest: Mapping[str, Any]) -> str:
    return json.dumps(
        manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
