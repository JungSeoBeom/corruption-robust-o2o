from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


IMPLEMENTATION_PROFILES = (
    "official_code_reference",
    "paper_reference",
    "common_budget_robustness",
    "locomotion_port",
    "legacy_current",
    "experimental_approximation",
)
IMPLEMENTATION_FIDELITIES = (
    "exact_upstream_port",
    "paper_code_conflict",
    "task_port",
    "approximation",
    "legacy_unknown",
)
SUITE_PROFILES = (
    "primary_research_benchmark",
    "common_budget_diagnostic",
    # Backward-compatible names retained for existing commands/results.
    "method_fidelity",
    "common_budget_robustness",
)
RUN_PURPOSES = ("smoke", "diagnostic", "final_benchmark")

ONLINE_REPLAY_PROFILES = (
    "official_code_online_only",
    "paper_offline_online_mixture",
    "fixed_offline_online_mixture",
    "balanced_density_replay",
)
EVALUATION_POLICY_PROFILES = (
    "official_code_epsilon_switching",
    "paper_greedy_highest_weight",
    "deterministic_diagnostic",
)
ATTACK_TIMINGS = (
    "official_code_post_transition_replay_poisoning",
    "paper_pre_action_sensor_actuator",
)
RANDOM_ATTACK_SEMANTICS = (
    "post_transition_replay_poisoning",
    "pre_action_sensor_actuator_corruption",
)
MIXED_CORRUPTION_PROFILES = ("generic_partitioned_mixed", "rpex_paper_mixed")
ACTION_EXECUTION_PROFILES = (
    "official_algorithm_behavior",
    "clip_to_action_space",
)
LEGACY_ACTION_EXECUTION_PROFILE_ALIASES = {
    "official_unclipped": "official_algorithm_behavior",
    "environment_clip": "clip_to_action_space",
}
POLICY_EXTRACTIONS = ("awr", "align_iql")
TASK_PROFILES = ("official_supported_task", "d4rl_locomotion_port")
ADVERSARIAL_ATTACK_PROFILES = (
    "rpex_official_adam",
    "experimental_sign_pgd",
)
ONLINE_CORRUPTION_SCALE_PROFILES = (
    "rpex_official_code",
    "dataset_std_scaled_extension",
)
WSRL_ENTROPY_PROFILES = (
    "official_negative_action_dim",
    "legacy_zero",
)
OFFLINE_ADVERSARIAL_REWARD_RULES = ("official_sign_flip",)
ONLINE_ADVERSARIAL_REWARD_RULES = (
    "official_uniform_replacement",
    "experimental_scaled_sign_flip",
)


UPSTREAM_COMMITS: Mapping[str, str] = {
    "rpex": "35da71ee5151b6179d21b9a2b4ce1b6408aedd04",
    "wsrl": "ad4dc1248a138bc15d6e053f2d1dba1b8cfbaca2",
    "cal_ql": "ac6eafec22e8d60836573e1f488c7f626ce8a77e",
    "pessimistic_q_ensemble": "6f298fa9ef040d725067d0f2775022bd2900d635",
}


@dataclass(frozen=True)
class RIQLReferenceRow:
    sigma: float
    quantile: float
    num_critics: int
    inverse_temperature: float = 3.0
    kappa: float = 0.1
    utd_ratio: int = 1
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    extension: bool = False


# Literal port of felix-thu/RPEX@35da71e:RIQL_TRAIN_CONFIG.py.  The upstream
# table has no clean or mixed row; callers must deliberately select the
# common-budget extension for those combinations.
_RANDOM = {
    "observations": {
        "halfcheetah": (0.1, 0.1, 5),
        "walker2d": (0.1, 0.25, 5),
        "hopper": (0.1, 0.25, 3),
    },
    "actions": {
        "halfcheetah": (0.5, 0.25, 3),
        "walker2d": (0.5, 0.1, 5),
        "hopper": (0.1, 0.25, 5),
    },
    "rewards": {
        "halfcheetah": (3.0, 0.25, 5),
        "walker2d": (3.0, 0.1, 5),
        "hopper": (1.0, 0.25, 3),
    },
    "dynamics": {
        "halfcheetah": (3.0, 0.25, 5),
        "walker2d": (1.0, 0.25, 3),
        "hopper": (1.0, 0.5, 5),
    },
}
_ADVERSARIAL = {
    "observations": {
        "halfcheetah": (0.1, 0.1, 5),
        "walker2d": (1.0, 0.25, 5),
        "hopper": (1.0, 0.25, 5),
    },
    "actions": {
        "halfcheetah": (1.0, 0.1, 5),
        "walker2d": (1.0, 0.1, 5),
        "hopper": (1.0, 0.25, 5),
    },
    "rewards": {
        "halfcheetah": (1.0, 0.1, 5),
        "walker2d": (3.0, 0.1, 5),
        "hopper": (0.1, 0.25, 5),
    },
    "dynamics": {
        "halfcheetah": (1.0, 0.1, 5),
        "walker2d": (1.0, 0.25, 5),
        "hopper": (1.0, 0.5, 5),
    },
}


def resolve_riql_reference_row(
    env_name: str,
    corruption: str,
    corruption_target: str,
    *,
    allow_extension: bool,
) -> tuple[str, RIQLReferenceRow]:
    domain = env_name.split("-", 1)[0]
    table = {"random": _RANDOM, "adversarial": _ADVERSARIAL}.get(corruption)
    if table is not None and corruption_target in table:
        sigma, quantile, critics = table[corruption_target][domain]
        key = f"{corruption}/{corruption_target}/{domain}"
        return key, RIQLReferenceRow(sigma, quantile, critics)
    if not allow_extension:
        raise ValueError(
            "RPEX RIQL_TRAIN_CONFIG.py has no official row for "
            f"{env_name} {corruption}/{corruption_target}; select "
            "suite_profile=common_budget_robustness for an explicit extension"
        )
    key = f"extension/{corruption}/{corruption_target}/{domain}"
    return key, RIQLReferenceRow(3.0, 0.1, 5, extension=True)


def canonical_json_sha256(payload: object) -> str:
    import hashlib
    import json

    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
