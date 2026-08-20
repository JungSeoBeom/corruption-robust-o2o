from __future__ import annotations

import argparse
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from .fidelity import (
    ACTION_EXECUTION_PROFILES,
    ADVERSARIAL_ATTACK_PROFILES,
    ATTACK_TIMINGS,
    EVALUATION_POLICY_PROFILES,
    IMPLEMENTATION_FIDELITIES,
    IMPLEMENTATION_PROFILES,
    LEGACY_ACTION_EXECUTION_PROFILE_ALIASES,
    MIXED_CORRUPTION_PROFILES,
    OFFLINE_ADVERSARIAL_REWARD_RULES,
    ONLINE_ADVERSARIAL_REWARD_RULES,
    ONLINE_CORRUPTION_SCALE_PROFILES,
    ONLINE_REPLAY_PROFILES,
    POLICY_EXTRACTIONS,
    RANDOM_ATTACK_SEMANTICS,
    RUN_PURPOSES,
    SUITE_PROFILES,
    TASK_PROFILES,
    UPSTREAM_COMMITS,
    WSRL_ENTROPY_PROFILES,
    resolve_riql_reference_row,
)


ALGORITHMS = (
    "rpex",
    "riql_pex",
    "riql_naive",
    "uwmsg",
    "pex",
    "cal_ql",
    "wsrl",
    "ro2o",
    "pessimistic_q_ensemble",
)

LEGACY_PROTOCOL = "rpex_d4rl_v2_legacy"
LOCAL_PROTOCOL = "local_gymnasium_v4_diagnostic"
LEGACY_LOCAL_PROTOCOL_ALIAS = "local_gymnasium_v4"
DEFAULT_PROTOCOL = LEGACY_PROTOCOL
PROTOCOLS = (LEGACY_PROTOCOL, LOCAL_PROTOCOL, LEGACY_LOCAL_PROTOCOL_ALIAS)
ALGORITHM_PROFILES = IMPLEMENTATION_PROFILES
CALIBRATION_MASK_MODES = ("all", "oracle_exclude_corrupted", "disabled")

ACTION_DIMS = {
    "halfcheetah": 6,
    "hopper": 3,
    "walker2d": 6,
}

ALGORITHM_TITLES = {
    "rpex": "Robust Policy Expansion for Offline-to-Online RL under Diverse Data Corruption",
    "riql_pex": "RPEX ablation: RIQL + Policy Expansion",
    "riql_naive": "Towards Robust Offline Reinforcement Learning under Diverse Data Corruption",
    "uwmsg": "Corruption-Robust Offline Reinforcement Learning with General Function Approximation",
    "pex": "Policy Expansion for Bridging Offline-to-Online Reinforcement Learning",
    "cal_ql": "Cal-QL: Calibrated Offline RL Pre-Training for Efficient Online Fine-Tuning",
    "wsrl": "Efficient Online Reinforcement Learning Fine-Tuning Need Not Retain Offline Data",
    "ro2o": "Towards Robust Offline-to-Online Reinforcement Learning via Uncertainty and Smoothness",
    "pessimistic_q_ensemble": (
        "Offline-to-Online Reinforcement Learning via Balanced Replay "
        "and Pessimistic Q-Ensemble"
    ),
}

ALGORITHM_ALIASES = {
    "riql+pex": "riql_pex",
    "riql-pex": "riql_pex",
    "riql_naive": "riql_naive",
    "riql-naive": "riql_naive",
    "cal-ql": "cal_ql",
    "calql": "cal_ql",
    "pqe": "pessimistic_q_ensemble",
    "pqe_shared_actor": "pessimistic_q_ensemble",
    "pessimistic-q-ensemble": "pessimistic_q_ensemble",
}

CORRUPTION_MODES = ("clean", "random", "adversarial")
INDIVIDUAL_CORRUPTION_TARGETS = (
    "observations",
    "actions",
    "rewards",
    "dynamics",
)
CORRUPTION_TARGETS = ("none", *INDIVIDUAL_CORRUPTION_TARGETS, "mixed")

# These are the MuJoCo locomotion tasks used throughout the supplied RPEX code.
BENCHMARK_ENVS = tuple(
    f"{domain}-{dataset}-v2"
    for domain in ("halfcheetah", "hopper", "walker2d")
    for dataset in ("medium", "medium-replay", "medium-expert")
)

ENV_NAME_PREFIX_ALIASES = {
    "half-cheetah-": "halfcheetah-",
    "half_cheetah-": "halfcheetah-",
    "walker-2d-": "walker2d-",
    "walker_2d-": "walker2d-",
}


def normalize_env_name(env_name: str) -> str:
    normalized = env_name.strip().lower()
    for alias, canonical in ENV_NAME_PREFIX_ALIASES.items():
        if normalized.startswith(alias):
            return canonical + normalized[len(alias) :]
    return normalized


@dataclass
class ExperimentConfig:
    algorithm: str
    env_name: str
    corruption: str = "clean"
    corruption_target: str = "none"
    stage: str = "both"
    seed: int = 0
    protocol: str = DEFAULT_PROTOCOL
    implementation_profile: Optional[str] = None
    implementation_fidelity: Optional[str] = None
    suite_profile: str = "common_budget_robustness"
    run_purpose: str = "diagnostic"
    # Input-only compatibility shim.  ``reference`` is accepted only so old
    # commands fail with a precise migration message instead of being silently
    # promoted to a paper reference.
    algorithm_profile: Optional[str] = None
    allow_diagnostic_protocol: bool = False
    allow_legacy_checkpoint_without_fingerprint: bool = False

    # ``seed`` remains the stable public/base seed. Role seeds are derived once
    # and serialized so preprocessing, replay and evaluation RNG streams cannot
    # silently affect learner initialization.
    learner_seed: Optional[int] = None
    corruption_seed: Optional[int] = None
    replay_seed: Optional[int] = None
    train_env_seed: Optional[int] = None
    eval_seed: Optional[int] = None

    output_dir: str = "results"
    dataset_dir: Optional[str] = None
    checkpoint: Optional[str] = None
    initialize_from_checkpoint: Optional[str] = None
    resume_run: Optional[str] = None
    attack_checkpoint: Optional[str] = None
    attack_checkpoint_sha256: Optional[str] = None
    comparison_name: Optional[str] = None
    device: str = "auto"
    cuda_device: int = 0
    diagnostic_mode: bool = False

    offline_steps: int = 500_000
    online_steps: int = 500_000
    initial_collection_steps: int = 5_000
    warmup_steps: int = 5_000
    updates_per_step: int = 1
    batch_size: int = 256
    replay_size: int = 1_000_000
    eval_period: int = 10_000
    eval_episodes: int = 10
    checkpoint_period: int = 100_000
    offline_checkpoint_period: Optional[int] = None
    online_checkpoint_period: Optional[int] = None
    keep_last_checkpoints: int = 5
    train_log_period: int = 1_000
    max_episode_steps: int = 1_000

    hidden_dim: int = 256
    hidden_layers: int = 2
    learning_rate: float = 3e-4
    actor_learning_rate: Optional[float] = None
    critic_learning_rate: Optional[float] = None
    temperature_learning_rate: Optional[float] = None
    max_grad_norm: Optional[float] = None
    discount: float = 0.99
    target_update_rate: float = 0.005
    normalize_states: bool = True
    state_normalization: str = "standard"
    deterministic_policy: bool = False
    action_distribution: str = "tanh_gaussian"
    evaluation_mode: Optional[str] = None
    online_replay_profile: str = "official_code_online_only"
    evaluation_policy_profile: str = "official_code_epsilon_switching"
    attack_timing: str = "official_code_post_transition_replay_poisoning"
    random_attack_semantics: str = "post_transition_replay_poisoning"
    mixed_corruption_profile: str = "generic_partitioned_mixed"
    action_execution_profile: str = "clip_to_action_space"
    policy_extraction: Optional[str] = None
    task_profile: Optional[str] = None
    adversarial_attack_profile: Optional[str] = None
    allow_experimental_adversarial_attack: bool = False
    online_corruption_scale_profile: Optional[str] = None
    offline_adversarial_reward_rule: Optional[str] = None
    online_adversarial_reward_rule: Optional[str] = None

    # IQL / RIQL / PEX / RPEX
    expectile: float = 0.7
    beta: float = 3.0
    riql_sigma: float = 3.0
    riql_quantile: float = 0.1
    num_critics: int = 5
    inv_temperature: float = 3.0
    kappa: float = 0.1
    riql_config_row: Optional[str] = None
    riql_config_extension: bool = False

    # SAC ensemble / UWMSG / RO2O / Pessimistic Q Ensemble
    sac_num_critics: int = 10
    lcb_ratio: float = 4.0
    uncertainty_ratio: float = 0.7
    uncertainty_basic: float = 0.0
    uncertainty_min: float = 1.0
    uncertainty_max: float = 10.0
    entropy_lr: float = 3e-4

    # CQL / Cal-QL
    cql_alpha: float = 5.0
    cql_alpha_online: float = 1.0
    cql_n_actions: int = 10
    cql_temperature: float = 1.0
    bc_steps: Optional[int] = None
    calql_bc_warmup_steps: Optional[int] = None
    backup_entropy: bool = False
    cql_max_target_backup: Optional[bool] = None
    calibration_mask_mode: str = "all"
    mc_return_source: str = "post_corruption"

    # WSRL / REDQ reference controls. ``updates_per_step`` remains a generic
    # legacy knob; the reference WSRL schedule is resolved independently.
    wsrl_num_critics: Optional[int] = None
    wsrl_target_critic_subsample_size: Optional[int] = None
    wsrl_layer_norm: Optional[bool] = None
    wsrl_utd_ratio: Optional[int] = None
    wsrl_per_critic_batch_size: int = 256
    wsrl_entropy_profile: str = "official_negative_action_dim"
    target_entropy: Optional[float] = None

    # Off2OnRL balanced replay. ``uniform`` is an explicit ablation.
    pqe_replay_mode: str = "balanced_density"
    balanced_replay_temperature: float = 5.0
    priority_floor: float = 1e-3
    implementation_variant: Optional[str] = None
    pqe_member_checkpoints: Tuple[str, ...] = ()

    # RO2O
    ro2o_beta_policy: float = 1.0
    ro2o_beta_ood: float = 0.1
    ro2o_q_smooth_eps: float = 0.01
    ro2o_policy_smooth_eps: float = 0.01
    ro2o_ood_smooth_eps: float = 0.01
    ro2o_sample_size: int = 20
    ro2o_uncertainty: float = 1.0
    ro2o_uncertainty_min: float = 0.1
    ro2o_uncertainty_decay: float = 5e-7

    # RPEX corruption defaults: offline rate=0.3, online rate=0.5.
    corruption_range: float = 1.0
    offline_corruption_rate: float = 0.3
    online_corruption_rate: float = 0.5
    offline_attack_steps: int = 100
    online_attack_steps: int = 2
    attack_step_size: float = 0.01
    online_attack_step_size: Optional[float] = None
    attack_min_step_size: float = 0.0
    attack_norm: str = "linf"
    force_regenerate_attack: bool = False
    mixed_ratios: Tuple[float, float, float, float] = (0.25, 0.25, 0.25, 0.25)

    # None selects the RPEX per-algorithm replay rule.
    offline_ratio: Optional[float] = None

    def __post_init__(self) -> None:
        self.algorithm = ALGORITHM_ALIASES.get(self.algorithm.lower(), self.algorithm.lower())
        self.env_name = normalize_env_name(self.env_name)
        self.corruption = self.corruption.lower()
        self.corruption_target = self.corruption_target.lower()
        self.stage = self.stage.lower()
        self.protocol = self.protocol.lower()
        if self.protocol == LEGACY_LOCAL_PROTOCOL_ALIAS:
            self.protocol = LOCAL_PROTOCOL
        if self.implementation_profile is not None:
            self.implementation_profile = self.implementation_profile.lower()
        if self.algorithm_profile is not None:
            self.algorithm_profile = self.algorithm_profile.lower()
        self.suite_profile = self.suite_profile.lower()
        self.run_purpose = self.run_purpose.lower()
        self.state_normalization = self.state_normalization.lower()
        self.action_distribution = self.action_distribution.lower()
        self.evaluation_mode = (
            self.evaluation_mode.lower() if self.evaluation_mode else None
        )
        self.online_replay_profile = self.online_replay_profile.lower()
        self.evaluation_policy_profile = self.evaluation_policy_profile.lower()
        self.attack_timing = self.attack_timing.lower()
        self.random_attack_semantics = self.random_attack_semantics.lower()
        self.mixed_corruption_profile = self.mixed_corruption_profile.lower()
        self.action_execution_profile = LEGACY_ACTION_EXECUTION_PROFILE_ALIASES.get(
            self.action_execution_profile.lower(), self.action_execution_profile.lower()
        )
        self.policy_extraction = (
            self.policy_extraction.lower() if self.policy_extraction else None
        )
        self.task_profile = self.task_profile.lower() if self.task_profile else None
        self.adversarial_attack_profile = (
            self.adversarial_attack_profile.lower()
            if self.adversarial_attack_profile is not None
            else "rpex_official_adam"
        )
        if self.adversarial_attack_profile == "official_adam":
            self.adversarial_attack_profile = "rpex_official_adam"
        self.online_corruption_scale_profile = (
            self.online_corruption_scale_profile.lower()
            if self.online_corruption_scale_profile is not None
            else None
        )
        self.offline_adversarial_reward_rule = (
            self.offline_adversarial_reward_rule.lower()
            if self.offline_adversarial_reward_rule is not None
            else None
        )
        self.online_adversarial_reward_rule = (
            self.online_adversarial_reward_rule.lower()
            if self.online_adversarial_reward_rule is not None
            else None
        )
        self.wsrl_entropy_profile = self.wsrl_entropy_profile.lower()
        self.mc_return_source = self.mc_return_source.lower()
        self.calibration_mask_mode = self.calibration_mask_mode.lower()
        self.pqe_replay_mode = self.pqe_replay_mode.lower()
        self.implementation_variant = (
            "pqe_shared_actor_approx"
            if self.algorithm == "pessimistic_q_ensemble"
            else None
        )
        self.pqe_member_checkpoints = tuple(self.pqe_member_checkpoints)
        self.attack_norm = self.attack_norm.lower()
        self.mixed_ratios = tuple(float(value) for value in self.mixed_ratios)

        self._resolve_role_seeds()
        self._resolve_implementation_profile()
        if self.online_corruption_scale_profile is None:
            self.online_corruption_scale_profile = (
                "rpex_official_code"
                if self.implementation_profile == "official_code_reference"
                else "dataset_std_scaled_extension"
            )
        if self.offline_adversarial_reward_rule is None:
            self.offline_adversarial_reward_rule = "official_sign_flip"
        if self.online_adversarial_reward_rule is None:
            self.online_adversarial_reward_rule = (
                "official_uniform_replacement"
                if self.adversarial_attack_profile == "rpex_official_adam"
                else "experimental_scaled_sign_flip"
            )
        self._resolve_algorithm_profile()
        if self.online_attack_step_size is None:
            self.online_attack_step_size = (
                0.1
                if self.adversarial_attack_profile == "rpex_official_adam"
                else self.attack_step_size
            )
        if (
            self.corruption == "adversarial"
            and self.corruption_target != "rewards"
            and self.adversarial_attack_profile == "rpex_official_adam"
            and (
                self.offline_attack_steps != 100
                or self.online_attack_steps != 2
                or not math.isclose(self.attack_step_size, 0.01)
                or not math.isclose(self.online_attack_step_size, 0.1)
            )
        ):
            raise ValueError(
                "rpex_official_adam requires the pinned upstream schedule: "
                "offline=100x0.01 and online=2x0.1"
            )
        if self.evaluation_mode is None:
            self.evaluation_mode = (
                "method_faithful"
                if self.implementation_profile
                in ("official_code_reference", "paper_reference")
                else "deterministic_diagnostic"
            )

        if not self.normalize_states:
            self.state_normalization = "none"
        self.normalize_states = self.state_normalization != "none"

        if self.algorithm not in ALGORITHMS:
            raise ValueError(f"Unknown algorithm {self.algorithm!r}; choose from {ALGORITHMS}")
        if self.env_name not in BENCHMARK_ENVS:
            raise ValueError(
                f"Unknown benchmark environment {self.env_name!r}; "
                f"choose from {BENCHMARK_ENVS}"
            )
        if self.corruption not in CORRUPTION_MODES:
            raise ValueError(f"Unknown corruption {self.corruption!r}")
        if self.corruption_target not in CORRUPTION_TARGETS:
            raise ValueError(f"Unknown corruption target {self.corruption_target!r}")
        if self.stage not in ("offline", "online", "both"):
            raise ValueError("stage must be offline, online, or both")
        if self.protocol not in (LEGACY_PROTOCOL, LOCAL_PROTOCOL):
            raise ValueError(
                f"Unknown protocol {self.protocol!r}; choose from {PROTOCOLS}"
            )
        if self.implementation_profile not in IMPLEMENTATION_PROFILES:
            raise ValueError(
                f"Unknown implementation_profile {self.implementation_profile!r}; "
                f"choose from {IMPLEMENTATION_PROFILES}"
            )
        if self.implementation_fidelity not in IMPLEMENTATION_FIDELITIES:
            raise ValueError(f"Unknown implementation_fidelity {self.implementation_fidelity!r}")
        if self.suite_profile not in SUITE_PROFILES:
            raise ValueError(f"Unknown suite_profile {self.suite_profile!r}")
        if self.run_purpose not in RUN_PURPOSES:
            raise ValueError(f"Unknown run_purpose {self.run_purpose!r}")
        for value, choices, label in (
            (self.online_replay_profile, ONLINE_REPLAY_PROFILES, "online_replay_profile"),
            (self.evaluation_policy_profile, EVALUATION_POLICY_PROFILES, "evaluation_policy_profile"),
            (self.attack_timing, ATTACK_TIMINGS, "attack_timing"),
            (self.random_attack_semantics, RANDOM_ATTACK_SEMANTICS, "random_attack_semantics"),
            (self.mixed_corruption_profile, MIXED_CORRUPTION_PROFILES, "mixed_corruption_profile"),
            (self.action_execution_profile, ACTION_EXECUTION_PROFILES, "action_execution_profile"),
            (self.policy_extraction, POLICY_EXTRACTIONS, "policy_extraction"),
            (self.task_profile, TASK_PROFILES, "task_profile"),
            (self.adversarial_attack_profile, ADVERSARIAL_ATTACK_PROFILES, "adversarial_attack_profile"),
            (
                self.online_corruption_scale_profile,
                ONLINE_CORRUPTION_SCALE_PROFILES,
                "online_corruption_scale_profile",
            ),
            (self.wsrl_entropy_profile, WSRL_ENTROPY_PROFILES, "wsrl_entropy_profile"),
            (
                self.offline_adversarial_reward_rule,
                OFFLINE_ADVERSARIAL_REWARD_RULES,
                "offline_adversarial_reward_rule",
            ),
            (
                self.online_adversarial_reward_rule,
                ONLINE_ADVERSARIAL_REWARD_RULES,
                "online_adversarial_reward_rule",
            ),
        ):
            if value not in choices:
                raise ValueError(f"Unknown {label} {value!r}; choose from {choices}")
        if (
            self.corruption == "adversarial"
            and self.adversarial_attack_profile == "experimental_sign_pgd"
            and not self.allow_experimental_adversarial_attack
        ):
            raise ValueError(
                "experimental_sign_pgd requires both an explicit "
                "--adversarial-attack-profile experimental_sign_pgd and "
                "--allow-experimental-adversarial-attack"
            )
        if (
            self.adversarial_attack_profile == "rpex_official_adam"
            and self.online_adversarial_reward_rule
            != "official_uniform_replacement"
        ):
            raise ValueError(
                "rpex_official_adam requires official_uniform_replacement for "
                "online adversarial rewards"
            )
        if (
            self.implementation_profile == "official_code_reference"
            and self.corruption_target in ("observations", "actions", "dynamics", "mixed")
            and self.online_corruption_scale_profile != "rpex_official_code"
        ):
            raise ValueError(
                "official_code_reference requires "
                "online_corruption_scale_profile=rpex_official_code"
            )
        if self.calibration_mask_mode not in CALIBRATION_MASK_MODES:
            raise ValueError(
                "calibration_mask_mode must be all, oracle_exclude_corrupted, "
                "or disabled"
            )
        if self.state_normalization not in ("standard", "robust_median_mad", "none"):
            raise ValueError(
                "state_normalization must be standard, robust_median_mad, or none"
            )
        if self.action_distribution not in (
            "tanh_gaussian",
            "legacy_gaussian",
            "official_unsquashed_gaussian",
        ):
            raise ValueError(
                "action_distribution must be tanh_gaussian, legacy_gaussian, "
                "or official_unsquashed_gaussian"
            )
        if self.evaluation_mode not in (
            "deterministic_diagnostic",
            "method_faithful",
            "both",
        ):
            raise ValueError(
                "evaluation_mode must be deterministic_diagnostic, "
                "method_faithful, or both"
            )
        if self.mc_return_source not in (
            "post_corruption",
            "legacy_pre_corruption",
        ):
            raise ValueError(
                "mc_return_source must be post_corruption or legacy_pre_corruption"
            )
        if self.pqe_replay_mode not in ("balanced_density", "uniform"):
            raise ValueError("pqe_replay_mode must be balanced_density or uniform")
        if self.attack_norm != "linf":
            raise ValueError("Only attack_norm=linf is currently implemented")
        if self.corruption == "clean":
            self.corruption_target = "none"
        elif self.corruption_target == "none":
            raise ValueError("random/adversarial corruption requires --corruption-target")
        if len(self.mixed_ratios) != len(INDIVIDUAL_CORRUPTION_TARGETS):
            raise ValueError(
                "mixed_ratios must contain four values ordered as "
                "observations actions rewards dynamics"
            )
        if any(value < 0.0 for value in self.mixed_ratios):
            raise ValueError("mixed_ratios cannot contain negative values")
        if abs(sum(self.mixed_ratios) - 1.0) > 1e-6:
            raise ValueError("mixed_ratios must sum to 1.0")
        if self.checkpoint and self.initialize_from_checkpoint:
            raise ValueError("use only --initialize-from-checkpoint; --checkpoint is deprecated")
        if self.checkpoint:
            self.initialize_from_checkpoint = self.checkpoint
        if self.initialize_from_checkpoint and self.resume_run:
            raise ValueError(
                "--initialize-from-checkpoint and --resume-run have different semantics "
                "and are mutually exclusive"
            )
        if self.stage == "online" and not self.initialize_from_checkpoint and not self.resume_run:
            raise ValueError(
                "--stage online requires --initialize-from-checkpoint or --resume-run"
            )
        for name in ("offline_corruption_rate", "online_corruption_rate"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.offline_ratio is not None and not 0.0 <= self.offline_ratio <= 1.0:
            raise ValueError("offline_ratio must be in [0, 1]")
        if self.balanced_replay_temperature <= 0.0:
            raise ValueError("balanced_replay_temperature must be positive")
        if self.priority_floor <= 0.0:
            raise ValueError("priority_floor must be positive")
        if (
            self.attack_step_size < 0.0
            or self.online_attack_step_size < 0.0
            or self.attack_min_step_size < 0.0
        ):
            raise ValueError("attack step sizes cannot be negative")
        if self.batch_size < 2:
            raise ValueError("batch_size must be at least 2")
        for name in ("offline_steps", "online_steps", "checkpoint_period"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} cannot be negative")
        for name in ("offline_checkpoint_period", "online_checkpoint_period"):
            value = getattr(self, name)
            if value is not None and value < 0:
                raise ValueError(f"{name} cannot be negative")
        if self.keep_last_checkpoints < 0:
            raise ValueError("keep_last_checkpoints cannot be negative")
        for name in (
            "eval_period",
            "eval_episodes",
            "train_log_period",
            "max_episode_steps",
            "replay_size",
            "updates_per_step",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.max_grad_norm is not None and self.max_grad_norm <= 0.0:
            raise ValueError("max_grad_norm must be positive when set")
        for name in (
            "actor_learning_rate",
            "critic_learning_rate",
            "temperature_learning_rate",
        ):
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be positive")
        if self.calql_bc_warmup_steps < 0:
            raise ValueError("calql_bc_warmup_steps cannot be negative")
        if self.cql_temperature <= 0.0:
            raise ValueError("cql_temperature must be positive")
        if self.wsrl_utd_ratio <= 0 or self.wsrl_per_critic_batch_size <= 0:
            raise ValueError("WSRL UTD ratio and per-critic batch size must be positive")
        if self.wsrl_target_critic_subsample_size is not None and not (
            1 <= self.wsrl_target_critic_subsample_size <= self.sac_num_critics
        ):
            raise ValueError(
                "wsrl_target_critic_subsample_size must be between 1 and "
                "sac_num_critics"
            )

    def _resolve_role_seeds(self) -> None:
        offsets = {
            "learner_seed": 0,
            "corruption_seed": 10_001,
            "replay_seed": 20_003,
            "train_env_seed": 30_007,
            "eval_seed": 40_009,
        }
        modulus = 2**31 - 1
        for name, offset in offsets.items():
            if getattr(self, name) is None:
                setattr(self, name, int((self.seed + offset) % modulus))

    def _resolve_implementation_profile(self) -> None:
        primary_suite = self.suite_profile in (
            "method_fidelity",
            "primary_research_benchmark",
        )
        common_budget_suite = self.suite_profile in (
            "common_budget_robustness",
            "common_budget_diagnostic",
        )
        if self.algorithm_profile == "reference":
            raise ValueError(
                "The generic algorithm_profile='reference' was removed because it "
                "mixed paper reproduction, upstream code, and task ports. Use "
                "--implementation-profile and --suite-profile explicitly."
            )
        if (
            self.algorithm_profile is not None
            and self.implementation_profile is not None
            and self.algorithm_profile != self.implementation_profile
        ):
            raise ValueError(
                "algorithm_profile and implementation_profile disagree; use only "
                "implementation_profile"
            )
        if self.implementation_profile is None:
            self.implementation_profile = self.algorithm_profile

        if primary_suite:
            if self.algorithm == "pessimistic_q_ensemble":
                raise ValueError(
                    "primary research fidelity is unavailable for Pessimistic Q-Ensemble: "
                    "the local implementation is pqe_shared_actor_approx, not the "
                    "official N=5 independently pretrained ensemble"
                )
            if self.algorithm == "cal_ql" and self.suite_profile == "method_fidelity":
                raise ValueError(
                    "method_fidelity is unavailable for Cal-QL on D4RL locomotion: "
                    "the official Cal-QL repository supports AntMaze/Adroit; select "
                    "locomotion_port or common_budget_robustness and report task_port"
                )
            expected = (
                "locomotion_port"
                if self.algorithm == "cal_ql"
                else "official_code_reference"
            )
            if self.implementation_profile is None:
                self.implementation_profile = expected
            allowed_profiles = (
                ("locomotion_port",)
                if self.algorithm == "cal_ql"
                else ("official_code_reference", "paper_reference")
            )
            if self.implementation_profile not in allowed_profiles:
                raise ValueError(
                    "primary research suite requires a method-specific official "
                    "profile (Cal-QL locomotion uses locomotion_port)"
                )
            if self.algorithm in ("rpex", "riql_naive", "riql_pex"):
                self.offline_steps = 2_000_001
                self.online_steps = 1_000_001
            elif self.algorithm == "wsrl":
                self.offline_steps = 250_000
                self.online_steps = 500_000
        else:
            if self.implementation_profile is None:
                self.implementation_profile = "common_budget_robustness"
            if self.algorithm == "pessimistic_q_ensemble":
                self.implementation_profile = "experimental_approximation"

        fidelity = {
            "official_code_reference": "exact_upstream_port",
            "paper_reference": "paper_code_conflict",
            "common_budget_robustness": "task_port",
            "locomotion_port": "task_port",
            "legacy_current": "legacy_unknown",
            "experimental_approximation": "approximation",
        }[self.implementation_profile]
        if common_budget_suite and fidelity not in ("approximation", "legacy_unknown"):
            fidelity = "task_port"
        if self.algorithm == "cal_ql" and fidelity == "exact_upstream_port":
            fidelity = "task_port"
        riql_extension = (
            self.algorithm in ("rpex", "riql_naive", "riql_pex")
            and (
                self.corruption not in ("random", "adversarial")
                or self.corruption_target not in INDIVIDUAL_CORRUPTION_TARGETS
            )
        )
        if riql_extension and self.suite_profile == "primary_research_benchmark":
            fidelity = "task_port"
        if self.implementation_fidelity not in (None, fidelity):
            raise ValueError(
                "implementation_fidelity is resolved from implementation_profile; "
                f"expected {fidelity!r}"
            )
        self.implementation_fidelity = fidelity
        self.algorithm_profile = self.implementation_profile

        if self.implementation_profile == "official_code_reference":
            self.action_execution_profile = "official_algorithm_behavior"
            if self.corruption == "adversarial":
                self.adversarial_attack_profile = "rpex_official_adam"
        elif self.implementation_profile == "paper_reference":
            if (
                self.algorithm in ("rpex", "riql_naive", "riql_pex")
                and self.corruption_target == "mixed"
            ):
                raise ValueError(
                    "rpex_paper_mixed is not executable: the pinned public RPEX "
                    "repository contains no mixed-corruption configuration or "
                    "implementation to port without guessing"
                )
            self.online_replay_profile = "paper_offline_online_mixture"
            self.evaluation_policy_profile = "paper_greedy_highest_weight"
            self.attack_timing = "paper_pre_action_sensor_actuator"
            self.random_attack_semantics = "pre_action_sensor_actuator_corruption"
            if self.corruption_target == "mixed":
                self.mixed_corruption_profile = "rpex_paper_mixed"
        if (
            self.algorithm in ("rpex", "riql_naive", "riql_pex")
            and self.implementation_profile not in ("legacy_current",)
        ):
            self.action_distribution = "official_unsquashed_gaussian"
        if self.algorithm == "cal_ql":
            self.task_profile = "d4rl_locomotion_port"
        elif self.task_profile is None:
            self.task_profile = "official_supported_task"
        if self.policy_extraction is None:
            self.policy_extraction = (
                "align_iql"
                if self.algorithm in ("rpex", "riql_naive", "riql_pex")
                and self.corruption_target == "observations"
                else "awr"
            )

        if self.algorithm in ("rpex", "riql_naive", "riql_pex"):
            row_key, row = resolve_riql_reference_row(
                self.env_name,
                self.corruption,
                self.corruption_target,
                allow_extension=(
                    common_budget_suite
                    or self.suite_profile == "primary_research_benchmark"
                ),
            )
            self.riql_config_row = row_key
            self.riql_config_extension = row.extension
            self.riql_sigma = row.sigma
            self.riql_quantile = row.quantile
            self.num_critics = row.num_critics
            self.inv_temperature = row.inverse_temperature
            self.kappa = row.kappa
            if primary_suite:
                self.updates_per_step = row.utd_ratio
                self.actor_learning_rate = row.actor_lr
                self.critic_learning_rate = row.critic_lr
        else:
            self.riql_config_row = None
            self.riql_config_extension = False

    def _resolve_algorithm_profile(self) -> None:
        reference = self.implementation_profile in (
            "official_code_reference",
            "paper_reference",
            "locomotion_port",
            "common_budget_robustness",
        )
        reference_actor_lr = (
            1e-4
            if self.algorithm in ("cal_ql", "wsrl") and reference
            else self.learning_rate
        )
        self.actor_learning_rate = (
            reference_actor_lr
            if self.actor_learning_rate is None
            else self.actor_learning_rate
        )
        self.critic_learning_rate = (
            self.learning_rate
            if self.critic_learning_rate is None
            else self.critic_learning_rate
        )
        reference_temperature_lr = (
            1e-4 if self.algorithm == "wsrl" and reference else self.entropy_lr
        )
        self.temperature_learning_rate = (
            reference_temperature_lr
            if self.temperature_learning_rate is None
            else self.temperature_learning_rate
        )
        requested_warmup = (
            self.calql_bc_warmup_steps
            if self.calql_bc_warmup_steps is not None
            else self.bc_steps
        )
        self.calql_bc_warmup_steps = (
            (0 if reference else 100_000)
            if requested_warmup is None
            else int(requested_warmup)
        )
        self.bc_steps = self.calql_bc_warmup_steps
        if self.cql_max_target_backup is None:
            self.cql_max_target_backup = bool(reference)
        if self.algorithm == "wsrl":
            if self.wsrl_num_critics is None:
                self.wsrl_num_critics = 10 if reference else self.sac_num_critics
            self.sac_num_critics = self.wsrl_num_critics
            if self.wsrl_target_critic_subsample_size is None:
                self.wsrl_target_critic_subsample_size = 2 if reference else self.sac_num_critics
            if self.wsrl_layer_norm is None:
                self.wsrl_layer_norm = reference
            if self.wsrl_utd_ratio is None:
                self.wsrl_utd_ratio = 4 if reference else self.updates_per_step
            action_dim = ACTION_DIMS[self.env_name.split("-", 1)[0]]
            resolved_target_entropy = (
                -float(action_dim)
                if self.wsrl_entropy_profile
                == "official_negative_action_dim"
                else 0.0
            )
            if self.target_entropy is not None and not math.isclose(
                float(self.target_entropy), resolved_target_entropy
            ):
                raise ValueError(
                    "target_entropy is resolved by wsrl_entropy_profile; "
                    f"expected {resolved_target_entropy}"
                )
            self.target_entropy = resolved_target_entropy
        else:
            self.wsrl_num_critics = self.sac_num_critics
            if self.wsrl_target_critic_subsample_size is None:
                self.wsrl_target_critic_subsample_size = self.sac_num_critics
            if self.wsrl_layer_norm is None:
                self.wsrl_layer_norm = False
            if self.wsrl_utd_ratio is None:
                self.wsrl_utd_ratio = self.updates_per_step
            if self.target_entropy is None:
                self.target_entropy = -float(
                    ACTION_DIMS[self.env_name.split("-", 1)[0]]
                )
        if self.algorithm not in ("rpex", "riql_pex"):
            self.evaluation_policy_profile = "deterministic_diagnostic"
        if self.algorithm in ("pex", "cal_ql"):
            self.online_replay_profile = "fixed_offline_online_mixture"
        elif self.algorithm == "pessimistic_q_ensemble":
            self.online_replay_profile = (
                "balanced_density_replay"
                if self.pqe_replay_mode == "balanced_density"
                else "fixed_offline_online_mixture"
            )

    @property
    def resolved_algorithm_profile(self) -> str:
        if self.algorithm == "cal_ql":
            base = (
                "calql_locomotion_port"
                if self.implementation_profile != "legacy_current"
                else "calql_legacy_bc100k"
            )
            if self.calibration_mask_mode == "oracle_exclude_corrupted":
                return f"{base}_oracle"
            if self.calibration_mask_mode == "disabled":
                return f"{base}_calibration_disabled"
            return base
        if self.algorithm == "wsrl":
            return (
                "wsrl_official_locomotion_redq10x2"
                if self.implementation_profile != "legacy_current"
                else "wsrl_legacy_min10"
            )
        if self.algorithm == "pessimistic_q_ensemble":
            return "pqe_shared_actor_approx"
        return f"{self.algorithm}_{self.implementation_profile}"

    @property
    def effective_offline_checkpoint_period(self) -> int:
        if self.offline_checkpoint_period is not None:
            return self.offline_checkpoint_period
        return self.checkpoint_period

    @property
    def effective_online_checkpoint_period(self) -> int:
        if self.online_checkpoint_period is not None:
            return self.online_checkpoint_period
        return self.checkpoint_period

    @property
    def effective_offline_ratio(self) -> float:
        if self.offline_ratio is not None:
            return self.offline_ratio
        if (
            self.algorithm in ("rpex", "riql_pex")
            and self.online_replay_profile == "paper_offline_online_mixture"
        ):
            return 0.5
        # Mirrors RPEX: robust IQL variants reduce online replay as an offline
        # problem; PEX and Cal-QL keep balanced offline samples.
        return {
            "rpex": 0.0,
            "riql_pex": 0.0,
            "riql_naive": 0.0,
            "uwmsg": 0.0,
            "pex": 0.5,
            "cal_ql": 0.5,
            "wsrl": 0.0,
            "ro2o": 0.0,
            "pessimistic_q_ensemble": 0.5,
        }[self.algorithm]

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result["effective_offline_ratio"] = self.effective_offline_ratio
        result[
            "effective_offline_checkpoint_period"
        ] = self.effective_offline_checkpoint_period
        result[
            "effective_online_checkpoint_period"
        ] = self.effective_online_checkpoint_period
        result["paper_title"] = ALGORITHM_TITLES[self.algorithm]
        result["base_seed"] = self.seed
        result["resolved_algorithm_profile"] = self.resolved_algorithm_profile
        result["implementation_profile"] = self.implementation_profile
        result["implementation_fidelity"] = self.implementation_fidelity
        result["suite_profile"] = self.suite_profile
        result["budget_profile"] = self.suite_profile
        result["offline_update_budget"] = self.offline_steps
        result["online_environment_step_budget"] = self.online_steps
        result["utd_ratio"] = (
            self.wsrl_utd_ratio if self.algorithm == "wsrl" else self.updates_per_step
        )
        result["not_paper_reproduction"] = (
            self.suite_profile
            in ("common_budget_robustness", "common_budget_diagnostic")
            or self.implementation_fidelity in ("task_port", "approximation")
        )
        result["paper_reproduction_eligible"] = (
            self.implementation_fidelity == "exact_upstream_port"
            and self.suite_profile
            in ("method_fidelity", "primary_research_benchmark")
        )
        result["oracle_information"] = (
            self.calibration_mask_mode == "oracle_exclude_corrupted"
        )
        result["upstream_commit"] = UPSTREAM_COMMITS.get(self.algorithm)
        result["score_semantics"] = (
            "d4rl_normalized_return"
            if self.protocol == LEGACY_PROTOCOL
            else "diagnostic_d4rl_reference_scaled_return"
        )
        if self.algorithm == "wsrl":
            result.update(
                {
                    "offline_pretrainer": "cql_redq",
                    "offline_stage_label": "CQL-REDQ pretrainer for WSRL",
                    "parameters_frozen_during_warmup": True,
                    "offline_data_retained_online": False,
                    "target_entropy": self.target_entropy,
                    "temperature_parameterization": (
                        "softplus_lagrange"
                        if self.wsrl_entropy_profile
                        == "official_negative_action_dim"
                        else "legacy_exponential_log_alpha"
                    ),
                    "wsrl_total_sampled_batch_size": self.wsrl_utd_ratio
                    * self.wsrl_per_critic_batch_size,
                }
            )
        if self.algorithm == "cal_ql":
            result["calql_actor_update_mode_at_start"] = (
                "bc_warmup" if self.calql_bc_warmup_steps > 0 else "sac"
            )
        return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Unified RPEX-benchmark corruption-robust offline-to-online RL"
    )
    parser.add_argument("--algorithm", required=True, help=", ".join(ALGORITHMS))
    parser.add_argument("--env-name", required=True)
    parser.add_argument("--corruption", choices=CORRUPTION_MODES, default="clean")
    parser.add_argument("--corruption-target", choices=CORRUPTION_TARGETS, default="none")
    parser.add_argument("--stage", choices=("offline", "online", "both"), default="both")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--learner-seed", type=int)
    parser.add_argument("--corruption-seed", type=int)
    parser.add_argument("--replay-seed", type=int)
    parser.add_argument("--train-env-seed", type=int)
    parser.add_argument("--eval-seed", type=int)
    parser.add_argument(
        "--implementation-profile",
        "--algorithm-profile",
        dest="implementation_profile",
        choices=IMPLEMENTATION_PROFILES,
        help="method-specific implementation provenance (generic reference was removed)",
    )
    parser.add_argument(
        "--suite-profile",
        choices=SUITE_PROFILES,
        default="common_budget_robustness",
    )
    parser.add_argument("--run-purpose", choices=RUN_PURPOSES, default="diagnostic")
    parser.add_argument(
        "--protocol",
        choices=PROTOCOLS,
        default=DEFAULT_PROTOCOL,
        help=(
            "rpex_d4rl_v2_legacy for exact reproduction, or "
            "local_gymnasium_v4_diagnostic for a non-benchmark local runtime"
        ),
    )
    parser.add_argument(
        "--allow-diagnostic-protocol",
        action="store_true",
        help="required acknowledgement for the non-benchmark Gymnasium protocol",
    )
    parser.add_argument(
        "--allow-legacy-checkpoint-without-fingerprint",
        action="store_true",
    )
    parser.add_argument("--output-dir", default="results")
    parser.add_argument(
        "--dataset-dir",
        help=(
            "D4RL dataset directory; passed to pinned D4RL in legacy mode and "
            "read directly in local Gymnasium mode"
        ),
    )
    parser.add_argument("--checkpoint", help=argparse.SUPPRESS)
    parser.add_argument(
        "--initialize-from-checkpoint",
        help="initialize model/normalizer for a new run; does not restore run position",
    )
    parser.add_argument(
        "--resume-run",
        help="resume an interrupted run from its full run-state checkpoint",
    )
    parser.add_argument("--attack-checkpoint")
    parser.add_argument(
        "--attack-checkpoint-sha256",
        help="required expected SHA256 for a custom official attacker checkpoint",
    )
    parser.add_argument(
        "--comparison-name",
        help="comparison group ID; generated automatically for a standalone run",
    )
    parser.add_argument("--device", default="auto", help="auto, cpu, mps, cuda, or cuda:N")
    parser.add_argument("--cuda-device", type=int, default=0)

    parser.add_argument("--offline-steps", type=int, default=500_000)
    parser.add_argument("--online-steps", type=int, default=500_000)
    parser.add_argument("--initial-collection-steps", type=int, default=5_000)
    parser.add_argument("--warmup-steps", type=int, default=5_000)
    parser.add_argument("--updates-per-step", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--replay-size", type=int, default=1_000_000)
    parser.add_argument("--eval-period", type=int, default=10_000)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument(
        "--checkpoint-period",
        type=int,
        default=100_000,
        help="shared periodic checkpoint interval; 0 disables periodic checkpoints",
    )
    parser.add_argument(
        "--offline-checkpoint-period",
        type=int,
        help="override --checkpoint-period for offline updates; 0 disables it",
    )
    parser.add_argument(
        "--online-checkpoint-period",
        type=int,
        help="override --checkpoint-period for online environment steps; 0 disables it",
    )
    parser.add_argument(
        "--keep-last-checkpoints",
        type=int,
        default=5,
        help="periodic checkpoints retained per phase; 0 keeps all",
    )
    parser.add_argument("--train-log-period", type=int, default=1_000)
    parser.add_argument("--max-episode-steps", type=int, default=1_000)

    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--hidden-layers", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--actor-learning-rate", type=float)
    parser.add_argument("--critic-learning-rate", type=float)
    parser.add_argument("--temperature-learning-rate", type=float)
    parser.add_argument("--max-grad-norm", type=float)
    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument("--target-update-rate", type=float, default=0.005)
    parser.add_argument("--no-normalize-states", dest="normalize_states", action="store_false")
    parser.add_argument(
        "--state-normalization",
        choices=("standard", "robust_median_mad", "none"),
        default="standard",
    )
    parser.add_argument("--deterministic-policy", action="store_true")
    parser.add_argument(
        "--action-distribution",
        choices=(
            "tanh_gaussian",
            "legacy_gaussian",
            "official_unsquashed_gaussian",
        ),
        default="tanh_gaussian",
        help="bounded default, or unsafe reproduction-only PEX/RPEX Gaussian",
    )
    parser.add_argument(
        "--evaluation-mode",
        choices=("deterministic_diagnostic", "method_faithful", "both"),
    )
    parser.add_argument(
        "--online-replay-profile", choices=ONLINE_REPLAY_PROFILES,
        default="official_code_online_only",
    )
    parser.add_argument(
        "--evaluation-policy-profile", choices=EVALUATION_POLICY_PROFILES,
        default="official_code_epsilon_switching",
    )
    parser.add_argument(
        "--attack-timing", choices=ATTACK_TIMINGS,
        default="official_code_post_transition_replay_poisoning",
    )
    parser.add_argument(
        "--random-attack-semantics", choices=RANDOM_ATTACK_SEMANTICS,
        default="post_transition_replay_poisoning",
    )
    parser.add_argument(
        "--mixed-corruption-profile", choices=MIXED_CORRUPTION_PROFILES,
        default="generic_partitioned_mixed",
    )
    parser.add_argument(
        "--action-execution-profile", choices=ACTION_EXECUTION_PROFILES,
        default="clip_to_action_space",
    )
    parser.add_argument("--policy-extraction", choices=POLICY_EXTRACTIONS)
    parser.add_argument("--task-profile", choices=TASK_PROFILES)
    parser.add_argument(
        "--adversarial-attack-profile", choices=ADVERSARIAL_ATTACK_PROFILES,
    )
    parser.add_argument(
        "--allow-experimental-adversarial-attack", action="store_true"
    )
    parser.add_argument(
        "--online-corruption-scale-profile",
        choices=ONLINE_CORRUPTION_SCALE_PROFILES,
    )
    parser.add_argument(
        "--offline-adversarial-reward-rule",
        choices=OFFLINE_ADVERSARIAL_REWARD_RULES,
    )
    parser.add_argument(
        "--online-adversarial-reward-rule",
        choices=ONLINE_ADVERSARIAL_REWARD_RULES,
    )

    parser.add_argument("--expectile", type=float, default=0.7)
    parser.add_argument("--beta", type=float, default=3.0)
    parser.add_argument("--riql-sigma", type=float, default=3.0)
    parser.add_argument("--riql-quantile", type=float, default=0.1)
    parser.add_argument("--num-critics", type=int, default=5)
    parser.add_argument("--inv-temperature", type=float, default=3.0)
    parser.add_argument("--kappa", type=float, default=0.1)

    parser.add_argument("--sac-num-critics", type=int, default=10)
    parser.add_argument("--lcb-ratio", type=float, default=4.0)
    parser.add_argument("--uncertainty-ratio", type=float, default=0.7)
    parser.add_argument("--uncertainty-basic", type=float, default=0.0)
    parser.add_argument("--uncertainty-min", type=float, default=1.0)
    parser.add_argument("--uncertainty-max", type=float, default=10.0)
    parser.add_argument("--entropy-lr", type=float, default=3e-4)

    parser.add_argument("--cql-alpha", type=float, default=5.0)
    parser.add_argument("--cql-alpha-online", type=float, default=1.0)
    parser.add_argument("--cql-n-actions", type=int, default=10)
    parser.add_argument("--cql-temperature", type=float, default=1.0)
    parser.add_argument("--bc-steps", type=int, help="deprecated alias for Cal-QL BC warmup")
    parser.add_argument("--calql-bc-warmup-steps", type=int)
    parser.add_argument("--backup-entropy", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--cql-max-target-backup", action=argparse.BooleanOptionalAction
    )
    parser.add_argument(
        "--calibration-mask-mode", choices=CALIBRATION_MASK_MODES, default="all"
    )
    parser.add_argument("--wsrl-num-critics", type=int)
    parser.add_argument("--wsrl-target-critic-subsample-size", type=int)
    parser.add_argument("--wsrl-layer-norm", action=argparse.BooleanOptionalAction)
    parser.add_argument("--wsrl-utd-ratio", type=int)
    parser.add_argument("--wsrl-per-critic-batch-size", type=int, default=256)
    parser.add_argument(
        "--wsrl-entropy-profile",
        choices=WSRL_ENTROPY_PROFILES,
        default="official_negative_action_dim",
    )
    parser.add_argument("--target-entropy", type=float)
    parser.add_argument(
        "--mc-return-source",
        choices=("post_corruption", "legacy_pre_corruption"),
        default="post_corruption",
    )
    parser.add_argument(
        "--pqe-replay-mode",
        choices=("balanced_density", "uniform"),
        default="balanced_density",
    )
    parser.add_argument("--balanced-replay-temperature", type=float, default=5.0)
    parser.add_argument("--priority-floor", type=float, default=1e-3)
    parser.add_argument("--pqe-member-checkpoints", nargs="*", default=())

    parser.add_argument("--ro2o-beta-policy", type=float, default=1.0)
    parser.add_argument("--ro2o-beta-ood", type=float, default=0.1)
    parser.add_argument("--ro2o-q-smooth-eps", type=float, default=0.01)
    parser.add_argument("--ro2o-policy-smooth-eps", type=float, default=0.01)
    parser.add_argument("--ro2o-ood-smooth-eps", type=float, default=0.01)
    parser.add_argument("--ro2o-sample-size", type=int, default=20)
    parser.add_argument("--ro2o-uncertainty", type=float, default=1.0)
    parser.add_argument("--ro2o-uncertainty-min", type=float, default=0.1)
    parser.add_argument("--ro2o-uncertainty-decay", type=float, default=5e-7)

    parser.add_argument("--corruption-range", type=float, default=1.0)
    parser.add_argument("--offline-corruption-rate", type=float, default=0.3)
    parser.add_argument("--online-corruption-rate", type=float, default=0.5)
    parser.add_argument("--offline-attack-steps", type=int, default=100)
    parser.add_argument("--online-attack-steps", type=int, default=2)
    parser.add_argument("--attack-step-size", type=float, default=0.01)
    parser.add_argument("--online-attack-step-size", type=float)
    parser.add_argument("--attack-min-step-size", type=float, default=0.0)
    parser.add_argument("--attack-norm", choices=("linf",), default="linf")
    parser.add_argument("--force-regenerate-attack", action="store_true")
    parser.add_argument(
        "--mixed-ratios",
        type=float,
        nargs=4,
        metavar=("OBS", "ACT", "REW", "DYN"),
        default=(0.25, 0.25, 0.25, 0.25),
        help=(
            "target allocation for --corruption-target mixed, ordered as "
            "observations actions rewards dynamics; must sum to 1"
        ),
    )
    parser.add_argument("--offline-ratio", type=float)
    return parser


def config_from_args(args: argparse.Namespace) -> ExperimentConfig:
    if getattr(args, "bc_steps", None) is not None:
        import warnings

        warnings.warn(
            "--bc-steps is deprecated; use --calql-bc-warmup-steps",
            DeprecationWarning,
            stacklevel=2,
        )
    config = ExperimentConfig(**vars(args))
    if config.protocol == LOCAL_PROTOCOL and not config.allow_diagnostic_protocol:
        raise ValueError(
            "The local Gymnasium protocol is diagnostic-only. Re-run with "
            "--allow-diagnostic-protocol to acknowledge that it is not a "
            "strict D4RL benchmark result."
        )
    return config


def default_attack_checkpoint(env_name: str) -> Optional[Path]:
    root = Path(__file__).resolve().parents[2]
    candidate = (
        root
        / "RIQL-main"
        / "pretrained_model"
        / "EDAC"
        / f"EDAC_baseline_seed0-{env_name}"
        / "2999.pt"
    )
    return candidate if candidate.exists() else None


DEFAULT_ATTACK_CHECKPOINT_SHA256 = {
    "halfcheetah-medium-replay-v2": (
        "7334ff2b658e95423ca520e258a15ebb6d32ac044cbb3bde27b352bb41e59beb"
    ),
    "hopper-medium-replay-v2": (
        "f5c558003cfd3814c4ea6cff4ce5319b61a8e3dc9013cf208c29e37e368680bd"
    ),
    "walker2d-medium-replay-v2": (
        "8028210fde88f2950115613b37df991358f6cb322f6ba06911662c99cd6d5852"
    ),
}


def default_attack_checkpoint_sha256(env_name: str) -> Optional[str]:
    return DEFAULT_ATTACK_CHECKPOINT_SHA256.get(env_name)
