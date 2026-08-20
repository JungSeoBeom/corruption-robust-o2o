from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


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
ALGORITHM_PROFILES = ("reference", "legacy_current")
CALIBRATION_MASK_MODES = ("all", "oracle_exclude_corrupted", "disabled")

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
    algorithm_profile: str = "reference"
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
    attack_checkpoint: Optional[str] = None
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
    evaluation_mode: str = "deterministic_diagnostic"

    # IQL / RIQL / PEX / RPEX
    expectile: float = 0.7
    beta: float = 3.0
    riql_sigma: float = 3.0
    riql_quantile: float = 0.1
    num_critics: int = 5
    inv_temperature: float = 3.0
    kappa: float = 0.1

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

    # Off2OnRL balanced replay. ``uniform`` is an explicit ablation.
    pqe_replay_mode: str = "balanced_density"
    balanced_replay_temperature: float = 5.0
    priority_floor: float = 1e-3
    implementation_variant: Optional[str] = None

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
        self.algorithm_profile = self.algorithm_profile.lower()
        self.state_normalization = self.state_normalization.lower()
        self.action_distribution = self.action_distribution.lower()
        self.evaluation_mode = self.evaluation_mode.lower()
        self.mc_return_source = self.mc_return_source.lower()
        self.calibration_mask_mode = self.calibration_mask_mode.lower()
        self.pqe_replay_mode = self.pqe_replay_mode.lower()
        self.implementation_variant = (
            "shared_actor_approx"
            if self.algorithm == "pessimistic_q_ensemble"
            else None
        )
        self.attack_norm = self.attack_norm.lower()
        self.mixed_ratios = tuple(float(value) for value in self.mixed_ratios)

        self._resolve_role_seeds()
        self._resolve_algorithm_profile()

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
        if self.algorithm_profile not in ALGORITHM_PROFILES:
            raise ValueError(
                f"Unknown algorithm_profile {self.algorithm_profile!r}; "
                f"choose from {ALGORITHM_PROFILES}"
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
        if self.action_distribution not in ("tanh_gaussian", "legacy_gaussian"):
            raise ValueError(
                "action_distribution must be tanh_gaussian or legacy_gaussian"
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
        if self.stage == "online" and not self.checkpoint:
            raise ValueError("--stage online requires --checkpoint")
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
        if self.attack_step_size < 0.0 or self.attack_min_step_size < 0.0:
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

    def _resolve_algorithm_profile(self) -> None:
        reference = self.algorithm_profile == "reference"
        self.actor_learning_rate = (
            (1e-4 if self.algorithm == "cal_ql" and reference else self.learning_rate)
            if self.actor_learning_rate is None
            else self.actor_learning_rate
        )
        self.critic_learning_rate = (
            self.learning_rate
            if self.critic_learning_rate is None
            else self.critic_learning_rate
        )
        self.temperature_learning_rate = (
            self.entropy_lr
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
        else:
            self.wsrl_num_critics = self.sac_num_critics
            if self.wsrl_target_critic_subsample_size is None:
                self.wsrl_target_critic_subsample_size = self.sac_num_critics
            if self.wsrl_layer_norm is None:
                self.wsrl_layer_norm = False
            if self.wsrl_utd_ratio is None:
                self.wsrl_utd_ratio = self.updates_per_step

    @property
    def resolved_algorithm_profile(self) -> str:
        if self.algorithm == "cal_ql":
            base = (
                "calql_reference"
                if self.algorithm_profile == "reference"
                else "calql_legacy_bc100k"
            )
            if self.calibration_mask_mode == "oracle_exclude_corrupted":
                return f"{base}_oracle"
            if self.calibration_mask_mode == "disabled":
                return f"{base}_calibration_disabled"
            return base
        if self.algorithm == "wsrl":
            return (
                "wsrl_reference_redq10x2"
                if self.algorithm_profile == "reference"
                else "wsrl_legacy_min10"
            )
        return f"{self.algorithm}_{self.algorithm_profile}"

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
        "--algorithm-profile", choices=ALGORITHM_PROFILES, default="reference"
    )
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
    parser.add_argument("--checkpoint")
    parser.add_argument("--attack-checkpoint")
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
        choices=("tanh_gaussian", "legacy_gaussian"),
        default="tanh_gaussian",
        help="bounded default, or unsafe reproduction-only PEX/RPEX Gaussian",
    )
    parser.add_argument(
        "--evaluation-mode",
        choices=("deterministic_diagnostic", "method_faithful", "both"),
        default="deterministic_diagnostic",
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
