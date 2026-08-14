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
LOCAL_PROTOCOL = "local_gymnasium_v4"
DEFAULT_PROTOCOL = LEGACY_PROTOCOL
PROTOCOLS = (LEGACY_PROTOCOL, LOCAL_PROTOCOL)

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

    output_dir: str = "results"
    dataset_dir: Optional[str] = None
    checkpoint: Optional[str] = None
    attack_checkpoint: Optional[str] = None
    comparison_name: Optional[str] = None
    device: str = "auto"
    cuda_device: int = 0

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
    bc_steps: int = 100_000
    backup_entropy: bool = False
    mc_return_source: str = "post_corruption"

    # Off2OnRL balanced replay. ``uniform`` is an explicit ablation.
    pqe_replay_mode: str = "balanced_density"
    balanced_replay_temperature: float = 5.0
    priority_floor: float = 1e-3

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
        self.state_normalization = self.state_normalization.lower()
        self.action_distribution = self.action_distribution.lower()
        self.evaluation_mode = self.evaluation_mode.lower()
        self.mc_return_source = self.mc_return_source.lower()
        self.pqe_replay_mode = self.pqe_replay_mode.lower()
        self.attack_norm = self.attack_norm.lower()
        self.mixed_ratios = tuple(float(value) for value in self.mixed_ratios)

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
        if self.protocol not in PROTOCOLS:
            raise ValueError(
                f"Unknown protocol {self.protocol!r}; choose from {PROTOCOLS}"
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
    parser.add_argument(
        "--protocol",
        choices=PROTOCOLS,
        default=DEFAULT_PROTOCOL,
        help=(
            "rpex_d4rl_v2_legacy for exact reproduction, or "
            "local_gymnasium_v4 for the modern local macOS runtime"
        ),
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
    parser.add_argument("--bc-steps", type=int, default=100_000)
    parser.add_argument("--backup-entropy", action=argparse.BooleanOptionalAction, default=False)
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
    return ExperimentConfig(**vars(args))


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
