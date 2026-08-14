from __future__ import annotations

import torch

from ..config import ExperimentConfig
from .base import BaseAgent
from .calql import CalQLAgent
from .iql_family import IQLFamilyAgent
from .sac_family import SACEnsembleAgent


def _apply_rpex_riql_defaults(config: ExperimentConfig) -> None:
    if config.algorithm not in ("rpex", "riql_pex", "riql_naive"):
        return
    # Explicit non-default CLI values win.
    if (config.riql_sigma, config.riql_quantile, config.num_critics) != (3.0, 0.1, 5):
        return
    domain = config.env_name.split("-")[0]
    target = config.corruption_target
    mode = config.corruption if config.corruption != "clean" else "random"
    tables = {
        "random": {
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
        },
        "adversarial": {
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
        },
    }
    # RPEX reports target-specific RIQL hyperparameters for the four
    # single-target settings. Mixed corruption keeps the user-supplied/general
    # defaults because there is no single target-specific row to select.
    if target in ("none", "mixed"):
        return
    config.riql_sigma, config.riql_quantile, config.num_critics = tables[mode][target][
        domain
    ]


def _apply_uwmsg_defaults(config: ExperimentConfig) -> None:
    if config.algorithm != "uwmsg":
        return
    domain = config.env_name.split("-")[0]
    target = config.corruption_target
    lcb = {
        "halfcheetah": 4.0,
        "walker2d": 6.0 if config.corruption == "random" else 4.0,
        "hopper": 6.0,
    }
    uncertainty = {
        ("halfcheetah", "rewards"): 0.7,
        ("halfcheetah", "dynamics"): 0.5 if config.corruption == "random" else 0.2,
        ("walker2d", "rewards"): 0.3 if config.corruption == "random" else 0.5,
        ("walker2d", "dynamics"): 0.5,
        ("hopper", "rewards"): 0.7,
        ("hopper", "dynamics"): 0.7 if config.corruption == "random" else 1.0,
    }
    if config.lcb_ratio == 4.0:
        config.lcb_ratio = lcb[domain]
    if config.uncertainty_ratio == 0.7:
        config.uncertainty_ratio = uncertainty.get((domain, target), 0.7)


def build_agent(
    config: ExperimentConfig,
    state_dim: int,
    action_dim: int,
    max_action: float,
    device: torch.device,
) -> BaseAgent:
    _apply_rpex_riql_defaults(config)
    _apply_uwmsg_defaults(config)
    if config.algorithm == "pessimistic_q_ensemble" and config.sac_num_critics == 10:
        config.sac_num_critics = 5
    if config.algorithm in ("rpex", "riql_pex", "riql_naive", "pex"):
        return IQLFamilyAgent(config, state_dim, action_dim, max_action, device)
    if config.algorithm == "cal_ql":
        return CalQLAgent(config, state_dim, action_dim, max_action, device)
    if config.algorithm in (
        "uwmsg",
        "wsrl",
        "ro2o",
        "pessimistic_q_ensemble",
    ):
        return SACEnsembleAgent(config, state_dim, action_dim, max_action, device)
    raise ValueError(f"Unsupported algorithm: {config.algorithm}")
