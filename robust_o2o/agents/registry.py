from __future__ import annotations

import torch

from ..config import ExperimentConfig, CANDIDATE_ALGORITHMS
from .cro2o import CRO2OAgent
from .base import BaseAgent
from .calql import CalQLAgent
from .iql_family import IQLFamilyAgent
from .pessimistic_q_ensemble import PessimisticQEnsembleAgent
from .sac_family import SACEnsembleAgent


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
    # RIQL/RPEX table selection is resolved once by ExperimentConfig so CLI
    # overrides and the serialized provenance cannot diverge at construction.
    _apply_uwmsg_defaults(config)
    if config.algorithm in CANDIDATE_ALGORITHMS:
        return CRO2OAgent(config, state_dim, action_dim, max_action, device)
    if config.algorithm in ("rpex", "riql_pex", "riql_naive", "pex"):
        return IQLFamilyAgent(config, state_dim, action_dim, max_action, device)
    if config.algorithm == "cal_ql":
        return CalQLAgent(config, state_dim, action_dim, max_action, device)
    if config.algorithm == "pessimistic_q_ensemble":
        return PessimisticQEnsembleAgent(
            config, state_dim, action_dim, max_action, device
        )
    if config.algorithm in (
        "uwmsg",
        "wsrl",
        "ro2o",
    ):
        return SACEnsembleAgent(config, state_dim, action_dim, max_action, device)
    raise ValueError(f"Unsupported algorithm: {config.algorithm}")
