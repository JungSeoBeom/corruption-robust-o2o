#!/usr/bin/env python3
"""Lightweight preflight for the practical custom research benchmark.

This checker deliberately validates executable benchmark semantics, not
upstream RNG/parity certificates or official paper budgets.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from robust_o2o.config import (  # noqa: E402
    BENCHMARK_ENVS,
    DEFAULT_PROTOCOL,
    PROTOCOLS,
    ExperimentConfig,
)
from robust_o2o.fidelity import (  # noqa: E402
    BASELINE_REPRODUCTION_REGISTRY,
    MAIN_BASELINES,
    OPTIONAL_ADAPTED_BASELINES,
    OPTIONAL_APPROXIMATION_BASELINES,
    OPTIONAL_BASELINES,
)
from robust_o2o.paths import comparison_directory  # noqa: E402


CORRUPTION_SUITES = ("clean", "random", "adversarial", "all")
RANDOM_TARGETS = ("observations", "actions", "rewards", "dynamics")


def _csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


@dataclass(frozen=True)
class Check:
    label: str
    ok: bool
    detail: str


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Check practical research-benchmark readiness without parity, "
            "exact RNG, certificate, official-budget, or fixed-seed gates"
        )
    )
    parser.add_argument("--env-name", choices=BENCHMARK_ENVS, required=True)
    parser.add_argument(
        "--corruption-suite", choices=CORRUPTION_SUITES, default="random"
    )
    parser.add_argument("--algorithms", type=_csv, default=list(MAIN_BASELINES))
    parser.add_argument("--optional-baselines", type=_csv, default=[])
    parser.add_argument("--protocol", choices=PROTOCOLS, default=DEFAULT_PROTOCOL)
    parser.add_argument(
        "--allow-diagnostic-protocol",
        action="store_true",
        help="acknowledge local_gymnasium_v4_diagnostic when selected",
    )
    parser.add_argument("--dataset-dir")
    parser.add_argument("--attack-checkpoint")
    parser.add_argument("--attack-checkpoint-sha256")
    parser.add_argument("--output-root", default="results")
    parser.add_argument("--experiment-name")
    parser.add_argument("--offline-steps", type=int, default=500_000)
    parser.add_argument("--online-steps", type=int, default=500_000)
    parser.add_argument("--evaluation-interval", type=int, default=10_000)
    parser.add_argument("--evaluation-episodes", type=int, default=10)
    parser.add_argument("--final-window-size", type=int, default=3)
    parser.add_argument(
        "--static-only",
        action="store_true",
        help=(
            "skip D4RL/MuJoCo loading for code inspection; a required runtime "
            "check is then reported as NOT CHECKED and readiness is not granted"
        ),
    )
    return parser


def _settings_for_suite(suite: str) -> tuple[tuple[str, str], ...]:
    clean = (("clean", "none"),)
    random = tuple(("random", target) for target in RANDOM_TARGETS)
    if suite == "clean":
        return clean
    if suite == "random":
        return (*clean, *random)
    if suite == "adversarial":
        from robust_o2o.corruption import SUPPORTED_ADVERSARIAL_TARGETS

        return tuple(
            ("adversarial", target) for target in SUPPORTED_ADVERSARIAL_TARGETS
        )
    if suite == "all":
        from robust_o2o.corruption import SUPPORTED_ADVERSARIAL_TARGETS

        return (
            *clean,
            *random,
            *tuple(
                ("adversarial", target)
                for target in SUPPORTED_ADVERSARIAL_TARGETS
            ),
        )
    raise ValueError(f"unknown corruption suite {suite!r}")


def _config(
    args: argparse.Namespace,
    algorithm: str,
    corruption: str = "clean",
    target: str = "none",
) -> ExperimentConfig:
    return ExperimentConfig(
        algorithm=algorithm,
        env_name=args.env_name,
        corruption=corruption,
        corruption_target=target,
        protocol=args.protocol,
        run_purpose="research_benchmark",
        suite_profile="research_benchmark",
        implementation_profile="research_benchmark",
        offline_steps=args.offline_steps,
        online_steps=args.online_steps,
        eval_period=args.evaluation_interval,
        eval_episodes=args.evaluation_episodes,
        final_window_size=args.final_window_size,
        attack_checkpoint=args.attack_checkpoint,
        attack_checkpoint_sha256=args.attack_checkpoint_sha256,
        allow_diagnostic_protocol=args.allow_diagnostic_protocol,
    )


def run_checks(
    args: argparse.Namespace,
    *,
    environment_loader: Callable[..., object] | None = None,
) -> list[Check]:
    checks: list[Check] = []

    requested_main = tuple(args.algorithms)
    checks.append(
        Check(
            "MAIN BASELINES",
            requested_main == MAIN_BASELINES,
            f"declared={','.join(MAIN_BASELINES)} requested={','.join(requested_main)}",
        )
    )

    optional_registry_ok = all(
        name not in MAIN_BASELINES
        and name in BASELINE_REPRODUCTION_REGISTRY
        and not BASELINE_REPRODUCTION_REGISTRY[name].main_table_eligible
        for name in OPTIONAL_BASELINES
    )
    optional_request_ok = set(args.optional_baselines).issubset(OPTIONAL_BASELINES)
    checks.append(
        Check(
            "OPTIONAL BASELINES EXCLUDED FROM MAIN",
            optional_registry_ok and optional_request_ok,
            "adapted="
            + ",".join(OPTIONAL_ADAPTED_BASELINES)
            + " approximation="
            + ",".join(OPTIONAL_APPROXIMATION_BASELINES),
        )
    )

    configs: list[ExperimentConfig] = []
    config_error: str | None = None
    try:
        for algorithm in (*requested_main, *tuple(args.optional_baselines)):
            configs.append(_config(args, algorithm))
    except (TypeError, ValueError) as exc:
        config_error = str(exc)
    checks.append(
        Check(
            "RESEARCH CONFIG",
            config_error is None,
            config_error or "custom budgets/seeds accepted without parity gates",
        )
    )

    try:
        wsrl = _config(args, "wsrl")
        wsrl_ok = (
            wsrl.warmup_steps == 5_000
            and wsrl.wsrl_num_critics == 10
            and wsrl.wsrl_target_critic_subsample_size == 2
            and wsrl.wsrl_utd_ratio == 4
            and wsrl.effective_offline_ratio == 0.0
            and wsrl.to_dict().get("offline_pretrainer") == "cql_redq"
            and wsrl.to_dict().get("offline_data_retained_online") is False
        )
        wsrl_detail = (
            f"warmup={wsrl.warmup_steps} critics={wsrl.wsrl_num_critics} "
            f"target_subset={wsrl.wsrl_target_critic_subsample_size} "
            f"utd={wsrl.wsrl_utd_ratio} offline_ratio={wsrl.effective_offline_ratio:g}"
        )
    except (TypeError, ValueError) as exc:
        wsrl_ok = False
        wsrl_detail = str(exc)
    checks.append(Check("WSRL PROTOCOL", wsrl_ok, wsrl_detail))

    labels_ok = bool(configs) and all(
        config.calibration_mask_mode != "oracle_exclude_corrupted"
        and config.to_dict().get("uses_corruption_labels") is False
        for config in configs
    )
    try:
        _config(args, "cal_ql_locomotion_adaptation").__class__(
            algorithm="cal_ql_locomotion_adaptation",
            env_name=args.env_name,
            run_purpose="research_benchmark",
            suite_profile="research_benchmark",
            implementation_profile="research_benchmark",
            calibration_mask_mode="oracle_exclude_corrupted",
        )
    except ValueError:
        oracle_rejected = True
    else:
        oracle_rejected = False
    checks.append(
        Check(
            "CORRUPTION LABEL ISOLATION",
            labels_ok and oracle_rejected,
            "learner metadata uses_corruption_labels=false; oracle mode rejected",
        )
    )

    try:
        from robust_o2o.corruption import (
            SUPPORTED_ADVERSARIAL_TARGETS,
            resolve_attack_checkpoint,
            validate_adversarial_target,
        )

        supported = tuple(SUPPORTED_ADVERSARIAL_TARGETS)
        expected_targets = {"observations", "actions", "rewards", "dynamics"}
        targets_ok = bool(supported) and set(supported).issubset(expected_targets)
        checkpoint_paths: list[str] = []
        if args.corruption_suite in ("adversarial", "all"):
            for target in supported:
                config = _config(args, requested_main[0], "adversarial", target)
                validate_adversarial_target(config)
                if target != "rewards":
                    checkpoint_paths.append(str(resolve_attack_checkpoint(config)))
        adversarial_ok = targets_ok
        adversarial_detail = "supported=" + ",".join(supported)
        if checkpoint_paths:
            adversarial_detail += " checkpoints=" + ",".join(
                sorted(set(checkpoint_paths))
            )
    except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
        adversarial_ok = False
        adversarial_detail = str(exc)
    checks.append(Check("ADVERSARIAL TARGETS", adversarial_ok, adversarial_detail))

    clean_eval_ok = bool(configs) and all(
        config.evaluation_mode == "deterministic_diagnostic"
        and config.evaluation_policy_profile == "deterministic_diagnostic"
        and config.to_dict().get("clean_evaluation") is True
        for config in configs
    )
    checks.append(
        Check(
            "CLEAN EVALUATION",
            clean_eval_ok,
            "separate clean environment; deterministic policy",
        )
    )

    common_reporting_ok = bool(configs) and all(
        (
            config.eval_period,
            config.eval_episodes,
            config.final_window_size,
        )
        == (
            args.evaluation_interval,
            args.evaluation_episodes,
            args.final_window_size,
        )
        for config in configs
    )
    checks.append(
        Check(
            "COMMON REPORTING",
            common_reporting_ok,
            f"interval={args.evaluation_interval} episodes={args.evaluation_episodes} "
            f"final_window={args.final_window_size}",
        )
    )

    collision_paths: list[str] = []
    if args.experiment_name:
        for corruption, target in _settings_for_suite(args.corruption_suite):
            path = comparison_directory(
                args.output_root,
                args.env_name,
                corruption,
                target,
                args.experiment_name,
                args.protocol,
                "research_benchmark__research_benchmark__research_benchmark",
            )
            if path.exists():
                collision_paths.append(str(path))
    output_ok = not collision_paths
    checks.append(
        Check(
            "OUTPUT PATH",
            output_ok,
            (
                "auto-generated UUID comparison name is collision-resistant"
                if not args.experiment_name
                else (
                    "no existing comparison directories"
                    if output_ok
                    else "already exists: " + ", ".join(collision_paths)
                )
            ),
        )
    )

    if args.static_only:
        checks.append(
            Check(
                "D4RL ENVIRONMENT",
                False,
                "NOT CHECKED (--static-only); run without this flag before training",
            )
        )
    else:
        try:
            if environment_loader is None:
                from robust_o2o.environment import preflight_runtime

                environment_loader = preflight_runtime
            environment_loader(
                args.env_name,
                args.dataset_dir,
                protocol=args.protocol,
            )
        except BaseException as exc:
            checks.append(
                Check(
                    "D4RL ENVIRONMENT",
                    False,
                    f"{type(exc).__name__}: {exc}",
                )
            )
        else:
            checks.append(
                Check(
                    "D4RL ENVIRONMENT",
                    True,
                    f"loaded {args.env_name} with protocol={args.protocol}",
                )
            )
    return checks


def _print_report(checks: Iterable[Check]) -> bool:
    checks = tuple(checks)
    for check in checks:
        print(f"{check.label}: {'READY' if check.ok else 'NOT READY'}")
        print(f"  {check.detail}")
    print(
        "CAL-QL LOCOMOTION ADAPTATION: EXCLUDED FROM MAIN "
        "(explicit optional_adapted result only)"
    )
    print(
        "PQE SHARED-ACTOR APPROXIMATION: EXCLUDED FROM MAIN "
        "(explicit optional_diagnostic result only)"
    )
    ready = all(check.ok for check in checks)
    print(f"RESEARCH BENCHMARK STATUS: {'READY' if ready else 'NOT READY'}")
    return ready


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    checks = run_checks(args)
    return 0 if _print_report(checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
