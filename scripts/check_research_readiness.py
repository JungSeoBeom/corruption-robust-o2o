#!/usr/bin/env python3
"""Preflight the five-main-baseline custom research benchmark.

The checker has two deliberately separate evidence levels:

* configuration/code checks establish that a launch is CONFIG-READY;
* runtime checks consume a completed run directory when one is supplied.

An absent D4RL runtime or run directory is reported as PENDING.  It is never
printed as a successful runtime check.  Upstream RNG trajectories, parity
certificates, fixed-batch equality, official paper budgets, and fixed seed
cohorts are intentionally outside this practical benchmark gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from robust_o2o.config import (  # noqa: E402
    ACTION_DIMS,
    BENCHMARK_ENVS,
    DEFAULT_PROTOCOL,
    PROTOCOLS,
    ExperimentConfig,
)
from robust_o2o.fidelity import (  # noqa: E402
    BASELINE_REPRODUCTION_REGISTRY,
    MAIN_BASELINES,
)
from robust_o2o.paths import comparison_directory  # noqa: E402


CORRUPTION_SUITES = ("clean", "random", "adversarial", "all")
RANDOM_TARGETS = ("observations", "actions", "rewards", "dynamics")
EXPECTED_MAIN_BASELINES = (
    "rpex",
    "riql_naive",
    "wsrl",
    "cal_ql",
    "pessimistic_q_ensemble",
)


def _csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


@dataclass(frozen=True)
class Check:
    label: str
    # True=ready, False=failed, None=runtime evidence not supplied/available.
    ok: bool | None
    detail: str
    scope: str = "config"


@dataclass(frozen=True)
class RunEvidence:
    run_dir: Path
    config: Mapping[str, Any]
    summary: Mapping[str, Any]
    completion: Mapping[str, Any]

    @property
    def algorithm(self) -> str:
        return str(self.config.get("algorithm", ""))

    @property
    def completed(self) -> bool:
        return self.summary.get("status") == "completed" and bool(self.completion)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Check the five-main-baseline research benchmark without parity, "
            "certificate, exact-RNG, official-budget, or fixed-seed gates"
        )
    )
    parser.add_argument("--env-name", choices=BENCHMARK_ENVS, required=True)
    parser.add_argument(
        "--corruption-suite", choices=CORRUPTION_SUITES, default="random"
    )
    parser.add_argument("--algorithms", type=_csv, default=list(MAIN_BASELINES))
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
        "--run-dir",
        action="append",
        default=[],
        help=(
            "completed run or comparison directory; repeat for multiple roots. "
            "Without it, MC-return/checkpoint evidence is reported PENDING"
        ),
    )
    parser.add_argument(
        "--static-only",
        action="store_true",
        help=(
            "skip D4RL/MuJoCo loading; environment evidence is reported "
            "PENDING rather than READY"
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

        adversarial = tuple(
            ("adversarial", target)
            for target in SUPPORTED_ADVERSARIAL_TARGETS
        )
        return (
            *clean,
            *adversarial,
            *random,
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


def _read_json(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, Mapping) else {}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _discover_runtime_evidence(paths: Iterable[str]) -> list[RunEvidence]:
    config_paths: set[Path] = set()
    for value in paths:
        path = Path(value).expanduser().resolve()
        if path.is_file() and path.name in ("config.json", "resolved_config.json"):
            config_paths.add(path)
        elif path.is_dir():
            direct = path / "config.json"
            if direct.is_file():
                config_paths.add(direct)
            config_paths.update(path.rglob("config.json"))

    records: list[RunEvidence] = []
    for config_path in sorted(config_paths):
        run_dir = config_path.parent
        config = _read_json(config_path)
        if str(config.get("algorithm", "")) not in EXPECTED_MAIN_BASELINES:
            continue
        if config.get("env_name") not in (None, "") and not str(
            config["env_name"]
        ).endswith("-v2"):
            # This does not reject the local diagnostic backend; env_name is
            # still the D4RL dataset identifier there.  It filters unrelated
            # JSON files found below a broad comparison root.
            continue
        records.append(
            RunEvidence(
                run_dir=run_dir,
                config=config,
                summary=_read_json(run_dir / "summary.json"),
                completion=_read_json(run_dir / "completed_experiment_manifest.json"),
            )
        )
    return records


def _latest_completed(
    records: Iterable[RunEvidence], algorithm: str, env_name: str
) -> RunEvidence | None:
    candidates = [
        record
        for record in records
        if record.algorithm == algorithm
        and record.config.get("env_name") == env_name
        and record.completed
    ]
    def completion_order(record: RunEvidence) -> tuple[int, str]:
        try:
            modified = (
                record.run_dir / "completed_experiment_manifest.json"
            ).stat().st_mtime_ns
        except OSError:
            modified = 0
        return modified, str(record.run_dir)

    return max(candidates, key=completion_order, default=None)


def _runtime_checks(
    args: argparse.Namespace, records: list[RunEvidence]
) -> list[Check]:
    if not args.run_dir:
        return [
            Check(
                "CAL-QL ONLINE MC EVIDENCE",
                None,
                "PENDING: pass --run-dir after training to validate completed "
                "trajectories and exact online MC returns",
                "runtime",
            ),
            Check(
                "PQE MEMBER CHECKPOINT EVIDENCE",
                None,
                "PENDING: stage=both is configured to generate five member "
                "checkpoints; pass --run-dir to validate the files and hashes",
                "runtime",
            ),
        ]

    calql = _latest_completed(records, "cal_ql", args.env_name)
    if calql is None:
        calql_check = Check(
            "CAL-QL ONLINE MC EVIDENCE",
            False,
            "no completed canonical cal_ql run with a completion manifest was "
            "found under --run-dir",
            "runtime",
        )
    else:
        from robust_o2o.reporting import (
            ReportingValidationError,
            validate_calql_completion_accounting,
        )

        completed_trajectories = int(
            calql.completion.get("completed_online_trajectories", 0)
        )
        completed_transitions = int(
            calql.completion.get("completed_online_transitions", 0)
        )
        valid_fraction = float(
            calql.completion.get("online_mc_return_valid_fraction", 0.0)
        )
        online_updates = int(
            calql.completion.get("online_critic_gradient_updates", 0)
        )
        try:
            accounting = validate_calql_completion_accounting(
                calql.completion, context=f"run {calql.run_dir}"
            )
        except ReportingValidationError as exc:
            accounting = None
            accounting_error = str(exc)
        else:
            accounting_error = None
        ok = (
            completed_trajectories > 0
            and completed_transitions > 0
            and math.isclose(valid_fraction, 1.0)
            and online_updates > 0
            and calql.completion.get("calql_online_cql_enabled") is True
            and accounting is not None
        )
        accounting_detail = (
            accounting_error
            if accounting_error is not None
            else (
                f"requested={accounting['requested_online_steps']} "
                f"actual={accounting['actual_online_steps']} "
                f"overshoot={accounting['episode_boundary_overshoot']} "
                f"pending={accounting['pending_episode_length']} "
                "effective="
                f"{accounting['effective_calql_training_transitions']}"
            )
        )
        calql_check = Check(
            "CAL-QL ONLINE MC EVIDENCE",
            ok,
            f"run={calql.run_dir} trajectories={completed_trajectories} "
            f"transitions={completed_transitions} valid_fraction={valid_fraction:g} "
            f"online_critic_updates={online_updates}; {accounting_detail}",
            "runtime",
        )

    pqe = _latest_completed(records, "pessimistic_q_ensemble", args.env_name)
    if pqe is None:
        pqe_check = Check(
            "PQE MEMBER CHECKPOINT EVIDENCE",
            False,
            "no completed canonical pessimistic_q_ensemble run with a completion "
            "manifest was found under --run-dir",
            "runtime",
        )
    else:
        member_paths = sorted(
            (pqe.run_dir / "checkpoints" / "offline" / "members").glob("*.pt")
        )
        file_hashes = [_sha256(path) for path in member_paths]
        recorded_hashes = pqe.completion.get("pqe_member_checkpoint_hashes", [])
        recorded_hashes = (
            list(recorded_hashes)
            if isinstance(recorded_hashes, (list, tuple))
            else []
        )
        ok = (
            len(member_paths) == 5
            and len(set(file_hashes)) == 5
            and len(recorded_hashes) == 5
            and all(recorded_hashes)
            and len(set(recorded_hashes)) == 5
            and set(recorded_hashes) == set(file_hashes)
            and pqe.completion.get("shared_actor") is False
            and pqe.completion.get("actor_independence") is True
            and pqe.completion.get("critic_independence") is True
            and pqe.completion.get("pqe_replay_mode") == "balanced_density"
        )
        pqe_check = Check(
            "PQE MEMBER CHECKPOINT EVIDENCE",
            ok,
            f"run={pqe.run_dir} files={len(member_paths)} "
            f"unique_file_hashes={len(set(file_hashes))} "
            f"recorded_unique_hashes={len(set(recorded_hashes))}",
            "runtime",
        )
    return [calql_check, pqe_check]


def run_checks(
    args: argparse.Namespace,
    *,
    environment_loader: Callable[..., object] | None = None,
) -> list[Check]:
    checks: list[Check] = []

    requested_main = tuple(args.algorithms)
    declared_main = tuple(MAIN_BASELINES)
    main_ok = (
        declared_main == EXPECTED_MAIN_BASELINES
        and requested_main == EXPECTED_MAIN_BASELINES
        and len(set(declared_main)) == 5
        and set(BASELINE_REPRODUCTION_REGISTRY) == set(EXPECTED_MAIN_BASELINES)
        and all(
            BASELINE_REPRODUCTION_REGISTRY[name].benchmark_role == "main"
            and BASELINE_REPRODUCTION_REGISTRY[name].main_table_eligible
            for name in EXPECTED_MAIN_BASELINES
        )
    )
    checks.append(
        Check(
            "FIVE MAIN BASELINES",
            main_ok,
            f"declared={','.join(declared_main)} requested={','.join(requested_main)}",
        )
    )

    configs: dict[str, ExperimentConfig] = {}
    config_error: str | None = None
    try:
        configs = {
            algorithm: _config(args, algorithm) for algorithm in requested_main
        }
    except (TypeError, ValueError) as exc:
        config_error = str(exc)
    checks.append(
        Check(
            "RESEARCH CONFIG",
            config_error is None and set(configs) == set(EXPECTED_MAIN_BASELINES),
            config_error
            or (
                "custom offline/online budgets and seeds are accepted; "
                "stage=both uses the five canonical names"
            ),
        )
    )

    calql = configs.get("cal_ql")
    try:
        if calql is None:
            raise ValueError("cal_ql configuration was not constructed")
        from robust_o2o.agents.registry import build_agent

        import torch
        import torch.nn as nn

        action_dim = ACTION_DIMS[args.env_name.split("-", 1)[0]]
        calql_agent = build_agent(
            calql, state_dim=5, action_dim=action_dim, max_action=1.0,
            device=torch.device("cpu")
        )
        actor_hidden = [
            module
            for module in calql_agent.actor.trunk
            if isinstance(module, nn.Linear)
        ]
        q1_linears = [
            module
            for module in calql_agent.q1.net
            if isinstance(module, nn.Linear)
        ]
        calql_network_ok = (
            calql.hidden_layers == 2
            and calql.hidden_dim == 256
            and len(actor_hidden) == 2
            and all(layer.out_features == 256 for layer in actor_hidden)
            and len(q1_linears) == 3
            and all(layer.out_features == 256 for layer in q1_linears[:2])
        )
        calql_network_detail = (
            f"configured={calql.hidden_layers}x{calql.hidden_dim} "
            f"actor_hidden={len(actor_hidden)} q_hidden={len(q1_linears) - 1}"
        )
    except ModuleNotFoundError as exc:
        calql_network_ok = None
        calql_network_detail = f"PENDING: executable topology check needs {exc.name}"
    except (ImportError, RuntimeError, TypeError, ValueError) as exc:
        calql_network_ok = False
        calql_network_detail = str(exc)
    checks.append(
        Check(
            "CAL-QL 2x256 NETWORK",
            calql_network_ok,
            calql_network_detail,
            "runtime" if calql_network_ok is None else "config",
        )
    )

    calql_calibration_ok = bool(
        calql is not None
        and calql.enable_calql is True
        and calql.calibration_mask_mode == "all"
        and calql.mc_return_source == "post_corruption"
        and calql.to_dict().get("calql_online_calibration_enabled") is True
        and calql.to_dict().get("uses_corruption_labels") is False
    )
    checks.append(
        Check(
            "CAL-QL ONLINE CALIBRATION",
            calql_calibration_ok,
            (
                "enabled for offline and completed online trajectories; "
                "MC source=post_corruption; corruption labels hidden"
            ),
        )
    )
    calql_objective_ok = bool(
        calql is not None
        and calql.calql_bc_warmup_steps == 0
        and calql.cql_alpha_online == 5.0
        and calql.cql_n_actions == 10
        and calql.backup_entropy is False
        and calql.cql_max_target_backup is True
        and calql.cql_importance_sample is True
        and calql.offline_ratio is None
    )
    checks.append(
        Check(
            "CAL-QL SAC/CQL PROTOCOL",
            calql_objective_ok,
            (
                f"bc_warmup={getattr(calql, 'calql_bc_warmup_steps', None)} "
                f"online_cql={getattr(calql, 'cql_alpha_online', None)} "
                "dynamic_replay=true"
            ),
        )
    )

    pqe = configs.get("pessimistic_q_ensemble")
    try:
        if pqe is None:
            raise ValueError("pessimistic_q_ensemble configuration was not constructed")
        from robust_o2o.agents.registry import build_agent

        import torch

        action_dim = ACTION_DIMS[args.env_name.split("-", 1)[0]]
        pqe_agent = build_agent(
            pqe, state_dim=5, action_dim=action_dim, max_action=1.0,
            device=torch.device("cpu")
        )
        pqe_agent.assert_independent_parameter_storage()
        pqe_structure_ok = (
            len(pqe_agent.actors) == 5
            and len(pqe_agent.q1_members) == 5
            and len(pqe_agent.q2_members) == 5
            and len(pqe_agent.target_q1_members) == 5
            and len(pqe_agent.target_q2_members) == 5
            and len({id(member) for member in pqe_agent.actors}) == 5
            and pqe_agent.algorithm_metadata().get("shared_actor") is False
            and pqe_agent.algorithm_metadata().get("actor_independence") is True
            and pqe_agent.algorithm_metadata().get("critic_independence") is True
        )
        pqe_structure_detail = (
            "actors=5 twin_critic_pairs=5 target_pairs=5 shared_actor=false"
        )
    except ModuleNotFoundError as exc:
        pqe_structure_ok = None
        pqe_structure_detail = f"PENDING: executable topology check needs {exc.name}"
    except (ImportError, RuntimeError, TypeError, ValueError) as exc:
        pqe_structure_ok = False
        pqe_structure_detail = str(exc)
    checks.append(
        Check(
            "PQE INDEPENDENT ENSEMBLE",
            pqe_structure_ok,
            pqe_structure_detail,
            "runtime" if pqe_structure_ok is None else "config",
        )
    )

    pqe_protocol_ok = bool(
        pqe is not None
        and pqe.pqe_ensemble_size == 5
        and pqe.pqe_replay_mode == "balanced_density"
        and pqe.to_dict().get("shared_actor") is False
        and pqe.pqe_init_online_fraction == 0.75
        and pqe.pqe_first_epoch_multiplier == 5
        and pqe.pqe_first_online_block_steps == 1_000
        and pqe.pqe_weight_batch_size == 256
        and pqe.stage == "both"
    )
    checks.append(
        Check(
            "PQE BALANCED REPLAY/CHECKPOINT PLAN",
            pqe_protocol_ok,
            (
                "balanced density-ratio priority replay; stage=both generates "
                "five independently pretrained member checkpoints"
            ),
        )
    )

    try:
        wsrl = configs.get("wsrl") or _config(args, "wsrl")
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

    try:
        riql = configs.get("riql_naive") or _config(args, "riql_naive")
        from robust_o2o.agents.registry import build_agent

        import torch

        action_dim = ACTION_DIMS[args.env_name.split("-", 1)[0]]
        riql_agent = build_agent(
            riql, state_dim=5, action_dim=action_dim, max_action=1.0,
            device=torch.device("cpu")
        )
        riql_agent.begin_online()
        online_lrs = [
            float(group["lr"])
            for group in riql_agent.actor_optimizer.param_groups
        ]
        riql_ok = bool(online_lrs) and all(value > 0.0 for value in online_lrs)
        riql_ok = riql_ok and riql_agent.actor_scheduler is None
        riql_detail = (
            f"configured_actor_lr={riql.actor_learning_rate:g} "
            f"online_optimizer_lrs={online_lrs} scheduler=None"
        )
    except ModuleNotFoundError as exc:
        riql_ok = None
        riql_detail = f"PENDING: online optimizer check needs {exc.name}"
    except (ImportError, RuntimeError, TypeError, ValueError) as exc:
        riql_ok = False
        riql_detail = str(exc)
    checks.append(
        Check(
            "RIQL ONLINE ACTOR LR",
            riql_ok,
            riql_detail,
            "runtime" if riql_ok is None else "config",
        )
    )

    labels_ok = bool(configs) and all(
        config.calibration_mask_mode != "oracle_exclude_corrupted"
        and config.to_dict().get("uses_corruption_labels") is False
        for config in configs.values()
    )
    try:
        _config(args, "cal_ql").__class__(
            algorithm="cal_ql",
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

    if args.corruption_suite not in ("adversarial", "all"):
        adversarial_ok = True
        adversarial_detail = "not selected; checkpoint validation is deferred until an adversarial suite is requested"
    else:
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
            for target in supported:
                config = _config(args, EXPECTED_MAIN_BASELINES[0], "adversarial", target)
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
    checks.append(
        Check("ADVERSARIAL TARGET/CHECKPOINT", adversarial_ok, adversarial_detail)
    )

    clean_eval_ok = bool(configs) and all(
        config.to_dict().get("clean_evaluation") is True
        and (
            (
                config.algorithm == "rpex"
                and config.evaluation_mode == "both"
                and config.evaluation_policy_profile
                == "official_code_epsilon_switching"
            )
            or (
                config.algorithm != "rpex"
                and config.evaluation_mode == "deterministic_diagnostic"
                and config.evaluation_policy_profile == "deterministic_diagnostic"
            )
        )
        for config in configs.values()
    )
    checks.append(
        Check(
            "CLEAN EVALUATION",
            clean_eval_ok,
            "separate clean environment; RPEX method-faithful primary plus "
            "deterministic secondary, deterministic policy for other baselines",
        )
    )

    common_reporting_ok = bool(configs) and all(
        (config.eval_period, config.eval_episodes, config.final_window_size)
        == (
            args.evaluation_interval,
            args.evaluation_episodes,
            args.final_window_size,
        )
        for config in configs.values()
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
                None,
                "PENDING (--static-only): run without this flag on the training "
                "host before starting the benchmark",
                "runtime",
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
                    "runtime",
                )
            )
        else:
            checks.append(
                Check(
                    "D4RL ENVIRONMENT",
                    True,
                    f"loaded {args.env_name} with protocol={args.protocol}",
                    "runtime",
                )
            )

    records = _discover_runtime_evidence(args.run_dir)
    checks.extend(_runtime_checks(args, records))
    return checks


def _print_report(checks: Iterable[Check]) -> bool:
    checks = tuple(checks)
    for check in checks:
        status = "READY" if check.ok is True else "NOT READY" if check.ok is False else "PENDING"
        print(f"{check.label}: {status}")
        print(f"  {check.detail}")

    config_checks = tuple(check for check in checks if check.scope == "config")
    runtime_checks = tuple(check for check in checks if check.scope == "runtime")
    config_ready = bool(config_checks) and all(check.ok is True for check in config_checks)
    runtime_failed = any(check.ok is False for check in runtime_checks)
    runtime_pending = any(check.ok is None for check in runtime_checks)
    runtime_ready = bool(runtime_checks) and all(check.ok is True for check in runtime_checks)

    print(f"RESEARCH CONFIG STATUS: {'CONFIG-READY' if config_ready else 'NOT READY'}")
    if runtime_failed:
        print("RUNTIME EVIDENCE STATUS: NOT READY")
    elif runtime_pending:
        print("RUNTIME EVIDENCE STATUS: PENDING")
    elif runtime_ready:
        print("RUNTIME EVIDENCE STATUS: READY")
    else:
        print("RUNTIME EVIDENCE STATUS: PENDING")

    if config_ready and runtime_ready:
        overall = "READY"
    elif config_ready and not runtime_failed:
        overall = "CONFIG-READY / RUNTIME EVIDENCE PENDING"
    else:
        overall = "NOT READY"
    print(f"RESEARCH BENCHMARK STATUS: {overall}")
    # Pending post-training evidence does not block a configuration-ready
    # launch.  A supplied but invalid runtime/run directory does fail closed.
    return config_ready and not runtime_failed


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    checks = run_checks(args)
    return 0 if _print_report(checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
