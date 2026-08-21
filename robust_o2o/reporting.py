from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np

from .fidelity import (
    COMMON_BENCHMARK_REPORTING_RULE,
    REPORTING_RULES,
    ReportingRule,
)


class ReportingValidationError(RuntimeError):
    """Raised when evaluation rows cannot support the requested statistic."""


PER_SEED_COLUMNS = (
    "algorithm",
    "environment",
    "condition",
    "seed",
    "aggregation_rule",
    "num_final_evaluations",
    "evaluation_episodes",
    "offline_steps",
    "online_environment_steps",
    "actual_online_environment_steps",
    "critic_gradient_updates",
    "actor_gradient_updates",
    "temperature_updates",
    "configured_utd",
    "actual_utd",
    "offline_corruption_rate",
    "online_corruption_rate",
    "corruption_range",
    "corruption_application_contract",
    "evaluation_corruption",
    "final_window_size",
    "selected_evaluation_steps",
    "seed_score",
    "temporal_std_ddof0",
    "reproduction_status",
    "source_commit",
    "publication_eligible",
    "paper_reproduction_eligible",
    "learner_parity_verified",
    "reporting_rule_verified",
    "condition_certificate_verified",
    "condition_status",
    "run_purpose",
    "implementation_type",
    "benchmark_role",
    "uses_corruption_labels",
    "run_status",
    "run_dir",
)

SUMMARY_COLUMNS = (
    "algorithm",
    "environment",
    "condition",
    "aggregation_rule",
    "num_seeds",
    "expected_seeds",
    "evaluation_episodes",
    "offline_steps",
    "online_environment_steps",
    "configured_utd",
    "offline_corruption_rate",
    "online_corruption_rate",
    "corruption_range",
    "corruption_application_contract",
    "evaluation_corruption",
    "final_window_size",
    "mean",
    "std",
    "std_ddof",
    "reproduction_status",
    "source_commit",
    "publication_eligible",
    "paper_reproduction_eligible",
    "learner_parity_verified",
    "reporting_rule_verified",
    "condition_certificate_verified",
    "condition_status",
    "run_purpose",
    "implementation_type",
    "benchmark_role",
    "uses_corruption_labels",
    "completed_seeds",
    "failed_seeds",
    "partial_seeds",
    "missing_seeds",
    "summary_status",
    "result_eligible",
)

SEED_STATUS_COLUMNS = (
    "algorithm",
    "environment",
    "condition",
    "seed",
    "run_status",
    "error_message",
    "implementation_type",
    "benchmark_role",
    "uses_corruption_labels",
    "run_purpose",
    "run_dir",
)

MAIN_BASELINES = frozenset(("rpex", "riql_naive", "wsrl"))
ADAPTED_BASELINES = frozenset(("cal_ql", "cal_ql_locomotion_adaptation"))
APPROXIMATION_BASELINES = frozenset(
    ("pessimistic_q_ensemble", "pqe_shared_actor_approx")
)


def _empty_score_frames(pd):
    return (
        pd.DataFrame(columns=PER_SEED_COLUMNS),
        pd.DataFrame(columns=SUMMARY_COLUMNS),
    )


def _condition(frame) -> str:
    corruption = str(frame["corruption"].iloc[0])
    target = str(frame["corruption_target"].iloc[0])
    return "clean" if corruption == "clean" else f"{corruption}_{target}"


def _single_value(frame, column: str, run_dir: str):
    values = frame[column].drop_duplicates().tolist()
    if len(values) != 1:
        raise ReportingValidationError(
            f"run {run_dir} mixes {column}: {values!r}"
        )
    return values[0]


def _infer_benchmark_role(algorithm: str, run_purpose: str) -> str:
    if algorithm in ADAPTED_BASELINES:
        return "optional_adapted"
    if algorithm in APPROXIMATION_BASELINES:
        return "optional_diagnostic"
    if algorithm in MAIN_BASELINES and run_purpose == "research_benchmark":
        return "main"
    return "diagnostic"


def _ensure_classification_columns(frame):
    result = frame.copy()
    if result.empty:
        for column in (
            "run_purpose",
            "benchmark_role",
            "implementation_type",
            "uses_corruption_labels",
            "error_message",
        ):
            if column not in result:
                result[column] = []
        return result
    if "run_purpose" not in result:
        result["run_purpose"] = "legacy_unknown"
    if "benchmark_role" not in result:
        result["benchmark_role"] = [
            _infer_benchmark_role(str(algorithm), str(purpose))
            for algorithm, purpose in zip(
                result["algorithm"], result["run_purpose"]
            )
        ]
    if "implementation_type" not in result:
        result["implementation_type"] = result.get(
            "implementation_fidelity", "legacy_unknown"
        )
    if "uses_corruption_labels" not in result:
        result["uses_corruption_labels"] = False
    if "error_message" not in result:
        result["error_message"] = ""
    optional_defaults = {
        "critic_gradient_updates": np.nan,
        "actor_gradient_updates": np.nan,
        "temperature_updates": np.nan,
        "configured_utd": np.nan,
        "actual_utd": np.nan,
        "offline_corruption_rate": np.nan,
        "online_corruption_rate": np.nan,
        "corruption_range": np.nan,
        "corruption_application_contract": "legacy_unknown",
        "evaluation_corruption": "legacy_unknown",
        "final_window_size": 3,
    }
    for column, value in optional_defaults.items():
        if column not in result:
            result[column] = value
    return result


def seed_run_statuses(frame):
    """Return one explicit status row per run, including failed/partial seeds."""

    import pandas as pd

    if frame.empty:
        return pd.DataFrame(columns=SEED_STATUS_COLUMNS)
    classified = _ensure_classification_columns(frame)
    rows = []
    for run_dir, run in classified.groupby("run_dir", sort=True):
        rows.append(
            {
                "algorithm": str(_single_value(run, "algorithm", run_dir)),
                "environment": str(_single_value(run, "env_name", run_dir)),
                "condition": _condition(run),
                "seed": int(_single_value(run, "seed", run_dir)),
                "run_status": str(_single_value(run, "run_status", run_dir)),
                "error_message": str(
                    _single_value(run, "error_message", run_dir)
                ),
                "implementation_type": str(
                    _single_value(run, "implementation_type", run_dir)
                ),
                "benchmark_role": str(
                    _single_value(run, "benchmark_role", run_dir)
                ),
                "uses_corruption_labels": bool(
                    _single_value(run, "uses_corruption_labels", run_dir)
                ),
                "run_purpose": str(
                    _single_value(run, "run_purpose", run_dir)
                ),
                "run_dir": str(run_dir),
            }
        )
    result = pd.DataFrame(rows, columns=SEED_STATUS_COLUMNS)
    identity = [
        "algorithm",
        "environment",
        "condition",
        "seed",
        "benchmark_role",
        "run_purpose",
    ]
    duplicates = result[result.duplicated(identity, keep=False)]
    if not duplicates.empty:
        raise ReportingValidationError(
            "multiple runs claim the same seed identity (no run was silently "
            f"selected): {duplicates[identity + ['run_dir']].to_dict('records')}"
        )
    return result


def common_reporting_rule(
    phase: str,
    *,
    final_evaluations: int | None = None,
    evaluation_episodes: int | None = None,
) -> ReportingRule:
    """Return the declared phase-specific common diagnostic rule."""

    if phase not in ("offline", "online"):
        raise ValueError(f"unsupported reporting phase: {phase!r}")
    final_evaluations = int(
        final_evaluations
        if final_evaluations is not None
        else COMMON_BENCHMARK_REPORTING_RULE.final_evaluations
    )
    evaluation_episodes = int(
        evaluation_episodes
        if evaluation_episodes is not None
        else COMMON_BENCHMARK_REPORTING_RULE.evaluation_episodes
    )
    if final_evaluations <= 0:
        raise ValueError("final_evaluations must be positive")
    if evaluation_episodes <= 0:
        raise ValueError("evaluation_episodes must be positive")
    if (
        phase == COMMON_BENCHMARK_REPORTING_RULE.phase
        and final_evaluations
        == COMMON_BENCHMARK_REPORTING_RULE.final_evaluations
        and evaluation_episodes
        == COMMON_BENCHMARK_REPORTING_RULE.evaluation_episodes
    ):
        return COMMON_BENCHMARK_REPORTING_RULE
    return ReportingRule(
        rule_id=(
            f"common_mean_last_{final_evaluations}_"
            f"{phase}_evaluations_per_seed_then_population_mean_std"
        ),
        phase=phase,
        final_evaluations=final_evaluations,
        evaluation_episodes=evaluation_episodes,
        source="repository common cross-algorithm benchmark metric",
        verified=True,
    )


def _joined_seeds(values: Iterable[int]) -> str:
    return ";".join(str(value) for value in sorted(set(values)))


def _one_group_value(frame, column: str, identity: str):
    values = frame[column].drop_duplicates().tolist()
    if len(values) != 1:
        raise ReportingValidationError(
            f"summary group {identity} mixes {column}: {values!r}"
        )
    return values[0]


def aggregate_seed_scores(
    frame,
    *,
    rule_by_algorithm: Mapping[str, ReportingRule],
    strict: bool,
    expected_seeds: Iterable[int] | None = None,
):
    """Apply a reporting rule within each seed before cross-seed statistics.

    `frame` is the canonical output of ``plot_results._load_runs``.  One run
    contributes exactly one seed scalar, preventing evaluation rows from being
    treated as independent seeds.
    """

    import pandas as pd

    if frame.empty:
        if strict:
            raise ReportingValidationError("no evaluation rows were supplied")
        return _empty_score_frames(pd)
    frame = _ensure_classification_columns(frame)
    required = {
        "run_dir",
        "algorithm",
        "seed",
        "env_name",
        "corruption",
        "corruption_target",
        "phase",
        "step",
        "env_steps",
        "normalized_return_mean",
        "run_status",
        "eval_episodes",
        "implementation_fidelity",
        "upstream_commit",
        "publication_eligible",
        "paper_reproduction_eligible",
        "learner_parity_verified",
        "reporting_rule_verified",
        "condition_certificate_verified",
        "condition_status",
        "run_purpose",
        "planned_online_steps",
        "planned_offline_steps",
        "actual_online_steps",
        "episode_boundary_overshoot",
        "eval_period",
        "online_budget_semantics",
        "environment_horizon",
        "implementation_type",
        "benchmark_role",
        "uses_corruption_labels",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ReportingValidationError(
            f"evaluation frame is missing columns: {missing}"
        )
    status_frame = seed_run_statuses(frame)
    completed = frame[frame["run_status"] == "completed"].copy()
    if completed.empty:
        if strict:
            raise ReportingValidationError("no completed runs are available")

    rows: list[dict] = []
    for run_dir, run in completed.groupby("run_dir", sort=True):
        algorithm = str(_single_value(run, "algorithm", run_dir))
        rule = rule_by_algorithm.get(algorithm)
        if rule is None:
            raise ReportingValidationError(
                f"no reporting rule is registered for {algorithm}"
            )
        if strict and not rule.verified:
            raise ReportingValidationError(
                f"{algorithm} reporting rule is not upstream-verified: {rule.rule_id}"
            )
        phase_rows = run[run["phase"] == rule.phase].copy()
        if phase_rows.empty:
            raise ReportingValidationError(
                f"run {run_dir} has no {rule.phase} evaluations"
            )
        step_column = "env_steps" if rule.phase == "online" else "step"
        phase_rows[step_column] = pd.to_numeric(
            phase_rows[step_column], errors="raise"
        )
        duplicate_steps = phase_rows[phase_rows.duplicated(step_column, keep=False)]
        if not duplicate_steps.empty:
            duplicates = sorted(duplicate_steps[step_column].astype(int).unique())
            raise ReportingValidationError(
                f"run {run_dir} has duplicate {rule.phase} evaluation steps: {duplicates}"
            )
        phase_rows = phase_rows.sort_values(step_column)
        available = len(phase_rows)
        if strict and available < rule.final_evaluations:
            raise ReportingValidationError(
                f"run {run_dir} has {available} {rule.phase} evaluations; "
                f"{rule.rule_id} requires {rule.final_evaluations}"
            )
        take = min(available, rule.final_evaluations)
        selected = phase_rows.tail(take)
        applied_rule_id = rule.rule_id
        if available < rule.final_evaluations:
            applied_rule_id = (
                f"{rule.rule_id}__partial_{take}_of_{rule.final_evaluations}"
            )
        configured_episodes = int(
            _single_value(selected, "eval_episodes", run_dir)
        )
        if strict and configured_episodes != rule.evaluation_episodes:
            raise ReportingValidationError(
                f"run {run_dir} used {configured_episodes} evaluation episodes; "
                f"{rule.rule_id} requires {rule.evaluation_episodes}"
            )
        if strict:
            eval_period = int(_single_value(run, "eval_period", run_dir))
            if eval_period <= 0:
                raise ReportingValidationError(
                    f"run {run_dir} has invalid eval_period={eval_period}"
                )
            if rule.phase == "online":
                actual_budget = int(
                    _single_value(run, "actual_online_steps", run_dir)
                )
                if actual_budget <= 0:
                    raise ReportingValidationError(
                        f"run {run_dir} has no recorded actual online budget"
                    )
                planned_budget = int(
                    _single_value(run, "planned_online_steps", run_dir)
                )
                budget_semantics = str(
                    _single_value(run, "online_budget_semantics", run_dir)
                )
                run_purpose = str(
                    _single_value(run, "run_purpose", run_dir)
                )
                if (
                    run_purpose == "final_benchmark"
                    and algorithm in ("rpex", "riql_naive", "riql_pex")
                    and budget_semantics
                    != "rpex_official_episode_boundary_strict_greater_than"
                ):
                    raise ReportingValidationError(
                        f"run {run_dir} uses {budget_semantics!r}; strict "
                        f"{algorithm} reporting requires official episode-boundary "
                        "budget semantics"
                    )
                environment_horizon = int(
                    _single_value(run, "environment_horizon", run_dir)
                )
                reported_overshoot = int(
                    _single_value(
                        run, "episode_boundary_overshoot", run_dir
                    )
                )
                if budget_semantics == (
                    "rpex_official_episode_boundary_strict_greater_than"
                ):
                    overshoot = actual_budget - planned_budget
                    if (
                        planned_budget <= 0
                        or environment_horizon <= 0
                        or overshoot <= 0
                        or overshoot > environment_horizon
                    ):
                        raise ReportingValidationError(
                            f"run {run_dir} violates official online budget "
                            "semantics: "
                            f"requested={planned_budget}, actual={actual_budget}, "
                            f"horizon={environment_horizon}, overshoot={overshoot}"
                        )
                    if reported_overshoot != overshoot:
                        raise ReportingValidationError(
                            f"run {run_dir} completion overshoot mismatch: "
                            f"recorded={reported_overshoot}, computed={overshoot}"
                        )
                elif budget_semantics == "exact_environment_steps":
                    if (
                        planned_budget <= 0
                        or actual_budget != planned_budget
                        or reported_overshoot != 0
                    ):
                        raise ReportingValidationError(
                            f"run {run_dir} violates exact online budget semantics: "
                            f"requested={planned_budget}, actual={actual_budget}, "
                            f"recorded_overshoot={reported_overshoot}"
                        )
                else:
                    raise ReportingValidationError(
                        f"run {run_dir} has unsupported online budget semantics: "
                        f"{budget_semantics!r}"
                    )
                expected_last_step = (
                    actual_budget // eval_period
                ) * eval_period
            else:
                planned_budget = int(
                    _single_value(run, "planned_offline_steps", run_dir)
                )
                expected_last_step = (
                    planned_budget // eval_period
                ) * eval_period
            actual_last_step = int(selected[step_column].iloc[-1])
            if actual_last_step != expected_last_step:
                raise ReportingValidationError(
                    f"run {run_dir} final {rule.phase} evaluation is stale: "
                    f"actual={actual_last_step}, expected={expected_last_step}"
                )
            phase_steps = phase_rows[step_column].astype(int).to_numpy()
            if np.any(phase_steps % eval_period != 0):
                raise ReportingValidationError(
                    f"run {run_dir} has {rule.phase} evaluations off the "
                    f"declared {eval_period}-step grid"
                )
            expected_window = [
                expected_last_step - eval_period * offset
                for offset in range(take - 1, -1, -1)
            ]
            actual_window = selected[step_column].astype(int).tolist()
            if actual_window != expected_window:
                raise ReportingValidationError(
                    f"run {run_dir} final {rule.phase} evaluation window is "
                    f"not contiguous: actual={actual_window}, "
                    f"expected={expected_window}"
                )
        scores = pd.to_numeric(
            selected["normalized_return_mean"], errors="raise"
        ).to_numpy(dtype=np.float64)
        if not np.isfinite(scores).all():
            raise ReportingValidationError(
                f"run {run_dir} contains non-finite final evaluation scores"
            )
        rows.append(
            {
                "algorithm": algorithm,
                "environment": str(_single_value(run, "env_name", run_dir)),
                "condition": _condition(run),
                "seed": int(_single_value(run, "seed", run_dir)),
                "aggregation_rule": applied_rule_id,
                "num_final_evaluations": int(take),
                "evaluation_episodes": configured_episodes,
                "offline_steps": int(
                    _single_value(run, "planned_offline_steps", run_dir)
                ),
                "online_environment_steps": int(
                    _single_value(run, "planned_online_steps", run_dir)
                ),
                "actual_online_environment_steps": int(
                    _single_value(run, "actual_online_steps", run_dir)
                ),
                "critic_gradient_updates": _single_value(
                    run, "critic_gradient_updates", run_dir
                ),
                "actor_gradient_updates": _single_value(
                    run, "actor_gradient_updates", run_dir
                ),
                "temperature_updates": _single_value(
                    run, "temperature_updates", run_dir
                ),
                "configured_utd": _single_value(
                    run, "configured_utd", run_dir
                ),
                "actual_utd": _single_value(run, "actual_utd", run_dir),
                "offline_corruption_rate": _single_value(
                    run, "offline_corruption_rate", run_dir
                ),
                "online_corruption_rate": _single_value(
                    run, "online_corruption_rate", run_dir
                ),
                "corruption_range": _single_value(
                    run, "corruption_range", run_dir
                ),
                "corruption_application_contract": str(
                    _single_value(
                        run, "corruption_application_contract", run_dir
                    )
                ),
                "evaluation_corruption": str(
                    _single_value(run, "evaluation_corruption", run_dir)
                ),
                "final_window_size": int(
                    _single_value(run, "final_window_size", run_dir)
                ),
                "selected_evaluation_steps": ";".join(
                    str(int(value)) for value in selected[step_column].tolist()
                ),
                "seed_score": float(scores.mean()),
                "temporal_std_ddof0": float(scores.std(ddof=0)),
                "reproduction_status": str(
                    _single_value(run, "implementation_fidelity", run_dir)
                ),
                "source_commit": str(
                    _single_value(run, "upstream_commit", run_dir)
                ),
                "publication_eligible": bool(
                    _single_value(run, "publication_eligible", run_dir)
                ),
                "paper_reproduction_eligible": bool(
                    _single_value(
                        run, "paper_reproduction_eligible", run_dir
                    )
                ),
                "learner_parity_verified": bool(
                    _single_value(run, "learner_parity_verified", run_dir)
                ),
                "reporting_rule_verified": bool(
                    _single_value(run, "reporting_rule_verified", run_dir)
                ),
                "condition_certificate_verified": bool(
                    _single_value(
                        run, "condition_certificate_verified", run_dir
                    )
                ),
                "condition_status": str(
                    _single_value(run, "condition_status", run_dir)
                ),
                "run_purpose": str(
                    _single_value(run, "run_purpose", run_dir)
                ),
                "implementation_type": str(
                    _single_value(run, "implementation_type", run_dir)
                ),
                "benchmark_role": str(
                    _single_value(run, "benchmark_role", run_dir)
                ),
                "uses_corruption_labels": bool(
                    _single_value(run, "uses_corruption_labels", run_dir)
                ),
                "run_status": "completed",
                "run_dir": str(run_dir),
            }
        )
    per_seed = pd.DataFrame(rows, columns=PER_SEED_COLUMNS)
    identity = [
        "algorithm",
        "environment",
        "condition",
        "aggregation_rule",
        "seed",
    ]
    duplicates = per_seed[per_seed.duplicated(identity, keep=False)]
    if not duplicates.empty:
        records = duplicates[identity + ["run_dir"]].to_dict("records")
        raise ReportingValidationError(
            f"multiple completed runs claim the same seed identity: {records}"
        )

    expected = (
        None if expected_seeds is None else sorted(set(int(x) for x in expected_seeds))
    )
    summaries: list[dict] = []
    status_group_columns = [
        "algorithm",
        "environment",
        "condition",
        "implementation_type",
        "benchmark_role",
        "uses_corruption_labels",
        "run_purpose",
    ]
    score_metadata_columns = [
        "aggregation_rule",
        "evaluation_episodes",
        "reproduction_status",
        "source_commit",
        "publication_eligible",
        "paper_reproduction_eligible",
        "learner_parity_verified",
        "reporting_rule_verified",
        "condition_certificate_verified",
        "condition_status",
        "offline_steps",
        "online_environment_steps",
        "configured_utd",
        "offline_corruption_rate",
        "online_corruption_rate",
        "corruption_range",
        "corruption_application_contract",
        "evaluation_corruption",
        "final_window_size",
    ]
    for keys, statuses in status_frame.groupby(status_group_columns, sort=True):
        identity_values = dict(zip(status_group_columns, keys))
        scored = per_seed
        for column, value in identity_values.items():
            scored = scored[scored[column] == value]

        identity = f"{keys[0]}/{keys[1]}/{keys[2]}"
        if not scored.empty:
            metadata = {
                column: _one_group_value(scored, column, identity)
                for column in score_metadata_columns
            }
        else:
            # A failed seed may have no evaluation row.  Keep an explicit NaN
            # summary row using immutable run metadata instead of silently
            # dropping the condition from the result table.
            algorithm = str(keys[0])
            rule = rule_by_algorithm.get(algorithm)
            if rule is None:
                raise ReportingValidationError(
                    f"no reporting rule is registered for {algorithm}"
                )
            source = frame[
                frame["run_dir"].isin(statuses["run_dir"].astype(str))
            ]
            metadata = {
                "aggregation_rule": rule.rule_id,
                "evaluation_episodes": int(
                    _one_group_value(source, "eval_episodes", identity)
                ),
                "reproduction_status": str(
                    _one_group_value(
                        source, "implementation_fidelity", identity
                    )
                ),
                "source_commit": str(
                    _one_group_value(source, "upstream_commit", identity)
                ),
                "publication_eligible": bool(
                    _one_group_value(source, "publication_eligible", identity)
                ),
                "paper_reproduction_eligible": bool(
                    _one_group_value(
                        source, "paper_reproduction_eligible", identity
                    )
                ),
                "learner_parity_verified": bool(
                    _one_group_value(source, "learner_parity_verified", identity)
                ),
                "reporting_rule_verified": bool(
                    _one_group_value(source, "reporting_rule_verified", identity)
                ),
                "condition_certificate_verified": bool(
                    _one_group_value(
                        source, "condition_certificate_verified", identity
                    )
                ),
                "condition_status": str(
                    _one_group_value(source, "condition_status", identity)
                ),
                "offline_steps": int(
                    _one_group_value(source, "planned_offline_steps", identity)
                ),
                "online_environment_steps": int(
                    _one_group_value(source, "planned_online_steps", identity)
                ),
                "configured_utd": _one_group_value(
                    source, "configured_utd", identity
                ),
                "offline_corruption_rate": _one_group_value(
                    source, "offline_corruption_rate", identity
                ),
                "online_corruption_rate": _one_group_value(
                    source, "online_corruption_rate", identity
                ),
                "corruption_range": _one_group_value(
                    source, "corruption_range", identity
                ),
                "corruption_application_contract": str(
                    _one_group_value(
                        source, "corruption_application_contract", identity
                    )
                ),
                "evaluation_corruption": str(
                    _one_group_value(source, "evaluation_corruption", identity)
                ),
                "final_window_size": int(
                    _one_group_value(source, "final_window_size", identity)
                ),
            }

        completed_seeds = sorted(scored["seed"].astype(int).tolist())
        failed_seeds = sorted(
            statuses.loc[
                statuses["run_status"] == "failed", "seed"
            ].astype(int).tolist()
        )
        partial_seeds = sorted(
            statuses.loc[
                ~statuses["run_status"].isin(("completed", "failed")),
                "seed",
            ].astype(int).tolist()
        )
        observed_seeds = sorted(statuses["seed"].astype(int).tolist())
        expected_for_group = expected if expected is not None else observed_seeds
        missing_seeds = sorted(set(expected_for_group) - set(observed_seeds))
        partial_window = bool(
            not scored.empty
            and scored["aggregation_rule"].astype(str).str.contains("__partial_").any()
        )
        uses_labels = bool(keys[5])
        summary_complete = bool(
            completed_seeds
            and not failed_seeds
            and not partial_seeds
            and not missing_seeds
            and not partial_window
            and not uses_labels
            and completed_seeds == expected_for_group
        )
        if strict and completed_seeds != expected_for_group:
            raise ReportingValidationError(
                f"seed set mismatch for {keys[0]}/{keys[2]}: "
                f"actual={completed_seeds}, expected={expected_for_group}; "
                f"failed={failed_seeds}, partial={partial_seeds}, "
                f"missing={missing_seeds}"
            )
        if strict and not summary_complete:
            raise ReportingValidationError(
                f"incomplete summary for {identity}: failed={failed_seeds}, "
                f"partial={partial_seeds}, missing={missing_seeds}, "
                f"partial_window={partial_window}, "
                f"uses_corruption_labels={uses_labels}"
            )

        scores = scored["seed_score"].to_numpy(dtype=np.float64)
        summaries.append(
            {
                "algorithm": keys[0],
                "environment": keys[1],
                "condition": keys[2],
                "aggregation_rule": metadata["aggregation_rule"],
                "num_seeds": len(scores),
                "expected_seeds": _joined_seeds(expected_for_group),
                "evaluation_episodes": metadata["evaluation_episodes"],
                "offline_steps": metadata["offline_steps"],
                "online_environment_steps": metadata[
                    "online_environment_steps"
                ],
                "configured_utd": metadata["configured_utd"],
                "offline_corruption_rate": metadata[
                    "offline_corruption_rate"
                ],
                "online_corruption_rate": metadata[
                    "online_corruption_rate"
                ],
                "corruption_range": metadata["corruption_range"],
                "corruption_application_contract": metadata[
                    "corruption_application_contract"
                ],
                "evaluation_corruption": metadata["evaluation_corruption"],
                "final_window_size": metadata["final_window_size"],
                "mean": float(scores.mean()) if len(scores) else float("nan"),
                # Population std is explicit. This is computed across the
                # per-seed last-K means, never across evaluation rows.
                "std": float(scores.std(ddof=0))
                if len(scores)
                else float("nan"),
                "std_ddof": 0,
                "reproduction_status": metadata["reproduction_status"],
                "source_commit": metadata["source_commit"],
                "publication_eligible": metadata["publication_eligible"],
                "paper_reproduction_eligible": metadata[
                    "paper_reproduction_eligible"
                ],
                "learner_parity_verified": metadata[
                    "learner_parity_verified"
                ],
                "reporting_rule_verified": metadata[
                    "reporting_rule_verified"
                ],
                "condition_certificate_verified": metadata[
                    "condition_certificate_verified"
                ],
                "condition_status": metadata["condition_status"],
                "run_purpose": keys[6],
                "implementation_type": keys[3],
                "benchmark_role": keys[4],
                "uses_corruption_labels": uses_labels,
                "completed_seeds": _joined_seeds(completed_seeds),
                "failed_seeds": _joined_seeds(failed_seeds),
                "partial_seeds": _joined_seeds(partial_seeds),
                "missing_seeds": _joined_seeds(missing_seeds),
                "summary_status": (
                    "complete" if summary_complete else "incomplete"
                ),
                "result_eligible": summary_complete,
            }
        )
    summary = pd.DataFrame(summaries, columns=SUMMARY_COLUMNS)
    return per_seed, summary


def write_reporting_outputs(
    frame,
    output_dir: Path,
    *,
    strict: bool,
    expected_seeds: Iterable[int] | None,
    phase: str = "online",
) -> dict[str, Path]:
    """Write source, common, and role-separated research artifacts."""

    frame = _ensure_classification_columns(frame)
    research_rows = (
        frame[frame["run_purpose"] == "research_benchmark"]
        if not frame.empty
        else frame
    )
    final_window = None
    evaluation_episodes = None
    if not research_rows.empty:
        if "final_window_size" in research_rows:
            windows = sorted(
                set(
                    int(value)
                    for value in research_rows["final_window_size"].dropna()
                )
            )
            if len(windows) > 1:
                raise ReportingValidationError(
                    "research runs mix final-window sizes: " f"{windows}"
                )
            final_window = windows[0] if windows else None
        episodes = sorted(
            set(int(value) for value in research_rows["eval_episodes"].dropna())
        )
        if len(episodes) > 1:
            raise ReportingValidationError(
                "research runs mix evaluation episode counts: " f"{episodes}"
            )
        evaluation_episodes = episodes[0] if episodes else None
    common_rule = common_reporting_rule(
        phase,
        final_evaluations=final_window,
        evaluation_episodes=evaluation_episodes,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    source_rules = {
        name: rule
        for name, rule in REPORTING_RULES.items()
        if rule.verified and rule.phase == phase
    }
    algorithms = (
        frame["algorithm"].drop_duplicates().astype(str).tolist()
        if "algorithm" in frame.columns
        else []
    )
    source_frame = (
        frame[
            frame["algorithm"].isin(source_rules)
            & frame["publication_eligible"].astype(bool)
            & frame["reporting_rule_verified"].astype(bool)
        ]
        if {
            "algorithm",
            "publication_eligible",
            "reporting_rule_verified",
        }.issubset(frame.columns)
        else frame
    )
    paper_frame = (
        source_frame[
            source_frame["paper_reproduction_eligible"].astype(bool)
            & source_frame["learner_parity_verified"].astype(bool)
            & source_frame["reporting_rule_verified"].astype(bool)
            & source_frame["condition_certificate_verified"].astype(bool)
            & (
                source_frame["condition_status"]
                == "paper_reproduction_condition"
            )
        ]
        if {
            "paper_reproduction_eligible",
            "learner_parity_verified",
            "reporting_rule_verified",
            "condition_certificate_verified",
            "condition_status",
        }.issubset(source_frame.columns)
        else source_frame
    )
    # No upstream registry entry currently declares an offline final-score
    # rule.  In that case an offline-only suite must still be able to emit its
    # common diagnostic summary without fabricating a paper result.  For a
    # strict phase that *does* have registered source rules, silently dropping
    # an algorithm would invalidate the benchmark and therefore fails.
    strict_source_frame = frame[frame["run_purpose"] == "final_benchmark"]
    if strict and source_rules and not strict_source_frame.empty:
        completed = strict_source_frame[
            strict_source_frame["run_status"] == "completed"
        ]
        completed_algorithms = set(completed["algorithm"].astype(str))
        unsupported = sorted(completed_algorithms - set(source_rules))
        if unsupported:
            raise ReportingValidationError(
                "no upstream-verified reporting rule for strict "
                f"{phase} summary: {', '.join(unsupported)}"
            )
        if not completed["publication_eligible"].astype(bool).all():
            ineligible = sorted(
                completed.loc[
                    ~completed["publication_eligible"].astype(bool),
                    "algorithm",
                ].astype(str).unique()
            )
            raise ReportingValidationError(
                "strict reporting received non-publication-eligible runs: "
                + ", ".join(ineligible)
            )
    source_seed, _ = aggregate_seed_scores(
        source_frame,
        rule_by_algorithm=source_rules,
        strict=strict and not source_frame.empty,
        expected_seeds=expected_seeds,
    )
    _, paper_summary = aggregate_seed_scores(
        paper_frame,
        rule_by_algorithm=source_rules,
        # An empty source frame is intentional when no paper rule exists for
        # the requested phase; it is written with the canonical CSV schema.
        strict=strict and not paper_frame.empty,
        expected_seeds=expected_seeds,
    )
    common_rules = {
        algorithm: common_rule for algorithm in algorithms
    }
    common_seed, common_summary = aggregate_seed_scores(
        frame,
        rule_by_algorithm=common_rules,
        strict=strict,
        expected_seeds=expected_seeds,
    )

    no_labels = ~frame["uses_corruption_labels"].astype(bool)
    research_mask = (
        (frame["benchmark_role"] == "main")
        & (frame["run_purpose"] == "research_benchmark")
        & no_labels
    )
    adapted_mask = (
        (frame["benchmark_role"] == "optional_adapted") & no_labels
    )
    diagnostic_mask = ~(research_mask | adapted_mask)

    research_frame = frame[research_mask]
    adapted_frame = frame[adapted_mask]
    diagnostic_frame = frame[diagnostic_mask]
    research_seed, research_summary = aggregate_seed_scores(
        research_frame,
        rule_by_algorithm=common_rules,
        strict=strict and not research_frame.empty,
        expected_seeds=expected_seeds,
    )
    adapted_seed, adapted_summary = aggregate_seed_scores(
        adapted_frame,
        rule_by_algorithm=common_rules,
        strict=False,
        expected_seeds=expected_seeds,
    )
    diagnostic_seed, diagnostic_summary = aggregate_seed_scores(
        diagnostic_frame,
        rule_by_algorithm=common_rules,
        strict=False,
        expected_seeds=expected_seeds,
    )
    seed_status = seed_run_statuses(frame)
    outputs = {
        "per_seed_final_scores": output_dir / "per_seed_final_scores.csv",
        "paper_reproduction_summary": output_dir
        / "paper_reproduction_summary.csv",
        "common_per_seed_final_scores": output_dir
        / "common_per_seed_final_scores.csv",
        "common_benchmark_summary": output_dir
        / "common_benchmark_summary.csv",
        "seed_run_status": output_dir / "seed_run_status.csv",
        "research_per_seed_final_scores": output_dir
        / "research_per_seed_final_scores.csv",
        "research_summary": output_dir / "research_summary.csv",
        "adapted_baselines_per_seed_final_scores": output_dir
        / "adapted_baselines_per_seed_final_scores.csv",
        "adapted_baselines_summary": output_dir
        / "adapted_baselines_summary.csv",
        "diagnostic_per_seed_final_scores": output_dir
        / "diagnostic_per_seed_final_scores.csv",
        "diagnostic_summary": output_dir / "diagnostic_summary.csv",
    }
    source_seed.to_csv(outputs["per_seed_final_scores"], index=False)
    paper_summary.to_csv(outputs["paper_reproduction_summary"], index=False)
    common_seed.to_csv(outputs["common_per_seed_final_scores"], index=False)
    common_summary.to_csv(outputs["common_benchmark_summary"], index=False)
    seed_status.to_csv(outputs["seed_run_status"], index=False)
    research_seed.to_csv(
        outputs["research_per_seed_final_scores"], index=False
    )
    research_summary.to_csv(outputs["research_summary"], index=False)
    adapted_seed.to_csv(
        outputs["adapted_baselines_per_seed_final_scores"], index=False
    )
    adapted_summary.to_csv(
        outputs["adapted_baselines_summary"], index=False
    )
    diagnostic_seed.to_csv(
        outputs["diagnostic_per_seed_final_scores"], index=False
    )
    diagnostic_summary.to_csv(outputs["diagnostic_summary"], index=False)
    return outputs


def reporting_registry_payload() -> dict[str, dict]:
    return {name: asdict(rule) for name, rule in REPORTING_RULES.items()}
