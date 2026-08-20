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
    "selected_evaluation_steps",
    "seed_score",
    "temporal_std_ddof0",
    "reproduction_status",
    "source_commit",
    "publication_eligible",
    "paper_reproduction_eligible",
    "condition_status",
    "run_purpose",
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
    "mean",
    "std",
    "std_ddof",
    "reproduction_status",
    "source_commit",
    "publication_eligible",
    "paper_reproduction_eligible",
    "condition_status",
    "run_purpose",
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


def common_reporting_rule(phase: str) -> ReportingRule:
    """Return the declared phase-specific common diagnostic rule."""

    if phase not in ("offline", "online"):
        raise ValueError(f"unsupported reporting phase: {phase!r}")
    if phase == COMMON_BENCHMARK_REPORTING_RULE.phase:
        return COMMON_BENCHMARK_REPORTING_RULE
    return ReportingRule(
        rule_id=(
            "common_mean_last_3_"
            f"{phase}_evaluations_per_seed_then_population_mean_std"
        ),
        phase=phase,
        final_evaluations=COMMON_BENCHMARK_REPORTING_RULE.final_evaluations,
        evaluation_episodes=(
            COMMON_BENCHMARK_REPORTING_RULE.evaluation_episodes
        ),
        source="repository common cross-algorithm benchmark metric",
        verified=True,
    )


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
        "condition_status",
        "run_purpose",
        "planned_online_steps",
        "planned_offline_steps",
        "actual_online_steps",
        "episode_boundary_overshoot",
        "eval_period",
        "online_budget_semantics",
        "environment_horizon",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ReportingValidationError(
            f"evaluation frame is missing columns: {missing}"
        )
    completed = frame[frame["run_status"] == "completed"].copy()
    if completed.empty:
        if strict:
            raise ReportingValidationError("no completed runs are available")
        return _empty_score_frames(pd)

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
                "condition_status": str(
                    _single_value(run, "condition_status", run_dir)
                ),
                "run_purpose": str(
                    _single_value(run, "run_purpose", run_dir)
                ),
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

    expected = None if expected_seeds is None else sorted(set(expected_seeds))
    summaries: list[dict] = []
    group_columns = [
        "algorithm",
        "environment",
        "condition",
        "aggregation_rule",
        "evaluation_episodes",
        "reproduction_status",
        "source_commit",
        "publication_eligible",
        "paper_reproduction_eligible",
        "condition_status",
        "run_purpose",
    ]
    for keys, group in per_seed.groupby(group_columns, sort=True):
        actual = sorted(group["seed"].astype(int).tolist())
        if strict and expected is not None and actual != expected:
            raise ReportingValidationError(
                f"seed set mismatch for {keys[0]}/{keys[2]}: "
                f"actual={actual}, expected={expected}"
            )
        scores = group["seed_score"].to_numpy(dtype=np.float64)
        summaries.append(
            {
                "algorithm": keys[0],
                "environment": keys[1],
                "condition": keys[2],
                "aggregation_rule": keys[3],
                "num_seeds": len(scores),
                "expected_seeds": (
                    ";".join(str(value) for value in expected)
                    if expected is not None
                    else ""
                ),
                "evaluation_episodes": keys[4],
                "mean": float(scores.mean()),
                # Population std is explicit; upstream RPEX does not publish
                # code specifying cross-seed ddof, so this is a declared
                # repository convention rather than an "exact" code claim.
                "std": float(scores.std(ddof=0)),
                "std_ddof": 0,
                "reproduction_status": keys[5],
                "source_commit": keys[6],
                "publication_eligible": keys[7],
                "paper_reproduction_eligible": keys[8],
                "condition_status": keys[9],
                "run_purpose": keys[10],
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
    """Write separate source-primary and common benchmark artifacts."""

    common_rule = common_reporting_rule(phase)
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
        ]
        if "algorithm" in frame.columns
        else frame
    )
    paper_frame = (
        source_frame[
            source_frame["paper_reproduction_eligible"].astype(bool)
            & (
                source_frame["condition_status"]
                == "paper_reproduction_condition"
            )
        ]
        if {
            "paper_reproduction_eligible",
            "condition_status",
        }.issubset(source_frame.columns)
        else source_frame
    )
    # No upstream registry entry currently declares an offline final-score
    # rule.  In that case an offline-only suite must still be able to emit its
    # common diagnostic summary without fabricating a paper result.  For a
    # strict phase that *does* have registered source rules, silently dropping
    # an algorithm would invalidate the benchmark and therefore fails.
    if strict and source_rules and "run_status" in frame.columns:
        completed = frame[frame["run_status"] == "completed"]
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
    outputs = {
        "per_seed_final_scores": output_dir / "per_seed_final_scores.csv",
        "paper_reproduction_summary": output_dir
        / "paper_reproduction_summary.csv",
        "common_per_seed_final_scores": output_dir
        / "common_per_seed_final_scores.csv",
        "common_benchmark_summary": output_dir
        / "common_benchmark_summary.csv",
    }
    source_seed.to_csv(outputs["per_seed_final_scores"], index=False)
    paper_summary.to_csv(outputs["paper_reproduction_summary"], index=False)
    common_seed.to_csv(outputs["common_per_seed_final_scores"], index=False)
    common_summary.to_csv(outputs["common_benchmark_summary"], index=False)
    return outputs


def reporting_registry_payload() -> dict[str, dict]:
    return {name: asdict(rule) for name, rule in REPORTING_RULES.items()}
