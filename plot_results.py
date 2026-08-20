#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Optional

from robust_o2o.fidelity import canonical_json_sha256
from robust_o2o.manifest import aggregation_signature


def _imports():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("Plotting requires matplotlib and pandas") from exc
    return plt, pd


def plot_single_run(run_dir: Path) -> Optional[Path]:
    plt, pd = _imports()
    metrics_path = run_dir / "metrics.csv"
    if not metrics_path.exists() or metrics_path.stat().st_size == 0:
        return None
    frame = pd.read_csv(metrics_path)
    if frame.empty:
        return None
    frame = add_global_plot_steps(frame)
    config_path = run_dir / "config.json"
    config = (
        json.loads(config_path.read_text(encoding="utf-8"))
        if config_path.exists()
        else {}
    )
    figure, axis = plt.subplots(figsize=(8, 5))
    for phase, phase_frame in frame.groupby("phase"):
        axis.plot(
            phase_frame["global_step"],
            phase_frame["normalized_return_mean"],
            marker="o",
            markersize=2,
            label=(
                "offline (gradient updates)"
                if phase == "offline"
                else "online (environment steps)"
            ),
        )
        axis.fill_between(
            phase_frame["global_step"],
            phase_frame["normalized_return_mean"]
            - phase_frame["normalized_return_std"],
            phase_frame["normalized_return_mean"]
            + phase_frame["normalized_return_std"],
            alpha=0.15,
        )
    completed_offline = int(
        frame.loc[frame["phase"] == "offline", "step"].max()
        if (frame["phase"] == "offline").any()
        else 0
    )
    axis.axvline(
        completed_offline,
        color="black",
        linestyle="--",
        linewidth=1.0,
        label="offline → online",
    )
    axis.set_xlabel("Completed offline updates + online environment steps")
    if config.get("score_semantics") == "d4rl_normalized_return":
        axis.set_ylabel("D4RL normalized return")
    else:
        axis.set_ylabel(
            "Diagnostic D4RL-reference-scaled return (not benchmark normalized return)"
        )
        axis.set_title(
            "DIAGNOSTIC: Gymnasium-v4 evaluation, not comparable to D4RL-v2"
        )
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    output = run_dir / "performance.png"
    figure.savefig(output, dpi=180)
    plt.close(figure)
    return output


def add_global_plot_steps(frame):
    """Return a copy with a monotonic offline-to-online x-coordinate."""
    result = frame.copy()
    offline = result["phase"] == "offline"
    online = result["phase"] == "online"
    completed_offline = int(result.loc[offline, "step"].max()) if offline.any() else 0
    result["global_step"] = result["step"]
    result.loc[online, "global_step"] = (
        completed_offline + result.loc[online, "env_steps"]
    )
    ordered = result.sort_values(["global_step", "phase"])["global_step"].to_numpy()
    if len(ordered) > 1 and (ordered[1:] < ordered[:-1]).any():
        raise ValueError("offline/online plot coordinates are not monotonic")
    return result


def _load_runs(root: Path):
    _, pd = _imports()
    frames = []
    for metrics_path in root.rglob("metrics.csv"):
        config_path = metrics_path.parent / "config.json"
        if not config_path.exists():
            continue
        with config_path.open(encoding="utf-8") as stream:
            config = json.load(stream)
        manifest_path = metrics_path.parent / "experiment_manifest.json"
        launch_manifest = None
        if manifest_path.exists():
            launch_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            recorded_launch_digest = launch_manifest.get("manifest_sha256")
            unhashed_launch = {
                key: value
                for key, value in launch_manifest.items()
                if key != "manifest_sha256"
            }
            actual_launch_digest = canonical_json_sha256(unhashed_launch)
            if recorded_launch_digest is None and int(
                launch_manifest.get("manifest_schema_version", 0)
            ) >= 2:
                raise RuntimeError(
                    f"launch manifest SHA256 missing: {manifest_path}"
                )
            if (
                recorded_launch_digest is not None
                and recorded_launch_digest != actual_launch_digest
            ):
                raise RuntimeError(
                    f"launch manifest SHA256 mismatch: {manifest_path}"
                )
        completion_path = metrics_path.parent / "completed_experiment_manifest.json"
        completion_manifest = None
        if completion_path.exists():
            completion_manifest = json.loads(
                completion_path.read_text(encoding="utf-8")
            )
            recorded_digest = completion_manifest.get(
                "completion_manifest_sha256"
            )
            unhashed = {
                key: value
                for key, value in completion_manifest.items()
                if key != "completion_manifest_sha256"
            }
            actual_digest = canonical_json_sha256(unhashed)
            if recorded_digest != actual_digest:
                raise RuntimeError(
                    f"completion manifest SHA256 mismatch: {completion_path}"
                )
            if launch_manifest is None:
                raise RuntimeError(
                    f"completion manifest has no launch manifest: {completion_path}"
                )
            if (
                completion_manifest.get("launch_manifest_sha256")
                != launch_manifest.get("manifest_sha256")
            ):
                raise RuntimeError(
                    f"completion/launch manifest mismatch: {completion_path}"
                )
            immutable_fields = (
                "algorithm",
                "dataset_id",
                "environment_protocol",
                "corruption",
                "corruption_target",
                "learner_seed",
                "offline_updates",
                "requested_online_steps",
                "evaluation_interval",
                "evaluation_episodes",
                "implementation_profile",
                "implementation_fidelity",
                "upstream_commit",
                "repository_commit",
                "repository_dirty",
                "repository_worktree_sha256",
                "suite_profile",
                "run_purpose",
                "condition_status",
                "reporting_rule",
                "reporting_rule_verified",
                "publication_eligible",
                "paper_reproduction_eligible",
                "score_semantics",
                "evaluation_env_id",
                "online_env_id",
                "dataset_sha256",
                "environment_fingerprint",
                "benchmark_seed_set",
                "controller_seed_cohort_attested",
                "final_audit_context_token",
                "corruption_rate",
                "corruption_range",
            )
            changed_launch_fields = {
                key: {
                    "launch": launch_manifest.get(key),
                    "completion": completion_manifest.get(key),
                }
                for key in immutable_fields
                if completion_manifest.get(key) != launch_manifest.get(key)
            }
            if changed_launch_fields:
                raise RuntimeError(
                    "completion manifest changed immutable launch fields at "
                    f"{completion_path}: {changed_launch_fields}"
                )
        if launch_manifest is not None:
            # The launch manifest is the immutable source of configuration and
            # provenance.  The completion manifest contributes measured
            # outcomes only; it must never relabel a run at report time.
            manifest = launch_manifest
        else:
            manifest = {
                "algorithm": config.get("algorithm"),
                "implementation_profile": "legacy_current",
                "implementation_fidelity": "legacy_unknown",
                "suite_profile": "legacy_current",
                "legacy_source_profile": config.get("algorithm_profile"),
            }
        current_manifest = int(manifest.get("manifest_schema_version", 0)) >= 2
        if current_manifest:
            comparison_contract = {
                "algorithm": (config.get("algorithm"), manifest.get("algorithm")),
                "env_name": (config.get("env_name"), manifest.get("dataset_id")),
                "corruption": (
                    config.get("corruption"),
                    manifest.get("corruption"),
                ),
                "corruption_target": (
                    config.get("corruption_target"),
                    manifest.get("corruption_target"),
                ),
                "seed": (config.get("seed"), manifest.get("learner_seed")),
                "protocol": (
                    config.get("protocol"),
                    manifest.get("environment_protocol"),
                ),
                "offline_steps": (
                    config.get("offline_steps"),
                    manifest.get("offline_updates"),
                ),
                "online_steps": (
                    config.get("online_steps"),
                    manifest.get("requested_online_steps"),
                ),
                "eval_period": (
                    config.get("eval_period"),
                    manifest.get("evaluation_interval"),
                ),
                "eval_episodes": (
                    config.get("eval_episodes"),
                    manifest.get("evaluation_episodes"),
                ),
            }
            mismatches = {
                name: {"config": left, "manifest": right}
                for name, (left, right) in comparison_contract.items()
                if left is not None and right is not None and left != right
            }
            if mismatches:
                raise RuntimeError(
                    f"config/manifest comparison contract mismatch at "
                    f"{metrics_path.parent}: {mismatches}"
                )
        summary_path = metrics_path.parent / "summary.json"
        summary = {}
        if summary_path.exists():
            with summary_path.open(encoding="utf-8") as stream:
                summary = json.load(stream)
        if (
            current_manifest
            and summary.get("status") == "completed"
            and not completion_path.exists()
        ):
            raise RuntimeError(
                "completed current-schema run has no verified completion "
                f"manifest: {metrics_path.parent}"
            )
        try:
            frame = pd.read_csv(metrics_path)
        except (OSError, pd.errors.ParserError):
            # A training process may be appending a row while plots refresh.
            continue
        if frame.empty:
            continue
        frame["algorithm"] = manifest.get("algorithm", config.get("algorithm"))
        frame["env_name"] = manifest.get("dataset_id", config.get("env_name"))
        frame["corruption"] = manifest.get(
            "corruption", config.get("corruption")
        )
        frame["corruption_target"] = manifest.get(
            "corruption_target", config.get("corruption_target")
        )
        frame["seed"] = int(manifest.get("learner_seed", config.get("seed", 0)))
        frame["protocol"] = manifest.get(
            "environment_protocol",
            config.get("protocol", "unknown_legacy_protocol"),
        )
        frame["algorithm_profile"] = manifest.get(
            "implementation_profile", "legacy_current"
        )
        frame["implementation_profile"] = frame["algorithm_profile"]
        frame["implementation_fidelity"] = manifest.get(
            "implementation_fidelity", "legacy_unknown"
        )
        frame["upstream_commit"] = manifest.get(
            "upstream_commit", "unknown"
        )
        frame["reporting_rule"] = manifest.get(
            "reporting_rule", "unknown"
        )
        frame["publication_eligible"] = bool(
            manifest.get("publication_eligible", False)
        )
        frame["paper_reproduction_eligible"] = bool(
            manifest.get("paper_reproduction_eligible", False)
        )
        frame["condition_status"] = manifest.get(
            "condition_status", "legacy_unknown"
        )
        frame["suite_profile"] = manifest.get("suite_profile", "legacy_current")
        frame["run_purpose"] = manifest.get("run_purpose", "legacy_unknown")
        frame["attack_timing"] = manifest.get("attack_timing", "legacy_unknown")
        frame["adversarial_attack_profile"] = manifest.get(
            "adversarial_attack_profile",
            manifest.get("attack_implementation", "legacy_unknown"),
        )
        frame["online_corruption_scale_profile"] = manifest.get(
            "online_corruption_scale_profile", "legacy_unknown"
        )
        frame["wsrl_entropy_profile"] = manifest.get(
            "wsrl_entropy_profile", "legacy_unknown"
        )
        frame["aggregation_signature"] = aggregation_signature(manifest)
        frame["resolved_algorithm_profile"] = manifest.get(
            "algorithm_profile",
            config.get("resolved_algorithm_profile", "unknown_legacy_profile"),
        )
        frame["score_semantics"] = manifest.get(
            "score_semantics",
            config.get("score_semantics", "unknown_legacy_score"),
        )
        frame["dataset_id"] = manifest.get(
            "dataset_id", config.get("dataset_id", "unknown_legacy_dataset")
        )
        frame["evaluation_env_id"] = manifest.get(
            "evaluation_env_id",
            config.get("evaluation_env_id", "unknown_legacy_evaluation_env"),
        )
        corruption_rates = manifest.get("corruption_rate") or {}
        frame["offline_corruption_rate"] = float(
            corruption_rates.get(
                "offline", config.get("offline_corruption_rate", -1.0)
            )
        )
        frame["online_corruption_rate"] = float(
            corruption_rates.get(
                "online", config.get("online_corruption_rate", -1.0)
            )
        )
        frame["learner_seed"] = int(
            manifest.get(
                "learner_seed", config.get("learner_seed", config["seed"])
            )
        )
        frame["corruption_seed"] = int(
            manifest.get(
                "corruption_seed", config.get("corruption_seed", config["seed"])
            )
        )
        frame["run_dir"] = str(metrics_path.parent)
        frame["run_status"] = summary.get("status", "unknown")
        elapsed = summary.get("elapsed_seconds")
        frame["run_elapsed_seconds"] = (
            float(elapsed) if elapsed is not None else float("nan")
        )
        frame["planned_offline_steps"] = int(
            manifest.get("offline_updates", config.get("offline_steps", 0))
        )
        offline_rows = frame[frame["phase"] == "offline"]
        frame["completed_offline_steps"] = (
            int(offline_rows["step"].max()) if not offline_rows.empty else 0
        )
        frame["planned_online_steps"] = int(
            manifest.get(
                "requested_online_steps", config.get("online_steps", 0)
            )
        )
        online_manifest_path = (
            metrics_path.parent / "online_corruption_manifest.json"
        )
        online_manifest = (
            json.loads(online_manifest_path.read_text(encoding="utf-8"))
            if online_manifest_path.exists()
            else {}
        )
        manifest_actual_steps = (
            completion_manifest.get("actual_online_steps")
            if completion_manifest is not None
            else manifest.get("actual_online_steps")
        )
        online_actual_steps = online_manifest.get("actual_online_steps")
        if (
            completion_path.exists()
            and manifest_actual_steps is not None
            and online_actual_steps is not None
            and int(manifest_actual_steps) != int(online_actual_steps)
        ):
            raise RuntimeError(
                "completion/online corruption manifest actual-step mismatch: "
                f"{metrics_path.parent}"
            )
        frame["actual_online_steps"] = int(
            manifest_actual_steps
            if manifest_actual_steps is not None
            else online_actual_steps or 0
        )
        frame["eval_episodes"] = int(
            manifest.get("evaluation_episodes", config.get("eval_episodes", 0))
        )
        frame["eval_period"] = int(
            manifest.get("evaluation_interval", config.get("eval_period", 0))
        )
        frame["online_budget_semantics"] = (
            completion_manifest.get("online_budget_semantics")
            if completion_manifest is not None
            else manifest.get(
                "online_budget_semantics", "unknown_legacy_semantics"
            )
        )
        frame["episode_boundary_overshoot"] = int(
            (
                completion_manifest.get("episode_boundary_overshoot", 0)
                if completion_manifest is not None
                else manifest.get("episode_boundary_overshoot", 0)
            )
            or 0
        )
        frame["environment_horizon"] = int(
            manifest.get(
                "environment_horizon", config.get("max_episode_steps", 0)
            )
            or 0
        )
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_runs(root: Path):
    """Public notebook/API entry point for canonical manifest-aware run loading."""

    return _load_runs(Path(root))


def _validate_aggregation_signatures(frame, context: str) -> None:
    if frame.empty:
        return
    signature_counts = frame.groupby("algorithm")[
        "aggregation_signature"
    ].nunique()
    mixed_algorithms = signature_counts[signature_counts > 1]
    if not mixed_algorithms.empty:
        raise RuntimeError(
            f"Manifest mismatch within {context}: "
            + ", ".join(mixed_algorithms.index.tolist())
        )


def write_final_score_summary(
    root: Path,
    output: Path,
    env_name: Optional[str] = None,
    corruption: Optional[str] = None,
    target: Optional[str] = None,
    phase: str = "online",
) -> Path:
    """Write the declared common last-three metric (never a paper metric)."""
    _, pd = _imports()
    from robust_o2o.reporting import (
        aggregate_seed_scores,
        common_reporting_rule,
    )

    frame = _load_runs(root)
    if frame.empty:
        raise RuntimeError(f"No metrics.csv files found below {root}")
    if env_name:
        frame = frame[frame["env_name"] == env_name]
    if corruption:
        frame = frame[frame["corruption"] == corruption]
    if target:
        frame = frame[frame["corruption_target"] == target]
    frame = frame[frame["run_status"] == "completed"]
    if frame.empty:
        raise RuntimeError("No completed runs match the requested summary filters")
    _validate_aggregation_signatures(frame, "final-score groups")

    common_rule = common_reporting_rule(phase)
    rules = {
        algorithm: common_rule
        for algorithm in frame["algorithm"].drop_duplicates().tolist()
    }
    per_seed, result = aggregate_seed_scores(
        frame,
        rule_by_algorithm=rules,
        strict=False,
    )
    elapsed_by_run = (
        frame[["run_dir", "run_elapsed_seconds"]]
        .drop_duplicates("run_dir")
        .set_index("run_dir")["run_elapsed_seconds"]
    )
    per_seed["run_elapsed_seconds"] = per_seed["run_dir"].map(elapsed_by_run)
    elapsed = (
        per_seed.groupby(["algorithm", "environment", "condition"])
        .agg(
            elapsed_seconds_mean=("run_elapsed_seconds", "mean"),
            elapsed_seconds_std=("run_elapsed_seconds", lambda values: values.std(ddof=0)),
        )
        .reset_index()
    )
    result = result.merge(
        elapsed,
        on=["algorithm", "environment", "condition"],
        how="left",
    )
    result["runs"] = result["num_seeds"]
    result["final_normalized_return_mean"] = result["mean"]
    result["final_normalized_return_std"] = result["std"]
    result["final_raw_return_mean"] = float("nan")
    result["final_raw_return_std"] = float("nan")
    result = result.sort_values(
        "final_normalized_return_mean", ascending=False, ignore_index=True
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    return output


def write_reproduction_summaries(
    root: Path,
    output_dir: Path,
    env_name: Optional[str] = None,
    corruption: Optional[str] = None,
    target: Optional[str] = None,
    *,
    strict: bool = False,
    expected_seeds: Optional[Iterable[int]] = None,
    phase: str = "online",
) -> dict[str, Path]:
    from robust_o2o.reporting import write_reporting_outputs

    frame = _load_runs(root)
    if not frame.empty:
        if env_name:
            frame = frame[frame["env_name"] == env_name]
        if corruption:
            frame = frame[frame["corruption"] == corruption]
        if target:
            frame = frame[frame["corruption_target"] == target]
    completed = (
        frame[frame["run_status"] == "completed"]
        if "run_status" in frame.columns
        else frame
    )
    _validate_aggregation_signatures(completed, "reproduction-summary groups")
    return write_reporting_outputs(
        frame,
        output_dir,
        strict=strict,
        expected_seeds=expected_seeds,
        phase=phase,
    )


def plot_aggregate(
    root: Path,
    output: Path,
    env_name: Optional[str] = None,
    corruption: Optional[str] = None,
    target: Optional[str] = None,
    phase: str = "online",
    include_running: bool = False,
) -> Path:
    plt, pd = _imports()
    frame = _load_runs(root)
    if frame.empty:
        raise RuntimeError(f"No metrics.csv files found below {root}")
    if env_name:
        frame = frame[frame["env_name"] == env_name]
    if corruption:
        frame = frame[frame["corruption"] == corruption]
    if target:
        frame = frame[frame["corruption_target"] == target]
    allowed_statuses = ("completed", "running", "unknown") if include_running else ("completed",)
    frame = frame[frame["run_status"].isin(allowed_statuses)]
    _validate_aggregation_signatures(frame, "algorithm curves")
    identity_columns = [
        "algorithm",
        "algorithm_profile",
        "protocol",
        "learner_seed",
        "corruption_seed",
    ]
    identities = frame[[*identity_columns, "run_dir"]].drop_duplicates()
    duplicates = identities.duplicated(identity_columns, keep=False)
    if duplicates.any():
        raise RuntimeError(
            "Duplicate runs for the same algorithm/profile/protocol/seeds; "
            "select an explicit comparison directory instead of silently "
            "choosing the latest run"
        )
    for column in ("protocol", "dataset_id", "evaluation_env_id"):
        if frame[column].nunique(dropna=False) > 1:
            raise RuntimeError(
                f"Mixed {column} values cannot be aggregated without an explicit "
                "diagnostic override"
            )
    frame = frame.copy()
    duplicate_rows = frame[
        frame.duplicated(
            ["run_dir", "phase", "step", "env_steps"], keep=False
        )
    ]
    if not duplicate_rows.empty:
        identity = duplicate_rows[
            ["run_dir", "phase", "step", "env_steps"]
        ].drop_duplicates().to_dict("records")
        raise RuntimeError(f"Duplicate evaluation rows cannot be plotted: {identity}")
    if phase == "offline_online":
        frame = frame[frame["phase"].isin(("offline", "online"))]
        frame["plot_step"] = frame["step"]
        online = frame["phase"] == "online"
        frame.loc[online, "plot_step"] = (
            frame.loc[online, "completed_offline_steps"]
            + frame.loc[online, "env_steps"]
        )
        x_column = "plot_step"
        x_label = "Offline updates + online environment steps"
    else:
        frame = frame[frame["phase"] == phase]
        x_column = "env_steps" if phase == "online" else "step"
        x_label = (
            "Online environment steps" if phase == "online" else "Offline updates"
        )

    figure, axis = plt.subplots(figsize=(10, 6))
    summary_frames = []
    group_keys = (
        "algorithm",
        "aggregation_signature",
        "protocol",
        "algorithm_profile",
        "resolved_algorithm_profile",
        "dataset_id",
        "evaluation_env_id",
        "env_name",
        "corruption",
        "corruption_target",
        "offline_corruption_rate",
        "online_corruption_rate",
    )
    for group, group_frame in frame.groupby(list(group_keys)):
        summary = (
            group_frame.groupby(x_column)
            .agg(
                mean=("normalized_return_mean", "mean"),
                seed_std=(
                    "normalized_return_mean",
                    lambda values: values.std(ddof=0),
                ),
                count=("seed", "nunique"),
            )
            .reset_index()
            .sort_values(x_column)
        )
        # Bands represent uncertainty across independent training seeds only.
        # A single seed therefore has no uncertainty band.
        summary["std"] = summary["seed_std"].fillna(0.0)
        for key, value in zip(group_keys, group):
            summary[key] = value
        summary_frames.append(summary)
        label = f"{group[0]} [{group[4]}]"
        if group_frame["suite_profile"].iloc[0] in (
            "common_budget_robustness",
            "common_budget_diagnostic",
        ):
            label += " | COMMON-BUDGET / NOT PAPER REPRO"
        if group_frame["implementation_fidelity"].iloc[0] == "approximation":
            label += " | PQE shared-actor approximation"
        if (
            group_frame["adversarial_attack_profile"].iloc[0]
            == "experimental_sign_pgd"
        ):
            label += " | EXPERIMENTAL SIGN-PGD"
        if group_frame["attack_timing"].iloc[0] != "legacy_unknown":
            timing_label = {
                "official_code_post_transition_replay_poisoning": "post-replay",
                "paper_pre_action_sensor_actuator": "pre-action",
            }.get(
                group_frame["attack_timing"].iloc[0],
                group_frame["attack_timing"].iloc[0],
            )
            label += f" | {timing_label}"
        if group_frame["online_corruption_scale_profile"].iloc[0] != "legacy_unknown":
            scale_label = {
                "rpex_official_code": "unit-scale",
                "dataset_std_scaled_extension": "dataset-std",
            }.get(
                group_frame["online_corruption_scale_profile"].iloc[0],
                group_frame["online_corruption_scale_profile"].iloc[0],
            )
            label += f" | {scale_label}"
        if (
            group[0] == "wsrl"
            and group_frame["wsrl_entropy_profile"].iloc[0] == "legacy_zero"
        ):
            label += " | LEGACY ENTROPY 0"
        if "oracle" in str(group[4]):
            label += " ORACLE"
        if phase in ("offline", "offline_online") and group[0] == "wsrl":
            label = f"CQL-REDQ pretrainer for WSRL [{group[4]}]"
        if len(frame["env_name"].unique()) > 1:
            label += f" | {group[7]}"
        axis.plot(summary[x_column], summary["mean"], label=label)
        axis.fill_between(
            summary[x_column],
            summary["mean"] - summary["std"],
            summary["mean"] + summary["std"],
            alpha=0.15,
        )
    if not summary_frames:
        axis.text(
            0.5,
            0.5,
            f"No {phase.replace('_', ' → ')} metrics yet",
            transform=axis.transAxes,
            ha="center",
            va="center",
        )
    if phase == "offline_online" and not frame.empty:
        boundary = int(frame["completed_offline_steps"].max())
        axis.axvline(
            boundary,
            color="black",
            linestyle="--",
            linewidth=1.2,
            label="offline → online",
        )
    axis.set_xlabel(x_label)
    score_semantics = set(frame.get("score_semantics", []))
    ylabel = (
        "D4RL normalized return"
        if score_semantics == {"d4rl_normalized_return"}
        else "Diagnostic D4RL-reference-scaled return (not benchmark normalized return)"
    )
    axis.set_ylabel(ylabel)
    if score_semantics != {"d4rl_normalized_return"}:
        axis.set_title(
            "DIAGNOSTIC: Gymnasium-v4 evaluation, not comparable to D4RL-v2"
        )
    axis.grid(alpha=0.25)
    if summary_frames:
        axis.legend(fontsize=8, ncol=2)
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output.with_name(f".{output.stem}.tmp{output.suffix}")
    figure.savefig(temporary_output, dpi=180)
    temporary_output.replace(output)
    plt.close(figure)
    csv_output = output.with_suffix(".csv")
    temporary_csv = csv_output.with_name(f".{csv_output.stem}.tmp.csv")
    if summary_frames:
        summary_frame = pd.concat(summary_frames, ignore_index=True)
    else:
        summary_frame = pd.DataFrame(
            columns=[x_column, "mean", "std", "count", *group_keys]
        )
    summary_frame.to_csv(temporary_csv, index=False)
    temporary_csv.replace(csv_output)
    return output


def update_live_comparison_plots(
    comparison_dir: Path,
    env_name: Optional[str] = None,
    corruption: Optional[str] = None,
    target: Optional[str] = None,
) -> dict[str, Path]:
    """Refresh only explicitly non-canonical running/partial plots."""

    runs_dir = comparison_dir / "runs"
    outputs = {}
    for phase in ("offline_online", "offline", "online"):
        output = comparison_dir / f"diagnostic_running_{phase}.png"
        outputs[phase] = plot_aggregate(
            runs_dir,
            output,
            env_name,
            corruption,
            target,
            phase,
            include_running=True,
        )
    return outputs


def update_comparison_plots(
    comparison_dir: Path,
    env_name: Optional[str] = None,
    corruption: Optional[str] = None,
    target: Optional[str] = None,
) -> dict[str, Path]:
    """Publish completed canonical plots after suite-level validation."""
    runs_dir = comparison_dir / "runs"
    outputs = {}
    for phase in ("offline_online", "offline", "online"):
        output = comparison_dir / f"comparison_{phase}.png"
        outputs[phase] = plot_aggregate(
            runs_dir,
            output,
            env_name,
            corruption,
            target,
            phase,
            include_running=False,
        )
    update_live_comparison_plots(
        comparison_dir, env_name, corruption, target
    )
    return outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot robust O2O performance logs")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--output", default="results/aggregate_performance.png")
    parser.add_argument("--env-name")
    parser.add_argument("--corruption")
    parser.add_argument("--target")
    parser.add_argument(
        "--phase",
        choices=("offline_online", "offline", "online"),
        default="online",
    )
    parser.add_argument("--single-run")
    parser.add_argument(
        "--include-running-diagnostic",
        action="store_true",
        help="include running/legacy-unknown runs in a diagnostic-only plot",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.single_run:
        output = plot_single_run(Path(args.single_run).expanduser().resolve())
        print(output)
        return
    output = plot_aggregate(
        Path(args.results_dir).expanduser().resolve(),
        Path(args.output).expanduser().resolve(),
        args.env_name,
        args.corruption,
        args.target,
        args.phase,
        include_running=args.include_running_diagnostic,
    )
    print(output)


if __name__ == "__main__":
    main()
