from __future__ import annotations

import csv
import json
import logging
import shutil
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from .paths import resolve_run_layout
from .manifest import build_experiment_manifest


METRIC_FIELDS = (
    "timestamp",
    "elapsed_seconds",
    "phase",
    "step",
    "env_steps",
    "updates",
    "return_mean",
    "return_std",
    "normalized_return_mean",
    "normalized_return_std",
    "diagnostic_d4rl_reference_scaled_return_mean",
    "diagnostic_d4rl_reference_scaled_return_std",
    "evaluation_mode",
    "return_deterministic",
    "normalized_return_deterministic",
    "return_method_faithful",
    "normalized_return_method_faithful",
)


class RunLogger:
    def __init__(self, config: object):
        self.start_wall = datetime.now().astimezone()
        self.start_monotonic = time.perf_counter()
        self.elapsed_offset = 0.0
        stamp = self.start_wall.strftime("%Y%m%d_%H%M%S")
        short_id = str(uuid.uuid4())[:8]
        run_id = f"{stamp}_{short_id}"
        self.run_id = run_id
        self.short_id = short_id
        if config.resume_run:
            supplied = Path(config.resume_run).expanduser().resolve()
            run_dir = supplied
            if supplied.is_file():
                # <run>/checkpoints/<phase>/<file>.pt
                run_dir = supplied.parents[2]
            if not (run_dir / "metrics.csv").exists():
                raise ValueError(
                    "--resume-run must name a run directory or one of its checkpoints"
                )
            runs_ancestor = next(
                (parent for parent in run_dir.parents if parent.name == "runs"), None
            )
            if runs_ancestor is None:
                raise ValueError("resume run is not inside a canonical runs directory")
            self.comparison_dir = runs_ancestor.parent
            self.run_dir = run_dir
            self.run_id = run_dir.name
            self.metrics_path = run_dir / "metrics.csv"
            self.train_metrics_path = run_dir / "train_metrics.jsonl"
            summary_path = run_dir / "summary.json"
            if summary_path.exists():
                previous_summary = json.loads(
                    summary_path.read_text(encoding="utf-8")
                )
                self.elapsed_offset = float(
                    previous_summary.get("elapsed_seconds", 0.0)
                )
                previous_start = previous_summary.get("start_time")
                if previous_start:
                    self.start_wall = datetime.strptime(
                        previous_start, "%Y-%m-%d %H:%M:%S"
                    ).replace(tzinfo=datetime.now().astimezone().tzinfo)
            else:
                with self.metrics_path.open(newline="", encoding="utf-8") as stream:
                    rows = list(csv.DictReader(stream))
                elapsed_values = [
                    float(row["elapsed_seconds"])
                    for row in rows
                    if row.get("elapsed_seconds") not in (None, "")
                ]
                self.elapsed_offset = max(elapsed_values, default=0.0)
                try:
                    parsed_start = datetime.strptime(
                        run_dir.name[:15], "%Y%m%d_%H%M%S"
                    )
                    self.start_wall = parsed_start.replace(
                        tzinfo=datetime.now().astimezone().tzinfo
                    )
                except ValueError:
                    pass
            self.logger = logging.getLogger(f"robust_o2o.{short_id}")
            self.logger.setLevel(logging.INFO)
            self.logger.propagate = False
            self._configure_handlers()
            self._manifest_written = True
            self.last_eval = None
            self.config = config
            return

        comparison_id = config.comparison_name or run_id
        self.comparison_dir, runs_dir = resolve_run_layout(
            config.output_dir,
            config.env_name,
            config.corruption,
            config.corruption_target,
            comparison_id,
            config.protocol,
            config.algorithm_profile,
        )
        self.run_dir = (
            runs_dir
            / config.algorithm
            / config.suite_profile
            / config.implementation_profile
            / config.implementation_fidelity
            / (
                f"budget_{config.suite_profile}_off{config.offline_steps}"
                f"_on{config.online_steps}_utd"
                f"{config.wsrl_utd_ratio if config.algorithm == 'wsrl' else config.updates_per_step}"
            )
            / config.resolved_algorithm_profile
            / config.online_replay_profile
            / config.evaluation_policy_profile
            / config.attack_timing
            / config.random_attack_semantics
            / config.adversarial_attack_profile
            / config.mixed_corruption_profile
            / config.action_execution_profile
            / config.task_profile
            / config.corruption
            / config.corruption_target
            / config.env_name
            / f"seed_l{config.learner_seed}_c{config.corruption_seed}"
            / run_id
        )
        self.run_dir.mkdir(parents=True, exist_ok=False)
        self.metrics_path = self.run_dir / "metrics.csv"
        self.train_metrics_path = self.run_dir / "train_metrics.jsonl"
        with self.metrics_path.open("w", newline="", encoding="utf-8") as stream:
            csv.DictWriter(stream, fieldnames=METRIC_FIELDS).writeheader()

        self.logger = logging.getLogger(f"robust_o2o.{short_id}")
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        self._configure_handlers()
        self._manifest_written = False
        self.last_eval: Optional[Dict[str, float]] = None
        self.config = config

    def _configure_handlers(self) -> None:
        formatter = logging.Formatter(
            "%(asctime)s %(levelname)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler = logging.FileHandler(self.run_dir / "result.log", encoding="utf-8")
        stream_handler = logging.StreamHandler()
        file_handler.setFormatter(formatter)
        stream_handler.setFormatter(formatter)
        self.logger.handlers = [file_handler, stream_handler]

    @property
    def elapsed(self) -> float:
        return self.elapsed_offset + time.perf_counter() - self.start_monotonic

    def write_config(self, config: Dict[str, Any]) -> None:
        if self.config.resume_run:
            manifest_path = self.run_dir / "experiment_manifest.json"
            if not manifest_path.exists():
                raise ValueError("resume run has no canonical experiment manifest")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            requested_manifest = build_experiment_manifest(config)
            if requested_manifest["manifest_sha256"] != manifest["manifest_sha256"]:
                raise ValueError(
                    "resume configuration does not match the original canonical "
                    "manifest; use --initialize-from-checkpoint for a new run"
                )
            setattr(self.config, "_manifest_sha256", manifest["manifest_sha256"])
            event = {
                "timestamp": datetime.now().astimezone().isoformat(),
                "resume_source": str(self.config.resume_run),
                "resolved_config_sha256": requested_manifest["manifest_sha256"],
            }
            with (self.run_dir / "resume_events.jsonl").open(
                "a", encoding="utf-8"
            ) as stream:
                stream.write(json.dumps(event, ensure_ascii=False) + "\n")
            return
        if not self._manifest_written:
            manifest = build_experiment_manifest(config)
            manifest_hash = manifest["manifest_sha256"]
            old_dir = self.run_dir
            desired_dir = (
                old_dir.parent
                / f"manifest_{manifest_hash[:16]}"
                / self.run_id
            )
            for handler in self.logger.handlers:
                handler.close()
            self.logger.handlers = []
            desired_dir.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(old_dir), str(desired_dir))
            self.run_dir = desired_dir
            self.metrics_path = self.run_dir / "metrics.csv"
            self.train_metrics_path = self.run_dir / "train_metrics.jsonl"
            self._configure_handlers()
            config = {
                **config,
                "run_dir": str(self.run_dir),
                "manifest_sha256": manifest_hash,
            }
            setattr(self.config, "_manifest_sha256", manifest_hash)
            with (self.run_dir / "experiment_manifest.json").open(
                "w", encoding="utf-8"
            ) as stream:
                json.dump(manifest, stream, indent=2, ensure_ascii=False)
            self._manifest_written = True
        for filename in ("config.json", "resolved_config.json"):
            with (self.run_dir / filename).open("w", encoding="utf-8") as stream:
                json.dump(config, stream, indent=2, ensure_ascii=False, default=str)

    def log_train(
        self,
        phase: str,
        step: int,
        env_steps: int,
        updates: int,
        metrics: Dict[str, float],
    ) -> None:
        record = {
            "timestamp": datetime.now().astimezone().isoformat(),
            "elapsed_seconds": self.elapsed,
            "phase": phase,
            "step": step,
            "env_steps": env_steps,
            "updates": updates,
            **metrics,
        }
        with self.train_metrics_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")

    def log_evaluation(
        self,
        phase: str,
        step: int,
        env_steps: int,
        updates: int,
        metrics: Dict[str, float],
    ) -> None:
        record = {
            "timestamp": datetime.now().astimezone().isoformat(),
            "elapsed_seconds": self.elapsed,
            "phase": phase,
            "step": step,
            "env_steps": env_steps,
            "updates": updates,
            **metrics,
        }
        with self.metrics_path.open("a", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=METRIC_FIELDS)
            writer.writerow({key: record.get(key, "") for key in METRIC_FIELDS})
        self.last_eval = metrics
        score_label = (
            "normalized"
            if self.config.protocol == "rpex_d4rl_v2_legacy"
            else "diagnostic_scaled"
        )
        self.logger.info(
            "%s step=%d env_steps=%d return=%.1f±%.1f %s=%.1f±%.1f",
            phase,
            step,
            env_steps,
            metrics["return_mean"],
            metrics["return_std"],
            score_label,
            metrics["normalized_return_mean"],
            metrics["normalized_return_std"],
        )
        try:
            from plot_results import update_comparison_plots

            update_comparison_plots(
                self.comparison_dir,
                self.config.env_name,
                self.config.corruption,
                self.config.corruption_target,
            )
        except Exception as exc:
            self.logger.warning("comparison plot refresh skipped: %s", exc)

    def finish(
        self, status: str, error: Optional[str] = None
    ) -> Dict[str, Any]:
        end = datetime.now().astimezone()
        elapsed = self.elapsed
        summary = {
            "status": status,
            "start_time": format_timestamp(self.start_wall),
            "end_time": format_timestamp(end),
            "timezone": str(self.start_wall.tzinfo),
            "elapsed_seconds": elapsed,
            "elapsed_hms": format_duration(elapsed),
            "last_evaluation": self.last_eval,
            "error": error,
            "run_dir": str(self.run_dir),
        }
        with (self.run_dir / "summary.json").open("w", encoding="utf-8") as stream:
            json.dump(summary, stream, indent=2, ensure_ascii=False)
        if status == "completed":
            try:
                from plot_results import update_comparison_plots

                update_comparison_plots(
                    self.comparison_dir,
                    self.config.env_name,
                    self.config.corruption,
                    self.config.corruption_target,
                )
            except Exception as exc:
                self.logger.warning("final comparison plot refresh skipped: %s", exc)
        self.logger.info("status=%s run_dir=%s", status, self.run_dir)
        # Keep these as the final three normal output lines, per the benchmark
        # requirement. The CLI prints tracebacks before calling finish().
        print(f"START_TIME: {format_timestamp(self.start_wall)}", flush=True)
        print(f"END_TIME: {format_timestamp(end)}", flush=True)
        print(f"ELAPSED: {format_duration(elapsed)} ({elapsed:.3f} seconds)", flush=True)
        return summary


def format_timestamp(value: datetime) -> str:
    return value.astimezone().strftime("%Y-%m-%d %H:%M:%S")


def format_duration(seconds: float) -> str:
    total = int(round(seconds))
    hours, remainder = divmod(total, 3_600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"
