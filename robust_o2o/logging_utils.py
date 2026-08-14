from __future__ import annotations

import csv
import json
import logging
import time
import uuid
from datetime import datetime
from typing import Any, Dict, Optional

from .paths import resolve_run_layout


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
)


class RunLogger:
    def __init__(self, config: object):
        self.start_wall = datetime.now().astimezone()
        self.start_monotonic = time.perf_counter()
        stamp = self.start_wall.strftime("%Y%m%d_%H%M%S")
        short_id = str(uuid.uuid4())[:8]
        run_id = f"{stamp}_{short_id}"
        comparison_id = config.comparison_name or run_id
        self.comparison_dir, runs_dir = resolve_run_layout(
            config.output_dir,
            config.env_name,
            config.corruption,
            config.corruption_target,
            comparison_id,
        )
        self.run_dir = (
            runs_dir
            / config.algorithm
            / config.corruption
            / config.corruption_target
            / config.env_name
            / f"seed_{config.seed}"
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
        formatter = logging.Formatter(
            "%(asctime)s %(levelname)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler = logging.FileHandler(self.run_dir / "result.log", encoding="utf-8")
        stream_handler = logging.StreamHandler()
        file_handler.setFormatter(formatter)
        stream_handler.setFormatter(formatter)
        self.logger.handlers = [file_handler, stream_handler]
        self.last_eval: Optional[Dict[str, float]] = None
        self.config = config

    @property
    def elapsed(self) -> float:
        return time.perf_counter() - self.start_monotonic

    def write_config(self, config: Dict[str, Any]) -> None:
        with (self.run_dir / "config.json").open("w", encoding="utf-8") as stream:
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
        self.logger.info(
            "%s step=%d env_steps=%d return=%.1f±%.1f normalized=%.1f±%.1f",
            phase,
            step,
            env_steps,
            metrics["return_mean"],
            metrics["return_std"],
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
