#!/usr/bin/env python3
from __future__ import annotations

import sys
import traceback

from robust_o2o.config import build_parser, config_from_args
from robust_o2o.experiment import run_experiment
from robust_o2o.logging_utils import RunLogger


def main() -> int:
    config = config_from_args(build_parser().parse_args())
    logger = RunLogger(config)
    try:
        run_dir = run_experiment(config, logger)
        try:
            from plot_results import plot_single_run

            plot_single_run(run_dir)
        except Exception as plot_error:
            logger.logger.warning("automatic plot skipped: %s", plot_error)
    except BaseException as exc:
        traceback.print_exc()
        logger.finish("failed", error=f"{type(exc).__name__}: {exc}")
        return 1
    logger.finish("completed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
