#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import os
import sys
import traceback
from typing import Any

from robust_o2o.config import build_parser, config_from_args
from robust_o2o.experiment import run_experiment
from robust_o2o.final_gate import (
    AUDIT_RECEIPT_SHA256_ENV,
    FinalAuditGateError,
    RECEIPT_SCHEMA,
    require_final_benchmark_audit,
)
from robust_o2o.logging_utils import RunLogger, resolve_resume_run_directory
from robust_o2o.manifest import verify_experiment_manifest


def _receipt_content_sha256(receipt: dict[str, Any]) -> str:
    content = (
        json.dumps(receipt, sort_keys=True, indent=2, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    return hashlib.sha256(content).hexdigest()


def inherit_verified_final_resume_attestation(
    config: object,
    audit_receipt: dict[str, Any] | None,
) -> None:
    """Inherit cohort status only from a verified launch in the same context.

    A new controller necessarily issues a new receipt (and therefore a new
    receipt digest).  The original launch's hashed manifest and persisted
    receipt evidence prove that the run was a publication-eligible member of
    a controller-attested cohort; the current PASS/READY receipt proves that
    resuming is still authorized in the identical code/runtime context.
    """

    if config.run_purpose != "final_benchmark" or not config.resume_run:
        return
    if not isinstance(audit_receipt, dict):
        raise ValueError("strict final resume requires a verified current audit receipt")
    current_receipt_digest = getattr(
        config, "_final_audit_receipt_sha256", None
    )
    if (
        not isinstance(current_receipt_digest, str)
        or current_receipt_digest != _receipt_content_sha256(audit_receipt)
    ):
        raise ValueError(
            "strict final resume current audit receipt digest is invalid"
        )
    current_result = audit_receipt.get("audit_result")
    if (
        audit_receipt.get("schema") != RECEIPT_SCHEMA
        or not isinstance(current_result, dict)
        or current_result.get("reproducibility_audit") != "PASS"
        or current_result.get("final_benchmark_status") != "READY"
    ):
        raise ValueError("strict final resume current audit receipt is not PASS/READY")

    run_dir = resolve_resume_run_directory(config.resume_run)
    manifest_path = run_dir / "experiment_manifest.json"
    if not manifest_path.exists():
        raise ValueError("strict final resume has no canonical launch manifest")
    launch = json.loads(manifest_path.read_text(encoding="utf-8"))
    verify_experiment_manifest(launch)
    if launch.get("run_purpose") != "final_benchmark":
        raise ValueError("strict final resume source was not a final benchmark run")
    if launch.get("controller_seed_cohort_attested") is not True:
        raise ValueError(
            "strict final resume source has no controller seed-cohort attestation"
        )
    if launch.get("publication_eligible") is not True:
        raise ValueError(
            "strict final resume source was not publication-eligible at launch"
        )

    original_context = launch.get("final_audit_context_token")
    current_context = audit_receipt.get("context_token")
    if not isinstance(original_context, str) or original_context != current_context:
        raise ValueError(
            "strict final resume audit context differs from the original launch"
        )

    evidence_path = run_dir / "final_audit_evidence.json"
    if not evidence_path.exists():
        raise ValueError("strict final resume has no persisted launch audit evidence")
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    evidence_digest = evidence.get("receipt_sha256")
    original_receipt_digest = launch.get("final_audit_receipt_sha256")
    if (
        not isinstance(evidence_digest, str)
        or evidence_digest != original_receipt_digest
    ):
        raise ValueError(
            "strict final resume launch receipt digest does not match its evidence"
        )
    persisted_receipt = {
        key: value
        for key, value in evidence.items()
        if key not in ("receipt_source", "receipt_sha256")
    }
    if _receipt_content_sha256(persisted_receipt) != evidence_digest:
        raise ValueError("strict final resume persisted launch receipt is corrupted")
    persisted_result = persisted_receipt.get("audit_result")
    if (
        persisted_receipt.get("schema") != RECEIPT_SCHEMA
        or persisted_receipt.get("context_token") != original_context
        or persisted_receipt.get("context") != audit_receipt.get("context")
        or not isinstance(persisted_result, dict)
        or persisted_result.get("reproducibility_audit") != "PASS"
        or persisted_result.get("final_benchmark_status") != "READY"
    ):
        raise ValueError(
            "strict final resume persisted launch evidence is not PASS/READY "
            "for the current context"
        )

    config._controller_seed_cohort_attested = True
    config._final_audit_context_token = current_context
    config._final_audit_receipt_sha256 = current_receipt_digest


def main() -> int:
    config = config_from_args(build_parser().parse_args())
    try:
        audit_receipt = require_final_benchmark_audit(config.run_purpose)
    except FinalAuditGateError as exc:
        print(f"FINAL_BENCHMARK_AUDIT_GATE_FAILED: {exc}", file=sys.stderr)
        return 2
    config._controller_seed_cohort_attested = bool(
        audit_receipt is not None
        and int(audit_receipt.get("origin_pid", os.getpid())) != os.getpid()
    )
    config._final_audit_context_token = (
        audit_receipt.get("context_token") if audit_receipt is not None else None
    )
    config._final_audit_receipt_sha256 = os.environ.get(
        AUDIT_RECEIPT_SHA256_ENV
    )
    if config.run_purpose in ("smoke", "diagnostic"):
        print("NOT A PAPER REPRODUCTION RUN", flush=True)
        print("NOT PUBLICATION-ELIGIBLE", flush=True)
    logger = None
    try:
        inherit_verified_final_resume_attestation(config, audit_receipt)
        if (
            config.run_purpose == "final_benchmark"
            and not config._controller_seed_cohort_attested
        ):
            raise ValueError(
                "a new final benchmark run must be launched by a controller "
                "that attests the complete benchmark seed cohort"
            )
        logger = RunLogger(config)
        run_dir = run_experiment(
            config,
            logger,
            final_audit_receipt=audit_receipt,
        )
        try:
            from plot_results import plot_single_run

            plot_single_run(run_dir)
        except Exception as plot_error:
            logger.logger.warning("automatic plot skipped: %s", plot_error)
    except BaseException as exc:
        traceback.print_exc()
        if logger is not None:
            if logger.resume_committed:
                logger.finish("failed", error=f"{type(exc).__name__}: {exc}")
            else:
                logger.close()
        return 1
    logger.finish("completed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
