#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from robust_o2o.config import ExperimentConfig  # noqa: E402
from robust_o2o.certificates import (  # noqa: E402
    CertificateContextError,
    CertificateSpec,
    build_certificate_context,
    certificate_directory,
    validate_certificate_receipt,
)
from robust_o2o.fidelity import (  # noqa: E402
    BASELINE_REPRODUCTION_REGISTRY,
    REPORTING_RULES,
    RPEX_GOLDEN_FIXTURE_CERTIFICATES,
    STRICT_FINAL_SEEDS,
    STRICT_FINAL_TASKS,
    strict_final_algorithms,
    validate_reproduction_fixture,
)
from robust_o2o.manifest import build_experiment_manifest  # noqa: E402


FIXTURES = ROOT / "tests" / "fixtures"
STRICT_REQUIREMENTS = ROOT / "requirements-rpex-v2.txt"
HOPPER_ATTACK_CHECKPOINT = (
    ROOT.parent
    / "RIQL-main"
    / "pretrained_model"
    / "EDAC"
    / "EDAC_baseline_seed0-hopper-medium-replay-v2"
    / "2999.pt"
)

EXPECTED_BASELINES = {
    "rpex",
    "riql_naive",
    "wsrl",
    "cal_ql_locomotion_adaptation",
    "pqe_shared_actor_approx",
}

CERTIFIED_STRICT_CONDITIONS = (
    ("random", "observations"),
    ("random", "actions"),
    ("random", "rewards"),
    ("random", "dynamics"),
)
SAVE_RESUME_REQUIRED_CELLS = [
    f"{algorithm}/{corruption}/{target}"
    for algorithm in ("rpex", "riql_naive")
    for corruption, target in CERTIFIED_STRICT_CONDITIONS
]
SAVE_RESUME_REQUIRED_STATE = [
    "actor_parameters",
    "critic_parameters",
    "value_parameters",
    "target_critic_parameters",
    "optimizer_states",
    "scheduler_states",
    "replay_buffer_content_and_order",
    "environment_state",
    "current_observation",
    "episode_return_and_length",
    "python_rng",
    "numpy_rng",
    "torch_cpu_cuda_rng",
    "corruption_rng",
    "action_selection_rng",
    "evaluation_counter",
    "metrics_sequence",
    "final_manifest",
]
RPEX_UPSTREAM = BASELINE_REPRODUCTION_REGISTRY["rpex"]
WSRL_UPSTREAM = BASELINE_REPRODUCTION_REGISTRY["wsrl"]
CALQL_UPSTREAM = BASELINE_REPRODUCTION_REGISTRY[
    "cal_ql_locomotion_adaptation"
]
PQE_UPSTREAM = BASELINE_REPRODUCTION_REGISTRY["pqe_shared_actor_approx"]
CERTIFICATE_SPECS: dict[str, CertificateSpec] = {
    "rpex_riql_learner_parity": CertificateSpec(
        "rpex_riql_learner_parity",
        "rpex_riql_learner_parity.json",
        RPEX_UPSTREAM.upstream_repository,
        RPEX_UPSTREAM.upstream_commit,
        (
            "tests/fixtures/rpex_riql_offline_update_v1.json",
            "tests/fixtures/rpex_online_transition_v1.json",
        ),
        required_claims={
            "algorithms": ["rpex", "riql_naive"],
            "parity_status": "end_to_end_verified",
            "scope": "offline_and_online_learner_end_to_end",
        },
    ),
    "rpex_online_rng_parity": CertificateSpec(
        "rpex_online_rng_parity",
        "rpex_online_rng_parity.json",
        RPEX_UPSTREAM.upstream_repository,
        RPEX_UPSTREAM.upstream_commit,
        ("tests/fixtures/rpex_online_rng_trajectory_v1.json",),
        required_claims={
            "algorithms": ["rpex", "riql_naive"],
            "scope": "fresh_process_full_online_rng_trajectory",
        },
    ),
    "rpex_evaluation_schedule_parity": CertificateSpec(
        "rpex_evaluation_schedule_parity",
        "rpex_evaluation_schedule_parity.json",
        RPEX_UPSTREAM.upstream_repository,
        RPEX_UPSTREAM.upstream_commit,
        ("tests/fixtures/rpex_evaluation_schedule_v1.json",),
        required_claims={
            "algorithms": ["rpex", "riql_naive"],
            "scope": "official_training_env_evaluation_rng_schedule",
        },
    ),
    "rpex_riql_reporting_parity": CertificateSpec(
        "rpex_riql_reporting_parity",
        "rpex_riql_reporting_parity.json",
        RPEX_UPSTREAM.upstream_repository,
        RPEX_UPSTREAM.upstream_commit,
        ("tests/fixtures/rpex_reporting_v1.json",),
        required_claims={
            "algorithms": ["rpex", "riql_naive"],
            "scope": "upstream_final_three_reporting",
        },
    ),
    "wsrl_learner_parity": CertificateSpec(
        "wsrl_learner_parity",
        "wsrl_learner_parity.json",
        WSRL_UPSTREAM.upstream_repository,
        WSRL_UPSTREAM.upstream_commit,
        ("tests/fixtures/wsrl_fixed_batch_end_to_end_v1.json",),
        required_claims={
            "algorithm": "wsrl",
            "parity_status": "end_to_end_verified",
            "scope": "official_jax_adapter_or_full_fixed_batch_parity",
        },
    ),
    "wsrl_reporting_parity": CertificateSpec(
        "wsrl_reporting_parity",
        "wsrl_reporting_parity.json",
        WSRL_UPSTREAM.upstream_repository,
        WSRL_UPSTREAM.upstream_commit,
        ("tests/fixtures/wsrl_reporting_v1.json",),
        required_claims={
            "algorithm": "wsrl",
            "scope": "upstream_reporting_end_to_end",
        },
    ),
    "calql_locomotion_parity": CertificateSpec(
        "calql_locomotion_parity",
        "calql_locomotion_parity.json",
        CALQL_UPSTREAM.upstream_repository,
        CALQL_UPSTREAM.upstream_commit,
        ("tests/fixtures/calql_locomotion_end_to_end_v1.json",),
        required_claims={
            "algorithm": "cal_ql_locomotion_adaptation",
            "official_task_support": "d4rl_mujoco_locomotion",
            "parity_status": "official_adapter_verified",
        },
    ),
    "calql_reporting_parity": CertificateSpec(
        "calql_reporting_parity",
        "calql_reporting_parity.json",
        CALQL_UPSTREAM.upstream_repository,
        CALQL_UPSTREAM.upstream_commit,
        ("tests/fixtures/calql_reporting_v1.json",),
        required_claims={
            "algorithm": "cal_ql_locomotion_adaptation",
            "scope": "upstream_reporting_end_to_end",
        },
    ),
    "pqe_independent_ensemble_parity": CertificateSpec(
        "pqe_independent_ensemble_parity",
        "pqe_independent_ensemble_parity.json",
        PQE_UPSTREAM.upstream_repository,
        PQE_UPSTREAM.upstream_commit,
        ("tests/fixtures/pqe_independent_ensemble_end_to_end_v1.json",),
        required_claims={
            "algorithm": "pqe_shared_actor_approx",
            "ensemble_size": 5,
            "independent_actors_and_twin_critics": True,
            "parity_status": "end_to_end_verified",
        },
    ),
    "pqe_reporting_parity": CertificateSpec(
        "pqe_reporting_parity",
        "pqe_reporting_parity.json",
        PQE_UPSTREAM.upstream_repository,
        PQE_UPSTREAM.upstream_commit,
        ("tests/fixtures/pqe_reporting_v1.json",),
        required_claims={
            "algorithm": "pqe_shared_actor_approx",
            "scope": "upstream_reporting_end_to_end",
        },
    ),
    "strict_runtime_random_fixture_alignment": CertificateSpec(
        "strict_runtime_random_fixture_alignment",
        "strict_runtime_random_fixture_alignment.json",
        RPEX_UPSTREAM.upstream_repository,
        RPEX_UPSTREAM.upstream_commit,
        (
            "tests/fixtures/rpex_random_corruption_v2.json",
            "tests/fixtures/rpex_riql_offline_update_v1.json",
            "tests/fixtures/rpex_online_transition_v1.json",
        ),
        required_claims={
            "runtime_profile": "requirements-rpex-v2.txt",
            "scope": "random_and_learner_fixtures",
        },
    ),
    "strict_runtime_adversarial_fixture_alignment": CertificateSpec(
        "strict_runtime_adversarial_fixture_alignment",
        "strict_runtime_adversarial_fixture_alignment.json",
        RPEX_UPSTREAM.upstream_repository,
        RPEX_UPSTREAM.upstream_commit,
        (
            "tests/fixtures/rpex_adversarial_observation_offline_v2.json",
            "tests/fixtures/rpex_adversarial_observation_online_v2.json",
        ),
        required_claims={
            "runtime_profile": "requirements-rpex-v2.txt",
            "scope": "adversarial_observation_fixtures",
        },
    ),
    "random_corruption_end_to_end": CertificateSpec(
        "random_corruption_end_to_end",
        "random_corruption_end_to_end.json",
        RPEX_UPSTREAM.upstream_repository,
        RPEX_UPSTREAM.upstream_commit,
        ("tests/fixtures/rpex_random_corruption_v2.json",),
        required_claims={
            "algorithms": ["rpex", "riql_naive"],
            "targets": ["observations", "actions", "rewards", "dynamics"],
            "scope": "offline_and_online_end_to_end",
        },
    ),
    "save_resume_coverage": CertificateSpec(
        "save_resume_coverage",
        "save_resume_coverage.json",
        RPEX_UPSTREAM.upstream_repository,
        RPEX_UPSTREAM.upstream_commit,
        ("tests/fixtures/rpex_riql_save_resume_v1.json",),
        required_claims={
            "cells": SAVE_RESUME_REQUIRED_CELLS,
            "state_fields": SAVE_RESUME_REQUIRED_STATE,
            "fresh_process_reload": True,
            "comparison": "uninterrupted_vs_resumed",
        },
    ),
    "adversarial_observation_end_to_end": CertificateSpec(
        "adversarial_observation_end_to_end",
        "adversarial_observation_end_to_end.json",
        RPEX_UPSTREAM.upstream_repository,
        RPEX_UPSTREAM.upstream_commit,
        (
            "tests/fixtures/rpex_adversarial_observation_offline_v2.json",
            "tests/fixtures/rpex_adversarial_observation_online_v2.json",
        ),
        required_claims={
            "algorithms": ["rpex", "riql_naive"],
            "target": "observations",
            "scope": "offline_and_online_full_wrapper_end_to_end",
        },
    ),
    "strict_environment_preflight": CertificateSpec(
        "strict_environment_preflight",
        "strict_environment_preflight.json",
        RPEX_UPSTREAM.upstream_repository,
        RPEX_UPSTREAM.upstream_commit,
        required_claims={
            "platform_system": "Linux",
            "platform_machine": "x86_64",
            "tasks": list(STRICT_FINAL_TASKS),
            "scope": "strict_preflight_and_dataset_environment_smoke",
        },
    ),
}
CERTIFICATE_CAPABILITIES: dict[str, tuple[str, ...]] = {
    "rpex_riql_learner": (
        "rpex_riql_learner_parity",
        "rpex_online_rng_parity",
        "rpex_evaluation_schedule_parity",
        "rpex_riql_reporting_parity",
    ),
    "five_baseline_learners": (
        "wsrl_learner_parity",
        "wsrl_reporting_parity",
        "calql_locomotion_parity",
        "calql_reporting_parity",
        "pqe_independent_ensemble_parity",
        "pqe_reporting_parity",
    ),
    "random_corruption": (
        "strict_runtime_random_fixture_alignment",
        "random_corruption_end_to_end",
    ),
    "adversarial_corruption": (
        "strict_runtime_adversarial_fixture_alignment",
        "adversarial_observation_end_to_end",
    ),
    "save_resume": ("save_resume_coverage",),
    "strict_environment": ("strict_environment_preflight",),
}


@dataclass
class Check:
    name: str
    passed: bool
    detail: str
    blocking: bool = True
    evidence: dict[str, Any] | None = None


def certificate_checks() -> list[Check]:
    """Validate every required receipt against one immutable current context."""

    fixture_paths = {
        path
        for spec in CERTIFICATE_SPECS.values()
        for path in spec.required_fixture_paths
    }
    try:
        context = build_certificate_context(ROOT, fixture_paths=fixture_paths)
        directory = certificate_directory(ROOT)
    except CertificateContextError as exc:
        return [
            Check(
                "certificate_context",
                False,
                str(exc),
                evidence={"status": "invalid", "valid": False},
            )
        ]
    validations = [
        validate_certificate_receipt(
            spec,
            certificate_dir=directory,
            context=context,
        )
        for spec in CERTIFICATE_SPECS.values()
    ]
    checks = [
        Check(
            f"certificate:{validation.certificate_id}",
            validation.valid,
            f"status={validation.status}; {validation.detail}",
            blocking=(
                validation.certificate_id
                not in CERTIFICATE_CAPABILITIES["adversarial_corruption"]
            ),
            evidence=validation.to_dict(),
        )
        for validation in validations
    ]
    checks.append(
        Check(
            "repository_clean",
            context.repository.get("clean") is True,
            (
                f"commit={context.repository.get('commit')} "
                f"tree={context.repository.get('tree_sha256')} "
                f"source_tree={context.repository.get('source_tree_sha256')}"
                if context.repository.get("clean") is True
                else "tracked or untracked working-tree changes are present"
            ),
            evidence=dict(context.repository),
        )
    )
    return checks


def fixture_check(filename: str, fixture_id: str) -> Check:
    path = FIXTURES / filename
    try:
        payload = validate_reproduction_fixture(path, fixture_id)
    except ValueError as exc:
        return Check(f"fixture:{fixture_id}", False, str(exc))
    certificate = RPEX_GOLDEN_FIXTURE_CERTIFICATES[fixture_id]
    detail = (
        f"source={payload.get('upstream_commit')} "
        f"content_sha256={certificate.content_sha256} "
        f"source_sha256={certificate.upstream_source_sha256} "
        f"python={payload.get('python_version')} "
        f"numpy={payload.get('numpy_version')} "
        f"torch={payload.get('pytorch_version')}"
    )
    if fixture_id == "rpex_adversarial_core_v1":
        detail += (
            f" oracle_source_sha256={certificate.oracle_source_sha256}; "
            "scope=observation-target optimizer core only (offline+online); "
            "action/reward/dynamics fixtures are not certified; upstream "
            "wrapper bugs remain"
        )
    return Check(f"fixture:{fixture_id}", True, detail)


def fixture_runtime_alignment_check() -> Check:
    """Require upstream fixtures to be generated with the strict numeric stack."""

    pinned: dict[str, str] = {}
    for raw_line in STRICT_REQUIREMENTS.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line.startswith("numpy=="):
            pinned["numpy"] = line.split("==", 1)[1]
        elif line.startswith("torch=="):
            pinned["torch"] = line.split("==", 1)[1]
    missing = sorted({"numpy", "torch"} - set(pinned))
    if missing:
        return Check(
            "fixture_strict_runtime_alignment",
            False,
            f"strict requirements have no exact pins for: {missing}",
        )

    mismatches: list[str] = []
    for fixture_id, certificate in RPEX_GOLDEN_FIXTURE_CERTIFICATES.items():
        if certificate.numpy_version != pinned["numpy"]:
            mismatches.append(
                f"{fixture_id}:numpy={certificate.numpy_version} "
                f"required={pinned['numpy']}"
            )
        if certificate.pytorch_version != pinned["torch"]:
            mismatches.append(
                f"{fixture_id}:torch={certificate.pytorch_version} "
                f"required={pinned['torch']}"
            )
    return Check(
        "fixture_strict_runtime_alignment",
        not mismatches,
        "all certified fixtures match the strict NumPy/Torch pins"
        if not mismatches
        else "; ".join(mismatches),
    )


def adversarial_checkpoint_check(
    path: Path = HOPPER_ATTACK_CHECKPOINT,
) -> Check:
    certificate = RPEX_GOLDEN_FIXTURE_CERTIFICATES[
        "rpex_adversarial_core_v1"
    ]
    expected = certificate.checkpoint_sha256
    if expected is None:
        return Check(
            "adversarial_checkpoint_certificate",
            False,
            "certificate has no checkpoint SHA256",
        )
    if not path.is_file():
        return Check(
            "adversarial_checkpoint_certificate",
            False,
            f"missing pinned checkpoint: {path}",
        )
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    return Check(
        "adversarial_checkpoint_certificate",
        actual == expected,
        f"path={path} expected={expected} actual={actual}",
    )


def manifest_check() -> Check:
    config = ExperimentConfig(
        "rpex",
        "hopper-medium-replay-v2",
        corruption="random",
        corruption_target="observations",
        seed=0,
        implementation_profile="official_code_reference",
    )
    resolved = config.to_dict()
    resolved.update(
        repository_commit="test",
        repository_dirty=True,
        dataset_id=config.env_name,
        dataset_sha256="dataset",
        runtime_package_versions={"python": sys.version.split()[0]},
        offline_corruption={
            "selected_transition_count": 1,
            "selected_transition_indices_sha256": "indices",
            "corruption_value_sha256": "values",
            "final_artifact_sha256": "artifact",
            "rng_implementation": "numpy.random.RandomState",
        },
    )
    manifest = build_experiment_manifest(resolved)
    required = {
        "repository_commit",
        "repository_dirty",
        "algorithm",
        "paper_title",
        "upstream_repository",
        "upstream_commit",
        "implementation_profile",
        "reproduction_status",
        "dataset_name",
        "dataset_sha256",
        "environment_versions",
        "learner_seed",
        "corruption_seed",
        "corruption",
        "corruption_target",
        "corruption_rate",
        "corruption_range",
        "corruption_fixture_id",
        "corruption_fixture_verified",
        "offline_updates",
        "requested_online_steps",
        "actual_online_steps",
        "online_budget_semantics",
        "episode_boundary_overshoot",
        "evaluation_interval",
        "evaluation_episodes",
        "reporting_rule",
        "publication_eligible",
        "replay_sampling_profile",
        "offline_replay_sampler",
        "online_replay_sampler",
        "replay_seed_mapping",
        "replay_rng_parity_verified",
        "online_optimizer_transition",
        "online_phase_rng_parity_verified",
        "evaluation_action_sampling",
        "evaluation_seed_schedule",
        "evaluation_protocol_parity_verified",
    }
    missing = sorted(required - set(manifest))
    return Check(
        "manifest_provenance",
        not missing,
        "all required fields present" if not missing else f"missing={missing}",
    )


def strict_config_contract_check() -> Check:
    failures: list[str] = []
    for algorithm in ("rpex", "riql_naive"):
        declared_eligible = BASELINE_REPRODUCTION_REGISTRY[
            algorithm
        ].strict_final_eligible
        try:
            config = ExperimentConfig(
                algorithm,
                "hopper-medium-replay-v2",
                corruption="random",
                corruption_target="observations",
                suite_profile="primary_research_benchmark",
                run_purpose="final_benchmark",
                benchmark_seed_set=STRICT_FINAL_SEEDS,
            )
        except Exception as exc:
            if declared_eligible:
                failures.append(f"{algorithm}: {type(exc).__name__}: {exc}")
            continue
        if not declared_eligible:
            failures.append(
                f"uncertified baseline was accepted in final mode: {algorithm}"
            )
            continue
        actual = (
            config.offline_steps,
            config.online_steps,
            config.offline_corruption_rate,
            config.online_corruption_rate,
            config.corruption_range,
            config.corruption_seed,
            config.implementation_profile,
            config.replay_seed,
            config.to_dict()["replay_sampling_profile"],
            config.to_dict()["replay_rng_parity_verified"],
        )
        required = (
            2_000_001,
            1_000_001,
            0.3,
            0.5,
            1.0,
            config.seed,
            "official_code_reference",
            config.seed,
            "rpex_official_global_rng",
            True,
        )
        if actual != required:
            failures.append(f"{algorithm}: actual={actual!r} required={required!r}")
    base: dict[str, Any] = {
        "algorithm": "rpex",
        "env_name": "hopper-medium-replay-v2",
        "corruption": "random",
        "corruption_target": "observations",
        "suite_profile": "primary_research_benchmark",
        "run_purpose": "final_benchmark",
        "benchmark_seed_set": STRICT_FINAL_SEEDS,
    }
    for field, value in {
        "benchmark_seed_set": (0,),
        "batch_size": 128,
        "hidden_dim": 128,
        "discount": 0.95,
        "normalize_states": False,
        "offline_ratio": 0.5,
        "replay_seed": 1,
        "initialize_from_checkpoint": "/tmp/unverified.pt",
    }.items():
        if BASELINE_REPRODUCTION_REGISTRY["rpex"].strict_final_eligible:
            try:
                ExperimentConfig(**{**base, field: value})
            except Exception:
                pass
            else:
                failures.append(
                    f"strict override was accepted: {field}={value!r}"
                )
    for target in ("actions", "rewards", "dynamics"):
        try:
            ExperimentConfig(
                **{
                    **base,
                    "corruption": "adversarial",
                    "corruption_target": target,
                }
            )
        except Exception:
            pass
        else:
            failures.append(
                "uncertified adversarial target was accepted: " + target
            )
    for env_name in (
        "halfcheetah-medium-replay-v2",
        "walker2d-medium-replay-v2",
    ):
        try:
            ExperimentConfig(
                **{
                    **base,
                    "env_name": env_name,
                    "corruption": "adversarial",
                    "corruption_target": "observations",
                }
            )
        except Exception:
            pass
        else:
            failures.append(
                "Hopper-only adversarial fixture was accepted for " + env_name
            )
    return Check(
        "strict_config_contract",
        not failures,
        "uncertified baselines are rejected; any future certified RPEX/RIQL "
        "configuration must satisfy exact budgets, hyperparameters, seed "
        "cohort, target/task-specific fixture scope, and profile"
        if not failures
        else " | ".join(failures),
    )


def run_unittest(name: str) -> Check:
    result = subprocess.run(
        [sys.executable, "-m", "unittest", name, "-q"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    combined_output = "\n".join((result.stdout, result.stderr))
    skipped_match = re.search(r"skipped=(\d+)", combined_output)
    skipped = int(skipped_match.group(1)) if skipped_match else 0
    passed = result.returncode == 0 and skipped == 0
    if passed:
        detail = "passed"
    elif result.returncode == 0 and skipped:
        detail = f"skipped={skipped}; skipped audit tests are not a pass"
    elif result.stderr.strip():
        detail = result.stderr.strip().splitlines()[-1]
    else:
        detail = f"returncode={result.returncode}"
    return Check(f"test:{name}", passed, detail)


def run_strict_preflight() -> tuple[Check, Check]:
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "preflight_strict.py")],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    passed = result.returncode == 0
    if passed:
        d4rl_detail = (
            "strict host/package checks and dataset/environment metadata for "
            "the three D4RL-v2 tasks passed; the executable diagnostic also "
            "completed"
        )
        resume_detail = (
            "passed only for RIQL-naive + random observation corruption under "
            "common_budget_diagnostic (10 offline updates, 20 online steps, "
            "one evaluation episode); an offline checkpoint was resumed and "
            "agent/normalizer/evaluation/online-corruption-audit matched, but "
            "online-checkpoint resume and full resume-state equality were not "
            "exercised; cache miss/hit matched"
        )
    else:
        lines = [
            line.strip()
            for line in (result.stderr + "\n" + result.stdout).splitlines()
            if line.strip()
        ]
        failure = lines[-1] if lines else f"returncode={result.returncode}"
        d4rl_detail = failure
        resume_detail = (
            "RIQL-naive + random observation corruption diagnostic smoke was "
            f"not completed: {failure}"
        )
    return (
        Check("strict_d4rl_preflight", passed, d4rl_detail),
        Check(
            "save_resume_smoke",
            passed,
            resume_detail,
            blocking=False,
        ),
    )


def audit(*, static_only: bool) -> list[Check]:
    checks: list[Check] = []
    expected = EXPECTED_BASELINES
    checks.append(
        Check(
            "baseline_registry",
            set(BASELINE_REPRODUCTION_REGISTRY) == expected,
            ",".join(BASELINE_REPRODUCTION_REGISTRY),
        )
    )
    exact_overclaims = [
        name
        for name, record in BASELINE_REPRODUCTION_REGISTRY.items()
        if record.reproduction_status == "exact_upstream_port"
        and record.parity_status
        not in {"end_to_end_verified", "official_adapter_verified"}
    ]
    checks.append(
        Check(
            "no_unverified_exact_labels",
            not exact_overclaims,
            f"overclaims={exact_overclaims}" if exact_overclaims else "none",
        )
    )
    missing_commits = [
        name
        for name, record in BASELINE_REPRODUCTION_REGISTRY.items()
        if not record.upstream_commit
    ]
    checks.append(
        Check(
            "source_commits",
            not missing_commits,
            f"missing={missing_commits}" if missing_commits else "all pinned",
        )
    )
    checks.extend(
        (
            fixture_check(
                "rpex_random_corruption_v1.json",
                "rpex_random_corruption_v1",
            ),
            fixture_check(
                "rpex_adversarial_core_v1.json",
                "rpex_adversarial_core_v1",
            ),
        )
    )
    checks.append(fixture_runtime_alignment_check())
    checks.append(adversarial_checkpoint_check())
    checks.append(
        Check(
            "reporting_registry",
            expected.issubset(REPORTING_RULES),
            ",".join(REPORTING_RULES),
        )
    )
    checks.append(
        Check(
            "final_seed_contract",
            STRICT_FINAL_SEEDS == (0, 1, 2, 3, 4),
            f"required_seeds={STRICT_FINAL_SEEDS}",
        )
    )
    checks.append(
        Check(
            "strict_task_contract",
            STRICT_FINAL_TASKS
            == (
                "hopper-medium-replay-v2",
                "halfcheetah-medium-replay-v2",
                "walker2d-medium-replay-v2",
            ),
            f"required_tasks={STRICT_FINAL_TASKS}",
        )
    )
    checks.append(strict_config_contract_check())
    checks.append(
        Check(
            "rpex_riql_registry_eligibility",
            set(strict_final_algorithms()) == {"rpex", "riql_naive"},
            "strict_final_algorithms=" + repr(strict_final_algorithms()),
        )
    )
    checks.append(
        Check(
            "calql_locomotion_excluded",
            not BASELINE_REPRODUCTION_REGISTRY[
                "cal_ql_locomotion_adaptation"
            ].strict_final_eligible,
            BASELINE_REPRODUCTION_REGISTRY[
                "cal_ql_locomotion_adaptation"
            ].reproduction_status,
        )
    )
    checks.append(
        Check(
            "pqe_shared_actor_excluded",
            not BASELINE_REPRODUCTION_REGISTRY[
                "pqe_shared_actor_approx"
            ].strict_final_eligible,
            BASELINE_REPRODUCTION_REGISTRY[
                "pqe_shared_actor_approx"
            ].reproduction_status,
        )
    )
    checks.append(
        Check(
            "wsrl_fixed_batch_parity",
            BASELINE_REPRODUCTION_REGISTRY["wsrl"].parity_status
            in {"end_to_end_verified", "official_adapter_verified"}
            and REPORTING_RULES["wsrl"].verified,
            "parity="
            + BASELINE_REPRODUCTION_REGISTRY["wsrl"].parity_status
            + f" reporting_verified={REPORTING_RULES['wsrl'].verified}",
            blocking=False,
        )
    )
    checks.append(
        Check(
            "five_baseline_final_coverage",
            set(strict_final_algorithms()) == expected,
            f"strict_final_algorithms={strict_final_algorithms()}",
            blocking=False,
        )
    )
    checks.extend(certificate_checks())
    checks.append(manifest_check())
    run55_text = (ROOT / "run_55_experiment.py").read_text(encoding="utf-8")
    checks.append(
        Check(
            "adversarial_suite_defined",
            "ADVERSARIAL_SETTINGS" in run55_text
            and '"--corruption-suite"' in run55_text,
            "clean/random/adversarial/all CLI present",
        )
    )

    if static_only:
        checks.append(
            Check(
                "strict_d4rl_preflight",
                False,
                "not executed (--static-only)",
            )
        )
        checks.append(
            Check(
                "save_resume_smoke",
                False,
                "not executed (--static-only)",
                blocking=False,
            )
        )
    else:
        checks.extend(run_strict_preflight())
        checks.extend(
            (
                run_unittest(
                    "tests.test_reproducibility_audit.RPEXGoldenFixtureTest."
                    "test_random_targets_match_pinned_upstream_fixture"
                ),
                run_unittest(
                    "tests.test_reproducibility_audit.RPEXGoldenFixtureTest."
                    "test_online_random_call_order_matches_upstream_fixture"
                ),
                run_unittest(
                    "tests.test_reproducibility_audit.RPEXGoldenFixtureTest."
                    "test_adversarial_offline_and_online_core_match_upstream_fixture"
                ),
                run_unittest(
                    "tests.test_fidelity_profiles.FidelityProfileTest."
                    "test_200_updates_equal_100_checkpoint_plus_100"
                ),
                run_unittest(
                    "tests.test_reproducibility_audit.RPEXGoldenFixtureTest."
                    "test_official_replay_samplers_match_pinned_upstream_rngs"
                ),
            )
        )
    return checks


def summarize_audit(checks: list[Check]) -> dict[str, object]:
    """Compute capability statuses without promoting a partial PASS to READY."""

    by_name = {check.name: check for check in checks}

    def passed(name: str) -> bool:
        check = by_name.get(name)
        return bool(check is not None and check.passed)

    certificate_statuses: dict[str, dict[str, Any]] = {}
    for certificate_id in CERTIFICATE_SPECS:
        check = by_name.get(f"certificate:{certificate_id}")
        if check is None or check.evidence is None:
            certificate_statuses[certificate_id] = {
                "certificate_id": certificate_id,
                "status": "invalid",
                "valid": False,
                "detail": "certificate validation did not run",
            }
        else:
            certificate_statuses[certificate_id] = dict(check.evidence)

    verified_certificate_ids = sorted(
        certificate_id
        for certificate_id, validation in certificate_statuses.items()
        if validation.get("valid") is True
    )
    capability_certificates = {
        capability: {
            "required_certificate_ids": list(required_ids),
            "verified": all(
                certificate_id in verified_certificate_ids
                for certificate_id in required_ids
            ),
        }
        for capability, required_ids in CERTIFICATE_CAPABILITIES.items()
    }

    common_integrity_ready = all(
        passed(name)
        for name in (
            "baseline_registry",
            "no_unverified_exact_labels",
            "source_commits",
            "reporting_registry",
            "final_seed_contract",
            "strict_task_contract",
            "strict_config_contract",
            "manifest_provenance",
            "repository_clean",
        )
    )
    subset_ready = (
        common_integrity_ready
        and passed("rpex_riql_registry_eligibility")
        and capability_certificates["rpex_riql_learner"]["verified"]
    )
    random_ready = (
        common_integrity_ready
        and capability_certificates["random_corruption"]["verified"]
    )
    strict_adversarial_enabled = any(
        corruption == "adversarial"
        for corruption, _target in CERTIFIED_STRICT_CONDITIONS
    )
    adversarial_ready = (
        common_integrity_ready
        and capability_certificates["adversarial_corruption"]["verified"]
    )
    adversarial_status = (
        "EXCLUDED"
        if not strict_adversarial_enabled
        else "READY"
        if adversarial_ready
        else "NOT READY"
    )
    save_resume_ready = (
        common_integrity_ready
        and capability_certificates["save_resume"]["verified"]
    )
    strict_environment_ready = (
        common_integrity_ready
        and capability_certificates["strict_environment"]["verified"]
        and passed("strict_d4rl_preflight")
    )
    five_baseline_ready = (
        subset_ready
        and passed("wsrl_fixed_batch_parity")
        and passed("five_baseline_final_coverage")
        and capability_certificates["five_baseline_learners"]["verified"]
    )
    final_ready = (
        subset_ready
        and five_baseline_ready
        and random_ready
        and save_resume_ready
        and strict_environment_ready
        and adversarial_status in {"READY", "EXCLUDED"}
    )

    def readiness(value: bool) -> str:
        return "READY" if value else "NOT READY"

    return {
        "checks": [asdict(check) for check in checks],
        "certificate_statuses": certificate_statuses,
        "verified_certificate_ids": verified_certificate_ids,
        "certificate_capabilities": capability_certificates,
        "reproducibility_audit": "PASS" if final_ready else "FAIL",
        "rpex_riql_eligible_subset_status": readiness(subset_ready),
        "five_baseline_status": readiness(five_baseline_ready),
        "random_corruption_status": readiness(random_ready),
        "adversarial_corruption_status": adversarial_status,
        "save_resume_status": readiness(save_resume_ready),
        "strict_environment_status": readiness(strict_environment_ready),
        "final_benchmark_status": readiness(final_ready),
        "final_benchmark_scope": list(BASELINE_REPRODUCTION_REGISTRY),
        # Compatibility aliases carry the same explicitly named scope.
        "eligible_subset_benchmark_status": readiness(subset_ready),
        "five_baseline_benchmark_status": readiness(five_baseline_ready),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--static-only",
        action="store_true",
        help="skip strict D4RL and executable smoke checks (cannot return READY)",
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    checks = audit(static_only=args.static_only)
    summary = summarize_audit(checks)
    ready = summary["final_benchmark_status"] == "READY"
    if args.json:
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    else:
        for check in checks:
            if check.passed:
                label = "PASS"
            elif check.blocking:
                label = "FAIL"
            else:
                label = "WARN"
            print(
                f"[{label}] {check.name}: {check.detail}"
            )
        print(f"REPRODUCIBILITY AUDIT: {summary['reproducibility_audit']}")
        print(
            "RPEX/RIQL ELIGIBLE SUBSET STATUS: "
            f"{summary['rpex_riql_eligible_subset_status']}"
        )
        print(
            "FIVE-BASELINE STATUS: "
            f"{summary['five_baseline_status']}"
        )
        print(
            "RANDOM CORRUPTION STATUS: "
            f"{summary['random_corruption_status']}"
        )
        print(
            "ADVERSARIAL CORRUPTION STATUS: "
            f"{summary['adversarial_corruption_status']}"
        )
        print(
            "SAVE/RESUME STATUS: "
            f"{summary['save_resume_status']}"
        )
        print(
            "STRICT ENVIRONMENT STATUS: "
            f"{summary['strict_environment_status']}"
        )
        print(
            "FINAL BENCHMARK STATUS: "
            f"{summary['final_benchmark_status']}"
        )
    return 0 if ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
