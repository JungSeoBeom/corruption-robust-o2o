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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from robust_o2o.config import ExperimentConfig  # noqa: E402
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
    "cal_ql",
    "pessimistic_q_ensemble",
}

CERTIFIED_STRICT_CONDITIONS = (
    ("clean", "none"),
    ("random", "observations"),
    ("random", "actions"),
    ("random", "rewards"),
    ("random", "dynamics"),
    ("adversarial", "observations"),
)
SAVE_RESUME_COVERED_CELLS = {
    ("riql_naive", "random", "observations"),
}


@dataclass
class Check:
    name: str
    passed: bool
    detail: str
    blocking: bool = True


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
            failures.append(f"{algorithm}: {type(exc).__name__}: {exc}")
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
    base = {
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
        try:
            ExperimentConfig(**{**base, field: value})
        except Exception:
            pass
        else:
            failures.append(f"strict override was accepted: {field}={value!r}")
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
        "RPEX/RIQL budgets, hyperparameters, seed cohort, target/task-specific "
        "fixture scope, and profile passed"
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
        Check("save_resume_smoke", passed, resume_detail),
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
        and record.parity_status != "verified"
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
            "rpex_online_phase_rng_parity",
            False,
            "RPEX/RIQL fresh online Adam state now matches the pinned phase "
            "transition, but the fresh-process Torch RNG consumption order "
            "(policy/critic/value and adversarial oracle construction before "
            "the first stochastic action) has no upstream parity fixture",
            blocking=True,
        )
    )
    checks.append(
        Check(
            "rpex_evaluation_rng_parity",
            False,
            "epsilon-greedy sample/mask call order now matches pinned RPEX, "
            "but strict evaluation intentionally uses a separate clean env "
            "and deterministic per-episode reseeding; pinned attack_online.py "
            "reuses the training env and advances its seeded RNG sequentially",
            blocking=True,
        )
    )
    checks.append(
        Check(
            "calql_locomotion_excluded",
            not BASELINE_REPRODUCTION_REGISTRY["cal_ql"].strict_final_eligible,
            BASELINE_REPRODUCTION_REGISTRY["cal_ql"].reproduction_status,
        )
    )
    checks.append(
        Check(
            "pqe_shared_actor_excluded",
            not BASELINE_REPRODUCTION_REGISTRY[
                "pessimistic_q_ensemble"
            ].strict_final_eligible,
            BASELINE_REPRODUCTION_REGISTRY[
                "pessimistic_q_ensemble"
            ].reproduction_status,
        )
    )
    checks.append(
        Check(
            "wsrl_fixed_batch_parity",
            BASELINE_REPRODUCTION_REGISTRY["wsrl"].parity_status
            == "framework_port_verified",
            BASELINE_REPRODUCTION_REGISTRY["wsrl"].parity_status,
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
    required_save_resume_cells = {
        (algorithm, corruption, target)
        for algorithm in strict_final_algorithms()
        for corruption, target in CERTIFIED_STRICT_CONDITIONS
    }
    missing_save_resume_cells = sorted(
        required_save_resume_cells - SAVE_RESUME_COVERED_CELLS
    )
    checks.append(
        Check(
            "save_resume_full_benchmark_coverage",
            not missing_save_resume_cells,
            "covered=RIQL-naive/random observation "
            "(common_budget_diagnostic, offline checkpoint only; online "
            "checkpoint and full resume-state comparison not exercised); "
            "missing="
            + ",".join("/".join(cell) for cell in missing_save_resume_cells),
            blocking=True,
        )
    )
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
            Check("save_resume_smoke", False, "not executed (--static-only)")
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
    """Return distinct status for the eligible subset and the five-baseline goal."""

    eligible_subset_ready = all(
        check.passed or not check.blocking for check in checks
    )
    by_name = {check.name: check for check in checks}
    five_baseline_ready = eligible_subset_ready and all(
        by_name[name].passed
        for name in ("wsrl_fixed_batch_parity", "five_baseline_final_coverage")
    )
    eligible_subset = list(strict_final_algorithms())
    return {
        "checks": [asdict(check) for check in checks],
        "reproducibility_audit": (
            "PASS" if eligible_subset_ready else "FAIL"
        ),
        # Kept for compatibility. Its scope is explicitly recorded below.
        "final_benchmark_status": (
            "READY" if eligible_subset_ready else "NOT READY"
        ),
        "final_benchmark_scope": eligible_subset,
        "eligible_subset_benchmark_status": (
            "READY" if eligible_subset_ready else "NOT READY"
        ),
        "five_baseline_benchmark_status": (
            "READY" if five_baseline_ready else "NOT READY"
        ),
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
    ready = summary["eligible_subset_benchmark_status"] == "READY"
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
            "FINAL BENCHMARK SCOPE: "
            + ",".join(summary["final_benchmark_scope"])
        )
        print(
            "ELIGIBLE-SUBSET BENCHMARK STATUS: "
            f"{summary['eligible_subset_benchmark_status']}"
        )
        print(
            "FINAL BENCHMARK STATUS: "
            f"{summary['final_benchmark_status']}"
        )
        print(
            "FIVE-BASELINE BENCHMARK STATUS: "
            f"{summary['five_baseline_benchmark_status']}"
        )
    return 0 if ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
