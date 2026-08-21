from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


IMPLEMENTATION_PROFILES = (
    "research_benchmark",
    "official_code_reference",
    "paper_reference",
    "common_budget_robustness",
    "locomotion_port",
    "legacy_current",
    "experimental_approximation",
)
IMPLEMENTATION_FIDELITIES = (
    # Retained only so historical manifests remain readable. New runs may not
    # select this label without an automated upstream parity certificate.
    "exact_upstream_port",
    "source_aligned_port",
    "framework_port_unverified",
    "framework_port_verified",
    "end_to_end_verified",
    "official_adapter_verified",
    "paper_code_conflict",
    "task_port",
    "diagnostic_extension",
    "approximation",
    "legacy_unknown",
)
PARITY_STATUSES = (
    "unverified",
    "formula_only",
    "corruption_only",
    "fixed_batch_partial",
    "end_to_end_verified",
    "official_adapter_verified",
)
STRICT_ELIGIBLE_PARITY_STATUSES = frozenset(
    {"end_to_end_verified", "official_adapter_verified"}
)
STRICT_ELIGIBLE_REPRODUCTION_STATUSES = frozenset(
    {
        "exact_upstream_port",
        "framework_port_verified",
        "end_to_end_verified",
        "official_adapter_verified",
    }
)
SUITE_PROFILES = (
    "research_benchmark",
    "primary_research_benchmark",
    "common_budget_diagnostic",
    # Backward-compatible names retained for existing commands/results.
    "method_fidelity",
    "common_budget_robustness",
)
RUN_PURPOSES = (
    "smoke",
    "diagnostic",
    "research_benchmark",
    "paper_reproduction",
    "final_benchmark",
)

# Canonical algorithms for new research-benchmark runs.  Cal-QL remains
# explicitly labelled as a locomotion adaptation and PQE as a D4RL-v2 port;
# those task-scope qualifications do not make either method optional.
MAIN_BASELINES = (
    "rpex",
    "riql_naive",
    "wsrl",
    "cal_ql",
    "pessimistic_q_ensemble",
)
# Kept as empty imports so older launcher/reporting code fails closed instead
# of breaking at import time while it migrates to the five-main-baseline
# contract.
OPTIONAL_ADAPTED_BASELINES: tuple[str, ...] = ()
OPTIONAL_APPROXIMATION_BASELINES: tuple[str, ...] = ()
OPTIONAL_BASELINES: tuple[str, ...] = ()
RESEARCH_BASELINES = MAIN_BASELINES

# Read-only normalization for manifests produced before the canonical names
# were introduced.  Launch configuration must not use this mapping: old names
# are not registry entries and must never appear in a new run directory.
HISTORICAL_RESULT_ALGORITHM_ALIASES: Mapping[str, str] = {
    "cal_ql_locomotion_adaptation": "cal_ql",
    "pqe_shared_actor_approx": "pessimistic_q_ensemble",
}
BENCHMARK_ROLES = ("main", "optional_adapted", "optional_diagnostic", "diagnostic")

ONLINE_REPLAY_PROFILES = (
    "official_code_online_only",
    "paper_offline_online_mixture",
    "fixed_offline_online_mixture",
    "dynamic_offline_online_mixture",
    "balanced_density_replay",
)
EVALUATION_POLICY_PROFILES = (
    "official_code_epsilon_switching",
    "paper_greedy_highest_weight",
    "deterministic_diagnostic",
)
ATTACK_TIMINGS = (
    "official_code_post_transition_replay_poisoning",
    "paper_pre_action_sensor_actuator",
)
RANDOM_ATTACK_SEMANTICS = (
    "post_transition_replay_poisoning",
    "pre_action_sensor_actuator_corruption",
)
MIXED_CORRUPTION_PROFILES = ("generic_partitioned_mixed", "rpex_paper_mixed")
ACTION_EXECUTION_PROFILES = (
    "official_algorithm_behavior",
    "clip_to_action_space",
)
LEGACY_ACTION_EXECUTION_PROFILE_ALIASES = {
    "official_unclipped": "official_algorithm_behavior",
    "environment_clip": "clip_to_action_space",
}
POLICY_EXTRACTIONS = ("awr", "align_iql")
TASK_PROFILES = (
    "official_supported_task",
    "d4rl_locomotion_adaptation",
    "d4rl_v2_port",
    # Historical manifest value retained for read compatibility only.
    "d4rl_locomotion_port",
)
ADVERSARIAL_ATTACK_PROFILES = (
    "rpex_official_adam",
    "experimental_sign_pgd",
)
ONLINE_CORRUPTION_SCALE_PROFILES = (
    "rpex_official_code",
    "dataset_std_scaled_extension",
)
WSRL_ENTROPY_PROFILES = (
    "official_negative_action_dim",
    "legacy_zero",
)
OFFLINE_ADVERSARIAL_REWARD_RULES = ("official_sign_flip",)
ONLINE_ADVERSARIAL_REWARD_RULES = (
    "official_uniform_replacement",
    "experimental_scaled_sign_flip",
)


UPSTREAM_COMMITS: Mapping[str, str] = {
    "rpex": "35da71ee5151b6179d21b9a2b4ce1b6408aedd04",
    "riql_naive": "35da71ee5151b6179d21b9a2b4ce1b6408aedd04",
    "riql_pex": "35da71ee5151b6179d21b9a2b4ce1b6408aedd04",
    "wsrl": "ad4dc1248a138bc15d6e053f2d1dba1b8cfbaca2",
    "cal_ql": "ac6eafec22e8d60836573e1f488c7f626ce8a77e",
    "pessimistic_q_ensemble": "6f298fa9ef040d725067d0f2775022bd2900d635",
}


@dataclass(frozen=True)
class BaselineReproductionRecord:
    paper_title: str
    upstream_repository: str
    upstream_commit: str
    official_task_support: str
    implementation_type: str
    reproduction_status: str
    parity_status: str
    strict_final_eligible: bool
    remaining_deviation: str
    benchmark_role: str = "diagnostic"
    main_table_eligible: bool = False
    display_name: str = ""
    task_scope: str = "unspecified"
    upstream_task_version: str | None = None
    benchmark_task_version: str | None = None
    offline_compute_multiplier: float = 1.0

    def __post_init__(self) -> None:
        if self.benchmark_role not in BENCHMARK_ROLES:
            raise ValueError(
                f"unknown benchmark_role {self.benchmark_role!r}; "
                f"choose from {BENCHMARK_ROLES}"
            )
        if self.main_table_eligible != (self.benchmark_role == "main"):
            raise ValueError(
                "main_table_eligible must be true exactly for benchmark_role='main'"
            )
        if self.parity_status not in PARITY_STATUSES:
            raise ValueError(
                f"unknown parity_status {self.parity_status!r}; "
                f"choose from {PARITY_STATUSES}"
            )
        if self.strict_final_eligible and not baseline_record_is_strict_eligible(
            self, require_declaration=False
        ):
            raise ValueError(
                "strict_final_eligible requires end_to_end_verified or "
                "official_adapter_verified parity and a non-diagnostic, "
                "non-port reproduction status"
            )


def baseline_record_is_strict_eligible(
    record: BaselineReproductionRecord,
    *,
    require_declaration: bool = True,
) -> bool:
    """Return strict eligibility only for an explicitly verified baseline.

    The registry declaration is necessary but never sufficient: handwritten
    source alignment, framework/task ports, and approximations remain excluded
    even if their boolean is accidentally changed.
    """

    return bool(
        (record.strict_final_eligible or not require_declaration)
        and record.parity_status in STRICT_ELIGIBLE_PARITY_STATUSES
        and record.reproduction_status in STRICT_ELIGIBLE_REPRODUCTION_STATUSES
    )


# This registry is deliberately conservative.  A source-aligned handwritten
# port is not promoted to "exact" merely because its formulas resemble the
# public implementation.  The strict controller consumes this registry rather
# than inferring eligibility from a user-facing profile name.
BASELINE_REPRODUCTION_REGISTRY: Mapping[str, BaselineReproductionRecord] = {
    "rpex": BaselineReproductionRecord(
        paper_title=(
            "RPEX: Robust Policy Expansion for Offline-to-Online RL under Diverse "
            "Data Corruption"
        ),
        upstream_repository="https://github.com/felix-thu/RPEX",
        upstream_commit=UPSTREAM_COMMITS["rpex"],
        official_task_support="D4RL MuJoCo locomotion v2",
        implementation_type="source_aligned_port",
        reproduction_status="source_aligned_port",
        parity_status="fixed_batch_partial",
        strict_final_eligible=False,
        remaining_deviation=(
            "no end-to-end fixed-batch optimizer parity certificate for the "
            "complete learner"
        ),
        benchmark_role="main",
        main_table_eligible=True,
        display_name="RPEX",
        task_scope="d4rl_locomotion_v2",
        upstream_task_version="v2",
        benchmark_task_version="v2",
    ),
    "riql_naive": BaselineReproductionRecord(
        paper_title="Towards Robust Offline Reinforcement Learning under Diverse Data Corruption",
        upstream_repository="https://github.com/felix-thu/RPEX",
        upstream_commit=UPSTREAM_COMMITS["riql_naive"],
        official_task_support="D4RL MuJoCo locomotion v2",
        implementation_type="source_aligned_port",
        reproduction_status="source_aligned_port",
        parity_status="fixed_batch_partial",
        strict_final_eligible=False,
        remaining_deviation=(
            "no end-to-end fixed-batch optimizer parity certificate for the "
            "complete learner"
        ),
        benchmark_role="main",
        main_table_eligible=True,
        display_name="RIQL-naive",
        task_scope="d4rl_locomotion_v2",
        upstream_task_version="v2",
        benchmark_task_version="v2",
    ),
    "wsrl": BaselineReproductionRecord(
        paper_title="Efficient Online Reinforcement Learning Fine-Tuning Need Not Retain Offline Data",
        upstream_repository="https://github.com/zhouzypaul/wsrl",
        upstream_commit=UPSTREAM_COMMITS["wsrl"],
        official_task_support="D4RL MuJoCo locomotion, AntMaze, Adroit, Kitchen",
        implementation_type="framework_port",
        reproduction_status="framework_port_unverified",
        parity_status="unverified",
        strict_final_eligible=False,
        remaining_deviation="optimizer-step parity against pinned JAX/Flax output is unverified",
        benchmark_role="main",
        main_table_eligible=True,
        display_name="WSRL",
        task_scope="d4rl_locomotion_v2",
        upstream_task_version="v2",
        benchmark_task_version="v2",
    ),
    "cal_ql": BaselineReproductionRecord(
        paper_title="Cal-QL: Calibrated Offline RL Pre-Training for Efficient Online Fine-Tuning",
        upstream_repository="https://github.com/nakamotoo/Cal-QL",
        upstream_commit=UPSTREAM_COMMITS["cal_ql"],
        official_task_support="official release recipes: AntMaze and Adroit",
        implementation_type="source_aligned_locomotion_adaptation",
        reproduction_status="task_port",
        parity_status="unverified",
        strict_final_eligible=False,
        remaining_deviation=(
            "the frozen Hopper/HalfCheetah/Walker2d configuration is a "
            "source-aligned adaptation because no official locomotion recipe "
            "is released upstream"
        ),
        benchmark_role="main",
        main_table_eligible=True,
        display_name="Cal-QL (D4RL locomotion adaptation)",
        task_scope="d4rl_locomotion_adaptation",
        upstream_task_version=None,
        benchmark_task_version="v2",
    ),
    "pessimistic_q_ensemble": BaselineReproductionRecord(
        paper_title="Offline-to-Online Reinforcement Learning via Balanced Replay and Pessimistic Q-Ensemble",
        upstream_repository="https://github.com/shlee94/Off2OnRL",
        upstream_commit=UPSTREAM_COMMITS["pessimistic_q_ensemble"],
        official_task_support="D4RL MuJoCo v0 recipes in the public release",
        implementation_type="source_aligned_d4rl_v2_port",
        reproduction_status="task_port",
        parity_status="unverified",
        strict_final_eligible=False,
        remaining_deviation=(
            "the public v0 method is ported to the common D4RL-v2 transition "
            "artifact; interaction budgets are matched but offline compute is "
            "approximately five times a single-agent baseline"
        ),
        benchmark_role="main",
        main_table_eligible=True,
        display_name="Pessimistic Q-Ensemble (D4RL-v2 port)",
        task_scope="d4rl_v2_port",
        upstream_task_version="v0",
        benchmark_task_version="v2",
        offline_compute_multiplier=5.0,
    ),
}


@dataclass(frozen=True)
class ReportingRule:
    rule_id: str
    phase: str
    final_evaluations: int
    evaluation_episodes: int
    source: str
    verified: bool


REPORTING_RULES: Mapping[str, ReportingRule] = {
    "rpex": ReportingRule(
        "mean_last_3_online_evaluations_per_seed_then_population_mean_std",
        "online",
        3,
        10,
        "felix-thu/RPEX result protocol",
        True,
    ),
    "riql_naive": ReportingRule(
        "mean_last_3_online_evaluations_per_seed_then_population_mean_std",
        "online",
        3,
        10,
        "felix-thu/RPEX result protocol",
        True,
    ),
    # These entries prevent accidental inheritance of the RPEX rule. They are
    # explicitly unverified and therefore cannot produce a paper-reproduction
    # summary until a pinned upstream reporting fixture is added.
    "wsrl": ReportingRule(
        "terminal_online_evaluation",
        "online",
        1,
        20,
        "zhouzypaul/wsrl finetune.py terminal evaluation",
        False,
    ),
    "cal_ql": ReportingRule(
        "upstream_reporting_unverified",
        "online",
        1,
        10,
        "nakamotoo/Cal-QL (locomotion unsupported)",
        False,
    ),
    "pessimistic_q_ensemble": ReportingRule(
        "upstream_reporting_unverified",
        "online",
        1,
        10,
        "shlee94/Off2OnRL (D4RL-v2 task port)",
        False,
    ),
}

COMMON_BENCHMARK_REPORTING_RULE = ReportingRule(
    "common_mean_last_3_online_evaluations_per_seed_then_population_mean_std",
    "online",
    3,
    10,
    "repository common cross-algorithm benchmark metric",
    True,
)

STRICT_FINAL_TASKS = (
    "hopper-medium-replay-v2",
    "halfcheetah-medium-replay-v2",
    "walker2d-medium-replay-v2",
)
STRICT_FINAL_SEEDS = (0, 1, 2, 3, 4)


class FinalBenchmarkValidationError(ValueError):
    """Raised before a run that cannot produce publication-eligible output."""


def strict_final_algorithms() -> tuple[str, ...]:
    return tuple(
        name
        for name, record in BASELINE_REPRODUCTION_REGISTRY.items()
        if baseline_record_is_strict_eligible(record)
    )


@dataclass(frozen=True)
class RIQLReferenceRow:
    sigma: float
    quantile: float
    num_critics: int
    inverse_temperature: float = 3.0
    kappa: float = 0.1
    utd_ratio: int = 1
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    extension: bool = False


# Literal port of felix-thu/RPEX@35da71e:RIQL_TRAIN_CONFIG.py.  The upstream
# table has no clean or mixed row; callers must deliberately select the
# common-budget extension for those combinations.
_RANDOM = {
    "observations": {
        "halfcheetah": (0.1, 0.1, 5),
        "walker2d": (0.1, 0.25, 5),
        "hopper": (0.1, 0.25, 3),
    },
    "actions": {
        "halfcheetah": (0.5, 0.25, 3),
        "walker2d": (0.5, 0.1, 5),
        "hopper": (0.1, 0.25, 5),
    },
    "rewards": {
        "halfcheetah": (3.0, 0.25, 5),
        "walker2d": (3.0, 0.1, 5),
        "hopper": (1.0, 0.25, 3),
    },
    "dynamics": {
        "halfcheetah": (3.0, 0.25, 5),
        "walker2d": (1.0, 0.25, 3),
        "hopper": (1.0, 0.5, 5),
    },
}
_ADVERSARIAL = {
    "observations": {
        "halfcheetah": (0.1, 0.1, 5),
        "walker2d": (1.0, 0.25, 5),
        "hopper": (1.0, 0.25, 5),
    },
    "actions": {
        "halfcheetah": (1.0, 0.1, 5),
        "walker2d": (1.0, 0.1, 5),
        "hopper": (1.0, 0.25, 5),
    },
    "rewards": {
        "halfcheetah": (1.0, 0.1, 5),
        "walker2d": (3.0, 0.1, 5),
        "hopper": (0.1, 0.25, 5),
    },
    "dynamics": {
        "halfcheetah": (1.0, 0.1, 5),
        "walker2d": (1.0, 0.25, 5),
        "hopper": (1.0, 0.5, 5),
    },
}


def resolve_riql_reference_row(
    env_name: str,
    corruption: str,
    corruption_target: str,
    *,
    allow_extension: bool,
) -> tuple[str, RIQLReferenceRow]:
    domain = env_name.split("-", 1)[0]
    table = {"random": _RANDOM, "adversarial": _ADVERSARIAL}.get(corruption)
    if table is not None and corruption_target in table:
        sigma, quantile, critics = table[corruption_target][domain]
        key = f"{corruption}/{corruption_target}/{domain}"
        return key, RIQLReferenceRow(sigma, quantile, critics)
    if not allow_extension:
        raise ValueError(
            "RPEX RIQL_TRAIN_CONFIG.py has no official row for "
            f"{env_name} {corruption}/{corruption_target}; select "
            "suite_profile=common_budget_robustness for an explicit extension"
        )
    key = f"extension/{corruption}/{corruption_target}/{domain}"
    return key, RIQLReferenceRow(3.0, 0.1, 5, extension=True)


def canonical_json_sha256(payload: object) -> str:
    import hashlib
    import json

    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class ReproductionFixtureCertificate:
    """Reviewed identity for an immutable upstream-generated golden fixture.

    The content digest lives outside the fixture so editing both a fixture value
    and its in-file provenance cannot make the modified artifact self-certify.
    """

    fixture_id: str
    filename: str
    content_sha256: str
    upstream_repository: str
    upstream_commit: str
    upstream_source_sha256: str
    python_version: str
    numpy_version: str
    pytorch_version: str
    checkpoint_sha256: str | None = None
    oracle_source_sha256: str | None = None


RPEX_GOLDEN_FIXTURE_CERTIFICATES: Mapping[
    str, ReproductionFixtureCertificate
] = {
    "rpex_random_corruption_v1": ReproductionFixtureCertificate(
        fixture_id="rpex_random_corruption_v1",
        filename="rpex_random_corruption_v1.json",
        content_sha256=(
            "06ef5d67cb104849cf373923271571f6b5ef69e598d24e16dea8b5f219618bae"
        ),
        upstream_repository="https://github.com/felix-thu/RPEX",
        upstream_commit=UPSTREAM_COMMITS["rpex"],
        upstream_source_sha256=(
            "1d854dc964fc40c11881f5aa53d1b3d712b7a27d33c27b0a7786d5d7699597ff"
        ),
        python_version="3.10.20",
        numpy_version="2.2.6",
        pytorch_version="2.13.0",
    ),
    "rpex_adversarial_core_v1": ReproductionFixtureCertificate(
        fixture_id="rpex_adversarial_core_v1",
        filename="rpex_adversarial_core_v1.json",
        content_sha256=(
            "e34123aada610b9488c832d39dcd3d98c97ecb6da4b4695bd1736c70c88206f3"
        ),
        upstream_repository="https://github.com/felix-thu/RPEX",
        upstream_commit=UPSTREAM_COMMITS["rpex"],
        upstream_source_sha256=(
            "1d854dc964fc40c11881f5aa53d1b3d712b7a27d33c27b0a7786d5d7699597ff"
        ),
        python_version="3.10.20",
        numpy_version="2.2.6",
        pytorch_version="2.13.0",
        checkpoint_sha256=(
            "f5c558003cfd3814c4ea6cff4ce5319b61a8e3dc9013cf208c29e37e368680bd"
        ),
        oracle_source_sha256=(
            "ec8c0f4554bab14d68e368403b3f522e69d1691e8e5c562c905d10697dd8d9e7"
        ),
    ),
}


def validate_reproduction_fixture(path: object, fixture_id: str) -> dict[str, object]:
    """Return a certified fixture payload or raise with the exact mismatch."""

    import hashlib
    import json
    from pathlib import Path

    certificate = RPEX_GOLDEN_FIXTURE_CERTIFICATES.get(fixture_id)
    if certificate is None:
        raise ValueError(f"unknown reproduction fixture certificate: {fixture_id}")
    candidate = Path(path)
    if candidate.name != certificate.filename:
        raise ValueError(
            "fixture filename mismatch: "
            f"expected={certificate.filename} actual={candidate.name}"
        )
    try:
        raw = candidate.read_bytes()
    except OSError as exc:
        raise ValueError(f"cannot read certified fixture {candidate}: {exc}") from exc
    actual_content_sha256 = hashlib.sha256(raw).hexdigest()
    if actual_content_sha256 != certificate.content_sha256:
        raise ValueError(
            "fixture content SHA256 mismatch: "
            f"expected={certificate.content_sha256} actual={actual_content_sha256}"
        )
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"certified fixture is invalid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("certified fixture root must be a JSON object")

    expected_metadata = {
        "fixture_id": certificate.fixture_id,
        "fixture_schema_version": 1,
        "upstream_repository": certificate.upstream_repository,
        "upstream_commit": certificate.upstream_commit,
        "upstream_source_sha256": certificate.upstream_source_sha256,
        "python_version": certificate.python_version,
        "numpy_version": certificate.numpy_version,
        "pytorch_version": certificate.pytorch_version,
    }
    if certificate.checkpoint_sha256 is not None:
        expected_metadata["checkpoint_sha256"] = certificate.checkpoint_sha256
    if certificate.oracle_source_sha256 is not None:
        expected_metadata["oracle_source_sha256"] = (
            certificate.oracle_source_sha256
        )
    mismatches = {
        key: {"expected": expected, "actual": payload.get(key)}
        for key, expected in expected_metadata.items()
        if payload.get(key) != expected
    }
    if mismatches:
        raise ValueError(f"fixture provenance metadata mismatch: {mismatches}")

    expected_targets = {"observations", "actions", "rewards", "dynamics"}
    if fixture_id == "rpex_random_corruption_v1":
        if payload.get("generator") != (
            "pinned attack.py Attack.sample_indexs + corrupt_* methods"
        ):
            raise ValueError("random fixture generator metadata mismatch")
        if payload.get("rng_implementation") != (
            "numpy.random.RandomState(MT19937)"
        ):
            raise ValueError("random fixture RNG metadata mismatch")
        if set(payload.get("targets", {})) != expected_targets or set(
            payload.get("online_random", {})
        ) != expected_targets:
            raise ValueError("random fixture does not cover all four RPEX targets")
    else:
        if payload.get("scope") != (
            "public attack optimizer core; not the broken end-to-end wrapper"
        ):
            raise ValueError("adversarial fixture scope metadata mismatch")
        if payload.get("device") != "cpu" or payload.get("dtype") != "float32":
            raise ValueError("adversarial fixture device/dtype metadata mismatch")
        online = payload.get("online")
        if not isinstance(online, dict) or online.get(
            "fresh_unseeded_cpu_generator"
        ) is not True:
            raise ValueError("adversarial fixture generator semantics mismatch")
        offline = payload.get("offline")
        offline_fields = {
            "selected_indices",
            "split_sizes",
            "initial_parameter_hash",
            "initial_effective_perturbation_hash",
            "post_first_step_objectives",
            "post_last_step_objectives",
            "final_perturbation_hash",
            "attacked_input_hash",
        }
        online_fields = {
            "input_index",
            "initial_parameter_hash",
            "initial_effective_perturbation_hash",
            "post_first_step_objective",
            "post_last_step_objective",
            "final_perturbation_hash",
            "attacked_input_hash",
        }
        if payload.get("synthetic_input_hash_scheme") != (
            "observations_float32_bytes_then_actions_float32_bytes"
        ) or not isinstance(payload.get("synthetic_input_hash"), str):
            raise ValueError("adversarial synthetic input provenance is incomplete")
        if not isinstance(offline, dict) or not offline_fields.issubset(offline):
            raise ValueError("adversarial offline trajectory metadata is incomplete")
        if not online_fields.issubset(online):
            raise ValueError("adversarial online trajectory metadata is incomplete")
        blockers = payload.get("known_upstream_wrapper_blockers")
        if not isinstance(blockers, list) or len(blockers) != 3:
            raise ValueError("adversarial fixture must preserve upstream blockers")
    return payload
