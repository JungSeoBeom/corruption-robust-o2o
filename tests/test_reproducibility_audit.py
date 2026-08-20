from __future__ import annotations

import hashlib
import io
import json
import logging
import random
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from robust_o2o.config import ExperimentConfig
from robust_o2o.corruption import (
    AttackOracle,
    corrupt_offline_dataset,
    corrupt_online_transition,
    dataset_fingerprint,
    make_numpy_corruption_rng,
    sample_online_corruption_target,
)
from robust_o2o.fidelity import (
    FinalBenchmarkValidationError,
    RPEX_GOLDEN_FIXTURE_CERTIFICATES,
    STRICT_FINAL_SEEDS,
    validate_reproduction_fixture,
)
from robust_o2o.environment import StateNormalizer
from robust_o2o.experiment import (
    _replay_transition_coordinates,
    _restore_agent_config,
    _run_offline,
    _run_online,
)
from robust_o2o.replay import OfflineDataset, ReplayBuffer
from robust_o2o.replay import RPEX_OFFICIAL_REPLAY_SAMPLING
from scripts.audit_reproducibility import (
    Check,
    adversarial_checkpoint_check,
    audit,
    fixture_runtime_alignment_check,
    main as audit_main,
    run_strict_preflight,
    summarize_audit,
)


FIXTURE = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "rpex_random_corruption_v1.json"
)
ADVERSARIAL_FIXTURE = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "rpex_adversarial_core_v1.json"
)
HOPPER_CHECKPOINT = (
    Path(__file__).resolve().parents[2]
    / "RIQL-main"
    / "pretrained_model"
    / "EDAC"
    / "EDAC_baseline_seed0-hopper-medium-replay-v2"
    / "2999.pt"
)


def synthetic_dataset() -> dict[str, np.ndarray]:
    return {
        "observations": (
            np.arange(36, dtype=np.float32).reshape(12, 3) / 10.0 - 1.0
        ),
        "actions": (
            np.arange(24, dtype=np.float32).reshape(12, 2) / 20.0 - 0.5
        ),
        "rewards": np.linspace(-2.0, 3.0, 12, dtype=np.float32),
        "next_observations": (
            np.arange(36, dtype=np.float32).reshape(12, 3) / 7.0 + 0.25
        ),
        "terminals": np.asarray(
            [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1], dtype=np.float32
        ),
    }


class RPEXGoldenFixtureTest(unittest.TestCase):
    def test_official_replay_samplers_match_pinned_upstream_rngs(self):
        python_state = random.getstate()
        torch_state = torch.random.get_rng_state()
        try:
            dataset = synthetic_dataset()
            torch.manual_seed(713)
            expected_offline = torch.randint(
                low=0, high=len(dataset["rewards"]), size=(7,)
            )
            torch.manual_seed(713)
            offline = OfflineDataset(
                dataset,
                seed=999,
                sampling_profile=RPEX_OFFICIAL_REPLAY_SAMPLING,
            )
            offline_batch = offline.sample(7, torch.device("cpu"))
            torch.testing.assert_close(
                offline_batch["_indices"], expected_offline
            )

            expected_online = random.Random(17).sample(range(10), 6)
            online = ReplayBuffer(
                3,
                2,
                16,
                seed=17,
                sampling_profile=RPEX_OFFICIAL_REPLAY_SAMPLING,
            )
            for index in range(10):
                online.add(
                    dataset["observations"][index],
                    dataset["actions"][index],
                    float(dataset["rewards"][index]),
                    dataset["next_observations"][index],
                    float(dataset["terminals"][index]),
                )
            online_batch = online.sample(6, torch.device("cpu"))
            self.assertEqual(
                online_batch["_indices"].tolist(), expected_online
            )

            config = ExperimentConfig(
                "rpex",
                "hopper-medium-replay-v2",
                seed=17,
                implementation_profile="official_code_reference",
            )
            resolved = config.to_dict()
            self.assertEqual(config.replay_seed, 17)
            self.assertEqual(
                resolved["replay_sampling_profile"],
                RPEX_OFFICIAL_REPLAY_SAMPLING,
            )
            self.assertTrue(resolved["replay_rng_parity_verified"])
        finally:
            random.setstate(python_state)
            torch.random.set_rng_state(torch_state)

    def test_completed_official_online_resume_is_a_noop(self):
        class NoStepEnv:
            def __init__(self):
                self.action_space = SimpleNamespace(
                    low=np.asarray([-1.0], dtype=np.float32),
                    high=np.asarray([1.0], dtype=np.float32),
                )

            def reset(self):
                raise AssertionError("completed resume must not reset the environment")

            def step(self, action):
                del action
                raise AssertionError("completed resume must not step the environment")

        class NoUpdateAgent:
            total_updates = 0

            def select_action(self, state, evaluate=False):
                del state, evaluate
                raise AssertionError("completed resume must not sample an action")

            def update(self, batch):
                del batch
                raise AssertionError("completed resume must not update")

        config = ExperimentConfig(
            "rpex",
            "hopper-medium-replay-v2",
            corruption="random",
            corruption_target="observations",
            implementation_profile="official_code_reference",
            online_steps=2,
            replay_size=8,
        )
        raw_dataset = synthetic_dataset()
        raw_dataset["observations"] = raw_dataset["observations"][:, :2]
        raw_dataset["next_observations"] = raw_dataset[
            "next_observations"
        ][:, :2]
        raw_dataset["actions"] = raw_dataset["actions"][:, :1]
        offline = OfflineDataset(raw_dataset, seed=0)
        normalizer = StateNormalizer(
            mean=np.zeros(2, dtype=np.float32),
            std=np.ones(2, dtype=np.float32),
        )
        completion_payloads = []
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            metrics_path = run_dir / "metrics.csv"
            train_metrics_path = run_dir / "train_metrics.jsonl"
            metrics_path.write_text("", encoding="utf-8")
            train_metrics_path.write_text("", encoding="utf-8")
            logger = SimpleNamespace(
                logger=logging.getLogger("rpex-completed-resume-fixture"),
                run_dir=run_dir,
                metrics_path=metrics_path,
                train_metrics_path=train_metrics_path,
                write_completion_manifest=lambda outcomes: completion_payloads.append(
                    outcomes
                ),
            )
            resume_state = {
                "phase": "online",
                "phase_step": 3,
                "episode_boundary": True,
                "writer_append_position": {
                    "metrics_csv": 0,
                    "train_metrics_jsonl": 0,
                },
                "online_corruption_audit": {
                    "selected_transition_count": 0,
                },
            }
            _run_online(
                NoStepEnv(),
                NoStepEnv(),
                raw_dataset,
                config,
                NoUpdateAgent(),
                offline,
                normalizer,
                None,
                torch.device("cpu"),
                logger,
                state_dim=2,
                action_dim=1,
                resume_state=resume_state,
            )
            online_manifest = json.loads(
                (run_dir / "online_corruption_manifest.json").read_text()
            )
        self.assertEqual(online_manifest["actual_online_steps"], 3)
        self.assertTrue(online_manifest["resume_noop_already_complete"])
        self.assertEqual(len(completion_payloads), 1)

    def test_completed_offline_resume_is_a_noop(self):
        class NoSampleDataset:
            def sample(self, batch_size, device):
                del batch_size, device
                raise AssertionError("completed resume must not sample replay")

        class NoUpdateAgent:
            total_updates = 0

            def update(self, batch):
                del batch
                raise AssertionError("completed resume must not update")

        config = ExperimentConfig(
            "riql_naive",
            "hopper-medium-replay-v2",
            stage="offline",
            offline_steps=3,
            eval_period=2,
        )
        normalizer = StateNormalizer(
            mean=np.zeros(2, dtype=np.float32),
            std=np.ones(2, dtype=np.float32),
        )
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            metrics_path = run_dir / "metrics.csv"
            train_metrics_path = run_dir / "train_metrics.jsonl"
            metrics_path.write_text("existing-metric\n", encoding="utf-8")
            train_metrics_path.write_text(
                "existing-train-metric\n", encoding="utf-8"
            )
            logger = SimpleNamespace(
                logger=logging.getLogger("completed-offline-resume-fixture"),
                metrics_path=metrics_path,
                train_metrics_path=train_metrics_path,
            )
            resume_state = {
                "phase": "offline",
                "phase_step": config.offline_steps,
                "episode_boundary": True,
                "writer_append_position": {
                    "metrics_csv": metrics_path.stat().st_size,
                    "train_metrics_jsonl": train_metrics_path.stat().st_size,
                },
            }
            before_metrics = metrics_path.read_bytes()
            before_train_metrics = train_metrics_path.read_bytes()
            with (
                patch(
                    "robust_o2o.experiment._evaluate",
                    side_effect=AssertionError(
                        "completed resume must not evaluate"
                    ),
                ),
                patch(
                    "robust_o2o.experiment._save_phase_checkpoint",
                    side_effect=AssertionError(
                        "completed resume must not write a checkpoint"
                    ),
                ),
            ):
                _run_offline(
                    object(),
                    config,
                    NoUpdateAgent(),
                    NoSampleDataset(),
                    normalizer,
                    torch.device("cpu"),
                    logger,
                    state_dim=2,
                    action_dim=1,
                    resume_state=resume_state,
                )

            self.assertEqual(metrics_path.read_bytes(), before_metrics)
            self.assertEqual(
                train_metrics_path.read_bytes(), before_train_metrics
            )

    def test_official_online_budget_and_pre_transition_update_order(self):
        replay_instances = []

        class CapturingReplay(ReplayBuffer):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                replay_instances.append(self)

        class OneStepLegacyEnv:
            def __init__(self):
                self.action_space = SimpleNamespace(
                    low=np.asarray([-1.0], dtype=np.float32),
                    high=np.asarray([1.0], dtype=np.float32),
                )

            def seed(self, seed):
                self.last_seed = seed

            def reset(self):
                return np.asarray([12.0, -3.5], dtype=np.float32)

            def step(self, action):
                del action
                return (
                    np.asarray([8.0, -5.0], dtype=np.float32),
                    0.0,
                    True,
                    {},
                )

        class RecordingAgent:
            total_updates = 0

            def __init__(self):
                self.replay_sizes_at_update = []

            def select_action(self, state, evaluate=False):
                del state, evaluate
                return torch.zeros(1)

            def update(self, batch):
                del batch
                self.replay_sizes_at_update.append(replay_instances[0].size)
                self.total_updates += 1
                return {"loss": 0.0}

        config = ExperimentConfig(
            "rpex",
            "hopper-medium-replay-v2",
            corruption="random",
            corruption_target="observations",
            implementation_profile="official_code_reference",
            online_steps=2,
            initial_collection_steps=1,
            batch_size=2,
            replay_size=8,
            eval_period=100,
            train_log_period=100,
            online_corruption_rate=0.0,
        )
        raw_dataset = synthetic_dataset()
        raw_dataset["observations"] = raw_dataset["observations"][:, :2]
        raw_dataset["next_observations"] = raw_dataset[
            "next_observations"
        ][:, :2]
        raw_dataset["actions"] = raw_dataset["actions"][:, :1]
        offline = OfflineDataset(raw_dataset, seed=0)
        normalizer = StateNormalizer(
            mean=np.asarray([10.0, -4.0], dtype=np.float32),
            std=np.asarray([2.0, 0.5], dtype=np.float32),
        )
        agent = RecordingAgent()
        with tempfile.TemporaryDirectory() as directory:
            metrics_path = Path(directory) / "metrics.csv"
            train_metrics_path = Path(directory) / "train_metrics.jsonl"
            metrics_path.write_text("", encoding="utf-8")
            train_metrics_path.write_text("", encoding="utf-8")
            logger = SimpleNamespace(
                logger=logging.getLogger("rpex-order-fixture"),
                run_dir=Path(directory),
                metrics_path=metrics_path,
                train_metrics_path=train_metrics_path,
                write_completion_manifest=lambda _outcomes: None,
            )
            with patch(
                "robust_o2o.experiment.ReplayBuffer", CapturingReplay
            ), patch("robust_o2o.experiment._save_phase_checkpoint"):
                _run_online(
                    OneStepLegacyEnv(),
                    OneStepLegacyEnv(),
                    raw_dataset,
                    config,
                    agent,
                    offline,
                    normalizer,
                    None,
                    torch.device("cpu"),
                    logger,
                    state_dim=2,
                    action_dim=1,
                )
            online_manifest = json.loads(
                (Path(directory) / "online_corruption_manifest.json").read_text(
                    encoding="utf-8"
                )
            )

        # The official loop updates on step 3 from the two transitions that
        # were already present, then takes/stores transition 3 and stops at the
        # episode boundary because 3 > the nominal budget 2.
        self.assertEqual(agent.replay_sizes_at_update, [2])
        self.assertEqual(replay_instances[0].size, 3)
        self.assertEqual(online_manifest["requested_online_steps"], 2)
        self.assertEqual(online_manifest["actual_online_steps"], 3)
        self.assertEqual(online_manifest["episode_boundary_overshoot"], 1)

    def test_official_online_replay_coordinates_are_not_normalized_twice(self):
        normalizer = StateNormalizer(
            mean=np.asarray([10.0, -4.0], dtype=np.float32),
            std=np.asarray([2.0, 0.5], dtype=np.float32),
        )
        normalized_state = np.asarray([0.25, -0.75], dtype=np.float32)
        normalized_next_state = np.asarray([1.5, 2.0], dtype=np.float32)
        stored_state, stored_next_state = _replay_transition_coordinates(
            normalized_state,
            normalized_next_state,
            normalizer,
            already_normalized=True,
        )
        np.testing.assert_array_equal(stored_state, normalized_state)
        np.testing.assert_array_equal(stored_next_state, normalized_next_state)

        raw_state = np.asarray([12.0, -3.5], dtype=np.float32)
        raw_next_state = np.asarray([8.0, -5.0], dtype=np.float32)
        stored_state, stored_next_state = _replay_transition_coordinates(
            raw_state,
            raw_next_state,
            normalizer,
            already_normalized=False,
        )
        np.testing.assert_array_equal(
            stored_state, normalizer.transform(raw_state)
        )
        np.testing.assert_array_equal(
            stored_next_state, normalizer.transform(raw_next_state)
        )

    def test_fixture_certificates_validate_content_and_provenance(self):
        for fixture_id, certificate in RPEX_GOLDEN_FIXTURE_CERTIFICATES.items():
            with self.subTest(fixture_id=fixture_id):
                payload = validate_reproduction_fixture(
                    FIXTURE.parent / certificate.filename, fixture_id
                )
                self.assertEqual(payload["upstream_commit"], certificate.upstream_commit)
                self.assertEqual(
                    payload["upstream_source_sha256"],
                    certificate.upstream_source_sha256,
                )

    def test_fixture_certificate_rejects_modified_content(self):
        original = FIXTURE.read_bytes()
        with tempfile.TemporaryDirectory() as directory:
            modified = Path(directory) / FIXTURE.name
            modified.write_bytes(original + b"\n")
            with self.assertRaisesRegex(ValueError, "content SHA256 mismatch"):
                validate_reproduction_fixture(
                    modified, "rpex_random_corruption_v1"
                )

    def test_missing_adversarial_checkpoint_is_blocking_not_skipped(self):
        with tempfile.TemporaryDirectory() as directory:
            check = adversarial_checkpoint_check(Path(directory) / "missing.pt")
        self.assertFalse(check.passed)
        self.assertTrue(check.blocking)
        self.assertIn("missing pinned checkpoint", check.detail)

    def test_random_targets_match_pinned_upstream_fixture(self):
        fixture = validate_reproduction_fixture(
            FIXTURE, "rpex_random_corruption_v1"
        )
        dataset = synthetic_dataset()
        self.assertEqual(dataset_fingerprint(dataset), fixture["dataset_hash"])
        keys = {
            "observations": "observations",
            "actions": "actions",
            "rewards": "rewards",
            "dynamics": "next_observations",
        }
        with tempfile.TemporaryDirectory() as directory:
            cache_root = Path(directory)
            for target, dataset_key in keys.items():
                expected = fixture["targets"][target]
                config = ExperimentConfig(
                    "rpex",
                    "hopper-medium-replay-v2",
                    corruption="random",
                    corruption_target=target,
                    seed=fixture["seed"],
                    implementation_profile="official_code_reference",
                    offline_corruption_rate=fixture["corruption_rate"],
                    corruption_range=fixture["epsilon"],
                )
                actual, metadata = corrupt_offline_dataset(
                    dataset, config, None, cache_root
                )
                indices = np.asarray(
                    expected["selected_indices"], dtype=np.int64
                )
                self.assertEqual(config.corruption_seed, config.seed)
                self.assertEqual(
                    metadata["rng_implementation"],
                    "numpy.random.RandomState",
                )
                self.assertEqual(
                    hashlib.sha256(indices.tobytes()).hexdigest(),
                    expected["mask_hash"],
                )
                self.assertTrue(
                    np.array_equal(
                        actual[dataset_key][indices],
                        np.asarray(
                            expected["corrupted_values"], dtype=np.float32
                        ),
                    ),
                    target,
                )
                self.assertEqual(
                    metadata["corruption_value_sha256"],
                    expected["corrupted_value_hash"],
                )
                self.assertEqual(
                    metadata["final_artifact_sha256"],
                    expected["final_dataset_hash"],
                )

    def test_official_seed_mapping_rejects_offset(self):
        with self.assertRaisesRegex(ValueError, "corruption_seed == seed"):
            ExperimentConfig(
                "rpex",
                "hopper-medium-replay-v2",
                corruption="random",
                corruption_target="observations",
                seed=3,
                corruption_seed=10_004,
                implementation_profile="official_code_reference",
            )

    def test_online_random_call_order_matches_upstream_fixture(self):
        fixture = validate_reproduction_fixture(
            FIXTURE, "rpex_random_corruption_v1"
        )
        inputs = {
            "observations": np.asarray([0.1, -0.2, 0.3], dtype=np.float32),
            "actions": np.asarray([0.25, -0.5], dtype=np.float32),
            "rewards": np.asarray([1.25], dtype=np.float32),
            "dynamics": np.asarray([-0.4, 0.5, 0.75], dtype=np.float32),
        }
        state = inputs["observations"]
        action = inputs["actions"]
        next_state = inputs["dynamics"]
        for target, original in inputs.items():
            config = ExperimentConfig(
                "rpex",
                "hopper-medium-replay-v2",
                corruption="random",
                corruption_target=target,
                seed=fixture["seed"],
                implementation_profile="official_code_reference",
                online_corruption_rate=0.5,
                corruption_range=fixture["epsilon"],
            )
            rng = make_numpy_corruption_rng(config)
            selected_flags = []
            values = []
            for _ in range(8):
                selected_target = sample_online_corruption_target(config, rng)
                result = corrupt_online_transition(
                    state,
                    action,
                    float(inputs["rewards"][0]),
                    next_state,
                    config,
                    None,
                    rng,
                    np.ones(3, dtype=np.float32),
                    np.asarray([0.2, 0.4], dtype=np.float32),
                    selected_target=selected_target,
                    selection_already_sampled=True,
                )
                selected_flags.append(int(result[-1]))
                if target == "observations":
                    value = result[0]
                elif target == "actions":
                    value = result[1]
                elif target == "rewards":
                    value = result[2]
                else:
                    value = result[3]
                values.append(np.asarray(value, dtype=np.float32).tolist())
            expected = fixture["online_random"][target]
            self.assertEqual(selected_flags, expected["selected"])
            self.assertTrue(
                np.array_equal(
                    np.asarray(values, dtype=np.float32),
                    np.asarray(expected["values"], dtype=np.float32),
                ),
                target,
            )
            self.assertTrue(
                np.array_equal(
                    np.asarray(rng.uniform(size=4)),
                    np.asarray(expected["rng_tail"]),
                ),
                f"{target} RNG tail",
            )

    def test_final_registry_rejects_unverified_and_task_port_baselines(self):
        for algorithm in ("wsrl", "cal_ql", "pessimistic_q_ensemble"):
            with self.subTest(algorithm=algorithm):
                with self.assertRaises((FinalBenchmarkValidationError, ValueError)):
                    ExperimentConfig(
                        algorithm,
                        "hopper-medium-replay-v2",
                        suite_profile="primary_research_benchmark",
                        run_purpose="final_benchmark",
                        benchmark_seed_set=STRICT_FINAL_SEEDS,
                    )

        with self.assertRaisesRegex(
            FinalBenchmarkValidationError, "non-allowlisted baseline"
        ):
            ExperimentConfig(
                "wsrl",
                "hopper-medium-replay-v2",
                suite_profile="primary_research_benchmark",
                run_purpose="diagnostic",
            )

    def test_final_benchmark_rejects_uncertified_corruption_fixture(self):
        with patch(
            "robust_o2o.config.validate_reproduction_fixture",
            side_effect=ValueError("fixture content SHA256 mismatch"),
        ):
            with self.assertRaisesRegex(
                FinalBenchmarkValidationError,
                "corruption_fixture_verification",
            ):
                ExperimentConfig(
                    "rpex",
                    "hopper-medium-replay-v2",
                    corruption="random",
                    corruption_target="observations",
                    suite_profile="primary_research_benchmark",
                    run_purpose="final_benchmark",
                    benchmark_seed_set=STRICT_FINAL_SEEDS,
                )

    def test_publication_eligibility_requires_validated_final_purpose(self):
        diagnostic = ExperimentConfig(
            "rpex",
            "hopper-medium-replay-v2",
            corruption="random",
            corruption_target="observations",
            implementation_profile="official_code_reference",
            run_purpose="diagnostic",
        ).to_dict()
        self.assertFalse(diagnostic["publication_eligible"])
        self.assertTrue(diagnostic["not_paper_reproduction"])

        final_config = ExperimentConfig(
            "rpex",
            "hopper-medium-replay-v2",
            corruption="random",
            corruption_target="observations",
            suite_profile="primary_research_benchmark",
            run_purpose="final_benchmark",
            benchmark_seed_set=STRICT_FINAL_SEEDS,
        )
        final = final_config.to_dict()
        self.assertFalse(final["publication_eligible"])
        self.assertFalse(final["controller_seed_cohort_attested"])
        final_config._controller_seed_cohort_attested = True
        final = final_config.to_dict()
        self.assertTrue(final["publication_eligible"])
        self.assertTrue(final["corruption_fixture_verified"])

    def test_final_benchmark_locks_core_upstream_contract(self):
        base = {
            "algorithm": "rpex",
            "env_name": "hopper-medium-replay-v2",
            "corruption": "random",
            "corruption_target": "observations",
            "suite_profile": "primary_research_benchmark",
            "run_purpose": "final_benchmark",
            "benchmark_seed_set": STRICT_FINAL_SEEDS,
        }
        invalid_overrides = {
            "stage": "offline",
            "benchmark_seed_set": (0,),
            "replay_seed": 7,
            "initial_collection_steps": 4_999,
            "warmup_steps": 4_999,
            "batch_size": 128,
            "replay_size": 999_999,
            "eval_period": 5_000,
            "eval_episodes": 1,
            "max_episode_steps": 999,
            "hidden_dim": 128,
            "hidden_layers": 3,
            "learning_rate": 1e-4,
            "max_grad_norm": 1.0,
            "discount": 0.95,
            "target_update_rate": 0.01,
            "normalize_states": False,
            "deterministic_policy": True,
            "evaluation_mode": "deterministic_diagnostic",
            "online_replay_profile": "fixed_offline_online_mixture",
            "evaluation_policy_profile": "paper_greedy_highest_weight",
            "policy_extraction": "align_iql",
            "expectile": 0.8,
            "beta": 2.0,
            "offline_ratio": 0.5,
            "offline_attack_steps": 99,
            "attack_min_step_size": 0.001,
            "initialize_from_checkpoint": "/tmp/not-an-official-checkpoint.pt",
        }
        for field, value in invalid_overrides.items():
            with self.subTest(field=field):
                with self.assertRaises(FinalBenchmarkValidationError):
                    ExperimentConfig(**{**base, field: value})

    def test_final_requires_full_seed_cohort_even_for_single_run_entrypoint(self):
        with self.assertRaisesRegex(
            FinalBenchmarkValidationError, "benchmark_seed_set"
        ):
            ExperimentConfig(
                "rpex",
                "hopper-medium-replay-v2",
                corruption="random",
                corruption_target="observations",
                suite_profile="primary_research_benchmark",
                run_purpose="final_benchmark",
            )

    def test_adversarial_fixture_scope_is_observation_only(self):
        base = {
            "algorithm": "rpex",
            "env_name": "hopper-medium-replay-v2",
            "corruption": "adversarial",
            "suite_profile": "primary_research_benchmark",
            "run_purpose": "final_benchmark",
            "benchmark_seed_set": STRICT_FINAL_SEEDS,
        }
        observations = ExperimentConfig(
            **base, corruption_target="observations"
        )
        self.assertEqual(
            observations.corruption_fixture_id, "rpex_adversarial_core_v1"
        )
        self.assertTrue(observations.to_dict()["corruption_fixture_verified"])

        for target in ("actions", "rewards", "dynamics"):
            with self.subTest(target=target):
                diagnostic = ExperimentConfig(
                    "rpex",
                    "hopper-medium-replay-v2",
                    corruption="adversarial",
                    corruption_target=target,
                    run_purpose="diagnostic",
                    implementation_profile="official_code_reference",
                ).to_dict()
                self.assertIsNone(diagnostic["corruption_fixture_id"])
                self.assertFalse(diagnostic["corruption_fixture_verified"])
                self.assertFalse(diagnostic["publication_eligible"])
                with self.assertRaisesRegex(
                    FinalBenchmarkValidationError, "corruption_fixture_scope"
                ):
                    ExperimentConfig(**base, corruption_target=target)

    def test_hopper_adversarial_fixture_cannot_certify_other_tasks(self):
        for env_name in (
            "halfcheetah-medium-replay-v2",
            "walker2d-medium-replay-v2",
        ):
            with self.subTest(env_name=env_name):
                diagnostic = ExperimentConfig(
                    "rpex",
                    env_name,
                    corruption="adversarial",
                    corruption_target="observations",
                    run_purpose="diagnostic",
                    implementation_profile="official_code_reference",
                )
                self.assertIsNone(diagnostic.corruption_fixture_id)
                resolved = diagnostic.to_dict()
                self.assertFalse(resolved["corruption_fixture_verified"])
                self.assertEqual(
                    resolved["condition_status"],
                    "paper_condition_fixture_unverified",
                )
                self.assertFalse(resolved["publication_eligible"])

                with self.assertRaisesRegex(
                    FinalBenchmarkValidationError, "corruption_fixture_scope"
                ):
                    ExperimentConfig(
                        "rpex",
                        env_name,
                        corruption="adversarial",
                        corruption_target="observations",
                        suite_profile="primary_research_benchmark",
                        run_purpose="final_benchmark",
                        benchmark_seed_set=STRICT_FINAL_SEEDS,
                    )

    def test_checkpoint_restored_fields_are_revalidated_for_final(self):
        config = ExperimentConfig(
            "rpex",
            "hopper-medium-replay-v2",
            corruption="random",
            corruption_target="observations",
            suite_profile="primary_research_benchmark",
            run_purpose="final_benchmark",
            benchmark_seed_set=STRICT_FINAL_SEEDS,
        )
        _restore_agent_config(config, {"config": {"hidden_dim": 64}})
        with self.assertRaisesRegex(FinalBenchmarkValidationError, "hidden_dim"):
            config._validate_final_benchmark()

    def test_paper_reproduction_is_reserved_until_contract_is_certified(self):
        with self.assertRaisesRegex(
            FinalBenchmarkValidationError, "paper_reproduction is reserved"
        ):
            ExperimentConfig(
                "rpex",
                "hopper-medium-replay-v2",
                corruption="random",
                corruption_target="observations",
                suite_profile="common_budget_diagnostic",
                run_purpose="paper_reproduction",
            )

    @unittest.skipUnless(
        HOPPER_CHECKPOINT.exists(),
        "pinned RPEX EDAC checkpoint is not installed; audit treats this as blocking",
    )
    def test_adversarial_offline_and_online_core_match_upstream_fixture(self):
        if not HOPPER_CHECKPOINT.is_file():
            self.fail(f"pinned RPEX EDAC checkpoint is missing: {HOPPER_CHECKPOINT}")
        fixture = validate_reproduction_fixture(
            ADVERSARIAL_FIXTURE, "rpex_adversarial_core_v1"
        )
        self.assertEqual(
            hashlib.sha256(HOPPER_CHECKPOINT.read_bytes()).hexdigest(),
            fixture["checkpoint_sha256"],
        )
        observations = (
            np.arange(64 * 11, dtype=np.float32).reshape(64, 11) / 200.0
            - 1.5
        )
        actions = (
            np.arange(64 * 3, dtype=np.float32).reshape(64, 3) / 100.0
            - 0.75
        )
        self.assertEqual(
            hashlib.sha256(
                np.ascontiguousarray(observations).tobytes()
                + np.ascontiguousarray(actions).tobytes()
            ).hexdigest(),
            fixture["synthetic_input_hash"],
        )
        dataset = {
            "observations": observations,
            "actions": actions,
            "rewards": np.zeros(64, dtype=np.float32),
            "next_observations": observations + np.float32(0.1),
            "terminals": np.zeros(64, dtype=np.float32),
        }
        config = ExperimentConfig(
            "rpex",
            "hopper-medium-replay-v2",
            corruption="adversarial",
            corruption_target="observations",
            seed=fixture["seed"],
            implementation_profile="official_code_reference",
            offline_corruption_rate=fixture["corruption_rate"],
            corruption_range=fixture["epsilon"],
            attack_checkpoint=str(HOPPER_CHECKPOINT),
            attack_checkpoint_sha256=fixture["checkpoint_sha256"],
        )
        oracle = AttackOracle(
            11,
            3,
            1.0,
            HOPPER_CHECKPOINT,
            torch.device("cpu"),
            seed=config.corruption_seed,
            implementation_profile="rpex_official_adam",
            record_trace=True,
        )
        try:
            with tempfile.TemporaryDirectory() as directory:
                result, metadata = corrupt_offline_dataset(
                    dataset, config, oracle, Path(directory)
                )
            indices = np.asarray(
                fixture["offline"]["selected_indices"], dtype=np.int64
            )
            attacked = np.ascontiguousarray(result["observations"][indices])
            self.assertEqual(
                hashlib.sha256(attacked.tobytes()).hexdigest(),
                fixture["offline"]["attacked_input_hash"],
            )
            self.assertEqual(
                metadata["selected_transition_indices_sha256"],
                fixture["offline"]["selected_indices_hash"],
            )
            offline_traces = list(oracle.attack_traces)
            self.assertEqual(
                [trace["attacked_input"].shape[0] for trace in offline_traces],
                fixture["offline"]["split_sizes"],
            )
            initial_parameters = np.ascontiguousarray(
                np.concatenate(
                    [trace["initial_parameter"] for trace in offline_traces]
                )
            )
            final_perturbations = np.ascontiguousarray(
                np.concatenate(
                    [trace["final_perturbation"] for trace in offline_traces]
                )
            )
            initial_effective_perturbations = np.ascontiguousarray(
                np.concatenate(
                    [
                        trace["initial_effective_perturbation"]
                        for trace in offline_traces
                    ]
                )
            )
            self.assertEqual(
                hashlib.sha256(initial_parameters.tobytes()).hexdigest(),
                fixture["offline"]["initial_parameter_hash"],
            )
            self.assertEqual(
                hashlib.sha256(
                    initial_effective_perturbations.tobytes()
                ).hexdigest(),
                fixture["offline"]["initial_effective_perturbation_hash"],
            )
            np.testing.assert_allclose(
                [trace["first_step_objective"] for trace in offline_traces],
                fixture["offline"]["post_first_step_objectives"],
                rtol=0.0,
                atol=1e-6,
            )
            np.testing.assert_allclose(
                [trace["last_step_objective"] for trace in offline_traces],
                fixture["offline"]["post_last_step_objectives"],
                rtol=0.0,
                atol=1e-6,
            )
            self.assertEqual(
                hashlib.sha256(final_perturbations.tobytes()).hexdigest(),
                fixture["offline"]["final_perturbation_hash"],
            )

            online_original = observations[3].copy()
            self.assertEqual(fixture["online"]["input_index"], 3)
            online = oracle.attack(
                online_original[None, :],
                np.ones((1, 11), dtype=np.float32),
                online_original[None, :],
                actions[3][None, :],
                "observations",
                1.0,
                2,
                0.1,
                online=True,
            )[0]
            online_trace = oracle.attack_traces[-1]
            self.assertTrue(online_trace["online"])
            self.assertEqual(
                hashlib.sha256(
                    np.ascontiguousarray(
                        online_trace["initial_parameter"]
                    ).tobytes()
                ).hexdigest(),
                fixture["online"]["initial_parameter_hash"],
            )
            self.assertEqual(
                hashlib.sha256(
                    np.ascontiguousarray(
                        online_trace["initial_effective_perturbation"]
                    ).tobytes()
                ).hexdigest(),
                fixture["online"]["initial_effective_perturbation_hash"],
            )
            self.assertAlmostEqual(
                online_trace["first_step_objective"],
                fixture["online"]["post_first_step_objective"],
                places=6,
            )
            self.assertAlmostEqual(
                online_trace["last_step_objective"],
                fixture["online"]["post_last_step_objective"],
                places=6,
            )
            self.assertEqual(
                hashlib.sha256(
                    np.ascontiguousarray(
                        online_trace["final_perturbation"]
                    ).tobytes()
                ).hexdigest(),
                fixture["online"]["final_perturbation_hash"],
            )
            self.assertEqual(
                hashlib.sha256(np.ascontiguousarray(online).tobytes()).hexdigest(),
                fixture["online"]["attacked_input_hash"],
            )
        finally:
            oracle.close()


class ReproducibilityAuditStatusTest(unittest.TestCase):
    @staticmethod
    def _eligible_ready_five_not_ready_checks() -> list[Check]:
        return [
            Check("eligible_subset_gate", True, "passed"),
            Check(
                "wsrl_fixed_batch_parity",
                False,
                "fixed_batch_numerical_parity_missing",
                blocking=False,
            ),
            Check(
                "five_baseline_final_coverage",
                False,
                "strict_final_algorithms=('rpex', 'riql_naive')",
                blocking=False,
            ),
            Check(
                "save_resume_full_benchmark_coverage",
                True,
                "all eligible baseline/condition paths established",
                blocking=True,
            ),
        ]

    def test_nonblocking_five_baseline_gaps_do_not_make_subset_unreachable(self):
        summary = summarize_audit(
            self._eligible_ready_five_not_ready_checks()
        )
        self.assertEqual(summary["reproducibility_audit"], "PASS")
        self.assertEqual(summary["final_benchmark_status"], "READY")
        self.assertEqual(
            summary["eligible_subset_benchmark_status"], "READY"
        )
        self.assertEqual(
            summary["five_baseline_benchmark_status"], "NOT READY"
        )
        self.assertEqual(
            summary["final_benchmark_scope"], ["rpex", "riql_naive"]
        )

    def test_audit_marks_known_matrix_gaps_as_nonblocking_warnings(self):
        checks = {check.name: check for check in audit(static_only=True)}
        for name in ("wsrl_fixed_batch_parity", "five_baseline_final_coverage"):
            with self.subTest(name=name):
                self.assertFalse(checks[name].passed)
                self.assertFalse(checks[name].blocking)
        self.assertFalse(checks["save_resume_full_benchmark_coverage"].passed)
        self.assertTrue(checks["save_resume_full_benchmark_coverage"].blocking)
        self.assertFalse(checks["rpex_online_phase_rng_parity"].passed)
        self.assertTrue(checks["rpex_online_phase_rng_parity"].blocking)
        self.assertFalse(checks["rpex_evaluation_rng_parity"].passed)
        self.assertTrue(checks["rpex_evaluation_rng_parity"].blocking)
        self.assertIn("RIQL-naive", checks["save_resume_full_benchmark_coverage"].detail)
        self.assertIn(
            "random observation",
            checks["save_resume_full_benchmark_coverage"].detail,
        )

    def test_fixture_runtime_mismatch_is_a_blocking_audit_failure(self):
        check = fixture_runtime_alignment_check()
        self.assertFalse(check.passed)
        self.assertTrue(check.blocking)
        self.assertIn("required=1.23.5", check.detail)
        self.assertIn("required=2.5.1", check.detail)

    def test_missing_full_save_resume_coverage_blocks_subset(self):
        checks = self._eligible_ready_five_not_ready_checks()
        checks[-1] = Check(
            "save_resume_full_benchmark_coverage",
            False,
            "diagnostic path only",
            blocking=True,
        )
        summary = summarize_audit(checks)
        self.assertEqual(summary["reproducibility_audit"], "FAIL")
        self.assertEqual(summary["eligible_subset_benchmark_status"], "NOT READY")
        self.assertEqual(summary["five_baseline_benchmark_status"], "NOT READY")

    def test_preflight_save_resume_result_names_its_narrow_scope(self):
        completed = SimpleNamespace(returncode=0, stdout="{}", stderr="")
        with patch(
            "scripts.audit_reproducibility.subprocess.run",
            return_value=completed,
        ):
            d4rl_check, resume_check = run_strict_preflight()
        self.assertTrue(d4rl_check.passed)
        self.assertTrue(resume_check.passed)
        self.assertIn("only for RIQL-naive", resume_check.detail)
        self.assertIn("random observation", resume_check.detail)
        self.assertIn("common_budget_diagnostic", resume_check.detail)
        self.assertIn("offline checkpoint", resume_check.detail)
        self.assertIn("full resume-state equality", resume_check.detail)

    def test_text_and_json_outputs_separate_subset_from_five_baselines(self):
        checks = self._eligible_ready_five_not_ready_checks()
        with patch(
            "scripts.audit_reproducibility.audit", return_value=checks
        ), patch("sys.argv", ["audit_reproducibility.py"]):
            stream = io.StringIO()
            with redirect_stdout(stream):
                returncode = audit_main()
        output = stream.getvalue()
        self.assertEqual(returncode, 0)
        self.assertIn("[WARN] wsrl_fixed_batch_parity", output)
        self.assertIn("ELIGIBLE-SUBSET BENCHMARK STATUS: READY", output)
        self.assertIn("FIVE-BASELINE BENCHMARK STATUS: NOT READY", output)

        with patch(
            "scripts.audit_reproducibility.audit", return_value=checks
        ), patch("sys.argv", ["audit_reproducibility.py", "--json"]):
            stream = io.StringIO()
            with redirect_stdout(stream):
                returncode = audit_main()
        payload = json.loads(stream.getvalue())
        self.assertEqual(returncode, 0)
        self.assertEqual(
            payload["eligible_subset_benchmark_status"], "READY"
        )
        self.assertEqual(
            payload["five_baseline_benchmark_status"], "NOT READY"
        )


if __name__ == "__main__":
    unittest.main()
