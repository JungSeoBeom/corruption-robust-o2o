from __future__ import annotations

import json
import logging
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from robust_o2o.config import ExperimentConfig
from robust_o2o.environment import StateNormalizer
from robust_o2o.experiment import _run_online
from robust_o2o.replay import OfflineDataset, ReplayBuffer


def _dataset(
    size: int = 16,
    state_dim: int = 2,
    action_dim: int = 1,
) -> dict[str, np.ndarray]:
    observations = np.arange(size * state_dim, dtype=np.float32).reshape(
        size, state_dim
    )
    actions = np.linspace(-0.5, 0.5, size * action_dim, dtype=np.float32).reshape(
        size, action_dim
    )
    rewards = np.linspace(0.1, 1.0, size, dtype=np.float32)
    return {
        "observations": observations,
        "actions": actions,
        "rewards": rewards,
        "next_observations": observations + 1.0,
        "terminals": np.zeros(size, dtype=np.float32),
        # Cal-QL's offline half must contain genuine finite calibration targets.
        "mc_returns": rewards.copy(),
    }


class _LegacyEnv:
    def __init__(
        self,
        *,
        terminal_at: int = 1,
        action_low: float = -1.0,
        action_high: float = 1.0,
    ) -> None:
        self.action_space = SimpleNamespace(
            low=np.asarray([action_low], dtype=np.float32),
            high=np.asarray([action_high], dtype=np.float32),
        )
        self.terminal_at = int(terminal_at)
        self.episode_step = 0
        self.total_steps = 0
        self.actions: list[np.ndarray] = []
        self.events: list[tuple[str, int]] = []
        self.seed_value: int | None = None

    def seed(self, seed: int) -> None:
        self.seed_value = int(seed)

    def reset(self) -> np.ndarray:
        self.episode_step = 0
        return np.asarray([0.0, 0.5], dtype=np.float32)

    def step(self, action: np.ndarray):
        self.episode_step += 1
        self.total_steps += 1
        self.actions.append(np.asarray(action, dtype=np.float32).copy())
        self.events.append(("step", self.total_steps))
        terminated = self.episode_step >= self.terminal_at
        return (
            np.asarray(
                [float(self.total_steps), float(self.total_steps) + 0.5],
                dtype=np.float32,
            ),
            float(self.episode_step),
            terminated,
            {},
        )


def _normalizer() -> StateNormalizer:
    return StateNormalizer(
        mean=np.zeros(2, dtype=np.float32),
        std=np.ones(2, dtype=np.float32),
    )


def _logger(directory: str, name: str):
    run_dir = Path(directory)
    metrics_path = run_dir / "metrics.csv"
    train_metrics_path = run_dir / "train_metrics.jsonl"
    metrics_path.write_text("", encoding="utf-8")
    train_metrics_path.write_text("", encoding="utf-8")
    completions: list[dict[str, object]] = []
    return SimpleNamespace(
        logger=logging.getLogger(name),
        run_dir=run_dir,
        metrics_path=metrics_path,
        train_metrics_path=train_metrics_path,
        log_train=lambda *_args, **_kwargs: None,
        write_completion_manifest=lambda payload: completions.append(payload),
        completions=completions,
    )


def _controller_config(
    algorithm: str,
    *,
    online_steps: int,
    corruption: str = "clean",
    corruption_target: str = "none",
) -> ExperimentConfig:
    config = ExperimentConfig(
        algorithm,
        "hopper-medium-replay-v2",
        implementation_profile="research_benchmark",
        corruption=corruption,
        corruption_target=corruption_target,
        online_steps=online_steps,
        eval_period=100_000,
        train_log_period=100_000,
        checkpoint_period=100_000,
    )
    # Tests shrink only controller-independent storage/batch knobs after the
    # research profile has resolved. The semantic fields under test remain the
    # source-aligned values unless a test states its exact miniature analogue.
    config.replay_size = max(online_steps + 2, 16)
    return config


class MainControllerContractTest(unittest.TestCase):
    def test_stage_both_with_zero_online_steps_never_resets_or_updates(self):
        class NoInteractionEnv:
            def __init__(self) -> None:
                self.action_space = SimpleNamespace(
                    low=np.asarray([-1.0], dtype=np.float32),
                    high=np.asarray([1.0], dtype=np.float32),
                )

            def reset(self):
                raise AssertionError("zero-step online phase must not reset the env")

            def step(self, action):
                del action
                raise AssertionError("zero-step online phase must not step the env")

        class NoInteractionAgent:
            total_updates = 0

            def select_action(self, state, evaluate=False):
                del state, evaluate
                raise AssertionError("zero-step online phase must not select an action")

            def update(self, batch):
                del batch
                raise AssertionError("zero-step online phase must not update")

        config = _controller_config("rpex", online_steps=0)
        config.stage = "both"
        offline = OfflineDataset(_dataset(), seed=13)

        with tempfile.TemporaryDirectory() as directory:
            logger = _logger(directory, "zero-online-step-controller")
            with (
                patch("robust_o2o.experiment._evaluate"),
                patch("robust_o2o.experiment._save_phase_checkpoint") as save_checkpoint,
            ):
                _run_online(
                    NoInteractionEnv(),
                    object(),
                    _dataset(),
                    config,
                    NoInteractionAgent(),
                    offline,
                    _normalizer(),
                    None,
                    torch.device("cpu"),
                    logger,
                    state_dim=2,
                    action_dim=1,
                )
            manifest = json.loads(
                (Path(directory) / "online_corruption_manifest.json").read_text(
                    encoding="utf-8"
                )
            )

        self.assertEqual(manifest["requested_online_steps"], 0)
        self.assertEqual(manifest["actual_online_steps"], 0)
        self.assertEqual(manifest["episode_boundary_overshoot"], 0)
        save_checkpoint.assert_called_once()
        self.assertIsNotNone(save_checkpoint.call_args.kwargs["resume_state"])

    def test_rpex_and_riql_research_update_before_transition_and_do_not_clip(self):
        for algorithm in ("rpex", "riql_naive"):
            with self.subTest(algorithm=algorithm), tempfile.TemporaryDirectory() as directory:
                replay_instances: list[ReplayBuffer] = []

                class CapturingReplay(ReplayBuffer):
                    def __init__(self, *args, **kwargs):
                        super().__init__(*args, **kwargs)
                        replay_instances.append(self)

                env = _LegacyEnv(terminal_at=1)

                class RecordingAgent:
                    def __init__(self) -> None:
                        self.total_updates = 0
                        self.update_env_steps: list[int] = []
                        self.sampled_actions: list[np.ndarray] = []

                    def select_action(self, state, evaluate=False):
                        del state, evaluate
                        # Deliberately outside the environment's [-1, 1] box.
                        return torch.asarray([2.5], dtype=torch.float32)

                    def update(self, batch):
                        self.update_env_steps.append(env.total_steps)
                        env.events.append(("update", env.total_steps))
                        self.sampled_actions.append(
                            batch["actions"].detach().cpu().numpy().copy()
                        )
                        self.total_updates += 1
                        return {"loss": 0.0}

                config = _controller_config(algorithm, online_steps=2)
                config.initial_collection_steps = 0
                config.batch_size = 1
                self.assertEqual(
                    config.action_execution_profile, "official_algorithm_behavior"
                )
                offline = OfflineDataset(_dataset(), seed=0)
                logger = _logger(directory, f"{algorithm}-controller")
                agent = RecordingAgent()

                with (
                    patch("robust_o2o.experiment.ReplayBuffer", CapturingReplay),
                    patch("robust_o2o.experiment._evaluate"),
                    patch("robust_o2o.experiment._save_phase_checkpoint"),
                ):
                    _run_online(
                        env,
                        object(),
                        _dataset(),
                        config,
                        agent,
                        offline,
                        _normalizer(),
                        None,
                        torch.device("cpu"),
                        logger,
                        state_dim=2,
                        action_dim=1,
                    )

                self.assertEqual(
                    env.events,
                    [("step", 1), ("update", 1), ("step", 2)],
                )
                self.assertEqual(agent.update_env_steps, [1])
                self.assertEqual(agent.total_updates, 1)
                np.testing.assert_array_equal(
                    np.stack(env.actions),
                    np.full((2, 1), 2.5, dtype=np.float32),
                )
                replay = replay_instances[0]
                np.testing.assert_array_equal(
                    replay.actions[: replay.size],
                    np.full((2, 1), 2.5, dtype=np.float32),
                )
                np.testing.assert_array_equal(
                    agent.sampled_actions[0],
                    np.asarray([[2.5]], dtype=np.float32),
                )

    def test_calql_holds_pending_episode_then_flushes_exact_rtg_and_utd_updates(self):
        replay_instances: list[ReplayBuffer] = []
        env = _LegacyEnv(terminal_at=3)

        class CapturingReplay(ReplayBuffer):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                replay_instances.append(self)

        class RecordingCalQLAgent:
            def __init__(self) -> None:
                self.total_updates = 0
                self.update_env_steps: list[int] = []
                self.source_counts: list[tuple[int, int]] = []
                self.seen_mc_returns: list[np.ndarray] = []

            def select_action(self, state, evaluate=False):
                del state, evaluate
                return torch.asarray([0.25], dtype=torch.float32)

            def update(self, batch):
                mc_returns = batch["mc_returns"].detach().cpu().numpy().copy()
                if not np.all(np.isfinite(mc_returns)):
                    raise AssertionError("Cal-QL received a non-finite MC return")
                source = batch["_source"]
                self.update_env_steps.append(env.total_steps)
                self.source_counts.append(
                    (int((source == 0).sum()), int((source == 1).sum()))
                )
                self.seen_mc_returns.append(mc_returns)
                self.total_updates += 1
                return {"loss": 0.0, "online_calibration_bound_rate": 1.0}

        config = _controller_config("cal_ql", online_steps=3)
        config.discount = 0.9
        config.updates_per_step = 2
        config.batch_size = 4
        config.replay_size = 16
        offline = OfflineDataset(_dataset(size=9), seed=3)
        agent = RecordingCalQLAgent()

        with tempfile.TemporaryDirectory() as directory:
            logger = _logger(directory, "calql-controller")
            with (
                patch("robust_o2o.experiment.ReplayBuffer", CapturingReplay),
                patch("robust_o2o.experiment._evaluate"),
                patch("robust_o2o.experiment._save_phase_checkpoint"),
            ):
                _run_online(
                    env,
                    object(),
                    _dataset(),
                    config,
                    agent,
                    offline,
                    _normalizer(),
                    None,
                    torch.device("cpu"),
                    logger,
                    state_dim=2,
                    action_dim=1,
                )
            manifest = json.loads(
                (Path(directory) / "online_corruption_manifest.json").read_text(
                    encoding="utf-8"
                )
            )

        # No optimizer step occurs while the three-step episode is pending;
        # the terminal transition flushes it and applies length * UTD updates.
        self.assertEqual(agent.update_env_steps, [3] * 6)
        self.assertEqual(agent.total_updates, 3 * 2)
        self.assertEqual(agent.source_counts, [(3, 1)] * 6)
        self.assertTrue(all(np.isfinite(x).all() for x in agent.seen_mc_returns))

        replay = replay_instances[0]
        self.assertEqual(replay.size, 3)
        np.testing.assert_allclose(
            replay.mc_returns[:3],
            np.asarray([5.23, 4.7, 3.0], dtype=np.float32),
            rtol=1e-6,
            atol=1e-6,
        )
        self.assertEqual(manifest["completed_online_trajectories"], 1)
        self.assertEqual(manifest["pending_episode_length"], 0)
        self.assertEqual(manifest["online_mc_return_valid_fraction"], 1.0)
        self.assertAlmostEqual(manifest["calql_dynamic_offline_ratio"], 0.75)

    def test_pqe_first_1000_step_block_maps_to_five_times_updates_and_poisoned_replay(self):
        replay_instances: list[ReplayBuffer] = []
        env = _LegacyEnv(terminal_at=1)
        counters = {"sample": 0, "priority": 0}

        class CapturingReplay(ReplayBuffer):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                replay_instances.append(self)

        fixed_batch = {
            "observations": torch.zeros((1, 2)),
            "actions": torch.zeros((1, 1)),
            "rewards": torch.zeros(1),
            "next_observations": torch.ones((1, 2)),
            "terminals": torch.zeros(1),
            "mc_returns": torch.ones(1),
            "_indices": torch.zeros(1, dtype=torch.long),
            "_source": torch.ones(1, dtype=torch.long),
        }

        def fake_pqe_batches(*args, **kwargs):
            del args, kwargs
            counters["sample"] += 1
            return fixed_batch, fixed_batch, fixed_batch

        def fake_priority_update(*args, **kwargs):
            del args, kwargs
            counters["priority"] += 1
            return {"number_of_priority_updates": float(counters["priority"])}

        def fake_corruption(
            state,
            action,
            reward,
            next_state,
            *args,
            **kwargs,
        ):
            del reward, args, kwargs
            return state.copy(), action.copy(), 99.0, next_state.copy(), True

        class RecordingPQEAgent:
            def __init__(self) -> None:
                self.total_updates = 0
                self.block_calls: list[tuple[int, int]] = []
                self.initial_priority_args: tuple[int, int] | None = None

            def select_action(self, state, evaluate=False):
                del state, evaluate
                return torch.asarray([0.25], dtype=torch.float32)

            def initial_online_priority(self, offline_size, block_size):
                self.initial_priority_args = (int(offline_size), int(block_size))
                return 7.0

            def online_update_count_for_block(self, block_index, normal_count):
                self.block_calls.append((int(block_index), int(normal_count)))
                return int(normal_count) * 5

            def update(self, **kwargs):
                self.assert_update_payload(kwargs)
                self.total_updates += 1
                return {"loss": 0.0}

            @staticmethod
            def assert_update_payload(kwargs):
                if not kwargs.get("rl_batch_prioritized"):
                    raise AssertionError("PQE RL batch was not prioritized")
                if kwargs["rl_batch"] is not fixed_batch:
                    raise AssertionError("controller did not route the PQE RL batch")

            @staticmethod
            def consume_priority_values():
                return torch.ones(1)

            @staticmethod
            def algorithm_metadata():
                return {"pqe_member_count": 5, "ensemble_size": 5}

        config = _controller_config(
            "pessimistic_q_ensemble",
            online_steps=1_000,
            corruption="random",
            corruption_target="rewards",
        )
        config.pqe_first_online_block_steps = 1_000
        config.pqe_online_buffer_size = 2_000
        config.updates_per_step = 1
        config.online_corruption_rate = 1.0
        offline = OfflineDataset(_dataset(size=12), seed=7)
        agent = RecordingPQEAgent()

        with tempfile.TemporaryDirectory() as directory:
            logger = _logger(directory, "pqe-controller")
            with (
                patch("robust_o2o.experiment.ReplayBuffer", CapturingReplay),
                patch(
                    "robust_o2o.experiment.sample_pqe_update_batches",
                    side_effect=fake_pqe_batches,
                ),
                patch(
                    "robust_o2o.experiment.update_sample_priorities",
                    side_effect=fake_priority_update,
                ),
                patch(
                    "robust_o2o.experiment.sample_online_corruption_target",
                    return_value="rewards",
                ),
                patch(
                    "robust_o2o.experiment.corrupt_online_transition",
                    side_effect=fake_corruption,
                ),
                patch("robust_o2o.experiment._evaluate"),
                patch("robust_o2o.experiment._save_phase_checkpoint"),
            ):
                _run_online(
                    env,
                    object(),
                    _dataset(),
                    config,
                    agent,
                    offline,
                    _normalizer(),
                    None,
                    torch.device("cpu"),
                    logger,
                    state_dim=2,
                    action_dim=1,
                )
            manifest = json.loads(
                (Path(directory) / "online_corruption_manifest.json").read_text(
                    encoding="utf-8"
                )
            )

        self.assertEqual(agent.initial_priority_args, (offline.size, 1_000))
        self.assertEqual(agent.block_calls, [(0, 1_000)])
        self.assertEqual(agent.total_updates, 5_000)
        self.assertEqual(counters, {"sample": 5_000, "priority": 5_000})
        replay = replay_instances[0]
        self.assertEqual(replay.size, 1_000)
        np.testing.assert_array_equal(
            replay.rewards[: replay.size], np.full(1_000, 99.0, dtype=np.float32)
        )
        np.testing.assert_array_equal(
            replay.priorities[: replay.size], np.full(1_000, 7.0, dtype=np.float64)
        )
        self.assertEqual(manifest["selected_transition_count"], 1_000)
        self.assertTrue(manifest["pqe_first_block_updates_applied"])
        self.assertEqual(manifest["pqe_first_block_update_count"], 5_000)

    def test_wsrl_warmup_collects_transitions_without_any_update(self):
        class NoWarmupUpdateAgent:
            def __init__(self) -> None:
                self.total_updates = 0
                self.actor_updates = 0
                self.critic_updates = 0
                self.temperature_updates = 0

            def select_action(self, state, evaluate=False):
                del state, evaluate
                return torch.asarray([0.0], dtype=torch.float32)

            def update(self, *args, **kwargs):
                del args, kwargs
                raise AssertionError("WSRL updated during its collection warmup")

        config = _controller_config("wsrl", online_steps=3)
        config.replay_size = 16
        env = _LegacyEnv(terminal_at=1)
        agent = NoWarmupUpdateAgent()
        offline = OfflineDataset(_dataset(), seed=11)

        with tempfile.TemporaryDirectory() as directory:
            logger = _logger(directory, "wsrl-controller")
            with (
                patch("robust_o2o.experiment._evaluate"),
                patch("robust_o2o.experiment._save_phase_checkpoint"),
            ):
                _run_online(
                    env,
                    object(),
                    _dataset(),
                    config,
                    agent,
                    offline,
                    _normalizer(),
                    None,
                    torch.device("cpu"),
                    logger,
                    state_dim=2,
                    action_dim=1,
                )
            manifest = json.loads(
                (Path(directory) / "online_corruption_manifest.json").read_text(
                    encoding="utf-8"
                )
            )

        self.assertEqual(env.total_steps, 3)
        self.assertEqual(agent.total_updates, 0)
        self.assertEqual(manifest["wsrl_first_update_env_step"], 5_002)
        self.assertEqual(manifest["wsrl_online_critic_updates"], 0)
        self.assertEqual(manifest["wsrl_online_actor_updates"], 0)
        self.assertEqual(manifest["wsrl_online_temperature_updates"], 0)


if __name__ == "__main__":
    unittest.main()
