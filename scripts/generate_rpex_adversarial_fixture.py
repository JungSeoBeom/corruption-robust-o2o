#!/usr/bin/env python3
"""Generate an adversarial-core fixture from pinned RPEX attack methods.

This validates the public optimizer trajectory around a fixed official EDAC
checkpoint.  It does not conceal the public wrapper's undefined-``std`` and
dynamics tuple bugs; those remain documented blockers to an exact end-to-end
adversarial reproduction claim.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import types
from pathlib import Path

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from rpex_fixture_provenance import (  # noqa: E402
    numpy_rng_state_hash,
    platform_metadata,
    require_pinned_clean_upstream,
    require_strict_runtime,
    sha256_file,
    torch_rng_state_hash,
)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

def sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def load_upstream(upstream: Path):
    gym_module = sys.modules.setdefault("gym", types.ModuleType("gym"))
    gym_module.Env = object
    sys.modules.setdefault("d4rl", types.ModuleType("d4rl"))
    sys.modules.setdefault("pyrallis", types.ModuleType("pyrallis"))
    sys.modules.setdefault("wandb", types.ModuleType("wandb"))
    source = upstream / "attack.py"
    spec = importlib.util.spec_from_file_location("pinned_rpex_attack_adv", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.modules["attack"] = module
    oracle_source = upstream / "EDAC.py"
    oracle_spec = importlib.util.spec_from_file_location(
        "pinned_rpex_edac_adv", oracle_source
    )
    if oracle_spec is None or oracle_spec.loader is None:
        raise RuntimeError(f"cannot load {oracle_source}")
    oracle_module = importlib.util.module_from_spec(oracle_spec)
    oracle_spec.loader.exec_module(oracle_module)
    return module, source, oracle_module, oracle_source


def make_attack(module, oracle_module, checkpoint: Path, seed: int):
    attack = module.Attack.__new__(module.Attack)
    attack.device = "cpu"
    attack.corruption_range = 1.0
    attack.update_times = 100
    attack.step_size = 0.01
    attack.corruption_tag = "observations"
    attack._th_rng = torch.Generator()
    attack._th_rng.manual_seed(seed)
    payload = torch.load(checkpoint, map_location="cpu")
    attack.actor = oracle_module.Actor(11, 3, 256, 1.0).eval()
    attack.critic = oracle_module.VectorizedCritic(11, 3, 256, 10).eval()
    attack.actor.load_state_dict(payload["actor"])
    attack.critic.load_state_dict(payload["critic"])
    for parameter in attack.actor.parameters():
        parameter.requires_grad_(False)
    for parameter in attack.critic.parameters():
        parameter.requires_grad_(False)
    attack.loss_Q = attack.loss_Q_for_obs
    return attack


def synthetic_inputs():
    observations = (
        np.arange(64 * 11, dtype=np.float32).reshape(64, 11) / 200.0 - 1.5
    )
    actions = (
        np.arange(64 * 3, dtype=np.float32).reshape(64, 3) / 100.0 - 0.75
    )
    return observations, actions


def traced_split(attack, observations, actions, std):
    initial_parts = []
    initial_effective_parts = []
    final_noise_parts = []
    attacked_parts = []
    first_objectives = []
    last_objectives = []
    pointer = 0
    size = observations.shape[0]
    for split_index in range(10):
        count = size // 10 if split_index < 9 else size - pointer
        obs = observations[pointer : pointer + count]
        act = actions[pointer : pointer + count]
        para = attack.sample_para(obs.shape, std)
        initial_parts.append(para.detach().cpu().numpy())
        initial_effective_parts.append((para * std).detach().cpu().numpy())
        first_post = None
        last_post = None
        for step in range(attack.update_times):
            para = torch.nn.Parameter(para.clone(), requires_grad=True)
            optimizer = torch.optim.Adam(
                [para], lr=attack.step_size * attack.corruption_range
            )
            loss = attack.loss_Q(para, obs, act, std)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            para = torch.clamp(
                para, -attack.corruption_range, attack.corruption_range
            ).detach()
            with torch.no_grad():
                post = float(attack.loss_Q(para, obs, act, std).item())
            if step == 0:
                first_post = post
            last_post = post
        final_noise = para * std
        final_noise_parts.append(final_noise.cpu().numpy())
        attacked_parts.append((obs + final_noise).cpu().numpy())
        first_objectives.append(first_post)
        last_objectives.append(last_post)
        pointer += count
    return {
        "initial": np.concatenate(initial_parts),
        "initial_effective": np.concatenate(initial_effective_parts),
        "final_noise": np.concatenate(final_noise_parts),
        "attacked": np.concatenate(attacked_parts),
        "first_objectives": first_objectives,
        "last_objectives": last_objectives,
    }


def traced_online(attack, original, std, observation, action):
    original_tensor = torch.from_numpy(original).view(1, -1)
    std_tensor = torch.from_numpy(std).view(1, -1)
    action_tensor = torch.from_numpy(action).view(1, -1)
    online_generator = torch.Generator()
    rng_before = torch_rng_state_hash(online_generator)
    para = 2.0 * std_tensor * (
        torch.rand(original_tensor.shape, generator=online_generator) - 0.5
    )
    rng_after = torch_rng_state_hash(online_generator)
    initial = para.detach().cpu().numpy()
    initial_effective = (para * std_tensor).detach().cpu().numpy()
    first_post = None
    last_post = None
    for step in range(2):
        para = torch.nn.Parameter(para.clone(), requires_grad=True)
        optimizer = torch.optim.Adam([para], lr=0.1)
        attacked = original_tensor + para * std_tensor
        loss = attack.critic(attacked.float(), action_tensor.float()).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        para = torch.clamp(para, -1.0, 1.0).detach()
        with torch.no_grad():
            attacked = original_tensor + para * std_tensor
            post = float(
                attack.critic(attacked.float(), action_tensor.float()).mean().item()
            )
        if step == 0:
            first_post = post
        last_post = post
    final_noise = para * std_tensor
    attacked = original_tensor + final_noise
    return {
        "initial": initial,
        "initial_effective": initial_effective,
        "first_objective": first_post,
        "last_objective": last_post,
        "final_noise": final_noise.cpu().numpy(),
        "attacked": attacked.cpu().numpy(),
        "torch_generator_state_before_sha256": rng_before,
        "torch_generator_state_after_sha256": rng_after,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--diagnostic-allow-runtime-mismatch",
        action="store_true",
        help=(
            "allow unsupported host versions for diagnostic output only; "
            "this generator never emits an end-to-end certificate"
        ),
    )
    args = parser.parse_args()
    upstream = args.upstream_dir.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve()
    runtime = require_strict_runtime(
        allow_diagnostic_mismatch=args.diagnostic_allow_runtime_mismatch
    )
    upstream_provenance = require_pinned_clean_upstream(upstream)
    commit = str(upstream_provenance["commit"])
    if not checkpoint.is_file():
        raise RuntimeError(f"checkpoint does not exist: {checkpoint}")
    module, source, oracle_module, oracle_source = load_upstream(upstream)
    torch.set_num_threads(1)
    process_torch_rng_before = torch_rng_state_hash()
    seed = 7
    all_observations, all_actions = synthetic_inputs()
    rng = np.random.RandomState(seed)
    selection_rng_before = numpy_rng_state_hash(rng.get_state())
    indices = np.flatnonzero(rng.random(len(all_observations)) < 0.3)
    selection_rng_after = numpy_rng_state_hash(rng.get_state())
    observations = torch.from_numpy(all_observations[indices].copy())
    actions = torch.from_numpy(all_actions[indices].copy())
    std = torch.from_numpy(all_observations.std(axis=0)).view(1, -1)

    official_attack = make_attack(module, oracle_module, checkpoint, seed)
    official_rng_before = torch_rng_state_hash(official_attack._th_rng)
    official = official_attack.split_gradient_attack(observations, actions, std)
    official_rng_after = torch_rng_state_hash(official_attack._th_rng)
    traced_attack = make_attack(module, oracle_module, checkpoint, seed)
    traced_rng_before = torch_rng_state_hash(traced_attack._th_rng)
    trace = traced_split(traced_attack, observations, actions, std)
    traced_rng_after = torch_rng_state_hash(traced_attack._th_rng)
    if not np.array_equal(official.astype(np.float32), trace["attacked"].astype(np.float32)):
        raise RuntimeError("instrumented trajectory differs from upstream final output")

    online_attack = make_attack(module, oracle_module, checkpoint, seed)
    online_original = all_observations[3].copy()
    online_std = np.ones(11, dtype=np.float32)
    online_obs = all_observations[3].copy()
    online_act = all_actions[3].copy()
    online = module.adversarial_attack(
        online_original,
        online_std,
        online_obs,
        online_act,
        "observations",
        types.SimpleNamespace(device="cpu"),
        online_attack.actor,
        online_attack.critic,
        1.0,
    ).astype(np.float32)
    online_trace = traced_online(
        online_attack,
        online_original,
        online_std,
        online_obs,
        online_act,
    )
    if not np.array_equal(
        online, online_trace["attacked"].astype(np.float32)
    ):
        raise RuntimeError("instrumented online trajectory differs from upstream output")

    generator_path = Path(__file__).resolve()
    helper_path = generator_path.with_name("rpex_fixture_provenance.py")
    payload = {
        "fixture_schema_version": 2,
        "fixture_id": "rpex_adversarial_optimizer_core_diagnostic_v2",
        "scope": (
            "diagnostic public optimizer core only; not the broken end-to-end "
            "wrapper and never an end-to-end adversarial certificate"
        ),
        "strict_or_publication_eligible": False,
        "end_to_end_verified": False,
        "strict_runtime_preflight_passed": bool(runtime["passed"]),
        "runtime_preflight": runtime,
        "generator_sha256": sha256_file(generator_path),
        "provenance_helper_sha256": sha256_file(helper_path),
        "upstream_repository": "https://github.com/felix-thu/RPEX",
        "upstream_commit": commit,
        "upstream_checkout_clean": upstream_provenance["clean"],
        "upstream_status_sha256": upstream_provenance["status_sha256"],
        "upstream_source_sha256": sha(source.read_bytes()),
        "oracle_source_sha256": sha(oracle_source.read_bytes()),
        "upstream_source_hashes": {
            "attack.py": sha256_file(source),
            "EDAC.py": sha256_file(oracle_source),
        },
        "platform": platform_metadata(execution_device="cpu"),
        "device": "cpu",
        "dtype": "float32",
        "synthetic_input_hash_scheme": (
            "observations_float32_bytes_then_actions_float32_bytes"
        ),
        "synthetic_input_hash": sha(
            np.ascontiguousarray(all_observations).tobytes()
            + np.ascontiguousarray(all_actions).tobytes()
        ),
        "input_hashes": {
            "synthetic_observations_and_actions": sha(
                np.ascontiguousarray(all_observations).tobytes()
                + np.ascontiguousarray(all_actions).tobytes()
            )
        },
        "checkpoint_sha256": sha(checkpoint.read_bytes()),
        "checkpoint_hashes": {
            "edac_oracle_checkpoint": sha(checkpoint.read_bytes())
        },
        "seed": seed,
        "corruption_rate": 0.3,
        "epsilon": 1.0,
        "offline": {
            "target": "observations",
            "selected_indices": indices.tolist(),
            "selected_indices_hash": sha(indices.astype(np.int64).tobytes()),
            "selection_numpy_rng_state_before_sha256": selection_rng_before,
            "selection_numpy_rng_state_after_sha256": selection_rng_after,
            "official_torch_generator_state_before_sha256": official_rng_before,
            "official_torch_generator_state_after_sha256": official_rng_after,
            "traced_torch_generator_state_before_sha256": traced_rng_before,
            "traced_torch_generator_state_after_sha256": traced_rng_after,
            "split_sizes": [len(indices) // 10] * 9
            + [len(indices) - 9 * (len(indices) // 10)],
            "initial_parameter_hash": sha(
                np.ascontiguousarray(trace["initial"]).tobytes()
            ),
            "initial_effective_perturbation_hash": sha(
                np.ascontiguousarray(trace["initial_effective"]).tobytes()
            ),
            "post_first_step_objectives": trace["first_objectives"],
            "post_last_step_objectives": trace["last_objectives"],
            "final_perturbation_hash": sha(
                np.ascontiguousarray(trace["final_noise"]).tobytes()
            ),
            "attacked_input_hash": sha(
                np.ascontiguousarray(trace["attacked"].astype(np.float32)).tobytes()
            ),
        },
        "online": {
            "target": "observations",
            "input_index": 3,
            "fresh_unseeded_cpu_generator": True,
            "steps": 2,
            "learning_rate": 0.1,
            "initial_parameter_hash": sha(
                np.ascontiguousarray(online_trace["initial"]).tobytes()
            ),
            "initial_effective_perturbation_hash": sha(
                np.ascontiguousarray(
                    online_trace["initial_effective"]
                ).tobytes()
            ),
            "post_first_step_objective": online_trace["first_objective"],
            "post_last_step_objective": online_trace["last_objective"],
            "final_perturbation_hash": sha(
                np.ascontiguousarray(online_trace["final_noise"]).tobytes()
            ),
            "attacked_input": online.reshape(-1).tolist(),
            "attacked_input_hash": sha(np.ascontiguousarray(online).tobytes()),
            "traced_torch_generator_state_before_sha256": online_trace[
                "torch_generator_state_before_sha256"
            ],
            "traced_torch_generator_state_after_sha256": online_trace[
                "torch_generator_state_after_sha256"
            ],
        },
        "process_rng_state_hashes": {
            "torch_cpu_before_sha256": process_torch_rng_before,
            "torch_cpu_after_sha256": torch_rng_state_hash(),
        },
        "known_upstream_wrapper_blockers": [
            "offline adversarial branches return undefined std",
            "cached attack returns two values while caller unpacks three",
            "offline dynamics passes Actor tuple directly to critic",
        ],
    }
    if not runtime["passed"]:
        payload["diagnostic_only_reason"] = (
            "unsupported runtime override was used; optimizer-core output "
            "cannot satisfy strict or publication eligibility"
        )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
