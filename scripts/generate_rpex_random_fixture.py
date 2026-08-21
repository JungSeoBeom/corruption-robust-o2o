#!/usr/bin/env python3
"""Print a random-corruption fixture by executing pinned RPEX methods.

The upstream checkout is intentionally an explicit input; the repository does
not vendor or silently download third-party code.  Redirect the reviewed JSON
to a temporary file and commit it only after checking the recorded commit and
source hash.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np

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

TARGETS = {
    "observations": ("observations", "corrupt_obs"),
    "actions": ("actions", "corrupt_act"),
    "rewards": ("rewards", "corrupt_rew"),
    "dynamics": ("next_observations", "corrupt_next_obs"),
}


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def dataset_hash(dataset: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for key in sorted(dataset):
        array = np.ascontiguousarray(dataset[key])
        digest.update(key.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def value_hash(
    target: str, indices: np.ndarray, values: np.ndarray
) -> str:
    digest = hashlib.sha256()
    digest.update(target.encode("utf-8"))
    digest.update(np.asarray(indices, dtype=np.int64).tobytes())
    contiguous = np.ascontiguousarray(values)
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(str(contiguous.shape).encode("ascii"))
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


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


def load_upstream_attack(upstream: Path):
    # attack.py imports Gym/D4RL at module import time, but the random methods
    # used here do not call either package.  Stubs make that dependency boundary
    # explicit without replacing any upstream Attack method.
    sys.modules.setdefault("gym", types.ModuleType("gym"))
    sys.modules.setdefault("d4rl", types.ModuleType("d4rl"))
    source = upstream / "attack.py"
    spec = importlib.util.spec_from_file_location("pinned_rpex_attack", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, source


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream-dir", type=Path, required=True)
    parser.add_argument(
        "--diagnostic-allow-runtime-mismatch",
        action="store_true",
        help=(
            "allow unsupported host versions for diagnostic output only; "
            "the result receives a non-certifiable diagnostic fixture ID"
        ),
    )
    args = parser.parse_args()
    upstream = args.upstream_dir.expanduser().resolve()
    runtime = require_strict_runtime(
        allow_diagnostic_mismatch=args.diagnostic_allow_runtime_mismatch
    )
    upstream_provenance = require_pinned_clean_upstream(upstream)
    commit = str(upstream_provenance["commit"])
    module, source = load_upstream_attack(upstream)
    clean = synthetic_dataset()
    seed = 7
    rate = 0.3
    epsilon = 0.5
    generator_path = Path(__file__).resolve()
    helper_path = generator_path.with_name("rpex_fixture_provenance.py")
    strict_generation = bool(runtime["passed"])
    payload: dict[str, object] = {
        "fixture_schema_version": 2,
        "fixture_id": (
            "rpex_random_corruption_v2"
            if strict_generation
            else "rpex_random_corruption_diagnostic_v2"
        ),
        "certification_scope": (
            "random corruption transformation fixture only; not learner, "
            "evaluator, online RNG, or end-to-end parity"
        ),
        "strict_runtime_preflight_passed": strict_generation,
        "runtime_preflight": runtime,
        "generator": "pinned attack.py Attack.sample_indexs + corrupt_* methods",
        "generator_sha256": sha256_file(generator_path),
        "provenance_helper_sha256": sha256_file(helper_path),
        "upstream_repository": "https://github.com/felix-thu/RPEX",
        "upstream_commit": commit,
        "upstream_checkout_clean": upstream_provenance["clean"],
        "upstream_status_sha256": upstream_provenance["status_sha256"],
        "upstream_source_sha256": sha256_bytes(source.read_bytes()),
        "upstream_source_hashes": {
            "attack.py": sha256_file(source),
        },
        "platform": platform_metadata(execution_device="cpu"),
        "dataset_hash": dataset_hash(clean),
        "input_hashes": {"synthetic_dataset": dataset_hash(clean)},
        "checkpoint_required": False,
        "checkpoint_sha256": None,
        "seed": seed,
        "corruption_rate": rate,
        "epsilon": epsilon,
        "rng_implementation": "numpy.random.RandomState(MT19937)",
        "targets": {},
    }
    for target, (dataset_key, method_name) in TARGETS.items():
        source_dataset = {key: value.copy() for key, value in clean.items()}
        result = {key: value.copy() for key, value in clean.items()}
        attack = module.Attack.__new__(module.Attack)
        attack.dataset = source_dataset
        attack._np_rng = np.random.RandomState(seed)
        attack.corruption_rate = rate
        attack.corruption_range = epsilon
        attack.corruption_random = True
        attack.corruption_tag = dataset_key
        rng_before = numpy_rng_state_hash(attack._np_rng.get_state())
        indices, original_indices = attack.sample_indexs()
        attack.attack_indexs = indices
        attack.original_indexs = original_indices
        result, _ = getattr(attack, method_name)(result)
        rng_after = numpy_rng_state_hash(attack._np_rng.get_state())
        values = np.asarray(result[dataset_key][indices])
        payload["targets"][target] = {
            "selected_indices": indices.astype(np.int64).tolist(),
            "mask_hash": sha256_bytes(indices.astype(np.int64).tobytes()),
            "corrupted_values": values.tolist(),
            "corrupted_values_dtype": str(values.dtype),
            "corrupted_value_hash": value_hash(target, indices, values),
            "final_dataset_hash": dataset_hash(result),
            "numpy_rng_state_before_sha256": rng_before,
            "numpy_rng_state_after_sha256": rng_after,
        }
    online_inputs = {
        "observations": (
            np.asarray([0.1, -0.2, 0.3], dtype=np.float32),
            np.ones(3, dtype=np.float32),
            False,
            "observations",
        ),
        "actions": (
            np.asarray([0.25, -0.5], dtype=np.float32),
            np.asarray([0.2, 0.4], dtype=np.float32),
            False,
            "actions",
        ),
        "rewards": (
            np.asarray([1.25], dtype=np.float32),
            np.asarray([1.0], dtype=np.float32),
            True,
            "rewards",
        ),
        "dynamics": (
            np.asarray([-0.4, 0.5, 0.75], dtype=np.float32),
            np.ones(3, dtype=np.float32),
            False,
            "next_observations",
        ),
    }
    payload["online_random"] = {}
    online_config = SimpleNamespace(
        corruption_range=epsilon, corruption_rate=0.5
    )
    observation = np.asarray([0.1, -0.2, 0.3], dtype=np.float32)
    action = np.asarray([0.25, -0.5], dtype=np.float32)
    torch_rng_before = torch_rng_state_hash()
    for target, (data, std, reward_target, upstream_tag) in online_inputs.items():
        np.random.seed(seed)
        rng_before = numpy_rng_state_hash(np.random.get_state())
        values = []
        selected = []
        for _ in range(8):
            value, flag = module.corrupt_trans(
                data,
                std,
                observation,
                action,
                None,
                None,
                corruption_random=True,
                corrupt_reward=reward_target,
                corruption_tag=upstream_tag,
                config=online_config,
            )
            values.append(np.asarray(value, dtype=np.float32).tolist())
            selected.append(int(flag))
        rng_after_calls = numpy_rng_state_hash(np.random.get_state())
        rng_tail = np.random.uniform(size=4).tolist()
        payload["online_random"][target] = {
            "selected": selected,
            "values": values,
            "rng_tail": rng_tail,
            "numpy_rng_state_before_sha256": rng_before,
            "numpy_rng_state_after_calls_sha256": rng_after_calls,
            "numpy_rng_state_after_tail_sha256": numpy_rng_state_hash(
                np.random.get_state()
            ),
        }
    payload["process_rng_state_hashes"] = {
        "torch_cpu_before_sha256": torch_rng_before,
        "torch_cpu_after_sha256": torch_rng_state_hash(),
    }
    if not strict_generation:
        payload["diagnostic_only_reason"] = (
            "unsupported runtime override was used; this output cannot satisfy "
            "strict or publication eligibility"
        )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
