# RPEX/D4RL-v2 Legacy Protocol

The default and only benchmark protocol in this repository is
`rpex_d4rl_v2_legacy`. It reproduces the environment/data interface used by the
official RPEX implementation.

## What the version names mean

MuJoCo is the simulator. Legacy Gym 0.23.1 supplies the locomotion task
implementation, and D4RL registers complete offline-RL environment IDs such as
`walker2d-medium-replay-v2`.

The `-v2` suffix in that name is the D4RL dataset/environment registration
revision. It does **not** mean the base Gym task `Walker2d-v2`, and it must not be
translated to Gymnasium's `Walker2d-v4` or `Walker2d-v5`.

The former implementation combined a D4RL-v2 HDF5 file with a separately
created Gymnasium-v4 MDP and manually copied normalization constants. Although
that could run on modern macOS, it was not the official RPEX protocol: offline
data, online interaction, and evaluation did not come from the same registered
D4RL environment.

The strict implementation is equivalent to:

```python
import gym
import d4rl  # registers the complete D4RL IDs

env = gym.make("walker2d-medium-replay-v2")
dataset = d4rl.qlearning_dataset(env, terminate_on_end=False)
normalized = d4rl.get_normalized_score(
    "walker2d-medium-replay-v2", returns
) * 100.0
```

There is no automatic Gymnasium-v4/v5 fallback.

## Pinned environment

Exact reproduction is recommended on Linux x86_64. The official stack depends
on `mujoco_py`, which expects a local MuJoCo 2.1 installation and is frequently
not buildable on Apple Silicon. A Gymnasium port is not an equivalent fallback.

```bash
cd /Users/seobeom/programming/project/corruption_robust_o2o
conda env create -f environment-rpex-v2.yml
conda activate corruption-rpex-v2
```

Install MuJoCo 2.1 under `~/.mujoco/mujoco210` according to the
[`mujoco-py` installation instructions](https://github.com/openai/mujoco-py#install-mujoco),
then expose its libraries on Linux:

```bash
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin:/usr/lib/nvidia"
```

The dependency file pins Python-compatible packages and D4RL at exactly:

```text
d842aa194b416e564e54b0730d9f934e3e32f854
```

The default PyPI `torch==2.5.1` wheel is suitable for CPU execution. For a CUDA
12.1 Linux host, install the official CUDA wheel after creating the environment:

```bash
python -m pip install --force-reinstall torch==2.5.1 \
  --index-url https://download.pytorch.org/whl/cu121
```

Do not install Gymnasium or the native `mujoco` Python package into this legacy
environment as replacements for Gym/mujoco_py.

## Dataset and MC returns

`--dataset-dir` is passed to `d4rl.set_dataset_path()`. D4RL itself resolves and
loads the dataset; this repository does not construct or download a D4RL URL
manually.

For Cal-QL, return-to-go is calculated on the raw `env.get_dataset()`
trajectories using their `terminals` and `timeouts` boundaries. Only afterward
are the exact indices retained by
`d4rl.qlearning_dataset(..., terminate_on_end=False)` selected. This prevents a
timeout-removed episode from leaking rewards from the next trajectory.

## Verification and commands

The integration smoke test checks package pins and the D4RL VCS commit, creates
the complete ID, performs one legacy Gym step, loads the official q-learning
dataset, computes a normalized score, and prints reproducibility metadata:

```bash
python scripts/smoke_rpex_d4rl_v2.py \
  --env-name hopper-medium-replay-v2 \
  --seed 0
```

Offline pretraining only:

```bash
python run_experiment.py \
  --algorithm rpex \
  --env-name hopper-medium-replay-v2 \
  --corruption clean \
  --stage offline \
  --offline-steps 500000 \
  --seed 0
```

Offline pretraining followed by online fine-tuning in one process:

```bash
python run_experiment.py \
  --algorithm rpex \
  --env-name hopper-medium-replay-v2 \
  --corruption clean \
  --stage both \
  --offline-steps 500000 \
  --online-steps 500000 \
  --seed 0
```

All nine algorithms for the same setting:

```bash
python run_all_algorithms.py \
  --env-name hopper-medium-replay-v2 \
  --corruption clean \
  --seeds 42 \
  --offline-steps 500000 \
  --online-steps 500000
```

## Saved provenance

Strict and local results share the canonical `results/comparisons/` tree; the
protocol is recorded in each run's `config.json` rather than encoded in its path.
Every run's `config.json` records the protocol, complete requested ID,
`env.spec.id`, unwrapped class/module, package versions, required and installed
D4RL commits, dataset URL/path/SHA-256 when available, dimensions, dataset size,
maximum episode length, and seed. A checkpoint also carries the protocol and is
rejected if used with a different backend.
