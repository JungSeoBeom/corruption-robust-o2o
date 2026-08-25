# Modern runtime contract

The repository uses one runtime for local macOS and Linux execution:

- Python 3.13
- Gymnasium 1.3.0
- native MuJoCo 3.11.0
- NumPy 2.5.2
- PyTorch 2.13.0
- direct HDF5 loading of the official D4RL-v2 locomotion datasets

The exact environment is declared in `environment.yml` and the Python package
pins are in `requirements.txt`. Legacy Gym, the D4RL Python package,
`mujoco_py`, PyBullet, Cython build shims, and a separately installed MuJoCo
2.1 tree are outside this contract.

## Task mapping

| Dataset ID | Online/evaluation task |
|---|---|
| `halfcheetah-medium-replay-v2` | `HalfCheetah-v4` |
| `hopper-medium-replay-v2` | `Hopper-v4` |
| `walker2d-medium-replay-v2` | `Walker2d-v4` |

The dataset suffix and simulator-task suffix describe different versioned
artifacts. Dataset IDs retain `-v2`; they are not rewritten to `-v4` or `-v5`.
The v4 task is retained intentionally because moving to v5 changes the MDP and
would exceed an API-only modernization.

## Data contract

`scripts/download_d4rl_datasets.py` downloads files from the official Berkeley
offline-RL dataset host. Files are streamed to a temporary file in the target
directory and atomically renamed after a successful download. Existing files
are reused unless `--force` is supplied.

The loader preserves the benchmark's D4RL q-learning conversion behavior,
including terminal/timeout boundaries and index-aligned Monte Carlo returns.
Reference min/max returns are used to retain D4RL-style normalized reporting
without importing the legacy D4RL package.

## Research scope

This runtime is the common execution basis for comparisons produced by this
repository. Package metadata, environment IDs, dataset paths and hashes, and
seeds are recorded with run outputs. Because the simulator binding and task
implementation differ from the historical Gym/`mujoco_py` stack, results from
the two stacks should not be mixed in one aggregate or called bitwise-equivalent
paper reproduction results.
