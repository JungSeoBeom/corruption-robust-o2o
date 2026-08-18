# Corruption-Robust Offline-to-Online RL Benchmark

This repository provides a unified benchmark for comparing nine algorithms under
`clean`, `random`, and `adversarial` conditions. It uses the D4RL MuJoCo
benchmarks and data-corruption protocol from RPEX, with a shared command-line
interface and logging format across all algorithms.

The default protocol is explicitly `rpex_d4rl_v2_legacy`: Gym 0.23.1, the full
D4RL-v2 environment ID, `mujoco_py`, and D4RL commit
`d842aa194b416e564e54b0730d9f934e3e32f854`. See
[RPEX_D4RL_V2_PROTOCOL.md](docs/RPEX_D4RL_V2_PROTOCOL.md) for the exact protocol,
installation, metadata, and platform limitations.

Long training runs were not executed while preparing this repository. Use the
commands below in the pinned legacy environment; exact reproduction is expected
to require a Linux x86_64 host rather than an Apple Silicon MacBook.

### KAIST RL Lab GCP Slurm (CPU only)

The files under `slurm/` install and run the strict legacy environment entirely
on the `cpu` partition. They do not request a GPU, and they enforce both
`--device cpu` and `MUJOCO_PY_FORCE_CPU=1`. Submit them from the repository root
on `slurm-login-001`:

```bash
# One-time setup per cluster user. This also runs the unit suite and the real
# D4RL/MuJoCo smoke.
sbatch --wait slurm/setup_cpu.sbatch

# Short end-to-end Slurm smoke: dataset, offline update, online interaction,
# online replay updates, evaluation, and checkpoints.
sbatch --wait slurm/smoke_cpu.sbatch

# Full default RPEX clean run (500k offline + 500k online).
sbatch slurm/run_cpu.sbatch
```

Pass normal `run_experiment.py` arguments after the batch script to select a
different experiment. Slurm resource overrides go before the script:

```bash
sbatch --time=12:00:00 slurm/run_cpu.sbatch \
  --algorithm uwmsg \
  --env-name hopper-medium-replay-v2 \
  --corruption random \
  --corruption-target rewards \
  --stage both \
  --offline-steps 500000 \
  --online-steps 500000 \
  --seed 0
```

The shared environment is stored under `~/.local/share/micromamba`, MuJoCo 2.1
under `~/.mujoco/mujoco210`, and datasets under `~/.d4rl/datasets`. All are
visible from the login and compute nodes through the shared home directory.
Batch output is written to `slurm-*.out` in the submission directory. The CPU
nodes are Spot VMs. Slurm may requeue the wrapper after preemption, but that
restarts the command from the beginning; periodic checkpoints survive on the
shared home directory but are not selected automatically. An online checkpoint
restores model state but does not exactly restore the live MuJoCo state or
online replay buffer.

For local Apple Silicon execution, the separate `local_gymnasium_v4` protocol
uses Gymnasium v4 with native MuJoCo and reads the cached D4RL-v2 HDF5 dataset
directly. This is convenient for local experiments but is not an exact legacy
RPEX environment reproduction.

## 1. Algorithms

The CLI names and reference papers or configurations are listed below.

| # | `--algorithm` | Paper/configuration | Offline → online behavior |
|---|---|---|---|
| 1 | `rpex` | *Robust Policy Expansion for Offline-to-Online RL under Diverse Data Corruption* | RIQL pretraining followed by IPW-based robust policy expansion |
| 2 | `riql_pex` | RIQL+PEX ablation from RPEX | RIQL pretraining followed by PEX without IPW |
| 3 | `riql_naive` | *Towards Robust Offline Reinforcement Learning under Diverse Data Corruption* | Applies the RIQL objective directly to online replay |
| 4 | `uwmsg` | *Corruption-Robust Offline Reinforcement Learning with General Function Approximation* | Applies the UWMSG objective directly to online replay |
| 5 | `pex` | *Policy Expansion for Bridging Offline-to-Online Reinforcement Learning* | IQL pretraining followed by PEX |
| 6 | `cal_ql` | *Cal-QL: Calibrated Offline RL Pre-Training for Efficient Online Fine-Tuning* | MC-return-calibrated CQL followed by online Cal-QL |
| 7 | `wsrl` | *Efficient Online Reinforcement Learning Fine-Tuning Need Not Retain Offline Data* | CQL pretraining, frozen-policy warmup, and online-only SAC |
| 8 | `ro2o` | *Towards Robust Offline-to-Online Reinforcement Learning via Uncertainty and Smoothness* | Q-ensemble/smoothness pretraining followed by replay-based online reduction |
| 9 | `pessimistic_q_ensemble` | *Offline-to-Online Reinforcement Learning via Balanced Replay and Pessimistic Q-Ensemble* | Ensemble+CQL pretraining followed by density-ratio-balanced replay |

As requested, the online stages of `riql_naive` and `uwmsg` store newly collected
transitions in a replay buffer and train on mini-batches using the same objective
as their offline updates. Following the default comparison protocol in the RPEX
code, the RIQL variants, UWMSG, WSRL, and RO2O use online replay only by default.
PEX and Cal-QL mix offline and online data at a 50:50 ratio. Pessimistic
Q-Ensemble uses density-ratio priority over a combined offline/online priority
mass; `--offline-ratio` sets its initial target mass. Use
`--pqe-replay-mode uniform` for the explicit fixed-ratio ablation.

## 2. Reproducibility environment

The official RPEX protocol is a legacy stack. Use Linux x86_64 for exact
reproduction; `mujoco_py` and its transitive PyBullet dependency are often not
buildable on Apple Silicon. The code intentionally does not fall back to
Gymnasium v4/v5 on macOS.

```bash
cd /Users/seobeom/programming/project/corruption_robust_o2o
conda env create -f environment-rpex-v2.yml
conda activate corruption-rpex-v2
```

Install MuJoCo 2.1 at `~/.mujoco/mujoco210` and configure its library path as
described in [the protocol document](docs/RPEX_D4RL_V2_PROTOCOL.md). Then run:

```bash
python scripts/smoke_rpex_d4rl_v2.py \
  --env-name hopper-medium-replay-v2 \
  --seed 0
```

The smoke test verifies Gym 0.23.1, NumPy 1.23.5, the exact D4RL commit,
`mujoco_py`, the complete environment registration, dataset loading, and score
normalization. `--dataset-dir /path/to/datasets` changes the D4RL cache through
`d4rl.set_dataset_path()`.

The default device option is `--device auto`:

1. `cuda:N` is selected and `torch.cuda.set_device` is called only when CUDA is
   actually available.
2. PyTorch MPS is selected when it is available on the Mac.
3. The code falls back to the CPU when neither accelerator is available.

There are no unconditional `.cuda()` calls. Use `--device cpu` if you encounter
MPS operation compatibility issues or need stricter reproducibility.

Do not interpret a Gymnasium-v4/v5 run as an RPEX reproduction. The D4RL `-v2`
suffix is a D4RL registration/dataset revision, not a request to create the base
Gym or Gymnasium `Walker2d-v2/v4/v5` task.

## 3. Running one experiment

The current default benchmark schedule is:

- Offline: `500,000` gradient updates
- Online: `500,000` environment steps
- Online updates: one gradient update per environment step by default

Use `--offline-steps`, `--online-steps`, and `--updates-per-step` to change these
values. A single command with `--stage both` runs offline pretraining and online
fine-tuning sequentially with the same agent:

### Clean

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

For `clean`, the corruption target is automatically set to `none`, so
`--corruption-target` is not required.

### Random corruption

```bash
python run_experiment.py \
  --algorithm uwmsg \
  --env-name hopper-medium-replay-v2 \
  --corruption random \
  --corruption-target dynamics \
  --stage both \
  --seed 0
```

### Adversarial corruption

```bash
python run_experiment.py \
  --algorithm riql_naive \
  --env-name hopper-medium-replay-v2 \
  --corruption adversarial \
  --corruption-target dynamics \
  --stage both \
  --seed 0
```

The following corruption targets are supported:

- `observations`: current observations stored in replay
- `actions`: actions stored in replay
- `rewards`: rewards stored in replay
- `dynamics`: next observations stored in replay
- `mixed`: allocate corrupted transitions across all four targets using
  `--mixed-ratios`

Online corruption is explicitly replay-only poisoning. The bounded clean policy
action is executed in `env.step()`. The selected observation/action/reward/next
observation field is then corrupted only in the transition stored in replay.
Thus clean runs require exact equality between executed and replayed actions,
while action-corruption runs intentionally record a mismatch.

The default corruption parameters match RPEX:

- Offline corruption rate: `0.3`
- Online corruption rate: `0.5`
- Corruption range: `1.0`
- Random reward corruption: replace selected rewards with `Uniform(-30, 30)`
- Adversarial reward corruption: change selected rewards to
  `-range × reward`

Example with custom corruption parameters:

```bash
python run_experiment.py \
  --algorithm rpex \
  --env-name walker2d-medium-replay-v2 \
  --corruption random \
  --corruption-target observations \
  --offline-corruption-rate 0.2 \
  --online-corruption-rate 0.4 \
  --corruption-range 0.5
```

### Mixed corruption

`mixed` is a corruption target and can be used with either `random` or
`adversarial` corruption. The four `--mixed-ratios` values are ordered as
`observations actions rewards dynamics` and must sum to `1.0`.

```bash
python run_experiment.py \
  --algorithm rpex \
  --env-name hopper-medium-replay-v2 \
  --corruption random \
  --corruption-target mixed \
  --mixed-ratios 0.1 0.2 0.3 0.4 \
  --stage both \
  --seed 0
```

The ratios allocate transitions *within* the corrupted subset. For example, with
the default offline corruption rate of `0.3`, the expected fractions of the full
offline dataset are 3% observations, 6% actions, 9% rewards, and 12% dynamics.
Each selected transition is assigned to exactly one target. Actual per-target
counts and fractions are written to `config.json` under `offline_corruption`.

### Adversarial attack checkpoint

Adversarial corruption of `observations`, `actions`, and `dynamics` requires an
EDAC gradient oracle, as in RPEX. Checkpoints for the following three
environments are discovered automatically from the original `RIQL-main`
directory:

- `halfcheetah-medium-replay-v2`
- `hopper-medium-replay-v2`
- `walker2d-medium-replay-v2`

For other dataset types, provide a checkpoint explicitly:

```bash
python run_experiment.py \
  --algorithm ro2o \
  --env-name hopper-medium-v2 \
  --corruption adversarial \
  --corruption-target actions \
  --attack-checkpoint /absolute/path/to/EDAC/2999.pt
```

Attack results are stored in `results/attack_cache/<protocol>/`. Cache keys hash
the dataset, attack checkpoint, target/rate/range, seed, attack steps and step
size, norm, preprocessing-relevant MC settings, and implementation version.
`--attack-min-step-size` is explicit and defaults to zero; no hidden runtime
lower bound is applied. Add `--force-regenerate-attack` to regenerate a cache.

### Correctness-sensitive modes

- `--action-distribution tanh_gaussian` is the safe RPEX/PEX default. It uses a
  bounded transformed Gaussian and Jacobian-corrected log density.
  `legacy_gaussian` is unbounded reproduction-only behavior; environment actions
  are still clipped before execution.
- `--evaluation-mode deterministic_diagnostic` uses policy means and a
  deterministic expansion branch. `method_faithful` preserves stochastic RPEX
  expansion, and `both` logs both. Evaluation saves/restores Python, NumPy,
  PyTorch CPU, and CUDA RNG state.
- `--mc-return-source post_corruption` is the Cal-QL default. Reward corruption
  recomputes return-to-go within trajectory boundaries, and corrupted
  state/action/next-state rows are excluded from calibration.
  `legacy_pre_corruption` is an explicit reproduction mode.
- `--backup-entropy` enables the entropy term in the Cal-QL Bellman backup. The
  default is disabled, matching the task configuration used by the reference.
- `--state-normalization` accepts `standard`, `robust_median_mad`, or `none`.
  Statistics are fitted after corruption and serialized in checkpoints.

## 4. Running the offline and online stages separately

Run the offline stage only:

```bash
python run_experiment.py \
  --algorithm riql_pex \
  --env-name halfcheetah-medium-replay-v2 \
  --corruption random \
  --corruption-target rewards \
  --stage offline \
  --seed 0
```

Then run the online stage using `checkpoints/offline/final.pt` from the resulting
run directory:

```bash
python run_experiment.py \
  --algorithm riql_pex \
  --env-name halfcheetah-medium-replay-v2 \
  --corruption random \
  --corruption-target rewards \
  --stage online \
  --checkpoint /absolute/path/to/run/checkpoints/offline/final.pt \
  --seed 0
```

The final offline checkpoint is stored at
`checkpoints/offline/final.pt` inside the run directory. The program fails
immediately if the checkpoint algorithm, environment, or
observation/action dimensions do not match the current command. The state
normalization mode, location, and scale are also restored from the checkpoint.
New runs write both `config.json` and `resolved_config.json`.

## Quick training diagnostics

The lightweight runner checks finite updates, action/replay invariants, and
basic return health:

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
python scripts/diagnose_training.py \
  --algorithm rpex \
  --env hopper-medium-replay-v2 \
  --offline-steps 20 \
  --online-steps 30 \
  --corruption-rate 0 \
  --quick
```

Each diagnostic run writes `diagnostics_summary.json`,
`diagnostics_summary.csv`, and `resolved_config.json` in its run directory.

### Checkpoint intervals and retention

Checkpoints are separated by algorithm through the run directory and by phase
inside each run:

```text
results/comparisons/<env>/<corruption>/<target>/<comparison_id>/runs/
└── <algorithm>/<corruption>/<target>/<env>/seed_<seed>/<run_id>/
└── checkpoints/
    ├── offline/
    │   ├── step_000100000.pt
    │   └── final.pt
    └── online/
        ├── step_000100000.pt
        └── final.pt
```

The shared periodic interval defaults to `100,000`. Five periodic checkpoints
per phase are retained by default; `final.pt` is always retained.

```bash
python run_experiment.py \
  --algorithm rpex \
  --env-name hopper-medium-replay-v2 \
  --corruption clean \
  --stage both \
  --offline-checkpoint-period 200000 \
  --online-checkpoint-period 50000 \
  --keep-last-checkpoints 3
```

- `--checkpoint-period N`: shared offline/online interval
- `--offline-checkpoint-period N`: override the offline interval
- `--online-checkpoint-period N`: override the online interval
- `--keep-last-checkpoints K`: retain the newest `K` periodic checkpoints per
  phase
- Interval `0`: disable periodic checkpoints for that phase
- `--keep-last-checkpoints 0`: keep every periodic checkpoint

## 5. Run all algorithms for one fixed setting

Use `run_all_algorithms.py` to run all nine algorithm classes for one fixed
environment and corruption configuration. The default stage is `both`, so each
algorithm performs offline pretraining followed immediately by online
fine-tuning.

```bash
python run_all_algorithms.py \
  --env-name hopper-medium-replay-v2 \
  --corruption random \
  --corruption-target mixed \
  --mixed-ratios 0.1 0.2 0.3 0.4 \
  --seeds 0,1,2 \
  --offline-steps 500000 \
  --online-steps 500000
```

Any unrecognized arguments are forwarded to every `run_experiment.py` command,
which allows shared step, evaluation, replay, and checkpoint settings. Use
`--dry-run` to print all generated commands without running them:

```bash
python run_all_algorithms.py \
  --env-name hopper-medium-replay-v2 \
  --corruption adversarial \
  --corruption-target dynamics \
  --seeds 0 \
  --dry-run
```

Each comparison is stored separately:

```text
results/comparisons/<env>/<corruption>/<target>/<comparison_id>/
├── runs/
│   └── <algorithm>/...
├── comparison_offline_online.png
├── comparison_offline.png
├── comparison_online.png
├── comparison_{offline_online,offline,online}.csv
├── final_scores.csv
├── timing.csv
└── manifest.json
```

- The three `comparison_*.png` files show the combined, offline-only, and
  online-only curves. They are refreshed after every evaluation, including the
  currently running algorithm.
- The matching CSV files contain mean/std/count at every evaluation step.
- `final_scores.csv`: final normalized/raw return and runtime mean/std by
  algorithm
- `timing.csv`: start/end time and elapsed time for every algorithm/seed run
- `manifest.json`: commands, return codes, per-algorithm timing summaries,
  overall timing, and artifact paths

After each seed run, the command prints a `RUN_FINISHED` line. After all seeds
for one algorithm finish, it prints an `ALGORITHM_FINISHED` line containing the
algorithm's accumulated runtime. At the end, `ALGORITHM_TIMING_SUMMARY` lists
every completed algorithm followed by the overall start, end, and elapsed time.

Use `--comparison-name NAME` to set the final directory name and `--keep-going`
to continue with the remaining algorithms if one run fails.

### Hopper 5 x 5 experiment suite

Run RPEX, RIQL naive, WSRL, Cal-QL, and Pessimistic Q-Ensemble on Hopper for
the clean setting and all four individual random-corruption targets:

```bash
conda activate corruption
python run_55_experiment.py
```

This runs 25 experiments for the default seed, with 500,000 offline updates and
500,000 online environment steps per experiment. Its default protocol is
`local_gymnasium_v4`, so the existing `corruption` environment and
`~/.d4rl/datasets/hopper_medium_replay-v2.hdf5` are sufficient. Use
`--seeds 0,1,2` for a multi-seed suite, `--dry-run` to print all generated
experiment commands, and `--keep-going` to continue after a failed algorithm or
setting. Additional experiment options such as `--device cpu` are forwarded to
every run. Pass `--protocol rpex_d4rl_v2_legacy` only on a machine with the
pinned legacy environment.

## 6. Full environment/corruption matrix

Run all algorithms under clean, random, and adversarial conditions for one
environment, the `dynamics` target, and three seeds:

```bash
python run_matrix.py \
  --envs hopper-medium-replay-v2 \
  --targets dynamics \
  --seeds 0,1,2
```

For each algorithm, this runs one `clean` experiment, one `random dynamics`
experiment, and one `adversarial dynamics` experiment.

Print the generated commands without starting experiments:

```bash
python run_matrix.py \
  --envs hopper-medium-replay-v2 \
  --targets dynamics \
  --seeds 0,1,2 \
  --dry-run
```

Compare all four individual corruption targets:

```bash
python run_matrix.py \
  --envs halfcheetah-medium-replay-v2,hopper-medium-replay-v2,walker2d-medium-replay-v2 \
  --targets observations,actions,rewards,dynamics \
  --seeds 0,1,2
```

Additional experiment arguments are forwarded to each run. To check the setup
with a small experiment on a supported legacy-runtime host:

```bash
python run_matrix.py \
  --envs hopper-medium-replay-v2 \
  --targets rewards \
  --seeds 0 \
  --offline-steps 1000 \
  --online-steps 1000 \
  --eval-period 500 \
  --eval-episodes 2
```

This small configuration is intended only to verify the installation and code
paths. Do not use it for paper-level performance comparisons.

## 7. Performance logs and visualization

Standalone, matrix, `run_all`, and `run_55` runs all use the same comparison
structure:

```text
results/
└── comparisons/<env>/<corruption>/<target>/<comparison_id>/
    ├── comparison_offline_online.png
    ├── comparison_offline.png
    ├── comparison_online.png
    └── runs/<algorithm>/<corruption>/<target>/<env>/seed_<seed>/<timestamp>_<id>/
    ├── config.json
    ├── result.log
    ├── metrics.csv
    ├── train_metrics.jsonl
    ├── performance.png
    ├── summary.json
    └── checkpoints/
        ├── offline/
        │   └── final.pt
        └── online/
            └── final.pt
```

- `metrics.csv`: evaluation step, raw return, D4RL normalized return, standard
  deviation, and cumulative elapsed time
- `train_metrics.jsonl`: losses, Q statistics, uncertainty,
  policy-expansion selection ratio, and other training metrics
- `performance.png`: normalized-return curve for one run
- `summary.json`: start/end timestamps, total elapsed time, final performance,
  and success/failure status
- `config.json`: complete arguments, exact protocol and D4RL environment ID,
  registered class, D4RL commit, dataset URL/path/SHA-256 when available, and
  installed Python/NumPy/PyTorch/Gym/D4RL/mujoco-py/h5py versions

Plot the mean ± standard deviation across multiple seeds and algorithms:

```bash
python plot_results.py \
  --results-dir results/comparisons \
  --env-name hopper-medium-replay-v2 \
  --corruption random \
  --target dynamics \
  --phase online \
  --output results/comparisons/hopper_random_dynamics.png
```

A CSV summary with the same base filename is generated together with the plot.

## 8. Timing information

The following three lines are printed at the end of every run, regardless of
whether it succeeds or fails:

```text
START_TIME: 2026-07-31 10:00:00
END_TIME: 2026-07-31 20:30:00
ELAPSED: 10:30:00 (37800.000 seconds)
```

The same values are stored in `summary.json`. The `elapsed_seconds` values in
`metrics.csv` and `train_metrics.jsonl` can be used to compare the
performance-versus-time trade-off during training.

Example final output from `run_all_algorithms.py`:

```text
ALGORITHM_TIMING_SUMMARY:
  ALGORITHM: RPEX (rpex) | runs=1/1 completed | START=2026-07-31 10:00:00 | END=2026-07-31 12:10:00 | ELAPSED=02:10:00 (7800.000 seconds)
  ALGORITHM: RIQL+PEX (riql_pex) | runs=1/1 completed | START=2026-07-31 12:10:00 | END=2026-07-31 14:05:00 | ELAPSED=01:55:00 (6900.000 seconds)
START_TIME: 2026-07-31 10:00:00
END_TIME: 2026-07-31 14:05:05
ELAPSED: 04:05:05 (14705.000 seconds)
```

## 9. Important points for performance and runtime comparisons

- Keep the environment, seed, corruption rate/range, offline/online step counts,
  and number of evaluation episodes identical across all nine algorithms.
- On supported hardware, RO2O with `--ro2o-sample-size 20` and the 10-critic
  UWMSG/RO2O configurations are particularly slow. Reduce the sample size or
  critic count only during exploratory runs, and restore the original values
  for final comparisons.
- MPS and CPU results may not be exactly identical because of floating-point
  implementation differences. Use the same device for all results in a
  comparison table.
- Adversarial offline-attack generation time is included in the total `ELAPSED`
  time. Check `offline_corruption.loaded_from_cache` in `config.json` to
  determine whether the attack cache was used.
- Before starting long experiments, run `scripts/smoke_rpex_d4rl_v2.py` to
  verify Gym/mujoco_py, the complete D4RL ID, dataset, normalization, and saved
  provenance.

## 10. Implementation sources

The provided `RPEX`, `RIQL-main`, and `UWMSG-main` codebases are the primary
references for the unified objectives and default values. The remaining
baselines were adapted to the shared PyTorch/replay interface using their
official public implementations and algorithm descriptions.

- RPEX: <https://github.com/felix-thu/RPEX>
- Pinned D4RL environment registry, dataset conversion, and normalization:
  <https://github.com/rail-berkeley/d4rl/tree/d842aa194b416e564e54b0730d9f934e3e32f854>
- RIQL: provided `RIQL-main` directory
- UWMSG: provided `UWMSG-main` directory
- WSRL: <https://github.com/zhouzypaul/wsrl>
- RO2O: <https://github.com/BattleWen/RO2O>
- Balanced Replay + Pessimistic Q-Ensemble:
  <https://github.com/shlee94/Off2OnRL>
# corruption=robust-o2o
