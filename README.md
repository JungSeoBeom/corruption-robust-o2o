# Corruption-Robust Offline-to-Online RL

This repository compares nine offline-to-online reinforcement-learning methods
on the RPEX locomotion benchmark. All methods share the same D4RL-v2 offline
datasets, corruption pipeline, online interaction environments, evaluation
schedule, logging format, and configurable offline/online budgets.

The runtime has one modern backend: Gymnasium with the native MuJoCo Python
package. Legacy Gym, D4RL's Python package, `mujoco_py`, PyBullet, and a local
MuJoCo 2.1 installation are not required. The migration changes package and
environment APIs only; it does not intentionally change an algorithm objective,
corruption rule, replay rule, optimizer schedule, or training budget.

## Algorithms

| # | CLI name | Paper or configuration |
|---:|---|---|
| 1 | `rpex` | *Robust Policy Expansion for Offline-to-Online RL under Diverse Data Corruption* |
| 2 | `riql_pex` | RIQL+PEX ablation defined by RPEX |
| 3 | `riql_naive` | *Towards Robust Offline Reinforcement Learning under Diverse Data Corruption* |
| 4 | `uwmsg` | *Corruption-Robust Offline Reinforcement Learning with General Function Approximation* |
| 5 | `pex` | *Policy Expansion for Bridging Offline-to-Online Reinforcement Learning* |
| 6 | `cal_ql` | *Cal-QL: Calibrated Offline RL Pre-Training for Efficient Online Fine-Tuning* |
| 7 | `wsrl` | *Efficient Online Reinforcement Learning Fine-Tuning Need Not Retain Offline Data* |
| 8 | `ro2o` | *Towards Robust Offline-to-Online Reinforcement Learning via Uncertainty and Smoothness* |
| 9 | `pessimistic_q_ensemble` | *Offline-to-Online Reinforcement Learning via Balanced Replay and Pessimistic Q-Ensemble* |

`calql` and `cal-ql` are accepted aliases for `cal_ql`; `pqe` is an alias for
`pessimistic_q_ensemble`.

PEX, RIQL+PEX, UWMSG, and RO2O have a separate
[implementation audit](docs/optional-algorithm-audit.md). Their execution tests
pass, but unresolved source differences and reproducibility issues mean that
the nine executable names are not nine verified paper baselines.

RIQL-naive and UWMSG use replay-buffer-based offline reduction during online
fine-tuning: online transitions are stored in replay and the same offline
objective is applied to sampled online batches. The other methods keep their
method-specific online behavior.

## Environment setup

Run the following commands from this repository directory:

```bash
cd /path/to/corruption_robust_o2o
conda env create -f environment.yml
conda activate corruption-robust-o2o
python -m pip check
```

The environment is pinned to a mutually compatible modern stack:

| Package | Version |
|---|---:|
| Python | 3.13 |
| NumPy | 2.5.2 |
| PyTorch | 2.13.0 |
| Gymnasium | 1.3.0 |
| MuJoCo | 3.11.0 |
| h5py | 3.16.0 |
| pandas | 3.0.5 |
| Matplotlib | 3.11.1 |
| JupyterLab | 4.6.3 |

See [the modern runtime contract](docs/MODERN_RUNTIME.md) for the task mapping,
dataset boundary handling, and research-interpretation details.

To update an existing environment created from this file:

```bash
conda env update --name corruption-robust-o2o --file environment.yml --prune
conda activate corruption-robust-o2o
python -m pip check
```

For a notebook kernel:

```bash
python -m ipykernel install --user \
  --name corruption-robust-o2o \
  --display-name "Python (corruption-robust-o2o)"
```

`--device auto` selects CUDA only when available, otherwise Apple MPS when
available, and otherwise CPU. On macOS, use `--device mps` explicitly for MPS
or `--device cpu` if an operation is unsupported by MPS.

## Environments and datasets

The commands in this README use the three medium-replay benchmark dataset IDs:

- `halfcheetah-medium-replay-v2`
- `hopper-medium-replay-v2`
- `walker2d-medium-replay-v2`

Download all three official medium-replay D4RL-v2 HDF5 files:

```bash
python scripts/download_d4rl_datasets.py
```

Download only selected datasets or use a custom cache directory:

```bash
python scripts/download_d4rl_datasets.py \
  --env-name halfcheetah-medium-replay-v2 hopper-medium-replay-v2 \
  --dataset-dir /path/to/d4rl/datasets
```

The same loader also accepts the medium and medium-expert D4RL-v2 variants for
these three domains. Supply those full IDs explicitly to the download script.
The default cache is `~/.d4rl/datasets`; pass the same custom directory to
training with `--dataset-dir /path/to/d4rl/datasets`.

The D4RL `-v2` suffix identifies the offline dataset revision. Online
interaction uses the corresponding Gymnasium MuJoCo v4 task internally
(`HalfCheetah-v4`, `Hopper-v4`, or `Walker2d-v4`). The repository deliberately
does not switch those tasks to v5: v5 changes the environment definition, so
doing so would change the experimental MDP rather than merely modernize syntax.

## Corruption settings

`--corruption` selects the corruption mechanism:

- `clean`: no corruption; the target is automatically `none`.
- `random`: random RPEX-style data corruption.
- `adversarial`: adversarial RPEX-style data corruption.

For a non-clean run, `--corruption-target` accepts `observations`, `actions`,
`rewards`, `dynamics`, or `mixed`. Mixed corruption allocates examples across
the four targets in that order. Ratios must be non-negative and sum to one:

```bash
--corruption-target mixed --mixed-ratios 0.1 0.2 0.3 0.4
```

The main severity controls are `--offline-corruption-rate` (default `0.3`),
`--online-corruption-rate` (default `0.5`), and `--corruption-range` (default
`1.0`).

Adversarial corruption of observations, actions, or dynamics also needs the
environment-specific EDAC attacker checkpoint. In the original three-folder
workspace it is found automatically below
`../RIQL-main/pretrained_model/EDAC/EDAC_baseline_seed0-<env>/2999.pt` and its
pinned SHA256 is verified. If this repository is moved on its own, pass the
same weights explicitly (these options are also forwarded by
`run_all_algorithms.py`):

```bash
--attack-checkpoint /path/to/2999.pt \
--attack-checkpoint-sha256 "$(shasum -a 256 /path/to/2999.pt | awk '{print $1}')"
```

Reward-only adversarial corruption does not load an attacker checkpoint.

## Five-baseline research benchmark contract

The `research_benchmark` suite contains exactly `rpex`, `riql_naive`, `wsrl`,
`cal_ql`, and the canonical `pessimistic_q_ensemble` name. Its source anchors
are [RPEX/RIQL `35da71e`](https://github.com/felix-thu/RPEX/tree/35da71ee5151b6179d21b9a2b4ce1b6408aedd04),
[WSRL `ad4dc12`](https://github.com/zhouzypaul/wsrl/tree/ad4dc1248a138bc15d6e053f2d1dba1b8cfbaca2),
[Cal-QL `ac6eafe`](https://github.com/nakamotoo/Cal-QL/tree/ac6eafec22e8d60836573e1f488c7f626ce8a77e),
and [Off2OnRL `6f298fa`](https://github.com/shlee94/Off2OnRL/tree/6f298fa9ef040d725067d0f2775022bd2900d635).
These are source-aligned PyTorch/runtime ports, not claims of bitwise paper
score reproduction. Cal-QL is a locomotion adaptation and PQE ports the public
v0 recipe to D4RL-v2.

The common interaction contract is:

1. Keep the policy/expansion proposal unchanged for the method's internal Q,
   IPW, and log-probability calculations.
2. Require that proposal to be finite, clip a copy to the environment's action
   bounds, and pass only that executed action to `env.step`.
3. Copy the resulting clean transition, poison only the replay copy, and choose
   the next real action from the environment's clean next observation.

Action poisoning is label poisoning: an out-of-range replay action is retained
for critic training and is never sent back to the simulator. Logs separately
record proposal, executed-environment, and poisoned-replay action OOB rates.
NaN/Inf actions fail explicitly; they are not repaired with `nan_to_num`.

Offline corruption is generated in raw dataset coordinates before learner
normalization. Online replay corruption also starts in raw transition
coordinates, then the learner normalizer is applied exactly once. Dataset-std
scales are frozen from the clean offline artifact (the source and actual
population standard deviations are serialized); the retained RPEX online
observation/dynamics rule uses its declared unit scale. Mixed corruption first
selects a transition and then assigns exactly one field. Reward-only
adversarial mixed runs do not instantiate EDAC.

The EDAC objective transforms raw attacked states with preprocessing declared
by the checkpoint. The supplied pinned EDAC payloads do not contain such
metadata, so the runner uses identity preprocessing and records it as
`checkpoint_metadata_missing_identity_unverified`; it does not substitute the
learner normalizer. Attack version
`corruption_v8_raw_coordinates_private_rng` retains the source Adam optimizer
recreation, 100 offline/2 online steps, and source step sizes. The research
adaptation uses resumable phase-private Torch RNGs and applies standard deviation
once when mapping a dimensionless perturbation to its declared budget. The
`official_code_reference` diagnostic retains the upstream fresh-generator and
double-std quirks, so the research attack is not labeled exact RNG parity.
Selected-transition rates and actual-value-change rates are both stored, and a
zero selection rate leaves the artifact byte-for-byte unchanged. Reward
replacement at epsilon zero remains replacement, not a clean shortcut.

Reliability revision (after `15d44da`): research vector attacks use
`phase_private_torch_v1`: offline seed = `corruption_seed`, online seed =
`(corruption_seed + 0x4F324F) % 2**63`. Both online perturbation initialization
and the stochastic EDAC dynamics policy use that persistent online stream.
Offline cache hit/miss/regeneration therefore cannot change the online attack
sequence. The RNG semantics field changes the research vector-attack cache
key and manifest identity; old artifacts remain untouched and are not reused
under the new key. The offline draw mapping itself is unchanged. Official-code
diagnostic RNG quirks remain unchanged.

Checkpoints record `attack_rng_schema`; online checkpoints store both private
states. Offline checkpoints may omit them because the unused online state is
deterministically reconstructible. Old single-stream research adversarial
checkpoints cannot exact-resume; compatible weights can still initialize a new
run. PQE also records `pqe_numerics_version` as
`centered_moments_strict_priorities_v1`: centered moment variance and a positive floor
before square root prevent cancellation/NaN gradients without detaching actors
or changing the pre-tanh Gaussian. Invalid priorities/weights now raise rather
than silently becoming uniform probabilities; valid floors, clipping, and the
initial online priority formula are unchanged. MC-return `-Inf` sentinels are
not priority errors. Old PQE checkpoints likewise support initialization, not
exact resume across this numerical revision.

Research vector-adversarial and PQE trajectories can change and must be grouped
by their recorded revisions, not silently merged with old runs. This does not
invalidate all historical clean/random results for other methods. No historical
result manifests, caches, or checkpoints are migrated in place. Regression
tests check local reproducibility and actual backward updates, not long-run
MuJoCo benchmark scores or paper reproduction.

Primary evaluation is clean deterministic deployment return for all five
methods. RIQL uses its Gaussian mean; WSRL and Cal-QL use the tanh of their
pre-tanh mean; PQE uses the tanh of the five-member pre-tanh mean average; RPEX
deterministically chooses between nominal offline/online actions with its
existing Q+IPW logits. Method-faithful RPEX epsilon switching remains an
explicit diagnostic and is never selected as the research primary or as a
best-of-two score. Use `--evaluation-seed-role tuning` for tuning-only runs;
only the default `final` role is benchmark/main-table eligible.

Native WSRL uses 10 critics, target subsampling of 2 with replacement,
LayerNorm, an online-only replay after warmup, and one 1024-sample update split
into four critic minibatches plus one full-batch actor/temperature update. PQE
uses five independent two-hidden-layer actors and twin critics, pre-tanh moment
matching, and balanced replay. The corrected PQE actor has member log-std bounds
`[-20, 2]`; old three-hidden-affine PQE checkpoints are rejected and require
retraining or an explicit converter.

RPEX/RIQL source-aligned extraction is AWR in offline and online phases. The
paper-motivated observation comparison is isolated with
`--online-policy-extraction align_iql`; it is accepted only for online RPEX
observation corruption, remains AWR offline, and is serialized as a distinct
implementation variant. Explicit RIQL table, learning-rate, and UTD overrides
are preserved. A matched clean control should pass the corrupted run's resolved
RIQL values explicitly so the clean extension row cannot alter the match.

Run the five methods for one condition with:

```bash
python run_all_algorithms.py \
  --env-name hopper-medium-replay-v2 \
  --corruption random \
  --corruption-target mixed \
  --suite-profile research_benchmark \
  --seeds 0 1 2 \
  --stage both \
  --keep-going
```

When this suite is selected and `--algorithms` is omitted, the launcher chooses
the five baselines above. Reporting first averages the last three evaluations
within each training seed and then reports the population mean/std across
seeds. It does not treat evaluation episodes as training seeds or silently
select the best member/checkpoint.

## Run one experiment

Offline pretraining and online fine-tuning are each `500,000` steps by default.
They are independently configurable. `--stage both` runs them sequentially in
one process and carries the trained model directly across the boundary:

```bash
python run_experiment.py \
  --algorithm rpex \
  --env-name hopper-medium-replay-v2 \
  --corruption clean \
  --stage both \
  --offline-steps 500000 \
  --online-steps 500000 \
  --seed 42
```

Offline pretraining only:

```bash
python run_experiment.py \
  --algorithm rpex \
  --env-name hopper-medium-replay-v2 \
  --corruption clean \
  --stage offline \
  --offline-steps 500000 \
  --seed 42
```

Random mixed corruption:

```bash
python run_experiment.py \
  --algorithm riql_naive \
  --env-name halfcheetah-medium-replay-v2 \
  --corruption random \
  --corruption-target mixed \
  --mixed-ratios 0.1 0.2 0.3 0.4 \
  --stage both \
  --offline-steps 500000 \
  --online-steps 500000 \
  --seed 0
```

HalfCheetah adversarial observation corruption for Cal-QL:

```bash
python run_experiment.py \
  --algorithm cal_ql \
  --env-name halfcheetah-medium-replay-v2 \
  --corruption adversarial \
  --corruption-target observations \
  --stage both \
  --offline-steps 500000 \
  --online-steps 500000 \
  --seed 0
```

The equivalent Pessimistic Q-Ensemble command is:

```bash
python run_experiment.py \
  --algorithm pessimistic_q_ensemble \
  --env-name halfcheetah-medium-replay-v2 \
  --corruption adversarial \
  --corruption-target observations \
  --stage both \
  --offline-steps 500000 \
  --online-steps 500000 \
  --seed 0
```

### Checkpoints and continuation

Periodic checkpoint frequency and retention can be controlled independently by
phase:

```bash
python run_experiment.py \
  --algorithm rpex \
  --env-name hopper-medium-replay-v2 \
  --corruption clean \
  --stage both \
  --offline-steps 500000 \
  --online-steps 500000 \
  --offline-checkpoint-period 50000 \
  --online-checkpoint-period 100000 \
  --keep-last-checkpoints 3 \
  --seed 42
```

Use `--checkpoint-period N` to set one interval for both phases. Its default is
`100000`, with the newest five periodic checkpoints retained per phase. A value
of `0` disables periodic checkpoints; phase-final checkpoints are still
written. Runs are separated by algorithm, environment, corruption, target, and
seed, and each run has separate `checkpoints/offline/` and
`checkpoints/online/` directories.

To start a new online-only run from an offline model checkpoint:

```bash
python run_experiment.py \
  --algorithm rpex \
  --env-name hopper-medium-replay-v2 \
  --corruption clean \
  --stage online \
  --initialize-from-checkpoint /path/to/checkpoints/offline/final.pt \
  --online-steps 500000 \
  --seed 42
```

To continue an interrupted run with its optimizer, replay, RNG, and progress
state, pass the run directory to `--resume-run` instead. Initialization and
resume have intentionally different semantics:

```bash
python run_experiment.py \
  --algorithm rpex \
  --env-name hopper-medium-replay-v2 \
  --corruption clean \
  --stage both \
  --resume-run /path/to/existing/run_directory
```

## Run all nine algorithms for one condition

`run_all_algorithms.py` runs all nine algorithms by default, prints the elapsed
time after every algorithm, and prints a per-algorithm and whole-suite summary
at the end:

```bash
python run_all_algorithms.py \
  --env-name hopper-medium-replay-v2 \
  --corruption clean \
  --seeds 42 \
  --stage both \
  --offline-steps 500000 \
  --online-steps 500000 \
  --keep-going
```

For a fixed random mixed condition:

```bash
python run_all_algorithms.py \
  --env-name walker2d-medium-replay-v2 \
  --corruption random \
  --corruption-target mixed \
  --mixed-ratios 0.1 0.2 0.3 0.4 \
  --seeds 0 1 2 \
  --offline-steps 500000 \
  --online-steps 500000 \
  --keep-going
```

If `--algorithms` is omitted, the launcher runs all nine methods in the table
above. Pass `--algorithms ...` only when you want a subset or a custom order.

## Run an experiment matrix

`run_matrix.py` expands environments, corruption modes, targets, severities,
algorithms, and seeds into individual commands. Comma-separated values are
used for environments, corruption modes, targets, seeds, and ranges:

```bash
python run_matrix.py \
  --algorithms \
    rpex riql_pex riql_naive uwmsg pex cal_ql wsrl ro2o pessimistic_q_ensemble \
  --envs halfcheetah-medium-replay-v2,hopper-medium-replay-v2,walker2d-medium-replay-v2 \
  --corruptions clean,random,adversarial \
  --targets observations,actions,rewards,dynamics,mixed \
  --corruption-ranges 1.0 \
  --seeds 0,1,2 \
  --stage both \
  --mixed-ratios 0.1 0.2 0.3 0.4 \
  --offline-steps 500000 \
  --online-steps 500000 \
  --keep-going
```

Add `--dry-run` first to inspect the expanded commands without training.

## Logs, timing, tables, and plots

Standalone runs write below:

```text
results/comparisons/<environment>/<corruption>/<target>/<comparison-id>/
```

Each completed run contains at least:

- `metrics.csv`: evaluation scores used by plotting and comparison notebooks.
- `train_metrics.jsonl`: optimizer and training metrics over time.
- `performance.png`: the automatic single-run performance plot.
- `config.json` and `summary.json`: resolved configuration and completion data.
- `checkpoints/offline/` and/or `checkpoints/online/`: checkpoints for each
  phase that the run actually executes.

An all-algorithms comparison additionally writes:

- `timing.csv`: start, end, elapsed time, status, and command for every
  algorithm/seed run.
- `final_scores.csv`: final-window performance summary.
- `comparison_offline_online.csv` and `.png`: the continuous offline-to-online
  curve.
- `comparison_offline.csv` and `.png`: offline detail.
- `comparison_online.csv` and `.png`: online detail.
- `manifest.json`: suite configuration, artifacts, per-run timing, and total
  timing.

Console timestamps use `YYYY-MM-DD HH:MM:SS`. The all-algorithms launcher prints
`RUN_FINISHED` for each seed, `ALGORITHM_FINISHED` after each algorithm, and a
final `ALGORITHM_TIMING_SUMMARY`, `START_TIME`, `END_TIME`, and `ELAPSED` block.
Standalone runs also print their start time, end time, and elapsed duration.

For interactive comparison, open `comparison.ipynb`, select the
`corruption-robust-o2o` kernel, and set the environment, corruption, and target
in its first cell. The notebook presents the table and the combined,
offline-only, and online-only plots.

You can also regenerate a standalone aggregate plot:

```bash
python plot_results.py \
  --results-dir results/comparisons/hopper-medium-replay-v2/clean/none/<comparison-id>/runs \
  --env-name hopper-medium-replay-v2 \
  --corruption clean \
  --target none \
  --phase offline_online \
  --output results/hopper_clean_offline_online.png
```

## CPU Slurm jobs

The batch files under `slurm/` use the same Python 3.13/Gymnasium/native-MuJoCo
runtime. They do not install MuJoCo 2.1 or build `mujoco_py`.

Submit from the repository root on a cluster whose CPU partition is named
`cpu`:

```bash
# Create/update the shared micromamba environment, download the Hopper dataset,
# and run the test suite.
sbatch --wait slurm/setup_cpu.sbatch

# Run a short end-to-end CPU smoke experiment.
sbatch --wait slurm/smoke_cpu.sbatch

# Start the default 500k + 500k RPEX CPU experiment.
sbatch slurm/run_cpu.sbatch
```

Normal experiment options may follow the batch script:

```bash
sbatch --time=24:00:00 slurm/run_cpu.sbatch \
  --algorithm uwmsg \
  --env-name hopper-medium-replay-v2 \
  --corruption random \
  --corruption-target rewards \
  --stage both \
  --offline-steps 500000 \
  --online-steps 500000 \
  --seed 0
```

The defaults are stored under `~/.local/share/micromamba` and
`~/.d4rl/datasets`. Override them with `CRO2O_MAMBA_ROOT_PREFIX`,
`CRO2O_ENV_PREFIX`, or `CRO2O_DATASET_DIR`. Batch logs are written to
`slurm-*.out` in the submission directory.

## Short integration smoke

After downloading the Hopper dataset, this command exercises dataset loading,
offline updates, online interaction, replay updates, evaluation, logging, and
both phase-final checkpoints without launching a full experiment:

```bash
python run_experiment.py \
  --algorithm rpex \
  --env-name hopper-medium-replay-v2 \
  --corruption clean \
  --stage both \
  --device cpu \
  --offline-steps 2 \
  --online-steps 12 \
  --initial-collection-steps 4 \
  --warmup-steps 4 \
  --batch-size 8 \
  --replay-size 64 \
  --eval-period 1000 \
  --eval-episodes 1 \
  --checkpoint-period 0 \
  --train-log-period 1 \
  --hidden-dim 32 \
  --hidden-layers 1 \
  --seed 0
```

Run the automated checks with:

```bash
python -m pytest -q
python -m compileall -q robust_o2o *.py scripts
```

## Research interpretation

The modern stack makes macOS and current Linux installation practical and
provides one consistent backend for all algorithms. It is suitable for a
controlled comparison performed entirely with this version of the repository.
It should not be described as a bitwise reproduction of results generated by
the historical Gym/`mujoco_py` runtime. Record the pinned package versions,
dataset hashes, configuration files, seeds, and commit together with reported
results.
