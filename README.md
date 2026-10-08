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

`--corruption-profile riql_rpex_code` is the default shared corruption
contract for every algorithm. It is independent of the learner's
`--implementation-profile`: changing the learner does not change the attack
recipe. The contract is anchored to the pinned RIQL/RPEX source code and is
recorded as `riql_rpex_code_v1` with attack implementation
`corruption_v9_source_contract`. The selected Gymnasium v4 training environment
and D4RL-v2 dataset protocol are unchanged.

The corruption source anchors are [RIQL `cc07d81`](https://github.com/YangRui2015/RIQL/tree/cc07d81201e4cd44415560523fcd719f57bbf12c)
and [RPEX `35da71e`](https://github.com/felix-thu/RPEX/tree/35da71ee5151b6179d21b9a2b4ce1b6408aedd04).

`--corruption` selects the mechanism:

- `clean`: no corruption; the target is automatically `none`.
- `random`: uniform RIQL/RPEX data corruption.
- `adversarial`: EDAC-based RIQL/RPEX data corruption, or its reward rule.

Source corruption accepts one non-clean target: `observations`, `actions`,
`rewards`, or `dynamics`. The pinned public code does not define a mixed-target
contract. Partitioned mixed corruption and dataset-standard-deviation online
noise remain explicit historical extensions through
`--corruption-profile legacy_extension`; they are not reported as source
settings. For partitioned mixed corruption, ratios are ordered as observation,
action, reward, dynamics, must be non-negative, and must sum to one:

```bash
--corruption-profile legacy_extension \
--corruption-target mixed --mixed-ratios 0.1 0.2 0.3 0.4
```

The experimental online dataset-std scale likewise requires both options:

```bash
--corruption-profile legacy_extension \
--online-corruption-scale-profile dataset_std_scaled_extension
```

The main severity controls are `--offline-corruption-rate` (default `0.3`),
`--online-corruption-rate` (default `0.5`), and `--corruption-range` (default
`1.0`, denoted ε). Source mask/noise draws use a private NumPy `RandomState`
MT19937 stream seeded by `corruption_seed`, which defaults to the experiment
seed. The stream preserves the source draw mapping without consuming learner
NumPy RNG.

Offline vector corruption uses each attacked field's clean offline population
standard deviation: random corruption adds `Uniform[-ε, ε] × σ`. Random reward
corruption **replaces** selected offline rewards with `Uniform[-30ε, 30ε]`.
Online random observation/dynamics noise has unit scale in normalized state
coordinates; online action noise uses the clean offline action standard
deviation. Source online random reward replacement is `Uniform[-30, 30]`
independent of ε, preserving the public code's exception even when ε≠1.
Cal-QL MC returns use the post-corruption reward sequence and stop at episode
boundaries; evaluation returns remain clean.

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

The default source interaction contract is:

1. Keep the policy/expansion proposal unchanged for the method's internal Q,
   IPW, and log-probability calculations.
2. Require that proposal to be finite and pass it to `env.step` with
   `action_execution_profile=official_algorithm_behavior`, preserving the
   public code's action handling. The runner adds no action-space clipping;
   bounded policies and the selected simulator may enforce their own bounds.
3. Copy the resulting clean transition and apply corruption to the replay copy
   after the environment step. Choose the next real action from the
   environment's clean next observation.

Action poisoning changes the replay label; the poisoned action is retained
for critic training and is never sent back to the simulator. Source action
noise is added to the original finite policy proposal, including a proposal
outside the declared action-space range. Logs separately record proposal,
environment-call, and poisoned-replay action OOB rates. NaN/Inf actions fail
explicitly. The application-clipped action path remains available only under
an explicit `legacy_extension` corruption profile.

Offline corruption is generated in raw dataset coordinates before learner
normalization. Source state normalization is fitted to the concatenated
corrupted offline observations and next observations; its denominator is
`population_std + 1e-3`. Online source attacks operate in the same normalized
state coordinates as the wrapped upstream environment. Random state noise is
therefore `Uniform[-ε, ε]` in that space. The runner maps it back with the
actual normalization denominator before a single learner normalization pass.
With normalization disabled, online state noise uses raw unit scale.
Adversarial state inputs and perturbations likewise use source online
coordinates; online action scales remain the offline action standard deviation.
The source coordinate field is `normalized_source_state_units_v2` when active.
The optional legacy dataset-std extension retains its separately recorded
raw-coordinate recipe.

The earlier `fix main 3` correction remains identifiable as
`normalized_random_state_units_v1` for historical extension runs. The new
source profile additionally changes mask RNG, normalization, action handling,
and adversarial details. Source-version cache keys and exact-resume checks
prevent those histories from being silently reused or pooled with the new
contract. Existing manifests, caches, and checkpoints are preserved; compatible
weights can initialize a new run. IQL-family `--max-grad-norm` clips actor,
critic and value gradients when explicitly set; its default remains disabled.

Comparison plots and final-score tables reject different corruption recipes,
reward supports, datasets, or evaluation/budget conditions across algorithms.
Method-specific objectives and UTD ratios may still differ and remain logged.
`plot_results.select_latest_comparable_records` selects a shared condition from
the newest usable RPEX run before choosing each method's latest variant/seeds;
it returns excluded records so a local notebook can explain missing methods.
It never falls back to a different corruption just to fill a missing curve.
Repository provenance includes source-content and tracked-diff SHA256 values,
so different dirty edits with the same `git status` are distinguishable.

Source EDAC objectives consume the states supplied by the source attack
pipeline: raw offline states and normalized online states when state
normalization is active. The pinned EDAC payload has no preprocessing metadata;
identity oracle preprocessing is recorded explicitly, and incompatible
checkpoint preprocessing is rejected for the source profile. Legacy extensions
continue to use their declared raw-state preprocessing.

Source adversarial attacks preserve the public code's double-standard-deviation
initialization: the sampled optimization parameter is scaled by σ, then the
effective perturbation is multiplied by σ again. They recreate Adam at every
step and use 100 steps with base step size `0.01` offline, and 2 steps with base
step size `0.1` online; the optimizer rate is scaled by ε. Online perturbation
initialization uses a fresh default Torch generator for each source attack.
The stochastic EDAC actor used by dynamics objectives has resumable,
phase-private RNG state so corruption does not consume the learner's Torch
stream and offline cache regeneration cannot shift the online actor stream.
This schema is recorded as `rpex_code_init_phase_private_actor_v1`.

The port fixes two upstream offline dynamics execution errors (an actor tuple
used as an action and an undefined standard-deviation name) while preserving
the intended stochastic objective. These runtime fixes, private actor RNGs,
and the user-selected v4 environment mean the port does not claim identical
upstream trajectories or benchmark scores. Selected-transition rates and
actual-value-change rates are both stored; a zero selection rate leaves the
artifact unchanged. Reward replacement at ε=0 remains replacement.

Checkpoints record the corruption profile, semantic version, coordinate field,
and `attack_rng_schema`, with private actor states where required. Exact resume
across incompatible contracts is rejected; compatible weights can initialize
new runs. PQE also records `pqe_numerics_version` as
`centered_moments_strict_priorities_v1`: centered moment variance and a positive floor
before square root prevent cancellation/NaN gradients without detaching actors
or changing the pre-tanh Gaussian. Invalid priorities/weights now raise rather
than silently becoming uniform probabilities; valid floors, clipping, and the
initial online priority formula are unchanged. MC-return `-Inf` sentinels are
not priority errors. Old PQE checkpoints likewise support initialization, not
exact resume across this numerical revision.

Corrupted and PQE trajectories can change and must be grouped by their recorded
revisions, rather than silently merged with old runs. Historical results remain
usable under their recorded settings; the new source profile does not relabel
them. No historical result manifests, caches, or checkpoints are migrated in
place. Regression tests check local reproducibility and actual backward
updates, not long-run MuJoCo benchmark scores or paper reproduction.

Primary evaluation is clean deterministic deployment return for all five
methods. RIQL uses its Gaussian mean; WSRL and Cal-QL use the tanh of their
pre-tanh mean; PQE uses the tanh of the five-member pre-tanh mean average; RPEX
deterministically chooses between nominal offline/online actions with its
existing Q+IPW logits. Method-faithful RPEX epsilon switching remains an
explicit diagnostic and is never selected as the research primary or as a
best-of-two score. `evaluation_mode` controls how an already trained policy
chooses actions during clean evaluation; it does not control training or the
corruption target. `--evaluation-mode deterministic` uses the policy's mean or
its deterministic expansion selector; `method_faithful` invokes the method's
native evaluation behavior, including RPEX's stochastic epsilon switching;
`both` logs both measurements and retains deterministic return as primary.
The research benchmark requires `deterministic`. Use
`--evaluation-seed-role tuning` for tuning-only runs; only the default `final`
role is benchmark/main-table eligible.

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
  --corruption-target observations \
  --corruption-profile riql_rpex_code \
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
  --corruption-profile legacy_extension \
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
  --corruption-profile legacy_extension \
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
  --corruption-profile legacy_extension \
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
