# Corruption-Robust Offline-to-Online RL Benchmark

This repository provides a custom-budget benchmark for corruption-robust
offline-to-online RL under `clean`, `random`, and `adversarial` conditions. The
default `research_benchmark` suite contains exactly three main baselines:
`rpex`, `riql_naive`, and `wsrl`. It uses one common clean evaluation rule and
one RPEX-inspired replay-transition-poisoning contract across those baselines.

Cal-QL locomotion is available only as the explicit optional task adaptation
`cal_ql_locomotion_adaptation`. The local PQE code is available only as
`pqe_shared_actor_approx`. This implementation is a shared-actor approximation
and is not the official independent-policy Pessimistic Q-Ensemble
implementation. Neither optional method is included in the main result table.
The retired name `pessimistic_q_ensemble` fails instead of silently selecting
the approximation.

The default protocol is explicitly `rpex_d4rl_v2_legacy`: Gym 0.23.1, the full
D4RL-v2 environment ID, `mujoco_py`, and D4RL commit
`d842aa194b416e564e54b0730d9f934e3e32f854`. See
[RPEX_D4RL_V2_PROTOCOL.md](docs/RPEX_D4RL_V2_PROTOCOL.md) for the pinned protocol,
installation, metadata, and platform limitations.

Long training runs were not executed while preparing this repository. Use the
commands below in the pinned legacy environment, which requires a Linux x86_64
host rather than an Apple Silicon MacBook. Passing that environment check does
not establish numerical parity of a learner implementation.

**Strict reproduction decision: FINAL BENCHMARK NOT READY.** The strict-final algorithm set
is empty. RPEX and RIQL-naive are handwritten `source_aligned_port`s with only
partial fixed-batch evidence, so neither is strict-eligible. WSRL is an
unverified cross-framework port and its source-primary reporting rule is also
unverified; locomotion Cal-QL is an unsupported-task port; local PQE is an
approximation. The existing v1 corruption fixtures are diagnostic evidence:
they were produced under a runtime different from the strict pins, and the
adversarial fixture covers only an optimizer core. Upstream-executed learner,
constructor/evaluation RNG, condition, save/resume, and strict Linux receipts
are missing. A local macOS run is diagnostic-only. No final-run command is
authorized or documented while these fail-closed blockers remain.

That strict decision does not block `run_purpose=research_benchmark`. The
research path permits explicit budgets, seeds, and evaluation settings; records
the implementation type and benchmark role; and does not require parity
receipts, exact-RNG fixtures, or certificates. It is a custom research
benchmark, not an official paper reproduction.

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

# Common-budget RPEX clean diagnostic (500k offline + 500k online).
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
nodes are Spot VMs. Initialization and interruption recovery are deliberately
separate: `--initialize-from-checkpoint` starts a new run, while `--resume-run`
restores replay/RNG/optimizer state from an episode-boundary resume checkpoint.

For local Apple Silicon execution, the separate
`local_gymnasium_v4_diagnostic` protocol
uses Gymnasium v4 with native MuJoCo and reads the cached D4RL-v2 HDF5 dataset
directly. It is diagnostic-only, is not a D4RL-v2 benchmark result, and requires
the explicit `--allow-diagnostic-protocol` acknowledgement. The old
`local_gymnasium_v4` spelling is accepted only as an alias and is recorded under
the canonical diagnostic name.

The generic `reference` profile has been removed. Every run records an
`implementation_profile`, an `implementation_fidelity`, a `suite_profile`, an
`implementation_type`, and a `benchmark_role`. The default research suite is
the 3×5 `research_benchmark`; it is always labelled as a custom benchmark and
never as paper reproduction. The separate `primary_research_benchmark` and
`final_benchmark` paths retain the conservative strict registry and certificate
gate. They currently select no eligible algorithm and fail before creating a
run directory. The old `common_budget_robustness`, `common_budget_diagnostic`,
and `method_fidelity` names remain compatibility profiles for existing
diagnostic commands and results. See
[reproduction_matrix.md](docs/reproduction_matrix.md) for the strict/research
distinction and
[baseline_fidelity_manifest.yaml](docs/baseline_fidelity_manifest.yaml) for
pinned source provenance.

## 1. Algorithms

The CLI names and reference papers or configurations are listed below.

| # | `--algorithm` | Paper/configuration | Offline → online behavior |
|---|---|---|---|
| 1 | `rpex` | *RPEX: Robust Policy Expansion for Offline-to-Online RL under Diverse Data Corruption* | RIQL pretraining followed by IPW-based robust policy expansion |
| 2 | `riql_pex` | RIQL+PEX ablation from RPEX | RIQL pretraining followed by PEX without IPW |
| 3 | `riql_naive` | *Towards Robust Offline Reinforcement Learning under Diverse Data Corruption* | Applies the RIQL objective directly to online replay |
| 4 | `uwmsg` | *Corruption-Robust Offline Reinforcement Learning with General Function Approximation* | Applies the UWMSG objective directly to online replay |
| 5 | `pex` | *Policy Expansion for Bridging Offline-to-Online Reinforcement Learning* | IQL pretraining followed by PEX |
| 6 | `cal_ql_locomotion_adaptation` | *Cal-QL: Calibrated Offline RL Pre-Training for Efficient Online Fine-Tuning* | Optional locomotion task adaptation; not an official locomotion recipe |
| 7 | `wsrl` | *Efficient Online Reinforcement Learning Fine-Tuning Need Not Retain Offline Data* | CQL pretraining, frozen-policy warmup, and online-only SAC |
| 8 | `ro2o` | *Towards Robust Offline-to-Online Reinforcement Learning via Uncertainty and Smoothness* | Q-ensemble/smoothness pretraining followed by replay-based online reduction |
| 9 | `pqe_shared_actor_approx` | *Offline-to-Online Reinforcement Learning via Balanced Replay and Pessimistic Q-Ensemble* | Optional shared-actor approximation; not the official independent-policy method |

For `research_benchmark`, RPEX and RIQL-naive are recorded as
`source_aligned_port`, WSRL as `framework_port`, Cal-QL locomotion as
`task_adaptation`, and the shared-actor PQE as `approximation`. Their benchmark
roles are respectively `main`, `main`, `main`, `optional_adapted`, and
`optional_diagnostic`. None is labelled `exact_upstream_port`.

As requested, the online stages of `riql_naive` and `uwmsg` store newly collected
transitions in a replay buffer and train on mini-batches using the same objective
as their offline updates. Following the default comparison protocol in the RPEX
code, the RIQL variants, UWMSG, WSRL, and RO2O use online replay only by default.
PEX and the Cal-QL adaptation mix offline and online data at a 50:50 ratio. The
shared-actor PQE approximation uses density-ratio priority over a combined
offline/online priority mass; `--offline-ratio` sets its initial target mass. Use
`--pqe-replay-mode uniform` for the explicit fixed-ratio ablation.

## 2. Reproducibility environment

The RPEX environment protocol is a pinned legacy stack. Use Linux x86_64 for
the strict environment path; `mujoco_py` and its transitive PyBullet dependency
are often not buildable on Apple Silicon. The code intentionally does not fall
back to Gymnasium v4/v5 on macOS. Environment fidelity and learner numerical
parity are recorded and gated separately.

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
`mujoco-py==2.1.2.14`, linked native MuJoCo version code `210`, the complete
environment registration, dataset loading, and score normalization.
`--dataset-dir /path/to/datasets` changes the D4RL cache through
`d4rl.set_dataset_path()`.

On a Linux x86_64 strict host, the bounded end-to-end preflight loads and hashes
all three medium-replay-v2 datasets, checks the official qlearning conversion,
then performs 10 offline updates, 20 online steps, evaluation, checkpoint
reload, deterministic interrupted/resumed comparison, and corruption-cache
miss/hit artifact comparison:

```bash
python scripts/preflight_strict.py
```

It fails on unsupported platforms or version mismatches. It never substitutes
the local Gymnasium diagnostic and reports that substitution as strict success.
The executable save/resume equivalence exercised by this preflight is narrowly
scoped to RIQL-naive with random observation corruption under a diagnostic
common-budget smoke configuration. It resumes an offline checkpoint and does
not exercise online-checkpoint restore or compare every saved replay/RNG state.
It is not a save/resume certificate. Strict eligibility requires validated
external receipts for the complete eligible-baseline × certified-condition
matrix, bound to the exact repository, runtime, command, and artifacts.

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

The current **diagnostic/default CLI** schedule is:

- Offline: `500,000` gradient updates
- Online: `500,000` environment steps
- Online updates: one gradient update per environment step by default; reference
  WSRL uses four critic updates and one actor/temperature update per step

Use `--offline-steps`, `--online-steps`, and `--updates-per-step` to change these
values. A single command with `--stage both` runs offline pretraining and online
fine-tuning sequentially with the same agent:

These `500,000/500,000` defaults are for common-budget diagnostics. The dormant
RPEX/RIQL strict contract records `2,000,001` offline updates, `1,000,001`
requested online steps, and exact seeds `0,1,2,3,4`, but no current learner is
eligible to launch that contract.

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

The command above is diagnostic-only because the target is `dynamics`. The v1
adversarial fixture covers only the **Hopper observation-target optimizer
core** and was generated under a runtime different from the strict pins. It is
not an end-to-end condition certificate and authorizes no strict adversarial
row. HalfCheetah/Walker2d observations and all adversarial `actions`, `rewards`,
and `dynamics` are likewise diagnostic-only. The strict adversarial condition
set is empty.

The following corruption targets are supported:

- `observations`: current observations stored in replay
- `actions`: actions stored in replay
- `rewards`: rewards stored in replay
- `dynamics`: next observations stored in replay
- `mixed`: allocate corrupted transitions across all four targets using
  `--mixed-ratios`

Corruption timing is explicit. `official_code_reference` uses
`post_transition_replay_poisoning`: the clean observation selects the action,
the clean action is executed, and the selected field is changed only before the
transition enters replay. `paper_reference` uses the separate
`paper_pre_action_sensor_actuator` path, where observation corruption precedes
action selection and action corruption precedes `env.step()`. The two profiles
have different manifest identities and cannot be aggregated.

The default corruption parameters match RPEX:

- Offline corruption rate: `0.3`
- Online corruption rate: `0.5`
- Corruption range: `1.0`
- Random reward corruption: replace selected rewards with
  `Uniform(-30 × range, 30 × range)`
- Offline adversarial reward corruption: official `-range × reward` sign flip
- Online adversarial reward corruption: official `Uniform(-1, 1)` replacement

Official RPEX online observation/dynamics poisoning uses unit scale; actions
use the offline action standard deviation. The historical dataset-standard-
deviation behavior is available only as
`--online-corruption-scale-profile dataset_std_scaled_extension` and has a
separate manifest/plot identity.

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
the dataset and normalizer, attack checkpoint, target/rate/range, attack seed,
offline and online steps/step sizes, device, timing, objective, optimizer,
source commit, MC settings, and implementation version. Writes use a per-key
file lock, temporary file, fsync, SHA-256 sidecar, and atomic replace; loads
validate the checksum, embedded metadata, shape, dtype, and indices.
`--attack-min-step-size` is explicit and defaults to zero; no hidden runtime
lower bound is applied. Add `--force-regenerate-attack` to regenerate a cache.

### Correctness-sensitive modes

- RPEX/RIQL non-legacy profiles resolve to
  `official_unsquashed_gaussian`: bounded mean, state-independent log standard
  deviation, ordinary Normal samples, and no tanh-Jacobian correction.
  `tanh_gaussian` remains available for explicit non-reference ablations.
  `--action-execution-profile` separately records
  `official_algorithm_behavior` or `clip_to_action_space`.
- `--evaluation-mode deterministic_diagnostic` uses policy means and a
  deterministic expansion branch. `method_faithful` preserves stochastic RPEX
  expansion, and `both` logs both. Evaluation saves/restores Python, NumPy,
  PyTorch CPU, and CUDA RNG state.
- `--mc-return-source post_corruption` is the Cal-QL default. Reward corruption
  recomputes return-to-go within trajectory boundaries. Reference benchmark
  mode uses `--calibration-mask-mode all`, so it does not reveal corrupted row
  indices to the learner. `oracle_exclude_corrupted` is explicitly labeled as
  an oracle ablation; `disabled` turns calibration off.
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

Then run the online stage using the manifest-tagged final checkpoint from the
resulting run directory:

```bash
python run_experiment.py \
  --algorithm riql_pex \
  --env-name halfcheetah-medium-replay-v2 \
  --corruption random \
  --corruption-target rewards \
  --stage online \
  --initialize-from-checkpoint /absolute/path/to/final_manifest_<sha>.pt \
  --seed 0
```

The final offline checkpoint is stored as
`checkpoints/offline/final_manifest_<sha>.pt` inside the run directory. The program fails
immediately if the checkpoint algorithm, environment, or
observation/action dimensions do not match the current command. The state
normalization mode, location, and scale are also restored from the checkpoint.
New runs write both `config.json` and `resolved_config.json`.

Interrupted-run continuation is different. Reissue the original semantic
arguments and point `--resume-run` at its run directory:

```bash
python run_experiment.py \
  --algorithm riql_naive \
  --env-name hopper-medium-replay-v2 \
  --corruption random \
  --corruption-target rewards \
  --stage both \
  --suite-profile common_budget_robustness \
  --resume-run /absolute/path/to/run_dir
```

Only an episode-boundary checkpoint with complete replay, optimizer, scheduler,
RNG, environment, counter, and writer state is eligible. The requested
canonical manifest must equal the original manifest; otherwise the command
fails and instructs the user to initialize a new run instead.

Current executable equivalence coverage is RIQL-naive plus random observation
corruption in the diagnostic smoke path only. No valid full-matrix
save/resume-equivalence receipt exists. This section documents the mechanism;
it does not certify a benchmark or authorize publication eligibility.

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
results/comparisons/<protocol>/<suite+profile>/<env>/<corruption>/<target>/<comparison_id>/runs/
└── <algorithm>/<suite>/<implementation>/<fidelity>/<budget>/.../
    └── manifest_<sha>/<run_id>/
        └── checkpoints/
    ├── offline/
    │   ├── step_000100000_manifest_<sha>.pt
    │   └── final_manifest_<sha>.pt
    └── online/
        ├── step_000100000_manifest_<sha>.pt
        └── final_manifest_<sha>.pt
```

The shared periodic interval defaults to `100,000`. Five periodic checkpoints
per phase are retained by default; the manifest-tagged final checkpoint is
always retained.

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
results/comparisons/<protocol>/<profile>/<env>/<corruption>/<target>/<comparison_id>/
├── runs/
│   └── <algorithm>/...
├── comparison_offline_online.png
├── comparison_offline.png
├── comparison_online.png
├── comparison_{offline_online,offline,online}.csv
├── final_scores.csv
├── per_seed_final_scores.csv
├── paper_reproduction_summary.csv
├── common_per_seed_final_scores.csv
├── common_benchmark_summary.csv
├── timing.csv
└── manifest.json
```

- The three `comparison_*.png` files show the combined, offline-only, and
  online-only curves and include completed runs only. Separate
  `diagnostic_running_{offline_online,offline,online}.png` files are refreshed
  after every evaluation and may include the currently running algorithm.
- The matching CSV files contain mean/std/count at every evaluation step.
- `final_scores.csv`: backward-compatible common last-three metric; it is not a
  paper metric
- `per_seed_final_scores.csv`: publication-eligible source-primary per-seed
  outputs with each row's reproduction and condition status preserved
- `paper_reproduction_summary.csv`: only verified rows with
  `paper_reproduction_eligible=true`; current source-aligned ports are excluded,
  so an empty file is an intentional fail-closed result
- `common_per_seed_final_scores.csv` and `common_benchmark_summary.csv`: rows
  passing the separate `common_benchmark_eligible` contract under one declared
  cross-algorithm rule; never an official-paper score
- `timing.csv`: start/end time and elapsed time for every algorithm/seed run
- `manifest.json`: commands, return codes, per-algorithm timing summaries,
  overall timing, and artifact paths

After each seed run, the command prints a `RUN_FINISHED` line. After all seeds
for one algorithm finish, it prints an `ALGORITHM_FINISHED` line containing the
algorithm's accumulated runtime. At the end, `ALGORITHM_TIMING_SUMMARY` lists
every completed algorithm followed by the overall start, end, and elapsed time.

Use `--comparison-name NAME` to set the final directory name and `--keep-going`
to continue with the remaining algorithms if one run fails.

### Research benchmark suite

`run_55_experiment.py` keeps its historical filename, but its default research
matrix is now 3 algorithms × 5 conditions: the three main baselines, clean, and
the four individual random-corruption targets. Hopper remains the default. Run
the readiness check first, then launch explicit seeds and budgets:

```bash
conda activate corruption
python scripts/check_research_readiness.py \
  --env-name hopper-medium-replay-v2 \
  --corruption-suite random \
  --protocol local_gymnasium_v4_diagnostic \
  --allow-diagnostic-protocol

python run_55_experiment.py \
  --run-purpose research_benchmark \
  --env-name hopper-medium-replay-v2 \
  --algorithms rpex,riql_naive,wsrl \
  --corruption-suite random \
  --seeds 0,1,2,3,4 \
  --offline-steps 500000 \
  --online-steps 500000 \
  --protocol local_gymnasium_v4_diagnostic \
  --allow-diagnostic-protocol
```

Use `--env-name halfcheetah-medium-replay-v2` for HalfCheetah. The local
Gymnasium-v4 backend remains explicitly identified in every manifest and its
scores must not be described as an official D4RL-v2 paper reproduction. Use
`--dry-run` to inspect commands and `--keep-going` to record later failures
instead of stopping the controller at the first one.

Optional methods require a separate opt-in, for example
`--optional-baselines cal_ql_locomotion_adaptation`. Their rows are written to
role-specific summaries rather than `research_summary.csv`.

Select suites explicitly with `--corruption-suite clean`, `random`,
`adversarial`, or `all`. In the custom research benchmark, `random` means clean
plus the four random targets. In the separate strict contract, it means only the four
actual random targets and excludes clean. Clean is a benchmark transfer and is
never auto-certified. A diagnostic adversarial suite may contain all four
single targets, but the strict adversarial setting is empty: the existing
Hopper observation fixture is optimizer-core-only and authorizes no end-to-end
condition. Applying an RPEX corruption condition to another baseline is
recorded as `benchmark_transfer`.

Before any strict run, on a supported Linux x86_64 host, execute:

```bash
python scripts/audit_reproducibility.py
```

Externally generated receipts default to the repository-private Git path
returned by `git rev-parse --git-path robust_o2o-certificates` (normally
`.git/robust_o2o-certificates`) so committing a receipt cannot change the commit
it attests. `ROBUST_O2O_CERTIFICATE_DIR` may point to an external immutable
artifact directory. The directory requires an indexed SHA-256 for each receipt;
the validator does not mint receipts or infer a missing pass.

The audit reports `RPEX/RIQL ELIGIBLE SUBSET STATUS`, `FIVE-BASELINE STATUS`,
`RANDOM CORRUPTION STATUS`, `ADVERSARIAL CORRUPTION STATUS`, `SAVE/RESUME
STATUS`, `STRICT ENVIRONMENT STATUS`, and `FINAL BENCHMARK STATUS` separately.
All are currently **NOT READY** except adversarial corruption, which is
explicitly **EXCLUDED** because the strict adversarial set is empty. RPEX/RIQL
lack upstream-executed end-to-end learner parity, so the strict algorithm set
is empty. The random v1 fixture has a strict-runtime mismatch; the adversarial
v1 fixture is optimizer-core-only; full online-constructor/evaluation RNG and
save/resume receipts are absent; and the current Mac is not the pinned Linux
runtime. WSRL also lacks numerical and reporting parity, Cal-QL locomotion is a
task port, and PQE is an approximation. The strict runner must reject before
output creation. No final benchmark command is published until the audit can
validate all required clean-tree, source-bound executable receipts.

For a local Mac smoke/debug run only:

```bash
conda activate corruption
python run_55_experiment.py \
  --env-name halfcheetah-medium-replay-v2 \
  --corruption-suite random \
  --suite-profile common_budget_diagnostic \
  --run-purpose diagnostic \
  --protocol local_gymnasium_v4_diagnostic \
  --allow-diagnostic-protocol
```

These local scores are stored and plotted as diagnostic D4RL-reference-scaled
returns, never as benchmark D4RL normalized returns.

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

This matrix command is diagnostic. No adversarial condition has an end-to-end
strict certificate. The v1 Hopper observation fixture is an optimizer-core
check with a strict-runtime mismatch and cannot authorize a final or paper
reproduction suite.

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
└── comparisons/<protocol>/<profile>/<env>/<corruption>/<target>/<comparison_id>/
    ├── comparison_offline_online.png
    ├── comparison_offline.png
    ├── comparison_online.png
    └── runs/<algorithm>/<suite>/<implementation>/<fidelity>/<budget>/.../
        └── manifest_<sha>/<timestamp>_<id>/
    ├── config.json
    ├── result.log
    ├── metrics.csv
    ├── train_metrics.jsonl
    ├── performance.png
    ├── summary.json
    └── checkpoints/
        ├── offline/
        │   └── final_manifest_<sha>.pt
        └── online/
            └── final_manifest_<sha>.pt
```

- `metrics.csv`: evaluation step, raw return, D4RL normalized return, standard
  deviation, and cumulative elapsed time
- `train_metrics.jsonl`: losses, Q statistics, uncertainty,
  policy-expansion selection ratio, and other training metrics
- `performance.png`: normalized-return curve for one run
- `summary.json`: start/end timestamps, total elapsed time, final performance,
  and success/failure status
- `config.json`: complete arguments, selected protocol and D4RL environment ID,
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

- Use `common_budget_diagnostic` when offline/online budgets must be identical.
  `primary_research_benchmark` uses the strict registry, which currently
  contains no eligible algorithm. RPEX/RIQL-naive are source-aligned but lack
  end-to-end learner certificates; WSRL lacks learner/reporting parity; Cal-QL
  locomotion is a task port; and local PQE is an approximation. The primary and
  final launch paths therefore fail closed. Never combine their diagnostic
  curves with future certified results.
- `paper_reproduction` is a reserved run purpose and currently fails closed:
  no baseline has a certified paper-specific task, seed, budget, environment,
  learner, corruption, and reporting contract. Use `diagnostic` for exploratory
  runs. No final contract is currently launchable.
- On supported hardware, RO2O with `--ro2o-sample-size 20` and the 10-critic
  UWMSG/RO2O configurations are particularly slow. Reduce the sample size or
  critic count only during exploratory runs, and restore the original values
  for final comparisons.
- MPS and CPU results may not be exactly identical because of floating-point
  implementation differences. Use the same device for all results in a
  comparison table.
- Never aggregate different environment protocols, implementation profiles, or
  suite profiles. Every run has a canonical manifest SHA in its path and
  checkpoints; plotting rejects non-seed manifest differences and duplicates.
- A single-seed curve has no seed-uncertainty band. Episode-return dispersion is
  not substituted for across-seed uncertainty.
- Adversarial offline-attack generation time is included in the total `ELAPSED`
  time. Check `offline_corruption.loaded_from_cache` in `config.json` to
  determine whether the attack cache was used.
- Before starting long experiments, run `scripts/smoke_rpex_d4rl_v2.py` to
  verify Gym/mujoco_py, the complete D4RL ID, dataset, normalization, and saved
  provenance.

## 10. Implementation sources

The provided and pinned upstream codebases are source references for the
unified objectives and default values. A reference URL does not mean that the
local learner is a thin wrapper or a numerically verified port.

See `docs/reproduction_matrix.md` and
`docs/baseline_fidelity_manifest.yaml` before interpreting any result. RPEX and
RIQL-naive are source-aligned ports, WSRL is an unverified framework port,
locomotion Cal-QL is a task port, and the current Pessimistic Q-Ensemble is an
approximation.

- RPEX: <https://github.com/felix-thu/RPEX>
- Pinned D4RL environment registry, dataset conversion, and normalization:
  <https://github.com/rail-berkeley/d4rl/tree/d842aa194b416e564e54b0730d9f934e3e32f854>
- RIQL: provided `RIQL-main` directory
- UWMSG: provided `UWMSG-main` directory
- WSRL: <https://github.com/zhouzypaul/wsrl>
- Cal-QL: <https://github.com/nakamotoo/Cal-QL>
- RO2O: <https://github.com/BattleWen/RO2O>
- Balanced Replay + Pessimistic Q-Ensemble:
  <https://github.com/shlee94/Off2OnRL>
# corruption=robust-o2o
