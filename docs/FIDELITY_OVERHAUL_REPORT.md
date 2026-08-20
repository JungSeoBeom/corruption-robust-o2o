# Baseline fidelity overhaul report

Audit date: 2026-08-21

This report describes the static audit and bounded verification used for the
research-facing `run_55` suite. No long reinforcement-learning run was
performed. The machine used for the checks was macOS, so strict D4RL-v2 runtime
success is intentionally not claimed.

## A. Baseline fidelity matrix

| Baseline | Upstream commit | Current status | Previous material difference | Remaining difference | `method_fidelity` |
|---|---|---|---|---|---|
| RPEX / RIQL-naive | `felix-thu/RPEX@35da71ee5151b6179d21b9a2b4ce1b6408aedd04` | `official_code_reference` / `exact_upstream_port` for published random/adversarial rows | Squashed/state-dependent policy path, generic hyperparameters, no AlignIQL selector, no actor cosine scheduler | Public code has no clean or mixed RIQL row; those are unsupported in method fidelity | Yes for the four published targets |
| RPEX paper interpretation | Same source plus paper semantics | `paper_reference` / `paper_code_conflict` | Paper replay/evaluation/corruption timing was collapsed into one generic reference | Public source has no executable mixed-corruption recipe; paper mixed fails rather than guessing | Yes for supported non-mixed rows |
| WSRL locomotion | `zhouzypaul/wsrl@ad4dc1248a138bc15d6e053f2d1dba1b8cfbaca2` | `official_code_reference` / `exact_upstream_port` (PyTorch port) | Extra actor hidden affine, generic initialization/LRs/entropy backup, all-head target/CQL, actor updated four times | Framework is PyTorch rather than JAX/Flax; strict runtime performance was not run locally | Yes |
| Cal-QL locomotion | `nakamotoo/Cal-QL@ac6eafec22e8d60836573e1f488c7f626ce8a77e` | `locomotion_port` / `task_port` | BC warm-up and shared LR differed; task qualification was missing | Official public recipes are AntMaze/Adroit, not D4RL locomotion | No; use locomotion port |
| Pessimistic Q-Ensemble | `shlee94/Off2OnRL@6f298fa9ef040d725067d0f2775022bd2900d635` | `pqe_shared_actor_approx` / `approximation` | Shared actor/critic ensemble was named like official PQE | Exact five checkpoint-loaded policies and member-wise twin-Q port is not implemented; public recipes are D4RL v0 | No; fail-fast |

The exact provenance file is `docs/baseline_fidelity_manifest.yaml`.

## B. Paper/code conflicts

| RPEX behavior | `official_code_reference` | `paper_reference` |
|---|---|---|
| Online replay | Online-only replay | Offline/online mixture |
| Evaluation | Epsilon policy switching | Greedy highest-weight branch |
| Observation/action corruption timing | Post-transition replay poisoning | Pre-action sensor/actuator corruption |
| Result identity | `exact_upstream_port` | `paper_code_conflict` |

These values are part of the canonical manifest, result path, checkpoint, and
aggregation signature. They cannot become one averaged curve. The pinned public
RPEX code contains no mixed setting, so `rpex_paper_mixed` is non-executable
until a citable source supplies the missing definition.

## C. Changed files

- `docs/baseline_fidelity_manifest.yaml`: pinned repositories, commits,
  file/function provenance, supported tasks, conflicts, and divergences.
- `robust_o2o/fidelity.py`: implementation/fidelity/suite vocabularies and the
  literal `RIQL_TRAIN_CONFIG.py` table.
- `robust_o2o/config.py`: profile resolution, upstream budgets, semantic
  conflict switches, fail-fast rules, role seeds, and separate initialize/resume
  arguments. Old `reference` now raises a migration error; old results remain
  legacy/unknown.
- `robust_o2o/networks.py`: official unsquashed RPEX Gaussian, exact RPEX hidden
  stack/initialization, WSRL two-block LayerNorm architecture and final-hidden
  `1e-2` initialization.
- `robust_o2o/agents/iql_family.py`: RIQL quantile/clipped robust loss, AlignIQL,
  actor/critic learning rates, cosine scheduler, and checkpointed scheduler.
- `robust_o2o/agents/sac_family.py`: WSRL REDQ 10/2 with replacement, full-10
  actor minimum, subset CQL, official `-action_dim` target entropy with a
  softplus Geq multiplier, no entropy backup, and 4:1:1
  critic/actor/temperature schedule. Entropy zero is legacy-only.
- `robust_o2o/corruption.py`: explicit pre/post timing, official Adam versus
  opt-in sign-PGD, separate offline/online reward rules, official unit-scale
  online observation/dynamics attacks, dedicated RNG, identity-complete locked
  atomic cache, checksum and shape/dtype/metadata validation, and mask/value
  hashes.
- `robust_o2o/environment.py`: profile-controlled action execution and strict
  environment/action/horizon checks.
- `robust_o2o/replay.py`, `robust_o2o/experiment.py`, and
  `robust_o2o/logging_utils.py`: replay/RNG/environment/counter serialization,
  episode-boundary exact resume, manifest-tagged atomic checkpoints, append-only
  metrics with cumulative elapsed time.
- `robust_o2o/manifest.py`: canonical sorted manifest and SHA-256 plus a
  seed-excluding aggregation signature.
- `plot_results.py`: manifest-strict aggregation, completed-only benchmark
  summaries, explicit common-budget/approximation/oracle labels, and opt-in
  running-run diagnostics.
- `run_55_experiment.py`, `run_all_algorithms.py`, and `run_matrix.py`: suite and
  implementation profile propagation. The `run_55` default remains 500k/500k
  and is explicitly common-budget, not paper reproduction.
- `scripts/preflight_strict.py`: Linux x86_64 pinned D4RL-v2 bounded preflight.
- `tests/`: fixed-tensor, architecture, schedule, paired-corruption, atomic
  multiprocessing cache, manifest, and exact checkpoint/resume tests.

## D. Corruption verification

- Random corruption uses a dedicated NumPy generator from `corruption_seed`.
  The same dataset/config/seed produces identical selected-index and corrupted-
  value SHA-256 values across algorithms.
- `generic_partitioned_mixed` assigns every selected transition to exactly one
  of observation/action/reward/dynamics. It is not promoted to RPEX paper mixed.
- Adversarial initialization uses a dedicated Torch generator. Official Adam
  keeps the upstream optimizer reinitialization, offline `100 × 0.01`, online
  `2 × 0.1`, projection, standard-deviation scaling, and mean-Q objective.
  Experimental sign-PGD has a different manifest and cache identity.
- Cache identity includes source commit, dataset/normalizer/attacker hashes,
  attack implementation/objective/optimizer/device/timing, both attack
  schedules, corruption parameters, and attack seed.
- Cache creation is protected by `flock`; data and checksum are atomically
  replaced after fsync. A two-process test verifies one miss, one hit, and
  byte-identical results.
- Post-transition replay poisoning and pre-action sensor/actuator corruption
  are separate execution paths and aggregation groups.

## E. Environment verification

- Benchmark default: `rpex_d4rl_v2_legacy`; no fallback is permitted.
- Diagnostic opt-in: `local_gymnasium_v4_diagnostic` plus
  `--allow-diagnostic-protocol`; scores are labeled diagnostic reference-scaled
  returns.
- Static/unit checks ran in the local `corruption` Conda environment on macOS.
  A real diagnostic smoke also completed on Hopper-v4 with 2 offline updates,
  3 online steps, one evaluation episode, manifest relocation, plots, and both
  phase checkpoint saves.
- Strict Linux x86_64 preflight was not run on this Mac because the pinned
  Gym 0.23.1/D4RL/mujoco_py runtime is platform-specific. This is an explicit
  unverified runtime, not a local-protocol success.
- `scripts/preflight_strict.py` verifies pinned versions, exact ID, dataset,
  dimensions/SHA, reset/step, 10 offline updates, 20 online steps, and checkpoint
  save/load.

## F. Saving and resume verification

- `--initialize-from-checkpoint` loads model/target/optimizer/normalizer state
  into a new run and new replay/environment lifecycle.
- `--resume-run` only selects a checkpoint whose full resume payload says it was
  saved at an episode boundary. It validates algorithm, environment, protocol,
  dimensions, environment fingerprint, implementation profile, and canonical
  manifest equality.
- Checkpoints store model/target/optimizers/scheduler/temperature, offline and
  online replay arrays and RNGs, Python/NumPy/Torch CPU/CUDA/MPS RNGs, corruption
  and attack RNGs, environment RNG, current episode state/counters, metric
  accumulator, and writer position.
- The deterministic equivalence test compares 200 uninterrupted RIQL updates
  with 100 updates + checkpoint/replay/RNG restore + 100 updates. Model tensors,
  optimizer/scheduler state, replay contents/RNG, and counters are bitwise equal
  (zero tolerance on CPU).
- Resume logging appends without a second CSV header or duplicate row and keeps
  cumulative elapsed time monotonic.

## G. Tests

Commands used:

```bash
conda run -n corruption env PYTHONPATH=. \
  python -m unittest discover -s tests -q
/opt/anaconda3/bin/ruff check .
git diff --check
conda run -n corruption env PYTHONPATH=. \
  python -m py_compile robust_o2o/*.py robust_o2o/agents/*.py scripts/*.py *.py
```

`pytest` is not installed in the `corruption` environment, so the same unittest
suite was run through `unittest discover`. Ruff is installed in the base Conda
environment and was run from there. No mypy configuration exists. The strict
preflight is skipped for the platform reason in section E.

Final result: 115 tests total: 114 passed, 0 failed, 1 skipped. Ruff, `diff --check`, and
byte-code compilation passed. The local diagnostic smoke passed; the strict
Linux preflight was not executed.

## H. Recommended commands

RPEX official-code reference, five seeds, random observations:

```bash
python run_all_algorithms.py --algorithms rpex \
  --env-name hopper-medium-replay-v2 \
  --corruption random --corruption-target observations \
  --seeds 0,1,2,3,4 --suite-profile method_fidelity \
  --implementation-profile official_code_reference
```

RPEX paper-conflict profile, same condition:

```bash
python run_all_algorithms.py --algorithms rpex \
  --env-name hopper-medium-replay-v2 \
  --corruption random --corruption-target observations \
  --seeds 0,1,2,3,4 --suite-profile method_fidelity \
  --implementation-profile paper_reference
```

WSRL method fidelity:

```bash
python run_experiment.py --algorithm wsrl \
  --env-name hopper-medium-replay-v2 --corruption clean --stage both \
  --suite-profile method_fidelity \
  --implementation-profile official_code_reference --seed 0
```

Cal-QL locomotion port:

```bash
python run_experiment.py --algorithm cal_ql \
  --env-name hopper-medium-replay-v2 --corruption clean --stage both \
  --suite-profile common_budget_robustness \
  --implementation-profile locomotion_port --seed 0
```

Exact Off2OnRL/PQE is intentionally unavailable. This command must fail before
training until the five official member checkpoints and exact port exist:

```bash
python run_experiment.py --algorithm pessimistic_q_ensemble \
  --env-name hopper-medium-replay-v2 --corruption clean --stage both \
  --suite-profile method_fidelity \
  --implementation-profile official_code_reference
```

Common-budget 5×5:

```bash
python run_55_experiment.py \
  --env-name hopper-medium-replay-v2 \
  --suite-profile common_budget_robustness --seeds 0
```

Paired adversarial RNG run (repeat for each intended learner seed):

```bash
python run_experiment.py --algorithm rpex \
  --env-name hopper-medium-replay-v2 \
  --corruption adversarial --corruption-target observations --stage both \
  --suite-profile method_fidelity \
  --implementation-profile official_code_reference \
  --learner-seed 0 --corruption-seed 10001
```

Episode-boundary resume smoke: start the bounded local diagnostic below,
interrupt only after an online checkpoint is logged, then reissue the identical
semantic arguments with `--resume-run <run_dir>`:

```bash
python run_experiment.py --algorithm riql_naive \
  --env-name hopper-medium-replay-v2 --corruption clean --stage both \
  --protocol local_gymnasium_v4_diagnostic --allow-diagnostic-protocol \
  --offline-steps 10 --online-steps 200 --eval-episodes 2 \
  --online-checkpoint-period 10 --output-dir /tmp/o2o-resume-smoke
```

## I. 2026-08-21 P0 re-audit and launch decision

### Pre-change findings

| Priority | Local file/function | Pre-change behavior | Pinned upstream behavior | Result impact |
|---|---|---|---|---|
| P0 | `agents/sac_family.py:SACEnsembleAgent.update` | WSRL target entropy was `0.0`; temperature used the generic exponential-log-alpha loss | `wsrl/agents/sac.py:SACAgent.create` resolves nonnegative entropy to `-action_dim`; `temperature_loss_fn` uses the softplus Geq multiplier | Changes entropy pressure and therefore actor/critic fine-tuning |
| P0 | `corruption.py:_corrupt_target_values` and `corrupt_online_transition` | Random reward support remained `[-30,30]` for every severity | Requested severity semantics preserve the upstream range-1 behavior and scale support by `corruption_range` | Invalidated reward severity sweeps |
| P0 | `corruption.py:corrupt_online_transition` | Observation/dynamics always used offline dataset std | `RPEX/attack_online.py` passes unit std for online observation/dynamics and dataset action std for actions | Changed dimension-wise perturbation bounds |
| P0 | `config.py` | Experimental sign-PGD was the default profile | RPEX official-code-aligned runs require the Adam attacker and verified weights/settings | Could silently mislabel an experimental attack as RPEX |
| P0 | `corruption.py:corrupt_online_transition` | Online adversarial reward used `-range * reward` | `RPEX/attack.py:corrupt_trans` uses selected `Uniform(-1,1)` replacement online; offline uses sign flip | Mixed two distinct attack definitions |
| P1 | `run_55_experiment.py` | Research and common-budget vocabulary did not expose the requested primary/diagnostic split | Primary must exclude non-exact PQE; diagnostic may retain the approximation | Could place an approximation in a primary suite |

### Result

- WSRL official entropy is `-3` for Hopper and `-6` for HalfCheetah/Walker2d;
  `legacy_zero` is the only zero-entropy profile. The resolved value and
  parameterization are in config, manifest, path hash, and checkpoints.
- Random reward supports are exactly `0`, `[-15,15]`, `[-30,30]`, and
  `[-60,60]` for ranges `0`, `0.5`, `1`, and `2`.
- Online scale, attack timing, adversarial optimizer, offline/online reward
  rules, selection hash, and value hash are persisted and aggregation-strict.
- `experimental_sign_pgd` requires both its profile flag and
  `--allow-experimental-adversarial-attack`.
- `primary_research_benchmark` excludes PQE. `common_budget_diagnostic` retains
  it only as `PQE shared-actor approximation` and labels the entire suite as
  not paper reproduction. `final_benchmark` requires seeds `0,1,2,3,4`.

### Verification classification

- Unit/mock tested: 115 total; 114 passed, 1 platform skip, 0 failed.
- Ruff, `git diff --check`, and byte-code compilation: passed.
- Local diagnostic tested: five clean algorithms, WSRL entropy path, random
  reward range `0.5`, official unit-scale random dynamics, and official online
  adversarial reward; all bounded runs completed. Outputs were written only
  below `/tmp`.
- Strict D4RL runtime: not tested. The strict preflight correctly failed on
  macOS before environment creation because it is Linux x86_64-only.
- Pytest: not installed in the `corruption` environment; base pytest cannot
  collect the suite because base Python has no Torch. The same tests were run
  with `unittest discover` in the target environment.
- Mypy: not run; the repository has no mypy configuration or dependency.

### Full experiment decision: CONDITIONAL GO

The five-algorithm common-budget diagnostic suite is runnable. The primary
research suite is ready for RPEX, RIQL-naive, WSRL, and the explicitly labelled
Cal-QL locomotion port only after the strict Linux preflight passes. A five-
algorithm paper-faithful primary claim remains blocked because exact
Off2OnRL/PQE is not implemented; the shared-actor approximation is never used
as an automatic replacement.
