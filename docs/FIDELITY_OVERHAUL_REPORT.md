# Baseline fidelity overhaul report (superseded snapshot)

Audit date: 2026-08-21

This report preserves the earlier overhaul history. Its former exactness and
launch-readiness conclusions are superseded by `docs/reproduction_matrix.md`
and `BASELINE_REPRODUCTION_REGISTRY`. No long reinforcement-learning run was
performed. The machine used for the checks was macOS, so strict D4RL-v2 runtime
success is intentionally not claimed.

Current conservative decision: **FINAL BENCHMARK NOT READY; strict-final
algorithm set empty**. RPEX and RIQL-naive remain handwritten
`source_aligned_port`s with `fixed_batch_partial` evidence, not verified
upstream adapters or end-to-end learner ports. WSRL is an unverified framework
port and its reporting rule is `verified=false`; Cal-QL locomotion is a task
port; PQE is an approximation. The random and adversarial v1 fixtures are
diagnostic/runtime-mismatched, and the latter covers only the Hopper
observation optimizer core. Required learner, online-constructor/evaluation
RNG, strict condition, full save/resume, reporting, and Linux receipts are
missing. The current Mac is diagnostic-only. No final-run command is authorized.

Condition scope is also fail-closed: diagnostic random is clean plus four
random targets, while strict random contains only the four actual corruption
targets. Clean is a benchmark transfer, never automatically certified. Strict
adversarial contains no condition because an optimizer-core fixture cannot
certify an end-to-end wrapper. A future eligibility decision must validate
external receipts bound to the exact clean repository state, upstream commits,
fixture/dataset/checkpoint hashes, runtime/platform, command, return code, and
timestamp; a boolean or successful local test is insufficient.

## A. Baseline fidelity matrix

| Baseline | Upstream commit | Current conservative status | Parity status | Strict locomotion status |
|---|---|---|---|---|
| RPEX / RIQL-naive | `felix-thu/RPEX@35da71ee5151b6179d21b9a2b4ce1b6408aedd04` | `source_aligned_port` | `fixed_batch_partial`; no upstream-executed complete learner/optimizer parity | Excluded; `strict_final_eligible=false` |
| RPEX paper interpretation | Same source plus paper semantics | `paper_code_conflict` sensitivity profile | No complete learner optimizer parity | Not the default final profile |
| WSRL locomotion | `zhouzypaul/wsrl@ad4dc1248a138bc15d6e053f2d1dba1b8cfbaca2` | `framework_port_unverified` | Fixed-batch JAX/PyTorch and source-backed reporting parity missing | Excluded |
| Cal-QL locomotion | `nakamotoo/Cal-QL@ac6eafec22e8d60836573e1f488c7f626ce8a77e` | `task_port` | Official public recipes are AntMaze/Adroit, not D4RL locomotion | Excluded; diagnostic only |
| Pessimistic Q-Ensemble | `shlee94/Off2OnRL@6f298fa9ef040d725067d0f2775022bd2900d635` | `approximation` | Five independent checkpoint-loaded actors/twin critics are not implemented | Excluded; diagnostic only |

The provenance file is `docs/baseline_fidelity_manifest.yaml`; the current
machine-readable eligibility source is `BASELINE_REPRODUCTION_REGISTRY`.

## B. Paper/code conflicts

| RPEX behavior | `official_code_reference` | `paper_reference` |
|---|---|---|
| Online replay | Online-only replay | Offline/online mixture |
| Evaluation | Epsilon policy switching | Greedy highest-weight branch |
| Observation/action corruption timing | Post-transition replay poisoning | Pre-action sensor/actuator corruption |
| Result identity | `source_aligned_port` | `paper_code_conflict` |

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
- `robust_o2o/networks.py`: source-aligned unsquashed RPEX Gaussian and hidden
  stack/initialization, WSRL two-block LayerNorm architecture and final-hidden
  `1e-2` initialization.
- `robust_o2o/agents/iql_family.py`: RIQL quantile/clipped robust loss, official
  AWR policy updates for every corruption target, diagnostic-only AlignIQL,
  source-matched epsilon-greedy sampling, actor/critic learning rates, and the
  offline-to-online fresh-Adam/no-scheduler transition.
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
  official offline global-Torch and online global-Python replay sampling,
  deterministic episode-boundary resume, manifest-tagged atomic checkpoints,
  append-only metrics with cumulative elapsed time.
- `robust_o2o/manifest.py`: canonical sorted manifest and SHA-256 plus a
  seed-excluding aggregation signature.
- `plot_results.py`: manifest-strict aggregation, completed-only benchmark
  summaries, explicit common-budget/approximation/oracle labels, and opt-in
  running-run diagnostics.
- `run_55_experiment.py`, `run_all_algorithms.py`, and `run_matrix.py`: suite and
  implementation profile propagation. The diagnostic `run_55` default remains
  500k/500k and is explicitly common-budget, not paper reproduction. Strict
  RPEX/RIQL-naive runs instead lock 2,000,001/1,000,001 and seeds 0..4.
- `scripts/preflight_strict.py`: Linux x86_64 pinned D4RL-v2 bounded preflight.
- `tests/`: fixed-tensor, architecture, schedule, paired-corruption, atomic
  multiprocessing cache, manifest, and deterministic checkpoint/resume tests.

## D. Corruption verification

- The RPEX public-code profile uses `np.random.RandomState(seed)` and maps the
  corruption seed to the experiment seed. Diagnostic extensions keep separate
  provenance and cannot be promoted to the strict result group.
- `generic_partitioned_mixed` assigns every selected transition to exactly one
  of observation/action/reward/dynamics. It is not promoted to RPEX paper mixed.
- Adversarial initialization uses the source-matched CPU Torch generator. The
  Adam optimizer core follows the pinned step schedules—offline `100 × 0.01`
  and online `2 × 0.1`—plus the public-code scaling and objective. The pinned
  wrapper contains execution defects, so the v1 fixture checks only the
  **Hopper observation-target optimizer core**, not HalfCheetah, Walker2d,
  actions, rewards, dynamics, the whole wrapper, or learner equivalence. Its
  runtime also differs from the strict pins; it certifies no strict condition.
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
  Gym 0.23.1/D4RL/`mujoco-py==2.1.2.14` runtime linked to native MuJoCo code
  `210` is platform-specific. This is an explicit unverified runtime, not a
  local-protocol success.
- `scripts/preflight_strict.py` verifies pinned versions, the complete ID,
  dataset, dimensions/SHA, reset/step, 10 offline updates, and 20 online steps.
  Its executable save/resume equivalence coverage is only RIQL-naive with
  random observation corruption in the bounded `common_budget_diagnostic`
  smoke path, resumed from an offline checkpoint. Online-checkpoint restore and
  equality of every saved replay/RNG state are not exercised. It does not
  certify every strict baseline/condition.

## F. Saving and resume mechanism and verification scope

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

The serialization and deterministic-update tests above do not establish full
benchmark coverage. The end-to-end preflight has exercised only RIQL-naive plus
random observation corruption under the diagnostic smoke configuration. It has
not produced a source-bound receipt for complete online-checkpoint state or a
full baseline × certified-condition matrix. The strict algorithm set is empty,
and no adversarial condition is certified. Serialization tests and this smoke
must not be promoted into a save/resume certificate.

## G. Historical bounded test snapshot

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

At the time of this superseded snapshot, 115 tests ran: 114 passed, 0 failed,
and 1 was skipped. Ruff, `diff --check`, and byte-code compilation passed. The
local diagnostic smoke passed; the strict Linux preflight was not executed.
These counts predate the current fixture, reporting, and registry tests and must
not be quoted as the current suite result.

## H. Superseded command examples

These examples document the earlier interface. They are not authorization for
a final run. Every command below is diagnostic/historical. The current strict
algorithm set is empty on every platform; neither a Linux environment smoke nor
an audit invocation turns these examples into final-run authorization.

RPEX source-aligned public-code profile, five seeds, random observations:

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

Historical WSRL method-profile example. The current strict registry rejects
WSRL because fixed-batch JAX/PyTorch parity is missing:

```bash
python run_experiment.py --algorithm wsrl \
  --env-name hopper-medium-replay-v2 --corruption clean --stage both \
  --suite-profile method_fidelity \
  --implementation-profile official_code_reference --seed 0
```

Optional Cal-QL locomotion adaptation:

```bash
python run_experiment.py --algorithm cal_ql_locomotion_adaptation \
  --env-name hopper-medium-replay-v2 --corruption clean --stage both \
  --run-purpose research_benchmark --suite-profile research_benchmark \
  --implementation-profile research_benchmark --seed 0
```

An upstream-equivalent Off2OnRL/PQE implementation is unavailable. This command
must fail before training until the five official member checkpoints and a
verified implementation exist:

```bash
python run_experiment.py --algorithm pessimistic_q_ensemble \
  --env-name hopper-medium-replay-v2 --corruption clean --stage both \
  --suite-profile method_fidelity \
  --implementation-profile official_code_reference
```

Custom-budget main 3×5 (clean plus four random targets):

```bash
python run_55_experiment.py \
  --env-name hopper-medium-replay-v2 \
  --run-purpose research_benchmark --suite-profile research_benchmark \
  --algorithms rpex,riql_naive,wsrl --seeds 0
```

Paired adversarial RNG run (repeat for each intended learner seed):

```bash
python run_experiment.py --algorithm rpex \
  --env-name hopper-medium-replay-v2 \
  --corruption adversarial --corruption-target observations --stage both \
  --suite-profile method_fidelity \
  --implementation-profile official_code_reference \
  --learner-seed 0 --corruption-seed 0
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

## I. Superseded 2026-08-21 P0 re-audit

### Pre-change findings

| Priority | Local file/function | Pre-change behavior | Pinned upstream behavior | Result impact |
|---|---|---|---|---|
| P0 | `agents/sac_family.py:SACEnsembleAgent.update` | WSRL target entropy was `0.0`; temperature used the generic exponential-log-alpha loss | `wsrl/agents/sac.py:SACAgent.create` resolves nonnegative entropy to `-action_dim`; `temperature_loss_fn` uses the softplus Geq multiplier | Changes entropy pressure and therefore actor/critic fine-tuning |
| P0 | `corruption.py:_corrupt_target_values` and `corrupt_online_transition` | Random reward support remained `[-30,30]` for every severity | Requested severity semantics preserve the upstream range-1 behavior and scale support by `corruption_range` | Invalidated reward severity sweeps |
| P0 | `corruption.py:corrupt_online_transition` | Observation/dynamics always used offline dataset std | `RPEX/attack_online.py` passes unit std for online observation/dynamics and dataset action std for actions | Changed dimension-wise perturbation bounds |
| P0 | `config.py` | Experimental sign-PGD was the default profile | RPEX official-code-aligned runs require the Adam attacker and verified weights/settings | Could silently mislabel an experimental attack as RPEX |
| P0 | `corruption.py:corrupt_online_transition` | Online adversarial reward used `-range * reward` | `RPEX/attack.py:corrupt_trans` uses selected `Uniform(-1,1)` replacement online; offline uses sign flip | Mixed two distinct attack definitions |
| P1 | `run_55_experiment.py` | Research and common-budget vocabulary did not expose the requested primary/diagnostic split | Primary must exclude approximate PQE; diagnostic may retain the approximation | Could place an approximation in a primary suite |

### Result

- The pinned WSRL source resolves entropy to `-3` for Hopper and `-6` for
  HalfCheetah/Walker2d;
  `legacy_zero` is the only zero-entropy profile. The resolved value and
  parameterization are in config, manifest, path hash, and checkpoints.
- Random reward supports are `0`, `[-15,15]`, `[-30,30]`, and
  `[-60,60]` for ranges `0`, `0.5`, `1`, and `2`.
- Online scale, attack timing, adversarial optimizer, offline/online reward
  rules, selection hash, and value hash are persisted and aggregation-strict.
- `experimental_sign_pgd` requires both its profile flag and
  `--allow-experimental-adversarial-attack`.
- At the time of this superseded snapshot, `primary_research_benchmark`
  conditionally allowlisted RPEX and RIQL-naive. That conclusion is withdrawn:
  both are `fixed_batch_partial` source-aligned ports and the current strict
  registry yields an empty algorithm set. `common_budget_diagnostic` may retain
  all five with non-publication labels. `paper_reproduction` remains reserved
  and rejected.

### Verification classification

- Unit/mock tested: 115 total; 114 passed, 1 platform skip, 0 failed.
- Ruff, `git diff --check`, and byte-code compilation: passed.
- Local diagnostic tested: five clean algorithms, WSRL entropy path, random
  reward range `0.5`, source-aligned unit-scale random dynamics, and the
  source-aligned online adversarial-reward rule; all bounded runs completed.
  The adversarial-reward result was diagnostic-only, not golden-fixture
  certification. Outputs were written only below `/tmp`.
- Strict D4RL runtime: not tested. The strict preflight correctly failed on
  macOS before environment creation because it is Linux x86_64-only.
- Pytest: not installed in the `corruption` environment; base pytest cannot
  collect the suite because base Python has no Torch. The same tests were run
  with `unittest discover` in the target environment.
- Mypy: not run; the repository has no mypy configuration or dependency.

### Current full experiment decision: NOT READY

The five-algorithm common-budget diagnostic suite may be used for code-path
checks, but it is not publication-eligible. The complete five-baseline strict
suite is blocked by missing WSRL fixed-batch parity, unsupported Cal-QL
locomotion recipes, the shared-actor PQE approximation, and the unexecuted
strict Linux runtime receipt. WSRL reporting is also unverified. Full
per-baseline/per-condition save/resume receipts are absent. RPEX and RIQL-naive
are source-aligned but not strict-eligible; their end-to-end learner,
online-constructor/evaluation RNG, and condition receipts are missing, and the
v1 fixture runtime differs from the strict pins. The strict-final algorithm set
is empty, so there is no narrower suite to launch and no final command to
publish.
`paper_reproduction_summary.csv` accepts only verified rows with
`paper_reproduction_eligible=true`; current source-aligned rows are excluded.
