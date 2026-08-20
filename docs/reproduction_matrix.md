# Reproduction matrix

Audit basis: pinned public repositories listed below, inspected on 2026-08-21.
No complete learner has an automated end-to-end optimizer parity certificate,
so no current baseline is classified as `exact_upstream_port`. The
`official_code_reference` profile name selects public-code semantics; it does
not by itself certify implementation equivalence.

**Complete five-baseline final status: NOT READY. Eligible-subset status: NOT
READY.** The narrower strict-eligible candidate subset is `rpex,riql_naive`,
both `source_aligned_port`. It is currently blocked because executable
save/resume equivalence covers only one diagnostic path, not every eligible
baseline × certified condition; the fresh-process online-constructor RNG
trajectory and the upstream evaluation environment/RNG schedule are also not
matched. In addition, both golden fixtures were generated under NumPy/PyTorch
versions different from the pinned strict runtime and must be regenerated
there. After those gaps are closed, the exact strict contract and executable
audit must still pass on a supported Linux x86_64 host before launch. A passing
environment check alone is not learner parity.

| Algorithm | Paper / pinned source | Official task support | Local implementation | Parity status | Strict locomotion inclusion | Remaining deviation |
|---|---|---|---|---|---|---|
| RPEX | *RPEX: Robust Policy Expansion for Offline-to-Online RL under Diverse Data Corruption*; [`felix-thu/RPEX`](https://github.com/felix-thu/RPEX) `35da71ee5151b6179d21b9a2b4ce1b6408aedd04` | D4RL MuJoCo v2; explicit random/adversarial observation, action, reward, dynamics rows | Handwritten PyTorch `source_aligned_port` | Registered random fixtures for all four single targets; adversarial golden coverage is Hopper observation-target optimizer core only; full learner parity is absent | Conditional allowlist only; strict D4RL preflight, audit, seed/budget contract, and required fixture/reporting checks must pass | Unified local controller; fresh online-constructor and evaluation RNG trajectories differ; fixture runtime differs from strict runtime; official adversarial wrapper has public-code bugs; clean has no RIQL config-table row |
| RIQL-naive | *Towards Robust Offline Reinforcement Learning under Diverse Data Corruption*; original [`YangRui2015/RIQL`](https://github.com/YangRui2015/RIQL), benchmarked implementation from pinned RPEX commit above | RPEX-configured D4RL MuJoCo v2 target rows | Handwritten PyTorch `source_aligned_port` | Partial formula plus shared corruption fixtures; adversarial certification is observation-core only; no complete optimizer parity | Same conditional allowlist and gates as RPEX | The compared implementation is RPEX-vendored RIQL; fresh online-constructor and evaluation RNG trajectories are not matched |
| WSRL | *Efficient Online Reinforcement Learning Fine-Tuning Need Not Retain Offline Data*; [`zhouzypaul/wsrl`](https://github.com/zhouzypaul/wsrl) `ad4dc1248a138bc15d6e053f2d1dba1b8cfbaca2` | AntMaze, Adroit, Kitchen, and D4RL locomotion | JAX/Flax-to-PyTorch `framework_port_unverified` | Local invariants only; no upstream-exported TD/CQL/actor/temperature/optimizer-step fixture | No | Fixed-batch numerical parity is missing; corruption conditions are RPEX benchmark transfers |
| Cal-QL | *Cal-QL: Calibrated Offline RL Pre-Training for Efficient Online Fine-Tuning*; [`nakamotoo/Cal-QL`](https://github.com/nakamotoo/Cal-QL) `ac6eafec22e8d60836573e1f488c7f626ce8a77e` | Public recipes: AntMaze and Adroit | PyTorch locomotion `task_port` | Unsupported-task port; no official locomotion fixture | No; diagnostic only | No official Hopper/HalfCheetah/Walker2d recipe or locomotion hyperparameter table |
| Pessimistic Q-Ensemble | *Offline-to-Online Reinforcement Learning via Balanced Replay and Pessimistic Q-Ensemble*; [`shlee94/Off2OnRL`](https://github.com/shlee94/Off2OnRL) `6f298fa9ef040d725067d0f2775022bd2900d635` | Public D4RL MuJoCo v0 recipes | `pqe_shared_actor_approx` / `approximation` | No official fixed-batch parity | No; diagnostic only | Official code loads five independent actor/twin-critic checkpoints (`seed + 4*i`) and moment-matches policies; local code has one shared actor |

The complete five-baseline locomotion suite is currently **not ready** as a
strict final benchmark: WSRL parity is unverified, Cal-QL locomotion is a task
port, and local PQE is an approximation. The registry conditionally allowlists
only RPEX and RIQL-naive as source-aligned strict candidates; this is a narrower
candidate benchmark scope and is not a claim that either learner exactly
reproduces the upstream optimizer trajectory. Its current status is still
`NOT READY`.

## Strict and diagnostic contracts

- `final_benchmark` is limited to the eligible subset, exact seeds
  `0,1,2,3,4`, the pinned legacy D4RL-v2 Linux environment, and the official
  RPEX/RIQL-naive budgets: `2,000,001` offline updates and `1,000,001`
  requested online steps. Full eligible-baseline × certified-condition
  save/resume equivalence, online-phase constructor RNG parity, evaluation RNG
  parity, strict-runtime fixture regeneration, and the audit gate must pass
  before a run directory is created.
- `500,000` offline plus `500,000` online is the default
  `common_budget_diagnostic` schedule. It is never upgraded to a final or paper
  reproduction by using the legacy environment backend.
- `paper_reproduction` is reserved and rejected. No current baseline has a
  complete paper-specific task, seed, budget, environment, learner,
  corruption, and reporting contract.
- Do not launch a final run on the current macOS machine or on Linux while any
  of the save/resume, online-constructor RNG, or evaluation RNG blockers remain.
  After those blockers are resolved, run
  `python scripts/audit_reproducibility.py` first on the intended Linux x86_64
  host and require `ELIGIBLE-SUBSET BENCHMARK STATUS: READY`. The separate
  five-baseline status remains `NOT READY` until the excluded baselines are
  resolved.

## Condition provenance

- RPEX/RIQL random observation/action/reward/dynamics rows are `paper_reproduction_condition` at the corruption-protocol level. The learner remains `source_aligned_port`, not exact, and the reserved `paper_reproduction` run purpose still rejects them.
- RPEX/RIQL adversarial **observations on Hopper** are the only adversarial row with a registered golden optimizer-core certificate. HalfCheetah/Walker2d observations and all adversarial actions/rewards/dynamics are `paper_condition_fixture_unverified` and diagnostic-only.
- RPEX/RIQL clean is `benchmark_transfer`: pinned `RIQL_TRAIN_CONFIG.py` has no clean row.
- Corruption applied to WSRL or another baseline is `benchmark_transfer` with `corruption_protocol_source=felix-thu/RPEX@35da71e…`; it is not a WSRL paper condition.
- Cal-QL locomotion is `non_publication_diagnostic`.
- Local shared-actor PQE is a `diagnostic_extension` and must never be labeled official PQE.

## Relevant upstream numerical protocol

- RPEX offline corruption uses `np.random.RandomState(seed)`, a Bernoulli mask, and a CPU `torch.Generator().manual_seed(seed)`. Online NumPy corruption restarts from the same experiment seed and consumes candidate random noise even when the Bernoulli mask is false.
- RPEX defaults are offline rate `0.3`, online rate `0.5`, and epsilon `1.0`. Online observation/dynamics use unit scale in normalized coordinates, actions use exact dataset standard deviation, and online random rewards use `30 × U[-1,1]` independent of epsilon.
- The pinned offline replay draws indices with global-device `torch.randint`
  with replacement. Online replay seeds global Python `random` with the
  experiment seed and draws batches with `random.sample` without replacement.
- The pinned RPEX attack entry point uses RIQL's AWR policy update for every
  corruption target; AlignIQL remains an explicitly diagnostic local option.
- The online phase creates fresh Adam optimizers and disables the offline actor
  scheduler. Local strict profiles now match that state transition, but do not
  reproduce the complete fresh-process constructor RNG consumption order.
- Official epsilon-greedy evaluation first samples the stochastic policy,
  consumes a CPU `torch.rand` mask, and replaces the selected rows with the
  greedy branch. That action-level call order is matched, while the local
  separate-evaluation-environment reset/seed schedule is not.
- The official online controller stops only at the first episode boundary where `total_numsteps > requested`; actual steps therefore overshoot. It does not add an evaluation at a non-divisible final step.
- WSRL locomotion uses CQL pretraining, 250k offline updates, REDQ 10 critics/2-target subset, critic UTD 4, a frozen 5k warmup, online-only replay, 500k online steps, and 20 deterministic evaluation trajectories.

## Golden fixtures

- `tests/fixtures/rpex_random_corruption_v1.json` records outputs generated by directly executing pinned `Attack.sample_indexs`, `corrupt_*`, and `corrupt_trans` methods. It covers offline observation/action/reward/dynamics plus online mask/value/RNG-tail sequences. A strict run still requires the automated fixture check to pass.
- `tests/fixtures/rpex_adversarial_core_v1.json` records observation-target outputs generated by directly loading pinned `attack.py`, pinned `EDAC.py`, and the pinned **Hopper** EDAC checkpoint. It covers selected rows, the ten-split offline observation optimizer trajectory, and the online observation two-step core, including initial internal/effective perturbations and post-update objectives. It does **not** certify HalfCheetah/Walker2d, adversarial actions/rewards/dynamics, or a whole wrapper/learner.
- The adversarial fixture deliberately claims observation-target optimizer-core parity only. The public wrapper at the pinned commit returns undefined `std` in adversarial branches, has a cache unpacking mismatch, and passes an actor tuple into the dynamics critic. The executable local bug-fix path is therefore `source_aligned_port`, not exact; adversarial actions/rewards/dynamics remain diagnostic-only.
- The fixtures currently record NumPy `2.2.6` and PyTorch `2.13.0`, while the
  strict environment pins NumPy `1.23.5` and PyTorch `2.5.1`. The audit treats
  this as a blocking mismatch until the upstream generators are rerun in the
  pinned strict environment and the external certificates are reviewed.

## Save/resume verification scope

The executable preflight save/resume equivalence check covers only RIQL-naive
with random observation corruption under the bounded
`common_budget_diagnostic` smoke path. It resumes an offline checkpoint and
compares the final agent, normalizer, evaluation metrics, and online corruption
audit; it does not exercise online-checkpoint restore or compare every saved
replay/RNG state. It also does not cover both strict candidates across clean,
every random target, and certified adversarial observations.
Full per-baseline/per-condition save/resume equivalence is therefore an
explicit **blocking** gate for the eligible subset, not an inferred property of
checkpoint serialization. Consequently the eligible subset is currently
`NOT READY` even on an otherwise compatible Linux host.

## Reporting rules

- RPEX and RPEX-vendored RIQL: sort online evaluations per run, average the final three per seed, then compute the mean and declared population standard deviation over seed scalars. Evaluations use 10 episodes. Missing final evaluations, duplicate steps/runs, or missing strict seeds fail.
- WSRL: terminal online evaluation with 20 deterministic trajectories. No source-backed final-three smoother exists; final-three WSRL output is a separately labeled common benchmark metric.
- Cal-QL locomotion and local PQE cannot emit a source-primary paper summary because the task/implementation is unsupported or approximate.

`paper_reproduction_summary.csv` accepts only rows whose verified provenance
sets `paper_reproduction_eligible=true`. Current `source_aligned_port` RPEX and
RIQL-naive rows therefore do **not** enter that file; they remain available in
`per_seed_final_scores.csv` after a publication-eligible final run and in the
explicitly labeled common-benchmark summary. An empty paper summary is the
intentional fail-closed result, not missing output.

The machine-readable source of truth is `BASELINE_REPRODUCTION_REGISTRY` and `REPORTING_RULES` in `robust_o2o/fidelity.py`. Strict eligibility is never inferred from a profile name alone.
