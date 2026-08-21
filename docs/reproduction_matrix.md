# Reproduction matrix

Audit basis: the pinned public repositories below and the local implementation,
reviewed on 2026-08-21. An implementation-profile name or a unit test is not an
end-to-end parity certificate.

## Custom research benchmark contract

The separate `run_purpose=research_benchmark` path is not an exact reproduction
claim and deliberately does not consume parity receipts or certificates. Its
main table contains only `rpex` (`source_aligned_port`), `riql_naive`
(`source_aligned_port`), and `wsrl` (`framework_port`). It permits explicit
offline/online budgets, seeds, evaluation interval, episode count, and final
window size.

`cal_ql_locomotion_adaptation` is an `optional_adapted` task adaptation because
the pinned Cal-QL release does not publish a locomotion recipe.
`pqe_shared_actor_approx` is an `optional_diagnostic` approximation and must not
be reported as the independent-policy Pessimistic Q-Ensemble. Both are excluded
from `research_summary.csv`.

All research main baselines consume the same cached offline corruption artifact
and use replay-transition poisoning: online policies observe the clean state,
the environment receives the clean policy action, and only the selected field
of the transition stored in replay is modified. Evaluation uses a separate,
clean environment and deterministic actions. The common final statistic first
averages each seed's last `K` evaluations (default `K=3`), then computes the
mean and population standard deviation across seed-level scores.

**Final benchmark status: NOT READY. Strict-final algorithm set: empty.** No
current `run_55` learner is an `exact_upstream_port`,
`official_adapter_verified`, or `end_to_end_verified` implementation. In
particular, RPEX and RIQL-naive remain handwritten `source_aligned_port`s with
only partial fixed-batch evidence; they are not strict-eligible. The audit must
therefore report the RPEX/RIQL eligible subset, the five-baseline suite, random
corruption, adversarial corruption, save/resume, strict environment, and the
final benchmark as separate fail-closed decisions.

| Algorithm | Pinned source | Local implementation | Learner parity | Reporting rule | Strict inclusion |
|---|---|---|---|---|---|
| RPEX | [`felix-thu/RPEX`](https://github.com/felix-thu/RPEX) `35da71ee5151b6179d21b9a2b4ce1b6408aedd04` | Handwritten PyTorch `source_aligned_port` | `fixed_batch_partial`; no upstream-executed end-to-end learner/optimizer certificate | RPEX final-three rule is source-backed | No |
| RIQL-naive | RPEX-vendored RIQL at the same pinned commit; original [`YangRui2015/RIQL`](https://github.com/YangRui2015/RIQL) is provenance only | Handwritten PyTorch `source_aligned_port` | `fixed_batch_partial`; no upstream-executed end-to-end learner/optimizer certificate | RPEX final-three rule is source-backed | No |
| WSRL | [`zhouzypaul/wsrl`](https://github.com/zhouzypaul/wsrl) `ad4dc1248a138bc15d6e053f2d1dba1b8cfbaca2` | JAX/Flax-to-PyTorch `framework_port_unverified` | TD/CQL/actor/temperature/target/optimizer-step parity missing | `verified=false`; the local terminal rule is not yet certified by an upstream reporting fixture | No |
| Cal-QL | [`nakamotoo/Cal-QL`](https://github.com/nakamotoo/Cal-QL) `ac6eafec22e8d60836573e1f488c7f626ce8a77e` | D4RL-locomotion `task_port` | Official recipes cover AntMaze and Adroit, not Hopper/HalfCheetah/Walker2d | Unverified for this unsupported task port | No; diagnostic only |
| Pessimistic Q-Ensemble | [`shlee94/Off2OnRL`](https://github.com/shlee94/Off2OnRL) `6f298fa9ef040d725067d0f2775022bd2900d635` | `pqe_shared_actor_approx` / `approximation` | Official five independently pretrained actor/twin-critic members and policy moment matching are absent | Unverified for the approximation | No; diagnostic only |

## Strict and diagnostic contracts

- `strict_final_algorithms()` currently returns `()`. Consequently
  `primary_research_benchmark` and `final_benchmark` must reject before creating
  a run directory. There is no authorized final-run command while this remains
  true.
- A future final run would additionally require a valid audit receipt bound to
  the exact clean repository commit/worktree, upstream commits, fixture hashes,
  runtime/platform, command, return code, and timestamp; exact seeds
  `0,1,2,3,4`; official budgets; pinned datasets/checkpoints; certified
  conditions; full save/resume coverage; and verified reporting. A passing
  environment smoke alone is insufficient.
- `500,000` offline plus `500,000` online is the default
  `common_budget_diagnostic` schedule. It never becomes publication-eligible by
  using the legacy environment backend or by completing successfully.
- `paper_reproduction` is reserved and rejected. No baseline currently has a
  complete paper-specific task, learner, condition, environment, seed, budget,
  save/resume, and reporting certificate.
- `common_benchmark_eligible` is a separate, explicitly non-paper comparison
  label. It must not be used as evidence for `publication_eligible` or
  `paper_reproduction_eligible`.

## Condition provenance

- Diagnostic `--corruption-suite random` means **clean plus four random
  targets**. The strict random condition set contains only the four actual
  random targets: observations, actions, rewards, and dynamics.
- Clean is a `benchmark_transfer` for RPEX/RIQL because pinned
  `RIQL_TRAIN_CONFIG.py` has no clean row. It is never auto-certified merely
  because a clean run completed.
- The v1 random fixture is upstream-derived but was generated under a runtime
  different from the strict pins. It is diagnostic evidence, not a strict
  condition certificate. A strict-runtime v2 receipt is missing.
- The v1 Hopper adversarial-observation fixture covers only the optimizer core.
  It cannot authorize a whole wrapper, a learner, or any strict adversarial
  condition. The strict adversarial condition set is therefore empty.
- HalfCheetah/Walker2d adversarial observations and all adversarial actions,
  rewards, dynamics, and mixed rows are diagnostic-only. The pinned public
  adversarial wrapper also contains execution defects; local repairs remain a
  source-aligned port, not upstream equivalence.
- Applying the RPEX corruption protocol to WSRL, Cal-QL, or PQE is a
  `benchmark_transfer`/diagnostic extension, not that baseline's paper
  condition.

## Missing executable evidence

The following evidence is absent and must stay fail-closed until produced by an
executable workflow on the exact pinned Linux stack:

- RPEX and RIQL-naive end-to-end learner/optimizer parity receipts generated
  against the pinned upstream implementation;
- RPEX online-constructor RNG-consumption and evaluation environment/reset/RNG
  parity receipts;
- strict-runtime v2 random-corruption fixture/receipt;
- an end-to-end adversarial-wrapper fixture/receipt (the existing optimizer-core
  v1 fixture is insufficient);
- per-baseline × certified-condition online checkpoint save/resume-equivalence
  receipts covering the complete serialized replay, RNG, optimizer, scheduler,
  environment, counters, and logs;
- a strict Linux x86_64 environment receipt with pinned Gym, D4RL, MuJoCo,
  NumPy, PyTorch, dataset hashes, and attacker-checkpoint hashes;
- WSRL fixed-batch numerical parity and source-backed reporting receipts.

Receipts are external evidence, not hand-edited booleans. Missing, malformed,
stale, source-mismatched, runtime-mismatched, nonzero-return-code, or dirty-tree
receipts must be rejected.

## Relevant upstream numerical protocol

- RPEX offline corruption uses `np.random.RandomState(seed)`, a Bernoulli mask,
  and a CPU `torch.Generator().manual_seed(seed)`. Online NumPy corruption
  restarts from the experiment seed and consumes candidate noise even when the
  mask is false.
- RPEX defaults are offline rate `0.3`, online rate `0.5`, and epsilon `1.0`.
  Online observation/dynamics use unit scale in normalized coordinates,
  actions use dataset standard deviation, and random online rewards use
  `30 × U[-1,1]`.
- Offline replay samples with global-device `torch.randint` with replacement.
  Online replay uses global Python `random` and `random.sample` without
  replacement.
- The pinned attack entry point uses RIQL's AWR policy update for every target.
  AlignIQL is a diagnostic local option.
- The upstream online phase creates fresh Adam optimizers and disables the
  offline actor scheduler. The local implementation follows that intended
  transition but lacks a certificate for the full constructor RNG trajectory.
- Upstream epsilon-greedy evaluation consumes a stochastic-policy sample and a
  CPU Torch mask before replacement by the greedy branch. Local action-level
  behavior is aligned, but the separate evaluation environment and reset/seed
  schedule are not certified equivalent.
- WSRL locomotion's intended structure is CQL pretraining, 250k offline
  updates, REDQ 10 critics/2-target subset, critic UTD 4, a frozen 5k warmup,
  online-only replay, 500k online steps, and 20 deterministic evaluations.

## Reporting eligibility

- RPEX/RPEX-vendored RIQL use the source-backed rule: sort online evaluations,
  average the final three per seed, then compute the mean and population
  standard deviation across seed scalars. This verified rule does not repair
  missing learner or condition parity.
- WSRL reporting remains `verified=false`. A terminal-20-episode local summary
  and the repository's common final-three metric may be emitted only with their
  distinct non-paper labels.
- Cal-QL locomotion and local PQE cannot emit source-primary paper summaries
  because one is an unsupported task port and the other is an approximation.
- `paper_reproduction_summary.csv` contains only rows satisfying every
  fail-closed paper eligibility field. An empty file is the correct current
  result. Diagnostic outputs belong only in explicitly labeled common or
  diagnostic summaries.

The machine-readable sources of truth are
`BASELINE_REPRODUCTION_REGISTRY`/`REPORTING_RULES` in
`robust_o2o/fidelity.py`, the condition sets in `run_55_experiment.py`, and the
receipt validator/audit. Documentation never grants eligibility.
