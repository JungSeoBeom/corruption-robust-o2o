# Corruption-robust O2O reproduction matrix

Audit basis: the pinned public repositories below and the local implementation,
reviewed on 2026-08-21. This document describes the executable custom research
benchmark. It does not claim bitwise or paper-budget reproduction.

## Five-main-baseline contract

`run_purpose=research_benchmark` contains exactly these five main baselines:

```text
rpex
riql_naive
wsrl
cal_ql
pessimistic_q_ensemble
```

Every method consumes the same condition-specific cached offline corruption
artifact. Online corruption follows replay-transition poisoning: the clean
state selects the action, that action is sent to the environment, and the
selected field is modified only in the transition written to learner replay.
Evaluation always uses a separate clean environment. The common result first
averages each seed's last `K` clean evaluations (default `K=3`), then reports
mean and population standard deviation across seed-level values.

Cal-QL and Pessimistic Q-Ensemble are main-table algorithms. Their qualified
implementation scope is provenance, not a reason to omit them. Historical
manifest values `cal_ql_locomotion_adaptation` and `pqe_shared_actor_approx`
are read-only aliases and cannot be used for a new run.

| Algorithm | Pinned source | Main local implementation | Task/version scope | Important limitation |
|---|---|---|---|---|
| RPEX | [`felix-thu/RPEX`](https://github.com/felix-thu/RPEX) `35da71ee5151b6179d21b9a2b4ce1b6408aedd04` | Source-aligned RIQL pretraining plus policy expansion and IPW | D4RL locomotion v2 | Method-faithful evaluation is stochastic; deterministic evaluation is secondary diagnostic output |
| RIQL-naive | RPEX-vendored RIQL at the same pinned commit | Source-aligned RIQL objective in offline and online phases | D4RL locomotion v2 | Online phase intentionally creates fresh positive-LR actor/critic/value optimizers |
| WSRL | [`zhouzypaul/wsrl`](https://github.com/zhouzypaul/wsrl) `ad4dc1248a138bc15d6e053f2d1dba1b8cfbaca2` | PyTorch framework port of CQL-REDQ pretraining, frozen warmup, and online SAC | D4RL locomotion v2 | Cross-framework numerical equality is not claimed |
| Cal-QL | [`nakamotoo/Cal-QL`](https://github.com/nakamotoo/Cal-QL) `ac6eafec22e8d60836573e1f488c7f626ce8a77e` | `source_aligned_locomotion_adaptation` | Frozen Hopper/HalfCheetah/Walker2d adaptation on D4RL v2 | Public recipes cover AntMaze/Adroit, so this is not an official locomotion recipe |
| Pessimistic Q-Ensemble | [`shlee94/Off2OnRL`](https://github.com/shlee94/Off2OnRL) `6f298fa9ef040d725067d0f2775022bd2900d635` | `source_aligned_d4rl_v2_port` with five independent actor/twin-critic members | Public v0 method ported to the common D4RL-v2 artifact | Offline gradient compute is about 5× a single-agent baseline; the suite is interaction-matched, not compute-matched |

## Method-specific executable invariants

### RPEX and RIQL-naive

- The research profile uses the pinned unsquashed Gaussian policy behavior.
  Raw actions are sent to the environment and stored in replay without silent
  clipping; out-of-bound diagnostics are logged.
- The online update occurs before `env.step()` and replay insertion, matching
  the public RPEX loop's use of the existing replay contents.
- RIQL-naive resets actor, critic, and value Adam state at the online boundary
  and removes the exhausted offline cosine scheduler. Its actor learning rate
  remains positive.
- RPEX's primary clean evaluation uses the public epsilon/Q policy-switching
  behavior. A deterministic expansion evaluation is logged only as a secondary
  diagnostic and is never silently substituted for the primary result.

### WSRL

- Offline pretrainer: CQL with the REDQ critic ensemble.
- Critics/target subset: `10/2`; online critic UTD: `4`.
- Frozen-policy online collection: `5,000` steps.
- Online learning uses online replay only; the offline dataset is not retained
  in the online mixture.
- Actor/critic/temperature learning rates are `1e-4/3e-4/1e-4`, target entropy
  is `-action_dim`, and entropy is not added to the Bellman backup.

### Cal-QL frozen locomotion adaptation

- Actor and both Q functions use exactly two 256-unit hidden layers with
  orthogonal initialization. Actor/Q/temperature learning rates are
  `1e-4/3e-4/1e-4`; target update rate is `0.005`.
- The actor uses the SAC policy loss from the first update. BC warmup is `0`.
- Cal-QL calibration is active offline and online. Current-state and next-state
  policy-action proposals are both lower-bounded by valid MC returns.
- CQL uses 10 sampled actions, temperature 1, importance sampling, max-target
  backup, no backup entropy, and weight 5 in both phases.
- Offline MC returns are recomputed after corruption. Online transitions stay
  in a pending episode buffer until terminal or timeout; only complete
  trajectories enter replay with exact post-corruption return-to-go. No fake
  zero return is used.
- Online batch mixing dynamically uses
  `|D_offline|/(|D_offline|+|D_online-completed|)`. A completed trajectory
  triggers trajectory-length × UTD updates.

### Pessimistic Q-Ensemble D4RL-v2 port

- Exactly five members are initialized with seeds `base_seed + 4*i`. Each owns
  its actor, Q1, Q2, target Q1, target Q2, and optimizer state without shared
  parameter storage.
- Every member receives independent CQL pretraining batches from the same
  immutable corrupted artifact. `stage=both` writes five content-distinct
  member checkpoints before online fine-tuning; `stage=online` requires five
  distinct checkpoint paths.
- The deployed stochastic policy moment-matches the five pre-tanh Gaussians:
  it averages means and uses
  `mean(std² + mean²) - mean(mean)²` for variance, then samples and applies
  `tanh`. Deterministic clean evaluation applies `tanh` to the average mean.
- Online SAC uses each member's clipped twin-Q target. The actor maximizes the
  ensemble mean of member-wise minimum Q, with temperature tuning and target
  update rate `0.005`. CQL is offline-only.
- Balanced replay uses the source density-ratio loss and priority formula with
  temperature 5 and clipping `[1e-3, 1e3]`. Source-derived controls are initial
  online fraction `0.75`, first epoch multiplier `5`, first online block
  `1,000`, online buffer `250,000`, and weight batch `256`.
- Clean evaluation always uses the ensemble moment policy; it never selects a
  best member or best checkpoint.

## Corruption and reporting provenance

- Diagnostic/research `--corruption-suite random` means clean plus four
  individual replay-poisoning targets: observations, actions, rewards, and
  dynamics.
- Defaults follow RPEX: offline corruption rate `0.3`, online rate `0.5`, and
  corruption range `1.0`.
- Cal-QL recomputes MC return-to-go when reward corruption changes the stored
  reward sequence. Observation, action, and dynamics corruption leave that
  reward sequence unchanged.
- The same clean-evaluation schedule, episode count, final window, seed list,
  environment protocol, and interaction budgets are used for all five methods.
  PQE's five-member offline compute is reported separately rather than hidden.
- Failed, partial, or non-finite runs are not promoted to completed seed rows.
  No best checkpoint, best evaluation, or best ensemble member selection is
  used in the common summary.

## Readiness semantics

`scripts/check_research_readiness.py` is the practical preflight for this
custom benchmark. Its launch conditions are algorithm structure, replay/update
semantics, clean evaluation, adversarial checkpoint validity when selected,
and collision-free output paths. It deliberately does not require an upstream
RNG trajectory, numerical parity certificate, official paper step budget, or
fixed seed cohort.

Before training, checks that need completed trajectories or files are printed
as `PENDING`; they are never represented as passing runtime evidence. A clean
static result is therefore `CONFIG-READY / RUNTIME EVIDENCE PENDING`. Supplying
`--run-dir` makes the checker validate Cal-QL's completed-trajectory MC-return
metadata and PQE's five unique member checkpoint files/hashes. A supplied but
invalid run directory fails closed.

The older strict publication/audit machinery remains a separate historical
workflow. It does not gate `run_purpose=research_benchmark`, and its receipt or
parity status must not be confused with this executable custom comparison.
