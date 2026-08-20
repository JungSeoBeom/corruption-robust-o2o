# Historical Cal-QL and WSRL audit (superseded)

This document records the earlier two-baseline audit. The active, method-level
source-of-truth is `docs/baseline_fidelity_manifest.yaml`; generic `reference`
is no longer a valid profile.

This audit was performed against repository `HEAD` `b203477` before the
reference-profile changes. No long reinforcement-learning run was used.

## Reference snapshots

- Cal-QL: `nakamotoo/Cal-QL` at `ac6eafec22e8d60836573e1f488c7f626ce8a77e`
  (`JaxCQL/conservative_sac.py`). Its defaults define policy LR `1e-4`, Q LR
  `3e-4`, ten CQL actions, importance correction, max target backup, and no
  entropy backup.
- WSRL: `zhouzypaul/wsrl` at
  `ad4dc1248a138bc15d6e053f2d1dba1b8cfbaca2`
  (`experiments/configs/wsrl_config.py`, locomotion launch scripts, and
  `SACAgent.update_high_utd`). It uses a 10-critic ensemble, a 2-head target
  sample, LayerNorm, UTD 4 over a total batch of 1024, then one actor and
  temperature update. Target-head sampling uses `jax.random.randint`, so it is
  with replacement and duplicate heads are allowed.

## Static analysis of the previous implementation

- `ExperimentConfig` had no implementation profile or role-specific seeds.
  `run_55_experiment.py` defaulted to the local Gymnasium backend even though
  the single-run and all-algorithm CLIs defaulted to strict D4RL-v2.
- Cal-QL used one `learning_rate` for actor and critics, entered a BC-dominant
  branch through update 100,000, and had no selectable oracle-mask policy. Its
  importance-sampled CQL proposals were otherwise structurally close to the
  official implementation.
- WSRL used the minimum over all ten target critics, had no LayerNorm, and a
  generic update call changed critic, actor, and temperature together once per
  environment step. Its private CQL penalty omitted next-state proposals and
  proposal-density corrections.
- Result directories omitted protocol/profile, aggregation silently selected
  the latest duplicate run, and a one-seed plot substituted episode dispersion
  for seed uncertainty. Local scores were labeled as D4RL normalized returns.
- Checkpoints checked algorithm, environment name, protocol, and tensor
  dimensions, but not dataset hash, simulator/runtime, action bounds, horizon,
  or algorithm profile. Preprocessing, replay, learner, and environment RNGs
  were derived ad hoc from one seed.

## Resolved behavior

The later fidelity overhaul replaced `reference` with explicit implementation
and suite profiles. Existing results from this period are read as
`legacy_current`/`legacy_unknown`; they are never silently promoted. Resolved
names, all role seeds, optimizer rates, CQL/REDQ settings, network architecture
flags, environment provenance, and manifest hashes are serialized in every new
run.

The strict benchmark remains a fail-fast Gym 0.23.1/D4RL-v2/mujoco_py path with
no Gymnasium fallback. The local Gymnasium-v4 protocol is explicitly diagnostic
and requires an acknowledgement flag. Result paths and aggregations separate
both dimensions.
