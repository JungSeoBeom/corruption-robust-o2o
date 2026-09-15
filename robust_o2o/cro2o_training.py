"""Honest offline audits and candidate-specific replay scheduling."""
from __future__ import annotations

import copy
import numpy as np
import torch

from .agents.cro2o import CRO2OAgent, SourceAudit, center_uncertainty, trust_score, register_offline_blocks
from .environment import StateNormalizer, apply_normalizer
from .replay import OfflineDataset, mixed_batch


def prepare_candidate(agent, raw_dataset, offline):
    register_offline_blocks(agent, raw_dataset)
    if agent.prepared:
        return
    config = agent.config
    blocks = np.unique(agent.offline_blocks)
    folds = config.candidate_audit_folds
    if len(blocks) < folds:
        raise ValueError("CRO2O cross-fitting needs at least one trajectory block per fold")
    assignment = {int(block): i % folds for i, block in enumerate(blocks)}
    fold_ids = np.asarray([assignment[int(block)] for block in agent.offline_blocks])
    residuals = np.empty(offline.size, np.float32)
    uncertainties = np.empty(offline.size, np.float32)
    scales = np.empty(offline.size, np.float32)
    # CPU audit models have their own seed scope, including their target/value
    # components and fold-specific normalization. No pretrained full-data model
    # or full-data fitted normalizer enters a held-out predictor.
    with torch.random.fork_rng(devices=[]):
        for fold in range(folds):
            train = np.flatnonzero(fold_ids != fold)
            heldout = np.flatnonzero(fold_ids == fold)
            training = {k: v[train] for k, v in raw_dataset.items()}
            normalizer = StateNormalizer.fit(training, enabled=config.normalize_states,
                                             mode=config.state_normalization)
            normalized = apply_normalizer(training, normalizer)
            fold_config = copy.deepcopy(config)
            fold_config.algorithm = "care_o2o"
            # manual_seed() also resets accelerator RNGs, which fork_rng([])
            # does not restore. Audit networks are CPU-only: seed CPU only.
            torch.random.default_generator.manual_seed(config.learner_seed + 100003 + fold)
            predictor = CRO2OAgent(fold_config, agent.state_dim, agent.action_dim,
                                   agent.max_action, torch.device("cpu"))
            register_offline_blocks(predictor, training)
            sampler = OfflineDataset(normalized, seed=config.replay_seed + 100003 + fold)
            calibration = SourceAudit(config)
            for _ in range(config.candidate_audit_steps):
                batch = sampler.sample(config.batch_size, torch.device("cpu"))
                # Scale estimation also excludes the scored fold. Use earlier
                # training-fold prediction residuals, before fitting this batch,
                # not pooled OOF scales whose predictors could have seen it.
                with torch.no_grad():
                    implied = (predictor.q(batch["observations"], batch["actions"])
                               - config.discount * (1 - batch["terminals"]) * predictor.value(batch["next_observations"]))
                    center, _ = center_uncertainty(implied, config.candidate_mad_multiplier)
                    calibration.residuals.extend((batch["rewards"] - center).tolist())
                predictor.update(batch)
                agent.audit_updates += 1
            # Every prediction is out-of-fold; no held-out training afterwards.
            with torch.no_grad():
                for start in range(0, len(heldout), config.batch_size):
                    rows = heldout[start:start + config.batch_size]
                    s = torch.as_tensor(normalizer.transform(raw_dataset["observations"][rows]))
                    ns = torch.as_tensor(normalizer.transform(raw_dataset["next_observations"][rows]))
                    a = torch.as_tensor(raw_dataset["actions"][rows])
                    g = config.discount * (1 - torch.as_tensor(raw_dataset["terminals"][rows]))
                    implied = predictor.q(s, a) - g * predictor.value(ns)
                    center, uncertainty = center_uncertainty(implied, config.candidate_mad_multiplier)
                    residuals[rows] = raw_dataset["rewards"][rows] - center.numpy()
                    uncertainties[rows] = uncertainty.numpy()
                    scales[rows] = calibration.scale()
    agent.offline_weights = np.empty(offline.size, np.float32)
    for i, (residual, uncertainty) in enumerate(zip(residuals, uncertainties)):
        suspicion, weight = trust_score(torch.tensor(residual), torch.tensor(uncertainty),
                                       float(scales[i]), config)
        if not torch.isfinite(weight):
            raise ValueError("CRO2O offline audit produced a non-finite weight")
        agent.offline_weights[i] = weight.item()
        agent.audit[0].record(agent.offline_blocks[i], residual, suspicion.item(), weight.item())
    if agent.generative:
        for _ in range(config.candidate_generator_steps):
            batch = offline.sample(config.batch_size, agent.device)
            loss = agent.generator_loss(batch, offline=True)
            agent._step(loss, agent.generator_optimizer, agent.proposal.parameters())
            agent.generator_updates += 1
            agent.actor_updates += 1
            agent.total_updates += 1
        agent.target_proposal.load_state_dict(agent.proposal.state_dict())
    agent.prepared = True


def candidate_update(agent, offline, online, config, *, critic_only=False):
    if agent.adaptive:
        offline_batch = offline.sample(config.batch_size, agent.device)
        online_batch = online.sample(config.batch_size, agent.device, replace=True)
        return agent.update_sources(offline_batch, online_batch, critic_only)
    batch = mixed_batch(offline, online, config.batch_size, config.effective_offline_ratio,
                        agent.device, online_replace=True)
    return agent.update_sources(batch, None, critic_only)
