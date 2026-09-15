"""CARE-O2O, ARW-O2O, RG-O2O: September 2026 working-note candidates.

These are experimental implementations, not published/verified baselines.
Audit metadata is inferred exclusively from logged tuples, never attack labels.
"""
from __future__ import annotations

import copy
import hashlib
import math
from collections import deque

import numpy as np
import torch
from torch import nn

from ..dataset import assert_no_corruption_labels
from ..networks import QNetwork, ValueNetwork, TanhGaussianPolicy, mlp
from .base import BaseAgent, soft_update


VERSION = "cro2o_working_notes_20260915_v1"


def weighted_mean(values, weights):
    weights = weights.detach()
    return (values * weights).sum(-1) / (weights.sum(-1) + 1e-8)


def huber(residual, delta):
    return torch.where(residual.abs() <= delta,
                       residual.square() / (2 * delta), residual.abs() - delta / 2)


def center_uncertainty(q, multiplier):
    center = torch.quantile(q, .5, dim=0)
    return center, multiplier * torch.quantile((q - center).abs(), .5, dim=0)


def trust_score(residual, uncertainty, scale, config):
    if (not torch.isfinite(residual).all() or not torch.isfinite(uncertainty).all()
            or not math.isfinite(float(scale)) or float(scale) <= 0):
        raise ValueError("CRO2O audit: non-finite residual/uncertainty or invalid scale")
    suspicion = ((residual.abs() - config.candidate_uncertainty_allowance * uncertainty)
                 / (scale + 1e-8) - config.candidate_residual_threshold).clamp_min(0)
    reliable = (-suspicion / config.candidate_trust_temperature).exp()
    weight = config.candidate_trust_min + (1 - config.candidate_trust_min) * reliable
    return suspicion.detach(), weight.detach()


def fixed_mask(seed, source, block, heads, probability):
    # No process-randomized hash and no replay-time resampling.
    rng = np.random.default_rng(np.random.SeedSequence([seed, source, int(block)]))
    return (rng.random(heads) < probability).astype(np.float32)


class SourceAudit:
    def __init__(self, config):
        self.config = config
        self.residuals = deque(maxlen=config.candidate_scale_window)
        self.blocks = {}

    def scale(self):
        if not self.residuals:
            return self.config.candidate_scale_floor
        values = np.asarray(self.residuals)
        mad = self.config.candidate_mad_multiplier * np.median(np.abs(values - np.median(values)))
        return max(float(mad), self.config.candidate_scale_floor)

    def record(self, block, residual, suspicion, weight):
        self.residuals.append(float(residual))
        row = self.blocks.setdefault(int(block), [0., 0., 0])
        row[0] += 1 - math.exp(-float(suspicion) / self.config.candidate_trust_temperature)
        row[1] += float(weight)
        row[2] += 1

    def diagnostics(self):
        if not self.blocks:
            return dict(count=0, suspicion=0., effective_count=0., scale=self.scale())
        values = np.asarray(list(self.blocks.values()))
        weights = values[:, 1] / values[:, 2]
        return dict(count=len(values), suspicion=float((values[:, 0] / values[:, 2]).mean()),
                    effective_count=float(weights.sum()**2 / np.square(weights).sum()), scale=self.scale())

    def state_dict(self):
        return {"residuals": list(self.residuals), "blocks": copy.deepcopy(self.blocks)}

    def load_state_dict(self, state):
        self.residuals = deque(state["residuals"], maxlen=self.config.candidate_scale_window)
        self.blocks = copy.deepcopy(state["blocks"])


def retention_target(offline, online, bias_scale):
    if not online["count"]:
        return 1.
    if not offline["count"]:
        return 0.
    bo, bn = bias_scale * offline["suspicion"], bias_scale * online["suspicion"]
    vo = offline["scale"]**2 / max(1., offline["effective_count"])
    vn = online["scale"]**2 / max(1., online["effective_count"])
    return float(np.clip((vn - bn * (bo - bn)) / ((bo - bn)**2 + vo + vn + 1e-8), 0., 1.))


class DiffusionProposal(nn.Module):
    """Conditional epsilon-prediction DDPM; no log-density API."""
    def __init__(self, state_dim, action_dim, config, max_action):
        super().__init__()
        self.steps = config.candidate_diffusion_steps
        self.action_dim, self.max_action = action_dim, max_action
        self.net = mlp(state_dim + action_dim + 1, config.hidden_dim, config.hidden_layers, action_dim)
        # Cosine cumulative alpha schedule, beta bounded away from singularity.
        t = torch.linspace(0, 1, self.steps + 1)
        cumulative = torch.cos((t + .008) / 1.008 * math.pi / 2).square()
        cumulative = cumulative / cumulative[0]
        beta = (1 - cumulative[1:] / cumulative[:-1]).clamp(1e-5, .999)
        self.register_buffer("beta", beta)
        self.register_buffer("abar", (1 - beta).cumprod(0))

    def epsilon(self, states, actions, time):
        return self.net(torch.cat((states, actions, time[:, None] / (self.steps - 1)), -1))

    def denoising_loss(self, states, actions):
        time = torch.randint(self.steps, (len(states),), device=states.device)
        noise = torch.randn_like(actions)
        abar = self.abar[time, None]
        # Poisoned labels are not clipped; bounds apply to generated actions.
        noisy = abar.sqrt() * (actions / self.max_action) + (1 - abar).sqrt() * noise
        return (noise - self.epsilon(states, noisy, time)).square().sum(-1)

    @torch.no_grad()
    def sample(self, states):
        action = torch.randn(len(states), self.action_dim, device=states.device)
        for step in reversed(range(self.steps)):
            time = torch.full((len(states),), step, device=states.device)
            epsilon = self.epsilon(states, action, time)
            x0 = ((action - (1 - self.abar[step]).sqrt() * epsilon)
                  / self.abar[step].sqrt()).clamp(-1, 1)
            previous = self.abar[step - 1] if step else self.abar.new_tensor(1.)
            denominator = 1 - self.abar[step]
            mean = (previous.sqrt() * self.beta[step] / denominator * x0
                    + (1 - self.beta[step]).sqrt() * (1 - previous) / denominator * action)
            variance = self.beta[step] * (1 - previous) / denominator
            action = mean + variance.sqrt() * torch.randn_like(action) if step else mean
        return action.clamp(-1, 1) * self.max_action


class CRO2OAgent(BaseAgent):
    def __init__(self, config, state_dim, action_dim, max_action, device):
        super().__init__(device)
        self.config, self.state_dim, self.action_dim = config, state_dim, action_dim
        self.max_action = max_action
        self.generative = config.algorithm == "rg_o2o"
        self.adaptive = config.algorithm == "arw_o2o"
        self.critics = nn.ModuleList([QNetwork(state_dim, action_dim, config.hidden_dim, config.hidden_layers)
                                      for _ in range(config.num_critics)])
        self.target_critics = copy.deepcopy(self.critics).requires_grad_(False)
        self.value = ValueNetwork(state_dim, config.hidden_dim, config.hidden_layers)
        self.actor = TanhGaussianPolicy(state_dim, action_dim, config.hidden_dim, config.hidden_layers, max_action)
        self.proposal = DiffusionProposal(state_dim, action_dim, config, max_action) if self.generative else None
        self.target_proposal = copy.deepcopy(self.proposal).requires_grad_(False) if self.generative else None
        self.log_alpha = nn.Parameter(torch.tensor(0.))
        self.to(device)
        self.q_optimizer = torch.optim.Adam(self.critics.parameters(), lr=config.critic_learning_rate)
        self.v_optimizer = torch.optim.Adam(self.value.parameters(), lr=config.critic_learning_rate)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config.actor_learning_rate)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=config.temperature_learning_rate)
        self.generator_optimizer = (torch.optim.Adam(self.proposal.parameters(), lr=config.actor_learning_rate)
                                    if self.generative else None)
        self.audit = [SourceAudit(config), SourceAudit(config)]
        self.offline_weights = self.offline_blocks = None
        self.offline_identity = None
        self.online_metadata = {}
        self.collection_steps = self.online_block = self.recalibration_updates = 0
        self.generator_updates = self.audit_updates = 0
        self.omega = (
            1. if self.adaptive and config.candidate_retention_mode == "adaptive"
            else 0. if self.adaptive and config.candidate_retention_mode == "none"
            else config.effective_offline_ratio
        )
        self.prepared = False

    def q(self, states, actions, target=False):
        return torch.stack([head(states, actions) for head in (self.target_critics if target else self.critics)])

    def _step(self, loss, optimizer, parameters):
        if not torch.isfinite(loss):
            raise ValueError("CRO2O: non-finite training loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        parameters = list(parameters)
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in parameters):
            raise ValueError("CRO2O: non-finite gradient")
        if self.config.max_grad_norm is not None:
            nn.utils.clip_grad_norm_(parameters, self.config.max_grad_norm)
        optimizer.step()

    def _metadata(self, batch, *, offline_training=False):
        assert_no_corruption_labels(batch)
        if "_indices" not in batch or "_source" not in batch:
            raise ValueError("CRO2O requires stable replay indices/source, not resampled bootstrap masks")
        weights, masks = [], []
        for source, index in zip(batch["_source"].long().tolist(), batch["_indices"].long().tolist()):
            if source == 0:
                if self.offline_blocks is None:
                    raise ValueError("CRO2O offline blocks are not registered")
                block = int(self.offline_blocks[index])
                weight = 1. if offline_training else float(self.offline_weights[index])
            elif source == 1:
                weight, block = self.online_metadata[index]
            else:
                raise ValueError("CRO2O unknown replay source")
            weights.append(weight)
            masks.append(fixed_mask(self.config.learner_seed, source, block, self.config.num_critics,
                                    self.config.candidate_bootstrap_probability))
        return (torch.tensor(weights, device=self.device),
                torch.as_tensor(np.stack(masks).T, device=self.device))

    def update(self, batch):
        if self.online_phase:
            return self.update_sources(batch, None)
        weights, masks = self._metadata(batch, offline_training=True)
        states, actions = batch["observations"], batch["actions"]
        with torch.no_grad():
            target_q = torch.quantile(self.q(states, actions, True), self.config.riql_quantile, dim=0)
            target = batch["rewards"] + self.config.discount * (1 - batch["terminals"]) * self.value(batch["next_observations"])
        diff = target_q - self.value(states)
        v_loss = (torch.where(diff < 0, 1 - self.config.expectile, self.config.expectile) * diff.square()).mean()
        self._step(v_loss, self.v_optimizer, self.value.parameters())
        q_loss = weighted_mean(huber(self.q(states, actions) - target, 1 / self.config.riql_sigma**2), masks).sum()
        self._step(q_loss, self.q_optimizer, self.critics.parameters())
        with torch.no_grad():
            advantage = target_q - self.value(states)
            quality = (self.config.beta * advantage).clamp(max=self.config.candidate_advantage_clip).exp()
        actor_loss = -(quality * self.actor.log_prob(states, actions)).mean()
        self._step(actor_loss, self.actor_optimizer, self.actor.parameters())
        soft_update(self.target_critics, self.critics, self.config.target_update_rate)
        self.total_updates += 1
        self.critic_updates += 1
        self.actor_updates += 1
        return dict(critic_loss=q_loss.item(), value_loss=v_loss.item(), actor_loss=actor_loss.item())

    @torch.no_grad()
    def candidates(self, states, *, target=False, optimistic=False, checking=False):
        count = self.config.candidate_candidates
        expanded = states[:, None].expand(-1, count, -1).reshape(-1, self.state_dim)
        if self.generative:
            actions = (self.target_proposal if target else self.proposal).sample(expanded)
        else:
            actions = self.actor(expanded)[0]
        q, uncertainty = center_uncertainty(self.q(expanded, actions, target), self.config.candidate_mad_multiplier)
        score = q + (self.config.candidate_optimism * uncertainty if optimistic else 0.)
        indices = score.reshape(len(states), count).argmax(-1)
        if checking:
            direct = torch.rand(len(states), device=self.device) < self.config.candidate_check_probability
            indices = torch.where(direct, torch.randint(count, indices.shape, device=self.device), indices)
        return actions.reshape(len(states), count, self.action_dim)[torch.arange(len(states), device=self.device), indices]

    @torch.no_grad()
    def select_action(self, state, evaluate=False, evaluation_mode="deterministic"):
        single = state.ndim == 1
        states = state[None] if single else state
        warming = self.collection_steps < max(self.config.initial_collection_steps, self.config.warmup_steps)
        if self.generative and self.prepared:
            action = self.candidates(states, optimistic=self.online_phase and not evaluate and not warming,
                                     checking=self.online_phase and not evaluate and not warming)
        elif evaluate:
            action = self.actor(states, deterministic=True)[0]
        elif not self.online_phase or warming:
            action = self.actor(states)[0]
        else:
            action = self.candidates(states, optimistic=True, checking=True)
        return action[0] if single else action

    @torch.no_grad()
    def _continuation(self, next_states, *, audit=False):
        if self.generative:
            action = self.candidates(next_states, target=True)
            return self.q(next_states, action, True)
        if audit:
            action, logp, _, _ = self.actor(next_states, need_log_prob=True)
            return self.q(next_states, action, True) - self.log_alpha.exp() * logp
        # Equation (2.3) samples per-head targets; equation (2.2) above uses a
        # common action so audit dispersion does not include proposal noise.
        values = []
        for head in self.target_critics:
            action, logp, _, _ = self.actor(next_states, need_log_prob=True)
            values.append(head(next_states, action) - self.log_alpha.exp() * logp)
        return torch.stack(values)

    @torch.no_grad()
    def observe_transition(self, state, action, reward, next_state, terminal, slot, episode_finished):
        if not self.prepared:
            raise ValueError("CRO2O must construct honest offline audit scores before collection")
        s = torch.as_tensor(state, dtype=torch.float32, device=self.device)[None]
        a = torch.as_tensor(action, dtype=torch.float32, device=self.device)[None]
        ns = torch.as_tensor(next_state, dtype=torch.float32, device=self.device)[None]
        implied = self.q(s, a) - self.config.discount * (1 - terminal) * self._continuation(ns, audit=True)
        center, uncertainty = center_uncertainty(implied, self.config.candidate_mad_multiplier)
        residual = torch.as_tensor(reward, device=self.device) - center
        suspicion, weight = trust_score(residual, uncertainty, self.audit[1].scale(), self.config)
        self.online_metadata[int(slot)] = (float(weight.item()), self.online_block)
        self.audit[1].record(self.online_block, residual.item(), suspicion.item(), weight.item())
        self.collection_steps += 1
        if episode_finished:
            self.online_block += 1
        if (self.adaptive and self.config.candidate_retention_mode == "adaptive"
                and self.collection_steps % self.config.candidate_retention_period == 0):
            target = retention_target(*(a.diagnostics() for a in self.audit), self.config.candidate_bias_scale)
            smoothing = self.config.candidate_retention_smoothing
            self.omega = (1 - smoothing) * self.omega + smoothing * target

    def _critic_loss(self, batch):
        weights, masks = self._metadata(batch)
        with torch.no_grad():
            target = batch["rewards"] + self.config.discount * (1 - batch["terminals"]) * self._continuation(batch["next_observations"])
        residual = self.q(batch["observations"], batch["actions"]) - target
        return weighted_mean(huber(residual, 1 / self.config.riql_sigma**2), masks * weights).sum()

    def generator_loss(self, batch, offline=False):
        weights, _ = self._metadata(batch)
        states, actions = batch["observations"], batch["actions"]
        with torch.no_grad():
            if offline:
                quality = torch.quantile(self.q(states, actions, True), self.config.riql_quantile, dim=0)
                baseline = self.value(states)
            else:
                quality, _ = center_uncertainty(self.q(states, actions), self.config.candidate_mad_multiplier)
                expanded = states[:, None].expand(-1, self.config.candidate_candidates, -1).reshape(-1, self.state_dim)
                proposals = self.proposal.sample(expanded)
                values, _ = center_uncertainty(self.q(expanded, proposals), self.config.candidate_mad_multiplier)
                baseline = values.reshape(len(states), -1).mean(-1)
            advantage = (self.config.beta * (quality - baseline)).clamp(-self.config.candidate_advantage_clip, self.config.candidate_advantage_clip)
            influence = weights * advantage.exp()
        return weighted_mean(self.proposal.denoising_loss(states, actions), influence)

    def update_sources(self, offline_batch, online_batch, critic_only=False):
        # ARW uses independent source normalization, never a jointly weighted batch.
        batches = [(offline_batch, self.omega), (online_batch, 1 - self.omega)] if online_batch is not None else [(offline_batch, 1.)]
        q_loss = sum(ratio * self._critic_loss(batch) for batch, ratio in batches if ratio > 0)
        self._step(q_loss, self.q_optimizer, self.critics.parameters())
        soft_update(self.target_critics, self.critics, self.config.target_update_rate)
        metrics = dict(critic_loss=q_loss.item(), offline_retention=self.omega)
        if not critic_only:
            if self.generative:
                actor_loss = sum(ratio * self.generator_loss(batch) for batch, ratio in batches if ratio > 0)
                self._step(actor_loss, self.generator_optimizer, self.proposal.parameters())
                soft_update(self.target_proposal, self.proposal, self.config.target_update_rate)
                self.generator_updates += 1
            else:
                actor_loss, alpha_loss = 0., 0.
                for batch, ratio in batches:
                    if not ratio:
                        continue
                    states = batch["observations"]
                    actions, logp, _, _ = self.actor(states, need_log_prob=True)
                    center, _ = center_uncertainty(self.q(states, actions), self.config.candidate_mad_multiplier)
                    actor_loss = actor_loss + ratio * (self.log_alpha.exp().detach() * logp - center).mean()
                    alpha_loss = alpha_loss - ratio * (self.log_alpha * (logp - self.action_dim).detach()).mean()
                self._step(actor_loss, self.actor_optimizer, self.actor.parameters())
                self._step(alpha_loss, self.alpha_optimizer, [self.log_alpha])
                self.temperature_updates += 1
            self.actor_updates += 1
            metrics["actor_loss"] = actor_loss.item()
        else:
            self.recalibration_updates += 1
        self.critic_updates += 1
        self.total_updates += 1
        return metrics

    def begin_online(self):
        if self.online_phase:
            return
        if not self.prepared:
            raise ValueError("CRO2O online phase requires offline cross-fit audit setup")
        self.online_phase = True

    def algorithm_metadata(self):
        return dict(candidate_version=VERSION, candidate_audit_updates=self.audit_updates,
                    candidate_audit_optimizer_steps=3 * self.audit_updates,
                    candidate_generator_updates=self.generator_updates,
                    candidate_recalibration_updates=self.recalibration_updates,
                    candidate_collected_transitions=self.collection_steps,
                    candidate_offline_retention=self.omega,
                    candidate_offline_distinct_blocks=len(self.audit[0].blocks),
                    candidate_online_distinct_blocks=len(self.audit[1].blocks),
                    candidate_evaluation=("stochastic_diffusion_exploitation_reranking" if self.generative else "deterministic_base_actor"))

    def optimizer_state(self):
        return {name: getattr(self, name).state_dict() for name in
                ("q_optimizer", "v_optimizer", "actor_optimizer", "alpha_optimizer", "generator_optimizer")
                if getattr(self, name) is not None}

    def checkpoint_state(self):
        result = super().checkpoint_state()
        result["candidate"] = {
            "version": VERSION, "offline_identity": self.offline_identity,
            "offline_blocks": self.offline_blocks, "offline_weights": self.offline_weights,
            "online_metadata": copy.deepcopy(self.online_metadata),
            "audit": [a.state_dict() for a in self.audit],
            **{k: getattr(self, k) for k in ("prepared", "collection_steps", "online_block", "omega", "recalibration_updates", "generator_updates", "audit_updates")},
        }
        return result

    def load_checkpoint_state(self, state):
        extra = state.get("candidate", {})
        if extra.get("version") != VERSION:
            raise ValueError("CRO2O checkpoint lacks compatible audit/bootstrap state")
        self.load_state_dict(state["model"], strict=True)
        for name, value in state["optimizers"].items():
            getattr(self, name).load_state_dict(value)
        for name in ("online_phase", "total_updates", "actor_updates", "critic_updates", "temperature_updates"):
            setattr(self, name, state[name])
        for name, value in extra.items():
            if name not in ("version", "audit"):
                setattr(self, name, copy.deepcopy(value))
        for audit, saved in zip(self.audit, extra["audit"]):
            audit.load_state_dict(saved)


def register_offline_blocks(agent, dataset):
    # Episode IDs are trajectory bookkeeping, not attack labels.
    if "episode_id" not in dataset:
        raise ValueError("CRO2O requires episode_id to cross-fit complete trajectory blocks")
    blocks = np.asarray(dataset["episode_id"], dtype=np.int64)
    digest = hashlib.sha256()
    for name in ("observations", "actions", "rewards", "next_observations", "terminals", "episode_id"):
        digest.update(np.ascontiguousarray(dataset[name]).tobytes())
    identity = digest.hexdigest()
    if agent.offline_identity is not None and agent.offline_identity != identity:
        raise ValueError("CRO2O checkpoint offline artifact differs from this run")
    agent.offline_identity = identity
    agent.offline_blocks = blocks.copy()
