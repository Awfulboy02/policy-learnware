"""Offline value estimates for a frozen policy on finite-horizon logged episodes."""

from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def _transitions(episodes, horizon, device):
    def tensor(rows):
        return torch.as_tensor(np.concatenate(rows), dtype=torch.float32, device=device)

    def field(name):
        return tensor([getattr(e, name)[:horizon] for e in episodes])

    obs = field("obs")
    actions = field("actions")
    rewards = field("rewards").flatten()
    next_obs = field("next_obs")
    terminal = field("terminated").flatten().bool()
    lengths = [min(len(e.rewards), horizon) for e in episodes]
    time = torch.as_tensor(np.concatenate([np.arange(n) for n in lengths]), device=device)
    valid = tensor([np.ones(n) if e.next_valid is None else e.next_valid[:n]
                    for e, n in zip(episodes, lengths)]).flatten().bool()
    if (~valid & ~terminal & (time + 1 < horizon)).any():
        raise ValueError("an unfinished transition requires its next observation")
    return obs, actions, rewards, next_obs, terminal, time, valid


def _scale(values):
    std = values.std(dim=0, unbiased=False)
    return torch.where(std > 1e-5, std, torch.ones_like(std))


def _network(inputs, outputs, seed, device):
    # Initialize on CPU without resetting the caller's random stream.
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(seed)
        net = nn.Sequential(nn.Linear(inputs, 64), nn.SiLU(), nn.Linear(64, 64),
                            nn.SiLU(), nn.Linear(64, outputs))
        nn.init.zeros_(net[-1].weight)
        nn.init.zeros_(net[-1].bias)
    return net.to(device)


def _actions(policy, obs):
    actions = np.asarray(policy(obs.detach().cpu().numpy()), dtype=np.float32)
    return torch.as_tensor(actions, device=obs.device).reshape(len(obs), -1)


def fit_fqe(episodes, policy, horizon, gamma, *, steps=4000, seed=0, device="cpu"):
    """Fit time-conditioned Q using the candidate's actions and logged rewards."""
    obs, actions, rewards, next_obs, terminal, time, _ = _transitions(episodes, horizon, device)
    inputs = torch.cat((obs, actions), dim=1)
    center, scale = inputs.mean(0), _scale(inputs)
    inputs = (inputs - center) / scale
    continuing = ~terminal & (time + 1 < horizon)

    # A constant per-step reward is a convenient finite-horizon Q initialization.
    mass = np.r_[0., np.cumsum(gamma ** np.arange(horizon))]
    baseline = rewards.mean() * torch.as_tensor(mass, dtype=obs.dtype, device=device)
    reward_scale = rewards.square().mean().sqrt().clamp_min(1e-5)
    net = _network(inputs.shape[1] + 1, 1, seed, device)
    target_net = deepcopy(net)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    rng = np.random.default_rng(seed)

    def q(model, x, t):
        features = torch.cat([x, (t / horizon).unsqueeze(1)], dim=1)
        return baseline[horizon - t] + reward_scale * model(features).flatten()

    for step in range(steps):
        i = torch.as_tensor(rng.integers(len(obs), size=min(128, len(obs))), device=device)
        with torch.no_grad():
            target = rewards[i].clone()
            mask = continuing[i]
            if mask.any():
                nxt = next_obs[i[mask]]
                x = (torch.cat((nxt, _actions(policy, nxt)), dim=1) - center) / scale
                target[mask] += gamma * q(target_net, x, time[i[mask]] + 1)
        loss = ((q(net, inputs[i], time[i]) - target) / reward_scale).square().mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (step + 1) % 10 == 0:
            target_net.load_state_dict(net.state_dict())

    initial = torch.as_tensor(np.stack([e.obs[0] for e in episodes]),
                              dtype=obs.dtype, device=device)
    with torch.no_grad():
        x = (torch.cat((initial, _actions(policy, initial)), dim=1) - center) / scale
        return q(net, x, torch.zeros(len(initial), dtype=torch.long, device=device)).mean().item()


@dataclass
class ApproximateModel:
    net: nn.Module
    center: torch.Tensor
    scale: torch.Tensor
    output_center: torch.Tensor
    output_scale: torch.Tensor
    has_terminal: bool

    @torch.no_grad()
    def value(self, policy, initial_states, horizon, gamma):
        """Roll out mean learned dynamics, weighting future rewards by survival."""
        state = torch.as_tensor(initial_states, dtype=self.center.dtype, device=self.center.device)
        weight = torch.ones(len(state), device=state.device)
        total = torch.zeros_like(weight)
        size = state.shape[1]
        for _ in range(horizon):
            x = torch.cat((state, _actions(policy, state)), dim=1)
            pred = self.net((x - self.center) / self.scale)
            output = pred[:, :size + 1] * self.output_scale + self.output_center
            total += weight * output[:, size]
            weight = gamma * weight
            if self.has_terminal:
                weight *= torch.sigmoid(-pred[:, -1])
            state = state + output[:, :size]
        return total.mean().item()


def fit_am(episodes, horizon, *, steps=4000, seed=0, device="cpu"):
    """Fit one model of mean next-state deltas, rewards, and natural termination."""
    obs, actions, rewards, nxt, terminal, _, valid = _transitions(episodes, horizon, device)
    inputs = torch.cat((obs, actions), dim=1)
    center, scale = inputs.mean(0), _scale(inputs)
    inputs = (inputs - center) / scale
    delta = nxt - obs
    # A terminal transition without its successor still supplies a reward label.
    if valid.any():
        delta_center, delta_scale = delta[valid].mean(0), _scale(delta[valid])
    else:
        delta_center = torch.zeros_like(obs[0])
        delta_scale = torch.ones_like(obs[0])
    output_center = torch.cat((delta_center, rewards.mean().view(1)))
    output_scale = torch.cat((delta_scale, _scale(rewards).view(1)))
    y = (torch.cat((delta, rewards.unsqueeze(1)), dim=1) - output_center) / output_scale
    has_terminal = bool(terminal.any())
    net = _network(inputs.shape[1], y.shape[1] + int(has_terminal), seed, device)
    if has_terminal:
        with torch.no_grad():
            net[-1].bias[-1] = torch.logit(terminal.float().mean().clamp(1e-4, 1 - 1e-4))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    rng = np.random.default_rng(seed)

    for _ in range(steps):
        i = torch.as_tensor(rng.integers(len(obs), size=min(128, len(obs))), device=device)
        pred = net(inputs[i])
        size = obs.shape[1]
        loss = F.mse_loss(pred[:, size], y[i, size])
        mask = valid[i]
        if mask.any():
            loss = loss + F.mse_loss(pred[mask, :size], y[i[mask], :size])
        if has_terminal:
            loss = loss + F.binary_cross_entropy_with_logits(pred[:, -1], terminal[i].float())
        opt.zero_grad()
        loss.backward()
        opt.step()
    return ApproximateModel(net.eval(), center, scale, output_center, output_scale, has_terminal)
