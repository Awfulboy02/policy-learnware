"""Online SAC with automatic temperature, and PPO-Clip for bounded actions."""
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .expansion_offpolicy import Replay


def mlp(inputs, width, outputs, activation):
    return nn.Sequential(nn.Linear(inputs, width), activation(), nn.Linear(width, width),
                         activation(), nn.Linear(width, outputs))


class GaussianActor(nn.Module):
    """Sample during training; forward is the deterministic frozen controller."""

    def __init__(self, obs_size, width, actions, state_dependent_std):
        super().__init__()
        self.state_dependent_std = state_dependent_std
        self.actions = actions
        activation = nn.ReLU if state_dependent_std else nn.Tanh
        self.net = mlp(obs_size, width, 2 * actions if state_dependent_std else actions,
                       activation)
        self.log_std = None if state_dependent_std else nn.Parameter(torch.zeros(actions))

    def forward(self, states):
        return self.net(states)[..., :self.actions].tanh()

    def distribution(self, states):
        outputs = self.net(states)
        if self.state_dependent_std:
            mean, log_std = outputs.chunk(2, dim=-1)
        else:
            mean, log_std = outputs, self.log_std
        return torch.distributions.Normal(mean, log_std.clamp(-20, 2).exp())


def soft_action(actor, states):
    dist = actor.distribution(states)
    raw = dist.rsample()
    # Stable log(1 - tanh(raw)**2), including saturated actions.
    correction = 2 * (math.log(2) - raw - F.softplus(-2 * raw))
    return raw.tanh(), (dist.log_prob(raw) - correction).sum(dim=-1)


def train(args, env, on_slice):
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    seeds = [args.seed + i for i in range(args.num_envs)]
    states = env.reset(seeds)
    if args.algorithm == "ppo":
        return _ppo(args, env, states, on_slice)
    return _sac(args, env, states, rng, on_slice)


def _sac(args, env, states, rng, on_slice):
    from copy import deepcopy

    device, actions = args.device, env.action_size
    obs_size = states.shape[-1]
    actor = GaussianActor(obs_size, args.width, actions, True).to(device)
    critics = nn.ModuleList(mlp(obs_size + actions, args.width, 1, nn.ReLU) for _ in range(2))
    critics.to(device)
    target = deepcopy(critics).requires_grad_(False)
    actor_opt = torch.optim.Adam(actor.parameters(), lr=args.learning_rate)
    critic_opt = torch.optim.Adam(critics.parameters(), lr=args.learning_rate)
    log_alpha = torch.tensor(0.0, device=device, requires_grad=True)
    alpha_opt = torch.optim.Adam([log_alpha], lr=args.learning_rate)
    replay = Replay(min(1_000_000, max(args.steps[-1], args.num_envs)), obs_size, actions)
    nenv, updates = args.num_envs, 0
    cuts = set(args.steps)
    if any(cut % nenv for cut in cuts):
        raise ValueError("SAC budget points must be multiples of num_envs")
    for step in range(nenv, args.steps[-1] + 1, nenv):
        if step - nenv < args.warmup:
            action = rng.uniform(-1, 1, (nenv, actions)).astype(np.float32)
        else:
            with torch.no_grad():
                dist = actor.distribution(torch.as_tensor(states, device=device))
                action = dist.sample().tanh().cpu().numpy()
        nxt, reward, terminated, truncated = env.step(action)
        replay.add(states, action, reward, nxt, 0.99 * (1 - np.asarray(terminated, dtype=float)))
        done = np.logical_or(terminated, truncated)
        states = env.reset_done(nxt, done) if done.any() else nxt
        if step >= args.warmup and len(replay) >= args.batch_size:
            s, a, r, s2, discount = replay.sample(args.batch_size, rng, device)
            with torch.no_grad():
                a2, logp2 = soft_action(actor, s2)
                sa2 = torch.cat((s2, a2), dim=-1)
                q2 = torch.minimum(target[0](sa2), target[1](sa2)).squeeze(-1)
                backup = r + discount * (q2 - log_alpha.exp() * logp2)
            sa = torch.cat((s, a), dim=-1)
            value_loss = sum(F.mse_loss(q(sa).squeeze(-1), backup) for q in critics)
            critic_opt.zero_grad()
            value_loss.backward()
            critic_opt.step()

            critics.requires_grad_(False)
            sampled, logp = soft_action(actor, s)
            sampled_sa = torch.cat((s, sampled), dim=-1)
            q = torch.minimum(critics[0](sampled_sa), critics[1](sampled_sa)).squeeze(-1)
            actor_loss = (log_alpha.detach().exp() * logp - q).mean()
            actor_opt.zero_grad()
            actor_loss.backward()
            actor_opt.step()
            critics.requires_grad_(True)

            # Target entropy = -action dimension.
            alpha_opt.zero_grad()
            (-(log_alpha * (logp.detach() - actions)).mean()).backward()
            alpha_opt.step()
            with torch.no_grad():
                for dst, src in zip(target.parameters(), critics.parameters()):
                    dst.lerp_(src, 0.005)
            updates += 1
        if step in cuts:
            on_slice(step, actor)
    return {"updates": updates}


def _ppo(args, env, states, on_slice):
    device = args.device
    nenv, length = args.num_envs, args.rollout_steps
    stride = nenv * length
    cuts = {((step + stride - 1) // stride) * stride for step in args.steps}
    actor = GaussianActor(states.shape[-1], args.width, env.action_size, False).to(device)
    value = mlp(states.shape[-1], args.width, 1, nn.Tanh).to(device)
    # Small initial action means and an unscaled linear value head.
    for net, final_gain in ((actor.net, 0.01), (value, 1.0)):
        for layer in net:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=final_gain if layer is net[-1] else 2**0.5)
                nn.init.zeros_(layer.bias)
    params = list(actor.parameters()) + list(value.parameters())
    optimizer = torch.optim.Adam(params, lr=args.learning_rate, eps=1e-5)
    updates = 0
    for step in range(stride, max(cuts) + 1, stride):
        visits, latents, old_logs, values, residuals, continuations = [], [], [], [], [], []
        with torch.no_grad():
            for _ in range(length):
                s = torch.as_tensor(states, device=device)
                dist = actor.distribution(s)
                raw = dist.sample()
                v = value(s).squeeze(-1)
                nxt, reward, terminated, truncated = env.step(raw.tanh().cpu().numpy())
                done = np.logical_or(terminated, truncated)
                v2 = value(torch.as_tensor(nxt, device=device)).squeeze(-1)
                bootstrap = torch.as_tensor(np.logical_not(terminated), device=device)
                r = torch.as_tensor(reward, device=device, dtype=v.dtype)
                residuals.append(r + 0.99 * bootstrap * v2 - v)
                continuations.append(torch.as_tensor(~done, device=device))
                visits.append(s)
                latents.append(raw)
                # For tanh actions its fixed Jacobian cancels in the PPO likelihood ratio.
                old_logs.append(dist.log_prob(raw).sum(-1))
                values.append(v)
                states = env.reset_done(nxt, done) if done.any() else nxt
        gae = torch.zeros(nenv, device=device)
        advantages = []
        for delta, keep in zip(reversed(residuals), reversed(continuations)):
            gae = delta + 0.99 * 0.95 * keep * gae
            advantages.append(gae)
        adv = torch.cat(advantages[::-1])
        returns = adv + torch.cat(values)
        adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)
        x, z, old = torch.cat(visits), torch.cat(latents), torch.cat(old_logs)
        for _ in range(4):
            for idx in torch.randperm(stride, device=device).split(args.batch_size):
                logp = actor.distribution(x[idx]).log_prob(z[idx]).sum(-1)
                ratio = (logp - old[idx]).exp()
                objective = torch.minimum(ratio * adv[idx], ratio.clamp(0.8, 1.2) * adv[idx])
                prediction = value(x[idx]).squeeze(-1)
                loss = -objective.mean() + 0.5 * F.mse_loss(prediction, returns[idx])
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(params, 0.5)
                optimizer.step()
                updates += 1
        if step in cuts:
            on_slice(step, actor)
    return {"updates": updates}
