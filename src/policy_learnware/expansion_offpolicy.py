"""Online TD3 and D4PG for bounded, continuous DMC actions.

D4PG uses synchronous vector actors, a five-step categorical critic and uniform
replay. Its 51 atoms cover [0, 100] for DMC rewards in [0, 1] at gamma=0.99.
Training bootstraps time limits from the last observation before reset.
"""

from collections import deque
from copy import deepcopy

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class Replay:
    """Uniform replay with explicit bootstrap discounts for one- or n-step data."""

    def __init__(self, capacity, obs_size, action_size):
        self.capacity = capacity
        self.size = self.cursor = 0
        self.columns = (
            np.empty((capacity, obs_size), np.float32),
            np.empty((capacity, action_size), np.float32),
            np.empty(capacity, np.float32),
            np.empty((capacity, obs_size), np.float32),
            np.empty(capacity, np.float32),
        )

    def __len__(self):
        return self.size

    def add(self, obs, actions, rewards, next_obs, discounts):
        count = len(obs)
        indices = (self.cursor + np.arange(count)) % self.capacity
        for column, values in zip(self.columns, (obs, actions, rewards, next_obs, discounts)):
            column[indices] = values
        self.cursor = (self.cursor + count) % self.capacity
        self.size = min(self.size + count, self.capacity)

    def sample(self, batch_size, rng, device):
        indices = rng.integers(self.size, size=batch_size)
        return tuple(torch.as_tensor(column[indices], device=device) for column in self.columns)


def _mlp(input_size, width, output_size):
    return nn.Sequential(
        nn.Linear(input_size, width), nn.ReLU(), nn.Linear(width, width), nn.ReLU(),
        nn.Linear(width, output_size),
    )


class FiveStep:
    """Accumulate per-environment returns, flushing short tails at episode ends."""

    def __init__(self, num_envs, gamma):
        self.queues = [deque() for _ in range(num_envs)]
        self.gamma = gamma

    def add(self, replay, obs, actions, rewards, next_obs, terminated, truncated):
        ready = []
        for i, queue in enumerate(self.queues):
            queue.append((obs[i].copy(), actions[i].copy(), float(rewards[i])))
            done = terminated[i] or truncated[i]
            if len(queue) < 5 and not done:
                continue
            # For a full window and for all tails at a boundary, the endpoint
            # is the newest transition. A true terminal suppresses bootstrap.
            while queue:
                window_return = sum(self.gamma ** j * row[2] for j, row in enumerate(queue))
                discount = 0.0 if terminated[i] else self.gamma ** len(queue)
                first_obs, first_action, _ = queue.popleft()
                ready.append((first_obs, first_action, window_return, next_obs[i], discount))
                if not done:
                    break
        if ready:
            replay.add(*(np.asarray(column) for column in zip(*ready)))


def _project(probabilities, rewards, discounts, atoms):
    """Project r + discount * Z onto evenly spaced categorical support."""
    shifted = rewards[:, None] + discounts[:, None] * atoms
    bins = ((shifted - atoms[0]) / (atoms[1] - atoms[0])).clamp(0, len(atoms) - 1)
    left = bins.long()
    right = (left + 1).clamp(max=len(atoms) - 1)
    frac = bins - left
    targets = torch.zeros_like(probabilities)
    targets.scatter_add_(1, left, probabilities * (1.0 - frac))
    targets.scatter_add_(1, right, probabilities * frac)
    return targets


def _soft_update(target, source, rate=0.005):
    with torch.no_grad():
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.lerp_(param, rate)


class _Learner:
    def __init__(self, args, obs_size, action_size):
        self.td3 = args.algorithm == "td3"
        self.actor = nn.Sequential(_mlp(obs_size, args.width, action_size), nn.Tanh()).to(args.device)
        n_critics, output_size = (2, 1) if self.td3 else (1, 51)
        self.critics = nn.ModuleList(
            _mlp(obs_size + action_size, args.width, output_size) for _ in range(n_critics)
        ).to(args.device)
        self.target_actor = deepcopy(self.actor).requires_grad_(False)
        self.target_critics = deepcopy(self.critics).requires_grad_(False)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=args.learning_rate)
        self.critic_opt = torch.optim.Adam(self.critics.parameters(), lr=args.learning_rate)
        self.atoms = torch.linspace(0, 100, 51, device=args.device)
        self.critic_updates = self.actor_updates = 0

    def update(self, batch):
        obs, actions, rewards, next_obs, discounts = batch
        with torch.no_grad():
            next_actions = self.target_actor(next_obs)
            if self.td3:
                noise = (torch.randn_like(next_actions) * 0.2).clamp(-0.5, 0.5)
                next_actions = (next_actions + noise).clamp(-1, 1)
            target_inputs = torch.cat((next_obs, next_actions), dim=-1)
            if self.td3:
                q1, q2 = (net(target_inputs).squeeze(-1) for net in self.target_critics)
                target = rewards + discounts * torch.minimum(q1, q2)
            else:
                next_probs = self.target_critics[0](target_inputs).softmax(dim=-1)
                target = _project(next_probs, rewards, discounts, self.atoms)

        critic_inputs = torch.cat((obs, actions), dim=-1)
        if self.td3:
            loss = sum(F.mse_loss(net(critic_inputs).squeeze(-1), target) for net in self.critics)
        else:
            log_probs = self.critics[0](critic_inputs).log_softmax(dim=-1)
            loss = -(log_probs * target).sum(dim=-1).mean()
        self.critic_opt.zero_grad()
        loss.backward()
        self.critic_opt.step()
        self.critic_updates += 1

        if self.td3 and self.critic_updates % 2:
            return
        self.critics.requires_grad_(False)
        prediction = self.critics[0](torch.cat((obs, self.actor(obs)), dim=-1))
        value = prediction if self.td3 else (prediction.softmax(dim=-1) * self.atoms).sum(dim=-1)
        self.actor_opt.zero_grad()
        (-value.mean()).backward()
        self.actor_opt.step()
        self.critics.requires_grad_(True)
        self.actor_updates += 1
        _soft_update(self.target_actor, self.actor)
        _soft_update(self.target_critics, self.critics)


def train(args, env, on_slice):
    """Train from scratch, emitting the deterministic controller at each budget."""
    n = args.num_envs
    if any(step % n for step in args.steps):
        raise ValueError("TD3/D4PG budgets must be multiples of num_envs")
    budgets = set(args.steps)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    obs = env.reset(rng.integers(0, 2**31, size=n).tolist())
    learner = _Learner(args, obs.shape[-1], env.action_size)
    replay = Replay(min(args.steps[-1], 1_000_000), obs.shape[-1], env.action_size)
    gamma = 0.99
    nstep = FiveStep(n, gamma) if args.algorithm == "d4pg" else None
    # Simultaneous exploratory actors share weights but draw independent noise.
    for start in range(0, args.steps[-1], n):
        if start < args.warmup:
            actions = rng.uniform(-1, 1, size=(n, env.action_size)).astype(np.float32)
        else:
            with torch.no_grad():
                actions = learner.actor(torch.as_tensor(obs, dtype=torch.float32, device=args.device))
                actions = actions.cpu().numpy()
            actions = np.clip(actions + rng.normal(0, 0.1, actions.shape), -1, 1).astype(np.float32)
        next_obs, rewards, terminated, truncated = env.step(actions)
        if nstep is None:
            replay.add(obs, actions, rewards, next_obs, gamma * (1 - terminated.astype(np.float32)))
        else:
            nstep.add(replay, obs, actions, rewards, next_obs, terminated, truncated)
        obs = env.reset_done(next_obs, terminated | truncated)
        steps = start + n
        if steps >= args.warmup and len(replay) >= args.batch_size:
            learner.update(replay.sample(args.batch_size, rng, args.device))
        if steps in budgets:
            on_slice(steps, learner.actor)
    return {
        "env_steps": args.steps[-1], "replay_size": len(replay),
        "critic_updates": learner.critic_updates, "actor_updates": learner.actor_updates,
    }
