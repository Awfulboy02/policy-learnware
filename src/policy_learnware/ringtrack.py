"""Small deterministic offline fixture with exactly enumerable returns."""
from pathlib import Path
import numpy as np

from .data import write_json

HORIZON = 52
STARTS = {"left": (0, 1), "right": (50, 51)}


def policy(spec):
    def act(obs):
        if spec["action"] == "expert":
            return (obs[:, :1] >= 50).astype(float)
        return np.full((len(obs), 1), spec["action"], dtype=float)
    return act


def _episode(path, start, act):
    obs = np.empty((HORIZON + 1, 2))
    acts = np.empty((HORIZON, 1))
    rewards = np.empty(HORIZON)
    obs[0] = (start, 0)
    for t in range(HORIZON):
        pos = int(obs[t, 0])
        acts[t] = act(obs[t:t+1])[0]
        forward = int(acts[t, 0]) == pos // 50
        rewards[t] = float(forward) / HORIZON
        obs[t + 1] = ((pos + (1 if forward else -1)) % 100, t + 1)
    end = np.zeros(HORIZON, dtype=bool)
    end[-1] = True
    np.savez(path, observations=obs[:-1], actions=acts, rewards=rewards,
             next_observations=obs[1:], terminated=np.zeros_like(end), truncated=end)
    return str(path)


def prepare(output):
    """Enumerate source, user and judge separately. No environment is used by ranking."""
    out = Path(output).resolve()
    logs = out / "ring_episodes"
    logs.mkdir(parents=True, exist_ok=True)
    specs = {side: {"kind": "ring", "action": index} for index, side in enumerate(STARTS)}
    expert = policy({"action": "expert"})
    sources, requests, truth = {}, {}, {}
    for side, starts in STARTS.items():
        spec = specs[side]
        src = [_episode(logs / f"src-{side}-{s}.npz", s, policy(spec)) for s in starts]
        sources[side] = {"policy": spec, "episodes": src}
        requests[side] = {"split": "development", "episodes": [
            _episode(logs / f"user-{side}-{s}.npz", s, expert) for s in starts]}
        truth[side] = {}
        for name, act in {**{i: policy(p) for i, p in specs.items()}, "__expert__": expert}.items():
            truth[side][name] = [
                _episode(logs / f"judge-{side}-{name}-{s}.npz", s, act) for s in starts]
    write_json(out / "source.json", {"policies": sources})
    write_json(out / "requests.json", {"requests": requests})
    write_json(out / "truth.json", {"requests": truth})
