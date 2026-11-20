"""One file per ordered episode. A missing successor is never an observed transition."""
from dataclasses import dataclass
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np


@dataclass
class Episode:
    obs: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_obs: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    next_valid: np.ndarray | None = None


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(value, indent=2, allow_nan=False) + "\n"
    with TemporaryDirectory(dir=path.parent) as tmp:
        pending = Path(tmp) / path.name
        pending.write_text(data, encoding="utf-8")
        pending.replace(path)


def load_episode(path, horizon=None):
    """Read one NPZ. Complete (s,a,r) sequences may omit the last successor."""
    with np.load(path, allow_pickle=False) as raw:
        r = raw["rewards"].reshape(-1)
        n = len(r)
        if not n or ("episode_offsets" in raw and list(raw["episode_offsets"].ravel()) != [0, n]):
            raise ValueError(f"{path} must contain exactly one nonempty episode")
        a = raw["actions"].reshape(n, -1)
        obs = raw["observations"]
        s = obs[:n].reshape(n, -1)
        term = np.asarray(raw.get("terminated", np.zeros(n)), dtype=bool).reshape(n)
        trunc = np.asarray(raw.get("truncated", np.zeros(n)), dtype=bool).reshape(n)
        if (term[:-1] | trunc[:-1]).any():
            raise ValueError(f"{path} contains an interior episode boundary")
        valid = None
        if "next_observations" in raw:
            nxt = raw["next_observations"].reshape(s.shape)
        elif len(obs) == n + 1:
            nxt = obs[1:].reshape(s.shape)
        elif len(obs) == n and (term[-1] or (horizon is not None and n >= horizon)):
            # This placeholder cannot be used as an AM dynamics target.
            nxt = np.concatenate([s[1:], s[-1:]])
            valid = np.ones(n, dtype=bool)
            valid[-1] = False
        else:
            raise ValueError(f"{path} needs its final successor or a known terminal/horizon")
        return Episode(s, a, r, nxt, term, trunc, valid)


def episode_prefix(episode, length):
    """Reveal at most length transitions, including the last observed successor."""
    if length < 1:
        raise ValueError("prefix length must be positive")
    if len(episode.rewards) <= length:
        return episode
    truncated = episode.truncated[:length].copy()
    truncated[-1] = True
    return Episode(episode.obs[:length], episode.actions[:length], episode.rewards[:length],
                   episode.next_obs[:length], episode.terminated[:length], truncated,
                   None if episode.next_valid is None else episode.next_valid[:length])
