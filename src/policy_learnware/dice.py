"""Exact tabular discounted OPE for a finite rollout embedded in an absorbing MDP.

Nachum et al. (2019), DualDICE, and Yang et al. (2020), eq. (9), BestDICE with
alpha_Q=0, alpha_zeta=1, alpha_R=0. With one Q coordinate per (time,state,action),
stationarity in Q forces the *empirical* discounted flow equations. Their unique
solution is nonnegative and has unit mass, so the additional BestDICE constraints
give the same optimum. This is a tabular reference, not a neural minimax fit.

The reference measure is uniform over logged transitions plus one model-known
zero-reward absorbing self-loop per episode. Reference mass cancels from the
estimated return. Values are returned in ordinary discounted-sum units.
"""
import numpy as np


def _unavailable(reason, **diagnostics):
    return {"status": "unavailable", "reason": reason, "value": None,
            "diagnostics": diagnostics}


def estimate(episodes, target_probs, *, method, horizon, gamma):
    """Time-major integer state/action logs; optional true termination before H."""
    if method not in ("dualdice", "bestdice"):
        raise ValueError(f"unknown DICE method {method}")
    if not 0 <= gamma < 1:
        return _unavailable("discounted tabular DICE requires 0 <= gamma < 1")
    if not episodes or horizon < 1:
        raise ValueError("expected episodes and a positive horizon")
    pi = np.asarray(target_probs, dtype=float)
    if pi.ndim == 2:
        pi = np.broadcast_to(pi, (horizon, *pi.shape))
    if pi.ndim != 3 or pi.shape[0] != horizon:
        raise ValueError("expected target_probs[state,action] or [horizon,state,action]")
    row_sums = pi.sum(axis=-1, keepdims=True)
    if not np.isfinite(pi).all() or (pi < 0).any() or not np.allclose(
            row_sums, 1, rtol=0, atol=1e-7):
        raise ValueError("target policy rows must be probability vectors")
    pi = pi / row_sums
    states, actions = pi.shape[1:]
    initial = np.zeros(states)
    # Each row carries (state, action, reward, successor); -1 is the absorber.
    steps = [[] for _ in range(horizon)]
    real_count = 0
    for ep in episodes:
        s, a, r = (np.asarray(ep[key]) for key in ("states", "actions", "rewards"))
        n = len(r)
        if not n or s.shape != (n,) or a.shape != (n,) or r.shape != (n,):
            raise ValueError("episode needs equal-length vectors of states, actions, rewards")
        if (not np.issubdtype(s.dtype, np.integer) or not np.issubdtype(a.dtype, np.integer)
                or (s < 0).any() or (s >= states).any() or (a < 0).any() or (a >= actions).any()):
            raise ValueError("DICE panel requires discrete state/action IDs in the table")
        if not np.isfinite(r).all():
            raise ValueError("episode rewards must be finite")
        term = np.asarray(ep.get("terminated", False), dtype=bool)
        if term.ndim:
            if term.shape != (n,) or term[:-1].any():
                raise ValueError("invalid episode terminal boundary")
            term = term[-1]
        if n < horizon and not term:
            return _unavailable("short log ends before the scoring horizon without true termination")
        length = min(n, horizon)
        initial[s[0]] += 1
        for t in range(length):
            successor = int(s[t+1]) if t + 1 < length else -1
            steps[t].append((s[t], a[t], r[t], successor))
        real_count += length

    total = real_count + len(episodes)
    incoming = (1 - gamma) * initial / len(episodes)
    sink_in = reward_mass = mass = squared_weights = max_ratio = max_residual = 0.
    for t, rows in enumerate(steps):
        if not incoming.any():
            break
        goal = incoming[:, None] * pi[t]
        samples = np.asarray(rows, dtype=float).reshape(-1, 4)
        s, a, successor = samples[:, [0, 1, 3]].astype(np.intp).T
        counts = np.bincount(s * actions + a, minlength=states * actions).reshape(states, actions)
        missing = (goal > 0) & (counts == 0)
        if missing.any():
            i, j = np.argwhere(missing)[0]
            return _unavailable("positive target occupancy outside empirical state-action support",
                                time=t, state=int(i), action=int(j),
                                missing_discounted_mass=float(goal[missing].sum()))

        # Solve the triangular KKT flow system for zeta = d_pi / d_D.
        ratio = np.divide(goal * total, counts, out=np.zeros_like(goal), where=counts > 0)
        weighted_rows = ratio[s, a] / total
        max_residual = max(max_residual, float(np.max(abs(ratio * counts / total - goal))))
        max_ratio = max(max_ratio, float(ratio.max()))
        mass += float(weighted_rows.sum())
        squared_weights += float(weighted_rows @ weighted_rows)
        reward_mass += float(weighted_rows @ samples[:, 2])
        alive = successor >= 0
        incoming = gamma * np.bincount(successor[alive], weights=weighted_rows[alive],
                                       minlength=states)
        sink_in += gamma * float(weighted_rows[~alive].sum())

    sink_mass = sink_in / (1 - gamma)
    real_mass = mass
    mass += sink_mass
    sink_ratio = sink_mass * total / len(episodes)
    if method == "bestdice" and not np.isclose(mass, 1., rtol=1e-7, atol=1e-9):
        return _unavailable("BestDICE unit-mass constraint violated", mass=mass)
    return {"status": "ok", "value": reward_mass / (1 - gamma),
            "diagnostics": {"method": method, "solver": "exact_tabular_flow",
                            "normalized_reward": reward_mass, "weight_mass": mass,
                            "max_real_ratio": max_ratio, "absorbing_ratio": sink_ratio,
                            "empirical_flow_residual_inf": max(max_residual,
                                abs(sink_mass - gamma * sink_mass - sink_in)),
                            "transition_ess": real_mass**2 / squared_weights}}
