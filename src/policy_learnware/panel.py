"""Full-pool OPE on an offline panel with discrete state and action IDs."""
from collections import Counter, defaultdict
from time import perf_counter

import numpy as np

from .data import read_json


def _validate(panel):
    horizon, gamma = panel["horizon"], panel["gamma"]
    if horizon < 1 or not 0 <= gamma <= 1 or not panel["episodes"] or not panel["policies"]:
        raise ValueError("expected nonempty episodes and policies and a valid horizon/discount")
    tables = {}
    for name, rows in panel["policies"].items():
        p = np.asarray(rows, dtype=float)
        if p.ndim not in (2, 3) or (p.ndim == 3 and p.shape[0] != horizon):
            raise ValueError(f"{name}: expected an S x A or H x S x A probability table")
        if min(p.shape) < 1 or np.any(p < 0) or not np.allclose(p.sum(axis=-1), 1):
            raise ValueError(f"{name}: invalid action distribution")
        tables[name] = p
    shapes = {p.shape[-2:] for p in tables.values()}
    if len(shapes) != 1:
        raise ValueError("all candidates must use the same discrete state/action space")
    states, actions = shapes.pop()
    for ep in panel["episodes"]:
        s, a, r = (np.asarray(ep[k]) for k in ("states", "actions", "rewards"))
        if (r.ndim != 1 or not np.isfinite(r).all() or s.shape != r.shape or a.shape != r.shape
                or not 1 <= len(r) <= horizon):
            raise ValueError("expected one complete ordered episode per record")
        if (s.dtype.kind not in "iu" or a.dtype.kind not in "iu" or s.min() < 0 or s.max() >= states
                or a.min() < 0 or a.max() >= actions):
            raise ValueError("logged state/action outside the declared discrete space")
        term = np.asarray(ep.get("terminated", False), dtype=bool)
        if term.ndim:
            if term.shape != r.shape or term[:-1].any():
                raise ValueError("invalid terminal flags within episode")
            term = term[-1]
        if len(r) < horizon and not term:
            raise ValueError("early cutoff is not a complete trial")
    return tables


def _model(episodes, horizon):
    """Time-dependent empirical transition/reward model. End of trial is absorbing."""
    visits, successors, rewards = Counter(), defaultdict(Counter), defaultdict(float)
    starts = Counter(ep["states"][0] for ep in episodes)
    for ep in episodes:
        for t, (s, a, r) in enumerate(zip(ep["states"], ep["actions"], ep["rewards"])):
            key = t, s, a
            visits[key] += 1
            rewards[key] += r
            if t + 1 < len(ep["states"]):
                successors[key][ep["states"][t + 1]] += 1
    return visits, successors, rewards, {s: n / len(episodes) for s, n in starts.items()}


def _tabular_value(model, policy, horizon, gamma, method):
    visits, successors, rewards, starts = model
    p = np.broadcast_to(policy, (horizon, *policy.shape[-2:]))
    # Forward propagation identifies exactly which actions require supported data.
    active, levels, total = starts, [], 0.
    stop = 1 if gamma == 0 else horizon
    for t in range(stop):
        levels.append(active)
        nxt = defaultdict(float)
        for s, weight in active.items():
            for a in np.flatnonzero(p[t, s]):
                key = t, s, a
                if not visits[key]:
                    return {"status": "unavailable", "value": None,
                            "reason": f"unobserved reachable (time,state,action) {key}"}
                factor = weight * p[t, s, a] / visits[key]
                total += gamma**t * factor * rewards[key]
                for s1, count in successors[key].items():
                    nxt[s1] += factor * count
        active = nxt
    if method == "fqe":
        # Exact tabular fitted Bellman regression at each time step, backwards.
        values = {}
        for t in reversed(range(stop)):
            before = {}
            for s in levels[t]:
                before[s] = 0.
                for a in np.flatnonzero(p[t, s]):
                    key = t, s, a
                    future = (sum(n * values[s1] for s1, n in successors[key].items())
                              if t + 1 < stop else 0.)
                    before[s] += p[t, s, a] * (rewards[key] + gamma * future) / visits[key]
            values = before
        total = sum(w * values[s] for s, w in starts.items())
    return {"status": "ok", "value": float(total)}


def rank_panel(path, method, folds=2, seed=0):
    start = perf_counter()
    panel = read_json(path)
    log_panel = panel.get("kind", "discrete") == "log_probs"
    if log_panel:
        # Only declared true densities on a common measure are comparable.
        policies = dict.fromkeys(panel["candidates"])
        if not policies or len(policies) != len(panel["candidates"]):
            raise ValueError("expected a nonempty full pool with distinct candidate IDs")
    else:
        policies = _validate(panel)
    episodes, h, gamma = panel["episodes"], panel["horizon"], panel["gamma"]
    cfg = dict(horizon=h, gamma=gamma)
    model = _model(episodes, h) if not log_panel and method in {"am", "fqe"} else None
    reports, times = {}, {}
    for name, policy in sorted(policies.items()):
        t = perf_counter()
        if log_panel:
            from .ratio import estimate_logs
            if method not in {"is", "wis", "pdis", "wpdis"}:
                report = {"status": "unavailable", "value": None, "reason": "action log ratios alone cannot run this estimator"}
            elif panel.get("common_support") is not True:
                report = {"status": "unavailable", "value": None, "reason": "common support and a common action measure must be established"}
            else:
                report = estimate_logs(episodes, name, method=method, **cfg)
        elif method in {"am", "fqe"}:
            report = _tabular_value(model, policy, h, gamma, method)
        elif method in {"dualdice", "bestdice"}:
            from .dice import estimate
            report = estimate(episodes, policy, method=method, **cfg)
        else:
            from .ratio import estimate
            report = estimate(episodes, policy, method=method, folds=folds, seed=seed, **cfg)
        reports[name] = report
        times[name] = perf_counter() - t
    estimates = {name: r["value"] for name, r in reports.items()}
    pool = sorted(policies)
    ranking = (sorted(pool, key=lambda i: (-estimates[i], i))
               if all(r["status"] == "ok" for r in reports.values()) else [])
    return {"request": panel["request"], "task": panel.get("task"), "split": panel.get("split"), "method": method, "horizon": h, "gamma": gamma,
            "pool": pool, "shortlist": pool, "retrieval": "all", "ranking": ranking,
            "selected": ranking[0] if ranking else None, "status": "ok" if ranking else "unavailable",
            "estimates": estimates, "candidates": reports, "episode_count": len(episodes),
            "folds": folds if method in {"dr", "wdr", "mrdr"} else None, "seed": seed,
            "seconds": {"total": perf_counter() - start, "by_candidate": times}}
