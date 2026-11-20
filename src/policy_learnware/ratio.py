"""Finite-horizon tabular OPE from complete, independent discrete-action episodes.

DR and WDR use the same out-of-fold Q estimates. MRDR fits the critic using
Farajtabar et al. (ICML 2018), Eq. 15 (Eq. 16 for deterministic targets).
The MRDR variance objective assumes one logged Markov behavior policy
(time may be part of the state). All values are discounted sums, with zero
reward after true termination.
"""
from dataclasses import dataclass

import numpy as np


METHODS = ("is", "wis", "pdis", "wpdis", "dr", "wdr", "mrdr")


def _distribution(p):
    return np.isfinite(p).all() and (p >= 0).all() and np.allclose(p.sum(-1), 1, rtol=1e-7, atol=1e-9)


@dataclass
class _Trace:
    states: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    beta: np.ndarray
    full_beta: np.ndarray | None
    log_ratio: np.ndarray


def _trace(ep, pi):
    h, n_states, n_actions = pi.shape
    actions = np.asarray(ep["actions"])
    states = np.asarray(ep["states"])
    rewards = np.asarray(ep["rewards"], dtype=float)
    n = len(rewards)
    if not n or states.shape != (n,) or actions.shape != (n,) or rewards.ndim != 1:
        raise ValueError("one ordered episode with scalar state/action IDs is required")
    terminal = np.asarray(ep.get("terminated", False), dtype=bool)
    if terminal.ndim and (terminal.shape != (n,) or np.any(terminal[:-1])):
        raise ValueError("episode contains an invalid or interior termination")
    last_terminal = bool(terminal[-1]) if terminal.ndim else bool(terminal)
    if n < h and not last_terminal:
        raise ValueError("incomplete trajectory before the evaluation horizon")
    t = np.arange(min(n, h))
    s, a, r = states[t], actions[t], rewards[t]
    if (s.dtype.kind not in "iu" or a.dtype.kind not in "iu" or
            np.any((s < 0) | (s >= n_states)) or np.any((a < 0) | (a >= n_actions))):
        raise ValueError("expected in-range discrete state and action IDs")
    if not np.isfinite(r).all():
        raise ValueError("nonfinite logged reward")
    full = None
    if "behavior_probs" in ep:
        full = np.asarray(ep["behavior_probs"], dtype=float)[t]
        if full.shape != (len(t), n_actions) or not _distribution(full):
            raise ValueError("invalid behavior action distributions")
        if np.any((pi[t, s] > 0) & (full == 0)):
            raise ValueError("known action support mismatch on a logged state")
        beta = full[t, a]
        if "behavior_action_prob" in ep and not np.allclose(beta, np.asarray(ep["behavior_action_prob"])[t],
                                                           rtol=1e-7, atol=0.):
            raise ValueError("chosen action propensities disagree with behavior_probs")
    elif "behavior_action_prob" in ep:
        beta = np.asarray(ep["behavior_action_prob"], dtype=float)[t]
    else:
        raise ValueError("actual logged behavior action probabilities required")
    if beta.shape != (len(t),) or not np.all(np.isfinite(beta) & (beta > 0) & (beta <= 1)):
        raise ValueError("observed action propensities must be in (0,1]")
    target = pi[t, s, a]
    log_target = np.full(len(t), -np.inf)
    np.log(target, out=log_target, where=target > 0)
    return _Trace(s, a, r, beta, full, log_target - np.log(beta))


def _probability_weights(logs):
    """Normalize each time step in log space. Keep zero denominators observable."""
    peak = logs.max(axis=1)
    keep = np.isfinite(peak)
    scaled = np.zeros_like(logs)
    scaled[keep] = np.exp(logs[keep] - peak[keep, None])
    scaled[keep] /= scaled[keep].sum(axis=1, keepdims=True)
    ess = np.zeros(len(peak))
    ess[keep] = 1 / np.square(scaled[keep]).sum(axis=1)
    return scaled, ess


def _fitted_q(traces, pi, gamma):
    """Backward tabular Bellman regression on training episodes only."""
    h, ns, na = pi.shape
    q = np.zeros_like(pi)
    for t in reversed(range(h)):
        counts = np.zeros((ns, na), dtype=int)
        for trace in traces:
            if t >= len(trace.rewards):
                continue
            s, a, y = trace.states[t], trace.actions[t], trace.rewards[t]
            if t + 1 < len(trace.rewards):
                nxt = trace.states[t + 1]
                y += gamma * (pi[t + 1, nxt] @ q[t + 1, nxt])
            q[t, s, a] += y
            counts[s, a] += 1
        np.divide(q[t], counts, out=q[t], where=counts > 0)
    return q


def _mrdr_q(traces, pi, gamma):
    # Eq. 15 is a separate positive-semidefinite quadratic in Q[t,s,:]
    # for each tabular state/time. Accumulate the A x A normal equations.
    blocks = {}
    for tr in traces:
        target = pi[np.arange(len(tr.rewards)), tr.states]
        tail = tr.rewards.copy()
        for t in reversed(range(len(tail) - 1)):
            tail[t] += gamma * np.exp(tr.log_ratio[t + 1]) * tail[t + 1]
        log_prefix = 0.
        for t, (s, a, p) in enumerate(zip(tr.states, tr.actions, target)):
            log_factor = 2 * log_prefix + tr.log_ratio[t]
            log_prefix += tr.log_ratio[t]
            if not np.isfinite(log_factor) or (gamma == 0 and t > 0):
                continue
            # gamma**(2*t) is common to all rows in this tabular block.
            if np.count_nonzero(p) == 1:
                curvature = (1 / tr.beta[t] - 1) * np.outer(p, p)
                linear = curvature[:, a] * tail[t]
            else:
                if tr.full_beta is None:
                    raise ValueError("stochastic MRDR requires full behavior action distributions")
                inv_beta = np.divide(1., tr.full_beta[t], out=np.zeros_like(p), where=tr.full_beta[t] > 0)
                curvature = np.diag(p**2 * inv_beta) - np.outer(p, p)
                linear = -p.copy()
                linear[a] += p[a] * inv_beta[a]
                linear *= tail[t]
            key = (t, s)
            if key not in blocks:
                blocks[key] = (log_factor, curvature.copy(), linear.copy())
            else:
                old, mat, rhs = blocks[key]
                peak = max(old, log_factor)
                blocks[key] = (peak, mat * np.exp(old-peak) + curvature * np.exp(log_factor-peak),
                               rhs * np.exp(old-peak) + linear * np.exp(log_factor-peak))
    q = np.zeros_like(pi)
    for (t, s), (_, mat, rhs) in blocks.items():
        q[t, s] = np.linalg.lstsq(mat, rhs, rcond=None)[0]
    return q


def _crossfit(traces, pi, gamma, folds, seed, variance_fit):
    if not 2 <= folds <= len(traces):
        raise ValueError("cross-fitted DR requires at least two independent episode folds")
    observed = np.zeros((pi.shape[0], len(traces)))
    expected = np.zeros_like(observed)
    order = np.random.default_rng(seed).permutation(len(traces))
    for held in np.array_split(order, folds):
        train = [traces[i] for i in np.setdiff1d(order, held)]
        q = _mrdr_q(train, pi, gamma) if variance_fit else _fitted_q(train, pi, gamma)
        for i in held:
            tr = traces[i]
            t = np.arange(len(tr.rewards))
            observed[t, i] = q[t, tr.states, tr.actions]
            expected[t, i] = np.sum(pi[t, tr.states] * q[t, tr.states], axis=-1)
    return observed, expected


def _compute(trajectories, horizon, gamma, method, support_check, critic=None):
    diag = {}
    try:
        n = len(trajectories)
        logs = np.empty((horizon, n))
        rewards = np.zeros_like(logs)
        for i, (ratios, observed_rewards) in enumerate(trajectories):
            end = len(observed_rewards)
            logs[:end, i] = np.cumsum(ratios)
            logs[end:, i] = logs[end - 1, i]  # absorbing padding has unit incremental ratio
            rewards[:end, i] = observed_rewards
        weights, ess = _probability_weights(logs)
        diag = dict(ess=float(ess[-1]), min_step_ess=float(ess.min()),
                    max_normalized_weight=float(weights[-1].max()),
                    zero_denominator_steps=np.flatnonzero(ess == 0).tolist(),
                    support_check=support_check)
        discount = gamma ** np.arange(horizon)
        with np.errstate(over="raise", invalid="raise"):
            if method in ("wis", "wpdis", "wdr"):
                needed = ess[-1:] if method == "wis" else ess[discount > 0]
                if np.any(needed == 0):
                    raise ValueError("zero normalization denominator")
            elif method == "is":
                weights = np.exp(logs[-1:] - np.log(n))
            else:
                weights = np.exp(logs - np.log(n))
            if method in ("dr", "wdr", "mrdr"):
                q, v = critic()
                prev = np.vstack([np.full(n, 1/n), weights[:-1]])
                value = discount @ (weights * (rewards - q) + prev * v).sum(axis=1)
            elif method in ("is", "wis"):
                value = (discount @ rewards) @ weights[-1]
            else:
                value = discount @ (weights * rewards).sum(axis=1)
        if not np.isfinite(value):
            raise ValueError("nonfinite value estimate")
        return dict(status="ok", value=float(value), **diag)
    except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
        return dict(status="unavailable", reason=str(exc), value=None, **diag)


def estimate(episodes, target_probs, *, method, horizon, gamma=1., folds=2, seed=0):
    """Assess one candidate. Unavailable diagnostics can be reported alongside all-pool scores.

    With only chosen propensities, support of unobserved actions remains a
    panel design assumption. Even full beta verifies only logged states.
    """
    if method not in METHODS:
        raise ValueError(f"unknown importance sampling method: {method}")
    try:
        if not episodes or horizon < 1 or not 0 <= gamma <= 1:
            raise ValueError("nonempty episodes, positive horizon and gamma in [0,1] required")
        pi = np.asarray(target_probs, dtype=float)
        if pi.ndim == 2:
            pi = np.broadcast_to(pi, (horizon, *pi.shape))
        if pi.ndim != 3 or pi.shape[0] != horizon or not _distribution(pi):
            raise ValueError("invalid target probabilities: expected [S,A] or [H,S,A]")
        traces = [_trace(ep, pi) for ep in episodes]
        if method == "mrdr" and any(tr.full_beta is None and np.any(np.count_nonzero(
                pi[np.arange(len(tr.rewards)), tr.states], axis=-1) > 1) for tr in traces):
            raise ValueError("stochastic MRDR requires full behavior action distributions")
        trajectories = [(tr.log_ratio, tr.rewards) for tr in traces]
        check = ("logged_state_distributions" if all(tr.full_beta is not None for tr in traces)
                 else "chosen_propensities_only")
        def critic():
            return _crossfit(traces, pi, gamma, folds, seed, method == "mrdr")
        result = _compute(trajectories, horizon, gamma, method, check, critic)
        if method in ("dr", "wdr", "mrdr") and result["status"] == "ok":
            result["folds"] = folds
        return result
    except (ValueError, IndexError, KeyError, TypeError) as exc:
        return dict(status="unavailable", reason=str(exc), value=None)


def estimate_logs(episodes, target_id, *, method, horizon, gamma=1.):
    """IS on true, same-measure log propensities or densities at recorded actions.

    Caller must establish target domination by the behavior policy. A continuous
    deterministic actor generally cannot be evaluated against a density logger.
    JSON null in target_log_probs denotes genuine zero target support.
    """
    if method not in ("is", "wis", "pdis", "wpdis"):
        return dict(status="unavailable", reason="logged ratios alone support IS/WIS/PDIS/WPDIS", value=None)
    try:
        if not episodes or horizon < 1 or not 0 <= gamma <= 1:
            raise ValueError("nonempty episodes, positive horizon and gamma in [0,1] required")
        trajectories = []
        for ep in episodes:
            if ep.get("common_support") is False:
                raise ValueError("declared support does not cover the candidate")
            rewards = np.asarray(ep["rewards"], dtype=float)
            n = len(rewards)
            if not n or rewards.ndim != 1 or not np.isfinite(rewards).all():
                raise ValueError("finite per-transition rewards required")
            term = np.asarray(ep.get("terminated", False), dtype=bool)
            if term.ndim and (term.shape != (n,) or np.any(term[:-1])):
                raise ValueError("episode contains an invalid or interior termination")
            if n < horizon and not bool(term[-1] if term.ndim else term):
                raise ValueError("incomplete trajectory before the evaluation horizon")
            beta = np.asarray(ep["behavior_log_prob"], dtype=float)
            target = np.array([float("-inf") if x is None else x for x in ep["target_log_probs"][target_id]],
                              dtype=float)
            if beta.shape != (n,) or not np.isfinite(beta).all():
                raise ValueError("finite true behavior log probability required at each logged action")
            if target.shape != (n,) or (np.isnan(target) | np.isposinf(target)).any():
                raise ValueError("invalid target log probability")
            trajectories.append((target[:horizon] - beta[:horizon], rewards[:horizon]))
        return _compute(trajectories, horizon, gamma, method, "declared_by_panel")
    except (KeyError, ValueError, TypeError, IndexError) as exc:
        return dict(status="unavailable", reason=str(exc), value=None)
