"""Gaussian reduced kernel mean embeddings of finite-horizon state-action visits."""

from dataclasses import dataclass
from functools import cached_property

import numpy as np


def _features(episode):
    return np.column_stack((episode.obs, episode.actions)).astype(np.float64, copy=False)


def fit_normalizer(episodes):
    """Pool source-side feature moments without retaining individual trajectories."""
    count = 0
    for episode in episodes:
        x = _features(episode)
        if count == 0:
            total = np.zeros(x.shape[1])
            squares = np.zeros_like(total)
        count += len(x)
        total += x.sum(axis=0)
        squares += np.einsum("ij,ij->j", x, x)
    if count == 0:
        raise ValueError("expected source episodes")
    mean = total / count
    scale = np.sqrt(np.maximum(squares / count - mean**2, 0))
    return mean, np.where(scale > 1e-8, scale, 1.0)


def _squared_distances(x, y):
    d = (x * x).sum(axis=1)[:, None] + (y * y).sum(axis=1) - 2 * (x @ y.T)
    return np.maximum(d, 0)


def _kernel(x, y, bandwidth):
    return np.exp(-_squared_distances(x, y) / (2 * bandwidth**2))


def _inner(left, right):
    result = 0.0
    for i in range(0, len(left.points), 1024):
        for j in range(0, len(right.points), 1024):
            ii, jj = slice(i, i + 1024), slice(j, j + 1024)
            block = _kernel(left.points[ii], right.points[jj], left.bandwidth)
            result += left.weights[ii] @ block @ right.weights[jj]
    return float(result)


@dataclass
class Specification:
    points: np.ndarray
    weights: np.ndarray
    bandwidth: float

    @cached_property
    def squared_norm(self):
        return _inner(self, self)

    def save(self, path):
        np.savez(path, points=self.points, weights=self.weights, bandwidth=self.bandwidth)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as f:
            return cls(f["points"], f["weights"], float(f["bandwidth"]))


def distance(left, right):
    """RKHS distance, including both embeddings' weighted self-interactions."""
    if left.bandwidth != right.bandwidth:
        raise ValueError("both specifications must use the same bandwidth")
    squared = left.squared_norm + right.squared_norm - 2 * _inner(left, right)
    return float(np.sqrt(max(0.0, squared)))


def _kmeans(x, weights, size, rng):
    """Weighted k-means++ initialization followed by Lloyd updates."""
    centers = [x[rng.choice(len(x), p=weights / weights.sum())]]
    nearest = _squared_distances(x, centers[0][None, :])[:, 0]
    for _ in range(1, size):
        probabilities = weights * nearest
        if probabilities.sum() <= 0:
            break
        center = x[rng.choice(len(x), p=probabilities / probabilities.sum())]
        centers.append(center)
        nearest = np.minimum(nearest, _squared_distances(x, center[None, :])[:, 0])
    centers = np.stack(centers)
    for _ in range(12):
        labels = np.argmin(_squared_distances(x, centers), axis=1)
        totals = np.bincount(labels, weights=weights, minlength=len(centers))
        sums = np.zeros_like(centers)
        np.add.at(sums, labels, weights[:, None] * x)
        updated = np.divide(sums, totals[:, None], out=centers.copy(), where=totals[:, None] > 0)
        if np.allclose(updated, centers):
            break
        centers = updated
    return centers


def _kernel_moments(z, x, weights, bandwidth):
    # K_zx w and K_zx diag(w) x suffice for the weight solve and support gradient.
    mass = np.zeros(len(z))
    first = np.zeros_like(z)
    for j in range(0, len(x), 1024):
        block = slice(j, j + 1024)
        weighted = _kernel(z, x[block], bandwidth) * weights[block]
        mass += weighted.sum(axis=1)
        first += weighted @ x[block]
    return mass, first


def specify(episodes, mean, scale, *, horizon, gamma=1.0, size=32, bandwidth=1.0,
            iterations=10, seed=0):
    """Fit Eq. 8 of RKME-03 with episode weights gamma**t / (n_episodes * Z_H).

    A true terminal stops contributing visits. An early recording cutoff leaves
    occupancy unknown. size=None retains the full empirical KME for local use.
    """
    if horizon < 1 or not 0 <= gamma <= 1 or bandwidth <= 0:
        raise ValueError("invalid horizon, discount or kernel bandwidth")
    if size is not None and (size < 1 or iterations < 0):
        raise ValueError("invalid reduced set size or iteration count")
    discounts = gamma ** np.arange(horizon, dtype=np.float64)
    features, masses = [], []
    for episode in episodes:
        if len(episode.obs) < horizon and not episode.terminated[-1]:
            raise ValueError("early recording cutoff leaves occupancy before the horizon unknown")
        x = _features(episode)[:horizon]
        features.append((x - mean) / scale)
        masses.append(discounts[:len(x)])
    if not features:
        raise ValueError("expected at least one episode")
    x = np.concatenate(features)
    weights = np.concatenate(masses) / (len(features) * discounts.sum())
    if size is None:
        return Specification(x, weights, bandwidth)

    z = _kmeans(x, weights, min(size, len(x)), np.random.default_rng(seed))
    for step in range(iterations + 1):
        kernel = _kernel(z, z, bandwidth)
        mass, first = _kernel_moments(z, x, weights, bandwidth)
        beta = np.linalg.pinv(kernel, rcond=1e-10, hermitian=True) @ mass
        if step == iterations:
            break
        internal = kernel @ (beta[:, None] * z) - (kernel @ beta)[:, None] * z
        external = first - mass[:, None] * z
        gradient = (2 / bandwidth**2) * beta[:, None] * (internal - external)
        # Bound curvature of the Gaussian objective when signed weights are large.
        abs_beta = np.abs(beta)
        curvature = abs_beta.max() * (4 * abs_beta.sum() + 2 * weights.sum())
        z = z - (bandwidth**2 / max(4.0, curvature)) * gradient
    return Specification(z, beta, bandwidth)
