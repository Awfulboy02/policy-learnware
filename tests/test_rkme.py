import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from policy_learnware.rkme import Specification, distance, fit_normalizer, specify


def episode(points, terminal=True):
    points = np.array(points, dtype=float)[:, None]
    done = np.zeros(len(points), dtype=bool)
    done[-1] = terminal
    return SimpleNamespace(obs=points, actions=np.zeros_like(points), terminated=done)


class RKMETest(unittest.TestCase):
    def test_discounted_occupancy_uses_common_horizon(self):
        episodes = [episode([0, 1]), episode([2])]
        spec = specify(episodes, np.zeros(2), np.ones(2), horizon=2, gamma=.5, size=None)
        np.testing.assert_allclose(spec.weights, [1 / 3, 1 / 6, 1 / 3])
        with self.assertRaisesRegex(ValueError, "cutoff"):
            specify([episode([0], terminal=False)], np.zeros(2), np.ones(2), horizon=2)

    def test_reduction_moves_support_and_preserves_kernel_geometry(self):
        episodes = [episode([-1.4, -1.3, -.8, .2, .3, 1.5])]
        mean, scale = fit_normalizer(episodes)
        options = dict(horizon=6, bandwidth=.8, size=2)
        reference = specify(episodes, mean, scale, horizon=6, bandwidth=.8, size=None)
        initial = specify(episodes, mean, scale, iterations=0, **options)
        fitted = specify(episodes, mean, scale, iterations=10, **options)
        self.assertLess(distance(fitted, reference), distance(initial, reference))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "spec.npz"
            fitted.save(path)
            self.assertAlmostEqual(distance(Specification.load(path), fitted), 0, places=6)

    def test_distance_includes_cross_terms_with_signed_weights(self):
        points = np.array([[0.], [1.]])
        left = Specification(points, np.array([1., -.5]), 1.)
        right = Specification(points, np.array([0., .5]), 1.)
        self.assertAlmostEqual(distance(left, right), np.sqrt(2 * (1 - np.exp(-.5))))
