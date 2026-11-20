import unittest
import numpy as np

from policy_learnware.dice import estimate


class TabularDICE(unittest.TestCase):
    def test_discounted_flow_stochastic_successors_and_absorbing_tail(self):
        # At time zero the logger sampled action 1 on 3/4 episodes.
        # Under the target the action mix is 1/2,1/2. Action 1 splits
        # equally between a terminal reward and the two successor returns below.
        episodes = [
            dict(states=[0, 1], actions=[0, 0], rewards=[2, 4]),
            dict(states=[0, 1], actions=[1, 0], rewards=[1, 4]),
            dict(states=[0, 2], actions=[1, 0], rewards=[1, 8]),
            dict(states=[0], actions=[1], rewards=[1], terminated=[True]),
        ]
        pi = np.array([[[.5,.5],[1,0],[1,0]], [[1,0],[1,0],[1,0]]])
        expected = .5*(2+.5*4) + .5*(1+.5*(4+8)/3)
        for method in ("dualdice", "bestdice"):
            with self.subTest(method=method):
                out = estimate(episodes, pi, method=method, horizon=2, gamma=.5)
                self.assertEqual(out["status"], "ok")
                self.assertAlmostEqual(out["value"], expected)
                info = out["diagnostics"]
                self.assertAlmostEqual(info["normalized_reward"], .5*expected)
                self.assertAlmostEqual(info["weight_mass"], 1.)
                self.assertAlmostEqual(info["max_real_ratio"], 2.75)
                self.assertAlmostEqual(info["empirical_flow_residual_inf"], 0.)
                scalar_terminal = [*episodes[:-1], dict(episodes[-1], terminated=True)]
                self.assertEqual(out, estimate(scalar_terminal, pi, method=method,
                                               horizon=2, gamma=.5))

    def test_coverage_censor_and_undiscounted_target(self):
        panel = [dict(states=[0, 1], actions=[0, 0], rewards=[3, 17])]
        policy = [[1,0], [0,1]]
        missing = estimate(panel, policy, method="dualdice", horizon=2, gamma=.9)
        self.assertIsNone(missing["value"])
        self.assertEqual(missing["diagnostics"]["time"], 1)
        for method in ("dualdice", "bestdice"):
            self.assertIsNone(estimate(panel, policy, method=method, horizon=2, gamma=1.)["value"])
            self.assertIsNone(estimate(panel, policy, method=method, horizon=3, gamma=.9)["value"])
            immediate = estimate(panel, policy, method=method, horizon=2, gamma=0)
            self.assertEqual(immediate["value"], 3.)
