import unittest

import numpy as np

from policy_learnware.ratio import METHODS, _mrdr_q, _trace, estimate, estimate_logs


def episode(actions, rewards, behavior=None, terminated=True):
    n = len(actions)
    return dict(states=[0] * n, actions=actions, rewards=rewards, terminated=terminated,
                behavior_probs=behavior if behavior is not None else [[.5, .5]] * n)


class RatioMethodsTest(unittest.TestCase):
    def test_two_step_trajectory_vs_step_and_absorbing_padding(self):
        # Under the target action 1, episode zero terminates immediately.
        # The second episode matches at t=0 and departs at t=1.
        eps = [episode([1], [3.]), episode([1, 0], [1., 99.])]
        pi = [[0, 1]]
        expected = {"is": 3., "wis": 3., "pdis": 4., "wpdis": 2.}
        for name, value in expected.items():
            result = estimate(eps, pi, method=name, horizon=2, gamma=.5)
            self.assertEqual(result["status"], "ok", name)
            self.assertAlmostEqual(result["value"], value)
        # Verify terminated paths keep their mass in the denominator at t=1.
        eps[1]["actions"][-1] = 1
        value = estimate(eps, pi, method="wpdis", horizon=2, gamma=.5)["value"]
        self.assertAlmostEqual(value, 2 + .5 * 99 * 4/6)

    def test_stationary_and_time_dependent_oracle(self):
        actions = [(a, b) for a in [0, 1] for b in [0, 1]]
        eps = [episode([a,b], [float(a), float(b)]) for a,b in actions] * 4
        # Equal, disjoint deterministic replicas form a balanced fold pair.
        seed = 11
        hold = np.random.default_rng(seed).permutation(16)
        eps = [eps[list(hold).index(i)] for i in range(16)]
        for method in METHODS:
            result = estimate(eps, [[[1., 0]], [[0., 1]]], horizon=2, gamma=.7,
                              method=method, folds=2, seed=seed)
            self.assertEqual(result["status"], "ok", (method, result))
            self.assertAlmostEqual(result["value"], .7, msg=method)

    def test_crossfit_holds_out_the_entire_evaluation_episode(self):
        # Train on reward 10 to score the mismatching action, then train on the
        # mismatching-action episode to score the episode earning 10.
        eps = [episode([1], [10.]), episode([0], [0.])]
        for method in ["dr", "wdr", "mrdr"]:
            value = estimate(eps, [[0, 1]], method=method, horizon=1)["value"]
            self.assertAlmostEqual(value, 15., msg=method)

    def test_mrdr_stochastic_variance_objective(self):
        pi = np.array([[[.75, .25]]])
        train = [_trace(episode([0], [2.], [[.5,.5]]), pi),
                 _trace(episode([1], [1.], [[.5,.5]]), pi)]
        q = _mrdr_q(train, pi, 1.)[0,0]
        # Evaluate Eq. 15 itself, independently of its implemented normal equations.
        def objective(v):
            val = 0.
            for tr in train:
                a = tr.actions[0]
                err = pi[0,0] * v - np.eye(2)[a] * tr.rewards[0]
                omega = np.diag(1. / tr.full_beta[0]) - np.ones((2,2))
                val += (pi[0,0,a] / tr.beta[0]) * err @ omega @ err
            return val
        for d in np.eye(2):
            self.assertAlmostEqual(objective(q+d) - objective(q-d), 0., places=8)
        self.assertLess(objective(q), objective(np.zeros(2)))

    def test_log_density_route_matches_discrete_ratios_at_tiny_densities(self):
        eps = [episode([1], [3.]), episode([1,0], [1., 99.])]
        direct = [dict(rewards=e["rewards"], terminated=True,
                       behavior_log_prob=[-900.] * len(e["actions"]),
                       target_log_probs={"x": [-900 + np.log(2) if a == 1 else None
                                                for a in e["actions"]]}) for e in eps]
        for m in ("is", "wis", "pdis", "wpdis"):
            v = estimate(eps, [[0, 1]], method=m, horizon=2, gamma=.5)
            logged = estimate_logs(direct, "x", method=m, horizon=2, gamma=.5)
            self.assertAlmostEqual(v["value"], logged["value"])
            self.assertAlmostEqual(v["ess"], logged["ess"])
        direct[0]["behavior_log_prob"] = [None]
        self.assertIsNone(estimate_logs(direct, "x", method="is", horizon=2)["value"])

    def test_permissions_overlap_and_degeneracy(self):
        pi = [[0., 1.]]
        miss = [episode([0], [1.]), episode([0], [2.])]
        result = estimate(miss, pi, method="wis", horizon=1)
        self.assertIsNone(result["value"])
        self.assertEqual(result["ess"], 0.)
        self.assertEqual(result["zero_denominator_steps"], [0])
        miss[0]["behavior_probs"] = [[1., 0.]]
        self.assertIn("mismatch", estimate(miss, pi, method="dr", horizon=1)["reason"])
        del miss[0]["behavior_probs"]
        self.assertIn("required", estimate(miss, pi, method="is", horizon=1)["reason"])
        miss[0]["behavior_action_prob"] = [.5]
        self.assertIn("full behavior", estimate(miss, [[.5,.5]], method="mrdr", horizon=1)["reason"])
        self.assertEqual(estimate(miss, pi, method="mrdr", horizon=1)["status"], "ok")
        self.assertIn("incomplete", estimate([episode([1], [0.], terminated=False)], pi,
                                              method="wis", horizon=2)["reason"])
