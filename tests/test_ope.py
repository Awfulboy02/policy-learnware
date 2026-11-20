import unittest
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from policy_learnware.ope import ApproximateModel, fit_am, fit_fqe


def single_step(reward, *, terminal=False, valid=True):
    return SimpleNamespace(obs=np.array([[0.]]), actions=np.array([[0.]]),
                           rewards=np.array([reward]), next_obs=np.array([[3.]]),
                           terminated=np.array([terminal]), truncated=np.array([not terminal]),
                           next_valid=np.array([valid]))


class OfflineValues(unittest.TestCase):
    def test_fqe_bootstraps_early_cutoff_but_skips_terminal_or_horizon(self):
        for terminal, valid, horizon, expected_next in (
                (False, True, 2, True), (True, False, 2, False), (False, False, 1, False)):
            seen = []

            def policy(obs):
                seen.extend(obs[:, 0])
                return np.zeros((len(obs), 1))

            with self.subTest(terminal=terminal, horizon=horizon):
                value = fit_fqe([single_step(2, terminal=terminal, valid=valid)], policy,
                                horizon, 0.5, steps=1)
                self.assertEqual(3. in seen, expected_next)
                self.assertIn(0., seen)
                if not terminal:
                    self.assertAlmostEqual(value, 3. if horizon == 2 else 2., places=5)

    def test_am_uses_existing_rewards_without_unobserved_deltas(self):
        model = fit_am([single_step(3, valid=False)], 1, steps=1)
        self.assertEqual(model.output_center.tolist(), [0., 3.])
        self.assertAlmostEqual(model.value(np.zeros_like, [[0.]], 1, 1.), 3.)

    def test_covered_one_step_actions_rank_by_reward(self):
        episodes = [single_step(r, terminal=True) for r in (-1., 1.)]
        for ep in episodes:
            ep.actions[:] = ep.rewards[0]
        policies = [lambda s: -np.ones_like(s), np.ones_like]
        model = fit_am(episodes, 1, steps=4)
        am = [model.value(p, [[0.]], 1, 1.) for p in policies]
        fqe = [fit_fqe(episodes, p, 1, 1., steps=4) for p in policies]
        self.assertLess(am[0], am[1])
        self.assertLess(fqe[0], fqe[1])

    def test_model_rollout_obeys_policy_discount_and_survival(self):
        # Zero delta, reward equal to the candidate action, survival 1/2.
        class FixedDynamics(nn.Module):
            def forward(self, x):
                return torch.stack([x[:, 0] * 0, x[:, 1], x[:, 0] * 0], dim=1)

        model = ApproximateModel(FixedDynamics(), torch.zeros(2), torch.ones(2),
                                 torch.zeros(2), torch.ones(2), True)
        for action in (1., -1.):
            value = model.value(lambda obs: np.full_like(obs, action), [[2.], [3.]], 3, 0.5)
            self.assertAlmostEqual(value, action * (1 + 0.25 + 0.25 ** 2))
