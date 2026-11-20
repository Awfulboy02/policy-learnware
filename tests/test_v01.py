from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from policy_learnware.cli import main
from policy_learnware.data import episode_prefix, load_episode, read_json, write_json
from policy_learnware.pipeline import _candidate_pool


class Version01Test(unittest.TestCase):
    def test_nested_pools_and_source_user_prefixes(self):
        assets = {str(i): {"stratum": str(i % 2)} for i in range(10)}
        two = _candidate_pool(assets, 2, 3)
        self.assertEqual(len({assets[i]["stratum"] for i in two}), 2)
        self.assertTrue(set(two) <= set(_candidate_pool(assets, 7, 3)))
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            main(["prepare", "ringtrack", "--out", tmp])
            main(["specify", "--source", str(d/"source.json"), "--settings", str(d/"settings.json"),
                  "--out", str(d/"cat"), "--size", "2", "--source-budget", "1", "--source-prefix", "7"])
            cat = read_json(d/"cat/catalog.json")
            self.assertEqual(cat["source_prefix"], 7)
            self.assertEqual({p["episode_count"] for p in cat["policies"].values()}, {1})
            ep = load_episode(read_json(d/"requests.json")["requests"]["left"]["episodes"][0])
            short = episode_prefix(ep, 3)
            self.assertTrue(short.truncated[-1] and not short.terminated[-1])
            np.testing.assert_equal(short.next_obs[-1], ep.obs[3])
            args = ["rank", "--catalog", str(d/"cat/catalog.json"), "--requests", str(d/"requests.json"),
                    "--request", "left", "--method", "fqe", "--user-prefix", "3", "--k", "2",
                    "--steps", "1", "--out", str(d/"run.json")]
            def inspect(episodes, policy, h, gamma, **kw):
                self.assertTrue(all(len(e.rewards) == 3 for e in episodes))
                self.assertEqual(h, 52)
                return float("nan")
            with patch("policy_learnware.ope.fit_fqe", side_effect=inspect):
                main(args)
                self.assertIsNone(read_json(d/"run.json")["selected"])
                main(args + ["--on-invalid", "raw"])
            run = read_json(d/"run.json")
            self.assertEqual((run["selected"], run["status"]), (run["shortlist"][0], "raw_fallback"))
            truth = read_json(d/"truth.json")
            del truth["requests"]["left"]["__expert__"]
            write_json(d/"judge.json", truth)
            main(["score", "--ranking", str(d/"run.json"), "--truth", str(d/"judge.json"),
                  "--out", str(d/"scores.json")])
            self.assertIsNone(read_json(d/"scores.json")["expert_gap"])
            main(["summarize", "--scores", str(d/"scores.json"), "--out", str(d/"summary.json")])
            self.assertEqual(read_json(d/"summary.json")["fallbacks"], 1)

    def test_full_pool_entry_point_and_permission(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            ep = [{"states":[0], "actions":[a], "rewards":[a + 1], "behavior_probs":[[.5,.5]],
                   "terminated": True} for a in (0, 1, 0, 1, 0, 1, 0, 1)]
            panel = {"request":"unit", "horizon":1, "gamma":.5, "episodes":ep,
                     "policies":{"first":[[1.,0.]], "second":[[0.,1.]]}}
            write_json(d/"panel.json", panel)
            def run(method):
                main(["full-pool", "--panel", str(d/"panel.json"), "--method", method,
                      "--out", str(d/"result.json")])
                return read_json(d/"result.json")
            for method in ("fqe","am","is","wis","pdis","wpdis","dr","wdr",
                           "mrdr","dualdice","bestdice"):
                with self.subTest(method=method):
                    result = run(method)
                    self.assertEqual(result["selected"], "second")
                    self.assertAlmostEqual(result["estimates"]["second"], 2.)
            for item in ep:
                del item["behavior_probs"]
            write_json(d/"panel.json", panel)
            self.assertIsNone(run("wdr")["selected"])
            self.assertEqual(run("dualdice")["selected"], "second")
            logged = {"kind":"log_probs","request":"unit","candidates":["p"],"horizon":1,"gamma":1,
                      "episodes":[{"rewards":[2.], "behavior_log_prob":[-904.],
                                   "target_log_probs":{"p":[-904.]}}]}
            write_json(d/"panel.json",logged)
            self.assertIsNone(run("wis")["selected"])
            logged["common_support"] = True
            write_json(d/"panel.json",logged)
            self.assertEqual(run("wis")["estimates"]["p"],2.)
