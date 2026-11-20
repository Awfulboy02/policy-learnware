import tempfile
import unittest
from pathlib import Path

import numpy as np

from policy_learnware.cli import main
from policy_learnware.data import load_episode, read_json


class PipelineTest(unittest.TestCase):
    def test_complete_ordered_triples_and_early_cutoff(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "episode.npz"
            np.savez(path, observations=[1., 3.], actions=[0., 1.], rewards=[1., 2.])
            episode = load_episode(path, horizon=2)
            np.testing.assert_array_equal(episode.next_obs[:, 0], [3., 3.])
            np.testing.assert_array_equal(episode.next_valid, [True, False])
            with self.assertRaisesRegex(ValueError, "successor"):
                load_episode(path, horizon=3)

    def test_rank_without_source_logs_or_truth_then_score_an_exact_panel(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            main(["prepare", "ringtrack", "--out", tmp])
            main(["specify", "--source", str(out / "source.json"), "--settings",
                  str(out / "settings.json"), "--out", str(out / "catalog"),
                  "--size", "5", "--iterations", "1"])
            for entry in read_json(out / "source.json")["policies"].values():
                for path in entry["episodes"]:
                    Path(path).unlink()
            (out / "truth.json").rename(out / "sealed.json")
            # Archived candidate rollouts are not accessible during ranking.
            judge = out / "judge"
            judge.mkdir()
            for path in (out / "ring_episodes").glob("judge-*.npz"):
                path.rename(judge / path.name)
            runs = {}
            for method, retrieval, k in [("rkme", "rkme", 1), ("fqe", "rkme", 2),
                                         ("fqe", "all", 2), ("am", "random", 1),
                                         ("am", "rkme", 2)]:
                result = out / f"{method}-{retrieval}.json"
                main(["rank", "--catalog", str(out / "catalog/catalog.json"), "--requests",
                      str(out / "requests.json"), "--request", "left", "--k", str(k),
                      "--method", method, "--retrieval", retrieval, "--steps", "1",
                      "--out", str(result)])
                runs[method, retrieval] = read_json(result)
                self.assertEqual(len(runs[method, retrieval]["shortlist"]), k)
                self.assertNotIn("mean_returns", runs[method, retrieval])
            self.assertEqual(runs["fqe", "rkme"]["estimates"], runs["fqe", "all"]["estimates"])
            for path in judge.iterdir():
                path.rename(out / "ring_episodes" / path.name)
            main(["score", "--ranking", str(out / "am-random.json"),
                  "--truth", str(out / "sealed.json"), "--out", str(out / "scores.json")])
            score = read_json(out / "scores.json")
            self.assertAlmostEqual(score["mean_returns"]["__expert__"], 1.)
            self.assertGreater(score["mean_returns"]["left"], score["mean_returns"]["right"])
            self.assertAlmostEqual(score["total_regret"], score["recall_loss"] + score["rerank_loss"])
            self.assertEqual(score["rerank_loss"], 0.)
