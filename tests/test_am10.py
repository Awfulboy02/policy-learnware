import tempfile
import unittest
from pathlib import Path

import numpy as np

from policy_learnware.am10 import _panel, _source
from policy_learnware.data import write_json


class AssetPreparation(unittest.TestCase):
    def test_qualification_uses_four_disjoint_final_episodes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = [root / f"episode_{i:03}.npz" for i in range(8)]
            for i, path in enumerate(paths):
                np.savez(path, rewards=[5.0 if i < 4 else 100.0])
            write_json(root / "bundle.json", {"parameters": "weights.npz", "policy_id": "p",
                                               "deployment_profile": {"mode": "stochastic"}})
            np.savez(root / "weights.npz", w=[0])
            asset = {"asset_id": "p", "source_id": "s", "source_certificate": "unknown",
                     "runtime": {"policy": {"bundle": str(root / "bundle.json")}},
                     "source_return_records": [{"binding": {"path": str(p)},
                                                "generator_policy_id": "p"} for p in reversed(paths)]}
            result = _source(asset, 5.1)
            self.assertFalse(result["passed"])
            self.assertEqual(result["mean_return"], 5.0)
            self.assertTrue(set(result["episodes"]).isdisjoint(result["qualification"]))
            self.assertEqual(result["strict_certificate"], "unknown")
            self.assertTrue(_source(asset, 5.0)["passed"])

    def test_panel_requires_completed_attempt_and_all_paired_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            attempt = root / "episodes/q__effect__p/attempt_0002"
            attempt.mkdir(parents=True)
            receipt = dict(asset_id="p", task_group="q", attempt_dir=str(attempt),
                           status="completed", completed=8)
            write_json(root / "receipts/cells/q__effect__p/latest.json", receipt)
            for i in range(7):
                (attempt / f"episode_{i:03}.npz").touch()
            self.assertEqual(_panel(root, "q", "p"), [])
            # Scoring files can be indexed without opening any reward arrays.
            (attempt / "episode_007.npz").touch()
            self.assertEqual(len(_panel(root, "q", "p")), 8)
