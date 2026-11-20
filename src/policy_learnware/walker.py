"""Adapter for the archived WalkerWalk experiment (b_v2_data_20260909)."""

from pathlib import Path

import numpy as np

from .data import read_json, write_json


def load_policy(spec):
    """Exported BC actor on raw Walker observations; accepts a batch of states."""
    with np.load(spec["path"], allow_pickle=False) as checkpoint:
        mean = checkpoint["mean"]
        scale = checkpoint["scale"]
        layers = [(checkpoint[f"w{i}"], checkpoint[f"b{i}"])
                  for i in range(int(checkpoint["layer_count"]))]

    def act(observations):
        x = ((np.asarray(observations, dtype=float) - mean) / scale).astype(np.float32)
        for weights, bias in layers[:-1]:
            x = np.maximum(x @ weights + bias, 0)
        weights, bias = layers[-1]
        return np.tanh(x @ weights + bias)

    return act


def _episodes(directory, count):
    paths = sorted(directory.glob("episode_*.npz"))
    if len(paths) != count:
        raise ValueError(f"{directory}: expected {count} episodes, found {len(paths)}")
    return [str(path) for path in paths]


def prepare(root, output):
    """Index the existing episodes in place. Only source qualification reads rewards."""
    root = Path(root).resolve()
    records = root / "records"
    assets = root.parent.parent / "assets" / "b_v2_20260909"

    policies = {}
    qualifications = {}
    for cohort in ("pilot", "asset_development"):
        for directory in sorted((records / cohort).iterdir()):
            if not directory.is_dir():
                continue
            candidate = directory.name
            asset = assets / candidate
            metadata = read_json(asset / "asset.json")
            paths = _episodes(directory, 12)
            returns = []
            for path in paths[:6]:
                with np.load(path, allow_pickle=False) as episode:
                    returns.append(episode["rewards"].sum(dtype=float))
            mean_return = float(np.mean(returns))
            # This MSE uses independent archived source expert episodes.
            behavior_mse = float(metadata["source_behavior"]["mean"])
            passed = mean_return >= 900 and behavior_mse <= 0.01
            qualifications[candidate] = {
                "mean_return": mean_return, "source_behavior_mse": behavior_mse,
                "passed": passed, "qualification": paths[:6],
            }
            if passed:
                checkpoint = asset / "policy.npz"
                if not checkpoint.is_file():
                    raise FileNotFoundError(checkpoint)
                policies[candidate] = {
                    "policy": {"kind": "walker_bc", "path": str(checkpoint)},
                    "episodes": paths[6:], "qualification": paths[:6],
                }

    requests = {}
    truth = {}
    for archive_split, split in (("train", "development"), ("validation", "validation"),
                                 ("test", "test")):
        groups = read_json(root / f"groups_{archive_split}.json")
        for group in groups:
            request = group["task_group"]
            directory = records / request
            requests[request] = {"split": split, "episodes": _episodes(directory / "query", 8)}
            panel = {candidate: _episodes(directory / f"effect_{candidate}", 4)
                     for candidate in policies}
            panel["__expert__"] = _episodes(directory / "effect_walker_sac_seed0_reference", 4)
            truth[request] = panel

    output = Path(output)
    manifests = {
        "source": {"policies": policies, "qualifications": qualifications},
        "requests": {"requests": requests},
        "truth": {"requests": truth},
    }
    for name, manifest in manifests.items():
        write_json(output / f"{name}.json", manifest)
