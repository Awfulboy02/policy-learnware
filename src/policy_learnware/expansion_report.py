"""Summarize independent expansion runs and compare qualified source policies."""

import argparse
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np

from .data import load_episode, read_json, write_json
from .policies import load_policy
from .rkme import Specification, distance


def _cost(rows):
    passed = [r for r in rows if r["qualification"]["passed"]]
    hours = sum(r["training_seconds"] for r in rows) / 3600
    gpu_rows = [r for r in rows if r["device"].startswith("cuda")]
    gpu_hours = sum(r["training_seconds"] for r in gpu_rows) / 3600
    first = {}
    for r in rows:
        floor = r["qualification"]["floor"]
        steps = [s["env_steps"] for s in r["slices"] if s["mean_return"] >= floor]
        first[r["id"]] = min(steps, default=None)
    return {
        "runs": len(rows), "seeds": sorted({r["seed"] for r in rows}), "qualified": len(passed),
        "devices": sorted({r["device"] for r in rows}), "training_device_hours": hours,
        "gpu_training_hours": gpu_hours,
        "gpu_training_hours_per_qualified": gpu_hours / len(passed)
        if passed and len(gpu_rows) == len(rows) else None,
        "best_validation_return": max(s["mean_return"] for r in rows for s in r["slices"]),
        "first_passing_validation_step_by_run": first,
    }


def _fold(episodes, rng, sample_size):
    # Pick an episode uniformly, then a time uniformly in that episode.
    # Keep the empirical episode weighting independent of episode length.
    features = []
    for idx in rng.integers(len(episodes), size=sample_size):
        ep = episodes[idx]
        t = rng.integers(len(ep.obs))
        features.append(np.concatenate((ep.obs[t], ep.actions[t])))
    return np.asarray(features, dtype=np.float64)


def _speed_curve(paths):
    """Equal-episode mean torso speed, aligned to the four 250-control-step windows."""
    edges = (0, 250, 500, 750, 1000)
    windows = [[] for _ in range(4)]
    dts = []
    for path in paths:
        with np.load(path, allow_pickle=False) as raw:
            if "forward_speed" not in raw:
                return None
            speed = raw["forward_speed"].reshape(-1)
            if "control_dt" in raw:
                dts.append(float(raw["control_dt"].item()))
        for values, start, end in zip(windows, edges, edges[1:]):
            if len(speed) > start:
                values.append(float(np.mean(speed[start:end])))
    curve = dict(step_edges=edges, mean_m_per_s=[np.mean(v).item() if v else None for v in windows],
                 episodes_per_window=[len(v) for v in windows])
    if len(dts) == len(paths) and np.allclose(dts, dts[0]):
        curve["second_edges"] = (np.array(edges) * dts[0]).tolist()
    return curve


def _behavior(rows, tolerance, sample_size):
    """Held-out source episodes split by parity, using a shared within-source kernel."""
    eligible = [r for r in rows if len(r.get("behavior_episodes", [])) >= 2]
    pairs = [(a, b) for a, b in combinations(eligible, 2)
             if abs(a["qualification"]["mean_return"] - b["qualification"]["mean_return"]) <= tolerance
             and (a["algorithm"] != b["algorithm"] or a["seed"] != b["seed"])]
    report = {
        "qualified_runs": len(rows),
        "runs": {r["id"]: dict(algorithm=r["algorithm"], seed=r["seed"],
                               mean_return=r["qualification"]["mean_return"]) for r in rows},
        "missing_two_behavior_episodes": [r["id"] for r in rows if len(r.get("behavior_episodes", [])) < 2],
        "pairs": [], "group_means": {},
    }
    for row in eligible:
        if row["task"] == "CheetahRun":
            curve = _speed_curve(row["behavior_episodes"])
            if curve is not None:
                report["runs"][row["id"]]["forward_speed"] = curve
    if not pairs:
        return report
    samples = {}
    for row in eligible:
        episodes = [load_episode(path) for path in row["behavior_episodes"]]
        obs_size = episodes[0].obs.shape[1]
        rng = np.random.default_rng(0)
        samples[row["id"]] = [_fold(episodes[i::2], rng, sample_size) for i in range(2)]
    # A common diagonal scale and bandwidth across both folds and all pairs.
    pooled = np.concatenate([fold for folds in samples.values() for fold in folds])
    mean, sd = pooled.mean(axis=0), pooled.std(axis=0)
    sd = np.where(sd > 1e-8, sd, 1.0)
    bandwidth = float(np.sqrt(len(sd)))
    report.update(bandwidth=bandwidth, feature_mean=mean.tolist(), feature_scale=sd.tolist())
    rng = np.random.default_rng(1)
    probe_features = pooled[rng.choice(len(pooled), size=sample_size, replace=False)]
    probes = probe_features[:, :obs_size].astype(np.float32)
    actions, specs = {}, {}
    for row in eligible:
        key = row["id"]
        actions[key] = load_policy(row["policy"])(probes)
        specs[key] = [Specification((v - mean) / sd, np.full(len(v), 1 / len(v)), bandwidth)
                      for v in samples[key]]

    groups = defaultdict(list)
    for a, b in pairs:
        left, right = a["id"], b["id"]
        group = "within_algorithm" if a["algorithm"] == b["algorithm"] else "across_algorithms"
        mse = float(np.mean((actions[left] - actions[right]) ** 2))
        mmd = [distance(x, y) for x, y in zip(specs[left], specs[right])]
        groups[group].append((mse, *mmd))
        report["pairs"].append(dict(left=left, right=right, group=group, action_mse=mse,
                                    occupancy_mmd_folds=mmd))
    for group, metrics in groups.items():
        mse, fold0, fold1 = np.mean(metrics, axis=0).tolist()
        report["group_means"][group] = dict(pairs=len(metrics), action_mse=mse,
                                            occupancy_mmd_folds=[fold0, fold1])
    return report


def summarize(directories, output, *, behavior=False, tolerance=30.0, sample_size=128):
    if tolerance < 0 or sample_size < 1:
        raise ValueError("return tolerance must be nonnegative and sample size must be positive")
    by_algorithm, by_source, catalogs = defaultdict(list), defaultdict(list), defaultdict(dict)
    missing = []
    identities = set()
    for directory in directories:
        path = Path(directory) / "result.json"
        if not path.is_file():
            missing.append(str(directory))
            continue
        row = read_json(path)
        task, source, algorithm = row["task"], row["source"], row["algorithm"]
        key = (task, row["id"])
        if key in identities:
            raise ValueError(f"duplicate run identity {key}")
        identities.add(key)
        by_algorithm[task, source, algorithm].append(row)
        qualified_runs = by_source[task, source]
        policies = catalogs[task]
        if row["qualification"]["passed"]:
            if not row["episodes"]:
                raise ValueError(f"{row['id']} passed but has no specification episodes")
            policies[row["id"]] = dict(policy=row["policy"], episodes=row["episodes"],
                                       stratum=[source, algorithm])
            qualified_runs.append(row)
    groups = [dict(task=t, source=s, algorithm=a, **_cost(rows))
              for (t, s, a), rows in sorted(by_algorithm.items())]
    output = Path(output)
    write_json(output / "summary.json", dict(groups=groups, missing_result_directories=missing))
    for task, policies in catalogs.items():
        write_json(output / task / "source.json", dict(policies=policies))
    if behavior:
        results = [dict(task=t, source=s, **_behavior(rows, tolerance, sample_size))
                   for (t, s), rows in sorted(by_source.items())]
        write_json(output / "behavior.json", dict(return_tolerance=tolerance,
                   samples_per_run_per_fold=sample_size, sources=results))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", nargs="+", type=Path, help="one directory per independent run")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--behavior", action="store_true", help="compare return-matched qualified runs")
    parser.add_argument("--return-tolerance", type=float, default=30.0)
    parser.add_argument("--sample-size", type=int, default=128, help="samples per occupancy fold and common probe size")
    args = parser.parse_args()
    summarize(args.directories, args.out, behavior=args.behavior,
              tolerance=args.return_tolerance, sample_size=args.sample_size)


if __name__ == "__main__":
    main()
