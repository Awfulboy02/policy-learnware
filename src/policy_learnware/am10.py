"""Index empirical source qualification and existing target panels in the AM10 archives."""
import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from .data import read_json, write_json

TASKS = ("WalkerWalk", "CheetahRun", "ReacherEasy")


def _floors(root):
    floors = {"torso_xx_negative": 900., "torso_xx_nonnegative": 900.}
    for directory in ("cheetah/repair_e1000", "reacher"):
        calibration = root / "b_v2_multienv_20260910" / directory / "data/SOURCE_CALIBRATION.json"
        floors.update(read_json(calibration)["source_floors"])
    return floors


def _source(asset, threshold):
    name = asset["asset_id"]
    records = sorted(asset["source_return_records"], key=lambda r: r["binding"]["path"])
    paths = [Path(r["binding"]["path"]) for r in records]
    if (len(paths) != 8 or len(set(paths)) != 8 or any(not p.is_file() for p in paths)
            or any(r["generator_policy_id"] != name for r in records)):
        raise ValueError(f"{name} lacks 8 distinct final-policy episodes")
    bundle = Path(asset["runtime"]["policy"]["bundle"])
    meta = read_json(bundle)
    if meta["policy_id"] != name or not (bundle.parent / meta["parameters"]).is_file():
        raise ValueError(f"{name} lacks a matching frozen bundle")
    returns = []
    for p in paths[:4]:
        with np.load(p, allow_pickle=False) as e:
            returns.append(float(e["rewards"].sum(dtype=float)))
    return dict(source=asset["source_id"], qualification=list(map(str, paths[:4])),
                episodes=list(map(str, paths[4:])), returns=returns, mean_return=float(np.mean(returns)),
                floor=threshold, passed=float(np.mean(returns)) >= threshold,
                strict_certificate=asset["source_certificate"],
                policy={"kind": "am10_bundle", "path": str(bundle),
                        "mode": meta["deployment_profile"]["mode"]})


def _panel(data, request, candidate):
    path = data / "receipts/cells" / f"{request}__effect__{candidate}" / "latest.json"
    if not path.exists():
        return []
    receipt = read_json(path)
    if (receipt["status"] != "completed" or receipt["completed"] != 8
            or receipt["asset_id"] != candidate or receipt["task_group"] != request):
        return []
    paths = sorted(Path(receipt["attempt_dir"]).glob("episode_*.npz"))
    return list(map(str, paths)) if len(paths) == 8 else []


def _requests(root, task):
    # B-v3.1 train entries reference the earlier B-v3 groups. Index them only once.
    requests = {}
    locations = {}
    for campaign, splits in (("am_b_v3_v1", ("train", "validation", "test")),
                             ("am_b_v3_1_v1", ("validation", "test"))):
        data = root / campaign / "data" / task
        for original in splits:
            for g in read_json(data / f"groups_{original}.json"):
                name = g["task_group"]
                paths = [r["binding"]["path"] for r in g["records"]]
                if len(set(paths)) != 16 or not all(Path(p).is_file() for p in paths):
                    raise ValueError(f"{name} lacks its 16 user query episodes")
                split = "development" if campaign == "am_b_v3_v1" else original
                requests[name] = {"episodes": paths, "split": split, "archived_split": original}
                locations[name] = data
    return requests, locations


def prepare(root, output):
    """Compute quality from source rewards; target files are indexed by receipt and filename."""
    root, output = Path(root), Path(output)
    index = read_json(root / "am10_expansion_900_v1/catalog_index.json")["policies"]
    old = {row["logical_id"] for row in read_json(root / "am10_expansion_v1/completion_summary.json")["rows"]}
    floors = _floors(root)
    summary = {}
    for task in TASKS:
        out = output / task
        requests, locations = _requests(root, task)
        truth = {name: {} for name in requests}
        audit, entries, scored = {}, {}, {}
        dims = None
        for item in (i for i in index if i["task"] == task):
            name = item["asset_id"]
            historical = item["logical_id"] in old
            row = {"asset_id": name, "source": item["source_id"], "algorithm": item["algorithm"],
                   "capacity": item["capacity"], "training_seed": item["training_seed"],
                   "in_historical_360": historical}
            audit[name] = row
            try:
                asset = read_json(item["asset_ref"])
                if asset["asset_id"] != name:
                    raise ValueError(f"source identity mismatch at {item['asset_ref']}")
                dims = asset["interface"]["observation_dim"] + asset["interface"]["action_dim"]
                row.update(_source(asset, floors[item["source_id"]]))
            except (OSError, ValueError, KeyError) as e:
                row.update(passed=None, reason=str(e))
                continue
            if not row["passed"]:
                continue
            entry = {key: row[key] for key in ("policy", "episodes", "qualification")}
            entry["stratum"] = [row["source"], row["algorithm"]]
            entries[name] = entry
            if historical:
                panels = {q: _panel(loc, q, name) for q, loc in locations.items()}
                row["complete_panels"] = sum(bool(v) for v in panels.values())
                if all(panels.values()):
                    scored[name] = entry
                    for q, paths in panels.items():
                        truth[q][name] = paths
        if dims is None:
            raise ValueError(f"no archived interface for {task}")
        counts = {}
        for scope, rows in (("900", list(audit.values())),
                            ("360", [r for r in audit.values() if r["in_historical_360"]])):
            counts[scope] = {"available": len(rows), "passed": sum(r["passed"] is True for r in rows),
                             "failed": sum(r["passed"] is False for r in rows),
                             "unknown": sum(r["passed"] is None for r in rows),
                             "per_source_passed": dict(Counter(r["source"] for r in rows if r["passed"])),
                             "per_algorithm_passed": dict(Counter(r["algorithm"] for r in rows if r["passed"]))}
        counts["qualified_with_all_panels"] = len(scored)
        counts["conditions"] = dict(Counter(r["split"] for r in requests.values()))
        summary[task] = counts
        for file, data in (("audit", audit), ("source", {"policies": scored}),
                           ("source_all", {"policies": entries}), ("requests", {"requests": requests}),
                           ("truth", {"requests": truth}), ("settings", {"task": task, "horizon": 1000, "gamma": 1.0,
                            "mean": [0.0]*dims, "scale": [1.0]*dims, "bandwidth": float(np.sqrt(dims))})):
            write_json(out / f"{file}.json", data)
    write_json(output / "summary.json", {"floors": floors, "tasks": summary})
    return summary


def load_policy(spec, seed=0):
    from .am10_policy import load
    return load(spec["path"], np.random.default_rng(seed))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiments", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    for task, counts in prepare(args.experiments, args.out).items():
        print(task, counts["900"]["passed"], counts["qualified_with_all_panels"])
