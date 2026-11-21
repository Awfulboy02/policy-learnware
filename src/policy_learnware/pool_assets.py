"""Publish accepted source policies and build the three production RKME catalogs."""
import argparse
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np

from .data import load_episode, read_json, write_json
from .expansion import FLOORS, panel_seeds
from .pipeline import rank
from .policies import load_policy
from .rkme import Specification, specify


HORIZON = 1000
# These are the established AM10 interfaces: observation plus action dimensions.
FEATURE_DIMS = {"WalkerWalk": 30, "CheetahRun": 23, "ReacherEasy": 8}
SIZE, ITERATIONS = 32, 10


def settings(task):
    dim = FEATURE_DIMS[task]
    return {"task": task, "horizon": HORIZON, "gamma": 1.0, "mean": [0.0] * dim,
            "scale": [1.0] * dim, "bandwidth": float(np.sqrt(dim))}


def check_panel(row, panel, count):
    returns = row["returns"]
    if (row["seeds"] != panel_seeds(panel, count) or len(returns) != count
            or len(row["lengths"]) != count or not all(0 < n <= HORIZON for n in row["lengths"])
            or not np.isfinite(returns).all()
            or not np.isclose(np.mean(returns), row["mean_return"], rtol=0, atol=1e-6)):
        raise ValueError(f"invalid {count}-episode source panel")


def check_actions(policy, episodes):
    # Include different starts and later times; compare with logged deterministic actions.
    obs, actions = [], []
    for ep in episodes:
        ii = np.linspace(0, len(ep.obs) - 1, 3, dtype=int)
        obs.append(ep.obs[ii])
        actions.append(ep.actions[ii])
    expected = np.concatenate(actions)
    actual = np.asarray(policy(np.concatenate(obs)))
    if actual.shape != expected.shape or not np.isfinite(actual).all():
        raise ValueError("invalid frozen policy output")
    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-4)


def publish(run_dir, pool_dir):
    evidence = (run_dir / "result.json").resolve()
    row = read_json(evidence)
    task, source, algorithm, seed = (row[k] for k in ("task", "source", "algorithm", "seed"))
    if source not in ("s0", "s1") or algorithm not in ("sac", "d4pg"):
        raise ValueError(f"unexpected production source or algorithm: {source} {algorithm}")
    ident = f"{algorithm}-{source}-r{seed}"
    asset_dir = pool_dir / task / "assets" / ident
    marker = asset_dir / "asset.json"
    if marker.is_file():
        if read_json(marker)["evidence"] != str(evidence):
            raise ValueError(f"{ident}: independent training ID already belongs to another run")
        return marker

    floor = FLOORS[task][int(source[-1])]
    audit = row["qualification"]
    for item in row["slices"]:
        check_panel(item, 1, 4)
    selected = max(row["slices"], key=lambda item: item["mean_return"])
    check_panel(audit, 2, 16)
    if (not audit["passed"] or audit["mean_return"] < floor or audit["floor"] != floor
            or selected["env_steps"] != row["selected_step"] or selected["policy"] != row["policy"]):
        raise ValueError(f"{ident}: verification or snapshot selection failed")

    paths, heldout = row["episodes"], row["behavior_episodes"]
    if len(paths) != 16 or len(heldout) != 4 or len(set(paths + heldout)) != 20:
        raise ValueError(f"{ident}: expected 16 distinct specification and 4 behavior episodes")
    episodes = [load_episode(path, HORIZON) for path in paths + heldout]
    cfg = settings(task)
    for ep in episodes:
        if (len(ep.obs) > HORIZON or (len(ep.obs) < HORIZON and not ep.terminated[-1])
                or ep.obs.shape[1] + ep.actions.shape[1] != FEATURE_DIMS[task]
                or not all(np.isfinite(x).all() for x in (ep.obs, ep.actions, ep.rewards, ep.next_obs))):
            raise ValueError(f"{ident}: incomplete or invalid source episode")
    check_actions(load_policy(row["policy"]), episodes)
    started = perf_counter()
    spec = specify(episodes[:len(paths)], np.array(cfg["mean"]), np.array(cfg["scale"]),
                   horizon=HORIZON, gamma=cfg["gamma"], bandwidth=cfg["bandwidth"],
                   size=SIZE, iterations=ITERATIONS, seed=0)

    asset_dir.parent.mkdir(parents=True, exist_ok=True)
    # Rename a fully populated directory; no reader sees a partially accepted asset.
    with TemporaryDirectory(dir=asset_dir.parent) as tmp:
        stage = Path(tmp) / ident
        stage.mkdir()
        spec.save(stage / "spec.npz")
        restored = Specification.load(stage / "spec.npz")
        if (restored.bandwidth != cfg["bandwidth"] or not np.isfinite(restored.squared_norm)
                or not np.isfinite(restored.points).all() or not np.isfinite(restored.weights).all()):
            raise ValueError(f"{ident}: invalid saved RKME")
        asset = dict(id=ident, task=task, source=source, algorithm=algorithm, seed=seed,
                     policy=row["policy"], qualification=audit, selected_step=row["selected_step"],
                     evidence=str(evidence), episodes=paths, behavior_episodes=heldout,
                     spec=f"assets/{ident}/spec.npz", stratum=[source, algorithm],
                     episode_count=len(paths), specification_seconds=perf_counter() - started)
        write_json(stage / "asset.json", asset)
        try:
            stage.rename(asset_dir)
        except OSError:
            if not marker.is_file() or read_json(marker)["evidence"] != str(evidence):
                raise
    return marker


def catalog(pool_dir, task, *, preview):
    root = pool_dir / task
    rows = [read_json(p) for p in sorted((root / "assets").glob("*/asset.json"))]
    counts = Counter(row["source"] for row in rows)
    if not rows or (not preview and counts != {"s0": 100, "s1": 100}):
        raise ValueError(f"{task}: accepted counts {dict(counts)}")
    policies, sources = {}, {}
    for row in rows:
        ident = row["id"]
        if ident in policies or row["task"] != task:
            raise ValueError(f"duplicate or misplaced asset {ident}")
        if not all(Path(p).exists() for p in [row["policy"]["path"], root / row["spec"],
                                              row["evidence"], *row["episodes"], *row["behavior_episodes"]]):
            raise FileNotFoundError(f"{task}/{ident}: missing accepted artifact")
        policies[ident] = {k: row[k] for k in ("policy", "spec", "stratum", "episode_count",
                                               "specification_seconds")}
        sources[ident] = {k: row[k] for k in ("policy", "stratum", "episodes", "qualification")}
    cfg = settings(task)
    write_json(root / "settings.json", cfg)
    write_json(root / "source.json", {"policies": sources})
    catalog_path = root / "catalog.json"
    write_json(catalog_path, dict(settings=cfg, size=SIZE, iterations=ITERATIONS, source_prefix=HORIZON,
                                  source_budget=16, source_seed=0, policies=policies))
    # A held-out source diagnostic only exercises the normal retrieval/call path.
    query = rows[0]["behavior_episodes"][:1]
    with TemporaryDirectory(dir=root) as tmp:
        req = Path(tmp) / "request.json"
        write_json(req, {"requests": {"smoke": {"episodes": query, "split": "source_smoke"}}})
        hit = rank(catalog_path, req, "smoke", k=1, method="rkme")
    if hit["status"] != "ok" or not all(np.isfinite(v) for v in hit["distances"].values()):
        raise ValueError(f"{task}: retrieval failed")
    obs = load_episode(query[0], HORIZON).obs[:2]
    policy = load_policy(policies[hit["selected"]]["policy"])
    actions = np.asarray(policy(obs))
    if actions.shape != (len(obs), FEATURE_DIMS[task] - obs.shape[1]) or not np.isfinite(actions).all():
        raise ValueError(f"{task}: selected policy failed")
    return dict(task=task, source_counts=dict(counts), accepted=len(rows), smoke_selected=hit["selected"],
                specification_episodes=sum(len(r["episodes"]) for r in rows),
                behavior_episodes=sum(len(r["behavior_episodes"]) for r in rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["publish", "preview", "finalize"])
    parser.add_argument("--pool-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--task", choices=sorted(FLOORS))
    args = parser.parse_args()
    root = args.pool_dir.resolve()
    if args.command == "publish":
        if args.run_dir is None:
            parser.error("publish requires --run-dir")
        print(publish(args.run_dir, root), flush=True)
    elif args.command == "preview":
        if args.task is None:
            parser.error("preview requires --task")
        print(catalog(root, args.task, preview=True), flush=True)
    else:
        tasks = [catalog(root, task, preview=False) for task in sorted(FLOORS)]
        write_json(root / "acceptance.json", {"accepted": sum(t["accepted"] for t in tasks), "tasks": tasks})
        print(root / "acceptance.json", flush=True)


if __name__ == "__main__":
    main()
