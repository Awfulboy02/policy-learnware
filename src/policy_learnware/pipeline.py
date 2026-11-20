"""Offline simulation of separate upload, retrieval and local reuse stages."""
from pathlib import Path
from time import perf_counter
from tempfile import TemporaryDirectory

import numpy as np

from .data import load_episode, read_json, write_json
from .policies import load_policy
from .rkme import Specification, distance, specify


def build_catalog(source_path, settings_path, output, *, size=32, iterations=10, seed=0):
    sources = read_json(source_path)["policies"]
    if not sources:
        raise ValueError("empty qualified policy pool")
    cfg = read_json(settings_path)
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"use a new catalog directory: {out}")
    # Publish specs and their index together; concurrent builds cannot mix catalogs.
    with TemporaryDirectory(dir=out.parent) as temp:
        staged = Path(temp) / "catalog"
        staged.mkdir()
        entries = {}
        for i, (name, source) in enumerate(sorted(sources.items())):
            episodes = [load_episode(p, cfg["horizon"]) for p in source["episodes"]]
            t = perf_counter()
            spec = specify(episodes, np.array(cfg["mean"]), np.array(cfg["scale"]),
                           horizon=cfg["horizon"], gamma=cfg["gamma"], size=size,
                           bandwidth=cfg["bandwidth"], iterations=iterations, seed=seed)
            filename = f"spec-{i}.npz"
            spec.save(staged / filename)
            entries[name] = {"spec": filename, "policy": source["policy"],
                             "specification_seconds": perf_counter() - t}
        write_json(staged / "catalog.json", {"settings": cfg, "size": size, "iterations": iterations,
                                          "policies": entries})
        staged.rename(out)


def rank(catalog_path, requests_path, request, *, k, method, retrieval="rkme",
         budget=None, steps=4000, seed=0, device="cpu", empirical=False):
    catalog = read_json(catalog_path)
    cfg = catalog["settings"]
    h, gamma = cfg["horizon"], cfg["gamma"]
    entry = read_json(requests_path)["requests"][request]
    paths = entry["episodes"]
    n = len(paths) if budget is None else budget
    if not 1 <= n <= len(paths):
        raise ValueError(f"episode budget must be in [1, {len(paths)}]")
    rng = np.random.default_rng(seed)
    picked = rng.permutation(len(paths))[:n]
    pool = sorted(catalog["policies"])
    if retrieval != "all" and not 1 <= k <= len(pool):
        raise ValueError(f"k must be in [1, {len(pool)}]")
    if method == "rkme" and retrieval != "rkme":
        raise ValueError("RKME-only selection requires RKME retrieval")
    if method != "rkme" and steps < 1:
        raise ValueError("OPE requires positive optimization steps")

    start = perf_counter()
    episodes = [load_episode(paths[i], h) for i in picked]
    loaded = perf_counter()
    distances = {}
    if retrieval == "rkme":
        query = specify(episodes, np.asarray(cfg["mean"]), np.asarray(cfg["scale"]),
                        horizon=h, gamma=gamma, bandwidth=cfg["bandwidth"],
                        size=None if empirical else catalog["size"],
                        iterations=catalog["iterations"], seed=seed)
        # The retrieval service needs only specifications to compute this distance.
        for name, candidate in catalog["policies"].items():
            spec = Specification.load(Path(catalog_path).parent / candidate["spec"])
            distances[name] = distance(query, spec)
        shortlist = sorted(pool, key=lambda name: (distances[name], name))[:k]
    elif retrieval == "random":
        shortlist = rng.choice(pool, size=k, replace=False).tolist()
    elif retrieval == "all":
        shortlist = pool
    else:
        raise ValueError(f"unknown retrieval {retrieval}")
    retrieved = perf_counter()

    estimates = {}
    fit_time, candidate_times = 0.0, {}
    if method == "rkme":
        ranking = shortlist
    else:
        from .ope import fit_am, fit_fqe
        options = dict(steps=steps, seed=seed, device=device)
        if method == "am":
            t = perf_counter()
            model = fit_am(episodes, h, **options)
            if device.startswith("cuda"):
                import torch
                torch.cuda.synchronize(device)
            fit_time = perf_counter() - t
            initial = np.stack([e.obs[0] for e in episodes])
        for name in shortlist:
            t = perf_counter()
            policy = load_policy(catalog["policies"][name]["policy"])
            if method == "fqe":
                estimates[name] = fit_fqe(episodes, policy, h, gamma, **options)
            elif method == "am":
                estimates[name] = model.value(policy, initial, h, gamma)
            else:
                raise ValueError(f"unknown estimator {method}")
            candidate_times[name] = perf_counter() - t
        ranking = sorted(shortlist, key=lambda name: (-estimates[name], name))
    return {"request": request, "split": entry["split"], "pool": pool, "shortlist": shortlist,
            "selected": ranking[0], "ranking": ranking, "distances": distances,
            "estimates": estimates, "method": method, "retrieval": retrieval,
            "query_kind": ("empirical" if empirical else "reduced") if retrieval == "rkme" else None,
            "episode_indices": picked.tolist(), "seed": seed, "horizon": h, "gamma": gamma,
            "steps": steps if method != "rkme" else None, "device": device,
            "seconds": {"read_user_data": loaded - start, "retrieval": retrieved - loaded,
                        "am_fit": fit_time, "by_candidate": candidate_times,
                        "total": perf_counter() - start}}
