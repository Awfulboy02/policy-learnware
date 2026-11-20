"""Offline simulation of separate upload, retrieval and local reuse stages."""
from collections import defaultdict
from pathlib import Path
from time import perf_counter
from tempfile import TemporaryDirectory

import numpy as np

from .data import episode_prefix, load_episode, read_json, write_json
from .policies import load_policy
from .rkme import Specification, distance, specify


def _visible_length(prefix, horizon):
    length = horizon if prefix is None else prefix
    if not 1 <= length <= horizon:
        raise ValueError(f"prefix must be between 1 and {horizon}")
    return length


def _candidate_pool(entries, size, seed):
    if size is None or size == len(entries):
        return sorted(entries)
    if not 1 <= size <= len(entries):
        raise ValueError(f"pool size exceeds available qualified candidates ({len(entries)})")
    rng = np.random.default_rng(seed)
    groups = defaultdict(list)
    for name in sorted(entries):
        groups[str(entries[name].get("stratum", ""))].append(name)
    # Round-robin across shuffled strata, then within strata. Prefixes are nested.
    order = rng.permutation(sorted(groups))
    groups = {key: list(rng.permutation(groups[key])) for key in order}
    pool = []
    while len(pool) < size:
        for key in order:
            if groups[key]:
                pool.append(str(groups[key].pop()))
                if len(pool) == size:
                    break
    return sorted(pool)


def build_catalog(source_path, settings_path, output, *, size=32, iterations=10, seed=0,
                  source_budget=None, source_prefix=None):
    sources = read_json(source_path)["policies"]
    if not sources:
        raise ValueError("empty qualified policy pool")
    cfg = read_json(settings_path)
    visible = _visible_length(source_prefix, cfg["horizon"])
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
            paths = source["episodes"]
            if source_budget is not None:
                if not 1 <= source_budget <= len(paths):
                    raise ValueError(f"{name} has {len(paths)} specification episodes")
                order = np.random.default_rng(seed).permutation(len(paths))[:source_budget]
                paths = [paths[j] for j in order]
            episodes = [episode_prefix(load_episode(p, cfg["horizon"]), visible) for p in paths]
            t = perf_counter()
            spec = specify(episodes, np.array(cfg["mean"]), np.array(cfg["scale"]),
                           horizon=visible, gamma=cfg["gamma"], size=size,
                           bandwidth=cfg["bandwidth"], iterations=iterations, seed=seed)
            filename = f"spec-{i}.npz"
            spec.save(staged / filename)
            entries[name] = {"spec": filename, "policy": source["policy"],
                             "stratum": source.get("stratum", ""), "episode_count": len(paths),
                             "specification_seconds": perf_counter() - t}
        write_json(staged / "catalog.json", {"settings": cfg, "size": size, "iterations": iterations,
                                             "source_prefix": visible, "source_budget": source_budget, "source_seed": seed,
                                             "policies": entries})
        staged.rename(out)


def rank(catalog_path, requests_path, request, *, k, method, retrieval="rkme",
         budget=None, steps=4000, seed=0, device="cpu", empirical=False,
         user_prefix=None, pool_size=None, pool_seed=0, on_invalid="abstain"):
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
    pool = _candidate_pool(catalog["policies"], pool_size, pool_seed)
    visible = _visible_length(user_prefix, h)
    if on_invalid == "raw" and retrieval != "rkme":
        raise ValueError("Raw fallback requires RKME retrieval")
    if retrieval != "all" and not 1 <= k <= len(pool):
        raise ValueError(f"k must be in [1, {len(pool)}]")
    if method == "rkme" and retrieval != "rkme":
        raise ValueError("RKME-only selection requires RKME retrieval")
    if method != "rkme" and steps < 1:
        raise ValueError("OPE requires positive optimization steps")

    start = perf_counter()
    episodes = [episode_prefix(load_episode(paths[i], h), visible) for i in picked]
    loaded = perf_counter()
    distances = {}
    if retrieval == "rkme":
        query = specify(episodes, np.asarray(cfg["mean"]), np.asarray(cfg["scale"]),
                        horizon=visible, gamma=gamma, bandwidth=cfg["bandwidth"],
                        size=None if empirical else catalog["size"],
                        iterations=catalog["iterations"], seed=seed)
        # The retrieval service needs only specifications to compute this distance.
        for name in pool:
            candidate = catalog["policies"][name]
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
            policy = load_policy(catalog["policies"][name]["policy"], seed=seed)
            if method == "fqe":
                estimates[name] = fit_fqe(episodes, policy, h, gamma, **options)
            elif method == "am":
                estimates[name] = model.value(policy, initial, h, gamma)
            else:
                raise ValueError(f"unknown estimator {method}")
            candidate_times[name] = perf_counter() - t
        estimates = {name: value if np.isfinite(value) else None for name, value in estimates.items()}
        if any(value is None for value in estimates.values()):
            ranking = []
        else:
            ranking = sorted(shortlist, key=lambda name: (-estimates[name], name))
    selected = ranking[0] if ranking else (shortlist[0] if on_invalid == "raw" else None)
    status = "ok" if ranking else ("raw_fallback" if selected else "unavailable")
    return {"request": request, "split": entry["split"], "pool": pool, "shortlist": shortlist,
            "selected": selected, "status": status, "ranking": ranking, "distances": distances,
            "estimates": estimates, "method": method, "retrieval": retrieval,
            "query_kind": ("empirical" if empirical else "reduced") if retrieval == "rkme" else None,
            "episode_indices": picked.tolist(), "source_prefix": catalog.get("source_prefix", h),
            "user_prefix": visible, "source_budget": catalog.get("source_budget"),
            "task": cfg.get("task"), "on_invalid": on_invalid, "pool_seed": pool_seed, "seed": seed, "horizon": h, "gamma": gamma,
            "steps": steps if method != "rkme" else None, "device": device,
            "seconds": {"read_user_data": loaded - start, "retrieval": retrieved - loaded,
                        "am_fit": fit_time, "by_candidate": candidate_times,
                        "total": perf_counter() - start}}
