"""Judge a saved ranking using only held-out closed-loop episodes."""
import numpy as np

from .data import read_json


def score(ranking_path, truth_path):
    run = read_json(ranking_path)
    panel = read_json(truth_path)["requests"][run["request"]]
    h, gamma = run["horizon"], run["gamma"]
    discount = gamma ** np.arange(h, dtype=float)
    rows = {}
    names = list(run["pool"])
    if "__expert__" in panel:
        names.append("__expert__")
    for name in names:
        returns = []
        for path in panel[name]:
            with np.load(path, allow_pickle=False) as f:
                r = f["rewards"].ravel()[:h]
                if len(r) < h and not ("terminated" in f and f["terminated"].ravel()[-1]):
                    raise ValueError(f"{path} is incomplete for horizon {h}")
                returns.append(float(r @ discount[:len(r)]))
        rows[name] = returns
    values = {name: float(np.mean(v)) for name, v in rows.items()}
    best = max(values[name] for name in run["pool"])
    recalled = max(values[name] for name in run["shortlist"])
    chosen = values.get(run["selected"])
    setting = {"task": run.get("task"), "split": run.get("split"), "method": run["method"],
               "retrieval": run.get("retrieval"), "horizon": h, "gamma": gamma,
               "pool_size": len(run["pool"]), "k": len(run["shortlist"]),
               "user_budget": run.get("episode_count", len(run.get("episode_indices", []))),
               "user_prefix": run.get("user_prefix"), "source_prefix": run.get("source_prefix"),
               "source_budget": run.get("source_budget"), "device": run.get("device"),
               "steps": run.get("steps"), "query_kind": run.get("query_kind"),
               "on_invalid": run.get("on_invalid"), "folds": run.get("folds")}
    return {"request": run["request"], "method": run["method"], "selected": run["selected"],
            "setting": setting, "status": run.get("status", "ok"),
            "chosen_return": chosen, "request_seconds": run.get("seconds", {}).get("total"),
            "recall_loss": best - recalled, "rerank_loss": None if chosen is None else recalled - chosen,
            "total_regret": None if chosen is None else best - chosen,
            "expert_gap": values["__expert__"] - chosen if chosen is not None and "__expert__" in values else None,
            "mean_returns": values, "episode_returns": rows}


def summarize(paths):
    """Summarize repeated requests from one task and one fixed experimental setting."""
    scored = [read_json(path) for path in paths]
    assignments = {tuple(sorted(r["setting"].items())) for r in scored}
    if len(assignments) != 1:
        raise ValueError("summarize one task and experimental setting at a time")
    groups = {}
    for r in scored:
        groups.setdefault(r["request"], []).append(r)
    fields = ("chosen_return", "recall_loss", "rerank_loss", "total_regret", "expert_gap")
    by_request = []
    for batch in groups.values():
        valid = [r for r in batch if r["selected"] is not None]
        means = {}
        for field in fields:
            vals = [r[field] for r in valid if r.get(field) is not None]
            means[field] = float(np.mean(vals)) if vals else None
        by_request.append(means)
    result = {}
    for field in fields:
        vals = [g[field] for g in by_request if g[field] is not None]
        result[field] = float(np.mean(vals)) if vals else None
    times = [r["request_seconds"] for r in scored]
    return {"setting": scored[0]["setting"], "requests": len(groups), "attempts": len(scored),
            "estimator_ok": sum(r["status"] == "ok" for r in scored),
            "fallbacks": sum(r["status"] == "raw_fallback" for r in scored),
            "selections": sum(r["selected"] is not None for r in scored),
            "valid_requests": sum(g["chosen_return"] is not None for g in by_request),
            "mean_on_selected_requests": result, "wall_seconds_p50": float(np.median(times)),
            "wall_seconds_p95": float(np.percentile(times, 95))}
