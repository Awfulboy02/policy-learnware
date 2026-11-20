"""Judge a saved ranking using only held-out closed-loop episodes."""
import numpy as np

from .data import read_json


def score(ranking_path, truth_path):
    run = read_json(ranking_path)
    panel = read_json(truth_path)["requests"][run["request"]]
    h, gamma = run["horizon"], run["gamma"]
    discount = gamma ** np.arange(h, dtype=float)
    rows = {}
    for name in [*run["pool"], "__expert__"]:
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
    chosen = values[run["selected"]]
    return {"request": run["request"], "method": run["method"], "selected": run["selected"],
            "recall_loss": best - recalled, "rerank_loss": recalled - chosen,
            "total_regret": best - chosen, "expert_gap": values["__expert__"] - chosen,
            "mean_returns": values, "episode_returns": rows}
