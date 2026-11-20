import argparse
from pathlib import Path

import numpy as np

from . import ringtrack, walker
from .data import write_json
from .pipeline import build_catalog, rank


def main(argv=None):
    parser = argparse.ArgumentParser(description="Policy Learnware v0.1")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare", help="index source, user and judge data separately")
    p.add_argument("benchmark", choices=["ringtrack", "walker"])
    p.add_argument("--archive", type=Path, help="existing b_v2_data_20260909 directory")
    p.add_argument("--out", type=Path, required=True)

    p = sub.add_parser("prepare-am10", help="audit frozen source assets and index independent target panels")
    p.add_argument("--experiments", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)

    p = sub.add_parser("specify", help="build source specifications and public catalog")
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--settings", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--size", type=int, default=32)
    p.add_argument("--iterations", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--source-budget", type=int, help="whole post-qualification episodes per upload")
    p.add_argument("--source-prefix", type=int, help="visible source transitions per trajectory")

    p = sub.add_parser("rank", help="rank a single request without judge data")
    p.add_argument("--catalog", type=Path, required=True)
    p.add_argument("--requests", type=Path, required=True)
    p.add_argument("--request", required=True)
    p.add_argument("--k", type=int, default=2)
    p.add_argument("--method", choices=["rkme", "fqe", "am"], required=True)
    p.add_argument("--retrieval", choices=["rkme", "random", "all"], default="rkme")
    p.add_argument("--budget", type=int)
    p.add_argument("--user-prefix", type=int, help="visible transitions; deployment horizon stays fixed")
    p.add_argument("--pool-size", type=int, help="nested stratified subset of qualified candidates")
    p.add_argument("--pool-seed", type=int, default=0)
    p.add_argument("--on-invalid", choices=["abstain", "raw"], default="abstain")
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--empirical", action="store_true", help="local full query KME comparator")
    p.add_argument("--out", type=Path, required=True)

    p = sub.add_parser("full-pool", help="offline comparison on a discrete or log-probability permission panel")
    p.add_argument("--panel", type=Path, required=True)
    p.add_argument("--method", required=True, choices=[
        "fqe", "am", "is", "wis", "pdis", "wpdis", "dr", "wdr", "mrdr", "dualdice", "bestdice"])
    p.add_argument("--folds", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, required=True)

    p = sub.add_parser("score", help="open the held-out judge for a completed ranking")
    p.add_argument("--ranking", type=Path, required=True)
    p.add_argument("--truth", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)

    p = sub.add_parser("summarize", help="equal-request metrics and local latency for one setting")
    p.add_argument("--scores", nargs="+", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        if args.benchmark == "ringtrack":
            ringtrack.prepare(args.out)
            h, mean, scale = ringtrack.HORIZON, [49.5, 25.5, .5], [28.87, 15., .5]
        else:
            if args.archive is None:
                parser.error("walker requires --archive")
            walker.prepare(args.archive, args.out)
            h, mean, scale = 1000, [0.] * 30, [1.] * 30
        # A public transform per benchmark lets both sides fit their own specs independently.
        write_json(args.out / "settings.json", {"task": args.benchmark, "horizon": h, "gamma": 1.0,
                   "mean": mean, "scale": scale, "bandwidth": float(np.sqrt(len(mean)))})
    elif args.command == "prepare-am10":
        from .am10 import prepare
        prepare(args.experiments, args.out)
    elif args.command == "specify":
        build_catalog(args.source, args.settings, args.out, size=args.size,
                      iterations=args.iterations, seed=args.seed, source_budget=args.source_budget,
                      source_prefix=args.source_prefix)
    elif args.command == "rank":
        result = rank(args.catalog, args.requests, args.request, k=args.k, method=args.method,
                      retrieval=args.retrieval, budget=args.budget, steps=args.steps,
                      seed=args.seed, device=args.device, empirical=args.empirical, pool_size=args.pool_size,
                      pool_seed=args.pool_seed, user_prefix=args.user_prefix, on_invalid=args.on_invalid)
        write_json(args.out, result)
    elif args.command == "full-pool":
        from .panel import rank_panel
        write_json(args.out, rank_panel(args.panel, args.method, args.folds, args.seed))
    elif args.command == "summarize":
        from .evaluation import summarize
        write_json(args.out, summarize(args.scores))
    else:
        from .evaluation import score
        write_json(args.out, score(args.ranking, args.truth))


if __name__ == "__main__":
    main()
