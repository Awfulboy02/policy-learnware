import argparse
from pathlib import Path

import numpy as np

from . import ringtrack, walker
from .data import write_json
from .pipeline import build_catalog, rank


def main(argv=None):
    parser = argparse.ArgumentParser(description="Policy Learnware v0")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare", help="index source, user and judge data separately")
    p.add_argument("benchmark", choices=["ringtrack", "walker"])
    p.add_argument("--archive", type=Path, help="existing b_v2_data_20260909 directory")
    p.add_argument("--out", type=Path, required=True)

    p = sub.add_parser("specify", help="build source specifications and public catalog")
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--settings", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--size", type=int, default=32)
    p.add_argument("--iterations", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)

    p = sub.add_parser("rank", help="rank a single request without judge data")
    p.add_argument("--catalog", type=Path, required=True)
    p.add_argument("--requests", type=Path, required=True)
    p.add_argument("--request", required=True)
    p.add_argument("--k", type=int, default=2)
    p.add_argument("--method", choices=["rkme", "fqe", "am"], required=True)
    p.add_argument("--retrieval", choices=["rkme", "random", "all"], default="rkme")
    p.add_argument("--budget", type=int)
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--empirical", action="store_true", help="local full query KME comparator")
    p.add_argument("--out", type=Path, required=True)

    p = sub.add_parser("score", help="open the held-out judge for a completed ranking")
    p.add_argument("--ranking", type=Path, required=True)
    p.add_argument("--truth", type=Path, required=True)
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
        write_json(args.out / "settings.json", {"horizon": h, "gamma": 1.0,
                   "mean": mean, "scale": scale, "bandwidth": float(np.sqrt(len(mean)))})
    elif args.command == "specify":
        build_catalog(args.source, args.settings, args.out, size=args.size,
                      iterations=args.iterations, seed=args.seed)
    elif args.command == "rank":
        result = rank(args.catalog, args.requests, args.request, k=args.k, method=args.method,
                      retrieval=args.retrieval, budget=args.budget, steps=args.steps,
                      seed=args.seed, device=args.device, empirical=args.empirical)
        write_json(args.out, result)
    else:
        from .evaluation import score
        write_json(args.out, score(args.ranking, args.truth))


if __name__ == "__main__":
    main()
