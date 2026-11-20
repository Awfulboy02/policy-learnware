"""Frozen policies always receive raw observations."""
from . import ringtrack, walker


def load_policy(spec):
    if spec["kind"] == "ring":
        return ringtrack.policy(spec)
    if spec["kind"] == "walker_bc":
        return walker.load_policy(spec)
    raise ValueError(f"Unknown policy kind {spec['kind']}")
