"""Frozen policies always receive raw observations."""
from . import ringtrack, walker


def load_policy(spec, seed=0):
    if spec["kind"] == "ring":
        return ringtrack.policy(spec)
    if spec["kind"] == "walker_bc":
        return walker.load_policy(spec)
    if spec["kind"] == "am10_bundle":
        from .am10 import load_policy as load_bundle
        return load_bundle(spec, seed=seed)
    raise ValueError(f"Unknown policy kind {spec['kind']}")
