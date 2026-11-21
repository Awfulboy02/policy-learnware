"""Frozen policies always receive raw observations."""
from . import ringtrack, walker


def load_policy(spec, seed=0):
    if spec["kind"] == "ring":
        return ringtrack.policy(spec)
    if spec["kind"] == "walker_bc":
        return walker.load_policy(spec)
    if spec["kind"] == "frozen_torch":
        import numpy as np
        import torch
        actor = torch.jit.load(spec["path"], map_location="cpu").eval()

        @torch.inference_mode()
        def act(obs):
            x = np.asarray(obs, dtype=np.float32)
            actions = actor(torch.tensor(np.atleast_2d(x))).numpy()
            return actions[0] if x.ndim == 1 else actions
        return act
    if spec["kind"] == "brax_checkpoint" and spec["algorithm"] == "sac":
        import jax
        import numpy as np
        from brax.training import checkpoint as common_checkpoint
        from brax.training.agents.sac import checkpoint, networks

        config = checkpoint.load_config(spec["path"])
        network = common_checkpoint.get_network(config, networks.make_sac_networks)
        policy = networks.make_inference_fn(network)(checkpoint.load(spec["path"]), deterministic=True)
        key = jax.random.PRNGKey(0)
        action = jax.jit(lambda observations: policy(observations, key)[0])

        def act(obs):
            x = np.asarray(obs, dtype=np.float32)
            actions = np.asarray(action(np.atleast_2d(x)))
            return actions[0] if x.ndim == 1 else actions
        return act
    if spec["kind"] == "am10_bundle":
        from .am10 import load_policy as load_bundle
        return load_bundle(spec, seed=seed)
    raise ValueError(f"Unknown policy kind {spec['kind']}")
