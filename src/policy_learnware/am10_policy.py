"""Inference kernels for AM10 frozen JAX bundles, including BCQ's selection critic."""
from pathlib import Path
import numpy as np
from .data import read_json


def load(path, rng):
    import jax
    import jax.numpy as jnp

    path = Path(path)
    meta = read_json(path)
    with np.load(path.parent / meta["parameters"], allow_pickle=False) as archive:
        def unpack(node):
            if "dict" in node:
                return {k: unpack(v) for k, v in node["dict"].items()}
            if "array" in node:
                return jnp.asarray(archive[node["array"]])
            return tuple(unpack(v) for v in next(iter(node.values())))
        params = unpack(meta["parameter_structure"])
        mean = jnp.asarray(archive["observation_mean"])
        scale = jnp.asarray(archive["observation_scale"])
    kind, cfg = meta["kind"], meta["settings"]

    def mlp(layers, x):
        for layer in layers[:-1]:
            x = jax.nn.relu(x @ layer["w"] + layer["b"])
        return x @ layers[-1]["w"] + layers[-1]["b"]

    def action(obs, seed):
        x = (obs - mean) / scale
        key = jax.random.PRNGKey(seed)
        if kind == "bcq":
            states = jnp.broadcast_to(x, (cfg["eval_candidates"], len(x)))
            z = jax.random.normal(key, (len(states), cfg["latent_dim"]))
            z = jnp.clip(z, -cfg["latent_clip"], cfg["latent_clip"])
            concat = lambda y: jnp.concatenate([states, y], axis=-1)
            decoded = jnp.tanh(mlp(params["decoder"], concat(z)))
            residual = cfg["phi"] * jnp.tanh(mlp(params["perturbation"], concat(decoded)))
            candidates = jnp.clip(decoded + residual, -1., 1.)
            scores = mlp(params["selection_q"], concat(candidates))[:, 0]
            return candidates[jnp.argmax(scores)]
        output = mlp(params["net"], x)
        if kind == "deterministic":
            return jnp.tanh(output)
        if kind == "tanh_gaussian":
            loc, log_std = jnp.split(output, 2)
        elif kind == "gaussian":
            loc, log_std = jnp.tanh(output), params["log_std"]
        else:
            raise ValueError(f"Unknown AM10 actor kind {kind}")
        sample = loc + jnp.exp(jnp.clip(log_std, -5., 2.)) * jax.random.normal(key, loc.shape)
        return jnp.tanh(sample) if kind == "tanh_gaussian" else jnp.clip(sample, -1., 1.)

    actions = jax.jit(lambda observations, seeds: jax.lax.map(lambda pair: action(*pair),
                                                               (observations, seeds)))

    def batch(observations):
        seeds = rng.integers(0, 2**32, size=len(observations), dtype=np.uint32)
        return np.asarray(actions(jnp.asarray(observations, dtype=jnp.float32), seeds))
    return batch
