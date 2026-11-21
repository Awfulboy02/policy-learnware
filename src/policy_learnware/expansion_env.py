"""Batched MuJoCo Playground tasks with the archived source reset distributions."""
from functools import lru_cache

import numpy as np


def source_bounds(task, source):
    limit = {"CheetahRun": np.pi, "WalkerWalk": 1.000001, "ReacherEasy": 1.0}[task]
    return {"s0": (-limit, 0.0), "s1": (0.0, limit)}[source]


def native_env(task):
    from mujoco_playground import registry

    config = registry.get_default_config(task)
    config.update({"action_repeat": 1, "vision": False, "episode_length": 1000})
    return registry.load(task, config=config)


def source_feature(task, env):
    if task == "ReacherEasy":
        target = int(env.mj_model.body("target").mocapid[0])
        return lambda state: state.data.mocap_pos[target, 0]
    index = {"CheetahRun": 3, "WalkerWalk": 0}[task]
    return lambda state: state.obs[index]


@lru_cache(maxsize=3)
def _native_task(task):
    import jax

    env = native_env(task)
    reset_feature = source_feature(task, env)

    def sample_source(seeds, lower, upper):
        def trial(i):
            state = env.reset(jax.random.PRNGKey(seeds[i]))
            feature = reset_feature(state)
            return i + 1, (lower <= feature) & (feature < upper), state

        def rejected(result):
            count, accepted, _ = result
            return (count < seeds.size) & ~accepted

        _, accepted, state = jax.lax.while_loop(rejected, lambda result: trial(result[0]), trial(0))
        return state, accepted

    reset_batch = jax.jit(jax.vmap(sample_source, in_axes=(0, None, None)))
    return env, reset_batch, jax.jit(jax.vmap(env.step))


class _SourceEnv:
    horizon = 1000

    def __init__(self, task, source, num_envs, seed):
        self.bounds = source_bounds(task, source)
        native, self._sample, self._step = _native_task(task)
        self.action_size = native.action_size
        self.control_dt = float(native.dt)
        self._speed_index = (int(native.mj_model.sensor("torso_subtreelinvel").adr[0])
                             if task == "CheetahRun" else None)
        self.num_envs = num_envs
        self._rng = np.random.default_rng(seed)
        self._elapsed = np.zeros(num_envs, dtype=int)

    def _source_states(self, seeds):
        # Historical reset uses the second child stream for native reset attempts.
        streams = [np.random.default_rng(np.random.SeedSequence(int(s)).spawn(2)[1]) for s in seeds]
        trials = np.stack([rng.integers(0, 2**32, size=64, dtype=np.uint32) for rng in streams])
        state, accepted = self._sample(trials, *self.bounds)
        if not np.asarray(accepted).all():
            raise RuntimeError("source reset exhausted 64 native attempts")
        return state

    def reset(self, seeds):
        if len(seeds) != self.num_envs:
            raise ValueError("provide one reset seed per environment")
        self._state = self._source_states(seeds)
        self._elapsed.fill(0)
        return np.asarray(self._state.obs)

    def step(self, actions):
        self._state = self._step(self._state, np.asarray(actions, dtype=np.float32))
        self._elapsed += 1
        native_done = np.asarray(self._state.done, dtype=bool)
        info = self._state.info
        cutoff = np.asarray(info.get("truncation", info.get("truncated", False)), dtype=bool)
        terminated = native_done & ~cutoff
        truncated = cutoff | ((self._elapsed >= self.horizon) & ~terminated)
        return np.asarray(self._state.obs), np.asarray(self._state.reward), terminated, truncated

    def forward_speed(self):
        """Signed torso subtree center-of-mass velocity along world x, in m/s."""
        if self._speed_index is None:
            raise ValueError("forward_speed is defined for CheetahRun")
        return np.asarray(self._state.data.sensordata)[:, self._speed_index]

    def reset_done(self, next_obs, done):
        import jax

        indices = np.flatnonzero(done)
        if not len(indices):
            return next_obs
        seeds = self._rng.integers(0, 2**32, size=len(indices), dtype=np.uint32)
        replacement = self._source_states(seeds)
        if len(indices) == self.num_envs:
            self._state = replacement
        else:
            self._state = jax.tree.map(
                lambda old, new: old.at[indices].set(new), self._state, replacement
            )
        self._elapsed[indices] = 0
        return np.asarray(self._state.obs)


def create_env(task, source, num_envs, seed):
    return _SourceEnv(task, source, num_envs, seed)
