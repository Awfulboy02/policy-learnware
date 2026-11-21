"""Train one independent source policy and evaluate frozen budget snapshots."""
import argparse
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np
import torch

from .data import write_json
from .expansion_env import create_env
from .policies import load_policy


FLOORS = {
    "WalkerWalk": (900., 900.),
    "CheetahRun": (828.6613473432138, 827.245831595175),
    "ReacherEasy": (869.1750000000001, 858.2062500000001),
}


def panel_seeds(panel, count, seed=0):
    return np.random.SeedSequence([71523, panel, seed]).generate_state(count).tolist()


def evaluate(policy, env, seeds, directory=None, *, record_speed=False):
    """Collect complete episodes on fixed resets; keep terminal successors."""
    obs = env.reset(seeds)
    active = np.ones(len(seeds), bool)
    returns = np.zeros(len(seeds), np.float64)
    lengths = np.zeros(len(seeds), int)
    records = []
    speeds = []
    for _ in range(env.horizon):
        actions = policy(obs)
        nxt, rewards, terminated, truncated = env.step(actions)
        returns[active] += rewards[active]
        lengths += active
        if directory is not None:
            records.append((obs, actions, rewards, nxt, terminated, truncated))
            if record_speed:
                speeds.append(env.forward_speed())
        active &= ~(terminated | truncated)
        if not active.any():
            break
        obs = nxt
    paths = []
    if directory is not None:
        directory.mkdir()
        columns = [np.stack(col) for col in zip(*records)]
        keys = ("observations", "actions", "rewards", "next_observations", "terminated", "truncated")
        for i, length in enumerate(lengths):
            path = directory / f"episode_{i:03}.npz"
            data = {k: v[:length, i] for k, v in zip(keys, columns)}
            if record_speed:
                data.update(forward_speed=np.asarray(speeds)[:length, i], control_dt=env.control_dt)
            np.savez_compressed(path, **data)
            paths.append(str(path))
    return {"returns": returns.tolist(), "mean_return": float(returns.mean()),
            "lengths": lengths.tolist(), "seeds": seeds}, paths


def qualify(policy, task, source, seed, out, *, qualification_episodes=16,
            spec_episodes=16, behavior_episodes=4):
    """Audit one selected snapshot, then export disjoint source data when it passes."""
    seeds = panel_seeds(2, qualification_episodes)
    audit_env = create_env(task, source, len(seeds), seed)
    audit, _ = evaluate(policy, audit_env, seeds)
    floor = FLOORS[task][int(source[-1])]
    audit.update(floor=floor, passed=audit["mean_return"] >= floor)
    if not audit["passed"]:
        return audit, [], []

    def collect(seeds, directory, record_speed=False):
        # Publish only complete sets of episodes. An interrupted native evaluation can retry.
        if not directory.is_dir():
            env = create_env(task, source, len(seeds), seed)
            with TemporaryDirectory(dir=out) as tmp:
                pending = Path(tmp) / directory.name
                evaluate(policy, env, seeds, pending, record_speed=record_speed)
                pending.rename(directory)
        return [str(directory / f"episode_{i:03}.npz") for i in range(len(seeds))]

    episodes = collect(panel_seeds(3, spec_episodes, seed), out / "episodes")
    behavior = []
    if behavior_episodes:
        behavior = collect(panel_seeds(4, behavior_episodes), out / "behavior",
                           record_speed=task == "CheetahRun")
    return audit, episodes, behavior


def run(args):
    from . import expansion_offpolicy, expansion_sacppo

    if min(args.num_envs, args.width, args.batch_size, args.learning_rate, args.rollout_steps,
           args.validation_episodes, args.qualification_episodes, args.spec_episodes) <= 0:
        raise ValueError("batch sizes, episode counts and learning rate must be positive")
    if not args.steps or args.steps[0] < 1 or sorted(set(args.steps)) != args.steps:
        raise ValueError("steps must contain increasing positive cumulative budgets")
    if args.behavior_episodes < 0 or args.warmup < 0:
        raise ValueError("behavior episode count and warmup must be nonnegative")
    requested = args.steps
    unit = args.num_envs * (args.rollout_steps if args.algorithm == "ppo" else 1)
    args.steps = sorted({((step + unit - 1) // unit) * unit for step in requested})
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    floor = FLOORS[args.task][int(args.source[-1])]
    recipe = {k: v for k, v in vars(args).items() if k != "out"}
    write_json(out / "recipe.json", {**recipe, "requested_steps": requested,
                                     "source_floor": floor, "deployment": "deterministic"})
    torch.set_num_threads(1)
    started = perf_counter()
    env = create_env(args.task, args.source, args.num_envs, args.seed)
    valid = create_env(args.task, args.source, args.validation_episodes, args.seed)
    val_seeds = panel_seeds(1, args.validation_episodes)
    probe = torch.tensor(valid.reset(val_seeds), dtype=torch.float32)
    slices = []
    slice_seconds = 0.0
    training_started = perf_counter()

    def snapshot(steps, actor):
        nonlocal slice_seconds
        if args.device.startswith("cuda"):
            torch.cuda.synchronize(args.device)
        now = perf_counter()
        training_seconds = now - training_started - slice_seconds
        path = out / f"actor-{steps}.pt"
        frozen = deepcopy(actor).cpu().eval()
        with torch.inference_mode():
            torch.jit.trace(frozen, probe).save(str(path))
        spec = {"kind": "frozen_torch", "path": str(path)}
        policy = load_policy(spec)
        with torch.inference_mode():
            np.testing.assert_allclose(policy(probe.numpy()), frozen(probe).numpy(), atol=1e-6)
        evidence, _ = evaluate(policy, valid, val_seeds)
        slices.append({**evidence, "env_steps": steps, "policy": spec,
                       "training_seconds": training_seconds})
        write_json(out / "curve.json", slices)
        print(f"{args.algorithm} {args.source} step={steps} validation={evidence['mean_return']:.2f}",
              flush=True)
        slice_seconds += perf_counter() - now

    trainer = expansion_sacppo if args.algorithm in ("sac", "ppo") else expansion_offpolicy
    updates = trainer.train(args, env, snapshot)
    training_seconds = perf_counter() - training_started - slice_seconds
    best = max(slices, key=lambda row: row["mean_return"])
    policy = load_policy(best["policy"])
    audit, episodes, behavior = qualify(
        policy, args.task, args.source, args.seed, out,
        qualification_episodes=args.qualification_episodes, spec_episodes=args.spec_episodes,
        behavior_episodes=args.behavior_episodes)
    result = {"id": out.name, "task": args.task, "source": args.source,
              "algorithm": args.algorithm, "seed": args.seed, "device": args.device,
              "selected_step": best["env_steps"], "slices": slices, "updates": updates,
              "policy": best["policy"], "qualification": audit, "episodes": episodes,
              "behavior_episodes": behavior,
              "training_seconds": training_seconds, "wall_seconds": perf_counter() - started}
    write_json(out / "result.json", result)
    print(f"selected={best['env_steps']} qualification={audit['mean_return']:.2f} passed={audit['passed']}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", choices=FLOORS, required=True)
    p.add_argument("--source", choices=["s0", "s1"], required=True)
    p.add_argument("--algorithm", choices=["sac", "td3", "ppo", "d4pg"], required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--steps", nargs="+", type=int, required=True,
                   help="cumulative environment steps, rounded up to complete batches")
    p.add_argument("--num-envs", type=int, default=16)
    p.add_argument("--rollout-steps", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--warmup", type=int, default=2048)
    p.add_argument("--width", type=int, default=256)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--validation-episodes", type=int, default=4)
    p.add_argument("--qualification-episodes", type=int, default=16)
    p.add_argument("--spec-episodes", type=int, default=16)
    p.add_argument("--behavior-episodes", type=int, default=4, help="separate common-reset diagnostic episodes per qualified policy")
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", type=Path, required=True)
    run(p.parse_args())


if __name__ == "__main__":
    main()
