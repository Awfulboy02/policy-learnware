"""Train and qualify a source-conditioned native Brax SAC policy."""
import argparse
import json
from functools import partial
from pathlib import Path
from time import monotonic

from .data import write_json
from .expansion import FLOORS, evaluate, panel_seeds, qualify
from .expansion_env import create_env, native_env, source_bounds, source_feature
from .policies import load_policy


ARCHIVE = Path("/share/songyf/RL_Learnware/KMEPoA_assets/derived/sac-dmc2-v1/runs")
MILLION = 1_000_000


def source_environment(task, source):
    import jax
    from mujoco_playground._src.wrapper import Wrapper

    env = native_env(task)
    feature = source_feature(task, env)
    low, high = source_bounds(task, source)

    class SourceReset(Wrapper):
        def reset(self, key):
            def draw(rng):
                rng, subkey = jax.random.split(rng)
                return rng, self.env.reset(subkey)

            def outside(carry):
                value = feature(carry[1])
                return (value < low) | (value >= high)

            return jax.lax.while_loop(outside, lambda carry: draw(carry[0]), draw(key))[1]

    return SourceReset(env)


def training_recipe(task, budget):
    # Reacher has no archived recipe here. Its pilot explicitly tries the Cheetah settings.
    origin = "CheetahRun" if task == "ReacherEasy" else task
    template = ARCHIVE / f"sac__{origin}__seed0" / "run.json"
    archived = json.loads(template.read_text())
    epochs = max(1, (budget + MILLION - 1) // MILLION)
    config = dict(archived["config"], num_timesteps=budget, num_evals=epochs + 1)
    targets = [x * MILLION for x in (2, 3, 4, 5, 8, 10) if x * MILLION < budget]
    return dict(recipe_path=str(template), recipe_task=origin, config=config,
                network_factory=archived["network_factory"], selection_budget_steps=targets + [budget])


def train_source(task, source, seed, out, budget):
    from brax.training.agents.sac import networks, train
    from mujoco_playground import wrapper

    recipe = training_recipe(task, budget)
    recipe.update(task=task, source=source, seed=seed, algorithm="sac", initialization="random",
                  source_bounds=source_bounds(task, source), deployment="deterministic")
    out.mkdir(parents=True, exist_ok=True)
    with (out / "recipe.json").open("x") as stream:
        json.dump(recipe, stream, indent=2)

    env = source_environment(task, source)
    net = partial(networks.make_sac_networks, **recipe["network_factory"])
    started = monotonic()

    def progress(step, metrics):
        row = dict(env_steps=int(step), mean_return=float(metrics["eval/episode_reward"]),
                   elapsed_seconds=monotonic() - started)
        with (out / "train_curve.jsonl").open("a") as stream:
            print(json.dumps(row), file=stream)
        print(json.dumps(row), flush=True)

    train.train(environment=env, network_factory=net, wrap_env_fn=wrapper.wrap_for_brax_training,
                seed=seed, checkpoint_logdir=str(out / "checkpoints"), progress_fn=progress,
                **recipe["config"])


def evaluate_source(task, source, seed, out):
    started = monotonic()
    recipe = json.loads((out / "recipe.json").read_text())
    if (task, source, seed) != (recipe["task"], recipe["source"], recipe["seed"]):
        raise ValueError("evaluation task, source and seed must match the trained policy")
    elapsed = {row["env_steps"]: row["elapsed_seconds"] for row in map(
        json.loads, (out / "train_curve.jsonl").read_text().splitlines())}
    checkpoints = {int(p.name): p for p in (out / "checkpoints").iterdir() if p.name.isdigit()}
    val_seeds = panel_seeds(1, 4)
    env = create_env(task, source, len(val_seeds), seed)
    slices = []
    for target in recipe["selection_budget_steps"]:
        step = min(checkpoints, key=lambda n: abs(n - target))
        if abs(step - target) > 0.5 * MILLION:
            raise ValueError(f"missing SAC checkpoint near {target} steps")
        spec = dict(kind="brax_checkpoint", algorithm="sac", path=str(checkpoints[step]),
                    deterministic=True)
        evidence, _ = evaluate(load_policy(spec), env, val_seeds)
        slices.append(dict(**evidence, requested_steps=target, env_steps=step,
                           training_seconds=elapsed[step], policy=spec))
        write_json(out / "curve.json", slices)
        print(f"sac {task} {source} step={step} validation={evidence['mean_return']:.2f}", flush=True)

    best = max(slices, key=lambda item: item["mean_return"])
    audit, episodes, behavior = qualify(load_policy(best["policy"]), task, source, seed, out)
    training_seconds = slices[-1]["training_seconds"]
    evaluation_seconds = monotonic() - started
    write_json(out / "result.json", dict(
        id=out.name, task=task, source=source, algorithm="sac", seed=seed, device="cuda",
        selected_step=best["env_steps"], slices=slices, policy=best["policy"], qualification=audit,
        episodes=episodes, behavior_episodes=behavior, training_seconds=training_seconds,
        evaluation_seconds=evaluation_seconds, wall_seconds=training_seconds + evaluation_seconds))
    print(f"selected={best['env_steps']} qualification={audit['mean_return']:.2f} passed={audit['passed']}",
          flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("train", "evaluate"))
    parser.add_argument("--task", choices=FLOORS, required=True)
    parser.add_argument("--source", choices=("s0", "s1"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--budget", type=int, default=5_000_000)
    args = parser.parse_args()
    out = args.out.resolve()
    if args.mode == "train":
        train_source(args.task, args.source, args.seed, out, args.budget)
    else:
        evaluate_source(args.task, args.source, args.seed, out)


if __name__ == "__main__":
    main()
