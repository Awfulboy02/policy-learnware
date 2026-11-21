"""Durable, single-coordinator queue for the three-task source policy pools."""
import argparse
from collections import Counter
from contextlib import contextmanager
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import shlex
import socket
import subprocess
import sys
import time

TASKS = ("CheetahRun", "WalkerWalk", "ReacherEasy")
SOURCES = ("s0", "s1")
CELLS = [(task, source) for task in TASKS for source in SOURCES]
ALGOS = ("sac", "d4pg")
REPO = Path(__file__).resolve().parents[1]
FIVE, TEN = 5_000_000, 10_000_000


def read(path):
    return json.loads(path.read_text())


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(obj, separators=(",", ":")) + "\n")
    tmp.replace(path)


@contextmanager
def locked(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def asset_marker(root, job):
    return root / job["task"] / "assets" / job["id"] / "asset.json"


def inventory(root, seen):
    for task in TASKS:
        directory = root / task / "assets"
        if not directory.exists():
            continue
        for d in directory.iterdir():
            key = (task, d.name)
            if key not in seen and (d / "asset.json").is_file():
                asset = read(d / "asset.json")
                seen[key] = (task, asset["source"], asset["algorithm"])
    return Counter(seen.values())


def environment():
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO / "src") + os.pathsep + env.get("PYTHONPATH", "")
    return env


def commands(root, j):
    out = root / "runs" / j["task"] / j["id"]
    common = ["--task", j["task"], "--source", j["source"], "--seed", str(j["seed"]), "--out", str(out)]

    def module(name):
        return [sys.executable, "-u", "-m", "policy_learnware." + name]

    if j["algorithm"] == "sac":
        return [
            (stage, module("production_sac") + [stage] + common + ["--budget", str(j["budget"])])
            for stage in ("train", "evaluate")
        ]
    return [
        (
            "train",
            module("expansion")
            + common
            + [
                "--algorithm",
                "d4pg",
                "--steps",
                "2000000",
                "3000000",
                "4000000",
                "5000000",
                "--num-envs",
                "16",
                "--batch-size",
                "128",
                "--width",
                "128",
                "--warmup",
                "1024",
            ],
        )
    ]


def execute(root, j, env):
    """Restart after a completed training or evaluation phase without repeating it."""
    out = root / "runs" / j["task"] / j["id"]
    result = out / "result.json"
    started = time.monotonic()
    log = root / "logs" / (j["key"] + ".log")
    log.parent.mkdir(exist_ok=True)
    stage = "start"
    with log.open("a", buffering=1) as stream:

        def launch(cmd):
            print("$", shlex.join(cmd), flush=True, file=stream)
            subprocess.run(cmd, cwd=REPO, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)

        try:
            if not result.exists():
                for stage, cmd in commands(root, j):
                    done = out / (stage + ".complete")
                    if done.exists():
                        continue
                    if stage == "train" and out.exists():
                        raise RuntimeError(f"incomplete training at {out}; preserving existing output")
                    launch(cmd)
                    done.touch()
            stage = "evaluation-output"
            r = read(result)
            if (r["task"], r["source"], r["algorithm"], r["seed"]) != tuple(
                j[k] for k in ("task", "source", "algorithm", "seed")
            ):
                raise ValueError("training identity differs from assignment")
            q = r["qualification"]
            if q["passed"]:
                stage = "publish"
                if not asset_marker(root, j).exists():
                    launch(
                        [
                            sys.executable,
                            "-u",
                            "-m",
                            "policy_learnware.pool_assets",
                            "publish",
                            "--run-dir",
                            str(out),
                            "--pool-dir",
                            str(root),
                        ]
                    )
                if not asset_marker(root, j).exists():
                    raise RuntimeError("publish returned without an asset marker")
            slices = r.get("slices", [])
            gain = slices[-1]["mean_return"] - slices[-2]["mean_return"] if len(slices) >= 2 else 0
            reply = dict(
                status="accepted" if q["passed"] else "rejected", score=q["mean_return"], last_gain=gain
            )
        except Exception as err:
            print(f"{stage}: {err}", file=stream, flush=True)
            reply = dict(status="error", stage=stage, error=str(err))
    return dict(**reply, seconds=time.monotonic() - started)


def gpu_query(gpu, query):
    return subprocess.check_output(
        ["nvidia-smi", "-i", str(gpu), query, "--format=csv,noheader,nounits"], text=True
    ).strip()


def worker(root, gpu):
    slot = socket.gethostname().split(".")[0] + "-" + str(gpu)
    qdir = root / "queue"
    beat = qdir / "workers" / (slot + ".json")
    inbox = qdir / "inbox" / (slot + ".json")
    env = environment()
    env.update(
        CUDA_VISIBLE_DEVICES=gpu_query(gpu, "--query-gpu=uuid"),
        XLA_PYTHON_CLIENT_PREALLOCATE="false",
        MUJOCO_GL="egl",
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        JAX_COMPILATION_CACHE_DIR=str(root / "jax-cache"),
    )
    cpus = sorted(os.sched_getaffinity(0))
    shard = cpus[gpu * len(cpus) // 8 : (gpu + 1) * len(cpus) // 8]
    if shard:
        os.sched_setaffinity(0, shard)
    with locked(qdir / "locks" / (slot + ".lock")):
        while not (qdir / "stop.json").exists():
            free = (
                not gpu_query(gpu, "--query-compute-apps=pid")
                and int(gpu_query(gpu, "--query-gpu=memory.used")) < 256
            )
            j = read(inbox) if inbox.exists() else None
            receipt = qdir / "outcomes" / (j["key"] + ".json") if j else None
            if free and j and not receipt.exists():
                write(beat, dict(state="running", time=time.time(), job=j["key"]))
                write(receipt, execute(root, j, env))
            else:
                write(beat, dict(state="ready" if free else "occupied", time=time.time()))
                time.sleep(5)
        write(beat, dict(state="stopped", time=time.time()))


def route(task, algo, history, inflight):
    """Return budget, available launch slots and reason for this task/source/algorithm."""
    errors = sum(j["status"] == "error" for j in history)
    if errors >= 2:
        return None, 0, "engineering errors (see logs)"
    rows = [j for j in history if j["status"] in ("accepted", "rejected")]
    if len(rows) >= 8 and not any(j["status"] == "accepted" for j in rows[-8:]):
        return None, 0, "8 unsuccessful audits"
    passed = sum(j["status"] == "accepted" for j in rows)
    if algo == "d4pg":
        if task != "CheetahRun" and not passed:
            if len(rows) >= 3:
                return None, 0, "0/3 pilot"
            quota = 1 if not rows else 3
            return FIVE, max(0, quota - len(rows) - inflight), "pilot"
        return FIVE, max(0, 2 - inflight), "production"
    if task == "CheetahRun":
        return FIVE, 20, "production"
    short = [j for j in rows if j["budget"] == FIVE]
    passed = sum(j["status"] == "accepted" for j in short)
    if len(short) < 3 or (len(short) < 6 and passed == 1):
        quota = 3 if len(short) < 3 else 6
        return FIVE, max(0, quota - len(short) - inflight), "pilot"
    if passed >= 2:
        return FIVE, 20, "production"
    if not any(j["last_gain"] > 0 for j in short):
        return None, 0, "5M pilot plateau; inspect source and curve"
    long = [j for j in rows if j["budget"] == TEN]
    if len(long) < 3:
        return TEN, max(0, 3 - len(long) - inflight), "extended pilot"
    if not any(j["status"] == "accepted" for j in long):
        return None, 0, "10M pilot below floor"
    return TEN, 20, "production"


def plan(jobs, assets, counts, rush=False):
    """Choose one job, recomputing remaining identities including assignments in flight."""
    options, reasons = [], {}
    active = [j for j in jobs if j["status"] == "assigned" and (j["task"], j["id"]) not in assets]
    for task, source in CELLS:
        local = [j for j in jobs if j["task"] == task and j["source"] == source]
        work = [j for j in active if j["task"] == task and j["source"] == source]
        accepted = {a: counts[task, source, a] for a in ALGOS}
        remaining = 100 - sum(accepted.values()) - len(work)
        if remaining <= 0:
            continue
        inflight = Counter(j["algorithm"] for j in work)
        histories = {a: [j for j in local if j["algorithm"] == a] for a in ALGOS}
        routes = {a: route(task, a, histories[a], inflight[a]) for a in ALGOS}
        costs = {}
        for a, h in histories.items():
            finished = [j for j in h if j["status"] in ("accepted", "rejected")]
            n = sum(j["status"] == "accepted" for j in finished)
            costs[a] = (
                sum(j["seconds"] for j in finished) / 3600 / n
                if n
                else (0.116 if a == "sac" else (0.59 if source == "s0" else 1.855))
            )
        failures = [j for j in histories["d4pg"] if j["status"] in ("accepted", "rejected")]
        low_yield = (
            len(failures) >= 6
            and sum(j["status"] == "accepted" for j in failures) / len(failures) <= 0.25
        )
        d_target = 5
        if routes["d4pg"][0] is None or ((rush or low_yield) and routes["sac"][2] == "production"):
            d_target = accepted["d4pg"] + inflight["d4pg"]
        if routes["sac"][0] is None:
            d_target = 100 - accepted["sac"] - inflight["sac"]
        needs = dict(
            d4pg=max(0, d_target - accepted["d4pg"] - inflight["d4pg"]),
            sac=max(0, 100 - d_target - accepted["sac"] - inflight["sac"]),
        )
        reasons[f"{task}/{source}"] = {a: routes[a][2] for a in ALGOS}
        task_jobs = [j for j in jobs if j["task"] == task]
        cell_weight = sum(needs[a] * costs[a] for a in ALGOS) / (1 + len(work))
        for a in ALGOS:
            budget, slots, phase = routes[a]
            if not (needs[a] and slots and budget):
                continue
            h = histories[a]
            # Give each task two distinct-source SACs plus one D4PG before opening extra slots.
            first_rank = (
                0
                if a == "sac" and not h
                else 1
                if a == "d4pg" and not any(j["algorithm"] == "d4pg" for j in task_jobs)
                else 2
            )
            boot = len(task_jobs) < 3
            score = (
                int(boot),
                -len(task_jobs) if boot else 0,
                -first_rank if boot else 0,
                int(phase in ("pilot", "extended pilot")),
                -len(h) if phase != "production" else 0,
                cell_weight,
                needs[a] * costs[a] / (1 + inflight[a]),
            )
            options.append((score, task, source, a, budget))
    if not options:
        return None, reasons
    _, task, source, algo, budget = max(options, key=lambda item: item[0])
    seed = 1001 + sum((j["task"], j["source"], j["algorithm"]) == (task, source, algo) for j in jobs)
    name = f"{algo}-{source}-r{seed}"
    return (
        dict(
            task=task,
            source=source,
            algorithm=algo,
            budget=budget,
            seed=seed,
            id=name,
            key=task + "-" + name,
            status="assigned",
        ),
        reasons,
    )


def coordinator(root, deadline):
    qdir = root / "queue"
    ledger = qdir / "jobs.json"
    stop = qdir / "stop.json"
    deadline_ts = datetime.fromisoformat(deadline).timestamp() if deadline else None
    with locked(qdir / "locks" / "coordinator.lock"):
        if stop.exists():
            print(read(stop), flush=True)
            return
        jobs = read(ledger) if ledger.exists() else []
        seen = {}
        while True:
            changed = False
            for j in jobs:
                if j["status"] == "assigned":
                    receipt = qdir / "outcomes" / (j["key"] + ".json")
                    if receipt.exists():
                        j.update(read(receipt))
                        changed = True
            counts = inventory(root, seen)
            slots = {p.stem: read(p) for p in (qdir / "workers").glob("*.json")}
            busy = {j["slot"]: j for j in jobs if j["status"] == "assigned"}
            live = [
                k
                for k, v in slots.items()
                if k in busy or (v["state"] == "ready" and time.time() - v["time"] < 75)
            ]
            # Fall back to the faster validated route when remaining GPU hours exceed the deadline.
            estimate = 0.0
            for task, src in CELLS:
                k = sum(counts[task, src, a] for a in ALGOS)
                want_d = min(100 - k, max(0, 5 - counts[task, src, "d4pg"]))
                for a, need in (("d4pg", want_d), ("sac", 100 - k - want_d)):
                    group = [
                        j
                        for j in jobs
                        if (j["task"], j["source"], j["algorithm"]) == (task, src, a)
                        and j["status"] in ("accepted", "rejected")
                    ]
                    n = sum(j["status"] == "accepted" for j in group)
                    hours = (
                        sum(j["seconds"] for j in group) / 3600 / n
                        if n
                        else (0.116 if a == "sac" else (0.59 if src == "s0" else 1.855))
                    )
                    estimate += need * hours
            rush = bool(
                deadline_ts
                and len(jobs) >= 16
                and estimate > max(0, deadline_ts - time.time()) / 3600 * len(live)
            )
            for slot in sorted(set(live) - busy.keys()):
                j, _ = plan(jobs, seen, counts, rush)
                if j is None:
                    break
                j.update(slot=slot, assigned_at=time.time())
                jobs.append(j)
                busy[slot] = j
                changed = True
            if changed:
                write(ledger, jobs)
            for slot, j in busy.items():
                inbox = qdir / "inbox" / (slot + ".json")
                if not inbox.exists() or read(inbox)["key"] != j["key"]:
                    write(inbox, j)
            possible, reasons = plan(jobs, seen, counts, rush)
            cells = {}
            for task, src in CELLS:
                rows = [j for j in jobs if (j["task"], j["source"]) == (task, src)]
                cells[task + "/" + src] = dict(
                    sac=counts[task, src, "sac"],
                    d4pg=counts[task, src, "d4pg"],
                    active=sum(j["status"] == "assigned" for j in rows),
                    rejected=sum(j["status"] == "rejected" for j in rows),
                    errors=sum(j["status"] == "error" for j in rows),
                )
            done = all(row["sac"] + row["d4pg"] == 100 for row in cells.values())
            state = "running" if busy or possible else "needs-review"
            if done and not busy:
                state = "finalizing"
            write(
                qdir / "status.json",
                dict(
                    state=state,
                    updated=time.time(),
                    cells=cells,
                    active={k: j["key"] for k, j in busy.items()},
                    routes=reasons,
                    rush=rush,
                    estimated_gpu_hours=round(estimate, 1),
                ),
            )
            if state == "finalizing":
                env = dict(
                    environment(),
                    JAX_PLATFORMS="cpu",
                    CUDA_VISIBLE_DEVICES="",
                    OPENBLAS_NUM_THREADS="1",
                    OMP_NUM_THREADS="1",
                )
                log = root / "logs" / "finalize.log"
                log.parent.mkdir(exist_ok=True)
                with log.open("a") as stream:
                    result = subprocess.run(
                        [
                            sys.executable,
                            "-u",
                            "-m",
                            "policy_learnware.pool_assets",
                            "finalize",
                            "--pool-dir",
                            str(root),
                        ],
                        cwd=REPO,
                        env=env,
                        stdout=stream,
                        stderr=subprocess.STDOUT,
                    )
                state = (
                    "complete"
                    if result.returncode == 0 and (root / "acceptance.json").exists()
                    else "finalize-error"
                )
            if state in ("complete", "needs-review", "finalize-error"):
                write(stop, dict(state=state, updated=time.time()))
                print(state, flush=True)
                return
            time.sleep(5)


def status(root):
    qdir = root / "queue"
    if not (qdir / "status.json").exists():
        print("Awaiting coordinator.")
        return
    s = read(qdir / "status.json")
    if (qdir / "stop.json").exists():
        s["state"] = read(qdir / "stop.json")["state"]
    print(
        s["state"],
        "| updated",
        datetime.fromtimestamp(s["updated"]).astimezone().isoformat(timespec="seconds"),
    )
    print("cell                 SAC   D4PG    accepted     in-flight   rejected   error")
    for k, row in s["cells"].items():
        total = row["sac"] + row["d4pg"]
        print(
            f"{k:20} {row['sac']:3}   {row['d4pg']:4}     {total:3}/100       "
            f"{row['active']:3}       {row['rejected']:4}    {row['errors']:3}"
        )
    print("total accepted", sum(v["sac"] + v["d4pg"] for v in s["cells"].values()), "/600")
    for slot, name in s["active"].items():
        print(slot, name)
    if s["state"] == "needs-review":
        print("routes:", json.dumps(s["routes"], ensure_ascii=False))
    print("remaining projected GPU hours", s["estimated_gpu_hours"], "deadline fallback", s["rush"])


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("coordinator")
    c.add_argument(
        "--deadline", default=None, help="ISO timestamp with UTC offset for the soft D4PG quota"
    )
    sub.add_parser("worker").add_argument("--gpu", type=int, required=True)
    sub.add_parser("status")
    args = p.parse_args()
    root = args.root.resolve()
    if args.cmd == "coordinator":
        coordinator(root, args.deadline)
    elif args.cmd == "worker":
        worker(root, args.gpu)
    else:
        status(root)
