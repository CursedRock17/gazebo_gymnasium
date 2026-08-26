# Copyright 2026 Lucas Wendland
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Throughput benchmark — simulation steps per second, per env and per scale.

Measures the *simulator* rate with a random policy (no learner in the loop), so
the numbers isolate what this package controls. Two figures are reported:

* ``env_steps/s``  — vectorized steps per second (one step advances all agents)
* ``agent_steps/s`` — env_steps/s x n_agents, the number comparable to a
  single-agent simulator such as MuJoCo

Because N agents share one Gazebo world, agent_steps/s rises with N while
env_steps/s falls only slowly — that scaling is the point of the architecture,
and the ``--scale`` mode measures it directly.

    python training_scripts/benchmark.py                       # all envs, N=8
    python training_scripts/benchmark.py --agent hopper --scale 1,4,8,16,32
    python training_scripts/benchmark.py --csv results.csv
"""

import argparse
import csv
from pathlib import Path
import time

import numpy as np

from gazebo_gymnasium_bridge.envs import make_inprocess
from gazebo_gymnasium_bridge.envs import registered_specs


def _random_actions(env, rng):
    """Sample a batch of valid actions for the vectorized env."""
    space = env.action_space
    if hasattr(space, "n"):  # Discrete
        return rng.integers(0, space.n, size=env.num_envs)
    low = np.broadcast_to(space.low, (env.num_envs,) + space.shape)
    high = np.broadcast_to(space.high, (env.num_envs,) + space.shape)
    return rng.uniform(low, high).astype(np.float32)


def measure(agent, n_agents, steps, warmup, seed=0):
    """Time `steps` vectorized steps of `agent` at `n_agents`; return a dict."""
    env = make_inprocess(agent, n_agents=n_agents, seed=seed)
    try:
        rng = np.random.default_rng(seed)
        env.reset()
        for _ in range(warmup):  # exclude first-touch costs
            env.step(_random_actions(env, rng))
        t0 = time.perf_counter()
        for _ in range(steps):
            env.step(_random_actions(env, rng))
        dt = time.perf_counter() - t0
    finally:
        env.close()
    env_sps = steps / dt
    return {
        "agent": agent,
        "n_agents": n_agents,
        "env_steps_per_s": round(env_sps, 1),
        "agent_steps_per_s": round(env_sps * n_agents, 1),
        "seconds": round(dt, 2),
    }


def _print_table(rows):
    head = f"{'environment':<26}{'N':>4}{'env steps/s':>14}{'agent steps/s':>16}"
    print(head)
    print("-" * len(head))
    for r in rows:
        print(
            f"{r['agent']:<26}{r['n_agents']:>4}"
            f"{r['env_steps_per_s']:>14,.1f}{r['agent_steps_per_s']:>16,.1f}"
        )


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--agent", default=None, help=f"one spec (default: all). Registered: {registered_specs()}"
    )
    p.add_argument("--n-agents", type=int, default=8)
    p.add_argument("--steps", type=int, default=300, help="timed vectorized steps per measurement")
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument(
        "--scale",
        default=None,
        help="comma-separated agent counts, e.g. 1,4,8,16,32 (requires --agent)",
    )
    p.add_argument("--csv", default=None, help="also write results to this CSV")
    args = p.parse_args()

    if args.scale:
        if not args.agent:
            raise SystemExit("--scale requires --agent")
        counts = [int(c) for c in args.scale.split(",")]
        jobs = [(args.agent, n) for n in counts]
    else:
        agents = [args.agent] if args.agent else registered_specs()
        jobs = [(a, args.n_agents) for a in agents]

    rows = []
    for agent, n in jobs:
        try:
            row = measure(agent, n, args.steps, args.warmup)
        except Exception as exc:  # noqa: B902
            print(f"  [{agent} N={n} skipped: {type(exc).__name__}: {exc}]")
            continue
        rows.append(row)
        print(
            f"  measured {agent:<24} N={n:<3} {row['env_steps_per_s']:>10,.1f} env steps/s",
            flush=True,
        )

    print()
    _print_table(rows)
    print(
        "\nRandom policy, no learner in the loop; in-process backend, "
        "headless. Vision environments are far slower — they render."
    )

    if args.csv and rows:
        out = Path(args.csv)
        with open(out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"Wrote {out}")


if __name__ == "__main__":
    main()
