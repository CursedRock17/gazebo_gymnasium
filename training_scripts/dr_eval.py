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
"""Per-agent solve-rate eval, precise enough to trust for a go/no-go call.

``deploy.py`` breaks its episode loop on ``dones.all()``, which reports
imprecisely once agents desync -- fine for watching a policy, not for deciding
whether a checkpoint is solved. This tracks each agent's own termination step
independently and reports the fraction that reached the full step cap.

    python training_scripts/dr_eval.py MODEL.zip --n-agents 8 --rounds 5
    python training_scripts/dr_eval.py MODEL.zip --track-shapes reset
"""

import argparse
from pathlib import Path
import sys

import numpy as np
import stable_baselines3 as sb3
from stable_baselines3.common.vec_env import VecMonitor

from gazebo_gymnasium_bridge.envs import make_inprocess
from gazebo_gymnasium_bridge.envs import wrap_for_observations
from gazebo_gymnasium_bridge.envs.agent_spec import get_spec
from gazebo_gymnasium_bridge.envs.agent_spec import register_track_randomized
from gazebo_gymnasium_bridge.envs.agent_spec import TRACK_SHAPE_MODES

_ALGOS = {"ppo": sb3.PPO, "a2c": sb3.A2C, "ddpg": sb3.DDPG, "sac": sb3.SAC}


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("model", help="path to the .zip checkpoint to evaluate")
    p.add_argument("--agent", default="line_follower")
    p.add_argument("--algo", default="ppo", choices=sorted(_ALGOS))
    p.add_argument("--n-agents", type=int, default=8)
    p.add_argument("--rounds", type=int, default=3, help="episodes per agent")
    p.add_argument("--track-shapes", default="reset", choices=TRACK_SHAPE_MODES)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--frame-stack", type=int, default=4, help="must match training")
    args = p.parse_args()

    name = register_track_randomized(args.agent, args.track_shapes)
    cap = get_spec(name).max_episode_steps
    env = VecMonitor(make_inprocess(name, n_agents=args.n_agents, seed=args.seed))
    env, _policy = wrap_for_observations(env, args.frame_stack)
    model = _ALGOS[args.algo].load(args.model)

    solved = total = 0
    term_steps = []
    try:
        for r in range(args.rounds):
            obs = env.reset()
            alive = np.ones(args.n_agents, dtype=bool)
            died_at = np.full(args.n_agents, cap, dtype=int)
            for t in range(cap):
                action, _ = model.predict(obs, deterministic=True)
                obs, _rew, dones, _info = env.step(action)
                died_at[alive & dones] = t + 1
                alive &= ~dones
                if not alive.any():
                    break
            solved += int((died_at >= cap).sum())
            total += args.n_agents
            term_steps.extend(died_at.tolist())
            print(
                f"  round {r}: solved {int((died_at >= cap).sum())}/{args.n_agents} "
                f"term_steps={died_at.tolist()}",
                flush=True,
            )
    finally:
        env.close()

    pct = 100.0 * solved / total
    print(
        f"EVAL model={Path(args.model).name} track_shapes={args.track_shapes} "
        f"n={args.n_agents} rounds={args.rounds} solved={solved}/{total} ({pct:.1f}%) "
        f"median_term_step={int(np.median(term_steps))}"
    )
    return 0 if solved == total else 1


if __name__ == "__main__":
    sys.exit(main())
