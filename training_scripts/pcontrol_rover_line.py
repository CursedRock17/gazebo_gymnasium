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
"""Classical line-following reference for `rover_line`. No learning involved.

Two jobs. First, it answers "is this task solvable at all?" independently of
any policy -- if a Stanley-style law cannot follow the line, a failing agent
is evidence about the ENVIRONMENT, not about the algorithm. Second, it is the
bar: an RL policy that cannot match a dozen lines of proportional control has
not earned the complexity.

It also caught a real bug. With the drive sign unflipped the same law manages
1.17 m before losing the line; flipped, it drives 30 m. That asymmetry is what
identified the rover driving away from its own camera view, and it is the
argument for keeping this script around rather than deleting it once the
policy works.

    python training_scripts/pcontrol_rover_line.py --speed 1.0

Steering uses the two terms a Stanley controller uses -- cross-track error
from the nearest scan band, plus heading across the bands -- both read from
the same features the policy sees, so nothing here is privileged state.
"""

import argparse
import os

import numpy as np

os.environ.setdefault("GAZEBO_GYM_ODOM", "1")

from gazebo_gymnasium_bridge.envs import make_inprocess  # noqa: E402
from gazebo_gymnasium_bridge.envs import rover_line as rl  # noqa: E402


def control(feats, kp, kh):
    """Features -> (speed, turn) action. The whole controller."""
    cross = float(feats[0])
    heading = float(feats[2 * rl.N_BANDS])
    return float(np.clip(-(kp * cross + kh * heading), -1.0, 1.0))


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--n-agents", type=int, default=4)
    p.add_argument("--rounds", type=int, default=2)
    p.add_argument(
        "--speed", type=float, default=1.0, help="constant action[0]; 1.0 is V_MAX, -1.0 is V_MIN"
    )
    p.add_argument("--kp", type=float, default=2.0, help="cross-track gain")
    p.add_argument("--kh", type=float, default=0.5, help="heading gain")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    n, cap = args.n_agents, rl.MAX_EPISODE_STEPS
    env = make_inprocess("rover_line", n_agents=n, seed=args.seed)
    lengths = []
    for r in range(args.rounds):
        env.reset()
        alive = np.ones(n, dtype=bool)
        died = np.full(n, cap, dtype=int)
        for t in range(cap):
            act = np.empty((n, 2), dtype=np.float32)
            for i in range(n):
                act[i] = (args.speed, control(rl.features(env._latest_obs[i]), args.kp, args.kh))
            _obs, _rew, dones, _info = env.step(act)
            died[alive & dones] = t + 1
            alive &= ~dones
            if not alive.any():
                break
        lengths.extend(died.tolist())
        print(f"  round {r}: ep_len={died.tolist()}  reached_cap={int((died >= cap).sum())}/{n}")

    print(
        f"\nP-CONTROL speed={args.speed} kp={args.kp} kh={args.kh}: "
        f"median ep_len {np.median(lengths):.0f}/{cap}, "
        f"{sum(v >= cap for v in lengths)}/{len(lengths)} reached the cap"
    )
    print("  (run eval_rover_line.py for odometry-measured lap distance)")
    env.close()


if __name__ == "__main__":
    main()
