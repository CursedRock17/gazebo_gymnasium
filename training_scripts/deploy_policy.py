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

r"""Load a trained SB3 policy and run it deterministically against its Gazebo env.

Use this to *evaluate* a model after training — no gradient updates, no
replay buffer, no exploration noise. The policy outputs the mean of its
action distribution, which is what you'd ship to a real robot.

Examples:
    python training_scripts/deploy_policy.py --env cartpole
    python training_scripts/deploy_policy.py --env inverted_pendulum \
        --checkpoint models/inverted_pendulum/sac_inverted_pendulum_30000_steps.zip
    python training_scripts/deploy_policy.py --env inverted_double_pendulum \
        --episodes 5 --stochastic

Prereq: the matching simulator launch must already be running in another
terminal, e.g. `ros2 launch gazebo_gymnasium_bringup cartpole.launch.py`.
"""

import argparse
import os
from pathlib import Path
import sys

from stable_baselines3 import A2C
from stable_baselines3 import PPO
from stable_baselines3 import SAC

from gazebo_gymnasium_bridge.envs import GazeboCartPoleEnv
from gazebo_gymnasium_bridge.envs import GazeboInvertedDoublePendulumEnv
from gazebo_gymnasium_bridge.envs import GazeboInvertedPendulumEnv


# Registry of supported envs. Each entry binds an env class to its default
# checkpoint path and the SB3 algorithm we use for it. Adding a new env to
# the project means appending one row here.
ENVS = {
    "cartpole": {
        "env_cls": GazeboCartPoleEnv,
        "algo_cls": PPO,
        "default_checkpoint": "models/cartpole/final.zip",
        "world_name": "cartpole",
    },
    "inverted_pendulum": {
        "env_cls": GazeboInvertedPendulumEnv,
        "algo_cls": SAC,
        "default_checkpoint": "models/inverted_pendulum/final.zip",
        "world_name": "inverted_pendulum",
    },
    "inverted_double_pendulum": {
        "env_cls": GazeboInvertedDoublePendulumEnv,
        "algo_cls": A2C,
        "default_checkpoint": "models/inverted_double_pendulum/final.zip",
        "world_name": "inverted_double_pendulum",
    },
}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--env", required=True, choices=sorted(ENVS),
        help="Which env to deploy the policy against.",
    )
    parser.add_argument(
        "--checkpoint", default=None,
        help="Path to the .zip checkpoint. Defaults to models/<env>/final.zip.",
    )
    parser.add_argument(
        "--episodes", type=int, default=3,
        help="How many episodes to roll out (default 3).",
    )
    parser.add_argument(
        "--stochastic", action="store_true",
        help="Sample actions from the policy distribution instead of taking the mean. "
             "Useful for SAC/PPO sanity-checks; default off (deterministic).",
    )
    args = parser.parse_args(argv)

    project_root = Path(__file__).resolve().parent.parent
    spec = ENVS[args.env]
    checkpoint = (Path(args.checkpoint) if args.checkpoint
                  else project_root / spec["default_checkpoint"])

    if not checkpoint.exists():
        print(f"ERROR: checkpoint not found at {checkpoint}", file=sys.stderr)
        print(f"       Run training first: training_scripts/train_{args.env}_*.py",
              file=sys.stderr)
        return 1

    env = spec["env_cls"](world_name=spec["world_name"])
    device = os.environ.get("GAZEBO_GYM_DEVICE", "cpu").lower()
    print(f"[deploy] Loading {spec['algo_cls'].__name__} policy from "
          f"{checkpoint} on device={device!r}")
    model = spec["algo_cls"].load(checkpoint, env=env, device=device)

    total_reward = 0.0
    total_steps = 0
    for ep in range(args.episodes):
        obs, _info = env.reset()
        ep_reward = 0.0
        ep_steps = 0
        while True:
            # `deterministic=True` returns the mean of the action distribution;
            # this is what you'd ship to a real robot. `--stochastic` flips it
            # for sanity-checking that the policy hasn't collapsed.
            action, _state = model.predict(obs, deterministic=not args.stochastic)
            obs, reward, terminated, truncated, _info = env.step(action)
            ep_reward += float(reward)
            ep_steps += 1
            if terminated or truncated:
                break
        total_reward += ep_reward
        total_steps += ep_steps
        print(f"[deploy] Episode {ep + 1}/{args.episodes}: "
              f"steps={ep_steps} reward={ep_reward:.1f}")

    if args.episodes > 0:
        print(f"[deploy] Mean reward over {args.episodes} episodes: "
              f"{total_reward / args.episodes:.1f} "
              f"(mean steps: {total_steps / args.episodes:.0f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
