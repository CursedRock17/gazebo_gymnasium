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
r"""Evaluate a trained policy deterministically against its Gazebo world.

The single eval entry point, mirroring train.py. Loads an SB3 model and runs
the greedy policy (distribution mean / argmax) for a few episodes — no
exploration, no updates.

Prereq: the matching world is running, e.g.
    ros2 launch gazebo_gymnasium_bringup cartpole_multi.launch.py n_agents:=4

Then:
    python training_scripts/deploy.py --agent cartpole --n_agents 4

Hugging Face Hub: ``--from-hub <repo_id>`` downloads and runs a model
published there directly instead of a local path (needs
``huggingface_hub`` installed -- soft dependency, matches ``train.py
--push-to-hub``'s treatment). Overrides ``--model``.
"""

import argparse
from pathlib import Path

import numpy as np
import stable_baselines3 as sb3

from gazebo_gymnasium_bridge.envs import make_harness
from gazebo_gymnasium_bridge.envs import make_inprocess
from gazebo_gymnasium_bridge.envs import make_multi
from gazebo_gymnasium_bridge.envs import wrap_for_observations

MODELS_ROOT = Path(__file__).resolve().parent.parent / "models"
_ALGOS = {"ppo": sb3.PPO, "a2c": sb3.A2C, "sac": sb3.SAC, "td3": sb3.TD3, "ddpg": sb3.DDPG}


def _scaled_timeouts(n_agents: int):
    return max(3.0, 0.9 * n_agents), max(1.0, 0.3 * n_agents)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--agent", default="cartpole")
    p.add_argument("--n_agents", type=int, default=4)
    p.add_argument("--algo", default="ppo", choices=sorted(_ALGOS))
    p.add_argument(
        "--model",
        default=None,
        help="path to .zip (default: models/<agent>_multi/final_<algo>_n<N>.zip)",
    )
    p.add_argument(
        "--from-hub",
        metavar="REPO_ID",
        default=None,
        help="download the model from this Hugging Face Hub repo "
        "(user/name) instead of a local path",
    )
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--world", default=None)
    p.add_argument("--backend", default="inprocess", choices=("inprocess", "harness", "peragent"))
    p.add_argument(
        "--frame-stack", type=int, default=4, help="must match training (image observations only)"
    )
    args = p.parse_args()

    if args.from_hub:
        from hub import pull_from_hub

        model_path = Path(pull_from_hub(args.from_hub))
        print(f"[deploy] downloaded {args.from_hub} -> {model_path}")
    else:
        model_path = (
            Path(args.model)
            if args.model
            else MODELS_ROOT / f"{args.agent}_multi" / f"final_{args.algo}_n{args.n_agents}.zip"
        )
        if not model_path.exists():
            raise SystemExit(f"model not found: {model_path}")

    reset_to, step_to = _scaled_timeouts(args.n_agents)
    _factories = {"inprocess": make_inprocess, "harness": make_harness, "peragent": make_multi}
    env = _factories[args.backend](
        args.agent,
        n_agents=args.n_agents,
        world_name=args.world,
        reset_timeout=reset_to,
        step_timeout=step_to,
    )
    env, _policy = wrap_for_observations(env, args.frame_stack)
    model = _ALGOS[args.algo].load(str(model_path))
    print(f"[deploy] {model_path} -> deterministic eval, {args.episodes} episode(s)")

    for ep in range(args.episodes):
        obs = env.reset()
        cur = np.zeros(args.n_agents, dtype=int)
        for _ in range(env.max_episode_steps + 20):
            action, _ = model.predict(obs, deterministic=True)
            env.step_async(action)
            obs, _r, dones, _i = env.step_wait()
            cur += (~dones).astype(int)
            if dones.all():
                break
        print(
            f"[deploy] episode {ep}: per-agent steps={cur.tolist()} "
            f"mean={float(cur.mean()):.0f}/{env.max_episode_steps}"
        )
    env.close()


if __name__ == "__main__":
    main()
