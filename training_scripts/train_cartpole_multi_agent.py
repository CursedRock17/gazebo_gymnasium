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

r"""Train CartPole with N agents in ONE gz sim — the cheap vectorized path.

This is the alternative to `train_cartpole_ppo_runner.py --n_envs N`,
which spins up N separate gz sim processes. Here N cartpoles live in a
single gz sim world; physics advances all N atomically per tick. Way
less overhead.

Mode comparison:

    train_cartpole_ppo_runner.py --n_envs 4   # 4 gz sim processes (~10 GB)
    train_cartpole_multi_agent.py --n_agents 4  # 1 gz sim process (~1 GB)

Both produce a (4, obs_dim) batch per step for PPO. The single-sim
version uses far less memory and CPU since the renderer / scene graph /
gz transport bus exist once instead of four times.

Prereq:
    ros2 launch gazebo_gymnasium_bringup cartpole_multi.launch.py n_agents:=4 \\
        headless:=true

The launch spawns N cartpoles dynamically (via ros_gz_sim Create) into
one empty `cartpole_multi` world, so any positive integer N works — no
pre-generated SDFs.
"""

import argparse
import os
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

from gazebo_gymnasium_bridge.envs import MultiCartPoleVecEnv


MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "cartpole_multi"


def _device() -> str:
    return os.environ.get("GAZEBO_GYM_DEVICE", "cpu").lower()


def _build_fresh_model(vec_env, n_agents: int):
    # n_steps is per-env, so rollout = n_steps × n_agents transitions
    # per update. Bump n_steps modestly when n_agents is small so the
    # rollout still has enough samples.
    n_steps = max(32, 256 // n_agents)
    return PPO(
        "MlpPolicy", vec_env,
        verbose=1,
        n_steps=n_steps,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.001,
        learning_rate=3e-4,
        device=_device(),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n_agents", type=int, default=4,
                        help="CartPoles in the one gz sim world (4, 8, or 16).")
    parser.add_argument("--timesteps", type=int, default=200_000,
                        help="Total env.step()s (across all agents).")
    args = parser.parse_args()

    world_name = "cartpole_multi"
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    vec_env = MultiCartPoleVecEnv(
        n_agents=args.n_agents,
        world_name=world_name,
    )

    final_path = MODEL_DIR / f"final_n{args.n_agents}.zip"
    resumed = False
    if final_path.exists():
        try:
            print(f"[train_cartpole_multi_agent] Resuming from {final_path}")
            model = PPO.load(final_path, env=vec_env)
            resumed = True
        except (ValueError, RuntimeError) as exc:
            print(f"[train_cartpole_multi_agent] WARNING: cannot load "
                  f"checkpoint ({type(exc).__name__}: {exc}). Starting fresh.")
            model = _build_fresh_model(vec_env, args.n_agents)
    else:
        model = _build_fresh_model(vec_env, args.n_agents)

    checkpoint_cb = CheckpointCallback(
        save_freq=max(1, 10_000 // args.n_agents),
        save_path=str(MODEL_DIR),
        name_prefix=f"ppo_cartpole_multi_n{args.n_agents}",
    )

    print(f"[train_cartpole_multi_agent] n_agents={args.n_agents}, "
          f"timesteps={args.timesteps}, model_dir={MODEL_DIR}")

    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=checkpoint_cb,
            reset_num_timesteps=not resumed,
        )
    finally:
        model.save(final_path)
        print(f"[train_cartpole_multi_agent] Saved final model to {final_path}")
        vec_env.close()


if __name__ == "__main__":
    main()
