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

"""Train Gazebo Reacher using Stable Baselines3 TD3.

Why TD3 here: continuous-action, dense-reward env. TD3's twin critics +
delayed policy updates handle the small action space (2 torques) and
short episodes (50 steps) cleanly. SAC also works; TD3 picked for
algorithm variety alongside SAC (InvertedPendulum) and PPO (CartPole).

Reacher converges quickly thanks to its dense distance reward — expect
the policy to start reliably reaching the target within ~50k steps.

Save/checkpoint/auto-resume mirror the other training scripts; see
train_line_follower_ppo.py for the resume-resilience pattern.

Prereq:
    ros2 launch gazebo_gymnasium_bringup reacher.launch.py
"""

import os
from pathlib import Path

import numpy as np
from stable_baselines3 import TD3
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.noise import NormalActionNoise

from gazebo_gymnasium_bridge.envs import GazeboReacherEnv


MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "reacher"
FINAL_PATH = MODEL_DIR / "final.zip"


def _device() -> str:
    return os.environ.get("GAZEBO_GYM_DEVICE", "cpu").lower()


def _build_fresh_model(env):
    # TD3 with Gaussian action noise for exploration (sigma=0.1 on a [-1,1]
    # action range is the SB3 recommended default for continuous control).
    n_actions = env.action_space.shape[0]
    action_noise = NormalActionNoise(
        mean=np.zeros(n_actions),
        sigma=0.1 * np.ones(n_actions),
    )
    return TD3(
        "MlpPolicy", env,
        verbose=1,
        learning_rate=1e-3,
        buffer_size=200_000,
        learning_starts=1_000,
        batch_size=256,
        tau=0.005,
        gamma=0.98,           # short episodes (50 steps) — lower gamma is fine
        train_freq=(1, "episode"),
        gradient_steps=-1,    # match the number of env steps in the rollout
        action_noise=action_noise,
        policy_kwargs={"net_arch": [256, 256]},
        device=_device(),
    )


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    env = GazeboReacherEnv(world_name="reacher")
    total_timesteps = int(os.environ.get("GAZEBO_GYM_TIMESTEPS", "100000"))

    resumed = False
    if FINAL_PATH.exists():
        try:
            print(f"[train_reacher_td3] Resuming from {FINAL_PATH}")
            model = TD3.load(FINAL_PATH, env=env)
            resumed = True
        except (ValueError, RuntimeError) as exc:
            print(f"[train_reacher_td3] WARNING: cannot load checkpoint "
                  f"({type(exc).__name__}: {exc}). Starting fresh.")
            model = _build_fresh_model(env)
    else:
        model = _build_fresh_model(env)

    checkpoint_cb = CheckpointCallback(
        save_freq=10_000,
        save_path=str(MODEL_DIR),
        name_prefix="td3_reacher",
    )

    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=checkpoint_cb,
            reset_num_timesteps=not resumed,
        )
    finally:
        model.save(FINAL_PATH)
        print(f"[train_reacher_td3] Saved final model to {FINAL_PATH}")


if __name__ == "__main__":
    main()
