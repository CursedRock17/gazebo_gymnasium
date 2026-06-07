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

"""Train Gazebo InvertedDoublePendulum using Stable Baselines3 A2C.

Why A2C: it's the synchronous Actor-Critic baseline — a simpler on-policy
alternative to PPO. Pairing it with InvertedDoublePendulum here gives us
algorithm variety alongside CartPole+PPO and InvertedPendulum+SAC.

Saves checkpoints every 10k steps to `models/inverted_double_pendulum/` and
a final `models/inverted_double_pendulum/final.zip` once training
completes. Re-running auto-resumes from `final.zip` if it exists.

Prereq:
    ros2 launch gazebo_gymnasium_bringup inverted_double_pendulum.launch.py
"""

import os
from pathlib import Path

from stable_baselines3 import A2C
from stable_baselines3.common.callbacks import CheckpointCallback

from gazebo_gymnasium_bridge.envs import GazeboInvertedDoublePendulumEnv


MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "inverted_double_pendulum"
FINAL_PATH = MODEL_DIR / "final.zip"


def _device() -> str:
    return os.environ.get("GAZEBO_GYM_DEVICE", "cpu").lower()


def _build_fresh_model(env):
    # SB3 A2C defaults, adjusted for our env's slower wall-clock per step:
    # smaller n_steps so updates happen more often, larger learning rate
    # because A2C is on-policy and benefits from frequent updates.
    return A2C(
        "MlpPolicy", env,
        verbose=1,
        learning_rate=7e-4,
        n_steps=16,
        gamma=0.99,
        gae_lambda=1.0,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        device=_device(),
    )


if __name__ == "__main__":
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    env = GazeboInvertedDoublePendulumEnv(world_name="inverted_double_pendulum")

    resumed = False
    if FINAL_PATH.exists():
        try:
            print(f"[train_inverted_double_pendulum_a2c] Resuming from {FINAL_PATH}")
            model = A2C.load(FINAL_PATH, env=env)
            resumed = True
        except (ValueError, RuntimeError) as exc:
            print(f"[train_inverted_double_pendulum_a2c] WARNING: cannot load "
                  f"checkpoint ({type(exc).__name__}: {exc}). Starting fresh.")
            model = _build_fresh_model(env)
    else:
        model = _build_fresh_model(env)

    checkpoint_cb = CheckpointCallback(
        save_freq=10_000,
        save_path=str(MODEL_DIR),
        name_prefix="a2c_inverted_double_pendulum",
    )

    try:
        model.learn(total_timesteps=200_000, callback=checkpoint_cb,
                    reset_num_timesteps=not resumed)
    finally:
        model.save(FINAL_PATH)
        print(f"[train_inverted_double_pendulum_a2c] Saved final model to {FINAL_PATH}")
