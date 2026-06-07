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

"""Train Gazebo InvertedPendulum using Stable Baselines3 SAC.

Why SAC: it's the standard MuJoCo/Gymnasium baseline for continuous-control benchmarks
(InvertedPendulum, HalfCheetah, Hopper, Walker2d). SAC is off-policy + entropy-regularized, so it
learns very sample-efficiently on continuous-action problems compared to PPO.

Saves checkpoints every 5k steps to `models/inverted_pendulum/` and a final
`models/inverted_pendulum/final.zip` once training completes. Re-running
the script auto-resumes from `final.zip` if it exists (delete to start fresh).

Prereq — start the simulator first:     ros2 launch gazebo_gymnasium_bringup
inverted_pendulum.launch.py
"""

import os
from pathlib import Path

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback

from gazebo_gymnasium_bridge.envs import GazeboInvertedPendulumEnv


MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "inverted_pendulum"
FINAL_PATH = MODEL_DIR / "final.zip"


def _device() -> str:
    return os.environ.get("GAZEBO_GYM_DEVICE", "cpu").lower()


def _build_fresh_model(env):
    # Hyperparameters match SB3's "good-enough" defaults for continuous
    # control on Pendulum/InvertedPendulum-style tasks. learning_starts=100
    # so SAC fills its replay buffer with a bit of data before training.
    return SAC(
        "MlpPolicy", env,
        verbose=1,
        learning_rate=3e-4,
        buffer_size=100_000,
        learning_starts=100,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
        ent_coef="auto",  # automatic entropy temperature tuning
        device=_device(),
    )


if __name__ == "__main__":
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    env = GazeboInvertedPendulumEnv(world_name="inverted_pendulum")

    resumed = False
    if FINAL_PATH.exists():
        try:
            print(f"[train_inverted_pendulum_sac] Resuming from {FINAL_PATH}")
            model = SAC.load(FINAL_PATH, env=env)
            resumed = True
        except (ValueError, RuntimeError) as exc:
            print(f"[train_inverted_pendulum_sac] WARNING: cannot load "
                  f"checkpoint ({type(exc).__name__}: {exc}). Starting fresh.")
            model = _build_fresh_model(env)
    else:
        model = _build_fresh_model(env)

    checkpoint_cb = CheckpointCallback(
        save_freq=5_000,
        save_path=str(MODEL_DIR),
        name_prefix="sac_inverted_pendulum",
    )

    try:
        model.learn(total_timesteps=50_000, callback=checkpoint_cb,
                    reset_num_timesteps=not resumed)
    finally:
        model.save(FINAL_PATH)
        print(f"[train_inverted_pendulum_sac] Saved final model to {FINAL_PATH}")
