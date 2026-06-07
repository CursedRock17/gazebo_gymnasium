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

"""Train Gazebo Point using Stable Baselines3 SAC.

Why SAC: continuous-action navigation env. SAC's automatic entropy
tuning handles the variable-radius reward landscape (sometimes the
target spawns close, sometimes far) without manual exploration
scheduling.

Point is the simplest env in the project — dense distance reward,
2D action, 6D obs. Expect convergence within ~20–50k steps.

Save/checkpoint/auto-resume mirror the other training scripts.

Prereq:
    ros2 launch gazebo_gymnasium_bringup point.launch.py
"""

import os
from pathlib import Path

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback

from gazebo_gymnasium_bridge.envs import GazeboPointEnv


MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "point"
FINAL_PATH = MODEL_DIR / "final.zip"


def _device() -> str:
    return os.environ.get("GAZEBO_GYM_DEVICE", "cpu").lower()


def _build_fresh_model(env):
    return SAC(
        "MlpPolicy", env,
        verbose=1,
        learning_rate=3e-4,
        buffer_size=100_000,
        learning_starts=500,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
        ent_coef="auto",
        device=_device(),
    )


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    env = GazeboPointEnv(world_name="point")
    total_timesteps = int(os.environ.get("GAZEBO_GYM_TIMESTEPS", "50000"))

    resumed = False
    if FINAL_PATH.exists():
        try:
            print(f"[train_point_sac] Resuming from {FINAL_PATH}")
            model = SAC.load(FINAL_PATH, env=env)
            resumed = True
        except (ValueError, RuntimeError) as exc:
            print(f"[train_point_sac] WARNING: cannot load checkpoint "
                  f"({type(exc).__name__}: {exc}). Starting fresh.")
            model = _build_fresh_model(env)
    else:
        model = _build_fresh_model(env)

    checkpoint_cb = CheckpointCallback(
        save_freq=5_000,
        save_path=str(MODEL_DIR),
        name_prefix="sac_point",
    )

    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=checkpoint_cb,
            reset_num_timesteps=not resumed,
        )
    finally:
        model.save(FINAL_PATH)
        print(f"[train_point_sac] Saved final model to {FINAL_PATH}")


if __name__ == "__main__":
    main()
