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

"""Train the diff_drive rover to follow a line using Stable Baselines3 PPO.

Vision-based control: the env's observation includes camera-derived line
offset (computed inside the sync-gate plugin so the trainer doesn't have
to subscribe to image topics). The action is a `[linear_x, angular_z]`
Twist tuple.

Saves checkpoints every 10k steps to `models/line_follower/` and a final
`models/line_follower/final.zip` once training completes. Re-running the
script auto-resumes from `final.zip` if it exists (delete the file to
start fresh).

Prereq:
    ros2 launch gazebo_gymnasium_bringup line_follower.launch.py
        OR
    ros2 launch gazebo_gymnasium_bringup line_follower_train.launch.py
        (runs the sim + this trainer in one composable container)
"""

import os
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

from gazebo_gymnasium_bridge.envs import GazeboLineFollowerEnv


MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "line_follower"
FINAL_PATH = MODEL_DIR / "final.zip"


def _device() -> str:
    return os.environ.get("GAZEBO_GYM_DEVICE", "cpu").lower()


def _build_fresh_model(env):
    """PPO with the project's defaults for the line-following env."""
    return PPO(
        "MlpPolicy", env,
        verbose=1,
        n_steps=128,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.005,
        learning_rate=3e-4,
        device=_device(),
    )


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    env = GazeboLineFollowerEnv(world_name="line_follower")

    # GAZEBO_GYM_TIMESTEPS is honored when launched via the composable
    # train launch (which sets the env var from its `timesteps:=N` arg).
    total_timesteps = int(os.environ.get("GAZEBO_GYM_TIMESTEPS", "200000"))

    # Resume if a checkpoint exists AND its obs/action spaces still match
    # the current env. SB3 raises ValueError from check_for_correct_spaces
    # when they don't (e.g. after the camera resolution changed, which
    # shifts the line_offset bounds). In that case we surface the mismatch
    # clearly and train fresh — the old policy's weights wouldn't be
    # meaningful for the new env anyway.
    resumed = False
    if FINAL_PATH.exists():
        try:
            print(f"[train_line_follower_ppo] Resuming from {FINAL_PATH}")
            model = PPO.load(FINAL_PATH, env=env)
            resumed = True
        except (ValueError, RuntimeError) as exc:
            print(f"[train_line_follower_ppo] WARNING: cannot load checkpoint "
                  f"({type(exc).__name__}: {exc})")
            print(f"[train_line_follower_ppo] The env's spaces or other "
                  f"properties have changed since the checkpoint was saved. "
                  f"Move/delete {FINAL_PATH} to suppress this and start fresh.")
            model = _build_fresh_model(env)
    else:
        model = _build_fresh_model(env)

    checkpoint_cb = CheckpointCallback(
        save_freq=10_000,
        save_path=str(MODEL_DIR),
        name_prefix="ppo_line_follower",
    )

    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=checkpoint_cb,
            reset_num_timesteps=not resumed,
        )
    finally:
        model.save(FINAL_PATH)
        print(f"[train_line_follower_ppo] Saved final model to {FINAL_PATH}")


if __name__ == "__main__":
    main()
