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

"""Train CartPole in Gazebo using Stable Baselines3 PPO.

Mirrors gymnasium_mujoco.py / sb3_pygame.py in shape — the only Gazebo-specific
import is `GazeboCartPoleEnv`. SB3 sees a plain `gym.Env`.

Saves checkpoints every 10k steps to `models/cartpole/` and a final
`models/cartpole/final.zip` once training completes. Re-running the script
auto-resumes from `final.zip` if it exists (delete the file to start fresh).

Prereq:
    Start the simulator separately (in another terminal):
        ros2 launch gazebo_gymnasium_bringup cartpole.launch.py
    Then run this script.
"""

from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

from gazebo_gymnasium_bridge.envs import GazeboCartPoleEnv


MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "cartpole"
FINAL_PATH = MODEL_DIR / "final.zip"


def _device() -> str:
    """Read GAZEBO_GYM_DEVICE env var.

    `cpu` (default) | `cuda` | `auto` (SB3 picks CUDA if available).
    """
    import os
    return os.environ.get("GAZEBO_GYM_DEVICE", "cpu").lower()


def _build_fresh_model(env):
    return PPO(
        "MlpPolicy", env,
        verbose=1,
        n_steps=256,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.001,
        learning_rate=3e-4,
        device=_device(),
    )


if __name__ == "__main__":
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    env = GazeboCartPoleEnv(world_name="cartpole")

    # Try to resume; fall back to fresh training if the saved spaces no
    # longer match the env (see train_line_follower_ppo.py for the
    # rationale).
    resumed = False
    if FINAL_PATH.exists():
        try:
            print(f"[train_cartpole_sb3] Resuming from {FINAL_PATH}")
            model = PPO.load(FINAL_PATH, env=env)
            resumed = True
        except (ValueError, RuntimeError) as exc:
            print(f"[train_cartpole_sb3] WARNING: cannot load checkpoint "
                  f"({type(exc).__name__}: {exc}). Starting fresh.")
            model = _build_fresh_model(env)
    else:
        model = _build_fresh_model(env)

    checkpoint_cb = CheckpointCallback(
        save_freq=10_000,
        save_path=str(MODEL_DIR),
        name_prefix="ppo_cartpole",
    )

    try:
        model.learn(total_timesteps=200_000, callback=checkpoint_cb,
                    reset_num_timesteps=not resumed)
    finally:
        # Save final model even if interrupted (Ctrl-C) so a long run isn't lost.
        model.save(FINAL_PATH)
        print(f"[train_cartpole_sb3] Saved final model to {FINAL_PATH}")
