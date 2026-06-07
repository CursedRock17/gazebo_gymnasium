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

"""All-in-one parallel-or-single CartPole trainer.

Spawns N headless `gz sim` subprocesses (each in its own GZ_PARTITION),
wraps the resulting envs in SubprocVecEnv (or DummyVecEnv if N=1), and
trains PPO on top. No separate `ros2 launch` step needed — this script
owns the entire stack.

Modes:
    --n_envs 1   : single env, headless. Same wall-time as the
                   `ros2 launch ... headless:=true` + train flow but
                   in one process group, no two-terminal dance.
    --n_envs 8   : 8 parallel headless envs. Rough sweet spot for a
                   14-core CPU (8 gz sim + 8 env workers + 1 trainer =
                   17 process slots; the OS happily oversubscribes the
                   remaining 3 cores for I/O).

Run:
    python training_scripts/train_cartpole_ppo_runner.py --n_envs 8
    python training_scripts/train_cartpole_ppo_runner.py --n_envs 1 --timesteps 50000

Prereqs (one-time):
    source /opt/ros/jazzy/setup.bash
    source install/setup.bash
"""

import argparse
import os
from pathlib import Path
import sys

# Make sibling _gz_runner.py importable when this script is invoked
# directly (training_scripts/ isn't a Python package).
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _gz_runner import make_headless_env_fn  # noqa: E402

from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.callbacks import CheckpointCallback  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv  # noqa: E402
from stable_baselines3.common.vec_env import SubprocVecEnv  # noqa: E402

from gazebo_gymnasium_bridge.envs import GazeboCartPoleEnv  # noqa: E402


MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "cartpole_parallel"
FINAL_PATH = MODEL_DIR / "final.zip"


def _device() -> str:
    return os.environ.get("GAZEBO_GYM_DEVICE", "cpu").lower()


def _build_fresh_model(vec_env, n_envs: int):
    # PPO defaults work fine across n_envs — n_steps is per-env, so total
    # rollout size is n_steps * n_envs. Bumped n_steps down a bit for
    # multi-env runs so updates happen more often per wall-second.
    n_steps = 128 if n_envs > 1 else 256
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
    parser.add_argument("--n_envs", type=int, default=1,
                        help="Number of parallel headless envs (1 = single).")
    parser.add_argument("--timesteps", type=int, default=200_000,
                        help="Total env.step()s summed across all envs.")
    args = parser.parse_args()

    if args.n_envs < 1:
        raise SystemExit("--n_envs must be >= 1")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Build N env factories. Each factory will spawn its own gz sim when
    # invoked (in the SubprocVecEnv worker process for n_envs > 1, or in
    # this process for n_envs == 1).
    env_fns = [
        make_headless_env_fn(
            env_cls=GazeboCartPoleEnv,
            env_kwargs={"world_name": "cartpole"},
            world_sdf_filename="cartpole.sdf",
            partition_id=i,
        )
        for i in range(args.n_envs)
    ]

    if args.n_envs == 1:
        # DummyVecEnv runs the env in-process — no SubprocVecEnv overhead.
        vec_env = DummyVecEnv(env_fns)
    else:
        vec_env = SubprocVecEnv(env_fns, start_method="spawn")

    # Auto-resume if a checkpoint exists with compatible spaces (same
    # pattern as the rest of the training scripts; see
    # train_line_follower_ppo.py for the rationale).
    resumed = False
    if FINAL_PATH.exists():
        try:
            print(f"[train_cartpole_ppo_runner] Resuming from {FINAL_PATH}")
            model = PPO.load(FINAL_PATH, env=vec_env)
            resumed = True
        except (ValueError, RuntimeError) as exc:
            print(f"[train_cartpole_ppo_runner] WARNING: cannot load "
                  f"checkpoint ({type(exc).__name__}: {exc}). Starting fresh.")
            model = _build_fresh_model(vec_env, args.n_envs)
    else:
        model = _build_fresh_model(vec_env, args.n_envs)

    checkpoint_cb = CheckpointCallback(
        save_freq=max(1, 10_000 // args.n_envs),  # account for vec env step counting
        save_path=str(MODEL_DIR),
        name_prefix=f"ppo_cartpole_n{args.n_envs}",
    )

    print(f"[train_cartpole_ppo_runner] n_envs={args.n_envs}, "
          f"timesteps={args.timesteps}, model_dir={MODEL_DIR}")

    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=checkpoint_cb,
            reset_num_timesteps=not resumed,
        )
    finally:
        model.save(FINAL_PATH)
        print(f"[train_cartpole_ppo_runner] Saved final model to {FINAL_PATH}")
        # SubprocVecEnv.close() terminates the workers; each worker's
        # atexit handler kills its own gz sim subprocess.
        vec_env.close()


if __name__ == "__main__":
    # SB3's SubprocVecEnv on Linux defaults to "fork" which can break with
    # CUDA / multi-threaded init. "spawn" is safer and avoids pickling
    # surprises in the env_fn closure.
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    main()
