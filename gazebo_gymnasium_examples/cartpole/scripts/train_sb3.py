#!/usr/bin/env python3
"""
Train CartPole in Gazebo using Stable-Baselines3.

Requires Gazebo to already be running with the cartpole world:
    ros2 launch gazebo_gymnasium_examples cartpole.launch.py

Then run this script:
    python3 train_sb3.py [--timesteps N] [--check-only] [--tensorboard-log DIR]

To view training curves live:
    tensorboard --logdir ./tb_logs

To use a different RL library, import CartPoleEnv into your own script instead.
"""
import argparse
import os
import sys

# Ensure cartpole_env is importable when this script is run from any directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import tensorboard  # noqa: F401
except ImportError:
    sys.exit(
        "tensorboard is required for training.\n"
        "Install it with: pip install tensorboard"
    )

import torch
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CallbackList
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy

import cartpole_env as _cartpole_env_mod
from cartpole_env import CartPoleEnv


class EpisodeLogger(BaseCallback):
    """Print per-episode reward and length to stdout as they complete.

    SB3's built-in verbose=1 output only appears at the end of each rollout
    buffer (every n_steps steps).  This callback gives immediate feedback
    after every individual episode so you can see whether the agent is
    learning without waiting for the next rollout to complete.
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self._ep_reward = 0.0
        self._ep_len = 0
        self._ep_count = 0

    def _on_step(self) -> bool:
        self._ep_reward += self.locals["rewards"][0]
        self._ep_len += 1
        if self.locals["dones"][0]:
            self._ep_count += 1
            print(
                f"  ep {self._ep_count:4d} | "
                f"steps={self._ep_len:4d} | "
                f"reward={self._ep_reward:.0f} | "
                f"total_steps={self.num_timesteps}"
            )
            self._ep_reward = 0.0
            self._ep_len = 0
        return True


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--timesteps", type=int, default=100_000,
                   help="Total training timesteps (default: 100000)")
    p.add_argument("--check-only", action="store_true",
                   help="Run environment checker with random actions then exit")
    p.add_argument("--save-path", default="cartpole_ppo",
                   help="Path to save trained model (default: cartpole_ppo)")
    p.add_argument("--tensorboard-log", default="./tb_logs",
                   help="Directory for TensorBoard logs (default: ./tb_logs)")
    p.add_argument("--eval-freq", type=int, default=5_000,
                   help="Evaluate and checkpoint the best model every N steps (default: 5000)")
    p.add_argument("--device", default="auto",
                   help="PyTorch device: 'auto' (default), 'cpu', 'cuda', 'cuda:0', etc.")
    p.add_argument("--debug", action="store_true",
                   help="Enable verbose per-callback/per-step tracing in CartPoleEnv")
    return p.parse_args()


def main():
    args = parse_args()
    if args.debug:
        _cartpole_env_mod.DEBUG = True
        print("[train] CartPoleEnv debug tracing ENABLED")
    # steps_per_action=5 → 50 ms per action step.  Halving this from 10 gives
    # the termination check twice as many chances to fire before the pole falls
    # past the 12° threshold, so episodes end sooner and more accurately.
    env = CartPoleEnv(steps_per_action=5)

    if args.check_only:
        print("Checking environment...")
        check_env(env, warn=True)
        print("Environment check passed.")
        print("Running 5 random-action episodes...")
        for ep in range(5):
            obs, _ = env.reset()
            total_reward = 0.0
            done = False
            while not done:
                action = env.action_space.sample()
                obs, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward
                done = terminated or truncated
            print(f"  Episode {ep + 1}: reward = {total_reward}")
        return

    # Report which device PyTorch will use. SB3 "auto" picks CUDA if available.
    if args.device == "auto":
        resolved = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        resolved = args.device
    print(f"PyTorch device: {resolved}" +
          (" (CUDA available)" if torch.cuda.is_available() else " (no CUDA detected)"))

    # PPO is a strong default for CartPole's discrete action space.
    # n_steps=256: SB3 only writes TensorBoard events at the end of each rollout
    # buffer, so a smaller buffer means more frequent live updates in TensorBoard.
    # 256 still gives enough samples for stable gradient estimates.
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        device=args.device,
        n_steps=256,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        tensorboard_log=args.tensorboard_log,
    )

    # EvalCallback: periodically evaluate the current policy and save the best model.
    # eval_freq halved from 10 000 to 5 000 to match the more frequent rollouts.
    eval_callback = EvalCallback(
        env,
        best_model_save_path=f"{args.save_path}_best",
        log_path=args.tensorboard_log,
        eval_freq=args.eval_freq,
        n_eval_episodes=5,
        deterministic=True,
        render=False,
    )

    # EpisodeLogger: print per-episode stats immediately after each episode ends,
    # independent of the rollout buffer size.
    episode_logger = EpisodeLogger()

    print(f"Training for {args.timesteps} timesteps...")
    print(f"TensorBoard logs → {args.tensorboard_log}")
    print("  View live: tensorboard --logdir " + args.tensorboard_log)
    model.learn(
        total_timesteps=args.timesteps,
        callback=CallbackList([eval_callback, episode_logger]),
    )

    model.save(args.save_path)
    print(f"Model saved to {args.save_path}.zip")
    print(f"Best model saved to {args.save_path}_best/")

    # Quick evaluation after training
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
    print(f"Evaluation over 10 episodes: mean={mean_reward:.1f} ± {std_reward:.1f}")


if __name__ == "__main__":
    main()
