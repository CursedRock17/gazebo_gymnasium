#!/usr/bin/env python3
"""
Train CartPole in Gazebo using Stable-Baselines3 A2C.

Requires Gazebo to already be running with the cartpole world:
    ros2 launch gazebo_gymnasium_examples cartpole.launch.py

Then run this script:
    python3 train_a2c.py [--timesteps N] [--continuous] [--check-only]

Use --continuous to train on CartPoleContinuousEnv (Box action space)
instead of the default CartPoleEnv (Discrete action space).

To view training curves live:
    tensorboard --logdir ./tb_logs_a2c
"""
import argparse
import os
import sys

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
from stable_baselines3 import A2C
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CallbackList
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy

import cartpole_env as _cartpole_env_mod
from cartpole_env import CartPoleEnv
from cartpole_continuous_env import CartPoleContinuousEnv


class EpisodeLogger(BaseCallback):
    """Print per-episode reward and length immediately after each episode ends."""

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
    p.add_argument("--continuous", action="store_true",
                   help="Use CartPoleContinuousEnv (Box action space) instead of Discrete")
    p.add_argument("--check-only", action="store_true",
                   help="Run environment checker with random actions then exit")
    p.add_argument("--save-path", default="cartpole_a2c",
                   help="Path to save trained model (default: cartpole_a2c)")
    p.add_argument("--tensorboard-log", default="./tb_logs_a2c",
                   help="Directory for TensorBoard logs (default: ./tb_logs_a2c)")
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
        print("[train_a2c] CartPoleEnv debug tracing ENABLED")

    if args.continuous:
        env = CartPoleContinuousEnv(steps_per_action=5)
        print("[train_a2c] Using CartPoleContinuousEnv (Box action space)")
    else:
        env = CartPoleEnv(steps_per_action=5)
        print("[train_a2c] Using CartPoleEnv (Discrete action space)")

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

    if args.device == "auto":
        resolved = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        resolved = args.device
    print(f"PyTorch device: {resolved}" +
          (" (CUDA available)" if torch.cuda.is_available() else " (no CUDA detected)"))

    # A2C hyperparameters tuned for CartPole with Gazebo physics.
    #
    # n_steps=128: each update sees ~1-3 full episodes (depending on pole balance),
    # giving stable gradient estimates without waiting for a large buffer.
    #
    # gae_lambda=1.0: use full Monte Carlo returns instead of GAE bootstrapping.
    # CartPole's dense +1/step reward makes this numerically stable.
    #
    # ent_coef=0.01: small entropy bonus prevents premature collapse to a
    # deterministic policy before the agent has explored enough.
    #
    # learning_rate=7e-4: the classic A2C rate from Mnih et al. (2016);
    # works reliably across CartPole variants.
    model = A2C(
        "MlpPolicy",
        env,
        verbose=1,
        device=args.device,
        n_steps=128,
        gamma=0.99,
        gae_lambda=1.0,
        ent_coef=0.01,
        vf_coef=0.25,
        max_grad_norm=0.5,
        learning_rate=7e-4,
        tensorboard_log=args.tensorboard_log,
    )

    eval_callback = EvalCallback(
        env,
        best_model_save_path=f"{args.save_path}_best",
        log_path=args.tensorboard_log,
        eval_freq=args.eval_freq,
        n_eval_episodes=5,
        deterministic=True,
        render=False,
    )

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

    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
    print(f"Evaluation over 10 episodes: mean={mean_reward:.1f} ± {std_reward:.1f}")


if __name__ == "__main__":
    main()
