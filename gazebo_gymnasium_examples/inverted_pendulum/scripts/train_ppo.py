#!/usr/bin/env python3
"""Train InvertedPendulum in Gazebo with PPO (Stable-Baselines3).

Launch Gazebo first:
    ros2 launch gazebo_gymnasium_examples inverted_pendulum_full.launch.py

Then run:
    python3 train_ppo.py [--timesteps N] [--check-only]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CallbackList
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy

from inverted_pendulum_env import InvertedPendulumEnv


class EpisodeLogger(BaseCallback):
    def __init__(self):
        super().__init__()
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
                f"total={self.num_timesteps}"
            )
            self._ep_reward = 0.0
            self._ep_len = 0
        return True


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--timesteps", type=int, default=200_000)
    p.add_argument("--check-only", action="store_true")
    p.add_argument("--save-path", default="inverted_pendulum_ppo")
    p.add_argument("--tensorboard-log", default="./tb_logs_inverted_pendulum")
    p.add_argument("--device", default="auto")
    return p.parse_args()


def main():
    args = parse_args()
    env = InvertedPendulumEnv(steps_per_action=4)

    if args.check_only:
        print("Checking environment...")
        check_env(env, warn=True)
        print("Environment check passed.")
        for ep in range(3):
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

    device = "cuda" if (args.device == "auto" and torch.cuda.is_available()) else args.device
    print(f"Device: {device}")

    # PPO with continuous actions — same hyperparams used for InvertedPendulum-v5
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        device=args.device,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        tensorboard_log=args.tensorboard_log,
    )

    eval_cb = EvalCallback(
        env,
        best_model_save_path=f"{args.save_path}_best",
        log_path=args.tensorboard_log,
        eval_freq=10_000,
        n_eval_episodes=5,
        deterministic=True,
        render=False,
    )

    print(f"Training for {args.timesteps} steps → {args.tensorboard_log}")
    model.learn(
        total_timesteps=args.timesteps,
        callback=CallbackList([eval_cb, EpisodeLogger()]),
    )

    model.save(args.save_path)
    print(f"Saved {args.save_path}.zip")

    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
    print(f"Eval (10 ep): mean={mean_reward:.1f} ± {std_reward:.1f}")


if __name__ == "__main__":
    main()
