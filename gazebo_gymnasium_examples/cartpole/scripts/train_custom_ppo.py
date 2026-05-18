#!/usr/bin/env python3
"""CartPole training using the built-in gazebo_gymnasium PPO (no SB3 required).

Demonstrates that non-SB3 code works with the GazeboEnv framework.
The custom PPO uses only numpy and PyTorch.

Launch Gazebo first:
    ros2 launch gazebo_gymnasium_examples cartpole_full.launch.py

Then run:
    python3 train_custom_ppo.py [--timesteps N] [--device cpu|cuda]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gazebo_gymnasium.ppo import PPO
from cartpole_env import CartPoleEnv


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--timesteps", type=int, default=100_000)
    p.add_argument("--save-path", default="cartpole_custom_ppo")
    p.add_argument("--tensorboard-log", default="./tb_logs_cartpole_custom")
    p.add_argument("--device", default="auto")
    return p.parse_args()


def main():
    args = parse_args()
    env = CartPoleEnv(steps_per_action=5)

    model = PPO(
        env,
        hidden_size=64,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        n_steps=512,
        n_epochs=10,
        batch_size=64,
        device=args.device,
        verbose=1,
        tensorboard_log=args.tensorboard_log,
    )

    print(f"Training for {args.timesteps} steps with custom PPO (no SB3)")
    model.learn(total_timesteps=args.timesteps)
    model.save(args.save_path)
    print(f"Saved {args.save_path}.pt")


if __name__ == "__main__":
    main()
