#!/usr/bin/env python3
"""
Train CartPole in Gazebo using Stable-Baselines3.

Requires Gazebo to already be running with the cartpole world:
    ros2 launch gazebo_gymnasium_examples cartpole/launch/cartpole.launch.py

Then run this script:
    python3 train_sb3.py [--timesteps N] [--check-only]

To use a different RL library, import CartPoleEnv into your own script instead.
"""
import argparse

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy

from cartpole_env import CartPoleEnv


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--timesteps", type=int, default=100_000,
                   help="Total training timesteps (default: 100000)")
    p.add_argument("--check-only", action="store_true",
                   help="Run environment checker with random actions then exit")
    p.add_argument("--save-path", default="cartpole_ppo",
                   help="Path to save trained model (default: cartpole_ppo)")
    return p.parse_args()


def main():
    args = parse_args()
    env = CartPoleEnv(steps_per_action=10)

    # Validate the environment follows the Gymnasium interface before training
    print("Checking environment...")
    check_env(env, warn=True)
    print("Environment check passed.")

    if args.check_only:
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

    # PPO is a strong default for CartPole's discrete action space.
    # The observation space has known bounds on position/angle but unbounded
    # velocities, so we rely on PPO's internal advantage normalization.
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
    )

    print(f"Training for {args.timesteps} timesteps...")
    model.learn(total_timesteps=args.timesteps)

    model.save(args.save_path)
    print(f"Model saved to {args.save_path}.zip")

    # Quick evaluation after training
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
    print(f"Evaluation over 10 episodes: mean={mean_reward:.1f} ± {std_reward:.1f}")


if __name__ == "__main__":
    main()
