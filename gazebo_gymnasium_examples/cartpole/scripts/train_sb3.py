#!/usr/bin/env python3
"""
Train CartPole in Gazebo using Stable-Baselines3 PPO.

Requires Gazebo to already be running with the cartpole world:
    ros2 launch gazebo_gymnasium_examples cartpole.launch.py

Then run this script:
    python3 train_sb3.py
    python3 train_sb3.py --timesteps 200000
    python3 train_sb3.py --check-only

To view training curves live:
    tensorboard --logdir ./tb_logs
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
        "tensorboard is required.\n"
        "Install it with: pip install tensorboard"
    )

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.env_checker import check_env

import cartpole_env as _cartpole_env_mod
from cartpole_env import CartPoleEnv


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

class EpisodeLogger(BaseCallback):
    """Print per-episode reward and length immediately after each episode ends.

    SB3 verbose=1 only logs at the end of each rollout buffer.  This gives
    live feedback after every individual episode without waiting.
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


class CheckpointCallback(BaseCallback):
    """Save the model every `save_freq` steps without touching the environment."""

    def __init__(self, save_freq: int, save_path: str, verbose=0):
        super().__init__(verbose)
        self._save_freq = save_freq
        self._save_path = save_path

    def _on_step(self) -> bool:
        if self.num_timesteps % self._save_freq == 0:
            path = f"{self._save_path}_{self.num_timesteps}"
            self.model.save(path)
            if self.verbose:
                print(f"  [checkpoint] saved → {path}.zip")
        return True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Train CartPole in Gazebo with SB3 PPO"
    )
    p.add_argument("--timesteps", type=int, default=100_000,
                   help="Total training timesteps (default: 100 000)")
    p.add_argument("--check-only", action="store_true",
                   help="Run env checker + 5 random episodes, then exit")
    p.add_argument("--save-path", default="cartpole_ppo",
                   help="Base path for saved models (default: cartpole_ppo)")
    p.add_argument("--tensorboard-log", default="./tb_logs",
                   help="TensorBoard log directory (default: ./tb_logs)")
    p.add_argument("--checkpoint-freq", type=int, default=25_000,
                   help="Save a model checkpoint every N steps (default: 25 000)")
    p.add_argument("--device", default="auto",
                   help="PyTorch device: auto (default), cpu, cuda")
    p.add_argument("--debug", action="store_true",
                   help="Enable per-callback tracing in CartPoleEnv")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # Print GZ_PARTITION early so the user can verify it matches the Gazebo terminal.
    # If unset, gz.transport uses hostname:username — fine when both are on same machine/user.
    gz_part = os.environ.get("GZ_PARTITION", "<not set — using hostname:username default>")
    print(f"[train] GZ_PARTITION={gz_part}", flush=True)

    if args.debug:
        _cartpole_env_mod.DEBUG = True
        print("[train] CartPoleEnv debug tracing ENABLED")

    env = CartPoleEnv(steps_per_action=5)

    # --check-only: validate the env and exit
    if args.check_only:
        print("Checking environment spec...")
        check_env(env, warn=True)
        print("Spec OK.  Running 5 random-action episodes...")
        for ep in range(5):
            obs, _ = env.reset()
            total_reward = 0.0
            done = False
            while not done:
                action = env.action_space.sample()
                obs, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward
                done = terminated or truncated
            print(f"  Episode {ep + 1}: reward = {total_reward:.0f}")
        return

    # Device selection
    if args.device == "auto":
        resolved = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        resolved = args.device
    print(f"Device: {resolved}" +
          (" (CUDA available)" if torch.cuda.is_available() else " (CPU only)"))

    # PPO hyperparameters tuned for CartPole in Gazebo.
    #
    # Note: EvalCallback is intentionally omitted.  Using the training env for
    #   evaluation corrupts Gazebo's sim state on resume, adding noise to training.
    #   The EpisodeLogger below gives live convergence feedback instead.
    #
    # ent_coef=0.05: higher entropy bonus keeps the policy from collapsing to
    #   always pushing one direction.  Early episodes should use both actions
    #   (cart oscillating left/right) with 15-30 steps of survival in the first
    #   50-100 episodes.  0.01 is too low and causes premature action-collapse.
    #
    # n_steps=512: ~25 s sim-time per rollout at 50 ms per step.  Enough
    #   episodes per update for stable GAE estimates with early short episodes.
    #
    # gamma=0.99: high discount keeps the agent focused on long-term survival
    #   (+1 every step up to 500 steps = maximum episode reward of 500).
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        device=args.device,
        n_steps=512,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.05,
        vf_coef=0.5,
        tensorboard_log=args.tensorboard_log,
    )

    callbacks = CallbackList([
        EpisodeLogger(),
        CheckpointCallback(
            save_freq=args.checkpoint_freq,
            save_path=args.save_path,
            verbose=1,
        ),
    ])

    print(f"\nTraining for {args.timesteps:,} timesteps...")
    print(f"TensorBoard: tensorboard --logdir {args.tensorboard_log}")
    print(f"Checkpoints every {args.checkpoint_freq:,} steps → {args.save_path}_N.zip\n")

    model.learn(total_timesteps=args.timesteps, callback=callbacks)

    model.save(args.save_path)
    print(f"\nFinal model saved → {args.save_path}.zip")


if __name__ == "__main__":
    main()
