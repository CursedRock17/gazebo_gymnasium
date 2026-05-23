#!/usr/bin/env python3
"""Train CartPole using the fixture-based env (no separate Gazebo process, no DDS).

gz.sim8.TestFixture embeds Gazebo inside this process.  server.run(True, N, False)
drives physics synchronously — no threads, no IPC, no network.

Usage:
    # Shortest path — SDF auto-resolved relative to this script:
    python3 train_fixture.py

    # Explicit SDF path:
    python3 train_fixture.py --sdf /path/to/cartpole_fixture.sdf

    # With options:
    python3 train_fixture.py --timesteps 200000 --device cuda
"""
import argparse
import os
import sys

# gz.sim8 Python bindings location under Gazebo Harmonic.
# These must be set before importing gz.sim8 (done lazily inside CartPoleFixtureEnv).
_GZ_PYTHON = "/usr/local/lib/python"
_GZ_LIB = "/usr/local/lib"
if _GZ_PYTHON not in sys.path:
    sys.path.insert(0, _GZ_PYTHON)
if "LD_LIBRARY_PATH" not in os.environ:
    os.environ["LD_LIBRARY_PATH"] = _GZ_LIB
elif _GZ_LIB not in os.environ["LD_LIBRARY_PATH"].split(":"):
    os.environ["LD_LIBRARY_PATH"] = _GZ_LIB + ":" + os.environ["LD_LIBRARY_PATH"]

_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
_WORLDS_DIR = os.path.join(_SCRIPTS_DIR, "..", "worlds")
sys.path.insert(0, _SCRIPTS_DIR)

from cartpole_fixture_env import CartPoleFixtureEnv  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="Train CartPole with gz.sim8 fixture env")
    p.add_argument(
        "--sdf",
        default=os.path.join(_WORLDS_DIR, "cartpole_fixture.sdf"),
        help="Path to cartpole_fixture.sdf (default: worlds/cartpole_fixture.sdf)",
    )
    p.add_argument("--timesteps", type=int, default=100_000)
    p.add_argument("--save-path", default="cartpole_fixture_ppo")
    p.add_argument("--tensorboard-log", default="./tb_logs_fixture")
    p.add_argument("--device", default="auto")
    p.add_argument(
        "--steps-per-action",
        type=int,
        default=5,
        help="Physics steps per env.step() call (default 5 = 50 ms sim time)",
    )
    return p.parse_args()


def main():
    args = parse_args()

    sdf = os.path.abspath(args.sdf)
    if not os.path.exists(sdf):
        print(f"ERROR: SDF not found: {sdf}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading SDF: {sdf}")
    env = CartPoleFixtureEnv(sdf_path=sdf, steps_per_action=args.steps_per_action)
    print("Fixture env ready — no separate Gazebo process required")

    try:
        from stable_baselines3 import PPO

        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            n_steps=512,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            verbose=1,
            device=args.device,
            tensorboard_log=args.tensorboard_log,
        )
        backend = "SB3"
    except ImportError:
        from gazebo_gymnasium.ppo import PPO

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
        backend = "custom PPO"

    print(f"Training {args.timesteps} steps with {backend} (fixture mode)")
    model.learn(total_timesteps=args.timesteps)
    model.save(args.save_path)
    print(f"Saved model → {args.save_path}")


if __name__ == "__main__":
    main()
