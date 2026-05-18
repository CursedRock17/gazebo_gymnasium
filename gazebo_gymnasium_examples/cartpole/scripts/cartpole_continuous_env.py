#!/usr/bin/env python3
"""
CartPole environment with a continuous (Box) action space.

Drop-in replacement for CartPoleEnv when you want to test continuous-control
algorithms (SAC, TD3, DDPG, A2C/PPO with Gaussian policy) on the same Gazebo
CartPole model without changing the world or SDF.

Action space (Box, shape=(1,)):
    Scalar in [-1.0, 1.0] linearly mapped to a joint position target:
        target_y = action[0] × 0.5 m
    action=-1 → cart left (-0.5 m), action=+1 → cart right (+0.5 m).

Observation space (Box, 4):
    [cart_position, cart_velocity, pole_angle, pole_angular_velocity]
    Identical to CartPoleEnv.

Reward / termination / truncation:
    Identical to CartPoleEnv (+1 per step, |pole| > 12°, |cart| > 2.4 m).
"""
import os
import sys

# Allow importing cartpole_env from the same scripts/ directory regardless of cwd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from gymnasium.spaces import Box

from gz.msgs10.double_pb2 import Double  # type: ignore

import cartpole_env as _cartpole_env_mod
from cartpole_env import CartPoleEnv, _POS_RIGHT, _dbg


class CartPoleContinuousEnv(CartPoleEnv):
    """CartPole with Box(-1, 1, shape=(1,)) continuous action space.

    Only apply_action() and action_space differ from CartPoleEnv;
    all physics, observations, rewards, and termination logic are inherited.
    """

    def __init__(self, steps_per_action: int = 5):
        super().__init__(steps_per_action=steps_per_action)
        # Override the Discrete(2) set by the parent constructor
        self.action_space = Box(
            low=np.float32(-1.0),
            high=np.float32(1.0),
            shape=(1,),
            dtype=np.float32,
        )

    def apply_action(self, action):
        # Clip defensively; RL algorithms can occasionally produce slight out-of-range values
        scalar = float(np.clip(action[0], -1.0, 1.0))
        # Linear map: [-1, 1] → [−_POS_RIGHT, +_POS_RIGHT] = [-0.5, 0.5] m
        target = scalar * _POS_RIGHT

        self._callbacks_enabled = True
        self._joint_state_ready.clear()
        msg = Double()
        msg.data = target
        self._cmd_pub.publish(msg)
        _dbg(f"apply_action(continuous={scalar:+.3f}) → target={target:+.3f} m")
