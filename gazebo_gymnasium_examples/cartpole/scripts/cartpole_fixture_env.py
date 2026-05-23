#!/usr/bin/env python3
"""CartPole environment using gz.sim8.TestFixture (ECM-direct, zero IPC).

No separate Gazebo process — this env IS the simulation server.
No gz-transport, no DDS, no ROS 2 required.

Usage:
    from cartpole_fixture_env import CartPoleFixtureEnv
    env = CartPoleFixtureEnv(sdf_path="/path/to/cartpole_fixture.sdf")
    obs, info = env.reset()
    obs, reward, terminated, truncated, info = env.step(0)
"""
import math
import os
import sys

import numpy as np
from gymnasium.spaces import Box, Discrete

_GZ_PYTHON = "/usr/local/lib/python"
if _GZ_PYTHON not in sys.path:
    sys.path.insert(0, _GZ_PYTHON)

from gz.sim8 import Joint, Model, World, world_entity  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gazebo_gymnasium.fixture_env import FixtureEnv  # noqa: E402

_FORCE_MAG = 10.0              # N applied to slider_to_cart (matches classic CartPole scale)
_MAX_CART_POS = 2.4            # m — episode terminates before ±4 m joint hard limit
_MAX_POLE_ANGLE = 12 * math.pi / 180  # rad (~12°)
_MAX_STEPS = 500


class CartPoleFixtureEnv(FixtureEnv):
    """CartPole using ECM-direct force control and observation reads.

    Action space: Discrete(2) — 0 = push left (−10 N), 1 = push right (+10 N)
    Observation:  [cart_pos, cart_vel, pole_angle, pole_ang_vel] (float32)
    Reward:       +1 per step while pole is upright
    """

    def __init__(self, sdf_path: str, steps_per_action: int = 5):
        obs_space = Box(
            low=np.array(
                [-_MAX_CART_POS * 2, -np.inf, -np.pi, -np.inf], dtype=np.float32
            ),
            high=np.array(
                [_MAX_CART_POS * 2, np.inf, np.pi, np.inf], dtype=np.float32
            ),
        )
        self._slider_joint: Joint = None  # type: ignore[assignment]
        self._pole_joint: Joint = None    # type: ignore[assignment]
        self._cart_pos = 0.0
        self._cart_vel = 0.0
        self._pole_angle = 0.0
        self._pole_ang_vel = 0.0
        super().__init__(sdf_path, obs_space, Discrete(2), steps_per_action)

    # ------------------------------------------------------------------ FixtureEnv hooks

    def configure(self, ecm):
        world = World(world_entity(ecm))
        model = Model(world.model_by_name(ecm, "cartpole"))
        self._slider_joint = Joint(model.joint_by_name(ecm, "slider_to_cart"))
        self._pole_joint = Joint(model.joint_by_name(ecm, "cart_to_pole"))
        # Must enable before position()/velocity() return non-None values.
        self._slider_joint.enable_position_check(ecm, True)
        self._slider_joint.enable_velocity_check(ecm, True)
        self._pole_joint.enable_position_check(ecm, True)
        self._pole_joint.enable_velocity_check(ecm, True)

    def apply_action_to_ecm(self, ecm, action):
        force = _FORCE_MAG if action == 1 else -_FORCE_MAG
        self._slider_joint.set_force(ecm, [force])

    def apply_reset(self, ecm):
        self._slider_joint.reset_position(ecm, [0.0])
        self._slider_joint.reset_velocity(ecm, [0.0])
        self._pole_joint.reset_position(ecm, [0.0])
        self._pole_joint.reset_velocity(ecm, [0.0])

    def read_observation(self, ecm) -> np.ndarray:
        sp = self._slider_joint.position(ecm)
        sv = self._slider_joint.velocity(ecm)
        pp = self._pole_joint.position(ecm)
        pv = self._pole_joint.velocity(ecm)
        if sp is not None:
            self._cart_pos = sp[0]
        if sv is not None:
            self._cart_vel = sv[0]
        if pp is not None:
            self._pole_angle = pp[0]
        if pv is not None:
            self._pole_ang_vel = pv[0]
        return np.array(
            [self._cart_pos, self._cart_vel, self._pole_angle, self._pole_ang_vel],
            dtype=np.float32,
        )

    # ------------------------------------------------------------------ Gymnasium interface

    def get_reward(self, action) -> float:
        return 1.0

    def is_terminated(self) -> bool:
        return (
            abs(self._pole_angle) > _MAX_POLE_ANGLE
            or abs(self._cart_pos) > _MAX_CART_POS
        )

    def is_truncated(self) -> bool:
        return self._current_step >= _MAX_STEPS

    def set_default_observation(self) -> np.ndarray:
        self._cart_pos = 0.0
        self._cart_vel = 0.0
        self._pole_angle = 0.0
        self._pole_ang_vel = 0.0
        return np.zeros(4, dtype=np.float32)

    def get_info(self) -> dict:
        return {
            "gz_episode": self._current_episode,
            "TimeLimit.truncated": self.is_truncated(),
        }
