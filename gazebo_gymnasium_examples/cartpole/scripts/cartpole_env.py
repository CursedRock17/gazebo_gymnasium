#!/usr/bin/env python3
"""
CartPole environment backed by Gazebo Sim.

Matches the classic CartPole-v1 interface. Compatible with any Gymnasium-compatible
RL library — SB3, RLlib, CleanRL, or a hand-rolled agent.
"""
import time

import numpy as np
from gymnasium.spaces import Box, Discrete

from gz.transport13 import Node, AdvertiseMessageOptions
from gz.msgs10.double_pb2 import Double
from gz.msgs10.model_pb2 import Model

from gazebo_gymnasium import GazeboEnv

WORLD_NAME = "cartpole"
MODEL_NAME = "cartpole"

_CMD_TOPIC = f"/model/{MODEL_NAME}/joint/slider_to_cart/0/cmd_pos"
_JOINT_STATE_TOPIC = f"/world/{WORLD_NAME}/model/{MODEL_NAME}/joint_state"

# Target positions for the joint position controller.
# Keep these well inside the termination bounds (±2.4 m) so that position
# termination is a last resort, not the default failure mode.  A ±0.5 m nudge
# produces force-impulse-like behaviour similar to the classic CartPole env.
_POS_LEFT = -0.5
_POS_RIGHT = 0.5

# Episode termination thresholds (matching CartPole-v1)
_MAX_CART_POS = 2.4      # meters
_MAX_POLE_ANGLE = 0.20944  # ~12 degrees in radians
_MAX_STEPS = 500


class CartPoleEnv(GazeboEnv):
    """
    Observation space (Box, 4):
        [cart_position, cart_velocity, pole_angle, pole_angular_velocity]

    Action space (Discrete 2):
        0 — push cart left
        1 — push cart right

    Reward: +1 for every step the pole remains upright.
    Terminated: pole angle > 12° or cart position > 2.4m from center.
    Truncated: episode exceeds _MAX_STEPS steps.
    """

    def __init__(self, steps_per_action: int = 10):
        # Bounds must cover all values the physics sim can produce during a single
        # action step (steps_per_action ticks of 10ms = 100ms).  The pole can swing
        # far past the 12° termination threshold in that window, so we use ±π for
        # angle and ±inf for velocities.  Termination still fires at 12°; these are
        # just the observation space limits used by SB3 for normalisation checks.
        obs_space = Box(
            low=np.array([-4.8, -np.inf, -np.pi, -np.inf], dtype=np.float32),
            high=np.array([4.8,  np.inf,  np.pi,  np.inf], dtype=np.float32),
        )
        super().__init__(WORLD_NAME, obs_space, Discrete(2), steps_per_action)

        # Publisher: send target position to the cart's joint position controller
        self._cmd_node = Node()
        self._cmd_pub = self._cmd_node.advertise(_CMD_TOPIC, Double, AdvertiseMessageOptions())

        # Subscriber: read joint positions and velocities after each physics step
        self._state_node = Node()
        self._state_node.subscribe(Model, _JOINT_STATE_TOPIC, self._on_joint_state)

        self._cart_position = 0.0
        self._cart_velocity = 0.0
        self._pole_angle = 0.0
        self._pole_ang_velocity = 0.0

        # Allow subscriber to establish connection before training starts
        time.sleep(0.5)

    # --- gz.transport callback ---

    def _on_joint_state(self, msg: Model):
        for joint in msg.joint:
            if joint.name == "slider_to_cart":
                self._cart_position = joint.axis1.position
                self._cart_velocity = joint.axis1.velocity
            elif joint.name == "cart_to_pole":
                self._pole_angle = joint.axis1.position
                self._pole_ang_velocity = joint.axis1.velocity

    # --- GazeboEnv abstract methods ---

    def apply_action(self, action: int):
        msg = Double()
        msg.data = _POS_LEFT if action == 0 else _POS_RIGHT
        self._cmd_pub.publish(msg)

    def get_observation(self):
        return np.array([
            self._cart_position,
            self._cart_velocity,
            self._pole_angle,
            self._pole_ang_velocity,
        ], dtype=np.float32)

    def get_reward(self, action) -> float:
        return 1.0

    def is_terminated(self) -> bool:
        return (
            abs(self._pole_angle) > _MAX_POLE_ANGLE
            or abs(self._cart_position) > _MAX_CART_POS
        )

    def is_truncated(self) -> bool:
        return self._current_step >= _MAX_STEPS

    def set_default_observation(self):
        self._cart_position = 0.0
        self._cart_velocity = 0.0
        self._pole_angle = 0.0
        self._pole_ang_velocity = 0.0
        return self.get_observation()

    def get_info(self) -> dict:
        return {
            "gz_episode": self._current_episode,  # "episode" is reserved by SB3
            "gz_step": self._current_step,
            # Required by SB3 for correct value estimates at truncation boundaries
            "TimeLimit.truncated": self.is_truncated(),
        }
