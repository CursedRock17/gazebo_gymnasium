#!/usr/bin/env python3
"""
InvertedPendulum environment backed by Gazebo Sim.

Matches the InvertedPendulum-v5 Gymnasium/MuJoCo interface.

Observation (Box 4):
    [cart_pos, cart_vel, pole_angle, pole_ang_vel]

Action (Box 1):
    Continuous force in [-1, 1], scaled to ±300 N on the slider joint.
    (MuJoCo convention: gear=100, ctrlrange=[-3,3] → ±300 N max.)

Reward: +1.0 every step the pole stays upright.
Terminated: |pole_angle| > 0.2 rad or |cart_pos| > 1.0 m.
Truncated: episode exceeds 1000 steps.
"""
import numpy as np
from gymnasium.spaces import Box

from gz.transport13 import Node, AdvertiseMessageOptions
from gz.msgs10.double_pb2 import Double
from gz.msgs10.model_pb2 import Model

from gazebo_gymnasium import GazeboEnv

WORLD_NAME = "inverted_pendulum"
MODEL_NAME = "inverted_pendulum"

_CMD_TOPIC         = f"/model/{MODEL_NAME}/joint/slider/cmd_force"
_JOINT_STATE_TOPIC = f"/world/{WORLD_NAME}/model/{MODEL_NAME}/joint_state"

_FORCE_SCALE    = 300.0
_MAX_CART_POS   = 1.0
_MAX_POLE_ANGLE = 0.2
_MAX_STEPS      = 1000


class InvertedPendulumEnv(GazeboEnv):
    """Port of gymnasium InvertedPendulum-v5 to Gazebo Sim."""

    def __init__(self, steps_per_action: int = 4):
        obs_space = Box(
            low=np.array([-1.0, -np.inf, -np.pi, -np.inf], dtype=np.float32),
            high=np.array([1.0,  np.inf,  np.pi,  np.inf], dtype=np.float32),
        )
        act_space = Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        super().__init__(WORLD_NAME, obs_space, act_space, steps_per_action)

        self._gz_node = Node()
        self._cmd_pub = self._gz_node.advertise(
            _CMD_TOPIC, Double, AdvertiseMessageOptions()
        )

        self._state_node = Node()
        self._state_node.subscribe(Model, _JOINT_STATE_TOPIC, self._on_joint_state)

        self._cart_pos = 0.0
        self._cart_vel = 0.0
        self._pole_angle = 0.0
        self._pole_vel = 0.0

        self._ping_world_control()

    # ------------------------------------------------------------------ gz cb

    def _on_joint_state(self, msg: Model):
        for joint in msg.joint:
            if joint.name == "slider":
                self._cart_pos = joint.axis1.position
                self._cart_vel = joint.axis1.velocity
            elif joint.name == "hinge":
                self._pole_angle = joint.axis1.position
                self._pole_vel   = joint.axis1.velocity
        self._count_physics_step()

    # ------------------------------------------------------------------ GazeboEnv overrides

    def reset(self, seed=None, options=None):
        print(f"[reset] ep={self._current_episode + 1}", flush=True)
        zero = Double(); zero.data = 0.0
        self._cmd_pub.publish(zero)
        obs, info = super().reset(seed=seed, options=options)
        self._advance_physics(1)
        return self.set_default_observation(), info

    def apply_action(self, action):
        msg = Double()
        msg.data = float(np.clip(action[0], -1.0, 1.0)) * _FORCE_SCALE
        self._cmd_pub.publish(msg)

    def get_observation(self):
        return np.array([
            self._cart_pos,
            self._cart_vel,
            self._pole_angle,
            self._pole_vel,
        ], dtype=np.float32)

    def get_reward(self, action) -> float:
        return 1.0

    def is_terminated(self) -> bool:
        pole_bad = abs(self._pole_angle) > _MAX_POLE_ANGLE
        cart_bad = abs(self._cart_pos)   > _MAX_CART_POS
        if pole_bad or cart_bad:
            parts = []
            if pole_bad: parts.append(f"pole={self._pole_angle:+.4f}")
            if cart_bad: parts.append(f"cart={self._cart_pos:+.4f}")
            print(f"[TERM] step={self._current_step}  " + "  ".join(parts), flush=True)
        return pole_bad or cart_bad

    def is_truncated(self) -> bool:
        return self._current_step >= _MAX_STEPS

    def set_default_observation(self):
        self._cart_pos = self._cart_vel = 0.0
        self._pole_angle = self._pole_vel = 0.0
        return np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)

    def get_info(self) -> dict:
        return {
            "gz_episode": self._current_episode,
            "gz_step": self._current_step,
        }
