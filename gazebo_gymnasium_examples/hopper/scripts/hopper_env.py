#!/usr/bin/env python3
"""
Hopper environment backed by Gazebo Sim.

Matches the Hopper-v5 Gymnasium/MuJoCo interface.

Observation (Box 11):
    [rootz, rooty, thigh, leg, foot,
     rootx_vel, rootz_vel, rooty_vel, thigh_vel, leg_vel, foot_vel]
    rootz = torso height, rooty = torso pitch angle.
    root velocities computed by finite difference over steps_per_action.

Action (Box 3):
    Torques on [thigh_joint, leg_joint, foot_joint] in [-1, 1].
    Scaled internally by gear=200 → ±200 N·m.

Reward: forward_velocity + 1.0 (alive) − 0.001 × Σ(action²)
Terminated: torso height < 0.7 m OR |torso_pitch| > 0.2 rad.
Truncated: episode exceeds 1000 steps.
"""
import math

import numpy as np
from gymnasium.spaces import Box

from gz.transport13 import Node, AdvertiseMessageOptions
from gz.msgs10.double_pb2 import Double
from gz.msgs10.model_pb2 import Model
from gz.msgs10.pose_v_pb2 import Pose_V

from gazebo_gymnasium import GazeboEnv

WORLD_NAME = "hopper"
MODEL_NAME = "hopper"

_THIGH_CMD = f"/model/{MODEL_NAME}/joint/thigh_joint/cmd_force"
_LEG_CMD   = f"/model/{MODEL_NAME}/joint/leg_joint/cmd_force"
_FOOT_CMD  = f"/model/{MODEL_NAME}/joint/foot_joint/cmd_force"
_JOINT_STATE_TOPIC = f"/world/{WORLD_NAME}/model/{MODEL_NAME}/joint_state"
_POSE_TOPIC        = f"/world/{WORLD_NAME}/dynamic_pose/info"

_GEAR = 200.0
_CTRL_COST = 0.001
_MIN_Z = 0.7
_MAX_ANGLE = 0.2
_MAX_STEPS = 1000
_DT = 0.01


class HopperEnv(GazeboEnv):
    """Port of gymnasium Hopper-v5 to Gazebo Sim."""

    def __init__(self, steps_per_action: int = 4):
        obs_space = Box(low=-np.inf, high=np.inf, shape=(11,), dtype=np.float32)
        act_space = Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        super().__init__(WORLD_NAME, obs_space, act_space, steps_per_action)
        self._act_dt = steps_per_action * _DT

        self._gz_node = Node()
        opts = AdvertiseMessageOptions()
        self._thigh_pub = self._gz_node.advertise(_THIGH_CMD, Double, opts)
        self._leg_pub   = self._gz_node.advertise(_LEG_CMD,   Double, opts)
        self._foot_pub  = self._gz_node.advertise(_FOOT_CMD,  Double, opts)

        self._state_node = Node()
        self._state_node.subscribe(Model, _JOINT_STATE_TOPIC, self._on_joint_state)

        self._pose_node = Node()
        self._pose_node.subscribe(Pose_V, _POSE_TOPIC, self._on_pose)

        self._rootx = 0.0;  self._rootz = 1.25; self._rooty = 0.0
        self._prev_rootx = 0.0; self._prev_rootz = 1.25; self._prev_rooty = 0.0
        self._thigh = 0.0; self._leg = 0.0; self._foot = 0.0
        self._thigh_vel = 0.0; self._leg_vel = 0.0; self._foot_vel = 0.0

        self._ping_world_control()

    # ------------------------------------------------------------------ gz cbs

    def _on_joint_state(self, msg: Model):
        for joint in msg.joint:
            if joint.name == "thigh_joint":
                self._thigh     = joint.axis1.position
                self._thigh_vel = joint.axis1.velocity
            elif joint.name == "leg_joint":
                self._leg     = joint.axis1.position
                self._leg_vel = joint.axis1.velocity
            elif joint.name == "foot_joint":
                self._foot     = joint.axis1.position
                self._foot_vel = joint.axis1.velocity
        self._count_physics_step()

    def _on_pose(self, msg: Pose_V):
        for pose in msg.pose:
            if pose.name in (MODEL_NAME, f"{MODEL_NAME}::torso", "torso"):
                self._rootx = pose.position.x
                self._rootz = pose.position.z
                q = pose.orientation
                self._rooty = math.atan2(
                    2.0 * (q.w * q.y - q.z * q.x),
                    1.0 - 2.0 * (q.y * q.y + q.z * q.z),
                )
                break

    # ------------------------------------------------------------------ GazeboEnv overrides

    def reset(self, seed=None, options=None):
        print(f"[reset] ep={self._current_episode + 1}", flush=True)
        z = Double(); z.data = 0.0
        self._thigh_pub.publish(z)
        self._leg_pub.publish(z)
        self._foot_pub.publish(z)
        obs, info = super().reset(seed=seed, options=options)
        self._advance_physics(1)
        return self.set_default_observation(), info

    def apply_action(self, action):
        self._prev_rootx = self._rootx
        self._prev_rootz = self._rootz
        self._prev_rooty = self._rooty
        for pub, a in zip([self._thigh_pub, self._leg_pub, self._foot_pub], action):
            msg = Double()
            msg.data = float(np.clip(a, -1.0, 1.0)) * _GEAR
            pub.publish(msg)

    def get_observation(self):
        rootx_vel = (self._rootx - self._prev_rootx) / self._act_dt
        rootz_vel = (self._rootz - self._prev_rootz) / self._act_dt
        rooty_vel = (self._rooty - self._prev_rooty) / self._act_dt
        return np.array([
            self._rootz, self._rooty,
            self._thigh, self._leg, self._foot,
            rootx_vel, rootz_vel, rooty_vel,
            self._thigh_vel, self._leg_vel, self._foot_vel,
        ], dtype=np.float32)

    def get_reward(self, action) -> float:
        rootx_vel = (self._rootx - self._prev_rootx) / self._act_dt
        ctrl_cost = _CTRL_COST * float(np.sum(np.square(action)))
        return rootx_vel + 1.0 - ctrl_cost

    def is_terminated(self) -> bool:
        z_bad     = self._rootz < _MIN_Z
        angle_bad = abs(self._rooty) > _MAX_ANGLE
        if z_bad or angle_bad:
            parts = []
            if z_bad:     parts.append(f"z={self._rootz:.3f}")
            if angle_bad: parts.append(f"rooty={self._rooty:+.4f}")
            print(f"[TERM] step={self._current_step}  " + "  ".join(parts), flush=True)
        return z_bad or angle_bad

    def is_truncated(self) -> bool:
        return self._current_step >= _MAX_STEPS

    def set_default_observation(self):
        self._rootx = 0.0; self._rootz = 1.25; self._rooty = 0.0
        self._prev_rootx = 0.0; self._prev_rootz = 1.25; self._prev_rooty = 0.0
        self._thigh = self._leg = self._foot = 0.0
        self._thigh_vel = self._leg_vel = self._foot_vel = 0.0
        return np.array([1.25, 0.0, 0.0, 0.0, 0.0,
                         0.0,  0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

    def get_info(self) -> dict:
        return {
            "gz_episode": self._current_episode,
            "gz_step": self._current_step,
            "torso_z": self._rootz,
        }
