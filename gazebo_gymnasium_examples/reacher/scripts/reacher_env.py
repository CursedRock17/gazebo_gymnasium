#!/usr/bin/env python3
"""
Reacher environment backed by Gazebo Sim.

Matches the Reacher-v5 Gymnasium/MuJoCo interface.

Observation (Box 10):
    [cos(theta0), sin(theta0),   ← shoulder joint angle
     cos(theta1), sin(theta1),   ← elbow joint angle
     target_x, target_y,         ← target position (world XY)
     theta0_vel, theta1_vel,      ← joint angular velocities
     tip_x - target_x,           ← fingertip error X
     tip_y - target_y]           ← fingertip error Y

Action (Box 2):
    [torque0, torque1] in [-1, 1], scaled by gear=200 N·m each.

Reward: -dist(fingertip, target) - 0.01 * sum(action²)
Truncated: 50 steps (no early termination).
"""
import math

import numpy as np
from gymnasium.spaces import Box

from gz.transport13 import Node, AdvertiseMessageOptions
from gz.msgs10.double_pb2 import Double
from gz.msgs10.model_pb2 import Model
from gz.msgs10.pose_v_pb2 import Pose_V
from gz.msgs10.pose_pb2 import Pose
from gz.msgs10.boolean_pb2 import Boolean

from gazebo_gymnasium import GazeboEnv

WORLD_NAME  = "reacher"
MODEL_NAME  = "reacher"
TARGET_NAME = "reacher_target"

_JOINT_STATE_TOPIC = f"/world/{WORLD_NAME}/model/{MODEL_NAME}/joint_state"
_POSE_TOPIC        = f"/world/{WORLD_NAME}/dynamic_pose/info"
_SET_POSE_SRV      = f"/world/{WORLD_NAME}/set_pose"

_JOINTS   = ["joint0", "joint1"]
_GEAR     = 200.0
_CMD_FMT  = f"/model/{MODEL_NAME}/joint/{{name}}/cmd_force"

_L1       = 0.1
_L2       = 0.11
_Z_PLANE  = 0.01

_CTRL_COST  = 0.01
_MAX_STEPS  = 50
_TARGET_MIN = 0.05
_TARGET_MAX = 0.20


class ReacherEnv(GazeboEnv):
    """Port of gymnasium Reacher-v5 to Gazebo Sim."""

    def __init__(self, steps_per_action: int = 2):
        obs_space = Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32)
        act_space = Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        super().__init__(WORLD_NAME, obs_space, act_space, steps_per_action)

        self._gz_node = Node()
        opts = AdvertiseMessageOptions()
        self._pubs = {
            name: self._gz_node.advertise(_CMD_FMT.format(name=name), Double, opts)
            for name in _JOINTS
        }

        self._state_node = Node()
        self._state_node.subscribe(Model, _JOINT_STATE_TOPIC, self._on_joint_state)

        self._pose_node = Node()
        self._pose_node.subscribe(Pose_V, _POSE_TOPIC, self._on_pose)

        self._theta0 = 0.0; self._theta1 = 0.0
        self._vel0   = 0.0; self._vel1   = 0.0
        self._target_x = 0.1; self._target_y = -0.1

        self._ping_world_control()

    # ------------------------------------------------------------------ gz cbs

    def _on_joint_state(self, msg: Model):
        for joint in msg.joint:
            if joint.name == "joint0":
                self._theta0 = joint.axis1.position
                self._vel0   = joint.axis1.velocity
            elif joint.name == "joint1":
                self._theta1 = joint.axis1.position
                self._vel1   = joint.axis1.velocity
        self._count_physics_step()

    def _on_pose(self, msg: Pose_V):
        for pose in msg.pose:
            if pose.name == TARGET_NAME:
                self._target_x = pose.position.x
                self._target_y = pose.position.y

    # ------------------------------------------------------------------ GazeboEnv overrides

    def reset(self, seed=None, options=None):
        print(f"[reset] ep={self._current_episode + 1}", flush=True)
        z = Double(); z.data = 0.0
        for pub in self._pubs.values():
            pub.publish(z)

        rng = np.random.default_rng(seed)
        while True:
            angle  = rng.uniform(0, 2 * math.pi)
            radius = rng.uniform(_TARGET_MIN, _TARGET_MAX)
            tx = radius * math.cos(angle)
            ty = radius * math.sin(angle)
            if math.hypot(tx, ty) >= _TARGET_MIN:
                break
        self._target_x = tx
        self._target_y = ty
        self._move_target(tx, ty)

        obs, info = super().reset(seed=seed, options=options)
        self._advance_physics(1)
        return self.set_default_observation(), info

    def apply_action(self, action):
        for name, a in zip(_JOINTS, action):
            msg = Double()
            msg.data = float(np.clip(a, -1.0, 1.0)) * _GEAR
            self._pubs[name].publish(msg)

    def get_observation(self):
        tip_x, tip_y = self._fingertip_xy()
        return np.array([
            math.cos(self._theta0), math.sin(self._theta0),
            math.cos(self._theta1), math.sin(self._theta1),
            self._target_x, self._target_y,
            self._vel0, self._vel1,
            tip_x - self._target_x,
            tip_y - self._target_y,
        ], dtype=np.float32)

    def get_reward(self, action) -> float:
        tip_x, tip_y = self._fingertip_xy()
        dist = math.hypot(tip_x - self._target_x, tip_y - self._target_y)
        ctrl_cost = _CTRL_COST * float(np.sum(np.square(action)))
        return -dist - ctrl_cost

    def is_terminated(self) -> bool:
        return False

    def is_truncated(self) -> bool:
        return self._current_step >= _MAX_STEPS

    def set_default_observation(self):
        self._theta0 = self._theta1 = 0.0
        self._vel0   = self._vel1   = 0.0
        tip_x, tip_y = self._fingertip_xy()
        return np.array([
            math.cos(self._theta0), math.sin(self._theta0),
            math.cos(self._theta1), math.sin(self._theta1),
            self._target_x, self._target_y,
            self._vel0, self._vel1,
            tip_x - self._target_x,
            tip_y - self._target_y,
        ], dtype=np.float32)

    def get_info(self) -> dict:
        tip_x, tip_y = self._fingertip_xy()
        return {
            "gz_episode": self._current_episode,
            "gz_step": self._current_step,
            "dist_to_target": math.hypot(tip_x - self._target_x, tip_y - self._target_y),
        }

    # ------------------------------------------------------------------ helpers

    def _fingertip_xy(self):
        x = _L1 * math.cos(self._theta0) + _L2 * math.cos(self._theta0 + self._theta1)
        y = _L1 * math.sin(self._theta0) + _L2 * math.sin(self._theta0 + self._theta1)
        return x, y

    def _move_target(self, x: float, y: float):
        pose = Pose()
        pose.name = TARGET_NAME
        pose.position.x = x
        pose.position.y = y
        pose.position.z = _Z_PLANE
        pose.orientation.w = 1.0
        try:
            ok, result = self._gz_node.request(
                _SET_POSE_SRV, pose, Pose, Boolean, timeout=500
            )
            if not ok or not result.data:
                print(f"[WARN] set_pose failed for target ({x:.3f}, {y:.3f})", flush=True)
        except Exception as e:
            print(f"[WARN] set_pose exception: {e}", flush=True)
