#!/usr/bin/env python3
"""
HalfCheetah environment backed by Gazebo Sim.

Matches the HalfCheetah-v5 Gymnasium/MuJoCo interface.

Observation (Box 17):
    [rootz, rooty,
     bthigh, bshin, bfoot, fthigh, fshin, ffoot,
     rootx_vel, rootz_vel, rooty_vel,
     bthigh_vel, bshin_vel, bfoot_vel,
     fthigh_vel, fshin_vel, ffoot_vel]

Action (Box 6):
    [bthigh, bshin, bfoot, fthigh, fshin, ffoot] in [-1, 1].
    Scaled by per-joint gear: [120, 90, 60, 120, 60, 30] N·m.

Reward: forward_velocity − 0.1 × Σ(action²)
    (no termination — truncation only at 1000 steps)
"""
import math

import numpy as np
from gymnasium.spaces import Box

from gz.transport13 import Node, AdvertiseMessageOptions
from gz.msgs10.double_pb2 import Double
from gz.msgs10.model_pb2 import Model
from gz.msgs10.pose_v_pb2 import Pose_V

from gazebo_gymnasium import GazeboEnv

WORLD_NAME = "half_cheetah"
MODEL_NAME = "half_cheetah"

_JOINT_STATE_TOPIC = f"/world/{WORLD_NAME}/model/{MODEL_NAME}/joint_state"
_POSE_TOPIC        = f"/world/{WORLD_NAME}/dynamic_pose/info"

_JOINTS  = ["bthigh", "bshin", "bfoot", "fthigh", "fshin", "ffoot"]
_GEARS   = [120.0,     90.0,    60.0,    120.0,    60.0,    30.0]
_CMD_FMT = "/model/" + MODEL_NAME + "/joint/{name}/cmd_force"

_CTRL_COST = 0.1
_MAX_STEPS = 1000
_DT = 0.01


class HalfCheetahEnv(GazeboEnv):
    """Port of gymnasium HalfCheetah-v5 to Gazebo Sim."""

    def __init__(self, steps_per_action: int = 5):
        obs_space = Box(low=-np.inf, high=np.inf, shape=(17,), dtype=np.float32)
        act_space = Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)
        super().__init__(WORLD_NAME, obs_space, act_space, steps_per_action)
        self._act_dt = steps_per_action * _DT

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

        self._rootx = 0.0; self._rootz = 0.7; self._rooty = 0.0
        self._prev_rootx = 0.0; self._prev_rootz = 0.7; self._prev_rooty = 0.0
        self._joint_pos = {n: 0.0 for n in _JOINTS}
        self._joint_vel = {n: 0.0 for n in _JOINTS}

        self._ping_world_control()

    # ------------------------------------------------------------------ gz cbs

    def _on_joint_state(self, msg: Model):
        for joint in msg.joint:
            if joint.name in self._joint_pos:
                self._joint_pos[joint.name] = joint.axis1.position
                self._joint_vel[joint.name] = joint.axis1.velocity
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
        for pub in self._pubs.values():
            pub.publish(z)
        obs, info = super().reset(seed=seed, options=options)
        self._advance_physics(1)
        return self.set_default_observation(), info

    def apply_action(self, action):
        self._prev_rootx = self._rootx
        self._prev_rootz = self._rootz
        self._prev_rooty = self._rooty
        for name, a, gear in zip(_JOINTS, action, _GEARS):
            msg = Double()
            msg.data = float(np.clip(a, -1.0, 1.0)) * gear
            self._pubs[name].publish(msg)

    def get_observation(self):
        rootx_vel = (self._rootx - self._prev_rootx) / self._act_dt
        rootz_vel = (self._rootz - self._prev_rootz) / self._act_dt
        rooty_vel = (self._rooty - self._prev_rooty) / self._act_dt
        return np.array([
            self._rootz, self._rooty,
            self._joint_pos["bthigh"], self._joint_pos["bshin"],
            self._joint_pos["bfoot"],  self._joint_pos["fthigh"],
            self._joint_pos["fshin"],  self._joint_pos["ffoot"],
            rootx_vel, rootz_vel, rooty_vel,
            self._joint_vel["bthigh"], self._joint_vel["bshin"],
            self._joint_vel["bfoot"],  self._joint_vel["fthigh"],
            self._joint_vel["fshin"],  self._joint_vel["ffoot"],
        ], dtype=np.float32)

    def get_reward(self, action) -> float:
        rootx_vel = (self._rootx - self._prev_rootx) / self._act_dt
        ctrl_cost = _CTRL_COST * float(np.sum(np.square(action)))
        return rootx_vel - ctrl_cost

    def is_terminated(self) -> bool:
        return False

    def is_truncated(self) -> bool:
        return self._current_step >= _MAX_STEPS

    def set_default_observation(self):
        self._rootx = 0.0; self._rootz = 0.7; self._rooty = 0.0
        self._prev_rootx = self._prev_rootz = self._prev_rooty = 0.0
        for name in _JOINTS:
            self._joint_pos[name] = 0.0
            self._joint_vel[name] = 0.0
        return np.zeros(17, dtype=np.float32)

    def get_info(self) -> dict:
        return {
            "gz_episode": self._current_episode,
            "gz_step": self._current_step,
        }
