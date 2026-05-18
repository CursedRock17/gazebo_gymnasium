#!/usr/bin/env python3
"""
Ant environment backed by Gazebo Sim.

Matches the Ant-v5 Gymnasium/MuJoCo interface.

Observation (Box 27):
    [torso_z,
     qw, qx, qy, qz,                              ← torso orientation quaternion
     hip_1, ankle_1, hip_2, ankle_2,               ← front joints
     hip_3, ankle_3, hip_4, ankle_4,               ← back joints
     vx, vy, vz,                                   ← torso linear velocity (finite diff)
     wx, wy, wz,                                   ← torso angular velocity (finite diff)
     hip_1_vel, ankle_1_vel, hip_2_vel, ankle_2_vel,
     hip_3_vel, ankle_3_vel, hip_4_vel, ankle_4_vel]

Action (Box 8):
    [hip_4, ankle_4, hip_1, ankle_1, hip_2, ankle_2, hip_3, ankle_3]
    in [-1, 1], scaled by gear=150 N·m each.

Reward: xy_velocity + 1.0 (alive) − 0.5 × Σ(action²)
Terminated: torso_z < 0.2 m or > 1.0 m.
Truncated: 1000 steps.
"""
import math

import numpy as np
from gymnasium.spaces import Box

from gz.transport13 import Node, AdvertiseMessageOptions
from gz.msgs10.double_pb2 import Double
from gz.msgs10.model_pb2 import Model
from gz.msgs10.pose_v_pb2 import Pose_V

from gazebo_gymnasium import GazeboEnv

WORLD_NAME = "ant"
MODEL_NAME = "ant"

_JOINT_STATE_TOPIC = f"/world/{WORLD_NAME}/model/{MODEL_NAME}/joint_state"
_POSE_TOPIC        = f"/world/{WORLD_NAME}/dynamic_pose/info"

_JOINTS = ["hip_4", "ankle_4", "hip_1", "ankle_1",
           "hip_2", "ankle_2", "hip_3", "ankle_3"]
_GEAR = 150.0
_CMD_FMT = "/model/" + MODEL_NAME + "/joint/{name}/cmd_force"

_CTRL_COST = 0.5
_MIN_Z = 0.2
_MAX_Z = 1.0
_MAX_STEPS = 1000
_DT = 0.01


class AntEnv(GazeboEnv):
    """Port of gymnasium Ant-v5 to Gazebo Sim."""

    def __init__(self, steps_per_action: int = 5):
        obs_space = Box(low=-np.inf, high=np.inf, shape=(27,), dtype=np.float32)
        act_space = Box(low=-1.0, high=1.0, shape=(8,), dtype=np.float32)
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

        self._tx = 0.0; self._ty = 0.0; self._tz = 0.75
        self._qw = 1.0; self._qx = 0.0; self._qy = 0.0; self._qz = 0.0
        self._prev_tx = 0.0; self._prev_ty = 0.0; self._prev_tz = 0.75
        self._prev_qw = 1.0; self._prev_qx = 0.0; self._prev_qy = 0.0; self._prev_qz = 0.0

        self._all_joints = [
            "hip_1", "ankle_1", "hip_2", "ankle_2",
            "hip_3", "ankle_3", "hip_4", "ankle_4",
        ]
        self._joint_pos = {n: 0.0 for n in self._all_joints}
        self._joint_vel = {n: 0.0 for n in self._all_joints}

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
                self._tx = pose.position.x
                self._ty = pose.position.y
                self._tz = pose.position.z
                self._qw = pose.orientation.w
                self._qx = pose.orientation.x
                self._qy = pose.orientation.y
                self._qz = pose.orientation.z
                break

    # ------------------------------------------------------------------ GazeboEnv overrides

    def reset(self, seed=None, options=None):
        print(f"[reset] ep={self._current_episode + 1}", flush=True)
        z = Double()
        z.data = 0.0
        for pub in self._pubs.values():
            pub.publish(z)
        obs, info = super().reset(seed=seed, options=options)
        # Wait for first fresh joint state confirming reset took effect
        self._advance_physics(1)
        return self.set_default_observation(), info

    def apply_action(self, action):
        # Save previous pose for velocity finite-diff before advancing
        self._prev_tx = self._tx; self._prev_ty = self._ty; self._prev_tz = self._tz
        self._prev_qw = self._qw; self._prev_qx = self._qx
        self._prev_qy = self._qy; self._prev_qz = self._qz
        for name, a in zip(_JOINTS, action):
            msg = Double()
            msg.data = float(np.clip(a, -1.0, 1.0)) * _GEAR
            self._pubs[name].publish(msg)

    def get_observation(self):
        vx = (self._tx - self._prev_tx) / self._act_dt
        vy = (self._ty - self._prev_ty) / self._act_dt
        vz = (self._tz - self._prev_tz) / self._act_dt
        dw  = self._qw*self._prev_qw + self._qx*self._prev_qx + self._qy*self._prev_qy + self._qz*self._prev_qz
        dqx = self._prev_qw*self._qx - self._prev_qx*self._qw + self._prev_qy*self._qz - self._prev_qz*self._qy
        dqy = self._prev_qw*self._qy - self._prev_qy*self._qw + self._prev_qz*self._qx - self._prev_qx*self._qz
        dqz = self._prev_qw*self._qz - self._prev_qz*self._qw + self._prev_qx*self._qy - self._prev_qy*self._qx
        scale = 2.0 / max(abs(dw), 1e-6) / self._act_dt
        wx = dqx * scale; wy = dqy * scale; wz = dqz * scale
        return np.array([
            self._tz,
            self._qw, self._qx, self._qy, self._qz,
            self._joint_pos["hip_1"],   self._joint_pos["ankle_1"],
            self._joint_pos["hip_2"],   self._joint_pos["ankle_2"],
            self._joint_pos["hip_3"],   self._joint_pos["ankle_3"],
            self._joint_pos["hip_4"],   self._joint_pos["ankle_4"],
            vx, vy, vz, wx, wy, wz,
            self._joint_vel["hip_1"],   self._joint_vel["ankle_1"],
            self._joint_vel["hip_2"],   self._joint_vel["ankle_2"],
            self._joint_vel["hip_3"],   self._joint_vel["ankle_3"],
            self._joint_vel["hip_4"],   self._joint_vel["ankle_4"],
        ], dtype=np.float32)

    def get_reward(self, action) -> float:
        xy_vel = math.sqrt(
            ((self._tx - self._prev_tx) / self._act_dt) ** 2 +
            ((self._ty - self._prev_ty) / self._act_dt) ** 2
        )
        ctrl_cost = _CTRL_COST * float(np.sum(np.square(action)))
        return xy_vel + 1.0 - ctrl_cost

    def is_terminated(self) -> bool:
        bad = self._tz < _MIN_Z or self._tz > _MAX_Z
        if bad:
            print(f"[TERM] step={self._current_step}  z={self._tz:.3f}", flush=True)
        return bad

    def is_truncated(self) -> bool:
        return self._current_step >= _MAX_STEPS

    def set_default_observation(self):
        self._tx = self._ty = 0.0; self._tz = 0.75
        self._qw = 1.0; self._qx = self._qy = self._qz = 0.0
        self._prev_tx = self._prev_ty = 0.0; self._prev_tz = 0.75
        self._prev_qw = 1.0; self._prev_qx = self._prev_qy = self._prev_qz = 0.0
        for n in self._all_joints:
            self._joint_pos[n] = 0.0
            self._joint_vel[n] = 0.0
        obs = np.zeros(27, dtype=np.float32)
        obs[0] = 0.75
        obs[1] = 1.0
        return obs

    def get_info(self) -> dict:
        return {
            "gz_episode": self._current_episode,
            "gz_step": self._current_step,
            "torso_z": self._tz,
        }
