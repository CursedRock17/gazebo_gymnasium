# Copyright 2026 Lucas Wendland
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Reacher sync-gate plugin.

Two-link planar arm. Forces on joint0 / joint1 drive the arm; the env
class scores based on fingertip-to-target distance (target is virtual,
managed env-side, not part of the SDF).

Protocol:
  /env/action (Float_V, in):  [action0, action1] each in [-1, 1].
                              Multiplied by GEAR=200 before publishing
                              as scalar force on joint0 / joint1.
  /env/state  (Float_V, out): 6 values
                              [theta0, theta1, dtheta0, dtheta1,
                               fingertip_x, fingertip_y]
                              The env expands these to the canonical
                              10-dim Gymnasium obs (sin/cos of angles,
                              plus the virtual target position).
"""

import math

from gz.msgs10.double_pb2 import Double
from gz.msgs10.float_v_pb2 import Float_V
from gz.msgs10.model_pb2 import Model as ModelMsg
from gz.msgs10.pose_v_pb2 import Pose_V
from gz.sim8 import World
from gz.sim8 import world_entity
from gz.transport13 import AdvertiseMessageOptions
from gz.transport13 import Node


# Matches MJCF `motor gear="200"`. Action range [-1, 1] -> force [-200, 200].
GEAR = 200.0


class ReacherSyncGate:

    def __init__(self):
        self.world_name = None
        self.agent_name = None
        self.frame_skip = 2

        self._pending_action = None
        self._ticks_since_action = 0
        self._action_active = False

        # Sensor cache.
        self._theta0 = 0.0
        self._theta1 = 0.0
        self._dtheta0 = 0.0
        self._dtheta1 = 0.0
        self._fingertip_x = 0.21  # default (arm fully extended along +X)
        self._fingertip_y = 0.0

        self._action_node = None
        self._state_node = None
        self._cmd0_node = None
        self._cmd1_node = None
        self._joint_state_node = None
        self._pose_node = None
        self._state_pub = None
        self._cmd0_pub = None
        self._cmd1_pub = None

    def configure(self, entity, element, ecm, eventManager):
        self.world_name = World(world_entity(ecm)).name(ecm)
        self.agent_name = element.get_string("agent_name") or "reacher"
        if not self.agent_name.startswith("/"):
            self.agent_name = "/" + self.agent_name

        sdf_frame_skip = int(element.get_double("frame_skip") or 0)
        if sdf_frame_skip > 0:
            self.frame_skip = sdf_frame_skip

        # Two cmd_force publishers, one per joint.
        self._cmd0_node = Node()
        self._cmd0_pub = self._cmd0_node.advertise(
            "/model" + self.agent_name + "/joint/joint0/cmd_force",
            Double, AdvertiseMessageOptions(),
        )
        self._cmd1_node = Node()
        self._cmd1_pub = self._cmd1_node.advertise(
            "/model" + self.agent_name + "/joint/joint1/cmd_force",
            Double, AdvertiseMessageOptions(),
        )

        # Joint state (theta + dtheta for joint0 / joint1).
        self._joint_state_node = Node()
        joint_topic = (
            "/world/" + self.world_name + "/model" + self.agent_name + "/joint_state"
        )
        self._joint_state_node.subscribe(ModelMsg, joint_topic, self._on_joint_state)

        # Pose info — used to recover the fingertip's world-frame x, y.
        self._pose_node = Node()
        pose_topic = "/world/" + self.world_name + "/pose/info"
        self._pose_node.subscribe(Pose_V, pose_topic, self._on_pose)

        # /env/action, /env/state.
        self._action_node = Node()
        self._action_node.subscribe(Float_V, "/env/action", self._on_action)
        self._state_node = Node()
        self._state_pub = self._state_node.advertise(
            "/env/state", Float_V, AdvertiseMessageOptions())

        print(f"[ReacherSyncGate] ready: world={self.world_name!r} "
              f"agent={self.agent_name!r} frame_skip={self.frame_skip} gear={GEAR}")

    def pre_update(self, info, ecm):
        if info.paused or self._pending_action is None:
            return
        force0 = Double()
        force1 = Double()
        if self._action_active:
            force0.data = float(self._pending_action[0]) * GEAR
            force1.data = float(self._pending_action[1]) * GEAR
        else:
            force0.data = 0.0
            force1.data = 0.0
        self._cmd0_pub.publish(force0)
        self._cmd1_pub.publish(force1)

    def post_update(self, info, ecm):
        if info.paused or self._pending_action is None or not self._action_active:
            return
        self._ticks_since_action += 1
        if self._ticks_since_action >= self.frame_skip:
            self._publish_state()
            self._action_active = False

    def reset(self, info, ecm):
        self._pending_action = None
        self._action_active = False
        self._ticks_since_action = 0
        self._theta0 = 0.0
        self._theta1 = 0.0
        self._dtheta0 = 0.0
        self._dtheta1 = 0.0
        self._fingertip_x = 0.21
        self._fingertip_y = 0.0
        self._publish_state()

    def _on_action(self, msg):
        if len(msg.data) < 2:
            return
        self._pending_action = (float(msg.data[0]), float(msg.data[1]))
        self._ticks_since_action = 0
        self._action_active = True

    def _on_joint_state(self, model_msg):
        for joint in model_msg.joint:
            if joint.name == "joint0":
                self._theta0 = joint.axis1.position
                self._dtheta0 = joint.axis1.velocity
            elif joint.name == "joint1":
                self._theta1 = joint.axis1.position
                self._dtheta1 = joint.axis1.velocity

    def _on_pose(self, pose_v_msg):
        # The fingertip is the end of `arm1` in the SDF — its world-frame
        # position has the +X offset of 0.11 m baked in by the arm1 link's
        # local frame. Track arm1's world pose, then push out by 0.11 along
        # arm1's local +X axis using the link's orientation.
        for pose in pose_v_msg.pose:
            if pose.name == "arm1":
                q = pose.orientation
                # 2D yaw from quaternion (only Z rotation matters here).
                yaw = math.atan2(
                    2.0 * (q.w * q.z + q.x * q.y),
                    1.0 - 2.0 * (q.y * q.y + q.z * q.z),
                )
                # arm1's local +X offset 0.11 m to the fingertip.
                self._fingertip_x = pose.position.x + 0.11 * math.cos(yaw)
                self._fingertip_y = pose.position.y + 0.11 * math.sin(yaw)

    def _publish_state(self):
        msg = Float_V()
        msg.data.extend([
            float(self._theta0),
            float(self._theta1),
            float(self._dtheta0),
            float(self._dtheta1),
            float(self._fingertip_x),
            float(self._fingertip_y),
        ])
        self._state_pub.publish(msg)


def get_system():
    """Expose the system so Gazebo's PythonSystemLoader can find it."""
    return ReacherSyncGate()
