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

"""Point sync-gate plugin.

Holonomic 2D point navigation. Two prismatic joints (`ballx`, `bally`)
slide the body along the world's X and Y axes — the rover is essentially
a top-down dot. The env class wraps a virtual target around it; this
plugin just translates action commands to joint forces and publishes
position + velocity back.

Protocol:
  /env/action (Float_V, in):  [fx, fy] in [-1, 1].
                              Each multiplied by GEAR=20 before being
                              published as scalar force on ballx / bally.
  /env/state  (Float_V, out): 4 values
                              [x, y, vx, vy]
                              The env adds the (virtual) target and
                              builds the canonical obs vector.
"""

from gz.msgs10.double_pb2 import Double
from gz.msgs10.float_v_pb2 import Float_V
from gz.msgs10.model_pb2 import Model as ModelMsg
from gz.sim8 import World
from gz.sim8 import world_entity
from gz.transport13 import AdvertiseMessageOptions
from gz.transport13 import Node


# Force per unit action. With body mass = 1 kg and damping = 1, this gives
# a terminal velocity of ~20 m/s with a 1 s time constant — fast enough to
# cross the arena in a few seconds, slow enough that PPO can learn the
# steady-state response.
GEAR = 20.0


class PointSyncGate:

    def __init__(self):
        self.world_name = None
        self.agent_name = None
        self.frame_skip = 4

        self._pending_action = None
        self._ticks_since_action = 0
        self._action_active = False

        # Sensor cache — directly from /joint_state.
        self._x = 0.0
        self._y = 0.0
        self._vx = 0.0
        self._vy = 0.0

        self._action_node = None
        self._state_node = None
        self._cmd_x_node = None
        self._cmd_y_node = None
        self._joint_state_node = None
        self._state_pub = None
        self._cmd_x_pub = None
        self._cmd_y_pub = None

    def configure(self, entity, element, ecm, eventManager):
        self.world_name = World(world_entity(ecm)).name(ecm)
        self.agent_name = element.get_string("agent_name") or "point"
        if not self.agent_name.startswith("/"):
            self.agent_name = "/" + self.agent_name

        sdf_frame_skip = int(element.get_double("frame_skip") or 0)
        if sdf_frame_skip > 0:
            self.frame_skip = sdf_frame_skip

        self._cmd_x_node = Node()
        self._cmd_x_pub = self._cmd_x_node.advertise(
            "/model" + self.agent_name + "/joint/ballx/cmd_force",
            Double, AdvertiseMessageOptions(),
        )
        self._cmd_y_node = Node()
        self._cmd_y_pub = self._cmd_y_node.advertise(
            "/model" + self.agent_name + "/joint/bally/cmd_force",
            Double, AdvertiseMessageOptions(),
        )

        self._joint_state_node = Node()
        joint_topic = (
            "/world/" + self.world_name + "/model" + self.agent_name + "/joint_state"
        )
        self._joint_state_node.subscribe(ModelMsg, joint_topic, self._on_joint_state)

        self._action_node = Node()
        self._action_node.subscribe(Float_V, "/env/action", self._on_action)
        self._state_node = Node()
        self._state_pub = self._state_node.advertise(
            "/env/state", Float_V, AdvertiseMessageOptions())

        print(f"[PointSyncGate] ready: world={self.world_name!r} "
              f"agent={self.agent_name!r} frame_skip={self.frame_skip} gear={GEAR}")

    def pre_update(self, info, ecm):
        if info.paused or self._pending_action is None:
            return
        fx = Double()
        fy = Double()
        if self._action_active:
            fx.data = float(self._pending_action[0]) * GEAR
            fy.data = float(self._pending_action[1]) * GEAR
        else:
            fx.data = 0.0
            fy.data = 0.0
        self._cmd_x_pub.publish(fx)
        self._cmd_y_pub.publish(fy)

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
        self._x = 0.0
        self._y = 0.0
        self._vx = 0.0
        self._vy = 0.0
        self._publish_state()

    def _on_action(self, msg):
        if len(msg.data) < 2:
            return
        self._pending_action = (float(msg.data[0]), float(msg.data[1]))
        self._ticks_since_action = 0
        self._action_active = True

    def _on_joint_state(self, model_msg):
        for joint in model_msg.joint:
            if joint.name == "ballx":
                self._x = joint.axis1.position
                self._vx = joint.axis1.velocity
            elif joint.name == "bally":
                self._y = joint.axis1.position
                self._vy = joint.axis1.velocity

    def _publish_state(self):
        msg = Float_V()
        msg.data.extend([
            float(self._x),
            float(self._y),
            float(self._vx),
            float(self._vy),
        ])
        self._state_pub.publish(msg)


def get_system():
    """Expose the system so Gazebo's PythonSystemLoader can find it."""
    return PointSyncGate()
