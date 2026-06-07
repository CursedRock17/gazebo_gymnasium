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

"""inverted_double_pendulum_learner — auto-generated SKELETON sync-gate plugin.

Status: NOT IMPLEMENTED. The model loads, but this plugin currently does
nothing with actions or state. To wire it up:

  1. Decide on action mapping (cmd_force / cmd_vel / cmd_pos) for each
     actuated joint. Browse the model SDF at
     gazebo_gymnasium_resources/models/inverted_double_pendulum/inverted_double_pendulum.sdf
     for joint names and the actuator plugin attached.
  2. Decide observation vector (look at canonical Gymnasium
     `gymnasium/envs/mujoco/inverted_double_pendulum_v5.py` for the reward
     + obs layout).
  3. Copy the body of cartpole_learner.py or inverted_pendulum_learner.py
     and adapt for this env's joints/links.

The empty `pre_update` / `post_update` here lets you `ros2 launch` the env
and verify the model renders without errors before doing the wiring work.
"""

from gz.msgs10.float_v_pb2 import Float_V
from gz.sim8 import World
from gz.sim8 import world_entity
from gz.transport13 import AdvertiseMessageOptions
from gz.transport13 import Node


class InvertedDoublePendulumSyncGate:

    def __init__(self):
        self.world_name = None
        self.agent_name = None
        self.frame_skip = 5
        self._action_node = None
        self._state_node = None
        self._state_pub = None
        self._has_warned = False

    def configure(self, entity, element, ecm, eventManager):
        self.world_name = World(world_entity(ecm)).name(ecm)
        self.agent_name = element.get_string("agent_name") or "inverted_double_pendulum"
        if not self.agent_name.startswith("/"):
            self.agent_name = "/" + self.agent_name
        sdf_frame_skip = int(element.get_double("frame_skip") or 0)
        if sdf_frame_skip > 0:
            self.frame_skip = sdf_frame_skip

        self._action_node = Node()
        self._action_node.subscribe(Float_V, "/env/action", self._on_action)
        self._state_node = Node()
        self._state_pub = self._state_node.advertise(
            "/env/state", Float_V, AdvertiseMessageOptions())

        print(
            f"[InvertedDoublePendulumSyncGate] LOADED but UNIMPLEMENTED "
            f"for env={self.agent_name!r}")
        print(
            "[InvertedDoublePendulumSyncGate] Actions arriving on "
            "/env/action will be ignored.")
        print(
            "[InvertedDoublePendulumSyncGate] Implement pre_update/post_update "
            "+ topic callbacks following the cartpole_learner.py / "
            "inverted_pendulum_learner.py pattern.")

    def pre_update(self, info, ecm):
        pass

    def post_update(self, info, ecm):
        pass

    def reset(self, info, ecm):
        msg = Float_V()
        msg.data.extend([0.0, 0.0, 0.0, 0.0])
        if self._state_pub is not None:
            self._state_pub.publish(msg)

    def _on_action(self, msg):
        if not self._has_warned:
            print(
                "[InvertedDoublePendulumSyncGate] received action — but "
                "plugin is UNIMPLEMENTED.")
            self._has_warned = True


def get_system():
    return InvertedDoublePendulumSyncGate()
