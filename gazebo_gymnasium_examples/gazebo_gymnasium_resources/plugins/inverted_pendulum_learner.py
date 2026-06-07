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

"""
InvertedPendulum sync-gate plugin.

Same protocol as `cartpole_learner.py`:
  /env/action  (Float_V, in)  — single-element continuous action in [-3, 3].
  /env/state   (Float_V, out) — 4 sensor values [cart_pos, pole_angle,
                                cart_vel, pole_ang_vel] published every
                                `frame_skip` post_updates after an action.

Differences from cartpole_learner:
  - Action is CONTINUOUS (Box(1,)) instead of Discrete(2). The float value
    arrives in `msg.data[0]` directly.
  - Action is applied as a JOINT FORCE on the slider, not a velocity command.
    Topic is `cmd_force` (Double), serviced by gz-sim-apply-joint-force-system.
  - Force = action * gear (gear=100, matches MuJoCo's `motor gear="100"`).
  - Observation order matches Gymnasium's InvertedPendulum-v5:
    [qpos[0], qpos[1], qvel[0], qvel[1]] = [cart_pos, pole_angle, cart_vel, pole_ang_vel].
    CartPole-v1 uses position-velocity interleaving; InvertedPendulum doesn't.
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

# Match MuJoCo's `motor gear="100"` so the action range [-3, 3] from the
# canonical env produces ±300 N on the slider — the same effective
# actuation as the original.
GEAR = 100.0


class InvertedPendulumSyncGate:
    """Sync gate between an external GazeboInvertedPendulumEnv and the live sim.

    Action is a scalar force (continuous), state is 4 floats.
    """

    def __init__(self):
        self.world_name = None
        self.agent_name = None
        self.frame_skip = 4

        # Action handling
        self._pending_action = None       # latest float action in [-3, 3]
        self._ticks_since_action = 0
        # Grace period: when False, publishes cmd_force=0 so the cart isn't
        # driven during agent thinking time (matches the cartpole pattern;
        # for SAC trainings that pause to update the network, this prevents
        # runaway motion).
        self._action_active = False

        # Sensor cache, updated by topic callbacks.
        self.cart_position = 0.0
        self.cart_velocity = 0.0
        self.pole_angle = 0.0
        self.pole_ang_velocity = 0.0

        # Transport nodes
        self._action_node = None
        self._state_node = None
        self._cmd_force_node = None
        self._pose_node = None
        self._joint_state_node = None

        self._state_pub = None
        self._cmd_force_pub = None

    # ------------------------------------------------------------------ #
    # Gazebo lifecycle hooks
    # ------------------------------------------------------------------ #

    def configure(self, entity, element, ecm, eventManager):
        self.world_name = World(world_entity(ecm)).name(ecm)
        self.agent_name = element.get_string("agent_name") or "inverted_pendulum"
        if not self.agent_name.startswith("/"):
            self.agent_name = "/" + self.agent_name

        sdf_frame_skip = int(element.get_double("frame_skip") or 0)
        if sdf_frame_skip > 0:
            self.frame_skip = sdf_frame_skip

        # cmd_force publisher → drives the ApplyJointForce plugin in the model.
        self._cmd_force_node = Node()
        cmd_force_topic = "/model" + self.agent_name + "/joint/slider/cmd_force"
        self._cmd_force_pub = self._cmd_force_node.advertise(
            cmd_force_topic, Double, AdvertiseMessageOptions())

        # Sensor subscribers
        self._pose_node = Node()
        pose_topic = "/world/" + self.world_name + "/pose/info"
        self._pose_node.subscribe(Pose_V, pose_topic, self._on_pose)

        self._joint_state_node = Node()
        joint_topic = "/world/" + self.world_name + "/model" + self.agent_name + "/joint_state"
        self._joint_state_node.subscribe(ModelMsg, joint_topic, self._on_joint_state)

        # /env/action, /env/state
        self._action_node = Node()
        self._action_node.subscribe(Float_V, "/env/action", self._on_action)

        self._state_node = Node()
        self._state_pub = self._state_node.advertise(
            "/env/state", Float_V, AdvertiseMessageOptions())

        print(f"[InvertedPendulumSyncGate] ready: world={self.world_name!r} "
              f"agent={self.agent_name!r} frame_skip={self.frame_skip} gear={GEAR}")

    def pre_update(self, info, ecm):
        if info.paused:
            return
        if self._pending_action is None:
            return
        msg = Double()
        if self._action_active:
            msg.data = float(self._pending_action) * GEAR
        else:
            msg.data = 0.0
        self._cmd_force_pub.publish(msg)

    def post_update(self, info, ecm):
        if info.paused:
            return
        if self._pending_action is None or not self._action_active:
            return
        self._ticks_since_action += 1
        if self._ticks_since_action >= self.frame_skip:
            self._publish_state()
            self._action_active = False

    def reset(self, info, ecm):
        self._pending_action = None
        self._action_active = False
        self._ticks_since_action = 0
        self.cart_position = 0.0
        self.cart_velocity = 0.0
        self.pole_angle = 0.0
        self.pole_ang_velocity = 0.0
        self._publish_state()

    # ------------------------------------------------------------------ #
    # Topic callbacks
    # ------------------------------------------------------------------ #

    def _on_action(self, msg):
        if len(msg.data) == 0:
            return
        # Continuous action — float scalar.
        self._pending_action = float(msg.data[0])
        self._ticks_since_action = 0
        self._action_active = True

    def _on_pose(self, pose_v_msg):
        # The model name in the world is "inverted_pendulum" — the cart's
        # link is also "cart" and the pole's link is "pole" (preserved from
        # URDF). Scene-broadcaster publishes link names without the model
        # prefix in this configuration.
        for pose in pose_v_msg.pose:
            if pose.name == "cart":
                # slider joint axis (1, 0, 0) — cart slides along X.
                self.cart_position = pose.position.x
            elif pose.name == "pole":
                # hinge joint axis (0, 1, 0) — pole rotates around Y. Recover
                # rotation angle from the quaternion (w, x, y, z).
                q = pose.orientation
                self.pole_angle = 2.0 * math.atan2(q.y, q.w)

    def _on_joint_state(self, model_msg):
        for joint in model_msg.joint:
            if joint.name == "slider":
                self.cart_velocity = joint.axis1.velocity
            elif joint.name == "hinge":
                self.pole_ang_velocity = joint.axis1.velocity

    # ------------------------------------------------------------------ #
    # State publishing
    # ------------------------------------------------------------------ #

    def _publish_state(self):
        # Observation ordering matches Gymnasium InvertedPendulum-v5:
        # [qpos[0], qpos[1], qvel[0], qvel[1]]
        # = [cart_position, pole_angle, cart_velocity, pole_ang_velocity]
        msg = Float_V()
        msg.data.extend([
            float(self.cart_position),
            float(self.pole_angle),
            float(self.cart_velocity),
            float(self.pole_ang_velocity),
        ])
        self._state_pub.publish(msg)


def get_system():
    return InvertedPendulumSyncGate()
