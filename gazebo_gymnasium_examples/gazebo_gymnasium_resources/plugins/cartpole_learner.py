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

"""CartPole sync-gate plugin.

This file is what `PythonSystemLoader` actually loads inside `gz sim`. It is deliberately small: it
does NOT contain a training loop. Its only job is to expose a topic-based "sync gate" so an
external Gymnasium env (running in any other Python process) can drive the simulation step-by-step.

Protocol (gz-transport topics):   /env/action  (Float_V, in)  — agent publishes a single-element
action.   /env/state   (Float_V, out) — plugin publishes 4 sensor values every
`frame_skip` post_updates after an action.   /env/state is also published once when the world
resets (Gazebo Reset event).

Why this pattern: it gives the external env a synchronous-feeling step loop (publish action -> wait
for state callback) while sensor reads happen inside post_update via gz-transport messages from the
model's existing plugins (JointStatePublisher, SceneBroadcaster). World-clock and agent-clock stay
in lockstep because the plugin only publishes /env/state when a step is in flight.

The plugin itself doesn't compute reward / terminated — that stays in the env on the agent side,
where it belongs.
"""

import math

from gz.msgs10.double_pb2 import Double
# Inbound action + outbound state both use Float_V — a variable-length float
# vector. Keeps the plugin generic enough to handle Discrete or Box actions
# (the env packs them into a list of floats either way).
from gz.msgs10.float_v_pb2 import Float_V
from gz.msgs10.model_pb2 import Model as ModelMsg
from gz.msgs10.pose_v_pb2 import Pose_V
from gz.sim8 import World
from gz.sim8 import world_entity
from gz.transport13 import AdvertiseMessageOptions
from gz.transport13 import Node


class CartPoleSyncGate:
    """Proxy actions and observations between a gz sim world and an external Gymnasium env.

    Minimal Gazebo system plugin that runs continuously inside the gz sim
    process and exposes the standard sync-gate protocol over gz-transport.

    Lifecycle:
      configure   — set up node, subscriptions, publishers
      pre_update  — publish the cached action's cmd_vel to JointController
      post_update — read sensor cache, on completed frame_skip publish /env/state
      reset       — Gazebo's Reset event; republish /env/state so env.reset() sees
                    fresh post-reset values.
    """

    def __init__(self):
        # Stored after configure runs:
        self.world_name = None
        self.agent_name = None
        self.frame_skip = 5    # physics ticks per env.step
        self.cart_speed = 1.0  # m/s applied when action == 1; -cart_speed for action == 0

        # Action handling. Mutated by the action subscriber's callback thread,
        # read by the sim main thread in pre_update.
        self._pending_action = None       # latest action (int 0 or 1)
        self._ticks_since_action = 0      # counts post_updates since pending_action arrived
        # When True, pre_update applies the cached action's cmd_vel each tick.
        # When False (between agent decisions), pre_update publishes cmd_vel=0
        # so the cart stops drifting while the agent is busy (e.g. SB3 train_step,
        # which freezes env.step() for ~100ms). Without this the last action
        # accelerates the cart unbounded across the rollout boundary and the
        # pole flips past threshold — that's what caused the "exactly 256
        # steps per episode" pattern.
        self._action_active = False

        # Latest sensor readings, updated by the pose / joint_state callbacks.
        # All assignments are single-float atomic under the GIL.
        self.cart_position = 0.0
        self.cart_velocity = 0.0
        self.pole_angle = 0.0
        self.pole_ang_velocity = 0.0

        # Transport nodes. One Node per logical channel keeps the docs simple
        # and avoids any in-process callback ordering surprises.
        self._action_node = None
        self._state_node = None
        self._cmd_vel_node = None
        self._pose_node = None
        self._joint_state_node = None

        self._state_pub = None
        self._cmd_vel_pub = None

    # ------------------------------------------------------------------ #
    # Gazebo lifecycle hooks
    # ------------------------------------------------------------------ #

    def configure(self, entity, element, ecm, eventManager):
        # Resolve world name from the ECM — needed to build topic paths.
        self.world_name = World(world_entity(ecm)).name(ecm)
        self.agent_name = element.get_string("agent_name") or "cartpole"
        if not self.agent_name.startswith("/"):
            self.agent_name = "/" + self.agent_name

        # frame_skip is optional in SDF; default 5 ticks per env step (50 ms
        # of sim time per agent decision at max_step_size=0.01).
        sdf_frame_skip = int(element.get_double("frame_skip") or 0)
        if sdf_frame_skip > 0:
            self.frame_skip = sdf_frame_skip

        # cmd_vel publisher → drives the existing JointController plugin in
        # the cartpole model. In-process publish, fast.
        self._cmd_vel_node = Node()
        cmd_vel_topic = "/model" + self.agent_name + "/joint/slider_to_cart/cmd_vel"
        self._cmd_vel_pub = self._cmd_vel_node.advertise(
            cmd_vel_topic, Double, AdvertiseMessageOptions())

        # Sensor subscribers — same plumbing as before, but we publish the
        # aggregated state on /env/state rather than letting the env subscribe
        # directly to pose/info. One fewer thing the env user has to know.
        self._pose_node = Node()
        pose_topic = "/world" + self.agent_name + "/pose/info"
        self._pose_node.subscribe(Pose_V, pose_topic, self._on_pose)

        self._joint_state_node = Node()
        joint_topic = "/world" + self.agent_name + "/model" + self.agent_name + "/joint_state"
        self._joint_state_node.subscribe(ModelMsg, joint_topic, self._on_joint_state)

        # /env/action — env -> plugin
        self._action_node = Node()
        self._action_node.subscribe(Float_V, "/env/action", self._on_action)

        # /env/state — plugin -> env
        self._state_node = Node()
        self._state_pub = self._state_node.advertise(
            "/env/state", Float_V, AdvertiseMessageOptions())

        print(f"[CartPoleSyncGate] ready: world={self.world_name!r} "
              f"agent={self.agent_name!r} frame_skip={self.frame_skip}")

    def pre_update(self, info, ecm):
        # Apply the pending action only while it's "active" (i.e. we haven't
        # yet finished its frame_skip window). When inactive, publish cmd_vel=0
        # so the cart coasts to a stop while the agent is busy. This is the
        # grace period that prevents runaway motion between SB3 rollouts.
        if info.paused:
            return
        if self._pending_action is None:
            return
        msg = Double()
        if self._action_active:
            msg.data = -self.cart_speed if self._pending_action == 0 else self.cart_speed
        else:
            msg.data = 0.0
        self._cmd_vel_pub.publish(msg)

    def post_update(self, info, ecm):
        # Tick gating: once frame_skip ticks have passed since the latest
        # action arrived, publish state and mark the action as inactive so
        # the cart stops driving until a new action arrives.
        if info.paused:
            return
        if self._pending_action is None or not self._action_active:
            return
        self._ticks_since_action += 1
        if self._ticks_since_action >= self.frame_skip:
            self._publish_state()
            self._action_active = False  # cart will publish cmd_vel=0 next tick

    def reset(self, info, ecm):
        # Gazebo Reset event — published when world_control.reset is called.
        # Publish fresh state once so env.reset() can return immediately.
        self._pending_action = None
        self._action_active = False
        self._ticks_since_action = 0
        # Zero the sensor cache so any callback latency doesn't leak terminal
        # values into the post-reset state.
        self.cart_position = 0.0
        self.cart_velocity = 0.0
        self.pole_angle = 0.0
        self.pole_ang_velocity = 0.0
        self._publish_state()

    # ------------------------------------------------------------------ #
    # Topic callbacks
    # ------------------------------------------------------------------ #

    def _on_action(self, msg):
        # Latest action wins. Reset the frame_skip counter and re-activate
        # cmd_vel publishing so the cart drives this action for exactly the
        # next frame_skip ticks.
        if len(msg.data) == 0:
            return
        self._pending_action = int(msg.data[0])
        self._ticks_since_action = 0
        self._action_active = True

    def _on_pose(self, pose_v_msg):
        for pose in pose_v_msg.pose:
            if pose.name == "cart":
                # slider_to_cart prismatic axis is (0,1,0) — cart slides on Y.
                self.cart_position = pose.position.y
            elif pose.name == "pole":
                # cart_to_pole revolute axis is (1,0,0) — angle around X from
                # the quaternion (w, x, y, z).
                q = pose.orientation
                self.pole_angle = 2.0 * math.atan2(q.x, q.w)

    def _on_joint_state(self, model_msg):
        for joint in model_msg.joint:
            if joint.name == "slider_to_cart":
                self.cart_velocity = joint.axis1.velocity
            elif joint.name == "cart_to_pole":
                self.pole_ang_velocity = joint.axis1.velocity

    # ------------------------------------------------------------------ #
    # State publishing
    # ------------------------------------------------------------------ #

    def _publish_state(self):
        msg = Float_V()
        msg.data.extend([
            float(self.cart_position),
            float(self.cart_velocity),
            float(self.pole_angle),
            float(self.pole_ang_velocity),
        ])
        self._state_pub.publish(msg)


def get_system():
    """Expose the system so Gazebo's PythonSystemLoader can find it."""
    return CartPoleSyncGate()
