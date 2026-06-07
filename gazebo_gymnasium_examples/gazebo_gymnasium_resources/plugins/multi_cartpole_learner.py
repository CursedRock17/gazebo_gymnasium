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

"""Multi-agent CartPole sync-gate plugin.

One INSTANCE of this plugin is loaded per cartpole model in the world
(see scripts/generate_multi_cartpole_world.py). Each instance manages
exactly one cartpole — its own cart/pole — and talks to the vectorized
env on the agent side using index-suffixed topics:

  /env/action_<i>  (Float_V, in)  — single-element discrete action for
                                    cartpole i.
  /env/state_<i>   (Float_V, out) — 4-element sensor vector for cartpole i.

The vec env on the agent side publishes N actions per step (one per
cartpole) and waits for N state replies before returning. The plugin
itself is otherwise nearly identical to `cartpole_learner.py`; the only
new logic is the `agent_index` SDF parameter that picks the topic suffix.

Group-reset (the one optimization the user spelled out): the plugin's
reset() handler simply zeroes its own cached state. The vec env decides
WHEN to call reset — typically only when ALL agents have terminated, so
a single `/world/<name>/control` reset call propagates to every cartpole
at once instead of N individual resets.
"""

from gz.msgs10.float_v_pb2 import Float_V
from gz.msgs10.model_pb2 import Model as ModelMsg
from gz.msgs10.pose_v_pb2 import Pose_V
from gz.msgs10.twist_pb2 import Twist
from gz.sim8 import World
from gz.sim8 import world_entity
from gz.transport13 import AdvertiseMessageOptions
from gz.transport13 import Node


class MultiCartPoleSyncGate:

    def __init__(self):
        self.world_name = None
        self.agent_name = None
        self.agent_index = 0
        self.frame_skip = 5

        # Action handling — same as single-agent cartpole_learner.
        self._pending_action = None
        self._ticks_since_action = 0
        self._action_active = False

        # Sensor cache.
        self.cart_position = 0.0
        self.cart_velocity = 0.0
        self.pole_angle = 0.0
        self.pole_ang_velocity = 0.0

        self._action_node = None
        self._state_node = None
        self._cmd_vel_node = None
        self._pose_node = None
        self._joint_state_node = None
        self._state_pub = None
        self._cmd_vel_pub = None

    def configure(self, entity, element, ecm, eventManager):
        self.world_name = World(world_entity(ecm)).name(ecm)
        self.agent_name = element.get_string("agent_name") or "cartpole_0"
        if not self.agent_name.startswith("/"):
            self.agent_name = "/" + self.agent_name
        self.agent_index = int(element.get_double("agent_index") or 0)

        sdf_frame_skip = int(element.get_double("frame_skip") or 0)
        if sdf_frame_skip > 0:
            self.frame_skip = sdf_frame_skip

        # cmd_vel publisher — drives JointController in the cartpole model.
        # Topic is scoped to this cartpole via agent_name, so each agent's
        # JointController only receives its own commands.
        self._cmd_vel_node = Node()
        cmd_topic = "/model" + self.agent_name + "/joint/slider_to_cart/cmd_vel"
        self._cmd_vel_pub = self._cmd_vel_node.advertise(
            cmd_topic, Twist, AdvertiseMessageOptions())

        # Sensor subscriptions — same per-agent scoping.
        self._pose_node = Node()
        pose_topic = "/world/" + self.world_name + "/pose/info"
        self._pose_node.subscribe(Pose_V, pose_topic, self._on_pose)

        self._joint_state_node = Node()
        joint_topic = (
            "/world/" + self.world_name + "/model" + self.agent_name
            + "/joint_state"
        )
        self._joint_state_node.subscribe(ModelMsg, joint_topic, self._on_joint_state)

        # Index-suffixed action / state channels so the vec env knows which
        # agent each message belongs to.
        action_topic = f"/env/action_{self.agent_index}"
        state_topic = f"/env/state_{self.agent_index}"
        self._action_node = Node()
        self._action_node.subscribe(Float_V, action_topic, self._on_action)
        self._state_node = Node()
        self._state_pub = self._state_node.advertise(
            state_topic, Float_V, AdvertiseMessageOptions())

        print(f"[MultiCartPoleSyncGate] ready: world={self.world_name!r} "
              f"agent={self.agent_name!r} index={self.agent_index} "
              f"frame_skip={self.frame_skip}")

    def pre_update(self, info, ecm):
        if info.paused:
            return
        if self._pending_action is None:
            return
        twist = Twist()
        if self._action_active:
            # Discrete action 0 -> -1.0 m/s, 1 -> +1.0 m/s.
            twist.linear.x = -1.0 if int(self._pending_action) == 0 else 1.0
        else:
            twist.linear.x = 0.0
        self._cmd_vel_pub.publish(twist)

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
        # Vec env-driven group reset: this fires once per cartpole when the
        # world-control reset propagates. Zero local cache so the env's
        # is_terminated() check on the just-reset cart doesn't see stale
        # data leaking from the previous episode (the bug we hit on the
        # single-agent cartpole).
        self._pending_action = None
        self._action_active = False
        self._ticks_since_action = 0
        self.cart_position = 0.0
        self.cart_velocity = 0.0
        self.pole_angle = 0.0
        self.pole_ang_velocity = 0.0
        self._publish_state()

    def _on_action(self, msg):
        if len(msg.data) == 0:
            return
        self._pending_action = float(msg.data[0])
        self._ticks_since_action = 0
        self._action_active = True

    def _on_pose(self, pose_v_msg):
        # Each cartpole has its own "cart" and "pole" link names scoped
        # under its model. The pose topic publishes ALL models' links;
        # we filter by checking that the link name matches AND its full
        # scope (parent model) ends with our agent_name.
        # gz publishes link poses with `name` set to just the link name
        # (e.g. "cart"). When multiple models share link names, we get
        # one Pose per agent. Track our own by checking the agent_index
        # equivalent: the X position roughly identifies which cart this
        # is, but that's brittle. Better: use header.frame_id metadata
        # if available; fall back to name-only.
        # Since gz pose_info doesn't disambiguate same-named links across
        # models, we use the model-scoped joint_state topic (already
        # subscribed above) for the cart velocity + pole velocity, and
        # only need pose for the absolute cart_position + pole_angle.
        # The joint state alone gives us position + velocity for both
        # joints — sufficient for the 4-dim cartpole observation.
        # Therefore: skip pose entirely for the multi-agent case. The
        # joint_state covers everything.
        pass

    def _on_joint_state(self, model_msg):
        # joint_state is model-scoped — already filtered to this cartpole.
        # Fields:
        #   slider_to_cart.axis1.position = cart X (m)
        #   slider_to_cart.axis1.velocity = cart velocity (m/s)
        #   cart_to_pole.axis1.position = pole angle (rad)
        #   cart_to_pole.axis1.velocity = pole angular velocity (rad/s)
        for joint in model_msg.joint:
            if joint.name == "slider_to_cart":
                self.cart_position = joint.axis1.position
                self.cart_velocity = joint.axis1.velocity
            elif joint.name == "cart_to_pole":
                self.pole_angle = joint.axis1.position
                self.pole_ang_velocity = joint.axis1.velocity

    def _publish_state(self):
        # Observation order: [cart_pos, cart_vel, pole_angle, pole_ang_vel]
        # Matches the single-agent cartpole_learner convention.
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
    return MultiCartPoleSyncGate()
