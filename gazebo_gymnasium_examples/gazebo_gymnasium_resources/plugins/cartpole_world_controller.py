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
"""World-level controller for all dynamically-spawned cartpoles.

Loaded once at world load — see `worlds/cartpole_multi.sdf`. Sidesteps
the gz-sim limitation that *model-level Python* system plugins are
unreliable for models created via the EntityFactory service. By living at
world scope, this controller IS loaded by the simulator's plugin
lifecycle and routes actions to every cartpole spawned later.

Wire protocol:
    /env/action_<i>  (Float_V, in)  — discrete action 0 or 1 for
                                       cartpole_<i>.
    /model/cartpole_<i>/joint/slider_to_cart/cmd_vel  (Double, out) —
                                       velocity command consumed by the
                                       per-model JointController system
                                       (which DOES activate on dynamic
                                       spawn). The env reads sensor data
                                       from each model's joint_state topic
                                       directly.

Actuation strategy:
    Publish a Double velocity command to each cartpole's JointController.
    This is the same proven path the single-agent `cartpole_learner.py`
    uses. We deliberately do NOT write the JointVelocityCmd component via
    `Joint.set_velocity` from here: the model's JointController also owns
    that component, and the two writers race (JointController wins and
    zeroes it), so the cart never moves. Driving JointController through
    its command topic avoids the conflict entirely.
"""

import threading

from gz.msgs10.double_pb2 import Double
from gz.msgs10.float_v_pb2 import Float_V
from gz.transport13 import AdvertiseMessageOptions
from gz.transport13 import Node

# Max number of cartpole agents we listen for. Subscribing to a topic
# nobody publishes to is cheap (no traffic), so being generous here
# costs nothing.
MAX_AGENTS = 64
# Linear speed commanded on the cart's slider joint. Matches the
# single-agent cartpole_learner.cart_speed.
CART_SPEED = 1.0


class CartpoleWorldController:
    def __init__(self):
        self.world_name = None
        # Per-agent latest action (None = not yet received; 0 or 1 = a
        # real command is pending).
        self._actions = [None] * MAX_AGENTS
        self._action_nodes = []
        # Per-agent Double cmd_vel publishers into each model's
        # JointController. Created in configure for all slots.
        self._cmd_pubs = [None] * MAX_AGENTS
        self._cmd_node = None
        # Mutex protects _actions across the callback thread and the
        # sim main thread that runs pre_update.
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ #
    # Gazebo lifecycle hooks
    # ------------------------------------------------------------------ #

    def configure(self, entity, element, ecm, eventManager):
        from gz.sim8 import World
        from gz.sim8 import world_entity

        self.world_name = World(world_entity(ecm)).name(ecm)

        self._cmd_node = Node()
        for i in range(MAX_AGENTS):
            node = Node()
            node.subscribe(Float_V, f"/env/action_{i}", self._make_action_callback(i))
            self._action_nodes.append(node)

            cmd_topic = f"/model/cartpole_{i}/joint/slider_to_cart/cmd_vel"
            self._cmd_pubs[i] = self._cmd_node.advertise(
                cmd_topic, Double, AdvertiseMessageOptions()
            )

        print(
            f"[CartpoleWorldController] ready world={self.world_name!r} "
            f"watching /env/action_0..{MAX_AGENTS - 1}, "
            f"commanding /model/cartpole_*/joint/slider_to_cart/cmd_vel"
        )

    def pre_update(self, info, ecm):
        if info.paused:
            return
        # Snapshot actions under the lock so the callback thread can keep
        # writing while we iterate.
        with self._lock:
            actions_snapshot = list(self._actions)

        # Publish every tick so JointController holds the commanded
        # velocity for the whole frame-skip window (it latches the last
        # command, exactly like the single-agent learner relies on).
        for i, action in enumerate(actions_snapshot):
            if action is None:
                continue
            target_vel = CART_SPEED if int(action) == 1 else -CART_SPEED
            msg = Double()
            msg.data = target_vel
            self._cmd_pubs[i].publish(msg)

    def post_update(self, info, ecm):
        # Sensors are read by the env via direct joint_state subscription,
        # so there's nothing to publish here.
        pass

    def reset(self, info, ecm):
        # World reset / entity churn: drop stale actions. The env always
        # sends a fresh action after reset.
        with self._lock:
            self._actions = [None] * MAX_AGENTS
        print("[CartpoleWorldController] reset — cleared all actions")

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _make_action_callback(self, idx):

        def cb(msg):
            if len(msg.data) > 0:
                with self._lock:
                    self._actions[idx] = int(msg.data[0])

        return cb


def get_system():
    """Entry point for PythonSystemLoader."""
    return CartpoleWorldController()
