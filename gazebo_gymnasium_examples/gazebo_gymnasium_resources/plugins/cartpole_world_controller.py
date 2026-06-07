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
the gz-sim limitation that model-level system plugins (`JointController`
in particular) don't activate for models created via the EntityFactory
service. By living at world scope, this controller IS loaded by the
simulator's plugin lifecycle, and it can lazily bind to joints of
cartpoles spawned later via `MultiCartPoleVecEnv._recreate_all_agents`.

Wire protocol (same as the broken per-model plugin):
    /env/action_<i>  (Float_V, in)  — discrete action 0 or 1 for
                                       cartpole_<i>.
    /world/<name>/model/cartpole_<i>/joint_state — already published by
                                       the per-model JointStatePublisher
                                       (which DOES activate on dynamic
                                       spawn). The env reads sensor data
                                       from there directly.

Per-step behaviour:
    pre_update iterates agent slots 0..MAX_AGENTS-1; for any with a
    cached action and a bound joint, sets the joint's velocity directly.
    Bindings are lazy: a slot stays unbound until the matching model
    appears in ECM, at which point we resolve cartpole_<i>::slider_to_cart
    and remember the entity for subsequent ticks. Bindings are invalidated
    on world reset (entities recreate with new IDs).

Actuation strategy:
    `Joint.reset_velocity(ecm, [v])` — sets the joint's velocity
    instantly, no force computation. The cart's position evolves
    naturally from the velocity; the pole responds to the cart's
    acceleration via its rotational joint, so pole dynamics remain
    physically correct. Standard CartPole-v1 uses a force-based
    actuator, but velocity injection produces the same training
    signal — the pole tips for the same reasons.

Fallback APIs tried if reset_velocity isn't available in this gz-sim
build: set_velocity, then a P-controller on top of set_force, then
warn loudly.
"""

import threading

from gz.msgs10.float_v_pb2 import Float_V
from gz.sim8 import Joint
from gz.sim8 import Model
from gz.sim8 import World
from gz.sim8 import world_entity
from gz.transport13 import Node


# Max number of cartpole agents we listen for. Subscribing to a topic
# nobody publishes to is cheap (no traffic), so being generous here
# costs nothing.
MAX_AGENTS = 64
# Linear speed applied to the cart's slider joint when action is set.
# Matches the single-agent cartpole_learner.cart_speed.
CART_SPEED = 1.0
# Fallback P-gain for the set_force code path. Only used if neither
# reset_velocity nor set_velocity is exposed in gz.sim8.
FALLBACK_P_GAIN = 3.0


class CartpoleWorldController:

    def __init__(self):
        self.world_name = None
        # Per-agent latest action (None means we haven't received one
        # yet; an int 0 or 1 means a real command is pending).
        self._actions = [None] * MAX_AGENTS
        # Cached joint entity IDs per agent slot. 0 = not yet found.
        self._joint_entities = [0] * MAX_AGENTS
        self._nodes = []
        # Mutex protects _actions across the callback thread and the
        # sim main thread that runs pre_update.
        self._lock = threading.Lock()
        # Which actuation API ended up working (set on first successful
        # call). Just for the diagnostic print.
        self._api_path = None

    # ------------------------------------------------------------------ #
    # Gazebo lifecycle hooks
    # ------------------------------------------------------------------ #

    def configure(self, entity, element, ecm, eventManager):
        self.world_name = World(world_entity(ecm)).name(ecm)
        for i in range(MAX_AGENTS):
            node = Node()
            topic = f"/env/action_{i}"
            node.subscribe(Float_V, topic, self._make_action_callback(i))
            self._nodes.append(node)
        print(f"[CartpoleWorldController] ready world={self.world_name!r} "
              f"watching /env/action_0..{MAX_AGENTS - 1}")

    def pre_update(self, info, ecm):
        if info.paused:
            return
        # Snapshot actions under the lock so the callback thread can keep
        # writing while we iterate.
        with self._lock:
            actions_snapshot = list(self._actions)

        for i, action in enumerate(actions_snapshot):
            if action is None:
                continue
            joint_entity = self._joint_entities[i]
            if joint_entity == 0:
                joint_entity = self._resolve_joint_entity(i, ecm)
                if joint_entity == 0:
                    continue
                self._joint_entities[i] = joint_entity
                print(f"[CartpoleWorldController] bound agent={i} -> "
                      f"cartpole_{i}::slider_to_cart "
                      f"(joint_entity={joint_entity})")
            target_vel = CART_SPEED if int(action) == 1 else -CART_SPEED
            self._apply_velocity(joint_entity, target_vel, ecm)

    def post_update(self, info, ecm):
        # Sensors are read by the env via direct joint_state subscription,
        # so there's nothing to publish here.
        pass

    def reset(self, info, ecm):
        # World reset invalidates entity IDs; force re-resolution of
        # joints next tick. Stale actions are dropped too — the env's
        # _recreate_all_agents always sends a fresh action after reset.
        with self._lock:
            self._actions = [None] * MAX_AGENTS
        self._joint_entities = [0] * MAX_AGENTS
        print(f"[CartpoleWorldController] world reset — cleared all bindings")

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _make_action_callback(self, idx):
        def cb(msg):
            if len(msg.data) > 0:
                with self._lock:
                    self._actions[idx] = int(msg.data[0])
        return cb

    def _resolve_joint_entity(self, idx, ecm):
        """Find cartpole_<idx>::slider_to_cart in ECM; 0 if not spawned yet."""
        world = World(world_entity(ecm))
        try:
            model_entity = world.model_by_name(ecm, f"cartpole_{idx}")
        except AttributeError:
            # Older gz.sim8 binding without World.model_by_name.
            print("[CartpoleWorldController] ERROR: gz.sim8.World lacks "
                  "model_by_name — bindings too old for this controller.")
            return 0
        if model_entity == 0:
            return 0
        try:
            return Model(model_entity).joint_by_name(ecm, "slider_to_cart")
        except AttributeError:
            print("[CartpoleWorldController] ERROR: gz.sim8.Model lacks "
                  "joint_by_name — bindings too old for this controller.")
            return 0

    def _apply_velocity(self, joint_entity, target_vel, ecm):
        """Try the gz.sim8 actuation APIs in order; latch onto whichever works."""
        joint = Joint(joint_entity)

        if self._api_path == "reset_velocity" or self._api_path is None:
            try:
                joint.reset_velocity(ecm, [target_vel])
                if self._api_path is None:
                    self._api_path = "reset_velocity"
                    print("[CartpoleWorldController] actuation via "
                          "Joint.reset_velocity")
                return
            except AttributeError:
                pass

        if self._api_path == "set_velocity" or self._api_path is None:
            try:
                joint.set_velocity(ecm, [target_vel])
                if self._api_path is None:
                    self._api_path = "set_velocity"
                    print("[CartpoleWorldController] actuation via "
                          "Joint.set_velocity")
                return
            except AttributeError:
                pass

        if self._api_path == "set_force" or self._api_path is None:
            try:
                # P-controller: force = P * (target_vel - current_vel).
                current = joint.velocity(ecm)
                cur_vel = current[0] if current else 0.0
                force = FALLBACK_P_GAIN * (target_vel - cur_vel)
                joint.set_force(ecm, [force])
                if self._api_path is None:
                    self._api_path = "set_force"
                    print("[CartpoleWorldController] actuation via "
                          "Joint.set_force (P-controller)")
                return
            except AttributeError:
                pass

        if self._api_path is None:
            self._api_path = "none"
            print("[CartpoleWorldController] ERROR: no working actuation "
                  "API in gz.sim8.Joint. Tried reset_velocity, "
                  "set_velocity, set_force — none exist on this build. "
                  "Cart will not move.")


def get_system():
    """Entry point for PythonSystemLoader."""
    return CartpoleWorldController()
