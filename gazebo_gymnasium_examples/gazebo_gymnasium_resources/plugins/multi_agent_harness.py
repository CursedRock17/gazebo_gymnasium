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

"""Batched in-sim harness — one world plugin drives all N agents.

Loaded once at world scope (PythonSystemLoader). It owns actuation and sensing
for every agent through the ECM (via ``HarnessCore``) and exposes exactly THREE
gz-transport topics regardless of N:

  /rl/actions       (Float_V, in)  — flattened N*act_dim action vector
  /rl/reset         (Float_V, in)  — reset command (optional [seed] payload)
  /rl/observations  (Float_V, out) — flattened N*obs_dim observation, per tick

So the client is O(1) in agent count (one publish + one subscribe), instead of
2N topics — and reset is in place (ECM reset_position), not a respawn.

Under Pixi the whole stack is one Python, so this plugin imports the same
AgentSpec the client uses (no duplicated per-agent wiring).
"""

import threading

from gz.msgs10.float_v_pb2 import Float_V
from gz.transport13 import AdvertiseMessageOptions
from gz.transport13 import Node
import numpy as np

from gazebo_gymnasium_bridge.envs.agent_spec import get_spec
from gazebo_gymnasium_bridge.harness.harness_core import HarnessCore


ACTION_TOPIC = "/rl/actions"
RESET_TOPIC = "/rl/reset"
OBS_TOPIC = "/rl/observations"


class MultiAgentHarness:

    def __init__(self):
        self.core = None
        self.n_agents = 0
        self._act_dim = 1
        self._actions = None
        self._lock = threading.Lock()
        self._pending_reset = False
        self._reset_seed = None
        self._rng = np.random.default_rng()
        self._node = None
        self._obs_pub = None
        # True only once ALL agents' joints are bound. Until then we publish
        # nothing — otherwise unresolved agents read as all-zeros, which look
        # like a perfectly balanced pole and silently fake a solved episode.
        self._ready = False

    # ------------------------------------------------------------------ #
    # Setup (callable directly for tests; configure() wires it from SDF)
    # ------------------------------------------------------------------ #

    def setup(self, agent_name: str, n_agents: int, seed=None):
        from gymnasium.spaces import Discrete
        spec = get_spec(agent_name)
        self.n_agents = int(n_agents)
        self.core = HarnessCore(spec, self.n_agents)
        self._act_dim = (1 if isinstance(spec.action_space, Discrete)
                         else int(spec.action_space.shape[0]))
        self._actions = np.zeros((self.n_agents, self._act_dim),
                                 dtype=np.float32)
        if seed is not None:
            self._rng = np.random.default_rng(int(seed))

        self._node = Node()
        self._node.subscribe(Float_V, ACTION_TOPIC, self._on_actions)
        self._node.subscribe(Float_V, RESET_TOPIC, self._on_reset)
        self._obs_pub = self._node.advertise(
            OBS_TOPIC, Float_V, AdvertiseMessageOptions())
        print(f"[MultiAgentHarness] ready: agent={spec.name!r} "
              f"n_agents={self.n_agents} act_dim={self._act_dim} "
              f"topics={ACTION_TOPIC}/{RESET_TOPIC}->{OBS_TOPIC}")

    def configure(self, entity, element, ecm, event_mgr):
        # Config comes from env vars (the launch exports GAZEBO_GYM_AGENT /
        # _N_AGENTS), so one fixed world SDF works for any N. The sdformat
        # `element` is readable too (get_string returns '' for missing keys),
        # but env vars are the single source of truth the launch already sets —
        # keeping configure trivial keeps this rarely-exercised surface small.
        import os
        agent = os.environ.get("GAZEBO_GYM_AGENT", "cartpole")
        n = int(os.environ.get("GAZEBO_GYM_N_AGENTS", "1"))
        self.setup(agent, n)

    # ------------------------------------------------------------------ #
    # Transport callbacks (async thread)
    # ------------------------------------------------------------------ #

    def _on_actions(self, msg):
        data = np.asarray(msg.data, dtype=np.float32)
        if data.size != self.n_agents * self._act_dim:
            return
        with self._lock:
            self._actions = data.reshape(self.n_agents, self._act_dim)

    def _on_reset(self, msg):
        with self._lock:
            self._pending_reset = True
            if len(msg.data) > 0:
                self._reset_seed = int(msg.data[0])

    # ------------------------------------------------------------------ #
    # Sim lifecycle
    # ------------------------------------------------------------------ #

    def pre_update(self, info, ecm):
        if info.paused or self.core is None:
            return
        self._ready = self.core.resolve(ecm)
        with self._lock:
            if self._pending_reset:
                if self._reset_seed is not None:
                    self._rng = np.random.default_rng(self._reset_seed)
                    self._reset_seed = None
                self.core.reset(ecm, self._rng)
                self._pending_reset = False
                return  # no actuation on the reset tick
            actions = self._actions.copy()
        self.core.apply_actions(ecm, actions)

    def post_update(self, info, ecm):
        if info.paused or self.core is None or self._obs_pub is None:
            return
        if not self._ready:
            return  # not all agents bound yet — don't publish zeros
        obs = self.core.read_obs(ecm)          # (n_agents, obs_dim)
        msg = Float_V()
        msg.data.extend(obs.reshape(-1).tolist())
        self._obs_pub.publish(msg)

    def reset(self, info, ecm):
        # World-control reset: also drop stale actions.
        with self._lock:
            self._actions[:] = 0.0
            self._pending_reset = False


def get_system():
    return MultiAgentHarness()
