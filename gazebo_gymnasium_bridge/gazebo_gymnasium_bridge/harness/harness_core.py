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

"""In-sim ECM core for the batched multi-agent harness.

``HarnessCore`` does the actual physics I/O for N agents through the Entity
Component Manager (ECM): it resolves each agent's joints, applies actions
(``Joint.set_velocity`` / ``set_force`` / ``reset_position``), reads the
observation vector (``Joint.position`` / ``velocity``), and resets joints in
place (``reset_position`` / ``reset_velocity`` — no respawn).

It's deliberately decoupled from gz-transport and the plugin lifecycle so it
can be unit-tested with ``gz.sim8.TestFixture`` (in-process server, no
simulator launch). The plugin (``multi_agent_harness``) wraps this with the
batched action/obs topics.

Actuation is via ECM, so the actuated models must NOT carry a competing
JointController (it would clobber the JointVelocityCmd). Note: actuated joints
must NOT declare an ``<effort>`` limit either — in this DART build an effort
limit silently disables the velocity command (the cart won't move). Position
bounds are enforced by the env's termination logic, not the joint limit.
"""

from gz.sim8 import Joint
from gz.sim8 import Model
from gz.sim8 import World
from gz.sim8 import world_entity
import numpy as np


class HarnessCore:
    """ECM apply / read / reset for N copies of one AgentSpec."""

    def __init__(self, spec, n_agents):
        self.spec = spec
        self.n_agents = n_agents
        self._joints = {}                 # (agent_i, joint_name) -> Joint
        self._resolved = [False] * n_agents
        self._needed = self._collect_needed_joints()

    # ------------------------------------------------------------------ #

    def _collect_needed_joints(self):
        names = {j.joint for j in self.spec.joint_obs}
        if self.spec.action_to_commands is not None:
            # sample both discrete extremes / a zero action to discover joints
            for probe in (0, 1):
                try:
                    for (jn, _m, _v) in self.spec.action_to_commands(probe):
                        names.add(jn)
                except Exception:
                    pass
        return names

    def resolve(self, ecm):
        """Bind each agent's joints by name; True once ALL agents are bound.

        Lazy + idempotent: models may not exist yet (dynamic spawn), so call
        this each tick until it returns True. Enables position/velocity
        component reads on first bind.
        """
        world = World(world_entity(ecm))
        all_ok = True
        for i in range(self.n_agents):
            if self._resolved[i]:
                continue
            model_entity = world.model_by_name(ecm, f"{self.spec.name}_{i}")
            if model_entity == 0:
                all_ok = False
                continue
            model = Model(model_entity)
            ok = True
            for jn in self._needed:
                je = model.joint_by_name(ecm, jn)
                if je == 0:
                    ok = False
                    break
                joint = Joint(je)
                joint.enable_velocity_check(ecm, True)
                joint.enable_position_check(ecm, True)
                self._joints[(i, jn)] = joint
            self._resolved[i] = ok
            all_ok = all_ok and ok
        return all_ok

    def apply_actions(self, ecm, actions):
        """Apply one action per agent via ECM (velocity/force/position)."""
        if self.spec.action_to_commands is None:
            return
        for i in range(self.n_agents):
            if not self._resolved[i]:
                continue
            for (jn, mode, value) in self.spec.action_to_commands(actions[i]):
                joint = self._joints.get((i, jn))
                if joint is None:
                    continue
                if mode == "velocity":
                    joint.set_velocity(ecm, [float(value)])
                elif mode == "force":
                    joint.set_force(ecm, [float(value)])
                elif mode == "position":
                    joint.reset_position(ecm, [float(value)])

    def read_obs(self, ecm):
        """Build the (n_agents, obs_dim) observation from ECM joint state."""
        obs = np.zeros((self.n_agents, self.spec.obs_dim), dtype=np.float32)
        for i in range(self.n_agents):
            if not self._resolved[i]:
                continue
            k = 0
            for jo in self.spec.joint_obs:
                joint = self._joints.get((i, jo.joint))
                if jo.position:
                    p = joint.position(ecm) if joint is not None else None
                    obs[i, k] = p[0] if p else 0.0
                    k += 1
                if jo.velocity:
                    v = joint.velocity(ecm) if joint is not None else None
                    obs[i, k] = v[0] if v else 0.0
                    k += 1
        return obs

    def reset(self, ecm, rng):
        """Reset every agent's joints in place (positions + velocities)."""
        if self.spec.reset_joint_state is None:
            return
        for i in range(self.n_agents):
            if not self._resolved[i]:
                continue
            state = self.spec.reset_joint_state(rng)
            for jn, (pos, vel) in state.items():
                joint = self._joints.get((i, jn))
                if joint is None:
                    continue
                joint.reset_position(ecm, [float(pos)])
                joint.reset_velocity(ecm, [float(vel)])
