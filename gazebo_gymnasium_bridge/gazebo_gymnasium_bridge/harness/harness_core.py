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

# gz.math7 MUST be imported before gz.sim8: gz.sim8's Link.world_pose() /
# world_linear_velocity() / world_angular_velocity() return gz.math7 types
# (Pose3d / Vector3d), and pybind11 only registers the C++->Python type
# caster for those when gz.math7 has already been imported into the
# process -- otherwise every call raises "Unable to convert function return
# value to a Python type", even though the call itself succeeds. Not needed
# for Joint.position()/velocity() (those return plain Python lists) or for
# passing a Pose3d as an ARGUMENT (e.g. Model.set_world_pose_cmd, the
# Python->C++ direction), which is why this went unnoticed until base_obs's
# read path (the first thing in this codebase to read a Pose3d/Vector3d back
# out of the ECM) started using it.
from gz.math7 import Pose3d
from gz.math7 import Vector3d
from gz.sim8 import Joint
from gz.sim8 import Link
from gz.sim8 import Model
from gz.sim8 import World
from gz.sim8 import world_entity
import numpy as np


class HarnessCore:
    """ECM apply / read / reset for N copies of one AgentSpec."""

    def __init__(self, spec, n_agents):
        self.spec = spec
        self.n_agents = n_agents
        self._joints = {}  # (agent_i, joint_name) -> Joint
        self._models = {}  # agent_i -> Model
        self._base_links = {}  # agent_i -> Link (spec.base_obs only)
        self._resolved = [False] * n_agents
        self._needed = self._collect_needed_joints()
        # (x, y, z, yaw) per agent — used to restore mobile bases on reset
        # when spec.reset_model_pose is set (joint resets alone can't bring a
        # free chassis home). Filled in by the world builder.
        self._spawn_poses = None
        # Per-agent actuator gain (multiplies velocity/force commands) — the
        # control-authority axis of domain randomization. Default 1.0 (off).
        self._action_gains = [1.0] * n_agents

    def set_spawn_poses(self, poses):
        """Record each agent's spawn pose(s) for model-pose restoring resets.

        Each entry is either a single ``(x, y, z, yaw)`` tuple (the
        original, fixed-per-agent behavior) or a LIST of such tuples --
        candidate poses `reset_agent` draws a fresh one from on every
        reset, not just once at world-build time (added for
        `AgentSpec.track_shape_reset_randomization`; see
        `inprocess_vec_env.py`'s `_build_world`, which is what actually
        builds the candidate list -- this class stays backend-agnostic and
        doesn't know or care where the poses came from).
        """
        self._spawn_poses = list(poses)

    def set_action_gains(self, gains):
        """Set the per-agent actuator gain multiplier (len n_agents)."""
        self._action_gains = [float(g) for g in gains]

    # ------------------------------------------------------------------ #

    def _collect_needed_joints(self):
        names = {j.joint for j in self.spec.joint_obs}
        if self.spec.action_to_commands is not None:
            # sample both discrete extremes / a zero action to discover joints
            for probe in (0, 1):
                try:
                    for jn, _m, _v in self.spec.action_to_commands(probe):
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
            self._models[i] = model
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
            if ok and self.spec.base_obs:
                link_entity = (
                    model.link_by_name(ecm, self.spec.base_link_name)
                    if self.spec.base_link_name
                    else model.canonical_link(ecm)
                )
                if link_entity == 0:
                    ok = False
                else:
                    link = Link(link_entity)
                    link.enable_velocity_checks(ecm, True)
                    self._base_links[i] = link
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
            gain = self._action_gains[i]
            for jn, mode, value in self.spec.action_to_commands(actions[i]):
                joint = self._joints.get((i, jn))
                if joint is None:
                    continue
                if mode == "velocity":
                    joint.set_velocity(ecm, [float(value) * gain])
                elif mode == "force":
                    joint.set_force(ecm, [float(value) * gain])
                elif mode == "position":
                    joint.reset_position(ecm, [float(value)])  # gain n/a

    def read_obs(self, ecm):
        """Build the (n_agents, obs_dim) observation from ECM joint state."""
        obs = np.zeros((self.n_agents, self.spec.obs_dim), dtype=np.float32)
        for i in range(self.n_agents):
            if not self._resolved[i]:
                continue
            if self.spec.base_obs:
                obs[i] = self._read_base_obs(ecm, i)
            else:
                obs[i] = self._read_joint_obs(ecm, i)
        return obs

    def _read_joint_obs(self, ecm, i):
        """Read one agent's obs the plain way (every spec but base_obs).

        One flat pass, position/velocity per entry in `joint_obs` order.
        """
        out = np.zeros(self.spec.obs_dim, dtype=np.float32)
        k = 0
        for jo in self.spec.joint_obs:
            joint = self._joints.get((i, jo.joint))
            if jo.position:
                p = joint.position(ecm) if joint is not None else None
                out[k] = p[0] if p else 0.0
                k += 1
            if jo.velocity:
                v = joint.velocity(ecm) if joint is not None else None
                out[k] = v[0] if v else 0.0
                k += 1
        return out

    def _read_base_obs(self, ecm, i):
        """Read one agent's obs the free-base way (MuJoCo's canonical layout).

        [z, quat, joint positions] then [lin_vel, ang_vel, joint
        velocities]. Requires `joint_obs` to itself be
        positions-only-then-velocities-only (i.e. built via
        `pos_then_vel_obs`), since the two blocks are assembled separately
        and concatenated, not interleaved per entry like `_read_joint_obs`.
        """
        link = self._base_links.get(i)
        pose = link.world_pose(ecm) if link is not None else None
        if pose is not None:
            q = pose.rot()
            pos_vals = [pose.pos().z(), q.w(), q.x(), q.y(), q.z()]
        else:
            pos_vals = [0.0] * 5
        linvel = link.world_linear_velocity(ecm) if link is not None else None
        angvel = link.world_angular_velocity(ecm) if link is not None else None
        vel_vals = [linvel.x(), linvel.y(), linvel.z()] if linvel is not None else [0.0, 0.0, 0.0]
        vel_vals += [angvel.x(), angvel.y(), angvel.z()] if angvel is not None else [0.0, 0.0, 0.0]
        for jo in self.spec.joint_obs:
            joint = self._joints.get((i, jo.joint))
            if jo.position:
                p = joint.position(ecm) if joint is not None else None
                pos_vals.append(p[0] if p else 0.0)
            if jo.velocity:
                v = joint.velocity(ecm) if joint is not None else None
                vel_vals.append(v[0] if v else 0.0)
        return np.array(pos_vals + vel_vals, dtype=np.float32)

    def reset_agent(self, ecm, i, rng):
        """Reset one agent in place (joints, and model pose/base velocity).

        `reset_joint_state` is optional (a base_obs spec may have no
        actuated joints at all, e.g. a body that just free-falls) -- guard
        only the joint-reset block on it, not the model-pose/base-velocity
        resets below, which apply independently.
        """
        if not self._resolved[i]:
            return
        if self.spec.reset_joint_state is not None:
            state = self.spec.reset_joint_state(rng)
            for jn, (pos, vel) in state.items():
                joint = self._joints.get((i, jn))
                if joint is None:
                    continue
                joint.reset_position(ecm, [float(pos)])
                joint.reset_velocity(ecm, [float(vel)])
        if (
            getattr(self.spec, "reset_model_pose", False)
            and self._spawn_poses is not None
            and i in self._models
        ):
            entry = self._spawn_poses[i]
            if entry and isinstance(entry[0], (tuple, list)):
                # A list of candidate poses (track_shape_reset_randomization):
                # draw a fresh one this reset, not the same one every time.
                entry = entry[rng.integers(len(entry))]
            x, y, z, yaw = entry
            self._models[i].set_world_pose_cmd(ecm, Pose3d(x, y, z, 0, 0, yaw))
        if self.spec.base_obs and i in self._base_links:
            # The free base has no "joint" reset_joint_state can zero -- do
            # it here. reset_model_pose (above) already restores its pose;
            # this zeroes the velocity so an episode doesn't inherit
            # momentum from how the last one ended.
            link = self._base_links[i]
            link.set_linear_velocity(ecm, Vector3d(0, 0, 0))
            link.set_angular_velocity(ecm, Vector3d(0, 0, 0))

    def reset(self, ecm, rng):
        """Reset every agent's joints in place (positions + velocities)."""
        for i in range(self.n_agents):
            self.reset_agent(ecm, i, rng)
