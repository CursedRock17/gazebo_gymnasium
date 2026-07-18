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

"""In-process VecEnv — the whole sim runs inside the training process.

The N-agent Gazebo world is hosted by ``gz.sim8.TestFixture`` right here in the
Python process. Each ``step`` advances the server ``frame_skip`` physics ticks
synchronously; actions and observations move through the ECM (``HarnessCore``),
never over a topic. So there is:

* **no gz-transport** — no serialization, no discovery, no per-step IPC,
* **no separate ``gz sim`` process** and no ``ros2 launch``,
* **no display** — it runs headless anywhere the gz Python bindings import.

That makes it the fastest backend (the PufferLib-style "sim in the loop"
approach applied to Gazebo) and the lowest-friction one for users: a single
``python train.py --backend inprocess`` trains end to end. It uses the exact
same ``AgentSpec`` + ``HarnessCore`` ECM actuation/sensing as the deployed
harness; only the transport is elided (and that layer is covered separately by
test_harness_plugin.py).
"""

import copy
import os
import tempfile
import xml.etree.ElementTree as ET

from gymnasium.spaces import Discrete
import numpy as np
from stable_baselines3.common.vec_env import VecEnv

from .agent_spec import AgentSpec
from .agent_spec import get_spec


# real_time_factor=0 removes gz-sim's wall-clock throttle (run as fast as the
# hardware allows) — the single biggest training-throughput lever.
_PHYSICS = ("""<physics name="fast" type="ignored">
      <max_step_size>0.01</max_step_size>
      <real_time_factor>0</real_time_factor>
      <real_time_update_rate>0</real_time_update_rate></physics>
    <plugin filename="gz-sim-physics-system" name="gz::sim::systems::Physics"/>
    <plugin filename="gz-sim-scene-broadcaster-system"
            name="gz::sim::systems::SceneBroadcaster"/>""")

_GROUND = ("""<model name="ground"><static>true</static>
      <link name="l"><collision name="c"><geometry><plane>
        <normal>0 0 1</normal><size>1000 1000</size></plane>
      </geometry></collision></link></model>""")


def _resolve_bare_model_sdf(spec: AgentSpec) -> str:
    """Return the on-disk model.sdf path for the spec's bare (ECM) model."""
    uri = spec.bare_model_uri or spec.model_uri
    if not uri.startswith("package://"):
        raise ValueError(f"expected a package:// uri, got {uri!r}")
    pkg, sub = uri[len("package://"):].split("/", 1)
    from ament_index_python.packages import get_package_share_directory
    return os.path.join(get_package_share_directory(pkg), sub, "model.sdf")


_INERTIA_TAGS = ("mass", "ixx", "iyy", "izz", "ixy", "ixz", "iyz")


def _load_model(model_sdf_path: str):
    """Return the <model> element of a model.sdf."""
    model = ET.parse(model_sdf_path).getroot().find("model")
    if model is None:
        raise ValueError(f"no <model> in {model_sdf_path}")
    return model


def _scaled_inner(model_el, mass_scale: float) -> str:
    """Serialize the model's children, scaling mass + inertia by mass_scale.

    Inertia scales with mass for fixed geometry (I = m r^2), so multiplying
    the mass and every inertia term by the same factor keeps the shape and
    just changes how heavy/dense the body is.
    """
    model = copy.deepcopy(model_el)
    if mass_scale != 1.0:
        for tag in _INERTIA_TAGS:
            for el in model.iter(tag):
                el.text = repr(float(el.text) * mass_scale)
    return "".join(ET.tostring(child, encoding="unicode") for child in model)


# Image worlds additionally need the render/sensors system, a light, and a
# bright ground VISUAL (the physics ground is collision-only): the camera-based
# specs detect the dark line against it.
_SENSORS = ("""<plugin filename="gz-sim-sensors-system"
            name="gz::sim::systems::Sensors">
      <render_engine>ogre2</render_engine></plugin>
    <light name="sun" type="directional">
      <pose>0 0 10 0 0 0</pose>
      <diffuse>0.9 0.9 0.9 1</diffuse><specular>0.1 0.1 0.1 1</specular>
      <direction>-0.3 0.2 -0.9</direction></light>
    <model name="ground_visual"><static>true</static>
      <link name="l"><visual name="v">
        <geometry><plane><normal>0 0 1</normal><size>1000 1000</size></plane>
        </geometry>
        <material><ambient>0.85 0.85 0.85 1</ambient>
          <diffuse>0.85 0.85 0.85 1</diffuse></material>
      </visual></link></model>""")


def _build_world(spec: AgentSpec, n_agents: int, spacing: float, rng):
    """Build an N-agent world SDF by inlining the bare model N times.

    Inline (rather than <include>) sidesteps the frame-graph quirks a merged
    include hits with a world-scope joint, and matches the geometry the ECM
    core is unit-tested against. Each agent's mass is optionally scaled for
    population-based dynamics randomization (``spec.mass_randomization``).
    Image specs additionally get the render/sensors system, a per-agent camera
    topic, and per-agent static scenery. Returns (sdf, spawn_poses).
    """
    model_el = _load_model(_resolve_bare_model_sdf(spec))
    mr = spec.mass_randomization
    offset = (n_agents - 1) * spacing / 2.0
    models, poses = [], []
    for i in range(n_agents):
        scale = float(rng.uniform(1.0 - mr, 1.0 + mr)) if mr > 0 else 1.0
        inner = _scaled_inner(model_el, scale)
        if spec.image_obs is not None:
            inner = inner.replace("<topic>camera</topic>",
                                  f"<topic>/rl/camera_{i}</topic>")
        joints = "".join(
            f'<joint name="{jn}" type="fixed">'
            f"<parent>{parent}</parent><child>{child}</child></joint>"
            for jn, parent, child in spec.extra_joints)
        x = i * spacing - offset
        poses.append((x, spec.spawn_y, spec.spawn_z, spec.spawn_yaw))
        models.append(
            f'<model name="{spec.name}_{i}">'
            f"<pose>{x} {spec.spawn_y} {spec.spawn_z} 0 0 {spec.spawn_yaw}"
            f"</pose>{inner}{joints}</model>")
        if spec.per_agent_include_uri:
            models.append(
                f"<include><uri>{spec.per_agent_include_uri}</uri>"
                f"<name>scenery_{i}</name>"
                f"<pose>{x} 0 0 0 0 0</pose></include>")
    extra = _SENSORS if spec.image_obs is not None else ""
    sdf = (f'<?xml version="1.0" ?><sdf version="1.8">'
           f'<world name="{spec.name}_inproc">{_PHYSICS}{extra}{_GROUND}'
           f'{"".join(models)}</world></sdf>')
    return sdf, poses


class InProcessHarnessVecEnv(VecEnv):
    """N agents in one in-process TestFixture sim (SB3 VecEnv, no transport)."""

    metadata = {"render_modes": [], "render_fps": 30}

    def __init__(self, spec: AgentSpec, n_agents: int = 8,
                 max_episode_steps=None, frame_skip=None,
                 spacing=None, seed=None, autoreset: bool = True,
                 **_ignored):
        if n_agents < 1:
            raise ValueError("n_agents must be >= 1")
        # Imported lazily so the module (and the rest of envs/) stays importable
        # without the native gz bindings present.
        from gazebo_gymnasium_bridge.harness.harness_core import HarnessCore
        import gz.sim8 as gz_sim

        self._spec = spec
        self.n_agents = n_agents
        self.max_episode_steps = (max_episode_steps if max_episode_steps
                                  is not None else spec.max_episode_steps)
        self.frame_skip = frame_skip if frame_skip is not None else spec.frame_skip
        super().__init__(n_agents, spec.observation_space, spec.action_space)
        self.spec = None  # gym EnvSpec slot (kept clear so VecMonitor etc. work)

        self._obs_dim = spec.obs_dim
        discrete = isinstance(spec.action_space, Discrete)
        self._act_dim = 1 if discrete else int(spec.action_space.shape[0])
        self._rng = np.random.default_rng(seed)
        self._debug = os.environ.get(
            "GAZEBO_GYM_VERBOSE", "false").lower() in ("1", "true", "yes", "on")

        # Per-agent same-step autoreset (the SB3 VecEnv convention, applied
        # independently per agent — an agent that terminates resets in place
        # on its own step, others keep going). Each agent tracks its own step
        # count so truncation is per-agent, not a shared clock.
        self._autoreset = autoreset
        self._agent_steps = np.zeros(n_agents, dtype=np.int64)
        self._episode_rewards = np.zeros(n_agents, dtype=np.float32)
        self._current_episode = 0
        self._latest_obs = np.zeros((n_agents, self._obs_dim), dtype=np.float32)
        # "reset" is a per-agent bool mask applied on the next sim tick (all
        # True at construction for the initial reset); None means "actuate".
        self._ctl = {"action": np.zeros((n_agents, self._act_dim)),
                     "reset": np.ones(n_agents, dtype=bool)}
        # Throughput: the sim callbacks fire every physics tick, but the RL loop
        # only needs the observation once per frame_skip and joints resolved
        # once. Reading obs only on the last tick of a run() is the main win.
        self._resolved = False
        self._post_ticks = 0
        self._read_at = 1

        self._core = HarnessCore(spec, n_agents)
        # A dedicated, seed-derived rng so the domain-randomization draws (mass
        # in the SDF, actuator gain on the core) are reproducible and
        # independent of the reset-randomization stream.
        dr_rng = np.random.default_rng(seed)
        if spacing is None:
            spacing = spec.x_spacing
        world, spawn_poses = _build_world(spec, n_agents, spacing, dr_rng)
        self._core.set_spawn_poses(spawn_poses)
        if spec.action_gain_randomization > 0:
            g = spec.action_gain_randomization
            self._core.set_action_gains(
                dr_rng.uniform(1.0 - g, 1.0 + g, size=n_agents))

        # Image observations: frames arrive over in-process gz-transport from
        # each agent's rewritten camera topic; _latest_obs becomes uint8
        # (n, H, W, C) and the joint read in _on_post is skipped.
        self._image_mode = spec.image_obs is not None
        if self._image_mode:
            from gz.msgs10.image_pb2 import Image
            from gz.transport13 import Node
            h, w, c = spec.image_obs
            self._latest_obs = np.zeros((n_agents, h, w, c), dtype=np.uint8)
            self._img_counts = np.zeros(n_agents, dtype=np.int64)
            self._img_node = Node()

            self._img_cb_error = None

            def _make_cb(idx):
                def _cb(msg):
                    # never let an exception escape into the transport thread
                    # (the binding dumps the raw message on callback errors)
                    try:
                        data = bytes(msg.data)
                        if msg.width * msg.height * c != len(data):
                            return
                        frame = np.frombuffer(data, dtype=np.uint8)
                        self._latest_obs[idx] = frame.reshape(
                            msg.height, msg.width, c)
                        self._img_counts[idx] += 1
                    except Exception as exc:   # noqa: B902
                        self._img_cb_error = exc
                return _cb

            for i in range(n_agents):
                self._img_node.subscribe(Image, f"/rl/camera_{i}",
                                         _make_cb(i))
        fd, path = tempfile.mkstemp(suffix=".sdf", prefix="inproc_world_")
        with os.fdopen(fd, "w") as fh:
            fh.write(world)
        self._world_path = path

        self._fx = gz_sim.TestFixture(path)
        self._fx.on_pre_update(self._on_pre)
        self._fx.on_post_update(self._on_post)
        self._fx.finalize()
        self._server = self._fx.server()
        self._step_server(1)  # resolve joints, settle
        if not self._resolved:
            # Without this, a failed world load (bad SDF, invalid inertia,
            # missing joint) would run silently on all-zero observations.
            raise RuntimeError(
                f"in-process world for {spec.name!r} did not resolve all "
                f"agents' joints after the first tick — the world SDF likely "
                f"failed to load (check stderr for gz [Err] lines) or the "
                f"model's joint names don't match the spec")
        if self._image_mode:
            # Warm up the render pipeline and demand a first frame from every
            # camera — a silent all-black obs stream would be worse than an
            # error here.
            prev = self._img_counts.copy()
            self._step_server(10)
            if not self._await_frames(prev, timeout=3.0):
                raise RuntimeError(
                    f"no camera frames received for {spec.name!r} "
                    f"(/rl/camera_0..{n_agents - 1}) — headless rendering "
                    f"unavailable or the model's camera sensor is missing")

        print(f"[InProcessHarnessVecEnv] ready (agent={spec.name!r}, "
              f"n_agents={n_agents}, frame_skip={self.frame_skip}, "
              f"headless in-process)")

    # ---- sim callbacks (run synchronously inside server.run) --------------- #

    def _step_server(self, ticks):
        """Advance the sim `ticks` and read obs only on the final tick."""
        self._post_ticks = 0
        self._read_at = ticks
        self._server.run(True, ticks, False)

    def _on_pre(self, info, ecm):
        if not self._resolved:
            self._core.resolve(ecm)
            self._resolved = all(self._core._resolved)
        mask = self._ctl["reset"]
        if mask is not None:
            for i in np.nonzero(mask)[0]:
                self._core.reset_agent(ecm, int(i), self._rng)
            self._ctl["reset"] = None
        else:
            self._core.apply_actions(ecm, self._ctl["action"])

    def _on_post(self, info, ecm):
        self._post_ticks += 1
        if self._post_ticks >= self._read_at and not self._image_mode:
            self._latest_obs = self._core.read_obs(ecm)

    def _await_frames(self, prev_counts, timeout=0.5):
        """Wait until every agent has a camera frame newer than prev_counts.

        Sensor publishes are asynchronous transport; frames may land shortly
        after server.run() returns. Falls back to the last frame on timeout
        (latest-frame semantics are fine for RL).
        """
        import time as _time
        deadline = _time.perf_counter() + timeout
        while _time.perf_counter() < deadline:
            if bool(np.all(self._img_counts > prev_counts)):
                return True
            _time.sleep(0.002)
        return False

    # ---- VecEnv API -------------------------------------------------------- #

    def _post_reset_settle(self):
        # image mode: run two camera periods and demand counts advance by 2 —
        # a single in-flight frame captured BEFORE the pose restore could
        # otherwise satisfy the wait and show the old (line-less) view.
        if self._image_mode:
            prev = self._img_counts.copy() + 1
            self._step_server(8)
            self._await_frames(prev)

    def reset(self):
        self._ctl["reset"] = np.ones(self.n_agents, dtype=bool)
        self._step_server(1)
        self._post_reset_settle()
        self._agent_steps[:] = 0
        self._episode_rewards[:] = 0.0
        self._current_episode += 1
        return self._latest_obs.copy()

    def _reset_agents(self, mask):
        """Reset the masked agents in place; return the full obs matrix."""
        self._ctl["reset"] = mask
        self._step_server(1)
        self._post_reset_settle()
        return self._latest_obs

    def step_async(self, actions):
        self._ctl["action"] = np.asarray(actions).reshape(self.n_agents, -1)

    def step_wait(self):
        if self._image_mode:
            prev = self._img_counts.copy()
            self._step_server(self.frame_skip)
            self._await_frames(prev)
        else:
            self._step_server(self.frame_skip)
        obs = self._latest_obs.copy()

        self._agent_steps += 1
        actions = self._ctl["action"]
        rewards = np.empty(self.n_agents, dtype=np.float32)
        terminated = np.zeros(self.n_agents, dtype=bool)
        for i in range(self.n_agents):
            # pass the applied action so specs can shape reward on effort
            rewards[i] = float(self._spec.reward_fn(obs[i], actions[i]))
            terminated[i] = bool(self._spec.terminated_fn(obs[i]))
        truncated = self._agent_steps >= self.max_episode_steps
        self._episode_rewards += rewards
        dones = terminated | truncated

        infos = [{} for _ in range(self.n_agents)]
        done_idx = np.nonzero(dones)[0]
        if len(done_idx):
            terminal_obs = obs.copy()
            reset_obs = self._reset_agents(dones.copy()) if self._autoreset \
                else None
            for i in done_idx:
                if truncated[i] and not terminated[i]:
                    infos[i]["TimeLimit.truncated"] = True   # bootstrap on step
                if self._autoreset:
                    infos[i]["terminal_observation"] = terminal_obs[i]
                    obs[i] = reset_obs[i]                     # same-step reset
                    self._agent_steps[i] = 0
                    self._episode_rewards[i] = 0.0
            self._current_episode += len(done_idx)
        return obs, rewards, dones, infos

    def close(self):
        if self._image_mode and getattr(self, "_img_node", None) is not None:
            # Quiesce the camera subscriptions BEFORE teardown: a transport
            # thread caught mid-callback during interpreter exit aborts with
            # "FATAL: exception not rethrown".
            for i in range(self.n_agents):
                try:
                    self._img_node.unsubscribe(f"/rl/camera_{i}")
                except Exception:   # noqa: B902
                    pass
            self._img_node = None
        try:
            if os.path.exists(self._world_path):
                os.remove(self._world_path)
        except OSError:
            pass

    def get_attr(self, attr_name, indices=None):
        idx = range(self.n_agents) if indices is None else indices
        return [getattr(self, attr_name, None) for _ in idx]

    def set_attr(self, attr_name, value, indices=None):
        setattr(self, attr_name, value)

    def env_method(self, method_name, *args, indices=None, **kwargs):
        idx = range(self.n_agents) if indices is None else indices
        method = getattr(self, method_name)
        return [method(*args, **kwargs) for _ in idx]

    def env_is_wrapped(self, wrapper_class, indices=None):
        idx = range(self.n_agents) if indices is None else indices
        return [False for _ in idx]

    def seed(self, seed=None):
        self._rng = np.random.default_rng(seed)
        return [seed for _ in range(self.n_agents)]


def make_inprocess(name: str, n_agents: int = 8, **kwargs):
    """Build an InProcessHarnessVecEnv for a registered agent (no launch)."""
    kwargs.pop("world_name", None)          # not applicable to the in-proc sim
    kwargs.pop("reset_timeout", None)
    kwargs.pop("step_timeout", None)
    return InProcessHarnessVecEnv(get_spec(name), n_agents=n_agents, **kwargs)
