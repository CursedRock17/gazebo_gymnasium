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
import uuid
import xml.etree.ElementTree as ET

from gymnasium.spaces import Discrete
import numpy as np
from stable_baselines3.common.vec_env import VecEnv

from . import line_track_shapes
from .agent_spec import AgentSpec
from .agent_spec import get_spec

# real_time_factor=0 removes gz-sim's wall-clock throttle (run as fast as the
# hardware allows) — the single biggest training-throughput lever.
# Whether a camera-backed env has EVER been built in this process. gz-sim's
# rendering scene is a process-wide singleton that is not torn down by close()
# — building a second one, even sequentially, corrupts the scene ("Visual [x]
# already exists") and then segfaults. Latches True permanently; see the guard
# in __init__.
_IMAGE_ENV_CREATED = False

_PHYSICS = """<physics name="fast" type="ignored">
      <max_step_size>0.01</max_step_size>
      <real_time_factor>0</real_time_factor>
      <real_time_update_rate>0</real_time_update_rate></physics>
    <plugin filename="gz-sim-physics-system" name="gz::sim::systems::Physics"/>
    <plugin filename="gz-sim-scene-broadcaster-system"
            name="gz::sim::systems::SceneBroadcaster"/>"""

_GROUND = """<model name="ground"><static>true</static>
      <link name="l"><collision name="c"><geometry><plane>
        <normal>0 0 1</normal><size>1000 1000</size></plane>
      </geometry></collision></link></model>"""


def _resolve_bare_model_sdf(spec: AgentSpec) -> str:
    """Return the on-disk model.sdf path for the spec's bare (ECM) model."""
    uri = spec.bare_model_uri or spec.model_uri
    if not uri.startswith("package://"):
        raise ValueError(f"expected a package:// uri, got {uri!r}")
    pkg, sub = uri[len("package://") :].split("/", 1)
    from ament_index_python.packages import get_package_share_directory

    return os.path.join(get_package_share_directory(pkg), sub, "model.sdf")


_INERTIA_TAGS = ("mass", "ixx", "iyy", "izz", "ixy", "ixz", "iyz")


def _apply_visual_dr(frame, brightness, noise_std, jpeg_quality, rng):
    """One agent's visual-DR draw applied to one rendered RGB frame.

    Order mirrors a real camera pipeline: exposure, then sensor noise, then
    JPEG compression (the actual last stage before transmission) -- these are
    gaps a synthetic renderer doesn't produce on its own. See
    ``AgentSpec.visual_randomization``.
    """
    import cv2

    out = frame.astype(np.float32) * brightness
    if noise_std > 0:
        out += rng.normal(0.0, noise_std, out.shape)
    out = np.clip(out, 0, 255).astype(np.uint8)
    bgr = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
    ok, enc = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_quality)])
    if not ok:
        return out
    bgr = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


# Matches agent_spec._LF_DARK (line_follower's "is this pixel the line"
# threshold) -- track_color_randomization only has a real subject (a dark
# line to lighten) on that spec today, so it's fine to share the constant
# rather than plumb it through AgentSpec for a single current user.
_TRACK_DARK_THRESH = 60


def _apply_track_color_dr(frame, lighten, target):
    """Blend a RAW frame's dark ("line") pixels toward a lighter color.

    Applied BEFORE _apply_visual_dr -- this models a property of the
    physical track (faded/grey paint, not pure black), upstream of the
    camera pipeline effects _apply_visual_dr models. See
    ``AgentSpec.track_color_randomization``.
    """
    if lighten <= 0:
        return frame
    out = frame.astype(np.float32)
    mask = out.max(axis=2) < _TRACK_DARK_THRESH
    if not mask.any():
        return frame
    out[mask] = out[mask] * (1.0 - lighten) + target * lighten
    return np.clip(out, 0, 255).astype(np.uint8)


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
_SENSORS = """<plugin filename="gz-sim-sensors-system"
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
      </visual></link></model>"""


# Grid-cell spacing for track_shape_reset_randomization's "all shapes near
# this agent" layout -- comfortably larger than the biggest shape's
# footprint (racetrack, ~3.3m x 1.5m) so adjacent cells never overlap or
# bleed into each other's camera view.
_SHAPE_GRID_SPACING = 4.5


def _grid_cell_offset(k, n):
    """Position of the k-th of n shapes in a compact, centered square grid."""
    import math

    cols = math.ceil(math.sqrt(n))
    row, col = divmod(k, cols)
    rows = math.ceil(n / cols)
    cx = (col - (cols - 1) / 2) * _SHAPE_GRID_SPACING
    cy = (row - (rows - 1) / 2) * _SHAPE_GRID_SPACING
    return cx, cy


# Per-agent ground-truth odometry, injected into each agent's model only when
# GAZEBO_GYM_ODOM=1. gz-sim's OdometryPublisher derives pose + twist from the
# model's canonical link, so it reports real motion regardless of how the
# wheels are driven (this spec writes joint velocities straight to the ECM and
# runs no DiffDrive plugin). Off by default: it's a diagnostic channel, not
# something training needs, and it costs a serialize+publish per agent per
# tick. training_scripts/verify_forward_motion.py is the consumer.
_ODOM_PLUGIN = (
    '<plugin filename="gz-sim-odometry-publisher-system" '
    'name="gz::sim::systems::OdometryPublisher">'
    "<odom_frame>world</odom_frame>"
    "<robot_base_frame>base_link</robot_base_frame>"
    "<odom_publish_frequency>60</odom_publish_frequency>"
    "<odom_topic>{topic}</odom_topic>"
    "<dimensions>2</dimensions>"
    "</plugin>"
)


def world_name(spec: AgentSpec, topic_prefix: str) -> str:
    """World name for this env, namespaced the same way its topics are.

    ``/rl/<pid>_<uuid8>`` -> ``<spec>_<pid>_<uuid8>``. The default prefix
    ``/rl`` (single env, no namespacing) keeps the original ``<spec>_inproc``.
    """
    tail = topic_prefix.strip("/").split("/")[-1]
    return f"{spec.name}_{tail}" if tail and tail != "rl" else f"{spec.name}_inproc"


def camera_pitch_degrees(world_sdf_path, model_prefix):
    """Per-agent ``camera_link`` pitch, in degrees down, from a built world.

    Read from the world SDF the server is actually handed (``_build_world``'s
    output, which the env writes to a temp file and loads), not from the model
    source tree -- this is the artifact describing the simulation that ran.

    The obvious alternative, subscribing to ``/world/<name>/pose/info``, does
    not work: that message names links unqualified (``camera_link``,
    ``base_link``) with no model scoping, so N agents' links collapse onto each
    other in any name-keyed read.
    """
    root = ET.parse(world_sdf_path).getroot()
    out = {}
    for model in root.iter("model"):
        name = model.get("name", "")
        if not name.startswith(model_prefix):
            continue
        for link in model.findall("link"):
            if link.get("name") != "camera_link":
                continue
            pose = link.find("pose")
            if pose is None or not pose.text:
                continue
            vals = [float(v) for v in pose.text.split()]
            if len(vals) >= 5:
                out[name] = float(np.degrees(vals[4]))  # x y z roll PITCH yaw
    return out


def odom_enabled():
    """True when GAZEBO_GYM_ODOM=1 asks for per-agent odometry topics."""
    return os.environ.get("GAZEBO_GYM_ODOM", "").strip() in ("1", "true", "True")


def _jitter_camera(inner, rng, mount_m, angle_deg):
    """Displace and re-aim one agent's camera in its inlined model text.

    Population DR: called once per agent at world-build time, so the draw is a
    fixed property of that agent for its whole lifetime -- which is what a
    printed mount's tolerance and a lens's unit-to-unit FOV variation actually
    are.

    Only the pose inside ``<link name="camera_link">`` is touched, and only
    its first, so the many other poses in the model are left alone.

    :param inner: the model's inlined SDF text.
    :param rng: this mechanism's own Generator.
    :param mount_m: metres of per-axis translation jitter, U(-x, x).
    :param angle_deg: degrees of pitch and horizontal-FOV jitter, U(-x, x).
    :return: the SDF text with this agent's camera moved.
    """
    start = inner.find('<link name="camera_link">')
    if start < 0:
        return inner
    end = inner.find("</link>", start)
    block = inner[start:end]

    if mount_m > 0 or angle_deg > 0:
        # Match "<pose", not "<pose>": rover_line_bare writes
        # `<pose relative_to="base_link">`, and matching the bare tag finds
        # nothing while everything still runs -- the camera simply never moves.
        # Found exactly that way, by a per-agent pitch that stayed at 45.00.
        tag = block.find("<pose")
        p0 = block.find(">", tag) + 1 if tag >= 0 else 0
        p1 = block.find("</pose>", p0) if p0 else -1
        if p0 and p1 > p0:
            vals = [float(v) for v in block[p0:p1].split()]
            if len(vals) >= 6:
                if mount_m > 0:
                    vals[0] += float(rng.uniform(-mount_m, mount_m))
                    vals[1] += float(rng.uniform(-mount_m, mount_m))
                    vals[2] += float(rng.uniform(-mount_m, mount_m))
                if angle_deg > 0:
                    # index 4 is pitch: x y z roll PITCH yaw
                    vals[4] += float(np.radians(rng.uniform(-angle_deg, angle_deg)))
                block = block[:p0] + " ".join(f"{v:.9g}" for v in vals) + block[p1:]
    if angle_deg > 0:
        f0 = block.find("<horizontal_fov>")
        f1 = block.find("</horizontal_fov>", f0)
        if f0 >= 0 and f1 > f0:
            fov = float(block[f0 + 16 : f1])
            fov += float(np.radians(rng.uniform(-angle_deg, angle_deg)))
            block = block[: f0 + 16] + f"{fov:.9g}" + block[f1:]
    return inner[:start] + block + inner[end:]


def _build_world(
    spec: AgentSpec,
    n_agents: int,
    spacing: float,
    rng,
    topic_prefix="/rl",
    camera_rng=None,
):
    """Build an N-agent world SDF by inlining the bare model N times.

    Inline (rather than <include>) sidesteps the frame-graph quirks a merged
    include hits with a world-scope joint, and matches the geometry the ECM
    core is unit-tested against. Each agent's mass is optionally scaled for
    population-based dynamics randomization (``spec.mass_randomization``).
    Image specs additionally get the render/sensors system, a per-agent camera
    topic, and per-agent static scenery. Returns (sdf, spawn_poses).

    ``topic_prefix`` namespaces the per-agent camera topics AND the world
    name. gz-transport discovery is machine-global, so a bare ``/rl/camera_i``
    makes two concurrent camera envs in DIFFERENT processes publish and
    subscribe to the SAME topic names -- each then reads the other world's
    frames, which is silent (the frames are real images, just of the wrong
    rover) and catastrophic (a known-good policy evaluates at 0% instead of
    ~96%).

    The world name has the same problem for everything gz scopes under
    ``/world/<name>/`` -- ``pose/info``, ``dynamic_pose/info``, ``stats`` --
    and for ``gz sim -g``, which attaches to a world by name. Reproduced
    2026-08-27: a single-agent env subscribed to ``pose/info`` and received
    four agents, from a sweep running in other processes. So the world name
    carries the same per-process suffix.
    """
    model_el = _load_model(_resolve_bare_model_sdf(spec))
    mr = spec.mass_randomization
    offset = (n_agents - 1) * spacing / 2.0
    models, poses = [], []
    for i in range(n_agents):
        scale = float(rng.uniform(1.0 - mr, 1.0 + mr)) if mr > 0 else 1.0
        inner = _scaled_inner(model_el, scale)
        if spec.image_obs is not None:
            inner = inner.replace(
                "<topic>camera</topic>", f"<topic>{topic_prefix}/camera_{i}</topic>"
            )
            if camera_rng is not None and (
                spec.camera_mount_randomization > 0 or spec.camera_angle_randomization > 0
            ):
                inner = _jitter_camera(
                    inner,
                    camera_rng,
                    spec.camera_mount_randomization,
                    spec.camera_angle_randomization,
                )
        joints = "".join(
            f'<joint name="{jn}" type="fixed">'
            f"<parent>{parent}</parent><child>{child}</child></joint>"
            for jn, parent, child in spec.extra_joints
        )
        x = i * spacing - offset
        scenery_x = i * spacing - offset
        if spec.track_shape_choices and spec.track_shape_reset_randomization:
            # Per-RESET track-shape randomization: every reset needs a
            # different pre-existing track to teleport onto (the world is
            # built once, not rebuilt per reset -- see
            # AgentSpec.track_shape_reset_randomization), so ALL choices get
            # spawned near this agent now, each at its own grid cell, and
            # EVERY segment of EVERY shape becomes one candidate spawn pose.
            # HarnessCore.reset_agent draws a fresh one from this list each
            # call.
            pose_choices = []
            for k, shape_name in enumerate(spec.track_shape_choices):
                builder, model_dir_name = line_track_shapes.PRESETS[shape_name]
                cell_x, cell_y = _grid_cell_offset(k, len(spec.track_shape_choices))
                include_uri = f"package://gazebo_gymnasium_resources/models/{model_dir_name}"
                models.append(
                    f"<include><uri>{include_uri}</uri>"
                    f"<name>scenery_{i}_{shape_name}</name>"
                    f"<pose>{scenery_x + cell_x} {cell_y} 0 0 0 0</pose></include>"
                )
                for (local_x, local_y, _z, _roll, _pitch, local_yaw), _size in builder():
                    # +pi/2: see the population-based branch below.
                    # The shape and its grid cell ride along, so whatever
                    # reset_agent draws can be reported back: a spec measuring
                    # progress or deviation has to know WHICH loop it is on.
                    pose_choices.append(
                        (
                            x + cell_x + local_x,
                            cell_y + local_y,
                            spec.spawn_z,
                            local_yaw + np.pi / 2,
                            shape_name,
                            scenery_x + cell_x,
                            cell_y,
                        )
                    )
            poses.append(pose_choices)
            agent_y, agent_yaw = pose_choices[0][1], pose_choices[0][3]
            x = pose_choices[0][0]
        elif spec.track_shape_choices:
            # Population-based track-SHAPE randomization: this agent draws
            # its own shape ONCE (for its whole lifetime), and its spawn
            # pose comes from a random point ON that shape (a segment's own
            # pose is tangent-aligned by construction, so it's always a
            # valid "start here, facing this way" choice with no extra
            # check needed). Overrides per_agent_include_uri and
            # spawn_y/spawn_yaw entirely -- see AgentSpec.track_shape_choices.
            shape_name = rng.choice(spec.track_shape_choices)
            builder, model_dir_name = line_track_shapes.PRESETS[shape_name]
            segments = builder()
            seg_idx = int(rng.integers(0, len(segments)))
            (local_x, local_y, _z, _roll, _pitch, local_yaw), _size = segments[seg_idx]
            # +pi/2: the rover's camera-forward axis is offset 90 degrees
            # from a segment's own tangent yaw (confirmed against the
            # original working spec: the bottom-straight rectangle segment
            # has tangent yaw=0, but its known-good spawn_yaw was pi/2).
            agent_y, agent_yaw = local_y, local_yaw + np.pi / 2
            include_uri = f"package://gazebo_gymnasium_resources/models/{model_dir_name}"
            x += local_x
            # Shape and origin ride along, as in the per-reset branch above.
            poses.append((x, agent_y, spec.spawn_z, agent_yaw, shape_name, scenery_x, 0.0))
            models.append(
                f"<include><uri>{include_uri}</uri>"
                f"<name>scenery_{i}</name>"
                f"<pose>{scenery_x} 0 0 0 0 0</pose></include>"
            )
        else:
            agent_y, agent_yaw = spec.spawn_y, spec.spawn_yaw
            poses.append((x, agent_y, spec.spawn_z, agent_yaw))
            if spec.per_agent_include_uri:
                models.append(
                    f"<include><uri>{spec.per_agent_include_uri}</uri>"
                    f"<name>scenery_{i}</name>"
                    f"<pose>{scenery_x} 0 0 0 0 0</pose></include>"
                )
        # The agent's own chassis, always spawned exactly once regardless of
        # which branch above ran -- x/agent_y/agent_yaw are set by all three.
        odom = _ODOM_PLUGIN.format(topic=f"{topic_prefix}/odom_{i}") if odom_enabled() else ""
        models.append(
            f'<model name="{spec.name}_{i}">'
            f"<pose>{x} {agent_y} {spec.spawn_z} 0 0 {agent_yaw}"
            f"</pose>{inner}{joints}{odom}</model>"
        )
    extra = _SENSORS if spec.image_obs is not None else ""
    sdf = (
        f'<?xml version="1.0" ?><sdf version="1.8">'
        f'<world name="{world_name(spec, topic_prefix)}">{_PHYSICS}{extra}{_GROUND}'
        f"{''.join(models)}</world></sdf>"
    )
    return sdf, poses


class InProcessHarnessVecEnv(VecEnv):
    """N agents in one in-process TestFixture sim (SB3 VecEnv, no transport)."""

    metadata = {"render_modes": [], "render_fps": 30}

    def __init__(
        self,
        spec: AgentSpec,
        n_agents: int = 8,
        max_episode_steps=None,
        frame_skip=None,
        spacing=None,
        seed=None,
        autoreset: bool = True,
        **_ignored,
    ):
        if n_agents < 1:
            raise ValueError("n_agents must be >= 1")
        # Imported lazily so the module (and the rest of envs/) stays importable
        # without the native gz bindings present.
        import gz.sim8 as gz_sim

        from gazebo_gymnasium_bridge.harness.harness_core import HarnessCore

        self._spec = spec
        self.n_agents = n_agents
        self.max_episode_steps = (
            max_episode_steps if max_episode_steps is not None else spec.max_episode_steps
        )
        self.frame_skip = frame_skip if frame_skip is not None else spec.frame_skip
        super().__init__(n_agents, spec.policy_observation_space, spec.action_space)
        self.spec = None  # gym EnvSpec slot (kept clear so VecMonitor etc. work)

        self._obs_dim = spec.obs_dim
        discrete = isinstance(spec.action_space, Discrete)
        self._act_dim = 1 if discrete else int(spec.action_space.shape[0])
        self._rng = np.random.default_rng(seed)
        self._debug = os.environ.get("GAZEBO_GYM_VERBOSE", "false").lower() in (
            "1",
            "true",
            "yes",
            "on",
        )

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
        self._ctl = {
            "action": np.zeros((n_agents, self._act_dim)),
            "reset": np.ones(n_agents, dtype=bool),
        }
        # Throughput: the sim callbacks fire every physics tick, but the RL loop
        # only needs the observation once per frame_skip and joints resolved
        # once. Reading obs only on the last tick of a run() is the main win.
        self._resolved = False
        self._post_ticks = 0
        self._read_at = 1

        self._core = HarnessCore(spec, n_agents)
        # A dedicated, seed-derived rng so the domain-randomization draws
        # (mass, actuator gain, visual) are reproducible and independent of
        # the reset-randomization stream. Each DR mechanism gets its OWN
        # child rng, seeded from a fixed draw off dr_rng -- not sequential
        # draws off dr_rng directly. Sequential draws meant turning one
        # mechanism off shifted what random values the OTHERS got (e.g.
        # visual-only vs. gain-only vs. both, at the same top-level seed,
        # silently drew different visual-DR values depending on whether
        # action_gain_randomization also consumed draws first) -- a real
        # confound for comparing DR configurations against each other.
        dr_rng = np.random.default_rng(seed)
        (
            mass_seed,
            gain_seed,
            visual_seed,
            track_color_seed,
            action_noise_seed,
            battery_seed,
            encoder_seed,
            camera_geom_seed,
        ) = dr_rng.integers(0, 2**63 - 1, size=8)
        if spacing is None:
            spacing = spec.x_spacing
        # Rendering n_agents frames costs roughly linearly in n_agents, and the
        # wait is wall-clock, so a fixed budget silently degrades as either the
        # agent count or the machine's load goes up. 0.5s stays the floor (the
        # long-standing value for small worlds).
        self._frame_wait_timeout = max(0.5, 0.25 * n_agents)
        self.stale_frame_waits = 0
        # Unique per env INSTANCE, not per process: two sequential camera envs
        # in one process would otherwise reuse the prefix, and gz-transport
        # keeps the retired publishers around long enough to matter.
        self._topic_prefix = f"/rl/{os.getpid()}_{uuid.uuid4().hex[:8]}"
        self._world_name = world_name(spec, self._topic_prefix)
        world, spawn_poses = _build_world(
            spec,
            n_agents,
            spacing,
            np.random.default_rng(mass_seed),
            self._topic_prefix,
            camera_rng=np.random.default_rng(camera_geom_seed),
        )
        self._core.set_spawn_poses(spawn_poses)
        # Each agent gets its OWN copy of the scenery, laid out along x. A spec
        # that measures against track geometry needs to know which copy is its
        # own, or it projects onto a track metres away.
        self._track_origins = np.array(
            [[i * spacing - (n_agents - 1) * spacing / 2.0, 0.0] for i in range(n_agents)],
            dtype=np.float32,
        )
        if spec.action_gain_randomization > 0:
            g = spec.action_gain_randomization
            gain_rng = np.random.default_rng(gain_seed)
            joints = spec.actuated_joints
            if joints:
                # One INDEPENDENT draw per actuator, not one shared by the
                # whole robot. A shared gain changes how fast a differential
                # drive goes; only a left/right mismatch changes where it
                # ends up, and veer is what actually breaks a line follower.
                # The rover's firmware carries separate TRIM_LEFT/TRIM_RIGHT
                # constants because the two motors really do differ.
                draws = gain_rng.uniform(1.0 - g, 1.0 + g, size=(n_agents, len(joints)))
                self._core.set_action_gains(
                    [dict(zip(joints, row, strict=True)) for row in draws.tolist()]
                )
            else:
                self._core.set_action_gains(gain_rng.uniform(1.0 - g, 1.0 + g, size=n_agents))

        # Per-step actuator noise: population-based magnitude (each agent's
        # own fixed U(0, x) draw), but the actual noise added to the action
        # is fresh every step -- see step_async. Applies to any spec's
        # action space, not just image_obs ones, so it lives outside the
        # image-mode block below.
        self._action_noise_enabled = spec.action_noise_randomization > 0
        if self._action_noise_enabled:
            an_rng = np.random.default_rng(action_noise_seed)
            self._action_noise_std = an_rng.uniform(
                0.0, spec.action_noise_randomization, size=n_agents
            )
            self._action_noise_rngs = [
                np.random.default_rng(s) for s in an_rng.integers(0, 2**63 - 1, size=n_agents)
            ]

        # Simulated battery discharge: per-agent depletion fraction resampled
        # every EPISODE (not fixed at construction like the DR fields above)
        # -- see _draw_battery_depletion, called from reset()/step_wait().
        self._battery_enabled = spec.battery_discharge_randomization > 0
        if self._battery_enabled:
            self._battery_rng = np.random.default_rng(battery_seed)
            self._battery_depletion = np.zeros(n_agents)
            self._draw_battery_depletion(np.ones(n_agents, dtype=bool))

        # Image observations: frames arrive over in-process gz-transport from
        # each agent's rewritten camera topic; _latest_obs becomes uint8
        # (n, H, W, C) and the joint read in _on_post is skipped.
        self._image_mode = spec.image_obs is not None
        self._aux_fn = spec.aux_obs_fn if spec.aux_obs_dim > 0 else None

        # Action low-pass (AgentSpec.action_lowpass). The retained state is
        # also observed, so the filter does not hide history from the policy.
        # Sensor joints (AgentSpec.sensor_joints): read every post-tick and
        # handed to aux_obs_fn/reward_fn. Zeros until the first tick lands.
        self._sensor_joints = tuple(spec.sensor_joints)
        self._sensors = np.zeros((n_agents, len(self._sensor_joints)), dtype=np.float32)

        # Encoder DR: calibration scale + reading noise + latency + dropout.
        # All three strengths share one RNG stream but independent draws off
        # it, and all are per-agent fixed properties, like the DR above.
        self._sensors_raw = np.zeros_like(self._sensors)
        self._encoder_dr = spec.encoder_noise_randomization if self._sensor_joints else 0.0
        self._encoder_lag_dr = spec.encoder_latency_randomization if self._sensor_joints else 0.0
        self._encoder_drop_dr = spec.encoder_dropout_randomization if self._sensor_joints else 0.0
        self._encoder_any = max(self._encoder_dr, self._encoder_lag_dr, self._encoder_drop_dr) > 0
        if self._encoder_any:
            e_rng = np.random.default_rng(encoder_seed)
            self._encoder_scale = e_rng.uniform(
                1.0 - self._encoder_dr,
                1.0 + self._encoder_dr,
                size=(n_agents, len(self._sensor_joints)),
            )
            # Fixed per-agent lag in control steps, fractional; and a fixed
            # per-agent probability that any one step's reply goes missing.
            self._encoder_lag = e_rng.uniform(0.0, self._encoder_lag_dr, size=n_agents)
            self._encoder_drop_p = e_rng.uniform(0.0, self._encoder_drop_dr, size=n_agents)
            self._encoder_rng = np.random.default_rng(e_rng.integers(0, 2**63 - 1))
            # One slot per step of lag the deepest-lagged agent can ask for,
            # newest first. Sized once so indexing never needs a bounds check.
            self._encoder_depth = int(np.ceil(self._encoder_lag_dr)) + 2
            self._encoder_hist = np.zeros(
                (self._encoder_depth, n_agents, len(self._sensor_joints)), dtype=np.float32
            )
            self._encoder_held = np.zeros_like(self._sensors)

        # Debounced termination: consecutive steps terminated_fn has held.
        self._grace = int(spec.termination_grace_steps)
        self._term_streak = np.zeros(n_agents, dtype=int)
        # Per-step state mapping for specs that ask for one: the previous
        # applied action, and the base link's pose. See
        # AgentSpec.provide_step_state.
        self._step_state = bool(spec.provide_step_state)
        self._reward_prev_action = (
            np.zeros((n_agents, int(np.prod(spec.action_space.shape))), dtype=np.float32)
            if self._step_state
            else None
        )
        self._base_poses = np.zeros((n_agents, 4), dtype=np.float32) if self._step_state else None

        self._lowpass = float(spec.action_lowpass)
        self._prev_action = None
        if self._lowpass > 0.0:
            act_dim = int(np.prod(spec.action_space.shape))
            self._prev_action = np.zeros((n_agents, act_dim), dtype=np.float32)
        if self._image_mode:
            # gz-sim's rendering/Sensors system is effectively a per-PROCESS
            # singleton: constructing a second camera-backed world while one is
            # still alive segfaults inside native code, which would kill the
            # caller outright. Fail clearly instead.
            global _IMAGE_ENV_CREATED
            if _IMAGE_ENV_CREATED:
                raise RuntimeError(
                    "a camera-based environment has already been created in "
                    "this process. gz-sim's rendering scene is a process-wide "
                    "singleton that close() does not tear down, so building a "
                    "second one corrupts the scene and then segfaults. Build "
                    "it in a FRESH PROCESS instead (e.g. multiprocessing, or "
                    "SB3's SubprocVecEnv). Note this limit is per process, not "
                    "per agent: one env can host many agents — use "
                    "n_agents=16 rather than 16 separate envs."
                )
            _IMAGE_ENV_CREATED = True
        if self._image_mode:
            from gz.msgs10.image_pb2 import Image
            from gz.transport13 import Node

            h, w, c = spec.image_obs
            self._latest_obs = np.zeros((n_agents, h, w, c), dtype=np.uint8)
            self._img_counts = np.zeros(n_agents, dtype=np.int64)
            self._img_node = Node()

            self._img_cb_error = None

            # Visual DR: its own independent substream (visual_seed, above),
            # reproducible from the same construction seed but not shifted by
            # whether mass/action-gain DR are also enabled. Each agent's
            # *strength* (0..v) is itself randomized, so the population spans
            # near-clean to heavily perturbed frames rather than applying one
            # fixed augmentation to everyone.
            vdr = spec.visual_randomization > 0
            if vdr:
                visual_rng = np.random.default_rng(visual_seed)
                v = spec.visual_randomization
                strength = visual_rng.uniform(0.0, v, size=n_agents)
                self._vdr_brightness = 1.0 + strength * visual_rng.uniform(
                    -1.0, 1.0, size=n_agents
                )
                self._vdr_noise_std = 25.0 * strength
                self._vdr_jpeg_quality = np.clip(95 - 65.0 * strength, 20, 95)
                self._vdr_rngs = [
                    np.random.default_rng(s)
                    for s in visual_rng.integers(0, 2**63 - 1, size=n_agents)
                ]
            self._vdr_enabled = vdr

            # Track-color DR: its own independent substream (track_color_seed,
            # above). Each agent draws a fixed lightening strength (0..x) and
            # a fixed target color to blend toward -- population-based, same
            # shape as visual DR, but a distinct mechanism/RNG so it can be
            # tuned or disabled without touching the already-tuned visual DR
            # strength.
            tcdr = spec.track_color_randomization > 0
            if tcdr:
                tc_rng = np.random.default_rng(track_color_seed)
                x = spec.track_color_randomization
                self._tcdr_lighten = tc_rng.uniform(0.0, x, size=n_agents)
                self._tcdr_target = tc_rng.uniform(180.0, 255.0, size=n_agents)
            self._tcdr_enabled = tcdr

            def _make_cb(idx):

                def _cb(msg):
                    # never let an exception escape into the transport thread
                    # (the binding dumps the raw message on callback errors).
                    # Store the RAW frame only -- visual DR (cv2) is applied
                    # later on the main thread (see _augment_obs), not here.
                    # cv2's JPEG codec reliably crashed the process (SIGSEGV/
                    # SIGABRT) when called from this thread, even with the
                    # module pre-imported on the main thread: this transport
                    # callback thread isn't Python/GIL-managed the way the
                    # gz-sim on_pre/on_post physics callbacks are.
                    try:
                        data = bytes(msg.data)
                        if msg.width * msg.height * c != len(data):
                            return
                        self._latest_obs[idx] = np.frombuffer(data, dtype=np.uint8).reshape(
                            msg.height, msg.width, c
                        )
                        self._img_counts[idx] += 1
                    except Exception as exc:  # noqa: B902
                        self._img_cb_error = exc

                return _cb

            for i in range(n_agents):
                self._img_node.subscribe(Image, f"{self._topic_prefix}/camera_{i}", _make_cb(i))
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
                f"model's joint names don't match the spec"
            )
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
                    f"unavailable or the model's camera sensor is missing"
                )

        print(
            f"[InProcessHarnessVecEnv] ready (agent={spec.name!r}, "
            f"n_agents={n_agents}, frame_skip={self.frame_skip}, "
            f"headless in-process)"
        )

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
        # Sensors are read on EVERY post-tick, not gated on _read_at: an image
        # spec never reaches that branch, and the last tick of the frame-skip
        # is the reading that pairs with the frame the policy is handed.
        if self._step_state:
            self._base_poses = self._core.read_base_poses(ecm)
        if self._sensor_joints:
            # Raw here; the DR that models the LINK (noise, latency, dropout)
            # is a per-CONTROL-STEP effect, so it is applied once per step in
            # _update_sensors rather than once per physics tick.
            self._sensors_raw = self._core.read_sensor_joints(ecm)
            if not self._encoder_any:
                self._sensors = self._sensors_raw

    def _update_sensors(self, reset_mask=None):
        """Advance the encoder link by one control step.

        Applies, in the order the physical chain does: the per-wheel
        calibration scale and reading noise (what the sensor gets wrong), then
        latency (when the reading describes), then dropout (whether it arrives
        at all). Sets self._sensors, which both the observation and the reward
        read.

        :param reset_mask: agents whose episode just restarted. Their history
            is refilled with the current reading rather than carrying the
            previous episode's motion across the boundary, and their held
            value is cleared.
        """
        if not self._sensor_joints:
            return
        if not self._encoder_any:
            self._sensors = self._sensors_raw
            return

        # What the sensor gets wrong: fixed per-wheel calibration, then noise
        # that is relative, so a stopped quadrature wheel still reads zero.
        reading = self._sensors_raw * self._encoder_scale
        if self._encoder_dr > 0:
            reading = reading * (
                1.0 + self._encoder_rng.normal(0.0, self._encoder_dr, reading.shape)
            )
        reading = reading.astype(np.float32)

        # When it describes: push newest-first, then read back at each agent's
        # own fractional lag, interpolating between the two steps it falls
        # between so a sub-step delay is representable.
        self._encoder_hist[1:] = self._encoder_hist[:-1]
        self._encoder_hist[0] = reading
        if reset_mask is not None and reset_mask.any():
            self._encoder_hist[:, reset_mask] = reading[reset_mask]
        if self._encoder_lag_dr > 0:
            lo = np.floor(self._encoder_lag).astype(int)
            frac = (self._encoder_lag - lo)[:, None]
            idx = np.arange(self.n_agents)
            delayed = (1.0 - frac) * self._encoder_hist[lo, idx] + frac * self._encoder_hist[
                lo + 1, idx
            ]
            reading = delayed.astype(np.float32)

        # Whether it arrives: a dropped reply holds the last good value, which
        # is what the deployment client does. Zeroing instead would claim the
        # rover had stopped, a different event entirely.
        if self._encoder_drop_dr > 0:
            dropped = self._encoder_rng.random(self.n_agents) < self._encoder_drop_p
            if reset_mask is not None:
                dropped &= ~reset_mask  # a fresh episode starts with a real read
            reading = np.where(dropped[:, None], self._encoder_held, reading).astype(np.float32)
        self._encoder_held = reading
        self._sensors = reading

    def _pack_obs(self, frames):
        """Policy-visible observation: the frames, plus aux features if any.

        Aux features are computed from the ALREADY-AUGMENTED frames, i.e. the
        image the policy actually sees, because on hardware they are extracted
        from a real compressed frame rather than from ground truth.
        """
        if self._aux_fn is None:
            return frames
        aux = np.empty((self.n_agents, self._spec.policy_aux_dim), dtype=np.float32)
        for i in range(self.n_agents):
            # Two-argument form only for specs that declare sensor_joints, so
            # every existing single-argument aux_obs_fn keeps working.
            aux[i, : self._spec.aux_obs_dim] = (
                self._aux_fn(frames[i], self._sensors[i])
                if self._sensor_joints
                else self._aux_fn(frames[i])
            )
        # Append the retained filter state: with a low-pass in the loop the
        # response depends on a_{t-1}, so the policy has to see it.
        if self._prev_action is not None:
            aux[:, self._spec.aux_obs_dim :] = self._prev_action
        # policy_image=False: the features ARE the observation. Returning the
        # bare array (not a Dict) keeps _obs_copy/_obs_take/_obs_assign on
        # their ndarray paths and lets wrap_for_observations pick MlpPolicy.
        return {"image": frames, "track": aux} if self._spec.policy_image else aux

    @staticmethod
    def _obs_copy(obs):
        return {k: v.copy() for k, v in obs.items()} if isinstance(obs, dict) else obs.copy()

    @staticmethod
    def _obs_take(obs, i):
        return {k: v[i] for k, v in obs.items()} if isinstance(obs, dict) else obs[i]

    @staticmethod
    def _obs_assign(dst, i, src):
        if isinstance(dst, dict):
            for k in dst:
                dst[k][i] = src[k][i]
        else:
            dst[i] = src[i]

    def _augment_obs(self, obs):
        """Apply per-agent track-color + visual DR to an already-copied obs.

        Main thread only: cv2's JPEG codec isn't safe to call from the
        gz-transport camera-callback thread (see the comment in _make_cb),
        so this runs here instead -- called from reset()/step_wait(), which
        the training loop always drives from the main thread. `obs` must
        already be a copy of self._latest_obs; mutated and returned in place.
        Track-color DR runs first (a property of the physical world), then
        visual DR (the camera pipeline) -- see each's own docstring.
        """
        if not self._image_mode:
            return obs
        if self._tcdr_enabled:
            for i in range(self.n_agents):
                obs[i] = _apply_track_color_dr(obs[i], self._tcdr_lighten[i], self._tcdr_target[i])
        if self._vdr_enabled:
            for i in range(self.n_agents):
                obs[i] = _apply_visual_dr(
                    obs[i],
                    self._vdr_brightness[i],
                    self._vdr_noise_std[i],
                    self._vdr_jpeg_quality[i],
                    self._vdr_rngs[i],
                )
        return obs

    def _draw_battery_depletion(self, mask):
        """Redraw this episode's max-depletion fraction for masked agents."""
        idx = np.nonzero(mask)[0]
        if len(idx):
            self._battery_depletion[idx] = self._battery_rng.uniform(
                0.0, self._spec.battery_discharge_randomization, size=len(idx)
            )

    def _await_frames(self, prev_counts, timeout=None):
        """Wait until every agent has a camera frame newer than prev_counts.

        Sensor publishes are asynchronous transport; frames may land shortly
        after server.run() returns. Falls back to the last frame on timeout,
        which hands the policy a STALE observation -- fine as an occasional
        latest-frame compromise, silent data corruption if it happens every
        step (the action then pairs with a frame from before it was taken).

        The timeout is wall-clock, so it is really a bet on how long this
        machine takes to render n_agents frames. That bet loses under load:
        several training processes sharing a machine is exactly the case
        where rendering slows down AND where nobody is watching a console.
        So it scales with agent count, and every fallback is counted and
        surfaced (``stale_frame_waits``) rather than passing silently.
        """
        import time as _time

        if timeout is None:
            timeout = self._frame_wait_timeout
        deadline = _time.perf_counter() + timeout
        while _time.perf_counter() < deadline:
            if bool(np.all(self._img_counts > prev_counts)):
                return True
            _time.sleep(0.002)
        self.stale_frame_waits += 1
        if self.stale_frame_waits in (1, 10, 100, 1000):
            print(
                f"[InProcessHarnessVecEnv] WARNING: camera frames did not arrive "
                f"within {timeout:.1f}s ({self.stale_frame_waits} so far) -- the "
                f"policy is seeing a stale frame. Usually means this machine is "
                f"oversubscribed; reduce concurrent envs or agents per env.",
                flush=True,
            )
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
        self._redraw_invalid_spawns(np.ones(self.n_agents, dtype=bool))
        self._agent_steps[:] = 0
        self._episode_rewards[:] = 0.0
        self._current_episode += 1
        if self._battery_enabled:
            self._draw_battery_depletion(np.ones(self.n_agents, dtype=bool))
        if self._prev_action is not None:
            # The zero action, not "stopped": a forward-only map has no
            # stopping command, so this is the neutral request the rover is
            # already being given while the reset settles.
            self._prev_action[:] = 0.0
        self._term_streak[:] = 0
        if self._reward_prev_action is not None:
            self._reward_prev_action[:] = 0.0
        self._update_sensors(np.ones(self.n_agents, dtype=bool))
        return self._pack_obs(self._augment_obs(self._latest_obs.copy()))

    def _redraw_invalid_spawns(self, mask):
        """Re-reset agents whose fresh spawn is already terminal.

        With track_shape_reset_randomization a drawn pose can land the agent
        where its camera sees no line at all. That episode is unwinnable from
        step 1 rather than merely hard, so it measures nothing about the
        policy. Redrawing costs one extra reset for the ~1% of draws that
        need it. Validity is spec.terminated_fn, so this is not
        line_follower-specific.
        """
        attempts = getattr(self._spec, "spawn_redraw_attempts", 0)
        if not attempts or self._spec.terminated_fn is None:
            return
        for _ in range(attempts):
            bad = np.zeros(self.n_agents, dtype=bool)
            for i in np.nonzero(mask)[0]:
                if self._spec.terminated_fn(self._latest_obs[i]):
                    bad[i] = True
            if not bad.any():
                return
            self._ctl["reset"] = bad
            self._step_server(1)
            self._post_reset_settle()
            mask = bad

    def _reset_agents(self, mask):
        """Reset the masked agents in place.

        Returns the full obs matrix, policy-visible (DR-augmented).
        """
        self._ctl["reset"] = mask
        self._step_server(1)
        self._post_reset_settle()
        self._redraw_invalid_spawns(mask)
        # Clear the ramp for the agents that restarted, and only those: the
        # rest are mid-episode and their filter state is still live.
        if self._prev_action is not None:
            self._prev_action[mask] = 0.0
        self._term_streak[mask] = 0
        if self._reward_prev_action is not None:
            self._reward_prev_action[mask] = 0.0
        self._update_sensors(mask)
        return self._pack_obs(self._augment_obs(self._latest_obs.copy()))

    def step_async(self, actions):
        actions = np.asarray(actions).reshape(self.n_agents, -1)
        if self._action_noise_enabled or self._battery_enabled:
            actions = actions.astype(np.float32, copy=True)
            if self._action_noise_enabled:
                for i in range(self.n_agents):
                    actions[i] += self._action_noise_rngs[i].normal(
                        0.0, self._action_noise_std[i], actions.shape[1]
                    )
            if self._battery_enabled:
                t = np.clip(self._agent_steps / self.max_episode_steps, 0.0, 1.0)
                actions *= (1.0 - self._battery_depletion * t)[:, None]
            np.clip(actions, -1.0, 1.0, out=actions)
        if self._lowpass > 0.0:
            # Ramp toward the request instead of stepping to it, mirroring the
            # firmware's per-wheel PID loop. Applied after noise/battery so it
            # smooths the command the hardware would truly receive.
            actions = (
                self._lowpass * self._prev_action
                + (1.0 - self._lowpass) * np.asarray(actions, dtype=np.float32)
            ).astype(np.float32)
            self._prev_action[:] = actions
        self._ctl["action"] = actions

    def step_wait(self):
        if self._image_mode:
            prev = self._img_counts.copy()
            self._step_server(self.frame_skip)
            self._await_frames(prev)
        else:
            self._step_server(self.frame_skip)
        # reward_fn/terminated_fn always see the RAW (ground-truth) frame --
        # only the policy-visible obs is DR-augmented, so visual DR perturbs
        # what the network learns to be robust to without also injecting
        # noise into the RL objective itself (spurious line-loss/reward from
        # augmented pixels rather than the actual simulated state).
        raw_obs = self._latest_obs.copy()
        self._update_sensors()
        obs = self._pack_obs(self._augment_obs(raw_obs.copy()))

        self._agent_steps += 1
        actions = self._ctl["action"]
        rewards = np.empty(self.n_agents, dtype=np.float32)
        terminated = np.zeros(self.n_agents, dtype=bool)
        for i in range(self.n_agents):
            # pass the applied action so specs can shape reward on effort
            if self._step_state:
                state = {
                    "agent": i,
                    # Steps since this agent's own reset, so a spec can cache
                    # per-step work that both reward_fn and terminated_fn need
                    # without doing it (or double-counting it) twice.
                    "step": int(self._agent_steps[i]),
                    "prev_action": self._reward_prev_action[i],
                    "pose": self._base_poses[i],
                    # Origin of THIS agent's copy of the scenery, and which
                    # shape it is currently on when the scenery varies.
                    "origin": self._track_origins[i],
                    "track": self._core.current_tracks.get(i),
                }
                rewards[i] = float(
                    self._spec.reward_fn(raw_obs[i], actions[i], self._sensors[i], state)
                )
                terminated[i] = bool(self._spec.terminated_fn(raw_obs[i], state))
            elif self._sensor_joints:
                rewards[i] = float(self._spec.reward_fn(raw_obs[i], actions[i], self._sensors[i]))
                terminated[i] = bool(self._spec.terminated_fn(raw_obs[i]))
            else:
                rewards[i] = float(self._spec.reward_fn(raw_obs[i], actions[i]))
                terminated[i] = bool(self._spec.terminated_fn(raw_obs[i]))
        if self._reward_prev_action is not None:
            self._reward_prev_action[:] = actions
        if self._grace:
            # The failure has to persist. Any non-failing step clears the
            # streak, so only a SUSTAINED condition ends the episode.
            self._term_streak = np.where(terminated, self._term_streak + 1, 0)
            terminated = self._term_streak >= self._grace
        truncated = self._agent_steps >= self.max_episode_steps
        self._episode_rewards += rewards
        dones = terminated | truncated

        infos = [{} for _ in range(self.n_agents)]
        done_idx = np.nonzero(dones)[0]
        if len(done_idx):
            terminal_obs = self._obs_copy(obs)
            if self._autoreset and self._battery_enabled:
                self._draw_battery_depletion(dones)
            reset_obs = self._reset_agents(dones.copy()) if self._autoreset else None
            for i in done_idx:
                if truncated[i] and not terminated[i]:
                    infos[i]["TimeLimit.truncated"] = True  # bootstrap on step
                if self._autoreset:
                    infos[i]["terminal_observation"] = self._obs_take(terminal_obs, i)
                    self._obs_assign(obs, i, reset_obs)  # same-step reset
                    self._agent_steps[i] = 0
                    self._episode_rewards[i] = 0.0
            self._current_episode += len(done_idx)
        return obs, rewards, dones, infos

    def close(self):
        # NOTE: _IMAGE_ENV_CREATED is deliberately NOT cleared here — the
        # renderer stays initialized for the life of the process.
        if self._image_mode and getattr(self, "_img_node", None) is not None:
            # Quiesce the camera subscriptions BEFORE teardown: a transport
            # thread caught mid-callback during interpreter exit aborts with
            # "FATAL: exception not rethrown".
            for i in range(self.n_agents):
                try:
                    self._img_node.unsubscribe(f"{self._topic_prefix}/camera_{i}")
                except Exception:  # noqa: B902
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
    kwargs.pop("world_name", None)  # not applicable to the in-proc sim
    kwargs.pop("reset_timeout", None)
    kwargs.pop("step_timeout", None)
    return InProcessHarnessVecEnv(get_spec(name), n_agents=n_agents, **kwargs)
