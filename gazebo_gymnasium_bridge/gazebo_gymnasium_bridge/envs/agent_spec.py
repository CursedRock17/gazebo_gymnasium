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
"""Per-agent specification for the generalized multi-agent Gazebo VecEnv.

One ``AgentSpec`` fully describes a *single* agent: which model to spawn,
how to turn its sensor topics into an observation vector, the action /
observation spaces, and the reward / termination rules. The
``MultiAgentGazeboVecEnv`` then runs N identical agents from one spec in a
single Gazebo world — so "MultiCartPole", "MultiAnt", etc. are just named
specs, not separate env classes.

This module is deliberately free of any ``gz.*`` import: it operates on
duck-typed messages (anything exposing ``msg.joint[k].name`` /
``.axis1.position`` / ``.axis1.velocity``). That keeps the spec layer unit
testable without a running simulator or the native bindings.
"""

from dataclasses import dataclass
from dataclasses import field
from dataclasses import replace
from typing import Callable
from typing import Optional
from typing import Sequence

from gymnasium import spaces
import numpy as np

# z(1) + quaternion(4) + lin_vel(3) + ang_vel(3), see AgentSpec.base_obs.
BASE_OBS_WIDTH = 11


@dataclass(frozen=True)
class JointObs:
    """Pull one joint's position and/or velocity into the observation vector.

    The order of fields in the final observation follows the order of the
    ``JointObs`` entries in ``AgentSpec.joint_obs``; within one entry,
    position precedes velocity.
    """

    joint: str
    position: bool = True
    velocity: bool = True

    @property
    def width(self) -> int:
        return int(self.position) + int(self.velocity)


@dataclass
class AgentSpec:
    """Everything the VecEnv needs to run N copies of one agent.

    Parameters
    ----------
    name:
        Base model name; agent ``i`` spawns as ``f"{name}_{i}"``.
    model_uri:
        SDF ``<uri>`` merged into each spawned model wrapper.
    observation_space / action_space:
        Per-agent Gymnasium spaces (SB3 batches across agents).
    joint_obs:
        Ordered joint reads that build the observation from a model's
        ``joint_state`` message. The summed widths must equal
        ``observation_space.shape[0]``.
    reward_fn(obs, action) -> float:
        Per-agent reward for a step.
    terminated_fn(obs) -> bool:
        Per-agent natural-termination test (e.g. pole fell).
    spawn_z / x_spacing / y_spacing:
        Spawn geometry. ``spawn_z`` must lift the model clear of the ground
        plane (z=0 buries it and contact friction pins the joints).
    """

    name: str
    model_uri: str
    observation_space: spaces.Space
    # (defaulted fields below) bare_model_uri: geometry-only model for the ECM
    # backends (harness / in-process) — no controllers, no effort limit. Falls
    # back to model_uri when empty.
    action_space: spaces.Space
    joint_obs: Sequence[JointObs]
    reward_fn: Callable[[np.ndarray, object], float]
    terminated_fn: Callable[[np.ndarray], bool]
    bare_model_uri: str = ""
    # --- Image observations (vision-in-the-loop) ---
    # (H, W, C) when the observation is the agent's onboard camera instead of
    # joint state. The world builder then loads the render/sensors system,
    # rewrites the model's camera <topic> to /rl/camera_<i> per agent, and the
    # in-process env reads frames from those topics. joint_obs may be empty.
    image_obs: Optional[tuple] = None
    # Static scenery included once per agent at that agent's spawn offset
    # (e.g. each rover gets its own line track). Any package:// model URI.
    per_agent_include_uri: str = ""
    # Population-based scenery-SHAPE randomization: names into
    # line_track_shapes.PRESETS, one drawn per agent at world-build time
    # (added 2026-08-25, after finding line_follower had trained against one
    # fixed track shape AND one fixed spawn point for this project's entire
    # history -- see docs/domain_randomization.md). When non-empty,
    # overrides per_agent_include_uri: each agent gets its own randomly
    # chosen shape, and its spawn pose is drawn from that shape's own
    # geometry (a random segment's pose -- tangent-aligned by construction,
    # so no separate "is this a valid point on the track" check is needed)
    # instead of the fixed spawn_y/spawn_yaw constants below. Empty tuple
    # (default) preserves the original single-shape, single-spawn-point
    # behavior via per_agent_include_uri + spawn_y/spawn_yaw, unchanged.
    track_shape_choices: tuple = ()
    # When True (and track_shape_choices is non-empty), EVERY reset draws a
    # fresh shape+spawn-point for that agent, rather than one fixed draw at
    # world-build time. Needs ALL of track_shape_choices spawned near each
    # agent simultaneously (not just the one chosen shape) so reset can
    # teleport onto a different pre-existing track each time -- real cost:
    # ~len(track_shape_choices)x more static geometry per agent. False
    # (default) keeps the cheaper population-based behavior: real variety
    # across the training population, but any one agent sees only the one
    # shape it originally drew for its whole lifetime.
    track_shape_reset_randomization: bool = False
    # Redraw a spawn pose up to this many times when the drawn one is
    # already terminal -- e.g. track_shape_reset_randomization landing the
    # rover where its camera sees no line at all, which makes the episode
    # unwinnable from step 1 rather than hard. Measured on line_follower:
    # ~1% of resets spawn blank, and every one of those died at step 1,
    # which accounted for essentially all of a 97.5% policy's residual
    # failures. 0 (default) keeps the old behavior: take whatever is drawn.
    # Validity reuses terminated_fn, so any spec gets this for free.
    spawn_redraw_attempts: int = 0
    # If True the whole model's world pose is restored to its spawn pose on
    # reset (mobile bases aren't world-pinned, so joint resets alone won't
    # bring them home).
    reset_model_pose: bool = False
    # --- Free-floating 3D base (Ant/Humanoid-style) ---
    # True when the model's root link has no parent joint at all (DART then
    # treats it as an unconstrained 6-DOF free body, unlike hopper/walker2d's
    # planar root, which is modeled as explicit prismatic/revolute joints and
    # needs no special handling here). When set, BASE_OBS_WIDTH extra floats
    # are prepended to the observation in MuJoCo's canonical order:
    # position section gets [z, quat_w, quat_x, quat_y, quat_z] before any
    # joint positions; velocity section gets [lin_vx, lin_vy, lin_vz, ang_wx,
    # ang_wy, ang_wz] before any joint velocities. x/y position are omitted
    # (MuJoCo convention -- absolute world position isn't policy-relevant).
    # ECM-only: HarnessCore.read_obs handles this; the duck-typed
    # obs_from_joint_state (peragent backend) does not.
    base_obs: bool = False
    # Name of the model's free/root link. Empty = the model's canonical link
    # (gz-sim's own notion of "the implicit free body"), which is correct for
    # every current use and only needs overriding for an unusual model.
    base_link_name: str = ""
    # Spawn offset/heading beyond the per-agent x_spacing grid (e.g. a rover
    # starting mid-track, aligned with the track's direction rather than at
    # the world origin facing +X).
    spawn_y: float = 0.0
    spawn_yaw: float = 0.0
    # Population-based dynamics randomization: fraction by which each agent's
    # mass + inertia is scaled, sampled per agent as U(1-x, 1+x) when the world
    # is built (0.0 = off). With N agents in one world this samples N points
    # from the dynamics distribution, so a policy trained across them is robust
    # to mass error — cheap domain randomization for sim-to-real, no ECM needed.
    mass_randomization: float = 0.0
    # Control-authority randomization: fraction by which each agent's actuator
    # command (velocity/force) is scaled, sampled per agent as U(1-x, 1+x)
    # (0.0 = off). Models real actuator-gain uncertainty and, unlike mass, bites
    # even under velocity control. Reproducible from the construction seed.
    action_gain_randomization: float = 0.0
    # Visual domain randomization for image_obs specs only (0.0 = off; ignored
    # for state-based specs). One per-agent draw at construction, applied to
    # every frame that agent renders for the rest of training — same
    # population-based, seed-reproducible pattern as the two DR fields above,
    # just in pixel space instead of physics. At strength x, each agent gets a
    # fixed: brightness scale U(1-x, 1+x), Gaussian pixel noise (std = 25x),
    # and JPEG re-encode quality (95 - 65x) — modeling a real camera's
    # auto-exposure drift and compression artifacts, neither of which the
    # renderer produces on its own. See inprocess_vec_env._apply_visual_dr.
    visual_randomization: float = 0.0
    # Track-color domain randomization for image_obs specs only (0.0 = off).
    # Same population-based pattern as visual_randomization (one per-agent
    # draw at construction, applied every frame that agent renders), but a
    # distinct mechanism and RNG stream so it can be tuned/disabled
    # independently: at strength x, each agent draws a fixed U(0, x)
    # lightening factor and blends its RAW frame's dark ("line") pixels
    # toward a lighter neutral color by that amount — modeling a real track
    # line that isn't perfectly black (grey/faded paint), which a synthetic
    # renderer with one fixed material color can't produce on its own.
    # Applied before the exposure/noise/compression stages above (it models
    # a property of the physical world, not the camera pipeline). See
    # inprocess_vec_env._apply_track_color_dr.
    track_color_randomization: float = 0.0
    # Per-step actuator noise (0.0 = off; in-process backend only, like
    # visual/track-color DR above — the harness backend doesn't implement
    # ANY DR mechanism yet, this included, not just this one) — distinct
    # from action_gain_randomization, which is a fixed per-agent bias
    # sampled once. Each agent draws a fixed U(0, x) noise magnitude at
    # construction (population-based, its own RNG stream), then every step
    # an independent N(0, magnitude) draw is added to that agent's raw
    # [-1, 1] action before it reaches action_to_commands — modeling real
    # per-tick motor jitter, not a static gain error.
    action_noise_randomization: float = 0.0
    # Simulated battery discharge (0.0 = off; in-process backend only, same
    # caveat as above) — a per-agent, per-EPISODE linear decay of actuator
    # authority: each agent draws a fixed U(0, x) max-depletion fraction at
    # every reset (resampled per episode, its own RNG stream), and the
    # action is scaled by (1 - depletion * min(t / max_episode_steps, 1)) as
    # the episode progresses, t = steps since that agent's last reset.
    # Models reduced available power over a deployment session; unlike the
    # other DR fields this one varies within an episode, not just across
    # the agent population.
    battery_discharge_randomization: float = 0.0
    spawn_z: float = 0.10
    x_spacing: float = 3.0
    y_spacing: float = 8.5
    frame_skip: int = 5
    max_episode_steps: int = 500
    # Whether the model's own links can collide with each other (SDF
    # <self_collide>). Off would let e.g. overlapping limbs pass through
    # each other — on is the physically correct default for every built-in.
    self_collide: bool = True
    # Extra SDF snippet spliced into the model wrapper (e.g. a world-fixed
    # joint pinning a rail). Kept generic so each model supplies its own.
    extra_sdf: str = ""
    extra_joints: tuple = field(default_factory=tuple)

    # --- In-sim harness (ECM) actuation + reset ---
    # action_to_commands(action) -> [(joint_name, mode, value), ...] with mode
    # in {"velocity", "force", "position"}. The in-sim harness applies these
    # via ECM (Joint.set_velocity / set_force / reset_position). None means
    # this spec isn't ECM-actuatable yet (harness path unavailable).
    action_to_commands: Optional[Callable[[object], Sequence]] = None
    # reset_joint_state(rng) -> {joint_name: (position, velocity)} set via ECM
    # at episode start — in place, no respawn. This is where clean per-agent
    # randomization lives (e.g. a small random initial pole angle), since ECM
    # reset_position sets the joint coordinate directly.
    reset_joint_state: Optional[Callable] = None
    # Auxiliary observation for image specs: a compact feature vector computed
    # from the SAME frame the policy sees, concatenated alongside the image as
    # a Dict observation ({"image": ..., "track": ...}, MultiInputPolicy).
    #
    # This exists because a CNN reading raw 64x64 pixels has to rediscover
    # quantities a classical line follower simply measures -- lateral error,
    # line angle, curvature, whether there is any track ahead. Handing them
    # over directly is what every classical implementation does, and all of
    # them are computable on the deployment laptop from the same JPEG, so
    # nothing here is privileged simulator state.
    #
    # aux_obs_fn(policy_visible_frame) -> float32 vector of length aux_obs_dim,
    # each element expected in [-1, 1]. Deliberately fed the AUGMENTED frame,
    # not the ground-truth one: on real hardware these features come off a
    # compressed, auto-exposed image, so they should be degraded in sim too.
    aux_obs_fn: Optional[Callable] = None
    aux_obs_dim: int = 0
    # Whether the raw frame reaches the POLICY. The camera renders either way,
    # and reward_fn/terminated_fn always see the frame; this only decides
    # whether the network is handed pixels alongside aux_obs_fn's features.
    # False means the policy observes the extracted features ALONE, so the
    # perception step is a fixed, portable function rather than learned
    # weights -- the same extractor can run on real JPEGs, and training drops
    # from a CNN to an MLP. Requires aux_obs_fn.
    policy_image: bool = True
    # Hard per-joint command ceiling, applied by HarnessCore.apply_actions
    # AFTER action_gain_randomization scales the command. action_to_commands
    # can only bound what the POLICY asks for; gain DR (and, on the harness
    # backend, anything else that scales a command downstream) can push the
    # result back outside the band the real hardware can execute. This clamps
    # the value that actually reaches the joint. Maps joint name -> max
    # absolute command in that joint's own units (rad/s for "velocity", N or
    # N*m for "force"). Empty = unclamped, the behavior every other spec has.
    command_limits: dict = field(default_factory=dict)
    # First-order low-pass on the action, applied by the in-process backend
    # AFTER every other action-side effect (noise, battery) and BEFORE the
    # command map: a_t = k * a_{t-1} + (1 - k) * a_raw, with k this value.
    # 0.0 disables it, which is what every other spec runs.
    #
    # It exists for hardware fidelity. The drive firmware does not step a
    # wheel to its setpoint, it ramps toward it under a per-wheel PID loop, so
    # a policy trained on instantly-realized setpoints learns commands the
    # real motors smear. Filtering in simulation trains against the same
    # smear.
    #
    # Enabling it also APPENDS the retained filter state (the previous
    # applied action) to the policy observation, which is not optional: the
    # filter makes the environment's response depend on history, and a policy
    # that cannot see a_{t-1} is acting on a partial observation. This is the
    # Markov break SB3's own custom-environment tips warn about, and the
    # reference implementation this was taken from has exactly that bug. The
    # state resets to the zero action at the start of every episode.
    action_lowpass: float = 0.0
    # Joints read as SENSORS rather than as observations in their own right.
    # Their velocities (rad/s, in this order) are handed to aux_obs_fn and
    # reward_fn as an extra argument each step, for specs whose observation is
    # an image and whose joint_obs is therefore empty. The spec decides how to
    # scale them; nothing normalizes on its behalf.
    #
    # This exists for hardware parity. The rover carries a quadrature encoder
    # per drive wheel, so wheel speed is information the deployed policy will
    # genuinely hold, and a reward computed from the commanded speed instead
    # pays out for a command the motors may not have executed.
    sensor_joints: tuple = ()
    # Encoder domain randomization for sensor_joints specs (0.0 = off;
    # in-process backend only, like the DR fields above). Two effects at
    # strength x, both modeling a real quadrature encoder rather than a
    # perfect joint-velocity read:
    #
    #   calibration -- each agent draws a FIXED per-wheel scale U(1-x, 1+x)
    #     at construction. Counts become metres through an assumed wheel
    #     diameter and counts-per-revolution, and neither is exact; the drive
    #     firmware carries separate TRIM_LEFT/TRIM_RIGHT constants precisely
    #     because the two wheels do not agree. A per-wheel scale error is what
    #     makes a rover think it is driving straight while it veers.
    #
    #   reading noise -- every step, the scaled reading is multiplied by
    #     1 + N(0, x). Relative rather than absolute, because a stopped
    #     quadrature wheel reports exactly zero counts, not noise.
    #
    # This matters more now than a spec without sensor_joints would suggest:
    # once the encoders are observed AND drive the reward, an unmodeled gap
    # between the simulated joint velocity and the real reading is a gap in
    # the policy's inputs and in its objective at the same time.
    encoder_noise_randomization: float = 0.0
    # Encoder LATENCY (0.0 = off; needs sensor_joints). A reply describes the
    # wheel as it was when the firmware sampled it, not as it is now: the
    # counts were accumulated over the previous interval, the UDP round trip
    # costs time, and the camera frame the reading is paired with was captured
    # at its own moment. At strength x each agent draws a fixed lag U(0, x),
    # measured in CONTROL STEPS and allowed to be fractional (the reading is
    # linearly interpolated between the two buffered steps it falls between),
    # held for the rest of training.
    #
    # Worth separating from noise because the failure is different in kind: a
    # noisy reading is wrong in a way that averages out, a late one is
    # systematically wrong in the direction the rover is accelerating, and a
    # policy that has only seen the former can be confidently wrong about the
    # latter.
    encoder_latency_randomization: float = 0.0
    # Encoder DROPOUT (0.0 = off; needs sensor_joints). UDP replies go missing.
    # At strength x each agent draws a fixed per-step drop probability U(0, x);
    # on a dropped step the previous reading is HELD rather than zeroed,
    # matching what the deployment client does (rover_line_deploy.py holds the
    # last good value and aborts after a run of failures). Substituting zero
    # would instead tell the policy the rover had stopped dead, which is a
    # different and much rarer event than a lost packet.
    encoder_dropout_randomization: float = 0.0
    # Camera MOUNT randomization for image_obs specs (0.0 = off), in metres.
    # Each agent's camera_link is displaced by U(-x, x) independently on each
    # axis, once at world-build time, and holds it for that agent's lifetime --
    # a printed mount's tolerance is a fixed property of one robot, not a
    # per-step disturbance.
    camera_mount_randomization: float = 0.0
    # Camera ANGLE randomization for image_obs specs (0.0 = off), in degrees.
    # Same population draw, applied to the camera_link's pitch and to the
    # sensor's horizontal field of view, U(-x, x) on each independently.
    #
    # These two are the camera-side DR that actually reaches a feature-based
    # observation. Photometric mechanisms do not: visual_randomization was
    # measured leaving rover_line's feature vector BIT-IDENTICAL at strength
    # 0.15, because brightness, noise and JPEG artifacts leave the dark-pixel
    # mask alone, and track_color_randomization is a cliff rather than a
    # gradient against a fixed threshold. Geometry is different in kind: move
    # or tilt the lens and the scan bands sample different ground, so every
    # offset, the heading and the curvature all shift continuously.
    camera_angle_randomization: float = 0.0
    # Consecutive steps `terminated_fn` must hold before the episode actually
    # ends (0 = end on the first True, which is every other spec's behaviour).
    #
    # Debounced termination, for tasks where the failure condition is a
    # sustained state rather than an instant. A line follower losing sight of
    # the track for one frame has not failed -- a corner can swing the line out
    # of view briefly -- so ending immediately throws away recoverable episodes
    # and teaches the policy nothing about recovering.
    termination_grace_steps: int = 0
    # Whether reward_fn and terminated_fn are handed a per-step STATE mapping
    # as one extra argument. Off by default: it changes both signatures.
    #
    # A mapping rather than more positional arguments, because the set of
    # things a task might need grows and a five-argument reward is unreadable.
    # It currently carries:
    #
    #   "prev_action" -- the action applied on the previous step, so a spec
    #     can price the CHANGE in command (jerk). An alternative to
    #     action_lowpass for the same problem: one filters the command, the
    #     other charges for it. Filtering needs the filter state in the
    #     observation to stay Markov; a penalty needs the previous action only
    #     inside the reward and leaves the observation alone.
    #
    #   "pose" -- the base link's (x, y, z, yaw) in world coordinates. This is
    #     privileged simulator state and must never reach the OBSERVATION; it
    #     is here so reward and termination can be judged against the track
    #     itself (how far round, how far off) rather than against what the
    #     camera happens to see.
    provide_step_state: bool = False

    @property
    def actuated_joints(self):
        """Joint names this spec commands, in action_to_commands order.

        Probed by calling action_to_commands on a zero action, which every
        mapping in this module tolerates. Empty when the spec is not
        ECM-actuatable.
        """
        if self.action_to_commands is None:
            return ()
        try:
            probe = np.zeros(int(np.prod(self.action_space.shape)), dtype=float)
            return tuple(jn for jn, _mode, _v in self.action_to_commands(probe))
        except Exception:  # noqa: B902 - a spec that cannot be probed has none
            return ()

    @property
    def policy_aux_dim(self):
        """Width of the feature vector the POLICY sees.

        aux_obs_fn's own output, plus the retained action-filter state when
        action_lowpass is on. Single source of truth for that width, so the
        declared space and the packed observation cannot drift apart.
        """
        if self.aux_obs_dim <= 0:
            return 0
        if self.action_lowpass <= 0.0:
            return self.aux_obs_dim
        return self.aux_obs_dim + int(np.prod(self.action_space.shape))

    @property
    def policy_observation_space(self):
        """The space the POLICY sees: Dict when aux_obs_fn is configured.

        `observation_space` stays the raw per-frame space so reward_fn,
        terminated_fn and every existing caller keep working unchanged.
        """
        if self.aux_obs_fn is None or self.aux_obs_dim <= 0:
            return self.observation_space
        aux_space = spaces.Box(low=-1.0, high=1.0, shape=(self.policy_aux_dim,), dtype=np.float32)
        if not self.policy_image:
            return aux_space
        return spaces.Dict(
            {
                "image": self.observation_space,
                "track": aux_space,
            }
        )

    def __post_init__(self):
        if self.aux_obs_fn is not None and self.image_obs is None:
            raise ValueError(
                f"AgentSpec({self.name!r}): aux_obs_fn is only supported "
                f"alongside an image observation"
            )
        if not self.policy_image and (self.aux_obs_fn is None or self.aux_obs_dim <= 0):
            raise ValueError(
                f"AgentSpec({self.name!r}): policy_image=False leaves the policy "
                f"with no observation at all; it requires aux_obs_fn"
            )
        for _field in ("camera_mount_randomization", "camera_angle_randomization"):
            if getattr(self, _field) and self.image_obs is None:
                raise ValueError(
                    f"AgentSpec({self.name!r}): {_field} needs image_obs -- "
                    f"there is no camera to move"
                )
        for _field in (
            "encoder_noise_randomization",
            "encoder_latency_randomization",
            "encoder_dropout_randomization",
        ):
            if getattr(self, _field) and not self.sensor_joints:
                raise ValueError(
                    f"AgentSpec({self.name!r}): {_field} needs sensor_joints -- "
                    f"there is no encoder reading to perturb"
                )
        if self.action_lowpass and self.aux_obs_fn is None:
            # The filter's retained state rides in the aux vector, so without
            # aux_obs_fn there is nowhere to put it and the policy would be
            # left acting on a partial observation -- silently, which is the
            # worst way for this to fail.
            raise ValueError(
                f"AgentSpec({self.name!r}): action_lowpass requires aux_obs_fn, "
                f"which is where the filter state is exposed to the policy"
            )
        if self.image_obs is not None:
            if tuple(self.observation_space.shape) != tuple(self.image_obs):
                raise ValueError(
                    f"AgentSpec({self.name!r}): observation_space shape "
                    f"{self.observation_space.shape} != image_obs "
                    f"{self.image_obs}"
                )
        else:
            obs_dim = self.observation_space.shape[0]
            joint_dim = sum(j.width for j in self.joint_obs)
            if self.base_obs:
                joint_dim += BASE_OBS_WIDTH
            if joint_dim != obs_dim:
                base_note = f" + base_obs({BASE_OBS_WIDTH})" if self.base_obs else ""
                raise ValueError(
                    f"AgentSpec({self.name!r}): joint_obs widths{base_note} "
                    f"sum to {joint_dim} but observation_space has {obs_dim} "
                    f"dims"
                )
        if self.spawn_z <= 0.0:
            raise ValueError(
                f"AgentSpec({self.name!r}): spawn_z must be > 0 to clear the "
                f"ground plane (got {self.spawn_z})"
            )

    @property
    def obs_dim(self) -> int:
        return int(self.observation_space.shape[0])

    def obs_from_joint_state(self, model_msg) -> np.ndarray:
        """Build the observation vector from a gz ``Model`` joint_state msg.

        Duck-typed: ``model_msg.joint`` is iterated; each element must expose
        ``.name`` and ``.axis1.position`` / ``.axis1.velocity``. Missing
        joints contribute zeros (matches the original cartpole behavior, and
        keeps a dropped message from raising mid-episode).
        """
        if self.base_obs:
            raise NotImplementedError(
                f"AgentSpec({self.name!r}): base_obs is ECM-only (the "
                f"joint_state message this method reads has no free-body "
                f"pose/twist) -- use the inprocess or harness backend, not "
                f"peragent."
            )
        by_name = {j.name: j for j in model_msg.joint}
        out = np.zeros(self.obs_dim, dtype=np.float32)
        k = 0
        for spec in self.joint_obs:
            j = by_name.get(spec.joint)
            if spec.position:
                out[k] = float(j.axis1.position) if j is not None else 0.0
                k += 1
            if spec.velocity:
                out[k] = float(j.axis1.velocity) if j is not None else 0.0
                k += 1
        return out


# --------------------------------------------------------------------------- #
# Spec-building helpers — the patterns every port repeats.
#
# Distilled from the eight built-in specs: torque/velocity action mapping with
# clipping and scalar-probe safety, MuJoCo's positions-then-velocities obs
# layout, uniform reset noise, forward-progress rewards, and planar health
# termination. Compose these instead of hand-writing the closures.
# --------------------------------------------------------------------------- #


def proportional_forces(joints, gears):
    """Map a Box(-1,1) action to per-joint forces/torques.

    ``gears`` is one float for all joints or a per-joint sequence. Robust to
    scalar probes (joint discovery calls with 0/1) via np.resize.
    """
    joints = tuple(joints)
    g = np.resize(np.asarray(gears, dtype=float), len(joints))

    def _cmds(action):
        a = np.resize(np.clip(np.asarray(action, dtype=float).ravel(), -1.0, 1.0), len(joints))
        return [(j, "force", float(a[i] * g[i])) for i, j in enumerate(joints)]

    return _cmds


def proportional_velocities(joints, speeds):
    """Map a Box(-1,1) action to per-joint velocity commands."""
    joints = tuple(joints)
    s = np.resize(np.asarray(speeds, dtype=float), len(joints))

    def _cmds(action):
        a = np.resize(np.clip(np.asarray(action, dtype=float).ravel(), -1.0, 1.0), len(joints))
        return [(j, "velocity", float(a[i] * s[i])) for i, j in enumerate(joints)]

    return _cmds


def clamped_velocities(joints, v_max, v_min=0.0, deadzone=0.0, reverse=True):
    """Map a Box(-1,1) action onto a velocity band the hardware can execute.

    ``proportional_velocities`` scales the action linearly onto plus/minus
    ``v_max``, which lets a policy spend most of its action range commanding
    speeds a real motor cannot hold, and spend half of it driving backwards.
    Two modes here, both bounded by ``v_max``:

    ``reverse=True`` maps onto {0} U +/-[v_min, v_max], keeping zero as a
    real stop and both directions available::

        |a| <  deadzone  ->  0
        |a| >= deadzone  ->  sign(a) * (v_min + |a|*(v_max-v_min))

    ``reverse=False`` maps the WHOLE action range onto [v_min, v_max]
    forward, with no stop and no reverse at all::

        a -> v_min + (a+1)/2 * (v_max - v_min)

    which is the right shape for a wheel that only ever needs to drive one
    way. It also makes creeping and back-and-forth shuffling physically
    unreachable rather than merely lower-scoring -- see
    docs/examples/line_follower.md. ``deadzone`` is ignored when
    ``reverse=False`` (there is no zero to sit in).

    ``v_min=0``, ``deadzone=0``, ``reverse=True`` reduces exactly to
    ``proportional_velocities``.
    """
    joints = tuple(joints)
    hi = np.resize(np.asarray(v_max, dtype=float), len(joints))
    lo = np.resize(np.asarray(v_min, dtype=float), len(joints))
    if np.any(lo < 0.0) or np.any(hi <= 0.0) or np.any(lo >= hi):
        raise ValueError(f"clamped_velocities: need 0 <= v_min < v_max (got {v_min}, {v_max})")

    def _cmds(action):
        a = np.resize(np.clip(np.asarray(action, dtype=float).ravel(), -1.0, 1.0), len(joints))
        if reverse:
            mag = np.abs(a)
            v = np.where(mag < deadzone, 0.0, np.sign(a) * (lo + mag * (hi - lo)))
        else:
            v = lo + 0.5 * (a + 1.0) * (hi - lo)
        return [(j, "velocity", float(v[i])) for i, j in enumerate(joints)]

    return _cmds


def mean_forward_velocities(joints, v_max, v_min):
    """Differential-drive map whose MEAN wheel speed is always forward.

    Each wheel is commanded over the full symmetric range ``[-v_max, v_max]``,
    then the pair is projected onto ``mean(v) >= v_min`` by shifting both
    wheels equally. The shift preserves the sum, so any excess above
    ``v_max`` on one wheel is handed to the other rather than clipped away,
    and the mean lands exactly on ``v_min`` instead of somewhere below it.

    This forbids exactly the behavior that needs forbidding and nothing more.
    Net creeping and back-and-forth shuffling are unreachable, because both
    require a mean below ``v_min``. A tight corner pivot stays reachable,
    because one wheel may still reverse as long as the other more than pays
    for it.

    ``clamped_velocities(reverse=False)`` is the stricter version, holding
    EACH wheel above ``v_min``. That also removes shuffling, but it caps the
    turn radius at ``(sep/2)*(v_max+v_min)/(v_max-v_min)``, which measured out
    at 0.12 m for the rover: wider than the corner it had to take (see
    docs/examples/line_follower.md, "The Corner Was The Wall"). Constraining
    the mean instead drops the reachable radius to about 0.02 m.
    """
    joints = tuple(joints)
    if len(joints) != 2:
        raise ValueError(f"mean_forward_velocities needs exactly 2 joints, got {joints}")
    v_max, v_min = float(v_max), float(v_min)
    if not 0.0 <= v_min < v_max:
        raise ValueError(f"mean_forward_velocities: need 0 <= v_min < v_max ({v_min}, {v_max})")

    def _cmds(action):
        a = np.resize(np.clip(np.asarray(action, dtype=float).ravel(), -1.0, 1.0), 2)
        v = a * v_max
        shortfall = v_min - 0.5 * (v[0] + v[1])
        if shortfall > 0.0:
            v = v + shortfall  # equal shift: raises the mean, keeps the turn
            # Re-cap without losing the sum: whatever one wheel cannot take,
            # the other absorbs. Reachable because the mean is at most v_min
            # here, so the pair together always has room under 2*v_max.
            for hi, lo in ((0, 1), (1, 0)):
                excess = v[hi] - v_max
                if excess > 0.0:
                    v[hi], v[lo] = v_max, v[lo] + excess
        return [(j, "velocity", float(v[i])) for i, j in enumerate(joints)]

    return _cmds


def pos_then_vel_obs(pos_joints, vel_joints):
    """Build the MuJoCo observation layout: positions, then velocities."""
    return tuple(JointObs(j, velocity=False) for j in pos_joints) + tuple(
        JointObs(j, position=False) for j in vel_joints
    )


def uniform_reset(joints, pos_eps, vel_eps=None):
    """Reset every joint to U(-pos_eps, pos_eps) position (and velocity)."""
    joints = tuple(joints)
    v_eps = pos_eps if vel_eps is None else vel_eps

    def _reset(rng):
        return {
            j: (float(rng.uniform(-pos_eps, pos_eps)), float(rng.uniform(-v_eps, v_eps)))
            for j in joints
        }

    return _reset


def forward_progress_reward(vel_index, alive_bonus=1.0, ctrl_cost=1e-3):
    """Locomotion reward: forward velocity + alive bonus - control cost."""

    def _reward(obs, action):
        a = np.clip(np.asarray(action, dtype=float).ravel(), -1.0, 1.0)
        return float(obs[vel_index] + alive_bonus - ctrl_cost * float(np.square(a).sum()))

    return _reward


def planar_health_termination(
    spawn_z, min_z, max_pitch, max_z=np.inf, z_index=0, pitch_index=1, obs_bound=100.0
):
    """MuJoCo-style planar health check (height band, pitch band, obs bound).

    ``obs[z_index]`` is the vertical-slide displacement (absolute height =
    spawn_z + displacement); ``obs[pitch_index]`` the root pitch.
    """

    def _terminated(obs):
        z = spawn_z + float(obs[z_index])
        healthy = (
            min_z < z < max_z
            and abs(float(obs[pitch_index])) < max_pitch
            and bool(np.all(np.abs(obs) < obs_bound))
        )
        return not healthy

    return _terminated


# --------------------------------------------------------------------------- #
# Built-in agent specs (the registry the make_multi() factory draws from)
# --------------------------------------------------------------------------- #

_POLE_ANGLE_THRESHOLD = 0.20944
_CART_POSITION_THRESHOLD = 2.4

# N pushed on the slider (bang-bang force), the classic CartPole actuation.
# The pole is genuinely unstable under force control: a constant push tips it
# in ~3 steps and a random policy survives ~7 steps (median 6), while a trained
# PPO policy reaches the 500-step cap — a textbook benchmark. NOTE: force
# actuation requires the model to spawn CLEAR of the ground (see spawn_z below);
# a cart resting on the ground plane is pinned by contact friction, which
# velocity control would silently override (kinematic) but force cannot.
_CART_FORCE = 10.0


def _cartpole_action_to_commands(action):
    # Discrete 1 -> +F, 0 -> -F, applied as a slider force command.
    # Robust to a scalar (offline) or a length-1 array (the harness passes each
    # agent's action as a row of the (n_agents, act_dim) matrix).
    a = int(round(float(np.ravel(action)[0])))
    f = _CART_FORCE if a == 1 else -_CART_FORCE
    return [("slider_to_cart", "force", f)]


def _cartpole_reset_joint_state(rng):
    # Cart centered, pole at a small random angle (standard CartPole ~+/-0.05).
    # Set directly via ECM — no free-fall/respawn, so it's exact per agent.
    return {
        "slider_to_cart": (0.0, 0.0),
        "cart_to_pole": (float(rng.uniform(-0.05, 0.05)), 0.0),
    }


def _cartpole_spec() -> AgentSpec:
    obs_space = spaces.Box(
        low=np.array([-4.8, -np.inf, -0.41887903, -np.inf], dtype=np.float32),
        high=np.array([4.8, np.inf, 0.41887903, np.inf], dtype=np.float32),
        shape=(4,),
        dtype=np.float32,
    )
    return AgentSpec(
        name="cartpole",
        model_uri="package://gazebo_gymnasium_resources/models/cartpole",
        bare_model_uri="package://gazebo_gymnasium_resources/models/cartpole_bare",
        observation_space=obs_space,
        action_space=spaces.Discrete(2),
        # obs order: cart_pos, cart_vel, pole_angle, pole_ang_vel
        joint_obs=(JointObs("slider_to_cart"), JointObs("cart_to_pole")),
        reward_fn=lambda obs, action: 1.0,
        terminated_fn=lambda obs: bool(
            abs(obs[2]) > _POLE_ANGLE_THRESHOLD or abs(obs[0]) > _CART_POSITION_THRESHOLD
        ),
        # Clear of the ground plane: the cart's collision box must NOT rest on
        # the ground or contact friction pins it against force actuation.
        spawn_z=0.60,
        extra_joints=(("world_to_slider", "world", "slider"),),
        action_to_commands=_cartpole_action_to_commands,
        reset_joint_state=_cartpole_reset_joint_state,
    )


_cartpole_continuous_action_to_commands = proportional_forces(("slider_to_cart",), _CART_FORCE)


def _cartpole_continuous_spec() -> AgentSpec:
    """Continuous-force cartpole: exercises Box actions (SAC/TD3/DDPG)."""
    return replace(
        _cartpole_spec(),
        name="cartpole_continuous",
        action_space=spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32),
        action_to_commands=_cartpole_continuous_action_to_commands,
    )


# --------------------------------------------------------------------------- #
# InvertedDoublePendulum (MuJoCo port): cart + two hinged 0.6 m poles.
# Obs layout: [cart_pos, cart_vel, th1, w1, th2, w2]  (th2 relative to pole 1).
# --------------------------------------------------------------------------- #

_IDP_POLE_LEN = 0.6
_IDP_TIP_MAX = 2 * _IDP_POLE_LEN
_IDP_TIP_THRESHOLD = 1.0  # terminate when the tip drops below this height
_IDP_FORCE = 100.0  # N full-scale slider force for a Box(-1,1) action


def _idp_tip(obs):
    """Tip (x, height) from joint state — x absolute, height cart-relative."""
    th1, th2 = float(obs[2]), float(obs[4])
    x = float(obs[0]) + _IDP_POLE_LEN * (np.sin(th1) + np.sin(th1 + th2))
    h = _IDP_POLE_LEN * (np.cos(th1) + np.cos(th1 + th2))
    return x, h


def _idp_reward(obs, action):
    # Shaped like MuJoCo's: alive bonus minus tip-distance and velocity
    # penalties (coefficients tuned to this model's scale, documented variant).
    x, h = _idp_tip(obs)
    w1, w2 = float(obs[3]), float(obs[5])
    return float(10.0 - 0.01 * x * x - 10.0 * (_IDP_TIP_MAX - h) ** 2 - 1e-3 * (w1 * w1 + w2 * w2))


_idp_action_to_commands = proportional_forces(("slider_to_cart",), _IDP_FORCE)


def _idp_reset_joint_state(rng):
    return {
        "slider_to_cart": (0.0, 0.0),
        "cart_to_pole": (float(rng.uniform(-0.05, 0.05)), 0.0),
        "pole_to_pole2": (float(rng.uniform(-0.05, 0.05)), 0.0),
    }


def _inverted_double_pendulum_spec() -> AgentSpec:
    obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)
    return AgentSpec(
        name="inverted_double_pendulum",
        model_uri=("package://gazebo_gymnasium_resources/models/inverted_double_pendulum_bare"),
        bare_model_uri=(
            "package://gazebo_gymnasium_resources/models/inverted_double_pendulum_bare"
        ),
        observation_space=obs_space,
        action_space=spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32),
        joint_obs=(
            JointObs("slider_to_cart"),
            JointObs("cart_to_pole"),
            JointObs("pole_to_pole2"),
        ),
        reward_fn=_idp_reward,
        terminated_fn=lambda obs: bool(_idp_tip(obs)[1] <= _IDP_TIP_THRESHOLD),
        spawn_z=0.60,  # force actuation: keep the cart off the ground
        max_episode_steps=1000,
        extra_joints=(("world_to_slider", "world", "slider"),),
        action_to_commands=_idp_action_to_commands,
        reset_joint_state=_idp_reset_joint_state,
    )


# --------------------------------------------------------------------------- #
# Hopper (MuJoCo port): planar monopod, root modeled as joints (MuJoCo-style),
# forward = +Y. Obs (11) = [z_disp, pitch, thigh, leg, foot,
#                           v_fwd, v_z, v_pitch, v_thigh, v_leg, v_foot].
# --------------------------------------------------------------------------- #

_HOPPER_TORQUE = 200.0  # N*m full-scale per actuated joint (MuJoCo gear)
_HOPPER_SPAWN_Z = 1.25  # torso-root height at spawn (MuJoCo initial z)
_HOPPER_MIN_Z = 0.7  # unhealthy below this absolute torso height
_HOPPER_MAX_PITCH = 0.2  # rad, unhealthy beyond

_HOPPER_ACT = ("thigh_joint", "leg_joint", "foot_joint")
_hopper_action_to_commands = proportional_forces(_HOPPER_ACT, _HOPPER_TORQUE)
_hopper_reward = forward_progress_reward(vel_index=5)
_hopper_terminated = planar_health_termination(_HOPPER_SPAWN_Z, _HOPPER_MIN_Z, _HOPPER_MAX_PITCH)
_hopper_reset_joint_state = uniform_reset(
    ("root_fwd", "root_up", "root_pitch") + _HOPPER_ACT, 0.005
)


def _hopper_spec() -> AgentSpec:
    obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(11,), dtype=np.float32)
    return AgentSpec(
        name="hopper",
        model_uri="package://gazebo_gymnasium_resources/models/hopper_bare",
        bare_model_uri=("package://gazebo_gymnasium_resources/models/hopper_bare"),
        observation_space=obs_space,
        action_space=spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32),
        # MuJoCo layout: positions (root x excluded), then all velocities.
        joint_obs=pos_then_vel_obs(
            ("root_up", "root_pitch") + _HOPPER_ACT,
            ("root_fwd", "root_up", "root_pitch") + _HOPPER_ACT,
        ),
        reward_fn=_hopper_reward,
        terminated_fn=_hopper_terminated,
        spawn_z=_HOPPER_SPAWN_Z,
        frame_skip=4,
        max_episode_steps=1000,
        extra_joints=(("world_to_anchor", "world", "anchor"),),
        action_to_commands=_hopper_action_to_commands,
        reset_joint_state=_hopper_reset_joint_state,
    )


# --------------------------------------------------------------------------- #
# Walker2d (MuJoCo port): planar biped — the hopper recipe with two legs.
# Obs (17) = [z, pitch, 6 leg angles | v_fwd, v_z, v_pitch, 6 leg velocities].
# --------------------------------------------------------------------------- #

_WALKER_TORQUE = 100.0
_WALKER_SPAWN_Z = 1.25
_WALKER_Z_RANGE = (0.8, 2.0)
_WALKER_MAX_PITCH = 1.0
_WALKER_LEG_JOINTS = (
    "thigh_joint",
    "leg_joint",
    "foot_joint",
    "thigh_left_joint",
    "leg_left_joint",
    "foot_left_joint",
)

_walker_action_to_commands = proportional_forces(_WALKER_LEG_JOINTS, _WALKER_TORQUE)
_walker_reward = forward_progress_reward(vel_index=8)
_walker_terminated = planar_health_termination(
    _WALKER_SPAWN_Z, _WALKER_Z_RANGE[0], _WALKER_MAX_PITCH, max_z=_WALKER_Z_RANGE[1]
)
_walker_reset_joint_state = uniform_reset(
    ("root_fwd", "root_up", "root_pitch") + _WALKER_LEG_JOINTS, 0.005
)


def _walker2d_spec() -> AgentSpec:
    obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(17,), dtype=np.float32)
    layout = pos_then_vel_obs(
        ("root_up", "root_pitch") + _WALKER_LEG_JOINTS,
        ("root_fwd", "root_up", "root_pitch") + _WALKER_LEG_JOINTS,
    )
    return AgentSpec(
        name="walker2d",
        model_uri="package://gazebo_gymnasium_resources/models/walker2d_bare",
        bare_model_uri=("package://gazebo_gymnasium_resources/models/walker2d_bare"),
        observation_space=obs_space,
        action_space=spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32),
        joint_obs=layout,
        reward_fn=_walker_reward,
        terminated_fn=_walker_terminated,
        spawn_z=_WALKER_SPAWN_Z,
        frame_skip=4,
        max_episode_steps=1000,
        extra_joints=(("world_to_anchor", "world", "anchor"),),
        action_to_commands=_walker_action_to_commands,
        reset_joint_state=_walker_reset_joint_state,
    )


# --------------------------------------------------------------------------- #
# HalfCheetah (MuJoCo port): planar, horizontal torso, back/front legs.
# No health termination — episodes end on truncation only (MuJoCo semantics).
# Obs (17) = [z, pitch, 6 leg angles | v_fwd, v_z, v_pitch, 6 leg velocities].
# --------------------------------------------------------------------------- #

_CHEETAH_SPAWN_Z = 0.77
_CHEETAH_JOINTS = ("bthigh", "bshin", "bfoot", "fthigh", "fshin", "ffoot")
_CHEETAH_GEARS = (120.0, 90.0, 60.0, 120.0, 60.0, 30.0)

_cheetah_action_to_commands = proportional_forces(_CHEETAH_JOINTS, _CHEETAH_GEARS)
_cheetah_reward = forward_progress_reward(vel_index=8, alive_bonus=0.0, ctrl_cost=0.1)
_cheetah_reset_joint_state = uniform_reset(
    ("root_fwd", "root_up", "root_pitch") + _CHEETAH_JOINTS, 0.005
)


def _half_cheetah_spec() -> AgentSpec:
    obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(17,), dtype=np.float32)
    layout = pos_then_vel_obs(
        ("root_up", "root_pitch") + _CHEETAH_JOINTS,
        ("root_fwd", "root_up", "root_pitch") + _CHEETAH_JOINTS,
    )
    return AgentSpec(
        name="half_cheetah",
        model_uri=("package://gazebo_gymnasium_resources/models/half_cheetah_bare"),
        bare_model_uri=("package://gazebo_gymnasium_resources/models/half_cheetah_bare"),
        observation_space=obs_space,
        action_space=spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32),
        joint_obs=layout,
        reward_fn=_cheetah_reward,
        # MuJoCo half-cheetah has no health termination; keep a pure numerical
        # guard so a solver blow-up can't silently poison training.
        terminated_fn=lambda obs: not bool(np.all(np.abs(obs) < 1000.0)),
        spawn_z=_CHEETAH_SPAWN_Z,
        frame_skip=5,
        max_episode_steps=1000,
        extra_joints=(("world_to_anchor", "world", "anchor"),),
        action_to_commands=_cheetah_action_to_commands,
        reset_joint_state=_cheetah_reset_joint_state,
    )


# --------------------------------------------------------------------------- #
# Reacher (MuJoCo port): 2-link horizontal-plane arm + goal-as-joints target.
# Obs (6) = [th0, th1, target_x, target_y, v0, v1]; dense distance reward.
# --------------------------------------------------------------------------- #

_REACHER_L0, _REACHER_L1 = 0.1, 0.11
_REACHER_TORQUE = 5.0


def _reacher_fingertip(obs):
    th0, th1 = float(obs[0]), float(obs[1])
    x = _REACHER_L0 * np.cos(th0) + _REACHER_L1 * np.cos(th0 + th1)
    y = _REACHER_L0 * np.sin(th0) + _REACHER_L1 * np.sin(th0 + th1)
    return x, y


def _reacher_reward(obs, action):
    fx, fy = _reacher_fingertip(obs)
    dist = float(np.hypot(fx - float(obs[2]), fy - float(obs[3])))
    a = np.clip(np.asarray(action, dtype=float).ravel(), -1.0, 1.0)
    return float(-dist - 0.1 * float(np.square(a).sum()))


_reacher_action_to_commands = proportional_forces(("joint0", "joint1"), _REACHER_TORQUE)


def _reacher_reset_joint_state(rng):
    # arm pose randomized like MuJoCo; the GOAL moves through the same reset
    # path because it is two prismatic joints (goal within reach, |g|<=0.198).
    return {
        "joint0": (float(rng.uniform(-0.1, 0.1)), 0.0),
        "joint1": (float(rng.uniform(-0.1, 0.1)), 0.0),
        "target_x": (float(rng.uniform(-0.14, 0.14)), 0.0),
        "target_y": (float(rng.uniform(-0.14, 0.14)), 0.0),
    }


def _reacher_spec() -> AgentSpec:
    obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)
    return AgentSpec(
        name="reacher",
        model_uri="package://gazebo_gymnasium_resources/models/reacher_bare",
        bare_model_uri=("package://gazebo_gymnasium_resources/models/reacher_bare"),
        observation_space=obs_space,
        action_space=spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32),
        joint_obs=(
            JointObs("joint0", velocity=False),
            JointObs("joint1", velocity=False),
            JointObs("target_x", velocity=False),
            JointObs("target_y", velocity=False),
            JointObs("joint0", position=False),
            JointObs("joint1", position=False),
        ),
        reward_fn=_reacher_reward,
        terminated_fn=lambda obs: not bool(np.all(np.abs(obs) < 100.0)),
        spawn_z=0.05,
        frame_skip=2,  # MuJoCo reacher dt = 0.02
        max_episode_steps=50,  # MuJoCo reacher truncates at 50
        extra_joints=(
            ("world_to_base", "world", "base"),
            ("world_to_target_anchor", "world", "target_anchor"),
        ),
        action_to_commands=_reacher_action_to_commands,
        reset_joint_state=_reacher_reset_joint_state,
    )


# --------------------------------------------------------------------------- #
# Ant (MuJoCo port): quadruped, free-floating 3D base (unlike hopper/walker's
# planar root, which is modeled as explicit joints -- the torso here has NO
# parent joint, so it's a genuine 6-DOF free body, read via AgentSpec.base_obs
# / HarnessCore's Link API). Obs (27) = MuJoCo's canonical layout: [z, quat(4),
# 8 joint angles | v_fwd, v_side, v_z, w_roll, w_pitch, w_yaw, 8 joint
# velocities]. First free-base spec in this codebase -- see docs/creating_
# your_own_agent.md if porting another one (humanoid, humanoidstandup).
# --------------------------------------------------------------------------- #

_ANT_GEAR = 150.0  # N*m full-scale per actuated joint (MJCF motor gear)
_ANT_SPAWN_Z = 0.75  # torso height at spawn (MuJoCo initial z)
_ANT_HEALTHY_Z = (0.2, 1.0)  # Gymnasium Ant-v4's own healthy_z_range default

_ANT_JOINTS = ("hip_1", "ankle_1", "hip_2", "ankle_2", "hip_3", "ankle_3", "hip_4", "ankle_4")
_ant_action_to_commands = proportional_forces(_ANT_JOINTS, _ANT_GEAR)
_ant_reward = forward_progress_reward(vel_index=13, ctrl_cost=0.5)
_ant_reset_joint_state = uniform_reset(_ANT_JOINTS, 0.005)


def _ant_terminated(obs):
    z = float(obs[0])
    healthy = _ANT_HEALTHY_Z[0] <= z <= _ANT_HEALTHY_Z[1] and bool(np.all(np.abs(obs) < 100.0))
    return not healthy


def _ant_spec() -> AgentSpec:
    obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(27,), dtype=np.float32)
    return AgentSpec(
        name="ant",
        model_uri="package://gazebo_gymnasium_resources/models/ant_bare",
        bare_model_uri=("package://gazebo_gymnasium_resources/models/ant_bare"),
        observation_space=obs_space,
        action_space=spaces.Box(low=-1.0, high=1.0, shape=(8,), dtype=np.float32),
        base_obs=True,
        # positions-only-then-velocities-only, required by base_obs's
        # two-block assembly (see HarnessCore._read_base_obs).
        joint_obs=pos_then_vel_obs(_ANT_JOINTS, _ANT_JOINTS),
        reward_fn=_ant_reward,
        terminated_fn=_ant_terminated,
        spawn_z=_ANT_SPAWN_Z,
        max_episode_steps=1000,
        reset_model_pose=True,  # free base: restore torso pose on reset
        action_to_commands=_ant_action_to_commands,
        reset_joint_state=_ant_reset_joint_state,
    )


# --------------------------------------------------------------------------- #
# Line follower: differential-drive rover, onboard 64x64 camera as the WHOLE
# observation. Reward/termination are computed from the image itself (line
# centroid in the lower half), so the env is self-contained vision-in-the-loop.
# --------------------------------------------------------------------------- #

_LF_IMAGE = (64, 64, 3)
# Hardware units are authoritative here: the firmware talks m/s and converts
# with WHEEL_DIAMETER_M, so the speed band is DEFINED in m/s and the joint
# commands are derived from it, not the other way around. Getting this
# backwards is how the sim and the rover end up disagreeing about what an
# action means.
#
# 0.035 m = the firmware's WHEEL_DIAMETER_M/2. The CAD wheel's own inertia
# implies 0.0343 (r = sqrt(2*Izz/m)), a 2 percent disagreement between the
# drawing and the measured part; the measured one wins, because it is what
# the rover actually converts commands with.
_LF_WHEEL_RADIUS = 0.035
# Speed band the PHYSICAL rover can actually execute, converted to wheel rad/s
# here (the firmware talks m/s). See docs/examples/line_follower.md,
# "Speed Range And Motor Response", for where each number comes from.
#
#   _LF_WHEEL_SPEED      0.51 m/s -- full scale. Unchanged; every shipped
#                        checkpoint was trained at it, and it leaves ~2x
#                        headroom under the caster-pop limit below.
#   _LF_WHEEL_SPEED_MIN  0.10 m/s -- the slowest wheel speed this spec will
#                        command. The firmware's own hard floor is lower
#                        (0.013 m/s: below that FLOOR_ENABLE_TICKS=2 counts
#                        per 50 ms tick at ENCODER_CPR_WHEEL=680 means the
#                        PWM floor never engages and the wheel produces no
#                        motion at all), but that gate is a control-law
#                        threshold, not a measurement of stiction under load,
#                        and a floor that low leaves room to creep. 0.10 m/s
#                        is set for the TRAINING reason below instead, and
#                        sits comfortably above the firmware gate.
#   _LF_WHEEL_SPEED_POP  1.0 m/s -- the front caster lifts past this, the
#                        driven wheels lose contact, and encoder counts stop
#                        tracking ground distance. Hard ceiling, enforced
#                        post-gain via AgentSpec.command_limits.
_LF_SPEED_MAX_MPS = 0.50  # full scale
_LF_SPEED_MIN_MPS = 0.10  # slowest this spec will command
_LF_SPEED_POP_MPS = 1.00  # front caster lifts past this
_LF_WHEEL_SPEED = _LF_SPEED_MAX_MPS / _LF_WHEEL_RADIUS  # ~14.29 rad/s
_LF_WHEEL_SPEED_MIN = _LF_SPEED_MIN_MPS / _LF_WHEEL_RADIUS  # ~2.86 rad/s
_LF_WHEEL_SPEED_POP = _LF_SPEED_POP_MPS / _LF_WHEEL_RADIUS  # ~28.6 rad/s
# Forward-only actuation. A line follower never needs to reverse, and the real
# rover's task is to complete the track, so both wheels are held in
# [_LF_WHEEL_SPEED_MIN, _LF_WHEEL_SPEED] and the policy steers purely by the
# difference between them. This is a deliberate task change, made because
# "solved" was being won by not driving: on the 2026-08-26 measurement every
# checkpoint reversed 38-45% of its steps and the BEST-scoring one covered the
# LEAST ground (1.08 m of wheel travel, vs 4.68 m for the earliest policy).
# terminated_fn fires only on line loss, so creeping was always safe and
# committing to speed always risked ending the episode. Weighting forward
# motion higher in the reward did not fix that, and an action-magnitude
# penalty collapsed the policy outright. Removing reverse from the action
# space makes shuffling unreachable instead of merely unattractive.
# Consequence to watch: the tightest reachable turn radius is now
# (wheel_sep/2)*(v_max+v_min)/(v_max-v_min) = ~0.12 m, so a sharp corner has
# to be taken as an arc rather than a pivot.
_LF_ALLOW_REVERSE = False
_LF_DARK = 60  # a pixel is "line" when max(R,G,B) < this


def _lf_band_centroid(img, y0, y1):
    """x-centroid (0..1) of dark pixels in rows [y0, y1) of `img`, or None."""
    img = np.asarray(img)
    band = img[y0:y1, :, :]
    mask = band.max(axis=2) < _LF_DARK
    if not mask.any():
        return None
    xs = np.nonzero(mask)[1]
    return float(xs.mean()) / (band.shape[1] - 1)


def _lf_line_centroid(img):
    """Return the x-centroid (0..1) of dark pixels in the lower half, or None.

    The near field: ground roughly under the rover's nose. This drives both
    the centering reward and termination.
    """
    return _lf_band_centroid(img, np.asarray(img).shape[0] // 2, None)


# Number of horizontal scan bands the frame is split into, nearest first.
# Sampling several rows at different distances rather than one aggregate blob
# is the standard camera line-follower feature: one row gives position only,
# several give position, heading AND curvature, which is what lets a tracker
# see a corner before it arrives.
_LF_SCAN_BANDS = 4
# Curvature (in normalized-offset units per band) treated as "as sharp as it
# gets" when regulating speed. Beyond this the speed cap is fully damped.
#
# MEASURED on this track, not chosen: driving the straight reads a median
# |curvature| of 0.091, and the 90 degree corner peaks at 0.551 (p90 across a
# whole episode is 0.419). 0.5 therefore puts the corner at essentially full
# damping while leaving straights almost untouched (straightness 0.82, cap
# 0.85). The initial guess of 1.0 left the cap at 0.56 through the corner,
# which is barely a gate at all -- and the policy duly charged the corner at
# full throttle and overshot to 0.27 m outside the track.
_LF_CURVATURE_FULL = 0.5


def _lf_track_features(img):
    """Classical line-tracker features from horizontal scan bands.

    Returns ``(offsets, cross_track, heading, curvature, ahead_visible)``.

    ``offsets`` holds one signed offset per band, nearest band first, each in
    [-1, 1] with 0 centred, and None where that band sees no line.

    * ``cross_track`` is the nearest band's offset: the classical lateral
      error, the same quantity the old single-centroid reward used.
    * ``heading`` approximates the line's angle in frame, as the difference
      between the furthest and nearest visible offsets. A pure lateral
      controller cannot distinguish "centred and parallel" from "centred and
      about to leave", which is exactly the case that was killing this spec;
      Stanley-style tracking adds this second term for that reason.
    * ``curvature`` is the second difference across bands: how much the line
      bends within the field of view. Used to regulate speed.
    * ``ahead_visible`` is whether the furthest band sees any line at all.
    """
    img = np.asarray(img)
    h = img.shape[0]
    step = h // _LF_SCAN_BANDS
    offsets = []
    for b in range(_LF_SCAN_BANDS):  # nearest band first: bottom of the frame
        y1 = h - b * step
        c = _lf_band_centroid(img, y1 - step, y1)
        offsets.append(None if c is None else 2.0 * (c - 0.5))

    seen = [(i, o) for i, o in enumerate(offsets) if o is not None]
    cross = offsets[0]
    heading = 0.0
    curvature = 0.0
    if len(seen) >= 2:
        heading = seen[-1][1] - seen[0][1]
    if len(seen) >= 3:
        # Second difference over the three furthest-apart visible bands.
        (_, a), (_, b_), (_, c_) = seen[0], seen[len(seen) // 2], seen[-1]
        curvature = c_ - 2.0 * b_ + a
    return offsets, cross, heading, curvature, offsets[-1] is not None


def _lf_lookahead_centroid(img):
    """Same, for the UPPER half: the track the rover is about to reach.

    Measured 2026-08-27, driving a straight into a 90 degree corner: the lower
    half's centroid sits at 0.41 to 0.52 -- dead centre -- for the entire
    approach, right up to the step the episode terminates. It carries no
    warning at all, because the line does not exit sideways, it recedes and
    shrinks straight ahead. The upper half reads 0.874 a full 20 steps (one
    second) earlier and decays toward 0.5 as the corner leaves the field of
    view, so the signal is strongest earliest.

    Note that the FULL-frame centroid is useless for this: 0.41 and 0.87
    average to 0.50 and cancel exactly. The two bands have to stay separate.
    """
    return _lf_band_centroid(img, 0, np.asarray(img).shape[0] // 2)


# Reward. Shape taken from classical line following and from published
# vision-based driving RL, not invented here; see
# docs/examples/line_follower.md, "What Classical Line Followers Do".
#
# The reward is MULTIPLICATIVE in speed:
#
#     r = (v / v_max) * (w_cross * quality_cross + w_heading * quality_heading)
#
# Every term is gated on actually moving, so standing still scores ~0 no
# matter how beautifully the line is centred. That structure is deliberate and
# was arrived at the hard way. An earlier version of this reward added its
# terms instead of multiplying, which handed out 1.75 per step for merely
# keeping the line in view: creeping at v_min for the full 600-step episode
# paid ~1050 while driving hard and losing the line at the first corner paid
# ~165, so the optimal policy was to barely move. Measured, not hypothesised
# -- that reward trained to a median episode of 257 steps covering 1.28 m,
# which is 0.10 m/s, exactly the floor. It is the same alive-bonus local
# optimum ROADMAP.md records for Ant, and adding survival terms to a reward
# whose episode ends on failure will reproduce it every time.
#
# Kendall et al., "Learning to Drive in a Day" (arXiv:1807.00412), use
# forward speed alone as the reward with termination on infraction, for
# exactly this reason. The two quality terms here only shape HOW the distance
# is earned; they can never pay for standing still.
#
#   cross    the nearest scan band's lateral error. The original reward was
#            this alone (plus an unconditional speed term), and it is provably
#            blind to a corner: measured driving a straight into a 90 degree
#            turn, it reads 0.41-0.52, dead centre, for the whole approach and
#            right up to the terminating step.
#   heading  the line's angle across the scan bands -- the second term a
#            Stanley controller adds to a pure lateral controller, and what
#            separates "centred and parallel" from "centred and about to
#            leave". On that same run the far band reads 0.874 a full second
#            before the near band notices anything.
#
# Speed reward is CAPPED by how straight the track ahead is -- Regulated Pure
# Pursuit's curvature heuristic, expressed as a reward.
#
# This gate was removed on 2026-08-28 on the argument that termination already
# supplies the pressure: entering a corner too fast ends the episode, which a
# ~100-step discount horizon can see. Measured, that argument is wrong. The
# policy enters the corner at [+1.00, +1.00], full speed, and only begins
# steering 4 steps before it dies, by which point it is committed to a 0.39 m
# turn radius against a corner needing 0.12 m. It overshoots to 0.27 m outside
# the track. It corners -- it just refuses to slow down for one, because every
# step it slows costs immediate reward and the payoff is 10+ steps away.
#
# The cap removes the INCENTIVE to charge without paying for standing still:
#
#     cap = v_min_frac + (1 - v_min_frac) * straightness
#     r   = quality * min(speed_frac, cap)
#
# On a straight (straightness 1) the cap is 1.0 and nothing changes. In a
# sharp corner (straightness 0) the cap falls to v_min_frac, so full throttle
# earns exactly what crawling earns and the rover may as well slow down.
# Creeping stays punished everywhere, because `min` still takes speed_frac
# when the rover is slower than the cap.
_LF_SPEED_FRAC_MIN = _LF_WHEEL_SPEED_MIN / _LF_WHEEL_SPEED
# Auxiliary observation layout, all in [-1, 1], nearest band first:
#   [0:4]  per-band lateral offset (0.0 where that band sees no line)
#   [4:8]  per-band visibility flag (1.0 seen, -1.0 not)
#   [8]    heading: line angle across the bands
#   [9]    curvature: how much the line bends within the field of view
#
# Every one of these is computable on the deployment laptop from the same
# camera frame, so none of it is privileged simulator state. Handing the CNN
# the quantities a classical tracker measures directly, rather than making it
# rediscover them from 64x64 pixels, is what every classical implementation
# does; see docs/examples/line_follower.md.
_LF_AUX_DIM = 10


def _lf_aux_obs(img):
    """Scan-band feature vector for the policy, from the frame IT sees."""
    offsets, _cross, heading, curvature, _ahead = _lf_track_features(img)
    out = np.zeros(_LF_AUX_DIM, dtype=np.float32)
    for i, o in enumerate(offsets[:_LF_SCAN_BANDS]):
        out[i] = 0.0 if o is None else np.clip(o, -1.0, 1.0)
        out[_LF_SCAN_BANDS + i] = -1.0 if o is None else 1.0
    out[2 * _LF_SCAN_BANDS] = np.clip(heading, -1.0, 1.0)
    out[2 * _LF_SCAN_BANDS + 1] = np.clip(curvature, -1.0, 1.0)
    return out


_LF_W_CROSS = 0.6
_LF_W_HEADING = 0.4


def _lf_reward(obs, action):
    _offsets, cross, heading, _curvature, ahead = _lf_track_features(obs)
    if cross is None:
        return 0.0
    a = np.clip(np.asarray(action, dtype=float).ravel(), -1.0, 1.0)

    # The heading term pays only when there IS track in the furthest band to
    # be aligned with. Without this it silently defaults to "perfectly
    # aligned" the moment the lookahead empties, which is precisely the corner
    # approach -- a frame with the line in the near bands and nothing beyond
    # them scored 0.99 out of 1.0, identical to an unobstructed straight. An
    # empty lookahead is the strongest corner cue available (the dark pixel
    # count collapses 961 -> 9 over the 16 steps before termination), so it
    # has to cost something.
    q_cross = 1.0 - abs(cross)
    q_heading = (1.0 - min(1.0, abs(heading))) if ahead else 0.0
    quality = _LF_W_CROSS * q_cross + _LF_W_HEADING * q_heading
    # Fraction of full scale the wheels are actually turning at. Uses the
    # spec's own action mapping rather than assuming one, so it stays correct
    # if the mapping changes: mean wheel speed over v_max, in [v_min/v_max, 1].
    speeds = [v for _j, _m, v in _lf_action_to_commands(a)]
    speed_frac = max(0.0, float(np.mean(speeds)) / _LF_WHEEL_SPEED)
    straightness = 1.0 - min(1.0, abs(_curvature) / _LF_CURVATURE_FULL)
    cap = _LF_SPEED_FRAC_MIN + (1.0 - _LF_SPEED_FRAC_MIN) * straightness
    return float(min(speed_frac, cap) * quality)


_LF_WHEELS = ("left_axle", "right_axle")
# Per-wheel forward-only, NOT the looser mean_forward_velocities.
#
# The mean-constrained map was tried on 2026-08-27 to buy a tighter turn
# radius (0.02 m against this map's 0.12 m) and was reverted the same day, for
# two measured reasons. Turn radius was not the binding constraint: distance
# before failure stayed at ~1.4 m under BOTH maps, across every speed and gain
# a hand-written controller could produce. And the pivot it unlocked is
# unphysical -- one wheel at +0.51 m/s against the other at -0.31 m/s skids
# and flings the chassis, which showed up as instantaneous ground speeds of
# 1.5 to 2.95 m/s against a 1.0 m/s caster-pop limit, on wheels that can only
# deliver 0.51. A 400k sweep scored worse under it (best 102.5 vs 113.4).
# mean_forward_velocities is kept and tested for when the policy can see a
# corner coming and a tight turn starts being worth its instability.
_lf_action_to_commands = clamped_velocities(
    _LF_WHEELS,
    _LF_WHEEL_SPEED,
    v_min=_LF_WHEEL_SPEED_MIN,
    reverse=_LF_ALLOW_REVERSE,
)


def _lf_reset_joint_state(rng):
    return {"left_axle": (0.0, 0.0), "right_axle": (0.0, 0.0)}


def _line_follower_spec() -> AgentSpec:
    return AgentSpec(
        name="line_follower",
        model_uri="package://gazebo_gymnasium_resources/models/rover_bare",
        bare_model_uri=("package://gazebo_gymnasium_resources/models/rover_bare"),
        observation_space=spaces.Box(low=0, high=255, shape=_LF_IMAGE, dtype=np.uint8),
        action_space=spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32),
        joint_obs=(),
        image_obs=_LF_IMAGE,
        per_agent_include_uri=("package://gazebo_gymnasium_resources/models/line_track"),
        reward_fn=_lf_reward,
        terminated_fn=lambda obs: _lf_line_centroid(obs) is None,
        # the old known-good start pose: on the y=-1 straight, aligned with it
        spawn_y=-1.0,
        spawn_yaw=1.5708,
        spawn_z=0.085,
        x_spacing=6.0,  # each agent gets its own ~2.4 m track loop
        frame_skip=5,
        # 600 steps = 30 s at the 20 Hz control rate. Raised from 300 on
        # 2026-08-27 because surviving the cap WAS the success criterion, and
        # at 300 steps a rover creeping at the v_min floor covered 1.5 m of a
        # ~10 m loop and still "solved" -- the cap arrived before the first
        # corner did. At 600 steps a full loop needs a 0.33 m/s average, which
        # creeping cannot reach, so lasting the episode now costs real ground.
        # Read alongside the distance bar in dr_eval.py's --solved-distance:
        # the cap makes creeping insufficient, the distance bar makes driving
        # necessary.
        max_episode_steps=600,
        reset_model_pose=True,  # mobile base: restore chassis pose on reset
        aux_obs_fn=_lf_aux_obs,
        aux_obs_dim=_LF_AUX_DIM,
        action_to_commands=_lf_action_to_commands,
        command_limits={j: _LF_WHEEL_SPEED_POP for j in _LF_WHEELS},
        reset_joint_state=_lf_reset_joint_state,
        # Sim-to-real DR (see docs/examples/line_follower.md). No
        # mass_randomization: this spec is velocity-actuated, and mass is
        # "largely masked" under pure velocity control per the field's own
        # docstring -- action_gain_randomization is the dynamics knob that
        # actually bites here (commanded vs. actual wheel speed on real
        # hardware). visual_randomization covers the camera-side gap: real
        # frames carry JPEG compression + auto-exposure drift this synthetic
        # renderer doesn't produce on its own.
        # Base (no-DR) task SOLVED 2026-08-07 (300/300, seed=0, 400k, see
        # docs/examples/line_follower.md "What good looks like"). DR
        # strengths tuned down from the first re-enable (0.1/0.15 -> 73%
        # solved, 8-agent eval) based on per-agent failure analysis: visual
        # DR's damage was concentrated at the TOP of its range (7/8 agents
        # were perfect, one harsh draw failed consistently) -- so the ceiling
        # is capped lower here. gain DR's damage was spread with no clean
        # per-agent cutoff -- so its overall strength is just reduced, not
        # capped. Re-validated: 40/40 episodes solved, all 8 agents, zero
        # failures -- matches the no-DR baseline's reliability exactly.
        action_gain_randomization=0.05,
        visual_randomization=0.08,
        # Three more mechanisms individually tuned 2026-08-22 via the same
        # isolated 8-agent-eval methodology, each resumed+fine-tuned from
        # the solved 45-degree-camera checkpoint rather than trained from
        # scratch (a from-scratch 100k screen never even solves the base
        # task, so its "failures" measure nothing about DR tolerance -- a
        # real methodology bug found and fixed mid-session). Individually,
        # each reaches a solved-or-near-solved result at real strengths:
        # action_noise=0.03 (40/40 after ~180k fine-tune steps),
        # battery_discharge=0.1 (40/40 at 60k), track_color=0.1 (38/40).
        # COMBINING all three at those individually-tuned values did not
        # reach the same reliability -- at every budget from 200k to 400k
        # fine-tune steps it either failed outright or plateaued around
        # 77-80%, stacking three new noise sources being genuinely harder
        # than any one alone. Repeatedly halving all three together (2026-
        # 08-23) traced a real, mostly-monotonic curve as combined strength
        # dropped -- half: ~80%, quarter: 85%, eighth: 92.5% -- before
        # collapsing to 0/40 at sixteenth-strength with the same tight,
        # near-deterministic failure signature seen elsewhere in this
        # tuning (almost certainly fine-tune instability/bad luck in that
        # one run, not a real strength effect, given it reverses a clean
        # trend). Shipped at eighth-strength, the best reliable point found
        # (37/40, 92.5%, the 3 remaining failures all at reset rather than
        # the harder mid-track corner) -- real progress, not a clean 100%
        # solve, so this is not claimed as fully solved. See
        # docs/domain_randomization.md for the full picture.
        action_noise_randomization=0.00375,
        battery_discharge_randomization=0.0125,
        track_color_randomization=0.0125,
        # Every episode must START on visible track. With
        # track_shape_reset_randomization, a drawn spawn pose sits on a track
        # segment but the camera can still see no line -- at a sharp corner
        # the line curves out of frame. Measured 2026-08-26: ~1% of resets
        # spawn blank, and every one died at step 1, which was essentially
        # the entire residual failure rate of a 97.5% policy. Those episodes
        # are unwinnable rather than hard, so they teach nothing during
        # training and measure nothing during eval. Redrawing is the only
        # place this can be checked: validity depends on what the camera
        # renders, which is not knowable at world-build time.
        spawn_redraw_attempts=5,
    )


def _line_follower_pivot_spec() -> AgentSpec:
    """`line_follower` with the mean-forward action map instead of per-wheel.

    Registered as an A/B variant rather than swapped in, so the two action
    spaces can be trained and evaluated against each other without editing
    the shipped spec out from under already-recorded results.

    The difference is what a corner can be taken with. The shipped map holds
    EACH wheel above v_min, capping the turn radius at ~0.12 m. This one
    constrains only the MEAN of the two, so one wheel may reverse and the
    reachable radius drops to ~0.02 m, while creeping stays unreachable
    because creeping needs a low mean by definition.

    Worth re-testing specifically because the reason it was dropped on
    2026-08-27 turned out to be wrong. Its pivots looked unphysical (chassis
    speeds of 1.5-2.95 m/s on wheels that deliver 0.50), but that was the
    wheel COLLISIONS being spheres: a point contact with no resistance to yaw
    about it. Cylinders fixed that. RoboCup Junior practice for 90 degree
    corners is explicitly to reverse the inner wheel, which the shipped map
    forbids outright.
    """
    return replace(
        _line_follower_spec(),
        name="line_follower_pivot",
        action_to_commands=mean_forward_velocities(
            _LF_WHEELS, _LF_WHEEL_SPEED, v_min=_LF_WHEEL_SPEED_MIN
        ),
    )


_SPEC_FACTORIES = {
    "cartpole": _cartpole_spec,
    "cartpole_continuous": _cartpole_continuous_spec,
    "inverted_double_pendulum": _inverted_double_pendulum_spec,
    "hopper": _hopper_spec,
    "walker2d": _walker2d_spec,
    "half_cheetah": _half_cheetah_spec,
    "reacher": _reacher_spec,
    "ant": _ant_spec,
    "line_follower": _line_follower_spec,
    "line_follower_pivot": _line_follower_pivot_spec,
}


def get_spec(name: str) -> AgentSpec:
    """Return a fresh AgentSpec for a registered agent name."""
    if name not in _SPEC_FACTORIES:
        raise KeyError(f"unknown agent spec {name!r}; registered: {sorted(_SPEC_FACTORIES)}")
    return _SPEC_FACTORIES[name]()


def register_spec(name: str, factory: Callable[[], AgentSpec]) -> None:
    """Register a new named agent spec (factory returning a fresh AgentSpec)."""
    _SPEC_FACTORIES[name] = factory


def registered_specs() -> list:
    return sorted(_SPEC_FACTORIES)


# Grid layout for track_shape_reset_randomization spawns every shape near
# every agent, so agents need far more room than the shipped single-track
# x_spacing leaves them. Measured against the 7 built-in presets.
_RESET_RANDOMIZATION_X_SPACING = 14.0

TRACK_SHAPE_MODES = ("off", "population", "reset")


def register_track_randomized(agent: str, mode: str, x_spacing: float = None) -> str:
    """Register a track-shape-randomized variant of ``agent``; return its name.

    ``mode`` is one of TRACK_SHAPE_MODES: ``off`` returns ``agent`` unchanged;
    ``population`` draws one shape per agent at world-build time (cheap, but
    any one agent sees a single shape for its whole lifetime); ``reset`` draws
    a fresh shape and spawn point every reset (stronger generalization, and
    measurably slower -- it spawns every shape near every agent).

    Registering a derived spec (rather than mutating the shipped one) keeps
    the default line_follower spec untouched for every other caller.
    """
    if mode not in TRACK_SHAPE_MODES:
        raise ValueError(f"track-shape mode must be one of {TRACK_SHAPE_MODES}, got {mode!r}")
    if mode == "off":
        return agent

    from .line_track_shapes import PRESETS

    base = get_spec(agent)
    if base.image_obs is None:
        raise ValueError(
            f"track-shape randomization needs an image-obs spec; {agent!r} is not one"
        )
    name = f"{agent}_track_{mode}"
    kwargs = {"name": name, "track_shape_choices": tuple(PRESETS)}
    if mode == "reset":
        kwargs["track_shape_reset_randomization"] = True
        kwargs["x_spacing"] = x_spacing or _RESET_RANDOMIZATION_X_SPACING
    elif x_spacing:
        kwargs["x_spacing"] = x_spacing
    spec = replace(base, **kwargs)
    register_spec(name, lambda: spec)
    return name
