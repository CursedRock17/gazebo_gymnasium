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
"""`rover_line` -- a line follower designed backwards from the physical rover.

Every constant here is a measured property of the real machine or of the track
we intend to lay down. The task definition itself -- observation, action and
reward -- lives in `rover_line_contract.py`, which is shared verbatim with the
deployment runtime; this module adds only what the simulator needs.

**The policy sees extracted features, not pixels** (``policy_image=False``).
The camera still renders and is still the only sensor that sees the line, but
the perception step is a fixed, portable function instead of learned weights:
the same extractor runs on a real JPEG, so the sim-to-real gap is a threshold
on a grayscale image -- something measurable -- rather than CNN features of a
synthetic renderer. It also drops training from a CNN to an MLP, which is what
makes a lap-length sample budget reachable on CPU.

Three properties of the task are worth stating up front, because each replaced
an earlier choice that did not survive contact with the problem:

**The speed command can stop.** ``forward`` spans 0 to twice the cruise speed,
so the middle of the action range is cruise and ``[-1, 0]`` is a halt. An
earlier forward-only map put a floor under every command to stay above motor
stiction, which also made "slow down for this corner" unrepresentable.

**Steering may reverse a wheel.** ``steering`` is added to one wheel and
subtracted from the other with no floor, so the rover can pivot -- the tactic
RoboCup Junior names for a 90 degree corner, and one the previous map forbade
outright while offering no replacement.

**Jerk is priced, not filtered.** The reward charges for the change in
command, which is why this spec runs no ``action_lowpass``. Filtering the
command needs the filter state in the observation to stay Markov; charging for
it needs the previous action only inside the reward, and leaves the
observation alone.

Losing the line does not end the episode immediately: ``termination_grace_steps``
requires the failure to persist for half a second, because a corner can swing
the line out of frame for a frame or two and an episode that ends there teaches
nothing about recovering.
"""

from dataclasses import replace

from gymnasium import spaces
import numpy as np

from .agent_spec import AgentSpec
from .agent_spec import register_spec
from .rover_line_contract import CONTROL_HZ
from .rover_line_contract import CRUISE_RAD_S
from .rover_line_contract import DARK
from .rover_line_contract import FAILURE_PENALTY
from .rover_line_contract import IMAGE
from .rover_line_contract import line_visible
from .rover_line_contract import MAX_WHEEL_RAD_S
from .rover_line_contract import MIN_LINE_PIXELS
from .rover_line_contract import N_BANDS
from .rover_line_contract import OBS_LEN
from .rover_line_contract import observation_from_sensors
from .rover_line_contract import step_reward
from .rover_line_contract import WHEEL_RADIUS
from .rover_line_contract import WHEEL_SEP
from .rover_line_contract import wheel_targets

# Re-exported so callers and tests resolve these off this module; the contract
# module is where they are defined.
__all__ = [
    "CRUISE_RAD_S",
    "DARK",
    "IMAGE",
    "MAX_WHEEL_RAD_S",
    "MIN_LINE_PIXELS",
    "N_BANDS",
    "OBS_LEN",
    "WHEEL_RADIUS",
    "WHEEL_SEP",
    "features",
    "reward",
    "spec",
    "terminated",
]

# Camera, from rover_line_bare (see scripts/make_rover_line_model.py):
# 0.1388 m above ground, pitched 45 degrees down, FACING FORWARD, so flat
# ground is visible from 0.059 m to 0.328 m ahead.
CAMERA_LOOKAHEAD_M = 0.328

WHEELS = ("left_axle", "right_axle")

# 10 Hz control: frame_skip x the 10 ms physics step. Matches the firmware's
# DEFAULT_CMD_RATE_HZ exactly, so one policy step is one real command rather
# than one the hardware would never see as a distinct setpoint.
FRAME_SKIP = 10
# Lap is ~9.31 m (3x2 m outer footprint, 0.4 m corners). 1200 steps at 10 Hz
# is 120 s, so a lap needs a 0.078 m/s average. The cap is generous on
# purpose: with a stop-capable action and jerk priced in the reward, the
# episode should end on a real outcome rather than on the clock.
MAX_EPISODE_STEPS = 1200
LAP_METERS = 9.31

# Real encoder resolution (wmala2/rover-firmware, 500 RPM profile): 680 counts
# per wheel revolution over a 0.2199 m circumference, sampled once per 0.1 s
# control step. Counts are integers, so the speed the firmware can report comes
# in steps of one count per step -- about 3.2 mm/s. Quantization is not
# randomization: it is there on every real reading, at every strength, so it
# is applied unconditionally rather than behind a DR knob.

# Encoder calibration + reading noise. 0.03 is a starting point, not a tuned
# value: the firmware's own TRIM_LEFT/TRIM_RIGHT exist because the two wheels
# disagree by a few percent, so this is the right order of magnitude for a
# wheel-diameter/CPR calibration error. Tune it the way line_follower tuned
# its own strengths -- an isolated multi-agent eval -- before trusting it.
ENCODER_DR = 0.03

# Encoder latency, in control steps. The firmware's counts accumulate over the
# interval BEFORE the reply, so even a perfect link hands back a value centred
# half a step in the past; the UDP round trip and the camera's own capture
# instant add the rest. One full step is the ceiling here, so an agent's draw
# spans "current" to "one command stale".
ENCODER_LATENCY_DR = 1.0

# Encoder dropout probability per step. UDP has no retransmission and the ESP32
# is also serving camera frames, so replies do go missing. Up to 5% is roughly
# one loss every 2 s at 10 Hz -- frequent enough to train against, far short of
# the sustained silence the deployment client aborts on.
ENCODER_DROPOUT_DR = 0.05

# --- Motor-side DR. Sourced from line_follower's own tuning history, which
# ran each mechanism through an isolated eight-agent evaluation. ---

# Per-wheel actuator gain error. The single mechanism line_follower's docs call
# "the dynamics knob that actually bites here (commanded vs. actual wheel speed
# on real hardware)", tuned there to this value. The framework draws it
# INDEPENDENTLY per actuator, which is what matters: a shared gain changes how
# fast the rover goes, a left/right mismatch changes where it ends up.
ACTION_GAIN_DR = 0.05

# Per-step motor jitter. line_follower reached 40/40 with this at 0.03 in
# isolation; it only had to be cut when stacked with two other new mechanisms.
ACTION_NOISE_DR = 0.03

# Battery sag across an episode. line_follower tuned 0.1 in isolation, over
# 300-step episodes; halved here because a 600-step episode at 10 Hz is 60 s of
# wall time, and losing a tenth of actuator authority in one minute would be a
# flat battery rather than a sag.
BATTERY_DR = 0.05

# --- Camera geometry. The one camera-side surface that reaches a
# feature-based observation: move or tilt the lens and the scan bands sample
# different ground, so every offset, the heading and the curvature shift
# continuously -- unlike brightness or track colour, which this observation
# was measured not to notice at all (see docs/examples/rover_line.md). ---

# Mount tolerance, metres. The lens is bolted to a printed mount, so a few
# millimetres of placement error per axis is the realistic build variation.
# Matches the +/-3 mm the MuJoCo build of this rover randomizes over.
CAMERA_MOUNT_DR = 0.003

# Pitch and field-of-view tolerance, degrees. The MuJoCo build randomizes
# tilt over 42-48 degrees against a 45 degree nominal and FOV over 57-63
# against 60, i.e. +/-3 on each; the same figure is used here for both.
CAMERA_ANGLE_DR = 3.0


# The rover drives caster-first, TOWARD the camera's view. Positive joint
# velocities move the chassis along body +Y, but the lens is mounted past the
# trailing caster looking along body -Y, so an unnegated command drives the
# rover away from everything it can see -- it follows the line receding behind
# it, and a corner only becomes visible after it has been passed. Negating
# here (rather than re-aiming the camera) is what matches the hardware: the
# real lens is bolted to the front caster mount and looks where the rover is
# going. Verified by rendering: aiming the sensor at +Y instead fills the
# frame with the rover's own chassis.
_DRIVE_SIGN = -1.0


# The known-good spawn: on the y=-1 straight, aligned along it.
SPAWN_Y = -1.0
SPAWN_YAW = 1.5708


class _Track:
    """The centreline a rover's progress and deviation are measured against.

    Built once per spec from the same functions that lay the track down, so it
    is the track's own geometry rather than an approximation recovered from a
    mesh. Projection keeps a per-agent hint of where the rover was last, and
    searches only a window around it: a closed loop has two arms that come
    close at corners, and a global nearest-point search would jump between
    them and report a lap completed in one step.
    """

    def __init__(self, points):
        self.points = np.asarray(points, dtype=np.float64)
        closed = np.vstack([self.points, self.points[0]])
        seg = np.linalg.norm(np.diff(closed, axis=0), axis=1)
        self.cum = np.concatenate([[0.0], np.cumsum(seg)])
        self.length = float(self.cum[-1])

    def project(self, xy, hint, window=40):
        """Nearest point on the loop, searched near `hint`.

        :param xy: the rover's (x, y).
        :param hint: index of the segment it was on last step.
        :param window: segments either side of the hint to consider.
        :return: (arc length in metres, deviation in metres, new hint).
        """
        n = len(self.points)
        idx = (np.arange(hint - window, hint + window + 1) % n).astype(int)
        a = self.points[idx]
        b = self.points[(idx + 1) % n]
        ab = b - a
        denom = np.einsum("ij,ij->i", ab, ab)
        t = np.clip(
            np.einsum("ij,ij->i", np.asarray(xy) - a, ab) / np.where(denom > 0, denom, 1), 0, 1
        )
        proj = a + t[:, None] * ab
        d = np.linalg.norm(proj - np.asarray(xy), axis=1)
        k = int(np.argmin(d))
        seg_i = int(idx[k])
        return float(self.cum[seg_i] + t[k] * np.linalg.norm(ab[k])), float(d[k]), seg_i


def _track_for(shape="rectangle"):
    """Centreline of one preset shape, built once and cached.

    Under track-shape randomization the loop under the rover changes between
    episodes, so measuring progress and deviation against a single fixed
    centreline would silently score against the wrong track. The step state
    reports which shape each agent is on; this hands back the matching one.
    """
    track = _TRACKS.get(shape)
    if track is None:
        track = _TRACKS[shape] = _load_track(shape)
    return track


def _load_track(shape="rectangle"):
    """Centreline of one track shape, from its own generator.

    Oriented so arc length increases the way the rover drives. The extraction
    chains segments by proximity and has no notion of which way round the loop
    is "forward", so half the time it comes out reversed -- and a reversed
    centreline makes every metre of honest driving count as negative progress,
    which no lap could ever complete.
    """
    from .line_track_shapes import centerline
    from .line_track_shapes import PRESETS

    pts = centerline(PRESETS[shape][0]())
    if shape == "rectangle":
        # This spec's own spawn is fixed, so orient against its heading: the
        # rover travels along body -Y (see _DRIVE_SIGN).
        heading = np.array([np.cos(SPAWN_YAW - np.pi / 2), np.sin(SPAWN_YAW - np.pi / 2)])
        spawn_xy = np.array([0.0, SPAWN_Y])
        nearest = int(np.argmin(np.linalg.norm(pts - spawn_xy, axis=1)))
        tangent = pts[(nearest + 1) % len(pts)] - pts[nearest]
        if float(tangent @ heading) < 0:
            pts = pts[::-1]
    else:
        # A randomized spawn is tangent-aligned to whichever segment is drawn,
        # so there is no fixed heading to orient against. Anticlockwise is the
        # convention; the signed delta in _track_state means a rover going the
        # other way simply accrues negative progress rather than being
        # mismeasured.
        area = float(
            np.sum(pts[:, 0] * np.roll(pts[:, 1], -1) - np.roll(pts[:, 0], -1) * pts[:, 1])
        )
        if area < 0:
            pts = pts[::-1]
    return _Track(pts)


_TRACKS = {}

# The shape `rover_line` itself spawns; randomized specs override per agent.
_TRACK = _track_for("rectangle")
# Per-agent projection hints and lap progress. Keyed by the id of the pose
# array the vec env hands in, which is stable per agent within an episode;
# cleared whenever a pose jumps, which a reset does.
_PROGRESS = {}

# A lap counts as finished this close to the start arc length, matching the
# tolerance a physical finish line would be judged to.
FINISH_TOLERANCE_M = 0.03
# Beyond this the chassis is off the tape: the track is 0.08 m wide, so 0.06 m
# of centre-to-centre error already has the wheels off one edge.
MAX_DEVIATION_M = 0.06
# Paid once, when a lap is completed.
COMPLETION_BONUS = 10.0
# A pose that moves more than this in one step is a reset, not driving.
_TELEPORT_M = 0.5
# Below this the chassis is on its side rather than on its wheels.
TIPPED_HEIGHT_M = 0.03


def _track_state(state):
    """Advance one agent's lap bookkeeping and return it.

    Idempotent within a step: reward_fn and terminated_fn both need this, and
    advancing twice would credit every step of progress twice over.

    :param state: the step state, carrying "agent", "step", "pose" and, when
        the scenery varies, "track".
    :return: (progress in metres, deviation in metres, z height).
    """
    pose = state["pose"]
    agent_key = state["agent"]
    # Which loop this agent is on, and where that copy sits. Every agent drives
    # a separate copy of the scenery laid out along x, and under shape
    # randomization the loop itself changes between episodes -- so world
    # coordinates alone would project onto a neighbour's track, or onto a
    # shape the rover is not driving.
    current = state.get("track")
    if current is None:
        track = _TRACK
        origin = np.asarray(state.get("origin", (0.0, 0.0)), dtype=np.float64)
    else:
        shape, origin_x, origin_y = current
        track = _track_for(shape)
        origin = np.array([origin_x, origin_y], dtype=np.float64)
    xy = np.asarray(pose[:2], dtype=np.float64) - origin

    prev = _PROGRESS.get(agent_key)
    if prev is not None and prev.get("step") == state["step"] and prev.get("track") is track:
        return prev["progress"], prev["dev"], float(pose[2])

    fresh = {"track": track, "step": state["step"], "xy": xy}
    if (
        prev is None
        or prev.get("track") is not track
        or np.linalg.norm(xy - prev["xy"]) > _TELEPORT_M
    ):
        # First step, moved onto a different loop, or teleported by a reset:
        # start the lap over rather than crediting the jump as progress.
        arc, dev, hint = track.project(xy, 0, window=len(track.points) // 2)
        # Which way round the loop this episode counts as forward. A randomized
        # spawn is tangent-aligned to whichever segment was drawn, so about
        # half of them face against the centreline's own ordering -- and
        # without this those episodes accrue negative progress and could never
        # complete a lap however well they drove.
        tangent = track.points[(hint + 1) % len(track.points)] - track.points[hint]
        heading = np.array([np.cos(pose[3] - np.pi / 2), np.sin(pose[3] - np.pi / 2)])
        fresh["sign"] = 1.0 if float(tangent @ heading) >= 0 else -1.0
        fresh.update({"arc": arc, "hint": hint, "progress": 0.0, "dev": dev})
        _PROGRESS[agent_key] = fresh
        return 0.0, dev, float(pose[2])

    arc, dev, hint = track.project(xy, prev["hint"])
    # Shortest signed step around the loop, so crossing the seam is not a lap.
    delta = (arc - prev["arc"] + track.length / 2) % track.length - track.length / 2
    delta *= prev.get("sign", 1.0)
    fresh["sign"] = prev.get("sign", 1.0)
    fresh.update({"arc": arc, "hint": hint, "progress": prev["progress"] + delta, "dev": dev})
    _PROGRESS[agent_key] = fresh
    return fresh["progress"], dev, float(pose[2])


def _sim_wheel_rad_s(sensors):
    """Simulated joint velocity -> the rad/s a real encoder would report.

    The whole simulator-side adaptation: undo the drive-sign negation, so that
    everything downstream speaks the units and signs the hardware uses.
    """
    return _DRIVE_SIGN * np.asarray(sensors, dtype=np.float32)


def features(img, sensors=None):
    """The policy observation, from SIMULATOR units.

    Thin wrapper over the contract's ``observation_from_sensors``: the only
    difference is the drive-sign convention on the joint velocities that
    ``AgentSpec.sensor_joints`` reads.

    :param img: the policy-visible camera frame, HxWx3 uint8.
    :param sensors: per-wheel joint velocity in rad/s, or None for stopped.
    :return: float32 vector of length OBS_LEN.
    """
    return observation_from_sensors(img, None if sensors is None else _sim_wheel_rad_s(sensors))


def action_to_commands(action):
    """Action -> per-wheel joint velocity commands.

    The contract returns the rad/s the firmware's ``'m'`` command takes;
    ``_DRIVE_SIGN`` then converts to the simulator's frame.
    """
    left, right = wheel_targets(action)
    return [
        ("left_axle", "velocity", _DRIVE_SIGN * left),
        ("right_axle", "velocity", _DRIVE_SIGN * right),
    ]


def outcome(obs, state):
    """Why this episode is over, or "running".

    Geometric and temporal, deliberately independent of the shaped reward: a
    high return does not establish success, and these are what the evaluator
    checks.

    :param obs: the raw camera frame.
    :param state: the step state, or None when the simulator supplies none, in
        which case only the camera-visible outcome can be judged.
    :return: one of "running", "tipped", "off_track", "line_lost", "completed".
    """
    if state is not None:
        progress, deviation, height = _track_state(state)
        if height < TIPPED_HEIGHT_M:
            return "tipped"
        if deviation > MAX_DEVIATION_M:
            return "off_track"
        lap = _PROGRESS[state["agent"]]["track"].length
        if progress >= lap - FINISH_TOLERANCE_M:
            return "completed"
    if not line_visible(obs):
        return "line_lost"
    return "running"


def reward(obs, action, sensors=None, state=None):
    """Per-step reward, plus the one-off terminal payments.

    The shaped part lives in the contract so the deployment side can compute
    the identical number from real sensors. The terminal payments are added
    here because only the simulator knows the episode is over, and only it can
    see the track well enough to say a lap was finished.

    :param obs: the raw camera frame.
    :param action: the action just applied.
    :param sensors: per-wheel joint velocity in rad/s; None reads as stopped.
    :param state: the per-step state mapping (prev_action, pose, ...).
    :return: this step's reward.
    """
    rad_s = np.zeros(2, np.float32) if sensors is None else _sim_wheel_rad_s(sensors)
    prev = action if state is None else state["prev_action"]
    r = step_reward(obs, action, rad_s, prev)
    why = outcome(obs, state)
    if why == "completed":
        r += COMPLETION_BONUS
    elif why != "running":
        # Charged on the step the failure is detected. Line loss is debounced,
        # so this may be the cost of nearly failing rather than of failing --
        # which is the intent: drifting off the line should hurt before it ends
        # the episode.
        r += FAILURE_PENALTY
    return r


def terminated(obs, state=None):
    """True on any terminal outcome. Line loss is debounced by the framework."""
    return outcome(obs, state) != "running"


def _reset_joint_state(rng):
    """Both wheels start the episode stopped.

    A module-level function rather than a lambda, so two specs built from the
    same source compare equal on this field -- which is what lets the A/B test
    pin the difference between the arms to exactly the fields it means.
    """
    return {j: (0.0, 0.0) for j in WHEELS}


def spec() -> AgentSpec:
    return AgentSpec(
        name="rover_line",
        model_uri="package://gazebo_gymnasium_resources/models/rover_line_bare",
        bare_model_uri="package://gazebo_gymnasium_resources/models/rover_line_bare",
        observation_space=spaces.Box(low=0, high=255, shape=IMAGE, dtype=np.uint8),
        action_space=spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32),
        joint_obs=(),
        image_obs=IMAGE,
        aux_obs_fn=features,
        aux_obs_dim=OBS_LEN,
        policy_image=False,  # features ARE the observation; MlpPolicy
        per_agent_include_uri="package://gazebo_gymnasium_resources/models/line_track",
        reward_fn=reward,
        terminated_fn=terminated,
        action_to_commands=action_to_commands,
        # Nothing may drive a wheel past what the motors deliver, whatever the
        # action map asks for and whatever gain DR does to it afterwards.
        command_limits={j: MAX_WHEEL_RAD_S for j in WHEELS},
        reset_joint_state=_reset_joint_state,
        spawn_y=SPAWN_Y,
        spawn_yaw=SPAWN_YAW,
        spawn_z=0.085,
        x_spacing=6.0,
        # Jerk is charged in the reward instead of filtered out of the command,
        # so the reward needs the previous action and the observation does not.
        # The same state mapping carries the chassis pose, which is what lets
        # progress, deviation and lap completion be judged against the track
        # rather than against what the camera happens to see.
        action_lowpass=0.0,
        provide_step_state=True,
        # Half a second of continuous line loss, at the 10 Hz control rate.
        termination_grace_steps=int(0.5 * CONTROL_HZ),
        # The two wheel encoders, the rover's other real sensors: read into
        # both the observation and the reward.
        sensor_joints=WHEELS,
        encoder_noise_randomization=ENCODER_DR,
        encoder_latency_randomization=ENCODER_LATENCY_DR,
        encoder_dropout_randomization=ENCODER_DROPOUT_DR,
        action_gain_randomization=ACTION_GAIN_DR,
        action_noise_randomization=ACTION_NOISE_DR,
        battery_discharge_randomization=BATTERY_DR,
        camera_mount_randomization=CAMERA_MOUNT_DR,
        camera_angle_randomization=CAMERA_ANGLE_DR,
        frame_skip=FRAME_SKIP,
        max_episode_steps=MAX_EPISODE_STEPS,
        reset_model_pose=True,
    )


def baseline_spec() -> AgentSpec:
    """`rover_line` with the encoders blinded, as an A/B control.

    Registered rather than swapped in, following `line_follower_pivot`, so both
    arms can be trained against each other without editing the shipped spec out
    from under recorded results. With `sensor_joints` empty the encoder slots
    read a constant zero -- the width is unchanged, so this isolates the
    INFORMATION rather than the shape -- and the reward's progress term falls
    back to whatever a stopped wheel implies.
    """
    return replace(
        spec(),
        name="rover_line_baseline",
        sensor_joints=(),
        # Nothing to perturb without sensors; all three must go together.
        encoder_noise_randomization=0.0,
        encoder_latency_randomization=0.0,
        encoder_dropout_randomization=0.0,
    )


register_spec("rover_line", spec)
register_spec("rover_line_baseline", baseline_spec)
