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
"""`rover_line`: the task contract, checked without starting a simulator.

Everything here is a pure function of a frame, an action and a wheel reading,
which is the point of extracting features rather than learning them -- the
contract can be tested at unit speed and the same functions run on hardware.
"""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from gazebo_gymnasium_bridge.envs import get_spec
from gazebo_gymnasium_bridge.envs import rover_line as rl
from gazebo_gymnasium_bridge.envs import rover_line_contract as c


def frame_with_line(x_centre=32, width=4, top=24):
    """A blank frame with one vertical line, for exercising the extractor."""
    img = np.full(c.IMAGE, 255, np.uint8)
    img[top:, x_centre - width // 2 : x_centre + width // 2] = 0
    return img


# --- observation ------------------------------------------------------------


def test_observation_is_eleven_bounded_floats():
    obs = c.observation_from_sensors(frame_with_line(), np.zeros(2))
    assert obs.shape == (11,)
    assert obs.dtype == np.float32
    assert np.all(np.abs(obs) <= 1.0)
    assert get_spec("rover_line").policy_observation_space.shape == (c.OBS_LEN,)


def test_each_band_carries_x_y_and_visibility():
    """Three bands of three, nearest first, then the two wheel speeds."""
    # Bands are fractions of FULL frame height: near 0.85-1.0 is rows 40-47,
    # far 0.25-0.55 is rows 12-25. A line starting at row 40 therefore fills
    # the near band and reaches neither of the others.
    feats = c.line_features(frame_with_line(top=40))
    assert feats.shape == (3, 3)
    assert feats[0][2] == 1.0, "near band should see a line drawn under it"
    assert feats[-1][2] == 0.0, "far band should be empty"
    # An unseen band reports zeros, not a stale centroid.
    assert feats[-1][0] == 0.0 and feats[-1][1] == 0.0


def test_centroid_x_follows_the_line_and_is_centred_on_zero():
    left = c.line_features(frame_with_line(x_centre=10))[0][0]
    middle = c.line_features(frame_with_line(x_centre=32))[0][0]
    right = c.line_features(frame_with_line(x_centre=54))[0][0]
    assert left < 0 < right
    assert middle == pytest.approx(0.0, abs=0.05)


def test_centroid_y_says_how_far_ahead_the_line_is():
    """How far ahead the line is, which one axis alone cannot say.

    The vertical centroid separates an approaching corner from one already
    under the wheels; without it the bands carry position only.
    """
    near_band = c.line_features(frame_with_line(top=44))[0][1]
    full = c.line_features(frame_with_line(top=0))[0][1]
    assert near_band > full, "line lower in frame must read further down"


def test_isolated_dark_pixels_are_not_tape():
    """Stray dark pixels must read as "nothing", not as a confident centroid.

    On real frames the caster, a shadow or sensor noise can put a few dark
    pixels in a band; without a floor they define that band's centroid.
    """
    img = np.full(c.IMAGE, 255, np.uint8)
    img[46, 2 : 2 + (c.MIN_LINE_PIXELS - 1)] = 0
    assert c.line_features(img)[0][2] == 0.0, "stray pixels were read as track"
    assert not c.line_visible(img)
    img[46, 2 : 2 + c.MIN_LINE_PIXELS] = 0
    assert c.line_features(img)[0][2] == 1.0, "a detection at the floor must count"


def test_wheel_speeds_are_quantized_to_the_encoder():
    """680 counts per revolution read once per control step is a real limit."""
    step = c.ENCODER_QUANTUM_RAD_S / c.MAX_WHEEL_RAD_S
    for rad_s in (1.3, 2.71, 5.999):
        got = c.wheel_speeds(np.full(2, rad_s))
        assert np.allclose(got / step, np.round(got / step), atol=1e-4)
        assert np.allclose(got, rad_s / c.MAX_WHEEL_RAD_S, atol=step)


# --- action -----------------------------------------------------------------


def test_the_rover_can_be_commanded_to_stop():
    """A forward-only map cannot express "slow down for this corner"."""
    assert c.wheel_targets([-1.0, 0.0]) == (0.0, 0.0)


def test_mid_range_action_is_cruise_speed():
    assert c.wheel_targets([0.0, 0.0]) == (c.CRUISE_RAD_S, c.CRUISE_RAD_S)


def test_full_steering_reverses_the_inner_wheel():
    """A pivot must be reachable.

    RoboCup Junior practice for a 90 degree corner is to reverse the inner
    wheel, which the previous action map forbade outright.
    """
    left, right = c.wheel_targets([0.0, 1.0])
    assert left < 0 < right


@pytest.mark.parametrize("a0", [-1.0, -0.5, 0.0, 0.5, 1.0])
@pytest.mark.parametrize("a1", [-1.0, 0.0, 1.0])
def test_no_action_can_exceed_the_motor_ceiling(a0, a1):
    assert max(abs(v) for v in c.wheel_targets([a0, a1])) <= c.MAX_WHEEL_RAD_S


def test_action_map_survives_the_joint_discovery_probe():
    """A scalar probe must not raise.

    HarnessCore probes the command map with the SCALARS 0 and 1 and swallows
    what it raises; an IndexError there leaves the rover unactuated while
    every command still computes.
    """
    for probe in (0, 1):
        assert len(rl.action_to_commands(probe)) == 2


def test_spec_reports_both_wheels_as_actuated():
    assert set(get_spec("rover_line").actuated_joints) == set(rl.WHEELS)


def test_both_wheels_are_driven_toward_the_camera_view():
    cmds = rl.action_to_commands([1.0, 0.0])
    assert all(np.sign(v) == np.sign(rl._DRIVE_SIGN) for _j, _m, v in cmds)


# --- reward -----------------------------------------------------------------


def test_reward_pays_for_measured_speed_not_commanded_speed():
    """The whole reason the encoders are in the loop.

    A stalled rover at full throttle must not score as though it were driving.
    """
    img = frame_with_line()
    full = np.array([1.0, 0.0], dtype=np.float32)
    stalled = c.step_reward(img, full, np.zeros(2), full)
    moving = c.step_reward(img, full, np.full(2, c.CRUISE_RAD_S), full)
    assert moving > stalled


def test_losing_the_line_costs_more_than_standing_still_on_it():
    """Losing the line must cost more than idling on it.

    Zero is what standing still on a visible line pays, so if losing it also
    paid zero the two outcomes would be worth the same.
    """
    stopped_on_line = c.step_reward(frame_with_line(), [-1, 0], np.zeros(2), [-1, 0])
    blank = np.full(c.IMAGE, 255, np.uint8)
    lost = c.step_reward(blank, [-1, 0], np.zeros(2), [-1, 0])
    assert lost < stopped_on_line


def test_centred_line_outscores_offset_line():
    a, s = np.array([0.0, 0.0]), np.full(2, c.CRUISE_RAD_S)
    centred = c.step_reward(frame_with_line(x_centre=32), a, s, a)
    offset = c.step_reward(frame_with_line(x_centre=10), a, s, a)
    assert centred > offset


def test_jerk_is_charged():
    """Jerk is priced, not filtered.

    Charging for the change in command is why this spec runs no
    action_lowpass.
    """
    img, s = frame_with_line(), np.full(2, c.CRUISE_RAD_S)
    steady = c.step_reward(img, [0.0, 0.0], s, [0.0, 0.0])
    jerked = c.step_reward(img, [0.0, 1.0], s, [0.0, -1.0])
    assert jerked < steady
    assert steady - jerked == pytest.approx(c.SMOOTHNESS_COST * 4.0, abs=1e-6)


def test_every_step_costs_something():
    """Dawdling on a perfectly centred line is never free."""
    img = frame_with_line()
    r = c.step_reward(img, [0.0, 0.0], np.zeros(2), [0.0, 0.0])
    assert r == pytest.approx(-c.STEP_COST, abs=1e-6)


def test_failure_is_penalised_once_by_the_simulator_wrapper():
    """The terminal penalty belongs to the simulator side.

    The shaped part is shared with hardware, but only the simulator knows the
    episode ended.
    """
    blank = np.full(c.IMAGE, 255, np.uint8)
    shaped = c.step_reward(blank, [0, 0], np.zeros(2), [0, 0])
    # No step state: the camera outcome alone still charges the failure.
    full = rl.reward(blank, [0, 0], np.zeros(2), None)
    assert full == pytest.approx(shaped + c.FAILURE_PENALTY, abs=1e-6)


def test_progress_reward_is_capped_at_cruise():
    """Speed past cruise earns nothing more.

    Centring is then the way to earn the rest of the reward, rather than raw
    speed.
    """
    img, a = frame_with_line(), np.array([0.0, 0.0])
    at_cruise = c.step_reward(img, a, np.full(2, c.CRUISE_RAD_S), a)
    flat_out = c.step_reward(img, a, np.full(2, c.MAX_WHEEL_RAD_S), a)
    assert flat_out == pytest.approx(at_cruise, abs=1e-6)


# --- termination ------------------------------------------------------------


def test_terminates_only_when_the_near_band_is_empty():
    assert not rl.terminated(frame_with_line())
    assert rl.terminated(np.full(c.IMAGE, 255, np.uint8))


def test_line_loss_is_debounced_for_half_a_second():
    """Brief line loss must not end the episode.

    A corner can swing the line out of frame for a frame or two, and an
    episode that ends there teaches nothing about recovering.
    """
    s = get_spec("rover_line")
    assert s.termination_grace_steps == int(0.5 * c.CONTROL_HZ)


# --- track geometry ---------------------------------------------------------


def _state(xy, z=0.085, agent=0, step=0, origin=(0.0, 0.0)):
    return {
        "agent": agent,
        "step": step,
        "prev_action": np.zeros(2, np.float32),
        "pose": np.array([xy[0], xy[1], z, 0.0], dtype=np.float32),
        "origin": np.asarray(origin, dtype=np.float32),
    }


def test_centreline_matches_the_documented_lap_length():
    """The centreline comes from the track's own generator, not a mesh.

    Every builder lays the track as boxes end to end along the path, so the
    geometry is analytic; recovering it by rasterizing would be both lossy and
    unnecessary.
    """
    assert rl._TRACK.length == pytest.approx(9.31, abs=0.02)
    assert len(rl._TRACK.points) > 100


def test_every_centreline_point_lies_on_the_tape():
    from gazebo_gymnasium_bridge.envs.line_track_shapes import centerline
    from gazebo_gymnasium_bridge.envs.line_track_shapes import PRESETS
    from gazebo_gymnasium_bridge.envs.line_track_shapes import TRACK_WIDTH

    for name, (builder, _model) in PRESETS.items():
        segs = builder()
        pts = centerline(segs)
        worst = 0.0
        for pt in pts:
            best = np.inf
            for (px, py, _z, _r, _p, yaw), (length, width, _h) in segs:
                d = np.asarray(pt) - np.array([px, py])
                cos, sin = np.cos(-yaw), np.sin(-yaw)
                local = np.abs([cos * d[0] - sin * d[1], sin * d[0] + cos * d[1]])
                over = np.maximum(local - np.array([length / 2, width / 2]), 0.0)
                best = min(best, float(np.linalg.norm(over)))
            worst = max(worst, best)
        assert worst < 1e-6, f"{name}: centreline strays {worst * 1000:.2f} mm off the tape"
        assert TRACK_WIDTH > 0


def test_progress_increases_in_the_direction_the_rover_faces():
    """Arc length must grow the way the rover drives.

    A reversed centreline makes honest driving count as negative progress, and
    no lap could ever complete.
    """
    rl._PROGRESS.clear()
    start = rl._TRACK.points[0]
    rl._track_state(_state(start, agent=50, step=0))
    ahead = rl._TRACK.points[5]
    progress, _dev, _z = rl._track_state(_state(ahead, agent=50, step=1))
    assert progress > 0


def test_progress_is_not_double_counted_within_a_step():
    """The bookkeeping is idempotent within a step.

    reward_fn and terminated_fn both need it, and advancing twice would credit
    every step of progress twice over.
    """
    rl._PROGRESS.clear()
    rl._track_state(_state(rl._TRACK.points[0], agent=51, step=0))
    once = rl._track_state(_state(rl._TRACK.points[5], agent=51, step=1))[0]
    twice = rl._track_state(_state(rl._TRACK.points[5], agent=51, step=1))[0]
    assert once == twice


def test_a_reset_restarts_the_lap_rather_than_crediting_the_jump():
    rl._PROGRESS.clear()
    rl._track_state(_state(rl._TRACK.points[0], agent=52, step=0))
    rl._track_state(_state(rl._TRACK.points[5], agent=52, step=1))
    far = rl._TRACK.points[len(rl._TRACK.points) // 2]
    progress, _d, _z = rl._track_state(_state(far, agent=52, step=2))
    assert progress == 0.0


def test_each_agent_is_measured_against_its_own_copy_of_the_track():
    """Each agent projects onto its own scenery.

    Agents drive separate copies laid out along x, so world coordinates alone
    would project onto a neighbour's track.
    """
    rl._PROGRESS.clear()
    on_line = rl._TRACK.points[10]
    _p, dev_here, _z = rl._track_state(_state(on_line, agent=53))
    shifted = on_line + np.array([6.0, 0.0])
    _p, dev_there, _z = rl._track_state(_state(shifted, agent=54, origin=(6.0, 0.0)))
    assert dev_here == pytest.approx(dev_there, abs=1e-6)


def test_every_preset_shape_has_a_usable_centreline():
    """Every shape needs its own centreline.

    Under shape randomization the loop changes between episodes, so one fixed
    centreline would score against the wrong track.
    """
    from gazebo_gymnasium_bridge.envs.line_track_shapes import PRESETS

    for shape in PRESETS:
        track = rl._track_for(shape)
        assert track.length > 1.0, shape
        assert len(track.points) > 50, shape


def test_progress_counts_forward_whichever_way_the_rover_faces():
    """Forward is decided per episode, not by the centreline's ordering.

    A randomized spawn is tangent-aligned to whichever segment was drawn, so
    about half face against that ordering; without a per-episode sign those
    episodes accrue negative progress and could never complete a lap however
    well they drove.
    """
    track = rl._track_for("rectangle")
    ahead = track.points[5] - track.points[0]
    forward_yaw = float(np.arctan2(ahead[1], ahead[0]) + np.pi / 2)

    def drive(agent, yaw, order):
        rl._PROGRESS.pop(agent, None)
        st = _state(track.points[order[0]], agent=agent, step=0)
        st["pose"][3] = yaw
        rl._track_state(st)
        st2 = _state(track.points[order[1]], agent=agent, step=1)
        st2["pose"][3] = yaw
        return rl._track_state(st2)[0]

    assert drive(80, forward_yaw, (0, 5)) > 0
    # Spawned facing the other way and driving that way is also progress.
    assert drive(81, forward_yaw + np.pi, (5, 0)) > 0


def test_each_agent_is_scored_on_the_shape_it_was_placed_on():
    """The drawn shape decides which loop scores the agent.

    The step state reports it; ignoring it would measure a rover driving a
    circle against a rectangle.
    """
    rl._PROGRESS.clear()
    circle = rl._track_for("circle")
    st = _state(circle.points[0], agent=90)
    st["track"] = ("circle", 0.0, 0.0)
    _p, dev, _z = rl._track_state(st)
    assert dev < 1e-6, "a point on the circle should read zero deviation"
    assert rl._PROGRESS[90]["track"] is circle


# --- outcomes ---------------------------------------------------------------


def test_outcomes_are_geometric_not_reward_based():
    """Outcomes are geometric, not reward-based.

    A high return does not establish success; these are what the evaluator
    checks.
    """
    img = frame_with_line(top=40)
    on_line = rl._TRACK.points[10]
    rl._PROGRESS.clear()
    assert rl.outcome(img, _state(on_line, agent=60)) == "running"
    rl._PROGRESS.clear()
    assert rl.outcome(img, _state(on_line, z=0.01, agent=61)) == "tipped"
    rl._PROGRESS.clear()
    rl.outcome(img, _state(on_line, agent=62, step=0))
    off = on_line + np.array([0.0, rl.MAX_DEVIATION_M * 3])
    assert rl.outcome(img, _state(off, agent=62, step=1)) == "off_track"
    rl._PROGRESS.clear()
    blank = np.full(c.IMAGE, 255, np.uint8)
    assert rl.outcome(blank, _state(on_line, agent=63)) == "line_lost"


def test_completing_a_lap_pays_a_bonus():
    rl._PROGRESS.clear()
    img = frame_with_line(top=40)
    key = 70
    rl._track_state(_state(rl._TRACK.points[0], agent=key, step=0))
    # Force the bookkeeping to a finished lap rather than driving one here.
    rl._PROGRESS[key]["progress"] = rl._TRACK.length
    rl._PROGRESS[key]["step"] = 1
    st = _state(rl._TRACK.points[0], agent=key, step=1)
    assert rl.outcome(img, st) == "completed"
    shaped = c.step_reward(img, [0, 0], np.zeros(2), [0, 0])
    assert rl.reward(img, [0, 0], np.zeros(2), st) == pytest.approx(
        shaped + rl.COMPLETION_BONUS, abs=1e-6
    )


def test_without_pose_only_the_camera_outcome_is_judged():
    """No pose still yields a usable answer.

    A caller without simulator state should get the camera-visible outcome,
    not a crash and not a silently wrong one.
    """
    assert rl.outcome(frame_with_line(top=40), None) == "running"
    assert rl.outcome(np.full(c.IMAGE, 255, np.uint8), None) == "line_lost"


# --- spec wiring ------------------------------------------------------------


def test_spec_matches_the_measured_rover():
    s = get_spec("rover_line")
    assert s.frame_skip == 10, "10 Hz control, matching the firmware command rate"
    assert s.max_episode_steps * s.frame_skip * 0.01 == 120.0
    assert s.sensor_joints == rl.WHEELS
    assert s.provide_step_state, "reward and termination need pose and a_{t-1}"
    assert s.action_lowpass == 0.0, "jerk is charged, not filtered"


def test_the_deployment_script_imports_the_contract_rather_than_copying_it():
    """The rover-side script must not carry its own copy of the extractor.

    The contract itself lives here and in the rover repo, nowhere in between:
    `results/for_rover_repo/` holds only the runtime, which imports it. That
    directory used to carry a duplicate, and two copies agreeing was a
    convention rather than a guarantee -- a divergent extractor fails on
    hardware in a way that still looks plausible.

    This guards the failure mode directly: re-inlining any of the contract's
    functions or thresholds into the runtime fails the test.
    """
    root = Path(__file__).resolve().parents[2]
    runtime = root / "results" / "for_rover_repo" / "rover_line_deploy.py"
    assert runtime.exists(), f"deployment runtime is missing: {runtime}"
    text = runtime.read_text()
    assert "from rover_line_contract import" in text, (
        "the runtime must import the contract, not re-implement it"
    )
    assert not (runtime.parent / "rover_line_contract.py").exists(), (
        "a second copy of the contract has reappeared; it belongs in the rover "
        "repo, copied from gazebo_gymnasium_bridge/envs/rover_line_contract.py"
    )
    for duplicated in ("def line_features", "def wheel_targets", "def band_centroid", "DARK ="):
        assert duplicated not in text, f"the runtime re-implements {duplicated!r}"


def test_contract_version_was_bumped_for_the_new_layout():
    """The version has to move with the layout.

    A runtime that knows only v1 must refuse a v2 policy rather than
    mis-slice eleven floats it expects to be fourteen.
    """
    assert c.SCHEMA_VERSION == c.MODEL_API == 2


# --- domain randomization ---------------------------------------------------


def test_all_dr_mechanisms_are_configured():
    s = get_spec("rover_line")
    for field, value in (
        ("encoder_noise_randomization", rl.ENCODER_DR),
        ("encoder_latency_randomization", rl.ENCODER_LATENCY_DR),
        ("encoder_dropout_randomization", rl.ENCODER_DROPOUT_DR),
        ("action_gain_randomization", rl.ACTION_GAIN_DR),
        ("action_noise_randomization", rl.ACTION_NOISE_DR),
        ("battery_discharge_randomization", rl.BATTERY_DR),
        ("camera_mount_randomization", rl.CAMERA_MOUNT_DR),
        ("camera_angle_randomization", rl.CAMERA_ANGLE_DR),
    ):
        assert getattr(s, field) == value > 0, field
    # Latency is capped at one control step: beyond that a reading would
    # describe a command older than the one being issued now.
    assert rl.ENCODER_LATENCY_DR <= 1.0


def test_camera_geometry_dr_moves_the_camera():
    """The jitter must reach the pose, not only the field of view.

    Regression guard: the first version matched "<pose>" while the model
    writes `<pose relative_to="base_link">`, so the camera never moved and
    nothing failed -- only the FOV, whose tag carries no attribute.
    """
    import re

    from gazebo_gymnasium_bridge.envs.inprocess_vec_env import _build_world

    s = get_spec("rover_line")
    world, _poses = _build_world(
        s, 4, 6.0, np.random.default_rng(0), "/rl/t", camera_rng=np.random.default_rng(1)
    )
    poses = [
        [float(v) for v in m.group(1).split()]
        for m in re.finditer(r"<link name=.camera_link.>\s*<pose[^>]*>([^<]*)<", world)
    ]
    assert len(poses) == 4
    pitch = np.degrees([p[4] for p in poses])
    assert np.ptp(pitch) > 0.5, f"camera pitch did not vary: {pitch}"
    assert np.all(np.abs(pitch - 45.0) <= s.camera_angle_randomization + 1e-6)
    xyz = np.array([p[:3] for p in poses])
    assert np.ptp(xyz, axis=0).max() > 0
    fov = np.degrees([float(m.group(1)) for m in re.finditer(r"<horizontal_fov>([^<]*)<", world)])
    assert np.ptp(fov) > 0.1


def test_dr_fields_require_what_they_perturb():
    for field in (
        "encoder_noise_randomization",
        "encoder_latency_randomization",
        "encoder_dropout_randomization",
    ):
        bare = dict.fromkeys(
            (
                "encoder_noise_randomization",
                "encoder_latency_randomization",
                "encoder_dropout_randomization",
            ),
            0.0,
        )
        bare["sensor_joints"] = ()
        with pytest.raises(ValueError, match=field):
            replace(get_spec("rover_line"), **{**bare, field: 0.05})
    for field in ("camera_mount_randomization", "camera_angle_randomization"):
        with pytest.raises(ValueError, match=field):
            replace(get_spec("cartpole"), **{field: 0.01})


def test_baseline_variant_isolates_the_encoders():
    """An A/B is only evidence if the arms differ in what is being tested."""
    from dataclasses import fields

    a, b = get_spec("rover_line"), get_spec("rover_line_baseline")
    differing = {f.name for f in fields(a) if repr(getattr(a, f.name)) != repr(getattr(b, f.name))}
    assert differing == {
        "name",
        "sensor_joints",
        # The encoder DR fields follow sensor_joints rather than being
        # independent knobs -- without encoders there is nothing to perturb.
        "encoder_noise_randomization",
        "encoder_latency_randomization",
        "encoder_dropout_randomization",
    }, differing
