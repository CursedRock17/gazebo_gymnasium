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
"""Line follower — the image-observation (vision-in-the-loop) environment.

Offline tests cover the image-processing spec math on synthetic frames; the
physics tests render the real onboard camera headlessly in the in-process sim.
"""

import functools
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from gazebo_gymnasium_bridge.envs import inprocess_vec_env as ip  # noqa: E402
from gazebo_gymnasium_bridge.envs.agent_spec import _lf_line_centroid  # noqa: E402
from gazebo_gymnasium_bridge.envs.agent_spec import get_spec  # noqa: E402


def _frame(dark_col=None):
    img = np.full((64, 64, 3), 220, dtype=np.uint8)
    if dark_col is not None:
        img[32:, dark_col - 2 : dark_col + 2, :] = 10  # line in lower half
    return img


class TestLineFollowerSpecMath:
    def test_spaces(self):
        spec = get_spec("line_follower")
        assert spec.image_obs == (64, 64, 3)
        assert spec.observation_space.dtype == np.uint8
        assert spec.action_space.shape == (2,)

    def test_no_mass_randomization(self):
        # Stable architectural call, not an active tuning knob like the two
        # DR strengths below (which change as docs/examples/line_follower.md
        # "Policies"/"Sim to Real" gets iterated on -- see test_visual_dr_*
        # for the DR *mechanism*, which must work regardless of the shipped
        # spec's current strengths).
        spec = get_spec("line_follower")
        assert spec.mass_randomization == 0.0, (
            "velocity-actuated: mass DR is largely masked here per its own "
            "docstring, so it's deliberately left off regardless"
        )

    def test_new_dr_mechanisms_tuned(self):
        # Shipped at eighth-strength of each mechanism's individually-tuned
        # value (real result: 37/40, 92.5%, not a clean solve) -- see
        # agent_spec.py's own comments for the full strength-vs-reliability
        # curve this came from.
        spec = get_spec("line_follower")
        assert spec.action_noise_randomization == 0.00375
        assert spec.battery_discharge_randomization == 0.0125
        assert spec.track_color_randomization == 0.0125

    def test_track_color_dr_lightens_dark_pixels_only(self):
        clean = _frame(32)
        dark_mask = clean.max(axis=2) < ip._TRACK_DARK_THRESH
        assert dark_mask.any()

        # lighten=0 is a no-op
        assert np.array_equal(ip._apply_track_color_dr(clean, 0.0, 220.0), clean)

        # lighten>0 pulls only the dark ("line") pixels toward target,
        # leaving the bright background untouched
        out = ip._apply_track_color_dr(clean, 0.6, 220.0)
        assert out[dark_mask].astype(int).mean() > clean[dark_mask].astype(int).mean()
        assert np.array_equal(out[~dark_mask], clean[~dark_mask])

    def test_visual_dr_brightness_and_noise(self):
        clean = _frame(32)
        rng = np.random.default_rng(0)

        # identity-ish: brightness=1, no noise, high quality -> close to clean
        out = ip._apply_visual_dr(clean, 1.0, 0.0, 95, rng)
        assert out.shape == clean.shape and out.dtype == np.uint8
        assert np.abs(out.astype(int) - clean.astype(int)).mean() < 5

        # brightness clearly shifts the mean
        bright = ip._apply_visual_dr(clean, 1.8, 0.0, 95, rng)
        assert bright.astype(int).mean() > clean.astype(int).mean()

        # noise makes two draws from the same rng differ
        rng2 = np.random.default_rng(1)
        a = ip._apply_visual_dr(clean, 1.0, 20.0, 95, rng2)
        b = ip._apply_visual_dr(clean, 1.0, 20.0, 95, rng2)
        assert not np.array_equal(a, b), "successive noisy draws must differ"

        # low jpeg_quality visibly degrades vs. high quality
        hi = ip._apply_visual_dr(clean, 1.0, 0.0, 95, rng)
        lo = ip._apply_visual_dr(clean, 1.0, 0.0, 20, rng)
        assert not np.array_equal(hi, lo)

    def test_centroid_detection(self):
        assert _lf_line_centroid(_frame(None)) is None
        c = _lf_line_centroid(_frame(32))
        assert 0.45 < c < 0.55
        assert _lf_line_centroid(_frame(8)) < 0.25
        # a line only in the UPPER half must not count (we track the ground
        # directly ahead, not the horizon)
        img = _frame(None)
        img[:20, 30:34, :] = 10
        assert _lf_line_centroid(img) is None

    def test_reward_prefers_centered_and_forward(self):
        spec = get_spec("line_follower")
        fwd = np.array([1.0, 1.0])
        stop = np.zeros(2)
        assert spec.reward_fn(_frame(32), fwd) > spec.reward_fn(_frame(8), fwd)
        assert spec.reward_fn(_frame(32), fwd) > spec.reward_fn(_frame(32), stop)
        assert spec.reward_fn(_frame(None), fwd) == 0.0

    def test_terminates_on_line_loss(self):
        spec = get_spec("line_follower")
        assert spec.terminated_fn(_frame(None)) is True
        assert spec.terminated_fn(_frame(32)) is False

    def test_action_maps_wheel_velocities(self):
        spec = get_spec("line_follower")
        cmds = spec.action_to_commands(np.array([1.0, -0.5]))
        assert [c[0] for c in cmds] == ["left_axle", "right_axle"]
        assert all(c[1] == "velocity" for c in cmds)
        assert cmds[1][2] == pytest.approx(-cmds[0][2] / 2)


# ---- real rendering + physics (in-process, headless) ---------------------- #

pytest.importorskip("gz.sim8", reason="gz.sim8 bindings not available")

_PROBE = """
from gazebo_gymnasium_bridge.envs import make_inprocess
env = make_inprocess("line_follower", n_agents=1, seed=0)
env.close()
"""

_VISUAL_DR_PROBE = """
from dataclasses import replace
from gazebo_gymnasium_bridge.envs import make_inprocess
from gazebo_gymnasium_bridge.envs.agent_spec import get_spec, register_spec

# Independent process (can't share lf_env's camera-env slot): construct a DR-
# ENABLED variant regardless of the shipped spec's current tuning state
# (docs/examples/line_follower.md "Policies" -- DR is off there right now,
# pending a base solve; the mechanism must still work when turned on).
spec = replace(get_spec("line_follower"), name="lf_vdr_probe",
               action_gain_randomization=0.2, visual_randomization=0.3)
register_spec(spec.name, lambda: spec)
env = make_inprocess(spec.name, n_agents=1, seed=0)
try:
    assert env._vdr_brightness.shape == (1,)
    assert 0.4 <= env._vdr_brightness[0] <= 1.6
    assert 20 <= env._vdr_jpeg_quality[0] <= 95
    print("VISUAL_DR_WIRED_OK")
finally:
    env.close()
"""

_NEW_DR_PROBE = """
from dataclasses import replace
import numpy as np
from gazebo_gymnasium_bridge.envs import make_inprocess
from gazebo_gymnasium_bridge.envs.agent_spec import get_spec, register_spec

# Independent process, same reasoning as _VISUAL_DR_PROBE above: a DR-ENABLED
# variant regardless of the shipped spec's current (off) tuning state.
spec = replace(get_spec("line_follower"), name="lf_newdr_probe",
               track_color_randomization=0.6, action_noise_randomization=0.3,
               battery_discharge_randomization=0.8)
register_spec(spec.name, lambda: spec)
env = make_inprocess(spec.name, n_agents=2, seed=0)
try:
    assert env._tcdr_lighten.shape == (2,) and (env._tcdr_lighten <= 0.6).all()
    assert (180.0 <= env._tcdr_target).all() and (env._tcdr_target <= 255.0).all()
    assert env._action_noise_std.shape == (2,) and (env._action_noise_std <= 0.3).all()
    assert env._battery_depletion.shape == (2,) and (env._battery_depletion <= 0.8).all()

    env.reset()
    fixed = np.tile([0.5, 0.5], (2, 1))
    env.step_async(fixed)
    if not np.allclose(env._ctl["action"], fixed):
        print("ACTION_NOISE_APPLIED")

    # battery: the SAME nominal action, late vs. early in the episode, must
    # come out smaller in magnitude once discharge has had time to bite.
    env._agent_steps[:] = env.max_episode_steps - 1
    env.step_async(np.tile([1.0, 1.0], (2, 1)))
    late = np.abs(env._ctl["action"]).mean()
    env._agent_steps[:] = 0
    env.step_async(np.tile([1.0, 1.0], (2, 1)))
    early = np.abs(env._ctl["action"]).mean()
    if late < early:
        print("BATTERY_DISCHARGE_APPLIED")

    env.step(np.tile([0.8, 0.8], (2, 1)))  # smoke-test a real step end to end
    print("NEW_DR_WIRED_OK")
finally:
    env.close()
"""

_TRACK_SHAPE_PROBE = """
from dataclasses import replace
from gazebo_gymnasium_bridge.envs import make_inprocess
from gazebo_gymnasium_bridge.envs.agent_spec import get_spec, register_spec
from gazebo_gymnasium_bridge.envs.agent_spec import _lf_line_centroid
from gazebo_gymnasium_bridge.envs.line_track_shapes import PRESETS

# Independent process, same reasoning as _VISUAL_DR_PROBE above: a
# track-shape-randomization-ENABLED variant regardless of the shipped
# spec's current (off) state -- see AgentSpec.track_shape_choices.
shapes = tuple(PRESETS.keys())
spec = replace(get_spec("line_follower"), name="lf_trackshape_probe",
               track_shape_choices=shapes)
register_spec(spec.name, lambda: spec)

# Oversample (3x the shape count) so a fixed seed's random draws land on
# more than one distinct shape -- this is what actually exercises the
# per-agent shape-choice + spawn-pose plumbing, not just the "off" default.
N = len(shapes) * 3
env = make_inprocess(spec.name, n_agents=N, seed=0)
try:
    obs = env.reset()
    seen = [_lf_line_centroid(obs[i]) for i in range(N)]
    print("centroids:", seen)
    if all(c is not None for c in seen):
        print("ALL_AGENTS_SEE_LINE_ON_RANDOM_SHAPE")
    print("TRACK_SHAPE_WIRED_OK")
finally:
    env.close()
"""

_TRACK_SHAPE_RESET_PROBE = """
from dataclasses import replace
import numpy as np
from gazebo_gymnasium_bridge.envs import make_inprocess
from gazebo_gymnasium_bridge.envs.agent_spec import get_spec, register_spec
from gazebo_gymnasium_bridge.envs.line_track_shapes import PRESETS

# Independent process: track_shape_reset_randomization=True spawns ALL
# choices near each agent (not just one), so a bigger x_spacing is needed
# to fit the grid -- see AgentSpec.track_shape_reset_randomization.
shapes = tuple(PRESETS.keys())
spec = replace(get_spec("line_follower"), name="lf_trackshape_reset_probe",
               track_shape_choices=shapes, track_shape_reset_randomization=True,
               x_spacing=14.0)
register_spec(spec.name, lambda: spec)

N, N_EPISODES = 4, 6
env = make_inprocess(spec.name, n_agents=N, seed=0)
try:
    n_no_line = 0
    frames = [[] for _ in range(N)]
    for _ in range(N_EPISODES):
        obs = env.reset()
        for i in range(N):
            frames[i].append(obs[i].copy())
            bottom = obs[i][obs[i].shape[0] // 2:, :, :]
            if not (bottom.max(axis=2) < 60).any():
                n_no_line += 1
    print(f"no_line_frames: {n_no_line}/{N * N_EPISODES}")
    if n_no_line == 0:
        print("ALL_RESETS_SEE_LINE")
    # Real per-reset diversity, not just "worked once": at least half the
    # resets should look visually distinct for at least one agent.
    distinct = sum(
        1 for a in range(N_EPISODES) for b in range(a + 1, N_EPISODES)
        if np.abs(frames[0][a].astype(int) - frames[0][b].astype(int)).mean() > 2.0
    )
    print(f"distinct_frame_pairs_agent0: {distinct}")
    if distinct > 0:
        print("RESET_RANDOMIZATION_VARIES")
    print("TRACK_SHAPE_RESET_WIRED_OK")
finally:
    env.close()
"""

_MULTI_AGENT_PROBE = """
import numpy as np
from gazebo_gymnasium_bridge.envs import make_inprocess
from gazebo_gymnasium_bridge.envs.agent_spec import _lf_line_centroid

N = 4
env = make_inprocess("line_follower", n_agents=N, seed=0)
try:
    obs = env.reset()
    seen = [_lf_line_centroid(obs[i]) for i in range(N)]
    print("centroids:", seen)
    if all(c is not None for c in seen):
        print("ALL_AGENTS_SEE_LINE")
    term = np.zeros(N, dtype=int)
    for _ in range(20):
        obs, _r, dones, _i = env.step(np.tile([0.8, 0.8], (N, 1)))
        term += dones.astype(int)
    print("terminations:", term.tolist())
    if term.sum() == 0:
        print("NO_SPURIOUS_TERMINATIONS")
finally:
    env.close()
"""


@functools.lru_cache(maxsize=1)
def _rendering_works():
    """Return whether a camera environment can be built on this machine.

    Run in a SUBPROCESS on purpose. On a host without a usable render device
    (GPU-less CI runners, minimal containers) gz-sim's rendering stack can die
    with a SIGSEGV inside native code rather than raising — and a native crash
    takes the whole pytest process with it, so no in-process ``try/except``
    can contain it. Probing in a child process keeps the suite alive and lets
    us skip honestly instead of reporting a false failure.
    """
    try:
        proc = subprocess.run([sys.executable, "-c", _PROBE], capture_output=True, timeout=180)
    except (subprocess.TimeoutExpired, OSError):
        return False
    return proc.returncode == 0


@pytest.fixture(scope="module")
def lf_env():
    from gazebo_gymnasium_bridge.envs import make_inprocess

    if not _rendering_works():  # pragma: no cover
        pytest.skip(
            "headless camera rendering unavailable on this machine "
            "(no usable render device); vision tests skipped"
        )
    env = make_inprocess("line_follower", n_agents=1, seed=0)
    yield env
    env.close()


def test_visual_dr_wired_on_real_env():
    # Construction smoke test (did adding visual DR break real camera-env
    # construction) for a DR-ENABLED variant, independent of whatever the
    # shipped line_follower spec's DR is currently tuned to (see
    # _VISUAL_DR_PROBE). test_visual_dr_brightness_and_noise above covers the
    # per-frame transform's correctness directly, offline.
    if not _rendering_works():  # pragma: no cover
        pytest.skip("headless camera rendering unavailable on this machine")
    # Own subprocess: can't share lf_env's one-camera-env-per-process slot.
    proc = subprocess.run(
        [sys.executable, "-c", _VISUAL_DR_PROBE], capture_output=True, text=True, timeout=60
    )
    assert proc.returncode == 0, (
        f"visual DR probe failed (rc={proc.returncode})\n"
        f"{proc.stdout[-600:]}\n{proc.stderr[-600:]}"
    )
    assert "VISUAL_DR_WIRED_OK" in proc.stdout


def test_new_dr_mechanisms_wired_on_real_env():
    # Construction + behavior smoke test for track_color/action_noise/
    # battery_discharge DR, mirroring test_visual_dr_wired_on_real_env.
    if not _rendering_works():  # pragma: no cover
        pytest.skip("headless camera rendering unavailable on this machine")
    proc = subprocess.run(
        [sys.executable, "-c", _NEW_DR_PROBE], capture_output=True, text=True, timeout=60
    )
    assert proc.returncode == 0, (
        f"new DR probe failed (rc={proc.returncode})\n{proc.stdout[-600:]}\n{proc.stderr[-600:]}"
    )
    assert "NEW_DR_WIRED_OK" in proc.stdout
    assert "ACTION_NOISE_APPLIED" in proc.stdout
    assert "BATTERY_DISCHARGE_APPLIED" in proc.stdout


def test_track_shape_choices_wired_on_real_env():
    # Construction + real-camera-frame smoke test for track_shape_choices,
    # mirroring test_new_dr_mechanisms_wired_on_real_env. Checks the two
    # real bugs found and fixed while building this: the +pi/2 camera-vs-
    # segment-tangent yaw offset, and the zigzag preset's corner-rounding
    # geometry (see line_track_shapes.py's preset_zigzag docstring).
    if not _rendering_works():  # pragma: no cover
        pytest.skip("headless camera rendering unavailable on this machine")
    proc = subprocess.run(
        [sys.executable, "-c", _TRACK_SHAPE_PROBE], capture_output=True, text=True, timeout=90
    )
    assert proc.returncode == 0, (
        f"track shape probe failed (rc={proc.returncode})\n"
        f"{proc.stdout[-800:]}\n{proc.stderr[-800:]}"
    )
    assert "TRACK_SHAPE_WIRED_OK" in proc.stdout
    assert "ALL_AGENTS_SEE_LINE_ON_RANDOM_SHAPE" in proc.stdout, proc.stdout[-800:]


def test_track_shape_reset_randomization_wired_on_real_env():
    # Construction + real-camera-frame smoke test for
    # track_shape_reset_randomization: every reset (not just world-build
    # time) must land on a real, visible line, AND the spawn point must
    # genuinely vary across resets, not just work once. Two real bugs were
    # found building this that a construction-only check wouldn't catch: a
    # dropped chassis-model append (F841 caught it at lint time) and the
    # same +pi/2 yaw / zigzag-geometry issues test_track_shape_choices_
    # wired_on_real_env already covers for the population-based mode.
    if not _rendering_works():  # pragma: no cover
        pytest.skip("headless camera rendering unavailable on this machine")
    proc = subprocess.run(
        [sys.executable, "-c", _TRACK_SHAPE_RESET_PROBE],
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert proc.returncode == 0, (
        f"track shape reset probe failed (rc={proc.returncode})\n"
        f"{proc.stdout[-800:]}\n{proc.stderr[-800:]}"
    )
    assert "TRACK_SHAPE_RESET_WIRED_OK" in proc.stdout
    assert "ALL_RESETS_SEE_LINE" in proc.stdout, proc.stdout[-800:]
    assert "RESET_RANDOMIZATION_VARIES" in proc.stdout, proc.stdout[-800:]


def test_camera_obs_shape_and_line_visible(lf_env):
    obs = lf_env.reset()
    assert obs.shape == (1, 64, 64, 3) and obs.dtype == np.uint8
    c = _lf_line_centroid(obs[0])
    assert c is not None and 0.3 < c < 0.7, f"line should start near-centered, centroid={c}"


def test_camera_is_live_while_driving(lf_env):
    obs0 = lf_env.reset().copy()
    obs = obs0
    for _ in range(15):
        obs, _r, _d, _i = lf_env.step(np.array([[0.8, 0.8]]))
    diff = float(np.abs(obs.astype(int) - obs0.astype(int)).mean())
    assert diff > 0.5, "camera frames must change as the rover drives"


def test_every_agent_gets_its_own_track():
    """N agents must each get their own scenery, not share agent 0's.

    Without `per_agent_include_uri`, agents 1..N-1 would spawn on bare ground,
    see no line, and terminate on step 1 — silently poisoning training with
    near-empty episodes while agent 0 looked fine.
    """
    spec = get_spec("line_follower")
    rng = np.random.default_rng(0)
    sdf, poses = ip._build_world(spec, 4, spec.x_spacing, rng)
    assert sdf.count('<model name="line_follower_') == 4
    assert sdf.count("scenery_") == 4, "one track per agent"
    # tracks must not overlap: the model is 2.2 m across, spacing is wider
    xs = sorted(p[0] for p in poses)
    # xs and xs[1:] are deliberately uneven lengths (pairwise iteration).
    assert min(b - a for a, b in zip(xs, xs[1:], strict=False)) > 2.2

    if not _rendering_works():  # pragma: no cover
        pytest.skip("headless camera rendering unavailable on this machine")
    # Must run in its own process: only ONE camera env may exist per process
    # (gz-sim's render scene is a process-wide singleton), and the module
    # fixture above has already claimed this process's slot.
    proc = subprocess.run(
        [sys.executable, "-c", _MULTI_AGENT_PROBE], capture_output=True, text=True, timeout=300
    )
    assert proc.returncode == 0, (
        f"multi-agent camera probe failed (rc={proc.returncode})\n"
        f"{proc.stdout[-600:]}\n{proc.stderr[-600:]}"
    )
    assert "ALL_AGENTS_SEE_LINE" in proc.stdout, (
        f"not every agent saw its own track:\n{proc.stdout[-600:]}"
    )
    assert "NO_SPURIOUS_TERMINATIONS" in proc.stdout, (
        f"an agent terminated early, likely a missing track:\n{proc.stdout[-600:]}"
    )


def test_line_loss_terminates_and_pose_reset_recovers(lf_env):
    lf_env.reset()
    lost = None
    for k in range(150):
        obs, _r, dones, _i = lf_env.step(np.array([[1.0, -1.0]]))  # spin
        if dones[0]:
            lost = k
            break
    assert lost is not None, "spinning in place should lose the line"
    # same-step autoreset restored the chassis pose: the line is back
    assert _lf_line_centroid(obs[0]) is not None


_TOPIC_NAMESPACE_PROBE = """
import numpy as np
from gazebo_gymnasium_bridge.envs import make_inprocess

env = make_inprocess("line_follower", n_agents=2, seed=0)
try:
    env.reset()
    prefix = env._topic_prefix
    # Namespaced per env instance, so two processes never share topic names.
    if prefix != "/rl" and str(os.getpid()) in prefix:
        print("TOPIC_PREFIX_NAMESPACED", prefix)
finally:
    env.close()
"""


def test_camera_topics_are_namespaced_per_process():
    # gz-transport discovery is machine-global, so unnamespaced /rl/camera_i
    # topics let two concurrent camera envs read each other's frames. That
    # failure is silent -- the frames are real images, just of the wrong
    # rover -- and it took a known-good policy from 95.8% solved to 0%.
    # Assert on the topic names rather than trying to reproduce the
    # cross-talk, which would need two real camera worlds at once.
    if not _rendering_works():  # pragma: no cover
        pytest.skip("headless camera rendering unavailable on this machine")
    proc = subprocess.run(
        [sys.executable, "-c", "import os\n" + _TOPIC_NAMESPACE_PROBE],
        capture_output=True,
        text=True,
        # Generous: building a camera world is slow on a loaded machine, and
        # this test is about topic NAMING, not about how fast gz starts up.
        timeout=300,
    )
    assert proc.returncode == 0, (
        f"topic-namespace probe failed (rc={proc.returncode})\n"
        f"{proc.stdout[-600:]}\n{proc.stderr[-600:]}"
    )
    assert "TOPIC_PREFIX_NAMESPACED" in proc.stdout, (
        f"camera topics are not namespaced per process:\n{proc.stdout[-600:]}"
    )
