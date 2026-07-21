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
from gazebo_gymnasium_bridge.envs.agent_spec import (  # noqa: E402
    _lf_line_centroid,
    get_spec,
)


def _frame(dark_col=None):
    img = np.full((64, 64, 3), 220, dtype=np.uint8)
    if dark_col is not None:
        img[32:, dark_col - 2:dark_col + 2, :] = 10   # line in lower half
    return img


class TestLineFollowerSpecMath:
    def test_spaces(self):
        spec = get_spec("line_follower")
        assert spec.image_obs == (64, 64, 3)
        assert spec.observation_space.dtype == np.uint8
        assert spec.action_space.shape == (2,)

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
        assert spec.reward_fn(_frame(32), fwd) > spec.reward_fn(_frame(32),
                                                                stop)
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
        proc = subprocess.run([sys.executable, "-c", _PROBE],
                              capture_output=True, timeout=180)
    except (subprocess.TimeoutExpired, OSError):
        return False
    return proc.returncode == 0


@pytest.fixture(scope="module")
def lf_env():
    from gazebo_gymnasium_bridge.envs import make_inprocess
    if not _rendering_works():                       # pragma: no cover
        pytest.skip("headless camera rendering unavailable on this machine "
                    "(no usable render device); vision tests skipped")
    env = make_inprocess("line_follower", n_agents=1, seed=0)
    yield env
    env.close()


def test_camera_obs_shape_and_line_visible(lf_env):
    obs = lf_env.reset()
    assert obs.shape == (1, 64, 64, 3) and obs.dtype == np.uint8
    c = _lf_line_centroid(obs[0])
    assert c is not None and 0.3 < c < 0.7, \
        f"line should start near-centered, centroid={c}"


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
    assert min(b - a for a, b in zip(xs, xs[1:])) > 2.2

    if not _rendering_works():                       # pragma: no cover
        pytest.skip("headless camera rendering unavailable on this machine")
    # Must run in its own process: only ONE camera env may exist per process
    # (gz-sim's render scene is a process-wide singleton), and the module
    # fixture above has already claimed this process's slot.
    proc = subprocess.run([sys.executable, "-c", _MULTI_AGENT_PROBE],
                          capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, (
        f"multi-agent camera probe failed (rc={proc.returncode})\n"
        f"{proc.stdout[-600:]}\n{proc.stderr[-600:]}")
    assert "ALL_AGENTS_SEE_LINE" in proc.stdout, (
        f"not every agent saw its own track:\n{proc.stdout[-600:]}")
    assert "NO_SPURIOUS_TERMINATIONS" in proc.stdout, (
        f"an agent terminated early, likely a missing track:\n"
        f"{proc.stdout[-600:]}")


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
