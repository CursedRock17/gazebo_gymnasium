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
"""Stress tests for the multi-agent environments.

Three axes, all headless (no simulator launch):

* **Scale** — many agents in one world stay independent and finite. Real
  physics via ``gz.sim8.TestFixture`` + ``HarnessCore``.
* **Endurance** — thousands of client steps and many group auto-resets keep the
  SB3 ``VecEnv`` contract and per-agent bookkeeping intact, with no unbounded
  growth. Mocked transport, so it runs fast and deterministically.
* **Robustness** — malformed observation frames, step timeouts, boundary
  agent counts (N=1, N=64) never corrupt state or raise.

These are the invariants a "standard RL environment" must hold before anyone
trusts a training run to it.
"""

from pathlib import Path
import sys
from unittest.mock import MagicMock

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))  # test_harness_core: the bare-cartpole SDF

# --------------------------------------------------------------------------- #
# Scale + endurance against real physics (needs the gz bindings)
# --------------------------------------------------------------------------- #

gz_sim = pytest.importorskip("gz.sim8", reason="gz.sim8 bindings not available")

import test_harness_core as thc  # noqa: E402

from gazebo_gymnasium_bridge.envs.agent_spec import get_spec  # noqa: E402,I100
from gazebo_gymnasium_bridge.harness.harness_core import HarnessCore  # noqa: E402


def _world_n(n, spacing=3.0):
    """Build a world SDF with ``n`` bare cartpoles spread along X."""
    offset = (n - 1) * spacing / 2.0
    models = "\n".join(thc._bare_cartpole(i, i * spacing - offset) for i in range(n))
    return f"""<?xml version="1.0" ?>
    <sdf version="1.8">
      <world name="stress">
        <physics name="10ms" type="ignored">
          <max_step_size>0.01</max_step_size>
          <real_time_update_rate>0</real_time_update_rate></physics>
        <plugin filename="gz-sim-physics-system"
                name="gz::sim::systems::Physics"/>
        <plugin filename="gz-sim-scene-broadcaster-system"
                name="gz::sim::systems::SceneBroadcaster"/>
        <model name="ground"><static>true</static>
          <link name="l"><collision name="c"><geometry><plane>
            <normal>0 0 1</normal><size>500 500</size></plane>
          </geometry></collision></link></model>
        {models}
      </world>
    </sdf>"""


def _run(world_sdf, on_pre, on_post, ticks, tmp_path):
    p = tmp_path / "stress_world.sdf"
    p.write_text(world_sdf)
    fx = gz_sim.TestFixture(str(p))
    fx.on_pre_update(on_pre)
    fx.on_post_update(on_post)
    fx.finalize()
    fx.server().run(True, ticks, False)


@pytest.mark.parametrize("n", [16, 32])
def test_many_agents_stay_independent_and_finite(n, tmp_path):
    """N agents in one world respond independently with finite observations."""
    core = HarnessCore(get_spec("cartpole"), n_agents=n)
    # alternate the discrete action: even agents 0 (-v), odd agents 1 (+v).
    actions = [i % 2 for i in range(n)]

    def on_pre(info, ecm):
        core.resolve(ecm)
        core.apply_actions(ecm, actions)

    box = {}

    def on_post(info, ecm):
        box["obs"] = core.read_obs(ecm)

    _run(_world_n(n), on_pre, on_post, 120, tmp_path)
    obs = box["obs"]
    assert obs.shape == (n, 4)
    assert np.isfinite(obs).all(), "non-finite observation under load"
    # Each agent's cart moved the commanded direction — no cross-talk.
    cart_x = obs[:, 0]
    assert (cart_x[0::2] < -0.1).all(), "even agents should move -y"
    assert (cart_x[1::2] > 0.1).all(), "odd agents should move +y"


def test_repeated_in_place_resets_stay_clean(tmp_path):
    """Many reset cycles keep carts centered and pole angles distinct + in band."""
    n = 12
    core = HarnessCore(get_spec("cartpole"), n_agents=n)
    rng = np.random.default_rng(7)
    interval = 15
    episodes = []  # captured pole-angle vectors, one per reset
    state = {"tick": 0, "captured_for": -1}

    def on_pre(info, ecm):
        core.resolve(ecm)
        if all(core._resolved) and state["tick"] % interval == 0:
            core.reset(ecm, rng)
            state["captured_for"] = state["tick"]
        state["tick"] += 1

    def on_post(info, ecm):
        # capture one frame right after each reset
        if state["captured_for"] >= 0:
            obs = core.read_obs(ecm)
            episodes.append(obs.copy())
            state["captured_for"] = -1

    _run(_world_n(n), on_pre, on_post, interval * 8, tmp_path)

    assert len(episodes) >= 5, "expected several reset cycles"
    for obs in episodes:
        assert np.abs(obs[:, 0]).max() < 0.05, "carts not recentered after reset"
        assert np.abs(obs[:, 2]).max() <= 0.06, "pole angle outside reset band"
        # per-agent randomization actually differs across agents
        assert len(np.unique(np.round(obs[:, 2], 6))) > n // 2


# --------------------------------------------------------------------------- #
# Endurance + robustness of the client VecEnv (mocked transport, fast)
# --------------------------------------------------------------------------- #

gz_transport13 = pytest.importorskip("gz.transport13", reason="gz-transport not available")
pytest.importorskip("gz.msgs10", reason="gz-msgs not available")


def _offline_harness(monkeypatch, n_agents, upright=True):
    monkeypatch.setattr(gz_transport13, "Node", MagicMock(name="Node"))
    from gazebo_gymnasium_bridge.envs import make_harness

    env = make_harness(
        "cartpole", n_agents=n_agents, world_name="stress", reset_timeout=0.01, step_timeout=0.01
    )
    monkeypatch.setattr(env, "_wait_frame", lambda timeout: True)
    monkeypatch.setattr(env._obs_event, "wait", lambda timeout=None: True)
    env._latest_obs[:] = 0.0 if upright else np.array([0, 0, 0.5, 0], dtype=np.float32)
    return env


def test_vecenv_endurance_thousands_of_steps(monkeypatch):
    """3000 steps + many truncation resets keep the contract and bookkeeping."""
    n = 8
    env = _offline_harness(monkeypatch, n)
    obs = env.reset()
    assert obs.shape == (n, 4)

    resets = 0
    for _ in range(3000):
        env.step_async(np.zeros(n, dtype=int))
        obs, rewards, dones, infos = env.step_wait()
        assert obs.shape == (n, 4) and obs.dtype == np.float32
        assert rewards.shape == (n,) and dones.shape == (n,)
        assert len(infos) == n
        if dones.all():
            resets += 1
            assert all("terminal_observation" in d for d in infos)

    # upright agents only ever end via truncation at max_episode_steps.
    assert resets >= 3000 // env.max_episode_steps - 1
    # no unbounded growth — bookkeeping arrays stay width N.
    assert env._episode_rewards.shape == (n,)
    assert env._dones.shape == (n,)
    assert env._current_episode == resets + 1  # +1 for the initial reset


def test_vecenv_mixed_deaths_trigger_group_reset(monkeypatch):
    """When every agent has fallen the group resets exactly once, together."""
    n = 6
    env = _offline_harness(monkeypatch, n, upright=False)
    env.reset()
    env.step_async(np.zeros(n, dtype=int))
    _obs, _rew, dones, infos = env.step_wait()
    assert dones.all()
    assert all("terminal_observation" in d for d in infos)
    # after the group reset the fresh episode starts clean
    assert not env._dones.any()
    assert env._steps_since_reset == 0


def test_vecenv_ignores_malformed_obs_frames(monkeypatch):
    """A wrong-sized observation frame is dropped, not reshaped into garbage."""
    n = 4
    env = _offline_harness(monkeypatch, n)
    good = np.arange(n * 4, dtype=np.float32)
    env._latest_obs[:] = good.reshape(n, 4)

    class _Msg:
        pass

    bad = _Msg()
    bad.data = [1.0, 2.0, 3.0]  # not n*obs_dim
    env._on_obs(bad)  # must be ignored
    np.testing.assert_array_equal(env._latest_obs, good.reshape(n, 4))


def test_vecenv_step_timeout_marks_all_done(monkeypatch):
    """A stalled sim (no obs before step_timeout) fails safe: all agents done."""
    n = 4
    env = _offline_harness(monkeypatch, n)
    env.reset()
    # make the wait time out instead of returning True
    monkeypatch.setattr(env._obs_event, "wait", lambda timeout=None: False)
    env.step_async(np.zeros(n, dtype=int))
    _obs, _rew, dones, _infos = env.step_wait()
    assert dones.all()


@pytest.mark.parametrize("n", [1, 64])
def test_vecenv_boundary_agent_counts(monkeypatch, n):
    """N=1 and a large N=64 both construct and step without error."""
    env = _offline_harness(monkeypatch, n)
    obs = env.reset()
    assert obs.shape == (n, 4)
    for _ in range(10):
        env.step_async(np.zeros(n, dtype=int))
        obs, rewards, dones, infos = env.step_wait()
        assert obs.shape == (n, 4)
        assert rewards.shape == (n,)
