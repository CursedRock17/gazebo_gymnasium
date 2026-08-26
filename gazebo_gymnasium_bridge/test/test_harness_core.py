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
"""In-process test of the harness ECM core via gz.sim8.TestFixture.

Loads a world with two bare cartpoles (geometry only — no JointController /
JointStatePublisher), drives HarnessCore from the sim's pre/post-update
callbacks, and checks that ECM actuation, observation reads, and in-place
reset all behave. No simulator launch, no gz-transport.

Requires the gz.sim8 Python bindings (Gazebo). Skipped otherwise.

Run with: pixi run -- python -m pytest \
    gazebo_gymnasium_bridge/test/test_harness_core.py -q
"""

from pathlib import Path
import sys

from gymnasium import spaces
import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

gz_sim = pytest.importorskip("gz.sim8", reason="gz.sim8 bindings not available")

from gazebo_gymnasium_bridge.envs.agent_spec import AgentSpec  # noqa: E402,I100
from gazebo_gymnasium_bridge.envs.agent_spec import get_spec  # noqa: E402
from gazebo_gymnasium_bridge.harness.harness_core import HarnessCore  # noqa: E402

_I_CART = "<inertia><ixx>0.00854</ixx><iyy>0.00667</iyy><izz>0.00854</izz></inertia>"
_I_POLE = "<inertia><ixx>0.08363</ixx><iyy>0.08347</iyy><izz>0.000433</izz></inertia>"


def _box(size):
    return f"<geometry><box><size>{size}</size></box></geometry>"


def _bare_cartpole(index, x):
    # z=0.60 keeps the cart clear of the ground plane — resting on it pins the
    # cart with contact friction, which silently defeats force actuation.
    return f"""
    <model name="cartpole_{index}">
      <pose>{x} 0 0.60 0 0 0</pose>
      <link name="slider">
        <inertial><mass>1</mass>{_I_CART}</inertial>
        <collision name="c">{_box("0.03 8 0.03")}</collision></link>
      <joint name="slider_to_cart" type="prismatic">
        <pose relative_to="slider">0 0 0 0 0 0</pose>
        <parent>slider</parent><child>cart</child>
        <axis><xyz>0 1 0</xyz>
          <limit><lower>-4</lower><upper>4</upper></limit></axis></joint>
      <link name="cart">
        <pose relative_to="slider_to_cart">0 0 0 0 0 0</pose>
        <inertial><mass>1</mass>{_I_CART}</inertial>
        <collision name="c">{_box("0.2 0.25 0.2")}</collision></link>
      <joint name="cart_to_pole" type="revolute">
        <pose relative_to="cart">0.12 0 0 0 0 0</pose>
        <parent>cart</parent><child>pole</child>
        <axis><xyz>1 0 0</xyz>
          <limit><effort>1000</effort><velocity>8</velocity>
            <lower>-1e9</lower><upper>1e9</upper></limit></axis></joint>
      <link name="pole">
        <pose relative_to="cart_to_pole">0 0 0 0 0 0</pose>
        <inertial><pose>0 0 0.47 0 0 0</pose><mass>1</mass>{_I_POLE}</inertial>
        <collision name="c">{_box("0.04 0.06 1")}</collision></link>
      <joint name="world_to_slider" type="fixed">
        <parent>world</parent><child>slider</child></joint>
    </model>"""


def _world_sdf():
    return f"""<?xml version="1.0" ?>
    <sdf version="1.8">
      <world name="harness_test">
        <physics name="10ms" type="ignored">
          <max_step_size>0.01</max_step_size>
          <real_time_update_rate>0</real_time_update_rate></physics>
        <plugin filename="gz-sim-physics-system"
                name="gz::sim::systems::Physics"/>
        <plugin filename="gz-sim-scene-broadcaster-system"
                name="gz::sim::systems::SceneBroadcaster"/>
        <model name="ground"><static>true</static>
          <link name="l"><collision name="c"><geometry><plane>
            <normal>0 0 1</normal><size>50 50</size></plane>
          </geometry></collision></link></model>
        {_bare_cartpole(0, -1.5)}
        {_bare_cartpole(1, 1.5)}
      </world>
    </sdf>"""


@pytest.fixture
def world_path(tmp_path):
    p = tmp_path / "harness_world.sdf"
    p.write_text(_world_sdf())
    return str(p)


def _run(world_path, on_pre, on_post, ticks):
    fx = gz_sim.TestFixture(world_path)
    fx.on_pre_update(on_pre)
    fx.on_post_update(on_post)
    fx.finalize()
    fx.server().run(True, ticks, False)


def test_apply_actions_moves_carts_independently(world_path):
    core = HarnessCore(get_spec("cartpole"), n_agents=2)

    def on_pre(info, ecm):
        core.resolve(ecm)
        core.apply_actions(ecm, [1, 0])  # agent0 +vel, agent1 -vel

    obs_box = {}

    def on_post(info, ecm):
        obs_box["obs"] = core.read_obs(ecm)

    _run(world_path, on_pre, on_post, 120)
    obs = obs_box["obs"]
    assert obs.shape == (2, 4)
    # cart position is obs[:,0]. agent0 driven +, agent1 driven -.
    assert obs[0, 0] > 0.3, f"agent0 should move +y, got {obs[0, 0]}"
    assert obs[1, 0] < -0.3, f"agent1 should move -y, got {obs[1, 0]}"


def test_read_obs_reports_pole_fall(world_path):
    # No actuation; nudge nothing — but reset the pole to a tilt, then let it
    # fall, and confirm read_obs tracks a growing pole angle.
    core = HarnessCore(get_spec("cartpole"), n_agents=2)
    angles = []
    did_reset = {"done": False}

    def on_pre(info, ecm):
        core.resolve(ecm)
        if not did_reset["done"] and all(core._resolved):
            # force a known tilt on agent 0's pole
            core._joints[(0, "cart_to_pole")].reset_position(ecm, [0.1])
            did_reset["done"] = True

    def on_post(info, ecm):
        obs = core.read_obs(ecm)
        angles.append(float(obs[0, 2]))  # agent0 pole angle

    _run(world_path, on_pre, on_post, 100)
    assert did_reset["done"]
    # pole released at 0.1 rad should fall further (angle magnitude grows)
    assert abs(angles[-1]) > abs(angles[len(angles) // 2]) > 0.05


def test_reset_sets_random_pole_angle_in_place(world_path):
    core = HarnessCore(get_spec("cartpole"), n_agents=2)
    rng = np.random.default_rng(42)
    captured = {}

    def on_pre(info, ecm):
        core.resolve(ecm)
        if all(core._resolved) and "after" not in captured:
            core.reset(ecm, rng)
            captured["did_reset"] = True

    def on_post(info, ecm):
        if captured.get("did_reset") and "after" not in captured:
            captured["after"] = core.read_obs(ecm).copy()

    _run(world_path, on_pre, on_post, 40)
    after = captured["after"]
    # both carts centered (~0), poles within the +/-0.05 reset band, and the
    # two agents got DIFFERENT angles (real per-agent randomization).
    assert abs(after[0, 0]) < 0.05 and abs(after[1, 0]) < 0.05
    assert abs(after[0, 2]) <= 0.06 and abs(after[1, 2]) <= 0.06
    assert after[0, 2] != after[1, 2]


# ---- base_obs (free-floating 3D base, e.g. ant/humanoid) ------------------ #
#
# A minimal synthetic world -- one unconstrained "torso" link (no parent
# joint, no <static>), so gz-sim treats it as a free 6-DOF body -- rather
# than the real ant model, to keep this a fast, isolated HarnessCore test
# per this file's own established pattern (see _bare_cartpole above).


def _free_body_spec():
    return AgentSpec(
        name="free_body",
        model_uri="unused://synthetic-test-only",
        observation_space=spaces.Box(low=-np.inf, high=np.inf, shape=(11,), dtype=np.float32),
        action_space=spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32),
        joint_obs=(),
        base_obs=True,
        reward_fn=lambda obs, action: 0.0,
        terminated_fn=lambda obs: False,
        spawn_z=2.0,
    )


def _free_body_model(index, x):
    return f"""
    <model name="free_body_{index}">
      <pose>{x} 0 2.0 0 0 0</pose>
      <link name="torso">
        <inertial><mass>1</mass>
          <inertia><ixx>0.1</ixx><iyy>0.1</iyy><izz>0.1</izz></inertia>
        </inertial>
        <collision name="c">{_box("0.2 0.2 0.2")}</collision></link>
    </model>"""


def _free_body_world_sdf():
    return f"""<?xml version="1.0" ?>
    <sdf version="1.8">
      <world name="free_body_test">
        <physics name="10ms" type="ignored">
          <max_step_size>0.01</max_step_size>
          <real_time_update_rate>0</real_time_update_rate></physics>
        <plugin filename="gz-sim-physics-system"
                name="gz::sim::systems::Physics"/>
        <plugin filename="gz-sim-scene-broadcaster-system"
                name="gz::sim::systems::SceneBroadcaster"/>
        <model name="ground"><static>true</static>
          <link name="l"><collision name="c"><geometry><plane>
            <normal>0 0 1</normal><size>50 50</size></plane>
          </geometry></collision></link></model>
        {_free_body_model(0, 0)}
        {_free_body_model(1, 3)}
      </world>
    </sdf>"""


@pytest.fixture
def free_body_world_path(tmp_path):
    p = tmp_path / "free_body_world.sdf"
    p.write_text(_free_body_world_sdf())
    return str(p)


def test_base_obs_free_fall_matches_physics(free_body_world_path):
    # No <static>, no parent joint on "torso" -> gz-sim treats it as an
    # unconstrained free body; nothing actuates it, so it just falls.
    core = HarnessCore(_free_body_spec(), n_agents=2)
    frames = []

    def on_pre(info, ecm):
        core.resolve(ecm)

    def on_post(info, ecm):
        if all(core._resolved):
            frames.append(core.read_obs(ecm).copy())

    _run(free_body_world_path, on_pre, on_post, 40)
    assert frames, "no obs read -- resolve() never bound the free link"
    first, last = frames[0], frames[-1]
    assert first.shape == (2, 11)
    # z (index 0) drops under gravity; quaternion (1:5) starts near-identity
    # (upright, no rotation imparted).
    assert last[0, 0] < first[0, 0] - 0.01, (
        f"z should fall under gravity, got {first[0, 0]} -> {last[0, 0]}"
    )
    np.testing.assert_allclose(first[0, 1:5], [1, 0, 0, 0], atol=0.05)
    # lin_vel z (index 7 = 5 pos + 2 lin_vel-xy) is negative and growing in
    # magnitude (accelerating downward): index layout is
    # [z, qw,qx,qy,qz | vx,vy,vz, wx,wy,wz] -> vz is index 7.
    assert last[0, 7] < first[0, 7] < 0, (
        f"downward speed should grow, got vz {first[0, 7]} -> {last[0, 7]}"
    )
    # two agents are independent -- different x doesn't matter, but neither
    # is a stale zero (both actually fell).
    assert last[1, 0] < first[1, 0] - 0.01


def test_base_obs_reset_zeroes_velocity(free_body_world_path):
    core = HarnessCore(_free_body_spec(), n_agents=1)
    rng = np.random.default_rng(0)
    captured = {}

    def on_pre(info, ecm):
        core.resolve(ecm)
        if (
            all(core._resolved)
            and "reset_at" not in captured
            and len(captured.get("pre_reset_vz", [])) >= 20
        ):
            core.reset(ecm, rng)
            captured["reset_at"] = True

    def on_post(info, ecm):
        if not all(core._resolved):
            return
        obs = core.read_obs(ecm)
        if "reset_at" not in captured:
            captured.setdefault("pre_reset_vz", []).append(float(obs[0, 7]))
        else:
            captured["post_reset_obs"] = obs.copy()

    _run(free_body_world_path, on_pre, on_post, 60)
    assert captured["pre_reset_vz"][-1] < -0.05, (
        "should have picked up real downward speed before reset"
    )
    post = captured["post_reset_obs"]
    # reset() runs in on_pre, obs is read in the SAME tick's on_post (after
    # physics), so one tick's worth of gravity (g*dt = 9.8*0.01 = 0.098)
    # always accumulates before the reset is visible -- same reasoning as
    # test_reset_sets_random_pole_angle_in_place's non-zero tolerance above.
    assert abs(post[0, 7]) < 0.15, f"velocity should be ~zeroed by reset, got vz={post[0, 7]}"
