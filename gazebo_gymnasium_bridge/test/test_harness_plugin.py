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

"""Headless integration test of the batched harness plugin.

TestFixture hosts an in-process server (two bare cartpoles); the plugin's
pre/post-update run from the sim callbacks; a real in-process gz-transport
client publishes batched actions / reset and receives batched observations.
This exercises the whole harness loop (transport in -> ECM apply -> ECM read
-> transport out) without launching a simulator.

Run with: pixi run -- python -m pytest \
    gazebo_gymnasium_bridge/test/test_harness_plugin.py -q
"""

from pathlib import Path
import sys
import time

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))  # test_harness_core (for the world SDF)
# Make the plugin importable (it lives with the resources plugins).
_PLUGINS = (HERE.parent.parent / "gazebo_gymnasium_examples"
            / "gazebo_gymnasium_resources" / "plugins")
sys.path.insert(0, str(_PLUGINS))

gz_sim = pytest.importorskip("gz.sim8", reason="gz.sim8 bindings not available")
pytest.importorskip("gz.transport13", reason="gz-transport not available")

from gz.msgs10.float_v_pb2 import Float_V           # noqa: E402,I100
from gz.transport13 import AdvertiseMessageOptions  # noqa: E402
from gz.transport13 import Node                     # noqa: E402
import multi_agent_harness as mah                   # noqa: E402
import test_harness_core as thc                     # noqa: E402


@pytest.fixture
def world_path(tmp_path):
    p = tmp_path / "harness_world.sdf"
    p.write_text(thc._world_sdf())
    return str(p)


def _client():
    node = Node()
    act = node.advertise(mah.ACTION_TOPIC, Float_V, AdvertiseMessageOptions())
    rst = node.advertise(mah.RESET_TOPIC, Float_V, AdvertiseMessageOptions())
    frames = []
    node.subscribe(Float_V, mah.OBS_TOPIC, lambda m: frames.append(list(m.data)))
    return node, act, rst, frames


def test_batched_action_obs_roundtrip(world_path):
    h = mah.MultiAgentHarness()
    h.setup("cartpole", 2)
    node, act_pub, _rst, frames = _client()
    time.sleep(0.4)  # transport pairing

    # one batched action msg for both agents: agent0 +1, agent1 0(-> -v)
    msg = Float_V()
    msg.data.extend([1.0, 0.0])
    for _ in range(3):
        act_pub.publish(msg)
        time.sleep(0.1)

    fx = gz_sim.TestFixture(world_path)
    fx.on_pre_update(h.pre_update)
    fx.on_post_update(h.post_update)
    fx.finalize()
    fx.server().run(True, 120, False)
    time.sleep(0.3)

    assert frames, "client received no /rl/observations"
    last = np.array(frames[-1]).reshape(2, 4)   # (n_agents, obs_dim)
    # cart positions (col 0): agent0 driven +, agent1 driven -
    assert last[0, 0] > 0.3, f"agent0 cart should be +, got {last[0, 0]}"
    assert last[1, 0] < -0.3, f"agent1 cart should be -, got {last[1, 0]}"


def test_reset_command_recenters_and_randomizes(world_path):
    h = mah.MultiAgentHarness()
    h.setup("cartpole", 2, seed=7)
    node, act_pub, rst_pub, frames = _client()
    time.sleep(0.4)

    # drive the carts out first
    drive = Float_V()
    drive.data.extend([1.0, 1.0])
    for _ in range(3):
        act_pub.publish(drive)
        time.sleep(0.1)

    fx = gz_sim.TestFixture(world_path)
    fx.on_pre_update(h.pre_update)
    fx.on_post_update(h.post_update)
    fx.finalize()
    server = fx.server()
    server.run(True, 80, False)          # carts move away
    moved = np.array(frames[-1]).reshape(2, 4)
    assert abs(moved[0, 0]) > 0.1

    # now command a reset and step a little; carts should recenter, poles get
    # small distinct random angles (in-place, no respawn)
    rst = Float_V()
    rst.data.extend([7.0])
    rst_pub.publish(rst)
    time.sleep(0.3)
    server.run(True, 5, False)
    time.sleep(0.3)
    after = np.array(frames[-1]).reshape(2, 4)
    # Recentered from ~2 m back toward 0; the small residual is the held drive
    # action re-accelerating the cart over the few post-reset ticks (magnitude
    # scales with _CART_SPEED), so allow a modest band rather than ~0.
    assert abs(after[0, 0]) < 0.2 and abs(after[1, 0]) < 0.2, \
        f"carts should recenter, got {after[:, 0]}"
    assert abs(after[0, 2]) <= 0.06 and after[0, 2] != after[1, 2]


def test_no_publish_until_all_agents_bound(world_path):
    # world_path has 2 cartpoles; ask the harness for 3 -> agent 2 never
    # resolves -> the plugin must publish NOTHING (no all-zeros stream that
    # would fake a solved episode).
    h = mah.MultiAgentHarness()
    h.setup("cartpole", 3)
    _node, _act, _rst, frames = _client()
    time.sleep(0.4)

    fx = gz_sim.TestFixture(world_path)
    fx.on_pre_update(h.pre_update)
    fx.on_post_update(h.post_update)
    fx.finalize()
    fx.server().run(True, 60, False)
    time.sleep(0.3)

    assert not frames, "harness must stay silent until every agent is bound"
