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

"""Unit tests for the AgentSpec layer.

Pure-Python: no gz bindings, no running simulator. Observation extraction is
exercised against duck-typed fake joint_state messages.

Run with: pytest gazebo_gymnasium_bridge/test/test_agent_spec.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # gazebo_gymnasium_bridge/

from gazebo_gymnasium_bridge.envs.agent_spec import (  # noqa: E402
    AgentSpec,
    JointObs,
    get_spec,
    register_spec,
    registered_specs,
)


# --- fake gz Model joint_state message ------------------------------------- #

class _FakeAxis:
    def __init__(self, position, velocity):
        self.position = position
        self.velocity = velocity


class _FakeJoint:
    def __init__(self, name, position, velocity):
        self.name = name
        self.axis1 = _FakeAxis(position, velocity)


class _FakeModelMsg:
    def __init__(self, joints):
        # joints: list of (name, pos, vel)
        self.joint = [_FakeJoint(n, p, v) for (n, p, v) in joints]


def _make_spec(**overrides):
    from gymnasium import spaces
    base = dict(
        name="t",
        model_uri="package://x/models/t",
        observation_space=spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32),
        action_space=spaces.Discrete(2),
        joint_obs=(JointObs("a"), JointObs("b")),
        reward_fn=lambda obs, action: 1.0,
        terminated_fn=lambda obs: False,
    )
    base.update(overrides)
    return AgentSpec(**base)


class TestObsExtraction:
    def test_order_is_pos_then_vel_per_joint(self):
        spec = _make_spec()
        msg = _FakeModelMsg([("a", 1.0, 2.0), ("b", 3.0, 4.0)])
        obs = spec.obs_from_joint_state(msg)
        np.testing.assert_allclose(obs, [1.0, 2.0, 3.0, 4.0])
        assert obs.dtype == np.float32

    def test_joint_order_follows_spec_not_message(self):
        spec = _make_spec()
        # message lists b before a; spec asks a then b
        msg = _FakeModelMsg([("b", 3.0, 4.0), ("a", 1.0, 2.0)])
        obs = spec.obs_from_joint_state(msg)
        np.testing.assert_allclose(obs, [1.0, 2.0, 3.0, 4.0])

    def test_missing_joint_contributes_zeros(self):
        spec = _make_spec()
        msg = _FakeModelMsg([("a", 1.0, 2.0)])  # b absent
        obs = spec.obs_from_joint_state(msg)
        np.testing.assert_allclose(obs, [1.0, 2.0, 0.0, 0.0])

    def test_position_only_and_velocity_only(self):
        from gymnasium import spaces
        spec = _make_spec(
            observation_space=spaces.Box(low=-1, high=1, shape=(2,),
                                         dtype=np.float32),
            joint_obs=(JointObs("a", position=True, velocity=False),
                       JointObs("b", position=False, velocity=True)),
        )
        msg = _FakeModelMsg([("a", 1.0, 9.0), ("b", 9.0, 4.0)])
        obs = spec.obs_from_joint_state(msg)
        np.testing.assert_allclose(obs, [1.0, 4.0])


class TestValidation:
    def test_mismatched_obs_dim_raises(self):
        from gymnasium import spaces
        with pytest.raises(ValueError, match="joint_obs widths"):
            _make_spec(
                observation_space=spaces.Box(low=-1, high=1, shape=(3,),
                                             dtype=np.float32),
            )  # joint_obs sums to 4, space says 3

    def test_zero_spawn_z_raises(self):
        with pytest.raises(ValueError, match="spawn_z"):
            _make_spec(spawn_z=0.0)


class TestRegistry:
    def test_cartpole_registered(self):
        assert "cartpole" in registered_specs()

    def test_get_cartpole_spec_shapes(self):
        spec = get_spec("cartpole")
        assert spec.observation_space.shape == (4,)
        assert spec.action_space.n == 2
        assert spec.spawn_z == 0.10

    def test_cartpole_reward_and_termination(self):
        spec = get_spec("cartpole")
        upright = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        fallen = np.array([0.0, 0.0, 0.30, 0.0], dtype=np.float32)
        out_of_bounds = np.array([3.0, 0.0, 0.0, 0.0], dtype=np.float32)
        assert spec.reward_fn(upright, 0) == 1.0
        assert spec.terminated_fn(upright) is False
        assert spec.terminated_fn(fallen) is True
        assert spec.terminated_fn(out_of_bounds) is True

    def test_cartpole_obs_extraction_canonical_order(self):
        spec = get_spec("cartpole")
        msg = _FakeModelMsg([
            ("slider_to_cart", 0.5, -0.3),   # cart_pos, cart_vel
            ("cart_to_pole", 0.1, 0.2),       # pole_angle, pole_ang_vel
        ])
        obs = spec.obs_from_joint_state(msg)
        np.testing.assert_allclose(obs, [0.5, -0.3, 0.1, 0.2])

    def test_unknown_spec_raises(self):
        with pytest.raises(KeyError, match="unknown agent spec"):
            get_spec("does_not_exist")

    def test_register_and_get(self):
        register_spec("t_dummy", _make_spec)
        assert "t_dummy" in registered_specs()
        assert get_spec("t_dummy").name == "t"


class TestHarnessActuation:
    """The in-sim harness (ECM) actuation + reset descriptors."""

    def test_cartpole_action_to_commands(self):
        spec = get_spec("cartpole")
        assert spec.action_to_commands is not None
        push_right = spec.action_to_commands(1)
        push_left = spec.action_to_commands(0)
        assert push_right == [("slider_to_cart", "velocity", 1.0)]
        assert push_left == [("slider_to_cart", "velocity", -1.0)]

    def test_cartpole_reset_joint_state_randomizes_pole(self):
        spec = get_spec("cartpole")
        assert spec.reset_joint_state is not None
        rng = np.random.default_rng(0)
        st = spec.reset_joint_state(rng)
        assert st["slider_to_cart"] == (0.0, 0.0)
        pole_pos, pole_vel = st["cart_to_pole"]
        assert -0.05 <= pole_pos <= 0.05 and pole_vel == 0.0
        # Different draws give different pole angles (real per-agent diversity).
        st2 = spec.reset_joint_state(rng)
        assert st2["cart_to_pole"][0] != pole_pos

    def test_command_joints_exist_in_obs(self):
        # Every actuated joint should be one we also observe (sanity).
        spec = get_spec("cartpole")
        obs_joints = {j.joint for j in spec.joint_obs}
        for (joint, _mode, _v) in spec.action_to_commands(1):
            assert joint in obs_joints
