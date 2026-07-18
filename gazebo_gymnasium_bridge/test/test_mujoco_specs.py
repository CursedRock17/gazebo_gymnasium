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

"""Tests for the MuJoCo-ported AgentSpecs (InvertedDoublePendulum, ...).

Spec math is tested offline; the physics tests run the real in-process sim
(TestFixture) headlessly — instability, termination, determinism.
"""

from pathlib import Path
import sys

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from gazebo_gymnasium_bridge.envs.agent_spec import (  # noqa: E402
    _idp_tip,
    _IDP_TIP_MAX,
    get_spec,
)


class TestIDPSpecMath:
    """Offline: observation layout, action map, tip kinematics, termination."""

    def test_spaces_and_layout(self):
        spec = get_spec("inverted_double_pendulum")
        assert spec.observation_space.shape == (6,)
        assert spec.action_space.shape == (1,)
        assert [j.joint for j in spec.joint_obs] == [
            "slider_to_cart", "cart_to_pole", "pole_to_pole2"]

    def test_action_proportional_and_clipped(self):
        spec = get_spec("inverted_double_pendulum")
        (_j, mode, full) = spec.action_to_commands(np.array([1.0]))[0]
        (_j, _m, half) = spec.action_to_commands(np.array([0.5]))[0]
        (_j, _m, over) = spec.action_to_commands(np.array([9.0]))[0]
        assert mode == "force"
        assert half == pytest.approx(full / 2)
        assert over == pytest.approx(full)

    def test_tip_kinematics(self):
        # upright: tip at max height, x = cart position
        x, h = _idp_tip(np.array([0.3, 0, 0.0, 0, 0.0, 0]))
        assert h == pytest.approx(_IDP_TIP_MAX)
        assert x == pytest.approx(0.3)
        # both segments horizontal: height 0
        _x, h = _idp_tip(np.array([0, 0, np.pi / 2, 0, 0.0, 0]))
        assert h == pytest.approx(0.0, abs=1e-9)
        # second joint folds the pole back: two segments cancel
        _x, h = _idp_tip(np.array([0, 0, np.pi / 2, 0, np.pi, 0]))
        assert h == pytest.approx(0.0, abs=1e-9)

    def test_termination_threshold(self):
        spec = get_spec("inverted_double_pendulum")
        upright = np.zeros(6, dtype=np.float32)
        assert spec.terminated_fn(upright) is False
        # 45 deg on the first hinge: tip_h = 1.2*cos(45deg) ~ 0.85 < 1.0
        tilted = np.array([0, 0, np.pi / 4, 0, 0, 0], dtype=np.float32)
        assert spec.terminated_fn(tilted) is True

    def test_reward_prefers_upright_and_still(self):
        spec = get_spec("inverted_double_pendulum")
        upright = np.zeros(6, dtype=np.float32)
        tilted = np.array([0, 0, 0.4, 0, 0.2, 0], dtype=np.float32)
        fast = np.array([0, 0, 0, 5.0, 0, 5.0], dtype=np.float32)
        a = np.zeros(1)
        assert spec.reward_fn(upright, a) > spec.reward_fn(tilted, a)
        assert spec.reward_fn(upright, a) > spec.reward_fn(fast, a)

    def test_reset_randomizes_both_hinges(self):
        spec = get_spec("inverted_double_pendulum")
        rng = np.random.default_rng(0)
        st = spec.reset_joint_state(rng)
        assert st["slider_to_cart"] == (0.0, 0.0)
        assert -0.05 <= st["cart_to_pole"][0] <= 0.05
        assert -0.05 <= st["pole_to_pole2"][0] <= 0.05


class TestHopperSpecMath:
    """Offline: hopper observation layout, actuation, health, reward."""

    def test_spaces_and_obs_layout(self):
        spec = get_spec("hopper")
        assert spec.observation_space.shape == (11,)
        assert spec.action_space.shape == (3,)
        # MuJoCo layout: 5 positions (forward slide excluded), 6 velocities.
        widths = [(j.joint, j.position, j.velocity) for j in spec.joint_obs]
        assert widths[0] == ("root_up", True, False)
        assert widths[5] == ("root_fwd", False, True)
        assert sum(j.width for j in spec.joint_obs) == 11

    def test_action_maps_three_torques(self):
        spec = get_spec("hopper")
        cmds = spec.action_to_commands(np.array([1.0, -0.5, 0.25]))
        assert [c[0] for c in cmds] == ["thigh_joint", "leg_joint",
                                        "foot_joint"]
        assert all(c[1] == "force" for c in cmds)
        assert cmds[1][2] == pytest.approx(-cmds[0][2] / 2)
        # scalar probes (used for joint discovery) must not raise
        assert len(spec.action_to_commands(0)) == 3

    def test_health_termination(self):
        spec = get_spec("hopper")
        healthy = np.zeros(11, dtype=np.float32)
        assert spec.terminated_fn(healthy) is False
        fallen = healthy.copy()
        fallen[0] = -0.6            # torso below min height
        assert spec.terminated_fn(fallen) is True
        pitched = healthy.copy()
        pitched[1] = 0.3            # beyond the pitch band
        assert spec.terminated_fn(pitched) is True

    def test_reward_rewards_forward_motion(self):
        spec = get_spec("hopper")
        still = np.zeros(11, dtype=np.float32)
        moving = still.copy()
        moving[5] = 1.5             # forward velocity
        a = np.zeros(3)
        assert spec.reward_fn(moving, a) > spec.reward_fn(still, a)
        # control cost bites
        assert spec.reward_fn(still, np.ones(3)) < spec.reward_fn(still, a)


# ---- real physics (in-process sim, headless) ------------------------------ #

pytest.importorskip("gz.sim8", reason="gz.sim8 bindings not available")


def test_hopper_passive_collapse_terminates():
    # Unactuated, the monopod must buckle and fall unhealthy within ~2 s.
    from gazebo_gymnasium_bridge.envs import make_inprocess
    env = make_inprocess("hopper", n_agents=1, seed=0)
    try:
        env.reset()
        fell = None
        for k in range(150):
            _o, _r, dones, _i = env.step(np.zeros((1, 3)))
            if dones[0]:
                fell = k
                break
        assert fell is not None and fell < 100
    finally:
        env.close()


def test_hopper_obs_finite_under_random_torques():
    from gazebo_gymnasium_bridge.envs import make_inprocess
    env = make_inprocess("hopper", n_agents=2, seed=1)
    try:
        env.reset()
        rng = np.random.default_rng(0)
        for _ in range(60):
            obs, rewards, _d, _i = env.step(rng.uniform(-1, 1, size=(2, 3)))
            assert np.isfinite(obs).all() and np.isfinite(rewards).all()
    finally:
        env.close()


def test_idp_is_unstable_and_terminates():
    # A genuinely inverted double pendulum must fall from a small tilt with no
    # actuation, within ~a second of sim time.
    from gazebo_gymnasium_bridge.envs import make_inprocess
    env = make_inprocess("inverted_double_pendulum", n_agents=1, seed=0)
    try:
        env.reset()
        fell_at = None
        for k in range(100):
            _o, _r, dones, infos = env.step(np.zeros((1, 1)))
            if dones[0]:
                fell_at = k
                assert "terminal_observation" in infos[0]
                break
        assert fell_at is not None and fell_at < 60, \
            f"passive fall should terminate quickly, got {fell_at}"
    finally:
        env.close()


def test_idp_obs_finite_and_deterministic():
    from gazebo_gymnasium_bridge.envs import make_inprocess

    def rollout(seed):
        env = make_inprocess("inverted_double_pendulum", n_agents=2, seed=seed)
        try:
            env.seed(seed)
            traj = [env.reset().copy()]
            for _ in range(15):
                obs, _r, _d, _i = env.step(np.full((2, 1), 0.3))
                assert np.isfinite(obs).all()
                traj.append(obs.copy())
            return np.array(traj)
        finally:
            env.close()

    assert np.array_equal(rollout(5), rollout(5))
