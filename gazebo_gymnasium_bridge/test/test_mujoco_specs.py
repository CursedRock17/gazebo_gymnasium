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

from gazebo_gymnasium_bridge.envs.agent_spec import _idp_tip  # noqa: E402
from gazebo_gymnasium_bridge.envs.agent_spec import _IDP_TIP_MAX  # noqa: E402
from gazebo_gymnasium_bridge.envs.agent_spec import get_spec  # noqa: E402


class TestIDPSpecMath:
    """Offline: observation layout, action map, tip kinematics, termination."""

    def test_spaces_and_layout(self):
        spec = get_spec("inverted_double_pendulum")
        assert spec.observation_space.shape == (6,)
        assert spec.action_space.shape == (1,)
        joints = [j.joint for j in spec.joint_obs]
        assert joints == ["slider_to_cart", "cart_to_pole", "pole_to_pole2"]

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
        assert [c[0] for c in cmds] == ["thigh_joint", "leg_joint", "foot_joint"]
        assert all(c[1] == "force" for c in cmds)
        assert cmds[1][2] == pytest.approx(-cmds[0][2] / 2)
        # scalar probes (used for joint discovery) must not raise
        assert len(spec.action_to_commands(0)) == 3

    def test_health_termination(self):
        spec = get_spec("hopper")
        healthy = np.zeros(11, dtype=np.float32)
        assert spec.terminated_fn(healthy) is False
        fallen = healthy.copy()
        fallen[0] = -0.6  # torso below min height
        assert spec.terminated_fn(fallen) is True
        pitched = healthy.copy()
        pitched[1] = 0.3  # beyond the pitch band
        assert spec.terminated_fn(pitched) is True

    def test_reward_rewards_forward_motion(self):
        spec = get_spec("hopper")
        still = np.zeros(11, dtype=np.float32)
        moving = still.copy()
        moving[5] = 1.5  # forward velocity
        a = np.zeros(3)
        assert spec.reward_fn(moving, a) > spec.reward_fn(still, a)
        # control cost bites
        assert spec.reward_fn(still, np.ones(3)) < spec.reward_fn(still, a)


class TestWalker2dSpecMath:
    """Offline: walker2d layout, actuation, health — the two-legged hopper."""

    def test_spaces_and_obs_layout(self):
        spec = get_spec("walker2d")
        assert spec.observation_space.shape == (17,)
        assert spec.action_space.shape == (6,)
        assert sum(j.width for j in spec.joint_obs) == 17
        # forward-slide position excluded, its velocity is obs[8]
        assert spec.joint_obs[8].joint == "root_fwd"
        assert not spec.joint_obs[8].position

    def test_action_maps_six_torques(self):
        spec = get_spec("walker2d")
        cmds = spec.action_to_commands(np.array([1, 0, 0, -1, 0, 0]))
        assert len(cmds) == 6
        assert cmds[0][2] == -cmds[3][2] != 0
        assert len(spec.action_to_commands(0)) == 6  # scalar probe safe

    def test_health_termination(self):
        spec = get_spec("walker2d")
        healthy = np.zeros(17, dtype=np.float32)
        assert spec.terminated_fn(healthy) is False
        low = healthy.copy()
        low[0] = -0.5  # z = 0.75 < 0.8
        assert spec.terminated_fn(low) is True
        pitched = healthy.copy()
        pitched[1] = 1.2
        assert spec.terminated_fn(pitched) is True


class TestHalfCheetahSpecMath:
    """Offline: cheetah layout, per-joint gears, no health termination."""

    def test_spaces_and_obs_layout(self):
        spec = get_spec("half_cheetah")
        assert spec.observation_space.shape == (17,)
        assert spec.action_space.shape == (6,)
        assert sum(j.width for j in spec.joint_obs) == 17

    def test_per_joint_gears(self):
        spec = get_spec("half_cheetah")
        cmds = spec.action_to_commands(np.ones(6))
        torques = [c[2] for c in cmds]
        assert torques == [120.0, 90.0, 60.0, 120.0, 60.0, 30.0]

    def test_no_health_termination(self):
        # MuJoCo semantics: cheetah episodes end on truncation only; the spec
        # keeps a pure numerical guard.
        spec = get_spec("half_cheetah")
        slumped = np.zeros(17, dtype=np.float32)
        slumped[0] = -0.6
        slumped[1] = 0.9
        assert spec.terminated_fn(slumped) is False
        blown_up = np.full(17, 2000.0, dtype=np.float32)
        assert spec.terminated_fn(blown_up) is True

    def test_reward_is_velocity_minus_ctrl_cost(self):
        spec = get_spec("half_cheetah")
        obs = np.zeros(17, dtype=np.float32)
        obs[8] = 2.0
        assert spec.reward_fn(obs, np.zeros(6)) == pytest.approx(2.0)
        assert spec.reward_fn(obs, np.ones(6)) == pytest.approx(2.0 - 0.6)


class TestReacherSpecMath:
    """Offline: reacher kinematics, goal-in-obs, dense reward."""

    def test_spaces_and_layout(self):
        spec = get_spec("reacher")
        assert spec.observation_space.shape == (6,)
        assert spec.action_space.shape == (2,)
        assert spec.max_episode_steps == 50

    def test_fingertip_kinematics(self):
        from gazebo_gymnasium_bridge.envs.agent_spec import _reacher_fingertip

        x, y = _reacher_fingertip(np.zeros(6))
        assert (x, y) == (pytest.approx(0.21), pytest.approx(0.0))
        x, y = _reacher_fingertip(np.array([np.pi / 2, 0, 0, 0, 0, 0]))
        assert (x, y) == (pytest.approx(0.0, abs=1e-9), pytest.approx(0.21))

    def test_reward_is_negative_distance(self):
        spec = get_spec("reacher")
        at_goal = np.array([0, 0, 0.21, 0.0, 0, 0], dtype=np.float32)
        far = np.array([0, 0, -0.14, -0.14, 0, 0], dtype=np.float32)
        a = np.zeros(2)
        assert spec.reward_fn(at_goal, a) == pytest.approx(0.0, abs=1e-6)
        assert spec.reward_fn(far, a) < spec.reward_fn(at_goal, a)

    def test_reset_randomizes_goal_within_reach(self):
        spec = get_spec("reacher")
        rng = np.random.default_rng(0)
        for _ in range(20):
            st = spec.reset_joint_state(rng)
            g = np.hypot(st["target_x"][0], st["target_y"][0])
            assert g <= 0.198 < 0.21, "goal must stay within arm reach"


# ---- real physics (in-process sim, headless) ------------------------------ #

pytest.importorskip("gz.sim8", reason="gz.sim8 bindings not available")


@pytest.mark.parametrize("agent,act_dim", [("hopper", 3), ("walker2d", 6)])
def test_locomotor_passive_collapse_terminates(agent, act_dim):
    # Unactuated, the legged robots must buckle and fall unhealthy quickly.
    from gazebo_gymnasium_bridge.envs import make_inprocess

    env = make_inprocess(agent, n_agents=1, seed=0)
    try:
        env.reset()
        fell = None
        for k in range(150):
            _o, _r, dones, _i = env.step(np.zeros((1, act_dim)))
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
        assert fell_at is not None and fell_at < 60, (
            f"passive fall should terminate quickly, got {fell_at}"
        )
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


def test_cheetah_runs_without_termination():
    # No health check: 80 random-torque steps must neither terminate nor
    # produce non-finite state.
    from gazebo_gymnasium_bridge.envs import make_inprocess

    env = make_inprocess("half_cheetah", n_agents=1, seed=3)
    try:
        env.reset()
        rng = np.random.default_rng(0)
        for _ in range(80):
            obs, rewards, dones, _i = env.step(rng.uniform(-1, 1, size=(1, 6)))
            assert not dones[0]
            assert np.isfinite(obs).all() and np.isfinite(rewards).all()
    finally:
        env.close()


def test_reacher_goal_moves_between_episodes():
    # The goal is two prismatic joints, so per-episode randomization flows
    # through the ordinary in-place reset — verify it actually moves.
    from gazebo_gymnasium_bridge.envs import make_inprocess

    env = make_inprocess("reacher", n_agents=1, seed=0)
    try:
        obs = env.reset()
        g1 = obs[0, 2:4].copy()
        for _ in range(60):  # run past the 50-step cap
            obs, _r, _d, _i = env.step(np.zeros((1, 2)))
        g2 = obs[0, 2:4].copy()
        assert not np.allclose(g1, g2), "goal should re-randomize on reset"
    finally:
        env.close()
