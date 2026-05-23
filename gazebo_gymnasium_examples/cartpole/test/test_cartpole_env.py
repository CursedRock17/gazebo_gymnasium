"""
Unit tests for CartPoleEnv.

All gz.transport and rclpy calls are mocked so these tests run without
Gazebo installed and without ROS 2 sourced (pure Python, no simulator).

Architecture notes (new pattern):
  - _on_joint_state() calls _count_physics_step() — no _joint_state_ready event
  - get_observation() reads state directly — no blocking wait
  - apply_action() just publishes — no event manipulation
  - set_default_observation() zeros state — no event
  - reset() does NOT call _advance_physics() (settle step removed; reset sends pause=True)

Run with:
    python3 -m unittest test_cartpole_env -v
"""
import os
import sys
import time
import types
import unittest
from unittest.mock import MagicMock, patch, call

import numpy as np

# Make cartpole_env importable from the scripts directory next to this test dir
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

# ---------------------------------------------------------------------------
# Stub the gz modules at import time so cartpole_env.py can be imported
# without Gazebo installed.
# ---------------------------------------------------------------------------

def _make_gz_stubs():
    gz = types.ModuleType("gz")
    transport = types.ModuleType("gz.transport13")
    msgs = types.ModuleType("gz.msgs10")
    double_mod = types.ModuleType("gz.msgs10.double_pb2")
    model_mod = types.ModuleType("gz.msgs10.model_pb2")

    class _Double:
        data = 0.0

    class _Model:
        joint = []

    transport.Node = MagicMock
    transport.AdvertiseMessageOptions = MagicMock
    double_mod.Double = _Double
    model_mod.Model = _Model

    sys.modules["gz"] = gz
    sys.modules["gz.transport13"] = transport
    sys.modules["gz.msgs10"] = msgs
    sys.modules["gz.msgs10.double_pb2"] = double_mod
    sys.modules["gz.msgs10.model_pb2"] = model_mod


_make_gz_stubs()


# ---------------------------------------------------------------------------
# Stub gazebo_gymnasium so we don't need the full library installed.
# ---------------------------------------------------------------------------

class _FakeGazeboEnv:
    """Minimal stand-in for GazeboEnv — matches the new callback-counting API."""

    def __init__(self, world_name, obs_space, act_space, steps_per_action):
        self.observation_space = obs_space
        self.action_space = act_space
        self._world_name = world_name
        self._world_control = MagicMock()
        self._current_step = 0
        self._current_episode = 0
        self._steps_per_action = steps_per_action

    def reset(self, seed=None, options=None):
        self._world_control.reset()
        self._current_step = 0
        self._current_episode += 1
        return self.set_default_observation(), self.get_info()

    def step(self, action):
        self.apply_action(action)
        self._advance_physics()
        self._current_step += 1
        obs = self.get_observation()
        return obs, self.get_reward(action), self.is_terminated(), self.is_truncated(), self.get_info()

    def _advance_physics(self, n=None):
        return True

    def _ping_world_control(self):
        pass

    def _count_physics_step(self):
        pass

    def _stop_counting(self):
        pass

    def get_info(self):
        return {}


gz_gym_mod = types.ModuleType("gazebo_gymnasium")
gz_gym_mod.GazeboEnv = _FakeGazeboEnv
sys.modules["gazebo_gymnasium"] = gz_gym_mod

# Now import the module under test
import cartpole_env as _mod  # noqa: E402
from cartpole_env import CartPoleEnv, _MAX_POLE_ANGLE, _MAX_CART_POS, _MAX_STEPS  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_env():
    """Return a CartPoleEnv with TF publishing disabled (rclpy not needed)."""
    with patch.object(_mod, "_try_init_tf", return_value=(None, None)):
        env = CartPoleEnv(steps_per_action=5)
    return env


def _make_joint_msg(slider_pos=0.0, slider_vel=0.0, pole_pos=0.0, pole_vel=0.0):
    """Build a minimal joint-state message matching what _on_joint_state expects."""
    class _Axis:
        pass

    class _Joint:
        def __init__(self, name, pos, vel):
            self.name = name
            self.axis1 = _Axis()
            self.axis1.position = pos
            self.axis1.velocity = vel

    class _Msg:
        pass

    msg = _Msg()
    msg.joint = [
        _Joint("slider_to_cart", slider_pos, slider_vel),
        _Joint("cart_to_pole", pole_pos, pole_vel),
    ]
    return msg


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestCartPoleEnvInit(unittest.TestCase):
    def test_observation_space_shape(self):
        env = _make_env()
        self.assertEqual(env.observation_space.shape, (4,))

    def test_action_space_size(self):
        env = _make_env()
        self.assertEqual(env.action_space.n, 2)

    def test_obs_space_bounds_allow_full_swing(self):
        """Obs space must accommodate pole angles beyond the 12° termination threshold."""
        env = _make_env()
        self.assertAlmostEqual(env.observation_space.high[2], np.pi, places=3)
        self.assertAlmostEqual(env.observation_space.low[2], -np.pi, places=3)

    def test_tf_disabled_when_unavailable(self):
        env = _make_env()
        self.assertIsNone(env._tf_broadcaster)


class TestJointStateCallback(unittest.TestCase):
    def test_callback_updates_cart_position(self):
        env = _make_env()
        env._on_joint_state(_make_joint_msg(slider_pos=1.23))
        self.assertAlmostEqual(env._cart_position, 1.23)

    def test_callback_updates_pole_angle(self):
        env = _make_env()
        env._on_joint_state(_make_joint_msg(pole_pos=0.15))
        self.assertAlmostEqual(env._pole_angle, 0.15)

    def test_callback_updates_cart_velocity(self):
        env = _make_env()
        env._on_joint_state(_make_joint_msg(slider_vel=0.42))
        self.assertAlmostEqual(env._cart_velocity, 0.42)

    def test_callback_updates_pole_velocity(self):
        env = _make_env()
        env._on_joint_state(_make_joint_msg(pole_vel=0.99))
        self.assertAlmostEqual(env._pole_ang_velocity, 0.99)

    def test_callback_calls_count_physics_step(self):
        """_on_joint_state must call _count_physics_step() on every message."""
        env = _make_env()
        count_mock = MagicMock()
        env._count_physics_step = count_mock
        env._on_joint_state(_make_joint_msg())
        count_mock.assert_called_once()

    def test_callback_ignores_nan_values(self):
        """NaN values in joint state must not update internal state."""
        env = _make_env()
        env._cart_position = 0.5
        env._on_joint_state(_make_joint_msg(slider_pos=float("nan"), slider_vel=0.0))
        self.assertAlmostEqual(env._cart_position, 0.5)


class TestGetObservation(unittest.TestCase):
    def test_get_observation_returns_current_state(self):
        """get_observation() reads state directly — no blocking."""
        env = _make_env()
        env._cart_position = 0.5
        env._cart_velocity = 0.1
        env._pole_angle = 0.2
        env._pole_ang_velocity = 0.3
        obs = env.get_observation()
        np.testing.assert_array_almost_equal(obs, [0.5, 0.1, 0.2, 0.3], decimal=5)

    def test_get_observation_returns_float32(self):
        env = _make_env()
        obs = env.get_observation()
        self.assertEqual(obs.dtype, np.float32)

    def test_get_observation_shape(self):
        env = _make_env()
        obs = env.get_observation()
        self.assertEqual(obs.shape, (4,))

    def test_get_observation_does_not_block(self):
        """get_observation() must return immediately — no event wait."""
        env = _make_env()
        start = time.time()
        env.get_observation()
        elapsed = time.time() - start
        self.assertLess(elapsed, 0.05)


class TestApplyAction(unittest.TestCase):
    def test_apply_action_publishes_left(self):
        env = _make_env()
        env.apply_action(0)
        env._cmd_pub.publish.assert_called_once()
        call_arg = env._cmd_pub.publish.call_args[0][0]
        self.assertLess(call_arg.data, 0)  # left → negative Y position

    def test_apply_action_publishes_right(self):
        env = _make_env()
        env.apply_action(1)
        env._cmd_pub.publish.assert_called_once()
        call_arg = env._cmd_pub.publish.call_args[0][0]
        self.assertGreater(call_arg.data, 0)  # right → positive Y position


class TestReset(unittest.TestCase):
    def test_reset_publishes_zero_command(self):
        """reset() must publish data=0.0 to centre the cart after world reset."""
        env = _make_env()
        env._cmd_pub.publish.reset_mock()
        env.reset()
        # First publish in reset() is the zero command
        first_call_arg = env._cmd_pub.publish.call_args_list[0][0][0]
        self.assertEqual(first_call_arg.data, 0.0)

    def test_reset_does_not_call_advance_physics(self):
        """reset() must NOT call _advance_physics() (settle step removed).

        WorldController.reset() now sends pause=True so Gazebo is deterministically
        paused at the initial state.  No settle step is needed — the first env.step()
        will run physics from the clean reset state.
        """
        env = _make_env()
        advance_mock = MagicMock(return_value=True)
        env._advance_physics = advance_mock
        env.reset()
        advance_mock.assert_not_called()

    def test_reset_returns_zeros_observation(self):
        env = _make_env()
        obs, info = env.reset()
        np.testing.assert_array_equal(obs, np.zeros(4, dtype=np.float32))


class TestTermination(unittest.TestCase):
    def test_not_terminated_at_origin(self):
        env = _make_env()
        env._pole_angle = 0.0
        env._cart_position = 0.0
        self.assertFalse(env.is_terminated())

    def test_terminated_when_pole_exceeds_threshold(self):
        env = _make_env()
        env._pole_angle = _MAX_POLE_ANGLE + 0.01
        self.assertTrue(env.is_terminated())

    def test_terminated_when_cart_out_of_bounds(self):
        env = _make_env()
        env._cart_position = _MAX_CART_POS + 0.01
        self.assertTrue(env.is_terminated())

    def test_not_truncated_before_max_steps(self):
        env = _make_env()
        env._current_step = _MAX_STEPS - 1
        self.assertFalse(env.is_truncated())

    def test_truncated_at_max_steps(self):
        env = _make_env()
        env._current_step = _MAX_STEPS
        self.assertTrue(env.is_truncated())


class TestDefaultObservation(unittest.TestCase):
    def test_set_default_observation_returns_zeros(self):
        env = _make_env()
        env._cart_position = 1.0
        env._pole_angle = 0.5
        obs = env.set_default_observation()
        np.testing.assert_array_equal(obs, np.zeros(4, dtype=np.float32))

    def test_set_default_observation_zeros_internal_state(self):
        env = _make_env()
        env._cart_position = 1.0
        env._cart_velocity = 2.0
        env._pole_angle = 0.5
        env._pole_ang_velocity = 0.3
        env.set_default_observation()
        self.assertEqual(env._cart_position, 0.0)
        self.assertEqual(env._cart_velocity, 0.0)
        self.assertEqual(env._pole_angle, 0.0)
        self.assertEqual(env._pole_ang_velocity, 0.0)

    def test_set_default_observation_does_not_block(self):
        """Must return immediately."""
        env = _make_env()
        start = time.time()
        env.set_default_observation()
        elapsed = time.time() - start
        self.assertLess(elapsed, 0.05)


class TestInfo(unittest.TestCase):
    def test_info_contains_gz_episode(self):
        env = _make_env()
        env._current_episode = 3
        self.assertIn("gz_episode", env.get_info())
        self.assertEqual(env.get_info()["gz_episode"], 3)

    def test_info_does_not_contain_episode_key(self):
        """SB3 reserves 'episode'; we must not use it."""
        env = _make_env()
        self.assertNotIn("episode", env.get_info())

    def test_info_contains_time_limit_truncated(self):
        env = _make_env()
        self.assertIn("TimeLimit.truncated", env.get_info())


if __name__ == "__main__":
    unittest.main()
