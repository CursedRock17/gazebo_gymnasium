"""
Unit tests for CartPoleEnv.

All gz.transport and rclpy calls are mocked so these tests run without
Gazebo installed and without ROS 2 sourced (pure Python, no simulator).

Run with:
    python3 -m unittest test_cartpole_env -v
"""
import os
import sys
import threading
import time
import types
import unittest
from unittest.mock import MagicMock, patch

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

    # Minimal stub classes
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
# Also stub gazebo_gymnasium so we don't need the full library installed.
# ---------------------------------------------------------------------------

class _FakeGazeboEnv:
    """Minimal stand-in for GazeboEnv base class."""

    def __init__(self, world_name, obs_space, act_space, steps_per_action):
        self.observation_space = obs_space
        self.action_space = act_space
        self._world_control = MagicMock()
        self._current_step = 0
        self._current_episode = 0

    def reset(self, seed=None, options=None):
        self._world_control.reset()
        self._current_step = 0
        self._current_episode += 1
        return self.set_default_observation(), self.get_info()

    def step(self, action):
        self.apply_action(action)
        self._world_control.step()
        self._current_step += 1
        obs = self.get_observation()
        return obs, self.get_reward(action), self.is_terminated(), self.is_truncated(), self.get_info()

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

    def test_callback_sets_ready_event(self):
        env = _make_env()
        env._joint_state_ready.clear()
        env._on_joint_state(_make_joint_msg())
        self.assertTrue(env._joint_state_ready.is_set())


class TestObservationSync(unittest.TestCase):
    def test_get_observation_blocks_until_event(self):
        """get_observation() must wait for _joint_state_ready before returning."""
        env = _make_env()
        env._joint_state_ready.clear()

        # Fire the callback from a background thread after a short delay
        def _fire():
            time.sleep(0.05)
            env._on_joint_state(_make_joint_msg(slider_pos=0.5, pole_pos=0.1))

        t = threading.Thread(target=_fire)
        t.start()

        obs = env.get_observation()
        t.join()

        self.assertAlmostEqual(obs[0], 0.5, places=5)  # cart_position
        self.assertAlmostEqual(obs[2], 0.1, places=5)  # pole_angle

    def test_get_observation_returns_float32(self):
        env = _make_env()
        env._joint_state_ready.set()
        obs = env.get_observation()
        self.assertEqual(obs.dtype, np.float32)

    def test_get_observation_shape(self):
        env = _make_env()
        env._joint_state_ready.set()
        obs = env.get_observation()
        self.assertEqual(obs.shape, (4,))


class TestApplyAction(unittest.TestCase):
    def test_apply_action_clears_event(self):
        """apply_action must clear the ready event before the physics step."""
        env = _make_env()
        env._joint_state_ready.set()  # pretend a previous step set it
        env.apply_action(0)
        self.assertFalse(env._joint_state_ready.is_set())

    def test_apply_action_publishes_left(self):
        env = _make_env()
        env.apply_action(0)
        # _cmd_pub is a MagicMock; just assert publish was called
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
        """reset() must zero the controller target after world_control.reset()."""
        env = _make_env()
        env._joint_state_ready.set()  # prevent blocking in settle step wait
        env.reset()
        # The last publish call after world_control.reset() should be data=0
        last_call = env._cmd_pub.publish.call_args[0][0]
        self.assertEqual(last_call.data, 0.0)

    def test_reset_triggers_settle_step(self):
        """reset() must call world_control.step() once for the settle step."""
        env = _make_env()
        env._joint_state_ready.set()
        env._world_control.step.reset_mock()
        env.reset()
        env._world_control.step.assert_called_once()

    def test_reset_returns_zeros_observation(self):
        env = _make_env()
        env._joint_state_ready.set()
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

    def test_set_default_observation_clears_event(self):
        env = _make_env()
        env._joint_state_ready.set()
        env.set_default_observation()
        self.assertFalse(env._joint_state_ready.is_set())

    def test_set_default_observation_does_not_block(self):
        """Must return immediately even when no joint-state message is coming."""
        env = _make_env()
        env._joint_state_ready.clear()
        start = time.time()
        env.set_default_observation()
        elapsed = time.time() - start
        # Should return in well under the _OBS_TIMEOUT_S window
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
