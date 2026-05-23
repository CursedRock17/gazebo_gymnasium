"""
Unit tests for GazeboEnv base class.

gz.transport13 and gz.msgs10 are mocked at the sys.modules level so these
tests run without a Gazebo installation (pure Python, no simulator required).
"""
import threading
import unittest
from unittest.mock import MagicMock, patch, call

import numpy as np
from gymnasium.spaces import Box, Discrete

# gz imports are lazy (inside WorldController.__init__), so this import works
# without Gazebo installed. Tests mock WorldController entirely.
from gazebo_gymnasium.gazebo_env import GazeboEnv

# --- Shared test fixtures ---
OBS_SPACE = Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float64)
ACT_SPACE = Discrete(2)
_DEFAULT_OBS = np.zeros(4, dtype=np.float64)


class _ConcreteEnv(GazeboEnv):
    """Minimal concrete subclass used to exercise the base class."""

    def apply_action(self, action):
        pass

    def get_observation(self):
        return _DEFAULT_OBS.copy()

    def get_reward(self, action):
        return 1.0

    def is_terminated(self):
        return False

    def is_truncated(self):
        return False

    def set_default_observation(self):
        return _DEFAULT_OBS.copy()


# --- Test cases ---

class TestGazeboEnvAbstract(unittest.TestCase):
    def test_cannot_instantiate_abstract_class(self):
        """GazeboEnv cannot be used directly — must be subclassed."""
        with self.assertRaises(TypeError):
            GazeboEnv("world", OBS_SPACE, ACT_SPACE)


@patch("gazebo_gymnasium.gazebo_env.WorldController")
class TestGazeboEnvReset(unittest.TestCase):
    def test_reset_calls_world_reset(self, mock_wc_cls):
        mock_wc_cls.return_value = MagicMock()
        env = _ConcreteEnv("world", OBS_SPACE, ACT_SPACE)
        env.reset()
        mock_wc_cls.return_value.reset.assert_called_once()

    def test_reset_returns_obs_and_info(self, mock_wc_cls):
        mock_wc_cls.return_value = MagicMock()
        env = _ConcreteEnv("world", OBS_SPACE, ACT_SPACE)
        obs, info = env.reset()
        self.assertEqual(obs.shape, (4,))
        self.assertIsInstance(info, dict)

    def test_reset_increments_episode_and_clears_step(self, mock_wc_cls):
        mock_wc_cls.return_value = MagicMock()
        env = _ConcreteEnv("world", OBS_SPACE, ACT_SPACE)
        env.reset()
        # Advance step counter manually (step() would block waiting for physics)
        env._current_step = 2
        env.reset()
        self.assertEqual(env._current_episode, 2)
        self.assertEqual(env._current_step, 0)


@patch("gazebo_gymnasium.gazebo_env.WorldController")
class TestGazeboEnvStep(unittest.TestCase):
    def test_step_calls_world_controller_step(self, mock_wc_cls):
        """_advance_physics() must call WorldController.step(n) for multi-step."""
        mock_wc = MagicMock()
        mock_wc_cls.return_value = mock_wc
        env = _ConcreteEnv("world", OBS_SPACE, ACT_SPACE)
        env.reset()

        # Fire physics event from a background thread so _advance_physics() completes.
        def _fire():
            import time
            time.sleep(0.01)
            with env._phys_lock:
                env._phys_steps = env._phys_target
                env._phys_event.set()

        t = threading.Thread(target=_fire, daemon=True)
        t.start()
        env.step(0)
        t.join(timeout=2.0)

        # _advance_physics uses multi_step via WorldController.step(n)
        mock_wc.step.assert_called()

    def test_step_increments_step_count(self, mock_wc_cls):
        mock_wc_cls.return_value = MagicMock()
        env = _ConcreteEnv("world", OBS_SPACE, ACT_SPACE)
        env.reset()

        with patch.object(env, '_advance_physics', return_value=True):
            env.step(0)
            env.step(1)
        self.assertEqual(env._current_step, 2)

    def test_step_returns_five_tuple(self, mock_wc_cls):
        mock_wc_cls.return_value = MagicMock()
        env = _ConcreteEnv("world", OBS_SPACE, ACT_SPACE)
        env.reset()

        with patch.object(env, '_advance_physics', return_value=True):
            obs, reward, terminated, truncated, info = env.step(0)
        self.assertEqual(obs.shape, (4,))
        self.assertIsInstance(reward, float)
        self.assertIsInstance(terminated, bool)
        self.assertIsInstance(truncated, bool)
        self.assertIsInstance(info, dict)


@patch("gazebo_gymnasium.gazebo_env.WorldController")
class TestGazeboEnvSpaces(unittest.TestCase):
    def test_observation_space_preserved(self, mock_wc_cls):
        mock_wc_cls.return_value = MagicMock()
        env = _ConcreteEnv("world", OBS_SPACE, ACT_SPACE)
        self.assertIs(env.observation_space, OBS_SPACE)

    def test_action_space_preserved(self, mock_wc_cls):
        mock_wc_cls.return_value = MagicMock()
        env = _ConcreteEnv("world", OBS_SPACE, ACT_SPACE)
        self.assertIs(env.action_space, ACT_SPACE)

    def test_get_info_default_returns_empty_dict(self, mock_wc_cls):
        mock_wc_cls.return_value = MagicMock()
        env = _ConcreteEnv("world", OBS_SPACE, ACT_SPACE)
        self.assertEqual(env.get_info(), {})


@patch("gazebo_gymnasium.gazebo_env.WorldController")
class TestPhysicsCountingMechanism(unittest.TestCase):
    """Test the thread-safe physics step counting used by _advance_physics()."""

    def test_count_physics_step_increments_when_counting(self, mock_wc_cls):
        mock_wc_cls.return_value = MagicMock()
        env = _ConcreteEnv("world", OBS_SPACE, ACT_SPACE)
        with env._phys_lock:
            env._phys_counting = True
            env._phys_target = 5
        env._count_physics_step()
        with env._phys_lock:
            self.assertEqual(env._phys_steps, 1)

    def test_count_physics_step_ignored_when_not_counting(self, mock_wc_cls):
        mock_wc_cls.return_value = MagicMock()
        env = _ConcreteEnv("world", OBS_SPACE, ACT_SPACE)
        with env._phys_lock:
            env._phys_counting = False
            env._phys_steps = 0
        env._count_physics_step()
        with env._phys_lock:
            self.assertEqual(env._phys_steps, 0)

    def test_count_physics_step_sets_event_at_target(self, mock_wc_cls):
        mock_wc_cls.return_value = MagicMock()
        env = _ConcreteEnv("world", OBS_SPACE, ACT_SPACE)
        with env._phys_lock:
            env._phys_counting = True
            env._phys_steps = 0
            env._phys_target = 1
            env._phys_event.clear()
        env._count_physics_step()
        self.assertTrue(env._phys_event.is_set())

    def test_ping_world_control_calls_ping(self, mock_wc_cls):
        """_ping_world_control() must call WorldController.ping()."""
        mock_wc = MagicMock()
        mock_wc.ping.return_value = True
        mock_wc_cls.return_value = mock_wc

        with patch("gazebo_gymnasium.gazebo_env.time") as mock_time:
            env = _ConcreteEnv("world", OBS_SPACE, ACT_SPACE)
            env._ping_world_control()

        mock_wc.ping.assert_called_once()

    def test_ping_world_control_raises_on_failure(self, mock_wc_cls):
        """_ping_world_control() must raise RuntimeError if ping fails."""
        mock_wc = MagicMock()
        mock_wc.ping.return_value = False
        mock_wc_cls.return_value = mock_wc

        with patch("gazebo_gymnasium.gazebo_env.time"):
            env = _ConcreteEnv("world", OBS_SPACE, ACT_SPACE)
            with self.assertRaises(RuntimeError):
                env._ping_world_control()


if __name__ == "__main__":
    unittest.main()
