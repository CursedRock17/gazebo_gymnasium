"""
Unit tests for GazeboEnv base class.
Uses a mocked WorldController — no running Gazebo instance required.
"""
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
from gymnasium.spaces import Box, Discrete

from gazebo_gymnasium.gazebo_env import GazeboEnv

OBS_SPACE = Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float64)
ACT_SPACE = Discrete(2)
_DEFAULT_OBS = np.zeros(4, dtype=np.float64)


class _ConcreteEnv(GazeboEnv):
    """Minimal concrete subclass used for testing the base class."""
    def apply_action(self, action): pass
    def get_observation(self): return _DEFAULT_OBS.copy()
    def get_reward(self, action): return 1.0
    def is_terminated(self): return False
    def is_truncated(self): return False
    def set_default_observation(self): return _DEFAULT_OBS.copy()


class TestGazeboEnvAbstract(unittest.TestCase):
    def test_cannot_instantiate_abstract_class(self):
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

    def test_reset_increments_episode_and_resets_step(self, mock_wc_cls):
        mock_wc_cls.return_value = MagicMock()
        env = _ConcreteEnv("world", OBS_SPACE, ACT_SPACE)
        env.reset()
        env.step(0)
        env.step(1)
        env.reset()
        self.assertEqual(env._current_episode, 2)
        self.assertEqual(env._current_step, 0)


@patch("gazebo_gymnasium.gazebo_env.WorldController")
class TestGazeboEnvStep(unittest.TestCase):
    def test_step_calls_world_step(self, mock_wc_cls):
        mock_wc = MagicMock()
        mock_wc_cls.return_value = mock_wc
        env = _ConcreteEnv("world", OBS_SPACE, ACT_SPACE)
        env.reset()
        env.step(0)
        mock_wc.step.assert_called_once()

    def test_step_increments_step_count(self, mock_wc_cls):
        mock_wc_cls.return_value = MagicMock()
        env = _ConcreteEnv("world", OBS_SPACE, ACT_SPACE)
        env.reset()
        env.step(0)
        env.step(1)
        self.assertEqual(env._current_step, 2)

    def test_step_returns_five_tuple(self, mock_wc_cls):
        mock_wc_cls.return_value = MagicMock()
        env = _ConcreteEnv("world", OBS_SPACE, ACT_SPACE)
        env.reset()
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


if __name__ == "__main__":
    unittest.main()
