"""
Unit tests for FixtureEnv base class.

gz.sim8 is mocked at the sys.modules level so these tests run without a
Gazebo installation (pure Python, no simulator required).

Architecture notes:
  - TestFixture has no on_configure — entity setup fires on the FIRST pre_update.
  - _configured flag gates the configure(ecm) call to happen exactly once.
  - server.run(True, N, False) fires pre_update + post_update N times.
  - reset() sets _do_reset=True, runs 1 step, then returns set_default_observation().

Run with:
    python3 -m unittest test_fixture_env -v
"""
import sys
import types
import unittest
from unittest.mock import MagicMock, call

import numpy as np
from gymnasium.spaces import Box, Discrete

# ---------------------------------------------------------------------------
# Stub gz.sim8 so we can import fixture_env without Gazebo.
# ---------------------------------------------------------------------------

class _FakeInfo:
    def __init__(self, paused=False):
        self.paused = paused
        self.iterations = 1


class _FakeServer:
    """Fires pre_update / post_update callbacks synchronously for N iterations."""

    def __init__(self, fixture):
        self._fixture = fixture
        self.run_calls = []

    def run(self, blocking, iterations, paused):
        self.run_calls.append((blocking, iterations, paused))
        info = _FakeInfo(paused=paused)
        ecm = MagicMock()
        for _ in range(iterations):
            if self._fixture._pre_cb:
                self._fixture._pre_cb(info, ecm)
            if self._fixture._post_cb:
                self._fixture._post_cb(info, ecm)
        return True


class _FakeTestFixture:
    def __init__(self, sdf_path):
        self.sdf_path = sdf_path
        self._pre_cb = None
        self._post_cb = None
        self._server = _FakeServer(self)

    def on_pre_update(self, cb):
        self._pre_cb = cb

    def on_post_update(self, cb):
        self._post_cb = cb

    def finalize(self):
        pass

    def server(self):
        return self._server


gz_pkg = types.ModuleType("gz")
gz_sim8_pkg = types.ModuleType("gz.sim8")
gz_sim8_pkg.TestFixture = _FakeTestFixture
sys.modules["gz"] = gz_pkg
sys.modules["gz.sim8"] = gz_sim8_pkg

# Now import the module under test.
from gazebo_gymnasium.fixture_env import FixtureEnv  # noqa: E402

# ---------------------------------------------------------------------------
# Minimal concrete subclass for testing.
# ---------------------------------------------------------------------------

OBS_SPACE = Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
ACT_SPACE = Discrete(2)
_DEFAULT_OBS = np.zeros(4, dtype=np.float32)


class _ConcreteEnv(FixtureEnv):
    def __init__(self, **kwargs):
        self.configure_calls = 0
        self.apply_action_calls = []
        self.apply_reset_calls = 0
        self.read_obs_calls = 0
        super().__init__("fake.sdf", OBS_SPACE, ACT_SPACE, **kwargs)

    def configure(self, ecm):
        self.configure_calls += 1

    def apply_action_to_ecm(self, ecm, action):
        self.apply_action_calls.append(action)

    def apply_reset(self, ecm):
        self.apply_reset_calls += 1

    def read_observation(self, ecm) -> np.ndarray:
        self.read_obs_calls += 1
        return _DEFAULT_OBS.copy()

    def get_reward(self, action) -> float:
        return 1.0

    def is_terminated(self) -> bool:
        return False

    def is_truncated(self) -> bool:
        return False

    def set_default_observation(self) -> np.ndarray:
        return _DEFAULT_OBS.copy()


def _make_env(**kwargs):
    return _ConcreteEnv(**kwargs)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestFixtureEnvInit(unittest.TestCase):
    def test_observation_space_preserved(self):
        env = _make_env()
        self.assertIs(env.observation_space, OBS_SPACE)

    def test_action_space_preserved(self):
        env = _make_env()
        self.assertIs(env.action_space, ACT_SPACE)

    def test_steps_per_action_default(self):
        env = _make_env()
        self.assertEqual(env._steps_per_action, 10)

    def test_steps_per_action_custom(self):
        env = _make_env(steps_per_action=5)
        self.assertEqual(env._steps_per_action, 5)

    def test_initial_step_count_is_zero(self):
        env = _make_env()
        self.assertEqual(env._current_step, 0)

    def test_initial_episode_count_is_zero(self):
        env = _make_env()
        self.assertEqual(env._current_episode, 0)


class TestFixtureEnvConfigure(unittest.TestCase):
    def test_configure_called_once_on_init(self):
        """configure(ecm) must be called exactly once — on the first pre_update."""
        env = _make_env()
        self.assertEqual(env.configure_calls, 1)

    def test_configure_not_called_again_on_step(self):
        """Step calls must not re-trigger configure."""
        env = _make_env()
        env.step(0)
        env.step(1)
        self.assertEqual(env.configure_calls, 1)

    def test_configured_flag_true_after_init(self):
        env = _make_env()
        self.assertTrue(env._configured)

    def test_no_action_applied_during_configure_step(self):
        """The initial configure step must not call apply_action_to_ecm."""
        env = _make_env()
        # __init__ runs 1 step for configure; no action should have been applied.
        self.assertEqual(len(env.apply_action_calls), 0)


class TestFixtureEnvStep(unittest.TestCase):
    def test_step_increments_step_count(self):
        env = _make_env()
        env.step(0)
        self.assertEqual(env._current_step, 1)
        env.step(1)
        self.assertEqual(env._current_step, 2)

    def test_step_applies_action(self):
        env = _make_env()
        env.step(0)
        self.assertIn(0, env.apply_action_calls)

    def test_step_calls_server_run_with_steps_per_action(self):
        env = _make_env(steps_per_action=7)
        server = env._server
        before = len(server.run_calls)
        env.step(0)
        call_args = server.run_calls[before]
        self.assertEqual(call_args, (True, 7, False))

    def test_step_returns_five_tuple(self):
        env = _make_env()
        result = env.step(0)
        self.assertEqual(len(result), 5)
        obs, reward, terminated, truncated, info = result
        self.assertEqual(obs.shape, (4,))
        self.assertIsInstance(reward, float)
        self.assertIsInstance(terminated, bool)
        self.assertIsInstance(truncated, bool)
        self.assertIsInstance(info, dict)

    def test_step_reads_observation_after_physics(self):
        env = _make_env()
        before = env.read_obs_calls
        env.step(0)
        # steps_per_action=10 means 10 post_update callbacks → 10 read_observation calls
        self.assertGreater(env.read_obs_calls, before)


class TestFixtureEnvReset(unittest.TestCase):
    def test_reset_returns_obs_and_info(self):
        env = _make_env()
        result = env.reset()
        self.assertEqual(len(result), 2)
        obs, info = result
        self.assertEqual(obs.shape, (4,))
        self.assertIsInstance(info, dict)

    def test_reset_returns_default_observation(self):
        env = _make_env()
        obs, _ = env.reset()
        np.testing.assert_array_equal(obs, _DEFAULT_OBS)

    def test_reset_clears_step_count(self):
        env = _make_env()
        env.step(0)
        env.step(1)
        env.reset()
        self.assertEqual(env._current_step, 0)

    def test_reset_increments_episode(self):
        env = _make_env()
        env.reset()
        env.reset()
        self.assertEqual(env._current_episode, 2)

    def test_reset_calls_apply_reset(self):
        env = _make_env()
        env.reset()
        self.assertEqual(env.apply_reset_calls, 1)

    def test_reset_clears_current_action(self):
        env = _make_env()
        env.step(0)
        env.reset()
        self.assertIsNone(env._current_action)


class TestPreUpdateDispatch(unittest.TestCase):
    def test_pre_update_skips_action_when_paused(self):
        """If info.paused, no action should be applied."""
        env = _make_env()
        env._current_action = 1
        info = _FakeInfo(paused=True)
        ecm = MagicMock()
        before = len(env.apply_action_calls)
        env._on_pre_update(info, ecm)
        self.assertEqual(len(env.apply_action_calls), before)

    def test_pre_update_applies_reset_before_action(self):
        """_do_reset=True takes priority over _current_action."""
        env = _make_env()
        env._do_reset = True
        env._current_action = 0
        info = _FakeInfo(paused=False)
        ecm = MagicMock()
        before_reset = env.apply_reset_calls
        before_action = len(env.apply_action_calls)
        env._on_pre_update(info, ecm)
        self.assertEqual(env.apply_reset_calls, before_reset + 1)
        self.assertEqual(len(env.apply_action_calls), before_action)  # not called

    def test_pre_update_clears_do_reset_flag(self):
        env = _make_env()
        env._do_reset = True
        info = _FakeInfo(paused=False)
        env._on_pre_update(info, MagicMock())
        self.assertFalse(env._do_reset)

    def test_pre_update_skips_action_when_none(self):
        """If _current_action is None (e.g. during reset), no action is applied."""
        env = _make_env()
        env._current_action = None
        info = _FakeInfo(paused=False)
        before = len(env.apply_action_calls)
        env._on_pre_update(info, MagicMock())
        self.assertEqual(len(env.apply_action_calls), before)


class TestPostUpdateDispatch(unittest.TestCase):
    def test_post_update_reads_obs_when_configured(self):
        env = _make_env()
        before = env.read_obs_calls
        env._on_post_update(_FakeInfo(paused=False), MagicMock())
        self.assertEqual(env.read_obs_calls, before + 1)

    def test_post_update_skips_when_paused(self):
        env = _make_env()
        before = env.read_obs_calls
        env._on_post_update(_FakeInfo(paused=True), MagicMock())
        self.assertEqual(env.read_obs_calls, before)

    def test_post_update_skips_when_not_configured(self):
        env = _make_env()
        env._configured = False
        before = env.read_obs_calls
        env._on_post_update(_FakeInfo(paused=False), MagicMock())
        self.assertEqual(env.read_obs_calls, before)


class TestGetInfo(unittest.TestCase):
    def test_default_get_info_returns_dict(self):
        env = _make_env()
        info = env.get_info()
        self.assertIsInstance(info, dict)


if __name__ == "__main__":
    unittest.main()
