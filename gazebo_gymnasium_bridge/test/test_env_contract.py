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

"""
Verify GazeboCartPoleEnv satisfies the canonical Gymnasium env contract without
needing a running Gazebo. The env normally communicates with the sync-gate
plugin over gz-transport — this test stubs the transport so step()/reset()
return immediately with mocked sensor values.

Run with: pytest gazebo_gymnasium_bridge/test/test_env_contract.py -v
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent.parent
# Source layout for the bridge is `gazebo_gymnasium_bridge/gazebo_gymnasium_bridge/`,
# so the outer dir needs to be on sys.path for `import gazebo_gymnasium_bridge.*`.
BRIDGE_SRC_DIR = PROJECT_ROOT / "gazebo_gymnasium_bridge"
sys.path.insert(0, str(BRIDGE_SRC_DIR))


# Skip up-front if gz native deps aren't importable.
gz_transport13 = pytest.importorskip(
    "gz.transport13",
    reason="gz-transport python bindings not available in this environment",
)
pytest.importorskip(
    "gz.msgs10",
    reason="gz-msgs python bindings not available in this environment",
)


@pytest.fixture
def cartpole_env(monkeypatch):
    """Construct a GazeboCartPoleEnv with transport mocked.

    Strategy:
      1. Mock `gz.transport13.Node` so neither the publisher advertise nor the
         subscribe call actually open a socket.
      2. Short-circuit `WorldController._send` so reset's service request
         returns success immediately.
      3. Pre-load the sensor cache and pre-set the state event so reset() and
         step() don't wait on a real /env/state callback.
    """
    monkeypatch.setattr(gz_transport13, "Node", MagicMock(name="Node"))

    from gazebo_gymnasium_bridge.backend.nodes import world_control
    monkeypatch.setattr(
        world_control.WorldController, "_send",
        lambda self, request, attempt_timeout_ms=500, max_attempts=60: True,
    )

    from gazebo_gymnasium_bridge.envs import GazeboCartPoleEnv

    env = GazeboCartPoleEnv(
        world_name="test_world",
        max_episode_steps=500,
        reset_timeout=0.01,
        step_timeout=0.01,
    )
    # Pre-set the event so reset/step don't block on a real callback.
    env._state_event.set()
    return env


class TestEnvContract:
    """Standard gym.Env contract checks."""

    def test_has_observation_space(self, cartpole_env):
        import gymnasium as gym
        assert cartpole_env.observation_space is not None
        assert isinstance(cartpole_env.observation_space, gym.spaces.Box)
        assert cartpole_env.observation_space.shape == (4,)

    def test_has_action_space(self, cartpole_env):
        import gymnasium as gym
        assert cartpole_env.action_space is not None
        assert isinstance(cartpole_env.action_space, gym.spaces.Discrete)
        assert cartpole_env.action_space.n == 2

    def test_reset_returns_obs_info_tuple(self, cartpole_env):
        import numpy as np
        result = cartpole_env.reset()
        assert isinstance(result, tuple) and len(result) == 2
        obs, info = result
        assert isinstance(obs, np.ndarray)
        assert obs.shape == (4,)
        assert obs.dtype == np.float32
        assert isinstance(info, dict)

    def test_step_returns_five_tuple(self, cartpole_env):
        import numpy as np
        cartpole_env.reset()
        # Keep _state_event pre-set so step() doesn't block on the real callback.
        cartpole_env._state_event.set()
        result = cartpole_env.step(0)
        assert isinstance(result, tuple) and len(result) == 5
        obs, reward, terminated, truncated, info = result
        assert isinstance(obs, np.ndarray)
        assert obs.shape == (4,) and obs.dtype == np.float32
        assert isinstance(reward, (int, float))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_random_actions_dont_explode(self, cartpole_env):
        cartpole_env.reset()
        for _ in range(5):
            cartpole_env._state_event.set()
            action = cartpole_env.action_space.sample()
            obs, _r, terminated, truncated, _i = cartpole_env.step(action)
            assert obs.shape == (4,)
            if terminated or truncated:
                cartpole_env._state_event.set()
                cartpole_env.reset()

    def test_terminates_on_pole_threshold(self, cartpole_env):
        """Termination logic lives in the env now (not the plugin) — verify
        that a stale-feeling sensor value past the threshold terminates the
        episode."""
        cartpole_env.reset()
        # Simulate the plugin publishing a state with pole_angle past the
        # 0.20944 rad threshold.
        cartpole_env._latest_state = [0.0, 0.0, 0.30, 0.0]
        cartpole_env._state_event.set()
        _obs, _r, terminated, _t, _i = cartpole_env.step(0)
        assert terminated is True

    def test_terminates_on_cart_bounds(self, cartpole_env):
        cartpole_env.reset()
        cartpole_env._latest_state = [3.0, 0.0, 0.0, 0.0]
        cartpole_env._state_event.set()
        _obs, _r, terminated, _t, _i = cartpole_env.step(0)
        assert terminated is True

    def test_passes_sb3_env_checker(self, cartpole_env):
        """SB3's env_checker — broader contract validation."""
        sb3_env_checker = pytest.importorskip(
            "stable_baselines3.common.env_checker",
            reason="stable_baselines3 not available in this environment",
        )
        # check_env will call reset() and step() — keep the state event set so
        # neither blocks on a real transport callback.
        cartpole_env._state_event.set()
        # Make is_terminated False during the contract checks by zeroing the
        # cached state. We also override _on_state to keep refreshing the event
        # in case check_env runs multiple steps.
        cartpole_env._latest_state = [0.0, 0.0, 0.0, 0.0]
        original_step = cartpole_env.step

        def step_with_event(action):
            cartpole_env._state_event.set()
            return original_step(action)

        cartpole_env.step = step_with_event
        sb3_env_checker.check_env(cartpole_env, warn=True, skip_render_check=True)
