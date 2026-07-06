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

"""Offline contract tests for the generalized MultiAgentGazeboVecEnv.

Transport is mocked and the recreate/wait helpers are short-circuited, so the
SB3 VecEnv contract (reset/step shapes, group auto-reset, reward/termination
from the AgentSpec) is exercised without a running Gazebo.

Run with: pytest gazebo_gymnasium_bridge/test/test_multi_agent_env.py -v
"""

from pathlib import Path
import sys
from unittest.mock import MagicMock

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # gazebo_gymnasium_bridge/

gz_transport13 = pytest.importorskip("gz.transport13",
                                     reason="gz bindings not available")
pytest.importorskip("gz.msgs10", reason="gz bindings not available")


@pytest.fixture
def cartpole_multi(monkeypatch):
    """Build a mocked MultiAgentGazeboVecEnv(cartpole spec, 3 agents).

    Transport is mocked and recreate/wait short-circuited so step/reset are
    deterministic.
    """
    monkeypatch.setattr(gz_transport13, "Node", MagicMock(name="Node"))
    from gazebo_gymnasium_bridge.backend.nodes import world_control
    monkeypatch.setattr(world_control.WorldController, "_send",
                        lambda self, *a, **kw: True)

    from gazebo_gymnasium_bridge.envs import make_multi
    env = make_multi("cartpole", n_agents=3, world_name="t",
                     reset_timeout=0.01, step_timeout=0.01)
    # Make reset/step non-blocking + side-effect free.
    monkeypatch.setattr(env, "_recreate_all_agents", lambda: None)
    monkeypatch.setattr(env, "_wait_all_states", lambda timeout: True)
    # Default to an upright (non-terminal) state for all agents.
    env._latest_states[:] = np.zeros((3, 4), dtype=np.float32)
    return env


class TestVecEnvContract:
    def test_spaces(self, cartpole_multi):
        import gymnasium as gym
        assert cartpole_multi.num_envs == 3
        assert cartpole_multi.observation_space.shape == (4,)
        assert isinstance(cartpole_multi.action_space, gym.spaces.Discrete)
        assert cartpole_multi.action_space.n == 2

    def test_reset_shape(self, cartpole_multi):
        obs = cartpole_multi.reset()
        assert isinstance(obs, np.ndarray)
        assert obs.shape == (3, 4)
        assert obs.dtype == np.float32

    def test_step_tuple_shapes(self, cartpole_multi):
        cartpole_multi.reset()
        cartpole_multi.step_async(np.zeros(3, dtype=int))
        obs, rewards, dones, infos = cartpole_multi.step_wait()
        assert obs.shape == (3, 4)
        assert rewards.shape == (3,)
        assert dones.shape == (3,)
        assert len(infos) == 3 and all(isinstance(d, dict) for d in infos)

    def test_alive_agents_get_unit_reward(self, cartpole_multi):
        cartpole_multi.reset()
        cartpole_multi._latest_states[:] = 0.0  # upright -> not terminal
        cartpole_multi.step_async(np.zeros(3, dtype=int))
        _obs, rewards, dones, _infos = cartpole_multi.step_wait()
        np.testing.assert_allclose(rewards, [1.0, 1.0, 1.0])
        assert not dones.any()

    def test_group_reset_when_all_fallen(self, cartpole_multi):
        cartpole_multi.reset()
        # All poles past the 0.20944 threshold -> all terminate this step ->
        # group reset fires, dones all True, terminal_observation stashed.
        cartpole_multi._latest_states[:] = np.array(
            [0.0, 0.0, 0.5, 0.0], dtype=np.float32)
        cartpole_multi.step_async(np.zeros(3, dtype=int))
        _obs, _rewards, dones, infos = cartpole_multi.step_wait()
        assert dones.all()
        assert all("terminal_observation" in d for d in infos)

    def test_truncation_marks_timelimit(self, cartpole_multi):
        cartpole_multi.max_episode_steps = 3
        cartpole_multi.reset()
        cartpole_multi._latest_states[:] = 0.0  # never naturally terminates
        for _ in range(3):
            cartpole_multi.step_async(np.zeros(3, dtype=int))
            _o, _r, dones, infos = cartpole_multi.step_wait()
        assert dones.all()
        assert all(d.get("TimeLimit.truncated") for d in infos)


class TestSB3:
    def test_ppo_accepts_generalized_env(self, cartpole_multi):
        sb3 = pytest.importorskip("stable_baselines3")
        cartpole_multi._latest_states[:] = 0.0
        model = sb3.PPO("MlpPolicy", cartpole_multi, n_steps=4,
                        batch_size=4, n_epochs=1, verbose=0)
        model.learn(total_timesteps=12)


class TestBackwardCompat:
    def test_multicartpole_is_generalized(self, monkeypatch):
        monkeypatch.setattr(gz_transport13, "Node", MagicMock(name="Node"))
        from gazebo_gymnasium_bridge.backend.nodes import world_control
        monkeypatch.setattr(world_control.WorldController, "_send",
                            lambda self, *a, **kw: True)
        from gazebo_gymnasium_bridge.envs import (
            MultiCartPoleVecEnv, MultiAgentGazeboVecEnv)
        env = MultiCartPoleVecEnv(n_agents=2, world_name="t")
        assert isinstance(env, MultiAgentGazeboVecEnv)
        assert env.num_envs == 2
        assert env.spec.name == "cartpole"
