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

"""Offline contract tests for HarnessVecEnv (batched-harness client).

Transport is mocked and the frame wait is short-circuited, so the SB3 VecEnv
contract (reset/step shapes, group auto-reset, reward/termination) is exercised
without a running harness. The live transport shape is separately proven by
test_harness_plugin.py.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

gz_transport13 = pytest.importorskip("gz.transport13",
                                     reason="gz bindings not available")
pytest.importorskip("gz.msgs10", reason="gz bindings not available")


@pytest.fixture
def harness_env(monkeypatch):
    monkeypatch.setattr(gz_transport13, "Node", MagicMock(name="Node"))
    from gazebo_gymnasium_bridge.envs import make_harness
    env = make_harness("cartpole", n_agents=3, world_name="t",
                       reset_timeout=0.01, step_timeout=0.01)
    # non-blocking frame waits; obs is whatever we seed.
    monkeypatch.setattr(env, "_wait_frame", lambda timeout: True)
    monkeypatch.setattr(env._obs_event, "wait", lambda timeout=None: True)
    env._latest_obs[:] = 0.0
    return env


class TestVecEnvContract:
    def test_spaces(self, harness_env):
        import gymnasium as gym
        assert harness_env.num_envs == 3
        assert harness_env.observation_space.shape == (4,)
        assert isinstance(harness_env.action_space, gym.spaces.Discrete)

    def test_reset_shape(self, harness_env):
        obs = harness_env.reset()
        assert obs.shape == (3, 4) and obs.dtype == np.float32

    def test_step_shapes(self, harness_env):
        harness_env.reset()
        harness_env.step_async(np.zeros(3, dtype=int))
        obs, rewards, dones, infos = harness_env.step_wait()
        assert obs.shape == (3, 4)
        assert rewards.shape == (3,) and dones.shape == (3,)
        assert len(infos) == 3

    def test_alive_reward(self, harness_env):
        harness_env.reset()
        harness_env._latest_obs[:] = 0.0
        harness_env.step_async(np.zeros(3, dtype=int))
        _o, rewards, dones, _i = harness_env.step_wait()
        np.testing.assert_allclose(rewards, [1.0, 1.0, 1.0])
        assert not dones.any()

    def test_group_reset_on_all_fallen(self, harness_env):
        harness_env.reset()
        harness_env._latest_obs[:] = np.array([0, 0, 0.5, 0], dtype=np.float32)
        harness_env.step_async(np.zeros(3, dtype=int))
        _o, _r, dones, infos = harness_env.step_wait()
        assert dones.all()
        assert all("terminal_observation" in d for d in infos)

    def test_ppo_accepts_harness_env(self, harness_env):
        sb3 = pytest.importorskip("stable_baselines3")
        harness_env._latest_obs[:] = 0.0
        model = sb3.PPO("MlpPolicy", harness_env, n_steps=4, batch_size=4,
                        n_epochs=1, verbose=0)
        model.learn(total_timesteps=12)


def test_step_async_flattens_actions(monkeypatch):
    monkeypatch.setattr(gz_transport13, "Node", MagicMock(name="Node"))
    from gazebo_gymnasium_bridge.envs import make_harness
    env = make_harness("cartpole", n_agents=4, world_name="t")
    published = {}
    env._act_pub.publish = lambda m: published.setdefault("data", list(m.data))
    env.step_async(np.array([1, 0, 1, 0]))
    assert published["data"] == [1.0, 0.0, 1.0, 0.0]
