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
"""Ecosystem wrapper compatibility (regression guard).

The cartpole observation has unbounded velocity components, so users will
normalize. These confirm the standard wrappers drive our envs unchanged:
SB3 ``VecNormalize`` (+ PPO), Gymnasium vector ``RecordEpisodeStatistics``,
and the single-env ``NormalizeObservation``.
"""

from pathlib import Path
import sys

import gymnasium as gym
import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import gazebo_gymnasium_bridge  # noqa: F401,E402  (registers ids)


def test_sb3_vecnormalize_and_ppo():
    pytest.importorskip("gz.sim8", reason="gz.sim8 bindings not available")
    sb3 = pytest.importorskip("stable_baselines3")
    from stable_baselines3.common.vec_env import VecNormalize

    from gazebo_gymnasium_bridge.envs import make_inprocess

    venv = VecNormalize(make_inprocess("cartpole", n_agents=4), norm_obs=True, norm_reward=True)
    try:
        obs = venv.reset()
        for _ in range(30):
            obs, _r, _d, _i = venv.step(np.random.randint(0, 2, size=4))
        assert np.isfinite(obs).all()
        sb3.PPO("MlpPolicy", venv, n_steps=16, batch_size=16, n_epochs=1, verbose=0).learn(
            total_timesteps=64
        )
    finally:
        venv.close()


def test_gym_vector_record_episode_statistics():
    pytest.importorskip("gz.sim8", reason="gz.sim8 bindings not available")
    from gymnasium.wrappers.vector import RecordEpisodeStatistics

    venv = RecordEpisodeStatistics(
        gym.make_vec("GazeboCartPole-v0", num_envs=4, vectorization_mode="vector_entry_point")
    )
    try:
        venv.reset(seed=0)
        seen = False
        for _ in range(120):
            _o, _r, _t, _tr, infos = venv.step(np.ones(4, dtype=int))
            if "episode" in infos:
                seen = True
        assert seen, "RecordEpisodeStatistics should surface episode stats"
    finally:
        venv.close()


def test_gym_single_normalize_observation():
    pytest.importorskip("gz.sim8", reason="gz.sim8 bindings not available")
    env = gym.wrappers.NormalizeObservation(gym.make("GazeboCartPole-v0"))
    try:
        obs, _ = env.reset(seed=0)
        for _ in range(20):
            obs, *_ = env.step(env.action_space.sample())
        assert obs.shape == (4,)
    finally:
        env.close()
