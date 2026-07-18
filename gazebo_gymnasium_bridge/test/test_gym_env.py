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

"""Gymnasium API-compliance tests for the registered single-agent env."""

from pathlib import Path
import sys

import gymnasium as gym
import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import gazebo_gymnasium_bridge  # noqa: F401,E402  (registers the env ids)


def test_env_id_registered():
    # Registration must NOT require the native gz bindings.
    assert "GazeboCartPole-v0" in gym.envs.registration.registry
    assert "GazeboCartPoleContinuous-v0" in gym.envs.registration.registry


def test_gym_make_step_reset_api():
    pytest.importorskip("gz.sim8", reason="gz.sim8 bindings not available")
    env = gym.make("GazeboCartPole-v0")
    try:
        obs, info = env.reset(seed=0)
        assert env.observation_space.contains(obs)
        assert isinstance(info, dict)
        obs, reward, terminated, truncated, info = env.step(
            env.action_space.sample())
        assert env.observation_space.contains(obs)
        assert np.isscalar(reward) or np.ndim(reward) == 0
        assert isinstance(terminated, bool) and isinstance(truncated, bool)
    finally:
        env.close()


@pytest.mark.parametrize("env_id", ["GazeboCartPole-v0",
                                    "GazeboCartPoleContinuous-v0",
                                    "GazeboInvertedDoublePendulum-v0",
                                    "GazeboHopper-v0"])
def test_passes_gymnasium_env_checker(env_id):
    pytest.importorskip("gz.sim8", reason="gz.sim8 bindings not available")
    from gymnasium.utils.env_checker import check_env
    env = gym.make(env_id)
    try:
        check_env(env.unwrapped, skip_render_check=True)
    finally:
        env.close()


def test_terminates_and_needs_reset():
    # No autoreset at the single-env layer: on termination the env reports
    # terminated=True and the caller must reset (standard gymnasium.Env).
    pytest.importorskip("gz.sim8", reason="gz.sim8 bindings not available")
    env = gym.make("GazeboCartPole-v0").unwrapped
    try:
        env.reset(seed=1)
        ended = False
        for _ in range(600):
            _o, _r, terminated, truncated, _i = env.step(1)  # push one way
            if terminated or truncated:
                ended = True
                break
        assert ended, "constant push should end the episode"
    finally:
        env.close()


def test_sb3_trains_on_registered_env():
    pytest.importorskip("gz.sim8", reason="gz.sim8 bindings not available")
    sb3 = pytest.importorskip("stable_baselines3")
    from stable_baselines3.common.env_util import make_vec_env
    vec = make_vec_env("GazeboCartPole-v0", n_envs=1)
    try:
        sb3.PPO("MlpPolicy", vec, n_steps=16, batch_size=16, n_epochs=1,
                verbose=0).learn(total_timesteps=64)
    finally:
        vec.close()


def test_sac_trains_on_continuous_env():
    # Off-policy continuous-control algorithms need a Box action space; the
    # continuous cartpole spec is their entry point.
    pytest.importorskip("gz.sim8", reason="gz.sim8 bindings not available")
    sb3 = pytest.importorskip("stable_baselines3")
    from gazebo_gymnasium_bridge.envs import make_inprocess
    env = make_inprocess("cartpole_continuous", n_agents=1)
    try:
        model = sb3.SAC("MlpPolicy", env, learning_starts=16, batch_size=32,
                        buffer_size=1000, verbose=0)
        model.learn(total_timesteps=64)
    finally:
        env.close()
