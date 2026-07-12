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

"""Tests for the native gymnasium.vector.VectorEnv (make_vec) surface."""

from pathlib import Path
import sys

import gymnasium as gym
import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import gazebo_gymnasium_bridge  # noqa: F401,E402  (registers ids + vector ep)


def test_vector_entry_point_registered():
    # No gz needed: the spec should carry a vector_entry_point so make_vec
    # returns our efficient N-in-one env, not SyncVectorEnv over N sims.
    spec = gym.spec("GazeboCartPole-v0")
    assert spec.vector_entry_point is not None


def _make(num_envs=4):
    return gym.make_vec("GazeboCartPole-v0", num_envs=num_envs,
                        vectorization_mode="vector_entry_point")


def test_make_vec_returns_native_env():
    pytest.importorskip("gz.sim8", reason="gz.sim8 bindings not available")
    from gazebo_gymnasium_bridge.envs import GazeboVectorEnv
    venv = _make(4)
    try:
        assert isinstance(venv.unwrapped, GazeboVectorEnv)
        assert venv.unwrapped.num_envs == 4
        assert venv.observation_space.shape == (4, 4)
    finally:
        venv.close()


def test_reset_step_api_and_spaces():
    pytest.importorskip("gz.sim8", reason="gz.sim8 bindings not available")
    venv = _make(4)
    try:
        obs, info = venv.reset(seed=0)
        assert venv.observation_space.contains(obs)
        assert isinstance(info, dict)
        obs, rew, term, trunc, infos = venv.step(
            venv.action_space.sample())
        assert obs.shape == (4, 4)
        assert rew.shape == (4,) and rew.dtype == np.float64
        assert term.shape == (4,) and term.dtype == bool
        assert trunc.shape == (4,) and trunc.dtype == bool
        assert isinstance(infos, dict)
    finally:
        venv.close()


def test_same_step_autoreset_carries_final_obs():
    pytest.importorskip("gz.sim8", reason="gz.sim8 bindings not available")
    venv = _make(4)
    try:
        venv.reset(seed=0)
        saw_final = False
        for _ in range(120):
            obs, _r, _t, _tr, infos = venv.step(np.ones(4, dtype=int))
            if "final_obs" in infos:
                saw_final = True
                assert "_final_obs" in infos  # boolean mask per convention
                for i in np.nonzero(infos["_final_obs"])[0]:
                    # SAME_STEP: the returned obs is already the reset (upright)
                    assert abs(obs[i][2]) < 0.21
        assert saw_final, "autoreset should surface final_obs"
    finally:
        venv.close()
