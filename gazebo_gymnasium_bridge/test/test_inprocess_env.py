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

"""Smoke tests for the in-process (TestFixture-hosted) VecEnv.

Real physics, no launch: builds the sim inside the process, checks the SB3
VecEnv contract, that episodes actually terminate, that VecMonitor can wrap it
(the .spec slot is clear), and that a PPO update runs end to end.
"""

from pathlib import Path
import sys

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

pytest.importorskip("gz.sim8", reason="gz.sim8 bindings not available")


def _env(n=2):
    from gazebo_gymnasium_bridge.envs import make_inprocess
    return make_inprocess("cartpole", n_agents=n)


def test_reset_and_step_shapes():
    env = _env(2)
    try:
        obs = env.reset()
        assert obs.shape == (2, 4) and obs.dtype == np.float32
        assert np.isfinite(obs).all()
        env.step_async(np.zeros(2, dtype=int))
        obs, rewards, dones, infos = env.step_wait()
        assert obs.shape == (2, 4)
        assert rewards.shape == (2,) and dones.shape == (2,)
        assert len(infos) == 2
    finally:
        env.close()


def test_agents_autoreset_independently():
    # Each agent that terminates resets in place ON ITS OWN STEP (SB3 same-step
    # autoreset), independently of the others: done[i]=True, a terminal_
    # observation is stashed, and the returned obs[i] is already the fresh
    # (upright) reset state.
    env = _env(3)
    try:
        obs = env.reset()
        rng = np.random.default_rng(0)
        saw_solo_done = False
        for _ in range(300):
            obs, _r, dones, infos = env.step(rng.integers(0, 2, size=3))
            for i in np.nonzero(dones)[0]:
                assert "terminal_observation" in infos[i]
                assert abs(obs[i][2]) < 0.21, "reset obs should be upright"
            if int(dones.sum()) == 1:
                saw_solo_done = True
        assert saw_solo_done, "agents should terminate/reset independently"
    finally:
        env.close()


def test_vecmonitor_wraps_inprocess():
    # regression: the AgentSpec must not occupy the gym `.spec` slot, or
    # VecMonitor(venv).spec.id blows up.
    from stable_baselines3.common.vec_env import VecMonitor
    env = VecMonitor(_env(2))
    try:
        env.reset()
        for _ in range(3):
            env.step(np.zeros(2, dtype=int))
    finally:
        env.close()


def test_ppo_learns_step_runs():
    sb3 = pytest.importorskip("stable_baselines3")
    env = _env(4)
    try:
        model = sb3.PPO("MlpPolicy", env, n_steps=16, batch_size=16,
                        n_epochs=1, verbose=0)
        model.learn(total_timesteps=128)
    finally:
        env.close()
