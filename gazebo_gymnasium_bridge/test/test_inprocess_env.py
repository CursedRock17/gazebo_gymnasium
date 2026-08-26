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


def test_deterministic_given_seed():
    # Reproducibility: same seed + same actions -> bit-identical trajectory,
    # across two fresh env instances (DART is deterministic; seed threads
    # through the reset randomization).
    def rollout(seed):
        env = _env(2)
        try:
            env.seed(seed)
            traj = [env.reset().copy()]
            for _ in range(20):
                obs, _r, _d, _i = env.step(np.array([1, 0]))
                traj.append(obs.copy())
            return np.array(traj)
        finally:
            env.close()

    assert np.array_equal(rollout(3), rollout(3)), "same seed + actions must reproduce exactly"


def test_mass_randomization_scales_sdf():
    # Population-based dynamics randomization: each agent's mass+inertia is
    # scaled by a per-agent, seed-reproducible factor. Off by default.
    from dataclasses import replace
    import re

    from gazebo_gymnasium_bridge.envs import inprocess_vec_env as ip
    from gazebo_gymnasium_bridge.envs.agent_spec import get_spec

    base = get_spec("cartpole")
    w0, _ = ip._build_world(base, 4, 3.0, np.random.default_rng(0))
    assert set(re.findall(r"<mass>([-\d.eE+]+)</mass>", w0)) == {"1"}, (
        "mass randomization off by default -> unscaled masses"
    )

    spec = replace(base, mass_randomization=0.4)
    w1, _ = ip._build_world(spec, 4, 3.0, np.random.default_rng(0))
    w2, _ = ip._build_world(spec, 4, 3.0, np.random.default_rng(0))
    assert w1 == w2, "same seed must reproduce the randomized world"
    masses = {round(float(m), 4) for m in re.findall(r"<mass>([-\d.eE+]+)</mass>", w1)}
    assert len(masses) >= 4, "each of the 4 agents should get a distinct mass"


def test_action_gain_randomization_sets_varied_gains():
    # Control-authority DR: each agent gets a distinct, in-band, seed-
    # reproducible actuator gain on the ECM core.
    from dataclasses import replace

    from gazebo_gymnasium_bridge.envs import make_inprocess
    from gazebo_gymnasium_bridge.envs.agent_spec import get_spec
    from gazebo_gymnasium_bridge.envs.agent_spec import register_spec

    spec = replace(get_spec("cartpole"), name="cp_gaintest", action_gain_randomization=0.4)
    register_spec("cp_gaintest", lambda: spec)
    env = make_inprocess("cp_gaintest", n_agents=4, seed=0)
    try:
        gains = env._core._action_gains
        assert len({round(g, 4) for g in gains}) == 4, "distinct per agent"
        assert all(0.6 <= g <= 1.4 for g in gains), "within +-40%"
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
        model = sb3.PPO("MlpPolicy", env, n_steps=16, batch_size=16, n_epochs=1, verbose=0)
        model.learn(total_timesteps=128)
    finally:
        env.close()
