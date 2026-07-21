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

"""Library-integration smoke tests against the generalized VecEnv.

Verifies the RL libraries we advertise accept ``make_multi(...)`` output
without raising. Each test constructs the env with gz-transport mocked (no
Gazebo), instantiates the algorithm, and runs ``learn`` for a few steps.
Convergence is NOT measured.

Discrete-action algorithms (PPO, A2C) are covered now via the ``cartpole``
spec. Continuous-action algorithms (SAC, TD3, DDPG) return once a
continuous-action spec lands (MuJoCo models / line-follower in later steps);
they are skipped until then so the suite documents the intended surface.

Run with: pytest gazebo_gymnasium_bridge/test/test_library_integration.py -v
"""

from pathlib import Path
import sys
from unittest.mock import MagicMock

import pytest


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "gazebo_gymnasium_bridge"))


gz_transport13 = pytest.importorskip(
    "gz.transport13", reason="gz-transport bindings not available")
pytest.importorskip("gz.msgs10", reason="gz-msgs bindings not available")


def _make_cartpole_multi(monkeypatch, n_agents=2):
    """Make a mocked make_multi('cartpole') in a non-terminal (upright) state.

    Transport is mocked and recreate/wait short-circuited.
    """
    monkeypatch.setattr(gz_transport13, "Node", MagicMock(name="Node"))
    from gazebo_gymnasium_bridge.backend.nodes import world_control
    monkeypatch.setattr(world_control.WorldController, "_send",
                        lambda self, *a, **kw: True)
    from gazebo_gymnasium_bridge.envs import make_multi
    env = make_multi("cartpole", n_agents=n_agents, world_name="t",
                     reset_timeout=0.01, step_timeout=0.01)
    monkeypatch.setattr(env, "_recreate_all_agents", lambda: None)
    monkeypatch.setattr(env, "_wait_all_states", lambda timeout: True)
    env._latest_states[:] = 0.0  # upright -> non-terminal
    return env


@pytest.fixture
def cartpole_multi(monkeypatch):
    return _make_cartpole_multi(monkeypatch, n_agents=2)


class TestStableBaselines3Discrete:
    """PPO + A2C against the generalized cartpole VecEnv."""

    def test_ppo_accepts_cartpole(self, cartpole_multi):
        sb3 = pytest.importorskip("stable_baselines3")
        model = sb3.PPO("MlpPolicy", cartpole_multi, n_steps=4, batch_size=4,
                        n_epochs=1, verbose=0)
        model.learn(total_timesteps=8)

    def test_a2c_accepts_cartpole(self, cartpole_multi):
        sb3 = pytest.importorskip("stable_baselines3")
        model = sb3.A2C("MlpPolicy", cartpole_multi, n_steps=4, verbose=0)
        model.learn(total_timesteps=8)


@pytest.mark.skip(reason="continuous-action specs (MuJoCo / line-follower) "
                         "land in a later step; SAC/TD3/DDPG coverage returns "
                         "with the first Box-action spec")
class TestStableBaselines3Continuous:
    """Placeholder documenting the continuous-algo surface to restore."""

    def test_sac_td3_ddpg_continuous(self):
        pass
