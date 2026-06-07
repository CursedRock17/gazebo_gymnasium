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
Library-integration smoke tests.

These tests verify that the env classes plug into the RL libraries we
advertise as supported. Each test:

  1. Constructs the env with gz-transport mocked (so no Gazebo needed)
  2. Instantiates the algorithm (SB3 PPO, A2C, SAC, TD3, DDPG, ...)
  3. Calls `learn(total_timesteps=…)` for a tiny number of steps

Each is a smoke test in the strictest sense: "does the library accept this
env without raising?" Convergence is NOT measured — that's the job of the
real training scripts. We just protect against accidental regressions like
"my env returns a list instead of an ndarray and SB3 silently breaks."

The Discrete-action algorithms (PPO, A2C) are tested with CartPole; the
Box-action algorithms (SAC, TD3, DDPG) with InvertedPendulum.

Run with: pytest gazebo_gymnasium_bridge/test/test_library_integration.py -v
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "gazebo_gymnasium_bridge"))


gz_transport13 = pytest.importorskip(
    "gz.transport13",
    reason="gz-transport bindings not available",
)
pytest.importorskip("gz.msgs10", reason="gz-msgs bindings not available")


def _patch_transport(monkeypatch):
    """Stub out gz-transport + WorldController._send."""
    monkeypatch.setattr(gz_transport13, "Node", MagicMock(name="Node"))
    from gazebo_gymnasium_bridge.backend.nodes import world_control
    monkeypatch.setattr(
        world_control.WorldController, "_send",
        lambda self, *a, **kw: True,
    )


def _force_state_event(env):
    """Make step() and reset() non-blocking — both pre-set and keep set."""
    env._state_event.set()
    original_step = env.step

    def step_keep_event(action):
        env._state_event.set()
        return original_step(action)

    env.step = step_keep_event
    return env


@pytest.fixture
def cartpole_env(monkeypatch):
    _patch_transport(monkeypatch)
    from gazebo_gymnasium_bridge.envs import GazeboCartPoleEnv
    env = GazeboCartPoleEnv(
        world_name="test_world", max_episode_steps=8,
        reset_timeout=0.01, step_timeout=0.01,
    )
    env._latest_state = [0.0, 0.0, 0.0, 0.0]
    return _force_state_event(env)


@pytest.fixture
def inverted_pendulum_env(monkeypatch):
    _patch_transport(monkeypatch)
    from gazebo_gymnasium_bridge.envs import GazeboInvertedPendulumEnv
    env = GazeboInvertedPendulumEnv(
        world_name="test_world", max_episode_steps=8,
        reset_timeout=0.01, step_timeout=0.01,
    )
    env._latest_state = [0.0, 0.0, 0.0, 0.0]
    return _force_state_event(env)


@pytest.fixture
def inverted_double_pendulum_env(monkeypatch):
    _patch_transport(monkeypatch)
    from gazebo_gymnasium_bridge.envs import GazeboInvertedDoublePendulumEnv
    env = GazeboInvertedDoublePendulumEnv(
        world_name="test_world", max_episode_steps=8,
        reset_timeout=0.01, step_timeout=0.01,
    )
    env._latest_state = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    return _force_state_event(env)


@pytest.fixture
def line_follower_env(monkeypatch):
    _patch_transport(monkeypatch)
    from gazebo_gymnasium_bridge.envs import GazeboLineFollowerEnv
    env = GazeboLineFollowerEnv(
        world_name="test_world", max_episode_steps=8,
        reset_timeout=0.01, step_timeout=0.01,
    )
    # Seed obs cache to mid-range valid values: 0.5 m/s forward, 0 yaw,
    # flat pitch, line dead-center.
    env._latest_state = [0.5, 0.0, 0.0, 0.0]
    return _force_state_event(env)


@pytest.fixture
def reacher_env(monkeypatch):
    _patch_transport(monkeypatch)
    from gazebo_gymnasium_bridge.envs import GazeboReacherEnv
    env = GazeboReacherEnv(
        world_name="test_world", max_episode_steps=8,
        reset_timeout=0.01, step_timeout=0.01,
    )
    # [theta0, theta1, dtheta0, dtheta1, fingertip_x, fingertip_y]
    env._latest_state = [0.0, 0.0, 0.0, 0.0, 0.21, 0.0]
    return _force_state_event(env)


@pytest.fixture
def point_env(monkeypatch):
    _patch_transport(monkeypatch)
    from gazebo_gymnasium_bridge.envs import GazeboPointEnv
    env = GazeboPointEnv(
        world_name="test_world", max_episode_steps=8,
        reset_timeout=0.01, step_timeout=0.01,
    )
    # [x, y, vx, vy]
    env._latest_state = [0.0, 0.0, 0.0, 0.0]
    return _force_state_event(env)


class TestStableBaselines3Discrete:
    """PPO + A2C against CartPole."""

    def test_ppo_accepts_cartpole(self, cartpole_env):
        sb3 = pytest.importorskip("stable_baselines3")
        model = sb3.PPO("MlpPolicy", cartpole_env, n_steps=8, batch_size=8, verbose=0)
        model.learn(total_timesteps=8)

    def test_a2c_accepts_cartpole(self, cartpole_env):
        sb3 = pytest.importorskip("stable_baselines3")
        model = sb3.A2C("MlpPolicy", cartpole_env, n_steps=4, verbose=0)
        model.learn(total_timesteps=8)


class TestStableBaselines3Continuous:
    """SAC + TD3 + DDPG against InvertedPendulum."""

    def test_sac_accepts_inverted_pendulum(self, inverted_pendulum_env):
        sb3 = pytest.importorskip("stable_baselines3")
        model = sb3.SAC(
            "MlpPolicy", inverted_pendulum_env,
            learning_starts=4, train_freq=1, batch_size=4,
            buffer_size=64, verbose=0,
        )
        model.learn(total_timesteps=8)

    def test_td3_accepts_inverted_pendulum(self, inverted_pendulum_env):
        sb3 = pytest.importorskip("stable_baselines3")
        model = sb3.TD3(
            "MlpPolicy", inverted_pendulum_env,
            learning_starts=4, train_freq=(1, "step"), batch_size=4,
            buffer_size=64, verbose=0,
        )
        model.learn(total_timesteps=8)

    def test_ddpg_accepts_inverted_pendulum(self, inverted_pendulum_env):
        sb3 = pytest.importorskip("stable_baselines3")
        model = sb3.DDPG(
            "MlpPolicy", inverted_pendulum_env,
            learning_starts=4, train_freq=(1, "step"), batch_size=4,
            buffer_size=64, verbose=0,
        )
        model.learn(total_timesteps=8)

    def test_a2c_accepts_inverted_double_pendulum(self, inverted_double_pendulum_env):
        """A2C on the 8-dim continuous env — same combo as our training script."""
        sb3 = pytest.importorskip("stable_baselines3")
        model = sb3.A2C(
            "MlpPolicy", inverted_double_pendulum_env,
            n_steps=4, verbose=0,
        )
        model.learn(total_timesteps=8)

    def test_ppo_accepts_line_follower(self, line_follower_env):
        """PPO on the 2D-action line-following env — matches our trainer."""
        sb3 = pytest.importorskip("stable_baselines3")
        model = sb3.PPO(
            "MlpPolicy", line_follower_env,
            n_steps=8, batch_size=8, verbose=0,
        )
        model.learn(total_timesteps=8)

    def test_td3_accepts_reacher(self, reacher_env):
        """TD3 on the 2D-action Reacher env — matches our trainer."""
        sb3 = pytest.importorskip("stable_baselines3")
        model = sb3.TD3(
            "MlpPolicy", reacher_env,
            learning_starts=4, train_freq=(1, "step"), batch_size=4,
            buffer_size=64, verbose=0,
        )
        model.learn(total_timesteps=8)

    def test_sac_accepts_point(self, point_env):
        """SAC on the 2D-action Point env — matches our trainer."""
        sb3 = pytest.importorskip("stable_baselines3")
        model = sb3.SAC(
            "MlpPolicy", point_env,
            learning_starts=4, train_freq=1, batch_size=4,
            buffer_size=64, verbose=0,
        )
        model.learn(total_timesteps=8)


class TestCleanRLStyle:
    """The CleanRL example file at the project root should be importable and
    expose a `main()` callable. We don't run it (it would loop forever), just
    that it loads and its public surface is intact — protects against
    accidental syntax breaks during refactors."""

    def test_cleanrl_script_is_importable(self):
        path = PROJECT_ROOT / "training_scripts" / "train_cartpole_cleanrl.py"
        assert path.exists(), f"missing {path}"
        # NOTE: don't exec the module (it would try to construct env immediately
        # via `main()` if we called it). Just verify the AST parses.
        with open(path) as f:
            source = f.read()
        compile(source, str(path), "exec")
        assert "def main" in source
        assert "Agent" in source
