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

from pathlib import Path
import sys
from unittest.mock import MagicMock

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

gz_transport13 = pytest.importorskip("gz.transport13", reason="gz bindings not available")
pytest.importorskip("gz.msgs10", reason="gz bindings not available")


@pytest.fixture
def harness_env(monkeypatch):
    monkeypatch.setattr(gz_transport13, "Node", MagicMock(name="Node"))
    from gazebo_gymnasium_bridge.envs import make_harness

    env = make_harness(
        "cartpole", n_agents=3, world_name="t", reset_timeout=0.01, step_timeout=0.01
    )
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
        model = sb3.PPO("MlpPolicy", harness_env, n_steps=4, batch_size=4, n_epochs=1, verbose=0)
        model.learn(total_timesteps=12)


def test_step_async_flattens_actions(monkeypatch):
    monkeypatch.setattr(gz_transport13, "Node", MagicMock(name="Node"))
    from gazebo_gymnasium_bridge.envs import make_harness

    env = make_harness("cartpole", n_agents=4, world_name="t")
    published = {}
    env._act_pub.publish = lambda m: published.setdefault("data", list(m.data))
    env.step_async(np.array([1, 0, 1, 0]))
    assert published["data"] == [1.0, 0.0, 1.0, 0.0]


def test_reset_raises_when_no_observations(monkeypatch):
    # A sim that never publishes (no agents spawned / plugin not loaded) must
    # fail loudly, not silently truncate at the cap on all-zeros.
    monkeypatch.setattr(gz_transport13, "Node", MagicMock(name="Node"))
    from gazebo_gymnasium_bridge.envs import make_harness

    env = make_harness("cartpole", n_agents=3, world_name="t", reset_timeout=0.01)
    with pytest.raises(RuntimeError, match="no observations"):
        env.reset()


def test_reset_raises_on_agent_count_mismatch(monkeypatch):
    monkeypatch.setattr(gz_transport13, "Node", MagicMock(name="Node"))
    from gazebo_gymnasium_bridge.envs import make_harness

    env = make_harness("cartpole", n_agents=3, world_name="t", reset_timeout=0.01)

    class _Msg:
        data = [1.0, 2.0, 3.0, 4.0]  # not 3 agents x 4 dims

    env._on_obs(_Msg())
    with pytest.raises(RuntimeError, match="does not match"):
        env.reset()


# ---- image-obs (line_follower) support ------------------------------------ #
#
# The harness plugin's /rl/observations channel is ECM/joint-state-only, so it
# carries nothing meaningful for an image spec (no joint_obs) -- these tests
# cover the two things that matter: (1) that meaningless channel is used
# purely as a pacing clock and must NOT clobber the real (image) observation,
# and (2) the per-agent camera subscriptions actually populate it. Live image
# transport shape (real pixels over a real launched world) is proven manually,
# not in the offline suite -- see docs/examples/line_follower.md.
#
# These patch harness_vec_env.Node directly (not gz_transport13.Node, the
# rest of this file's pattern): harness_vec_env does `from gz.transport13
# import Node` at module scope, so once that's imported once in the pytest
# session, patching gz_transport13's own attribute no longer reaches it --
# every HarnessVecEnv built afterward can end up sharing ONE frozen mock
# instance (fine for the other tests here, which never need to tell one
# env's recorded calls apart from another's; not fine for these, which do).


def _image_env(monkeypatch, **kwargs):
    import gazebo_gymnasium_bridge.envs.harness_vec_env as harness_vec_env

    monkeypatch.setattr(harness_vec_env, "Node", MagicMock(name="Node"))
    from gazebo_gymnasium_bridge.envs import make_harness

    return make_harness("line_follower", n_agents=2, world_name="t", **kwargs)


def test_image_spec_sets_up_camera_mode(monkeypatch):
    env = _image_env(monkeypatch)
    assert env._image_mode is True
    assert env._latest_obs.shape == (2, 64, 64, 3)
    assert env._latest_obs.dtype == np.uint8

    # one subscribe call per agent for /rl/camera_i, on top of /rl/observations
    topics = [c.args[1] for c in env._node.subscribe.call_args_list]
    assert "/rl/camera_0" in topics and "/rl/camera_1" in topics


def test_image_spec_state_channel_is_clock_only(monkeypatch):
    # The plugin still publishes /rl/observations every tick for an image
    # spec (harmless zero-floats -- obs_dim comes from observation_space.
    # shape[0], not joint_obs, so it's nonzero even with no joints). That
    # message must advance the pacing clock but never touch _latest_obs.
    env = _image_env(monkeypatch, frame_skip=2)
    env._latest_obs[:] = 7  # sentinel -- must survive _on_obs untouched

    class _Msg:
        data = [0.0] * (2 * env._obs_dim)

    assert env._obs_event.is_set() is False
    env._on_obs(_Msg())
    assert not env._obs_event.is_set()  # 1 of frame_skip=2
    env._on_obs(_Msg())
    assert env._obs_event.is_set()  # 2 of 2 -> pacing fires
    assert (env._latest_obs == 7).all(), (
        "the meaningless state payload must not overwrite the image obs"
    )


def test_image_spec_camera_callback_fills_latest_obs(monkeypatch):
    env = _image_env(monkeypatch)
    cam0_cb = next(
        c.args[2] for c in env._node.subscribe.call_args_list if c.args[1] == "/rl/camera_0"
    )

    class _ImgMsg:
        width, height = 64, 64
        data = bytes([42]) * (64 * 64 * 3)

    cam0_cb(_ImgMsg())
    assert (env._latest_obs[0] == 42).all()
    assert (env._latest_obs[1] == 0).all(), "agent 1's frame must be untouched"
