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

"""Standard single-agent ``gymnasium.Env`` + registration.

This is the universal-interop surface. It wraps the in-process engine
(``n_agents=1``, autoreset off) as a fully compliant ``gymnasium.Env`` — the
0.26+ ``(obs, reward, terminated, truncated, info)`` step and ``(obs, info)``
reset — so any Gymnasium-speaking tool (RLlib, CleanRL, Tianshou, TorchRL, the
``gymnasium.utils.env_checker``) can just::

    import gazebo_gymnasium_bridge          # registers the ids
    env = gymnasium.make("GazeboCartPole-v0")

For efficient training you still want the N-in-one-sim VecEnvs
(``make_inprocess`` / ``make_harness``); this single-agent env spins up one
Gazebo world per instance and exists for compliance + tools that vectorize
themselves.
"""

import gymnasium as gym
import numpy as np

from .agent_spec import get_spec


# env id -> registered agent spec name
_REGISTERED = {"GazeboCartPole-v0": "cartpole"}


class GazeboEnv(gym.Env):
    """One agent of a spec as a standard gymnasium.Env (own in-process sim)."""

    metadata = {"render_modes": []}

    def __init__(self, agent: str = "cartpole", render_mode=None, **kwargs):
        # Imported here (not at module top) so registration + this module stay
        # importable without the native gz bindings; only construction needs them.
        from .inprocess_vec_env import InProcessHarnessVecEnv
        spec = get_spec(agent)
        self.observation_space = spec.observation_space
        self.action_space = spec.action_space
        self.render_mode = render_mode
        kwargs.pop("n_agents", None)
        self._vec = InProcessHarnessVecEnv(
            spec, n_agents=1, autoreset=False, **kwargs)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._vec.seed(seed)
        obs = self._vec.reset()
        return obs[0], {}

    def step(self, action):
        self._vec.step_async(np.asarray([action]))
        obs, rewards, dones, infos = self._vec.step_wait()
        info = infos[0]
        truncated = bool(info.pop("TimeLimit.truncated", False))
        terminated = bool(dones[0]) and not truncated
        return obs[0], float(rewards[0]), terminated, truncated, info

    def render(self):
        return None

    def close(self):
        self._vec.close()


def register_envs():
    """Register the built-in Gazebo Gymnasium ids (idempotent)."""
    from gymnasium.envs.registration import registry
    for env_id, agent in _REGISTERED.items():
        if env_id in registry:
            continue
        gym.register(
            id=env_id,
            entry_point="gazebo_gymnasium_bridge.envs.gym_env:GazeboEnv",
            kwargs={"agent": agent},
        )
