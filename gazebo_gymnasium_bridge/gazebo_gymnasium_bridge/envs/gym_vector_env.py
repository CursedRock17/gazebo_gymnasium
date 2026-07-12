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

"""Native ``gymnasium.vector.VectorEnv`` over the N-in-one-sim engine.

This exposes the efficient batched env (N agents in ONE Gazebo world) through
the standard Gymnasium vector API, so ``gymnasium.make_vec("GazeboCartPole-v0",
num_envs=16)`` returns *this* — not 16 separate simulators. It reports
**SAME_STEP** autoreset (``metadata["autoreset_mode"]``): a sub-env that
terminates is reset in place on the same step, its observation is the reset
observation, and the terminal observation is carried in ``infos["final_obs"]``
(with the ``infos["_final_obs"]`` boolean mask, per the Gymnasium 1.0 vector
info convention). That mode is exactly what the in-process engine already does
per agent, so this is a thin, correct adapter.

SB3 keeps its own VecEnv (``make_inprocess`` / ``make_harness``) — it cannot
consume a ``gymnasium.vector.VectorEnv`` directly.
"""

from gymnasium.vector import VectorEnv
from gymnasium.vector.utils import batch_space
import numpy as np

from .agent_spec import get_spec

try:                                            # gymnasium >= 1.0
    from gymnasium.vector import AutoresetMode
    _SAME_STEP = AutoresetMode.SAME_STEP
except ImportError:                             # pragma: no cover
    _SAME_STEP = "SameStep"


class GazeboVectorEnv(VectorEnv):
    """N agents in one Gazebo world as a gymnasium.vector.VectorEnv."""

    def __init__(self, agent: str = "cartpole", num_envs: int = 8, **kwargs):
        from .inprocess_vec_env import InProcessHarnessVecEnv
        spec = get_spec(agent)
        kwargs.pop("n_agents", None)
        kwargs.pop("render_mode", None)
        self._engine = InProcessHarnessVecEnv(
            spec, n_agents=num_envs, autoreset=True, **kwargs)

        self.num_envs = num_envs
        self.single_observation_space = spec.observation_space
        self.single_action_space = spec.action_space
        self.observation_space = batch_space(spec.observation_space, num_envs)
        self.action_space = batch_space(spec.action_space, num_envs)
        self.metadata = {"autoreset_mode": _SAME_STEP}
        self.render_mode = None
        self.spec = None
        self.closed = False

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._engine.seed(seed)
        return self._engine.reset(), {}

    def step(self, actions):
        self._engine.step_async(np.asarray(actions))
        obs, rewards, dones, infos_list = self._engine.step_wait()
        terminations = np.zeros(self.num_envs, dtype=bool)
        truncations = np.zeros(self.num_envs, dtype=bool)
        infos = {}
        for i in np.nonzero(dones)[0]:
            trunc = "TimeLimit.truncated" in infos_list[i]
            truncations[i] = trunc
            terminations[i] = not trunc
            infos = self._add_info(
                infos, {"final_obs": infos_list[i]["terminal_observation"]},
                int(i))
        return (obs, rewards.astype(np.float64), terminations, truncations,
                infos)

    def render(self):
        return None

    def close(self, **kwargs):
        self._engine.close()
        self.closed = True


def make_gym_vector(num_envs: int = 1, agent: str = "cartpole", **kwargs):
    """vector_entry_point for gymnasium.make_vec (builds a GazeboVectorEnv)."""
    return GazeboVectorEnv(agent=agent, num_envs=num_envs, **kwargs)
