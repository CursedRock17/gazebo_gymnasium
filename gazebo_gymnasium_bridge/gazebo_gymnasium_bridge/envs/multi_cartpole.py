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

"""Backward-compatible cartpole multi env.

The implementation now lives in the generalized
:class:`~gazebo_gymnasium_bridge.envs.multi_agent_env.MultiAgentGazeboVecEnv`.
``MultiCartPoleVecEnv`` is kept as a thin wrapper bound to the ``cartpole``
:class:`AgentSpec` so existing training scripts / checkpoints keep working;
new code should prefer ``make_multi("cartpole", n_agents=...)``.
"""

from .agent_spec import get_spec
from .multi_agent_env import MultiAgentGazeboVecEnv


class MultiCartPoleVecEnv(MultiAgentGazeboVecEnv):
    """N cartpoles in one gz sim (cartpole spec over the generalized VecEnv)."""

    def __init__(self, n_agents: int = 4,
                 world_name: str = "cartpole_multi", **kwargs):
        super().__init__(get_spec("cartpole"), n_agents=n_agents,
                         world_name=world_name, **kwargs)
