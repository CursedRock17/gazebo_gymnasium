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

"""Generalized multi-agent Gymnasium VecEnv for Gazebo.

There is a single env path: ``MultiAgentGazeboVecEnv``, parameterized by an
``AgentSpec`` and built for N agents in one gz world. Single-agent training
is just ``n_agents=1`` — there are no per-model or single-agent env classes.

    from gazebo_gymnasium_bridge.envs import make_multi
    env = make_multi("cartpole", n_agents=16)
"""

from .agent_spec import AgentSpec
from .agent_spec import JointObs
from .agent_spec import get_spec
from .agent_spec import register_spec
from .agent_spec import registered_specs
from .multi_agent_env import MultiAgentGazeboVecEnv
from .multi_agent_env import make_multi
from .multi_cartpole import MultiCartPoleVecEnv

__all__ = [
    "MultiAgentGazeboVecEnv",
    "make_multi",
    "AgentSpec",
    "JointObs",
    "get_spec",
    "register_spec",
    "registered_specs",
    "MultiCartPoleVecEnv",
]
